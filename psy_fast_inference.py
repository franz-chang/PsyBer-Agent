#!/usr/bin/env python3
"""Fast policy runtime + shared utilities for PsyBer alignment training/inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


STATE_KEYS = ("P", "C", "E", "S")
STATE_DIM = 4
ACTION_COUNT = 5  # click1..click4 + skip
# Bump when network architecture changes to keep backward compatibility.
FEATURE_VERSION = 2
FEATURE_DIM = 13
DEFAULT_HIDDEN_DIM = 64


class FastPolicyNetV1(nn.Module):
    """Original 1-layer scorer kept for loading old checkpoints (feature_version=1)."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 32):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastPolicyNet(nn.Module):
    """Two-layer scorer with LayerNorm+GELU for stabler training/inference."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        mid_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.mid_dim = int(mid_dim or hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.mid_dim),
            nn.GELU(),
            nn.Linear(self.mid_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastPolicyLoadError(RuntimeError):
    pass


def state_dict_to_vec(state: dict[str, Any]) -> np.ndarray:
    return np.array([float(state.get(k, 0.0)) for k in STATE_KEYS], dtype=np.float32)


def pareto_front_indices(deltas: np.ndarray, eps: float = 1e-8) -> list[int]:
    """Return non-dominated action indices on objective deltas."""
    arr = np.array(deltas, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != STATE_DIM:
        raise ValueError(f"deltas shape must be [N, {STATE_DIM}], got {arr.shape}")
    n = int(arr.shape[0])
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        ge = np.all(arr >= (arr[i] - eps), axis=1)
        gt = np.any(arr > (arr[i] + eps), axis=1)
        ge[i] = False
        gt[i] = False
        if np.any(ge & gt):
            dominated[i] = True
    return [int(i) for i in np.where(~dominated)[0].tolist()]


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.array(logits, dtype=np.float32)
    if x.size == 0:
        return x
    x = x - float(np.max(x))
    exp_x = np.exp(x)
    z = float(np.sum(exp_x))
    if not np.isfinite(z) or z <= 1e-12:
        return np.full_like(exp_x, 1.0 / max(1, exp_x.size))
    return exp_x / z


def apply_skip_transition(
    state_vec: np.ndarray,
    *,
    recovery_boost: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    """Heuristic skip transition used both in offline and online inference.

    Behavior:
    - If a dimension is depleted (<=0), skip recovers it by `recovery_boost`.
    - Otherwise skip causes slight natural drift (less arousal, mild cognitive recovery).
    """
    before = np.array(state_vec, dtype=np.float32).reshape(-1)
    if before.size != STATE_DIM:
        raise ValueError(f"state_vec must have {STATE_DIM} dims, got {before.shape}")

    after = before.copy()
    recovered: list[str] = []

    for i, key in enumerate(STATE_KEYS):
        if float(after[i]) <= 0.0:
            after[i] = float(np.clip(after[i] + recovery_boost, 0.0, 1.0))
            recovered.append(key)

    if not recovered:
        drift = np.array([-0.03, 0.04, -0.01, 0.02], dtype=np.float32)
        after = np.clip(after + drift, 0.0, 1.0)

    delta = (after - before).astype(np.float32)
    metrics = {
        "skip_delta_norm": float(np.linalg.norm(delta)),
        "skip_recovered_dims": float(len(recovered)),
    }
    return after.astype(np.float32), delta, recovered, metrics


def apply_skip_to_engine(
    engine: Any,
    *,
    recovery_boost: float = 0.3,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    before = state_dict_to_vec(engine.export_state())
    after, delta, recovered, metrics = apply_skip_transition(
        before, recovery_boost=recovery_boost
    )
    engine.P = float(after[0])
    engine.C = float(after[1])
    engine.E = float(after[2])
    engine.S = float(after[3])
    engine.Surf = 1 if np.all(after > 0.0) else 0
    return delta, recovered, metrics


def simulate_action_matrix(
    engine: Any,
    candidates: list[dict[str, Any]],
    *,
    include_skip: bool = True,
    recovery_boost: float = 0.3,
) -> dict[str, Any]:
    """Simulate 4-click (+skip) action consequences from current engine state."""
    actions: list[dict[str, Any]] = []
    deltas: list[np.ndarray] = []

    for cand in candidates:
        v = np.array(cand.get("v"), dtype=np.float32)
        n = float(cand.get("n", 0.0))
        c = float(cand.get("c", 0.0))
        pop = float(cand.get("pop", 0.0))
        _, delta, _, metrics = engine._calculate_transitions(v, n, c, pop)
        delta_arr = np.array(delta, dtype=np.float32)
        deltas.append(delta_arr)
        actions.append(
            {
                "action": "click",
                "item_id": str(cand.get("item_id") or ""),
                "title": str(cand.get("title") or ""),
                "v": v,
                "n": n,
                "c": c,
                "pop": pop,
                "metrics": {k: float(vv) for k, vv in metrics.items()},
            }
        )

    if include_skip:
        state_before = state_dict_to_vec(engine.export_state())
        _, skip_delta, recovered, skip_metrics = apply_skip_transition(
            state_before,
            recovery_boost=recovery_boost,
        )
        deltas.append(skip_delta)
        actions.append(
            {
                "action": "skip",
                "item_id": "",
                "title": "[SKIP]",
                "v": np.zeros_like(candidates[0]["v"], dtype=np.float32)
                if candidates
                else np.zeros((512,), dtype=np.float32),
                "n": 0.0,
                "c": 0.0,
                "pop": 0.0,
                "metrics": {
                    **skip_metrics,
                    "skip_recovered": float(len(recovered)),
                },
            }
        )

    delta_mat = np.stack(deltas, axis=0) if deltas else np.zeros((0, STATE_DIM), dtype=np.float32)
    return {
        "actions": actions,
        "deltas": delta_mat,
    }


def build_action_features(
    state_before: np.ndarray,
    actions: list[dict[str, Any]],
    deltas: np.ndarray,
) -> np.ndarray:
    state_vec = np.array(state_before, dtype=np.float32).reshape(-1)
    if state_vec.size != STATE_DIM:
        raise ValueError(f"state_before must have {STATE_DIM} dims, got {state_vec.shape}")
    delta_mat = np.array(deltas, dtype=np.float32)
    if delta_mat.ndim != 2 or delta_mat.shape[1] != STATE_DIM:
        raise ValueError(f"deltas shape must be [N, {STATE_DIM}], got {delta_mat.shape}")

    rows: list[np.ndarray] = []
    for idx, action in enumerate(actions):
        delta = delta_mat[idx]
        is_skip = 1.0 if action.get("action") == "skip" else 0.0
        item_n = float(action.get("n", 0.0))
        item_c = float(action.get("c", 0.0))
        item_pop = float(action.get("pop", 0.0))
        delta_norm = float(np.linalg.norm(delta))

        row = np.array(
            [
                state_vec[0],
                state_vec[1],
                state_vec[2],
                state_vec[3],
                delta[0],
                delta[1],
                delta[2],
                delta[3],
                item_n,
                item_c,
                item_pop,
                is_skip,
                delta_norm,
            ],
            dtype=np.float32,
        )
        rows.append(row)

    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


def select_action_from_logits(
    logits: np.ndarray,
    valid_indices: list[int],
    *,
    rng: np.random.Generator,
    stochastic: bool,
) -> tuple[int, np.ndarray]:
    arr = np.array(logits, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("logits is empty")

    valid = [int(i) for i in valid_indices if 0 <= int(i) < arr.size]
    if not valid:
        valid = list(range(arr.size))

    probs_valid = softmax_np(arr[valid])
    if stochastic:
        local_idx = int(rng.choice(len(valid), p=probs_valid))
    else:
        local_idx = int(np.argmax(probs_valid))
    chosen_idx = int(valid[local_idx])

    probs_full = np.zeros_like(arr)
    probs_full[valid] = probs_valid
    return chosen_idx, probs_full


def save_fast_policy_bundle(
    path: Path,
    model: FastPolicyNet,
    metadata: dict[str, Any],
) -> None:
    bundle = {
        "state_dict": model.state_dict(),
        "metadata": {
            "feature_version": FEATURE_VERSION,
            "feature_dim": int(metadata.get("feature_dim", model.feature_dim)),
            "hidden_dim": int(metadata.get("hidden_dim", model.hidden_dim)),
            **{k: v for k, v in metadata.items() if k not in {"feature_dim", "hidden_dim"}},
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load_fast_policy_bundle(path: Path, *, device: str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FastPolicyLoadError(f"Policy file not found: {path}")
    bundle = torch.load(path, map_location=device)
    if not isinstance(bundle, dict):
        raise FastPolicyLoadError("Invalid policy bundle format")
    metadata = bundle.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise FastPolicyLoadError("Invalid metadata in policy bundle")

    feature_dim = int(metadata.get("feature_dim", FEATURE_DIM))
    hidden_dim = int(metadata.get("hidden_dim", DEFAULT_HIDDEN_DIM))
    feature_version = int(metadata.get("feature_version", 1))

    if feature_version <= 1:
        model = FastPolicyNetV1(feature_dim=feature_dim, hidden_dim=hidden_dim)
    else:
        mid_dim = int(metadata.get("mid_dim", hidden_dim))
        model = FastPolicyNet(feature_dim=feature_dim, hidden_dim=hidden_dim, mid_dim=mid_dim)

    state_dict = bundle.get("state_dict")
    if not isinstance(state_dict, dict):
        raise FastPolicyLoadError("Missing state_dict in policy bundle")
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "metadata": metadata,
        "path": str(path),
    }
