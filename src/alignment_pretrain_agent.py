#!/usr/bin/env python3
"""Alignment pretraining agent.

Pipeline (between prelearning_agent.py and web_sim_camel_agent.py):
1) Rebuild implicit click trajectories from ML-1M ratings.dat.
2) Simulate 5-action slate decisions (Click1..Click4 + Skip) and collect agent states.
3) Build structure distances for agent-state and expert-item trajectories.
4) Learn regression prior and initialize UGW transport from pseudo-state matching.
5) Solve UGW-style transport and convert it into per-step action soft labels.
6) Train lightweight fast policy network via supervised CE + entropy regularization.
7) Export fast policy for online inference in web_sim_camel_agent.py.

Legacy RL-reward objective is still available via `--alignment-mode legacy_rl`.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from psy_fast_inference import (
    FEATURE_DIM,
    FastPolicyNet,
    apply_skip_guard_to_candidates,
    apply_skip_to_engine,
    build_action_features,
    load_fast_policy_bundle,
    pareto_front_indices,
    save_fast_policy_bundle,
    simulate_action_matrix,
    state_dict_to_vec,
)


SRC_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = Path("/Users/chongzhang/WebSim_Dataset/MM-ML-1M-main")
DEFAULT_EMBEDDINGS_FILE = DEFAULT_DATASET_DIR / "Embeddings" / "item_embeddings.json"
DEFAULT_PROFILES_FILE = DEFAULT_DATASET_DIR / "Embeddings" / "agent_profiles.json"
DEFAULT_POLICY_OUT = DEFAULT_DATASET_DIR / "Embeddings" / "fast_inference_policy.pt"
DEFAULT_REPORT_OUT = DEFAULT_DATASET_DIR / "Embeddings" / "alignment_pretrain_report.json"
DEFAULT_TRAJ_PREVIEW_OUT = DEFAULT_DATASET_DIR / "Embeddings" / "expert_intent_preview.json"
DEFAULT_INTENT_ENCODER_OUT = DEFAULT_DATASET_DIR / "Embeddings" / "intent_encoder.pt"
DEFAULT_INTENT_STATS_OUT = DEFAULT_DATASET_DIR / "Embeddings" / "intent_stats.json"
PSY_ENGINE_FILE = SRC_DIR / "Psy-engine.py"


@dataclass
class ItemFeature:
    item_id: str
    item_v: np.ndarray
    item_n: float
    item_c: float
    item_pop: float


class ICVFIntentModel(nn.Module):
    """Intent-conditioned value function with psi encoder used as intent mapper."""

    def __init__(self, state_dim: int, intent_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = int(state_dim)
        self.intent_dim = int(intent_dim)
        h = max(8, int(hidden_dim))

        self.phi_net = nn.Sequential(
            nn.Linear(self.state_dim, h),
            nn.GELU(),
            nn.Linear(h, self.intent_dim),
        )
        self.psi_net = nn.Sequential(
            nn.Linear(self.state_dim, h),
            nn.GELU(),
            nn.Linear(h, self.intent_dim),
        )
        self.trans_net = nn.Sequential(
            nn.Linear(self.intent_dim, h),
            nn.GELU(),
            nn.Linear(h, self.intent_dim * self.intent_dim),
        )

    def encode_phi(self, s: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.phi_net(s), dim=-1)

    def encode_intent(self, s: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.psi_net(s), dim=-1)

    def value(self, s: torch.Tensor, s_plus: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        phi = self.encode_phi(s)
        psi = self.encode_intent(s_plus)
        trans = self.trans_net(z).view(-1, self.intent_dim, self.intent_dim)
        proj = torch.bmm(trans, psi.unsqueeze(-1)).squeeze(-1)
        return (phi * proj).sum(dim=-1)

    def forward(self, s: torch.Tensor, s_plus: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.value(s, s_plus, z)


class UGWInitRegressor(nn.Module):
    """Regresses expert pseudo-psychological states for UGW prior initialization."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        h = max(16, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def stable_float_01(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16)
    return raw / float(0xFFFFFFFF)


def deterministic_unit_vector(key: str, dim: int) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    return vec


def as_float_01(value: Any, default: float) -> float:
    try:
        x = float(value)
    except Exception:
        x = float(default)
    return float(max(0.0, min(1.0, x)))


def pairwise_l2(x: np.ndarray) -> np.ndarray:
    arr = np.array(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"pairwise_l2 expects 2D input, got {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    sq = np.sum(arr * arr, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (arr @ arr.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.sqrt(dist2 + 1e-12).astype(np.float32)


def normalized_time_distance(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 1), dtype=np.float32)
    pos = np.arange(n, dtype=np.float32) / float(n - 1)
    return np.abs(pos[:, None] - pos[None, :]).astype(np.float32)


def cross_normalized_time_distance(n: int, m: int) -> np.ndarray:
    if n <= 0 or m <= 0:
        return np.zeros((max(0, n), max(0, m)), dtype=np.float32)
    if n == 1:
        pos_n = np.zeros((1,), dtype=np.float32)
    else:
        pos_n = np.arange(n, dtype=np.float32) / float(n - 1)
    if m == 1:
        pos_m = np.zeros((1,), dtype=np.float32)
    else:
        pos_m = np.arange(m, dtype=np.float32) / float(m - 1)
    return np.abs(pos_n[:, None] - pos_m[None, :]).astype(np.float32)


def build_expert_structure_distance(
    item_vectors: np.ndarray,
    *,
    time_weight: float,
) -> np.ndarray:
    embeds = np.array(item_vectors, dtype=np.float32)
    if embeds.ndim != 2:
        raise ValueError(f"item_vectors must be 2D, got {embeds.shape}")
    d_item = pairwise_l2(embeds)
    d_time = normalized_time_distance(embeds.shape[0])
    return d_item + float(max(0.0, time_weight)) * d_time


def build_agent_structure_distance(
    states: np.ndarray,
    *,
    delta_weight: float,
    time_weight: float,
) -> np.ndarray:
    arr = np.array(states, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"agent states shape must be [N,4], got {arr.shape}")
    d_state = pairwise_l2(arr)
    delta = np.diff(arr, axis=0, prepend=arr[0:1])
    d_delta = pairwise_l2(delta)
    d_time = normalized_time_distance(arr.shape[0])
    return d_state + float(max(0.0, delta_weight)) * d_delta + float(max(0.0, time_weight)) * d_time


def unbalanced_sinkhorn_transport(
    cost: np.ndarray,
    *,
    epsilon: float,
    mass_reg: float,
    n_iter: int,
) -> np.ndarray:
    c = np.array(cost, dtype=np.float64)
    n, m = c.shape
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    a = np.full((n,), 1.0 / n, dtype=np.float64)
    b = np.full((m,), 1.0 / m, dtype=np.float64)

    eps = max(float(epsilon), 1e-4)
    reg = max(float(mass_reg), 1e-6)
    tau = reg / (reg + eps)

    k = np.exp(-c / eps)
    k = np.clip(k, 1e-300, None)

    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max(1, int(n_iter))):
        kv = k @ v + 1e-12
        u = np.power(a / kv, tau)
        ktu = k.T @ u + 1e-12
        v = np.power(b / ktu, tau)

    pi = (u[:, None] * k) * v[None, :]
    z = float(pi.sum())
    if not np.isfinite(z) or z <= 1e-12:
        return np.outer(a, b)
    return pi / z


def ugw_transport(
    agent_dist: np.ndarray,
    expert_dist: np.ndarray,
    *,
    epsilon: float,
    mass_reg: float,
    outer_iters: int,
    inner_iters: int,
    tol: float,
    initial_plan: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    da = np.array(agent_dist, dtype=np.float64)
    de = np.array(expert_dist, dtype=np.float64)
    n, n2 = da.shape
    m, m2 = de.shape
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64), np.zeros((n, m), dtype=np.float64)
    if n != n2 or m != m2:
        raise ValueError(f"UGW expects square distance matrices, got agent={da.shape}, expert={de.shape}")

    if initial_plan is None:
        p = np.full((n, m), 1.0 / (n * m), dtype=np.float64)
    else:
        p0 = np.array(initial_plan, dtype=np.float64)
        if p0.shape != (n, m):
            raise ValueError(
                f"UGW initial_plan shape mismatch: expected {(n, m)}, got {p0.shape}"
            )
        p0 = np.clip(p0, 0.0, None)
        z = float(p0.sum())
        if not np.isfinite(z) or z <= 1e-12:
            p = np.full((n, m), 1.0 / (n * m), dtype=np.float64)
        else:
            p = p0 / z
    da2 = da * da
    de2 = de * de
    cost = np.zeros((n, m), dtype=np.float64)

    for _ in range(max(1, int(outer_iters))):
        row_mass = p.sum(axis=1)
        col_mass = p.sum(axis=0)
        term_a = da2 @ row_mass
        term_b = de2 @ col_mass
        cross = da @ p @ de.T
        cost = np.maximum(term_a[:, None] + term_b[None, :] - 2.0 * cross, 0.0)

        p_new = unbalanced_sinkhorn_transport(
            cost,
            epsilon=epsilon,
            mass_reg=mass_reg,
            n_iter=inner_iters,
        )
        delta = float(np.mean(np.abs(p_new - p)))
        p = p_new
        if delta <= float(max(1e-8, tol)):
            break

    row_mass = p.sum(axis=1)
    col_mass = p.sum(axis=0)
    term_a = da2 @ row_mass
    term_b = de2 @ col_mass
    cross = da @ p @ de.T
    cost = np.maximum(term_a[:, None] + term_b[None, :] - 2.0 * cross, 0.0)
    return p, cost


def build_cost_matrix(
    agent_states: np.ndarray,
    expert_states: np.ndarray,
    *,
    lookahead_lambda: float,
) -> np.ndarray:
    a = np.array(agent_states, dtype=np.float32)
    b = np.array(expert_states, dtype=np.float32)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] <= 0
        or b.shape[1] <= 0
        or a.shape[1] != b.shape[1]
    ):
        raise ValueError(f"state/intent shape invalid: agent={a.shape}, expert={b.shape}")

    dist_state = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    if lookahead_lambda > 0:
        da = np.diff(a, axis=0, prepend=a[0:1])
        db = np.diff(b, axis=0, prepend=b[0:1])
        dist_delta = np.linalg.norm(da[:, None, :] - db[None, :, :], axis=2)
        return dist_state + float(lookahead_lambda) * dist_delta
    return dist_state


def sinkhorn_transport(
    cost: np.ndarray,
    *,
    epsilon: float,
    n_iter: int,
) -> np.ndarray:
    c = np.array(cost, dtype=np.float64)
    n, m = c.shape
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    a = np.full((n,), 1.0 / n, dtype=np.float64)
    b = np.full((m,), 1.0 / m, dtype=np.float64)

    eps = max(float(epsilon), 1e-4)
    k = np.exp(-c / eps)
    k = np.clip(k, 1e-300, None)

    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max(1, int(n_iter))):
        kv = k @ v + 1e-12
        u = a / kv
        ktu = k.T @ u + 1e-12
        v = b / ktu

    pi = (u[:, None] * k) * v[None, :]
    z = float(pi.sum())
    if z > 1e-12:
        pi = pi / z
    return pi


def dense_rewards_from_transport(pi: np.ndarray, cost: np.ndarray) -> np.ndarray:
    plan = np.array(pi, dtype=np.float64)
    c = np.array(cost, dtype=np.float64)
    if plan.size == 0:
        return np.zeros((0,), dtype=np.float32)

    row_mass = plan.sum(axis=1)
    row_cost = (plan * c).sum(axis=1) / (row_mass + 1e-12)
    rewards = -row_cost

    rmin = float(np.min(rewards))
    rmax = float(np.max(rewards))
    if rmax - rmin > 1e-9:
        rewards = (rewards - rmin) / (rmax - rmin)
    else:
        rewards = np.full_like(rewards, 0.5)

    return rewards.astype(np.float32)


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = float(rewards[i]) + float(gamma) * g
        out[i] = g
    return out


class AlignmentPretrainAgent:
    def __init__(
        self,
        *,
        dataset_dir: Path,
        embeddings_file: Path,
        profiles_file: Path,
        policy_out: Path,
        report_out: Path,
        traj_preview_out: Path,
        agent_step_logs_out: Path | None,
        intent_encoder_out: Path,
        intent_stats_out: Path,
        max_users: int,
        max_steps: int,
        min_seq_len: int,
        slate_size: int,
        epochs: int,
        lr: float,
        weight_decay: float,
        gamma: float,
        entropy_coef: float,
        imitation_coef: float,
        imitation_decay: float,
        sinkhorn_eps: float,
        sinkhorn_iters: int,
        lookahead_lambda: float,
        skip_recovery_boost: float,
        alignment_mode: str,
        ugw_eps: float,
        ugw_mass_reg: float,
        ugw_outer_iters: int,
        ugw_inner_iters: int,
        ugw_tol: float,
        expert_time_weight: float,
        agent_delta_weight: float,
        agent_time_weight: float,
        ugw_init_temperature: float,
        ugw_init_delta_weight: float,
        ugw_init_time_weight: float,
        ugw_init_sinkhorn_iters: int,
        ugw_init_reg_epochs: int,
        ugw_init_reg_batch_size: int,
        ugw_init_reg_lr: float,
        ugw_init_hidden_dim: int,
        ugw_init_user_dim: int,
        hidden_dim: int,
        stochastic: bool,
        ablation_wo_template_random_selection: bool,
        seed: int,
        device: str,
        disable_icvf: bool,
        icvf_intent_dim: int,
        icvf_hidden_dim: int,
        icvf_epochs: int,
        icvf_steps_per_epoch: int,
        icvf_batch_size: int,
        icvf_lr: float,
        icvf_weight_decay: float,
        icvf_expectile: float,
        icvf_future_window: int,
        icvf_lin_alpha: float,
        icvf_lin_coef: float,
        icvf_norm_coef: float,
        icvf_target_tau: float,
    ):
        self.dataset_dir = dataset_dir
        self.embeddings_file = embeddings_file
        self.profiles_file = profiles_file
        self.policy_out = policy_out
        self.report_out = report_out
        self.traj_preview_out = traj_preview_out
        self.agent_step_logs_out = agent_step_logs_out
        self.intent_encoder_out = intent_encoder_out
        self.intent_stats_out = intent_stats_out

        self.max_users = max(1, int(max_users))
        self.max_steps = max(5, int(max_steps))
        self.min_seq_len = max(2, int(min_seq_len))
        self.slate_size = max(2, int(slate_size))
        self.epochs = max(1, int(epochs))
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.imitation_coef = float(imitation_coef)
        self.imitation_decay = float(imitation_decay)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.lookahead_lambda = float(lookahead_lambda)
        self.skip_recovery_boost = float(skip_recovery_boost)
        mode = str(alignment_mode).strip().lower()
        if mode not in {"softlabel_ugw", "legacy_rl"}:
            raise ValueError(f"unsupported alignment_mode: {alignment_mode}")
        self.alignment_mode = mode
        self.ugw_eps = float(ugw_eps)
        self.ugw_mass_reg = float(ugw_mass_reg)
        self.ugw_outer_iters = int(ugw_outer_iters)
        self.ugw_inner_iters = int(ugw_inner_iters)
        self.ugw_tol = float(ugw_tol)
        self.expert_time_weight = float(max(0.0, expert_time_weight))
        self.agent_delta_weight = float(max(0.0, agent_delta_weight))
        self.agent_time_weight = float(max(0.0, agent_time_weight))
        self.ugw_init_temperature = float(max(1e-4, ugw_init_temperature))
        self.ugw_init_delta_weight = float(max(0.0, ugw_init_delta_weight))
        self.ugw_init_time_weight = float(max(0.0, ugw_init_time_weight))
        self.ugw_init_sinkhorn_iters = max(5, int(ugw_init_sinkhorn_iters))
        self.ugw_init_reg_epochs = max(1, int(ugw_init_reg_epochs))
        self.ugw_init_reg_batch_size = max(32, int(ugw_init_reg_batch_size))
        self.ugw_init_reg_lr = float(max(1e-6, ugw_init_reg_lr))
        self.ugw_init_hidden_dim = max(16, int(ugw_init_hidden_dim))
        self.ugw_init_user_dim = max(4, int(ugw_init_user_dim))
        self.hidden_dim = max(4, int(hidden_dim))
        self.stochastic = bool(stochastic)
        self.ablation_wo_template_random_selection = bool(
            ablation_wo_template_random_selection
        )
        self.seed = int(seed)
        self.device = self._resolve_device(device)
        # ICVF pretraining is removed from runtime pipeline.
        self.icvf_enabled = False
        self.icvf_intent_dim = max(4, int(icvf_intent_dim))
        self.icvf_hidden_dim = max(8, int(icvf_hidden_dim))
        self.icvf_epochs = max(1, int(icvf_epochs))
        self.icvf_steps_per_epoch = max(1, int(icvf_steps_per_epoch))
        self.icvf_batch_size = max(16, int(icvf_batch_size))
        self.icvf_lr = float(icvf_lr)
        self.icvf_weight_decay = float(icvf_weight_decay)
        self.icvf_expectile = float(max(0.01, min(0.99, icvf_expectile)))
        self.icvf_future_window = max(1, int(icvf_future_window))
        self.icvf_lin_alpha = float(icvf_lin_alpha)
        self.icvf_lin_coef = float(max(0.0, icvf_lin_coef))
        self.icvf_norm_coef = float(max(0.0, icvf_norm_coef))
        self.icvf_target_tau = float(max(1e-4, min(1.0, icvf_target_tau)))

        self.rng = np.random.default_rng(self.seed)

        self.item_features: dict[str, ItemFeature] = {}
        self.item_pool: list[str] = []
        self.emb_dim = 512
        self.user_sequences: list[tuple[str, list[str]]] = []
        self.expert_states: dict[str, np.ndarray] = {}
        self.expert_intents: dict[str, np.ndarray] = {}
        self.profile_pool: list[dict[str, Any]] = []
        self.icvf_model: ICVFIntentModel | None = None
        self.ugw_init_regressor: UGWInitRegressor | None = None
        self.ugw_init_feature_dim: int = 0
        self.ugw_init_regression_stats: dict[str, Any] = {}
        self.icvf_training_stats: dict[str, Any] = {}
        self.agent_step_logs: list[dict[str, Any]] = []

        self.psy_engine_cls: type | None = None

    @staticmethod
    def _resolve_device(device_text: str) -> str:
        text = (device_text or "auto").strip().lower()
        if text != "auto":
            return text
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_psy_engine_class(self) -> type:
        if not PSY_ENGINE_FILE.exists():
            raise FileNotFoundError(f"Missing Psy-engine.py: {PSY_ENGINE_FILE}")
        spec = importlib.util.spec_from_file_location("psy_engine_runtime", str(PSY_ENGINE_FILE))
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load Psy-engine module spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, "PsyBerEngine", None)
        if cls is None:
            raise RuntimeError("PsyBerEngine class not found in Psy-engine.py")
        return cls

    def _load_item_features(self) -> None:
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")

        raw = json.loads(self.embeddings_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Embeddings json must be dict: {self.embeddings_file}")

        first_dim = None
        for item_id, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            psy = payload.get("psy_input") or {}
            if not isinstance(psy, dict):
                continue
            v = psy.get("item_v")
            if not isinstance(v, list) or not v:
                continue
            first_dim = len(v)
            break
        if first_dim is not None:
            self.emb_dim = int(first_dim)

        features: dict[str, ItemFeature] = {}
        for item_id, payload in raw.items():
            sid = str(item_id)
            if not isinstance(payload, dict):
                continue
            psy = payload.get("psy_input")
            if not isinstance(psy, dict):
                continue

            item_v = psy.get("item_v")
            if isinstance(item_v, list) and len(item_v) == self.emb_dim:
                vec = np.array(item_v, dtype=np.float32)
            else:
                vec = deterministic_unit_vector(f"item::{sid}", self.emb_dim)

            feat = ItemFeature(
                item_id=sid,
                item_v=vec,
                item_n=as_float_01(psy.get("item_n"), stable_float_01(f"n::{sid}")),
                item_c=as_float_01(psy.get("item_c"), stable_float_01(f"c::{sid}")),
                item_pop=as_float_01(psy.get("item_pop"), stable_float_01(f"pop::{sid}")),
            )
            features[sid] = feat

        if not features:
            raise RuntimeError("No valid item psy_input found in embeddings file")

        self.item_features = features
        self.item_pool = sorted(features.keys())

    def _load_profiles(self) -> None:
        self.profile_pool = []
        if not self.profiles_file.exists():
            return
        try:
            raw = json.loads(self.profiles_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(raw, list):
            self.profile_pool = [x for x in raw if isinstance(x, dict)]

    def _ensure_persona_vectors(self, profile: dict[str, Any], user_id: str) -> dict[str, Any]:
        out = dict(profile)
        for key in ("u_base", "u_ideal", "u_real"):
            value = out.get(key)
            if isinstance(value, list) and len(value) == self.emb_dim:
                arr = np.array(value, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 1e-12:
                    arr = arr / norm
                out[key] = [float(x) for x in arr.tolist()]
            else:
                arr = deterministic_unit_vector(f"{key}::{user_id}", self.emb_dim)
                out[key] = [float(x) for x in arr.tolist()]
        return out

    def _profile_for_user(self, user_id: str) -> dict[str, Any]:
        if self.profile_pool:
            idx = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8], 16) % len(self.profile_pool)
            return self._ensure_persona_vectors(self.profile_pool[idx], user_id)
        return self._ensure_persona_vectors({}, user_id)

    def _new_engine_for_user(self, user_id: str) -> Any:
        if self.psy_engine_cls is None:
            raise RuntimeError("psy_engine_cls not loaded")
        profile = self._profile_for_user(user_id)
        return self.psy_engine_cls(
            emb_dim=self.emb_dim,
            persona=profile,
            stochastic=False,
            seed=self.seed,
        )

    def _get_item_feature(self, item_id: str) -> ItemFeature:
        sid = str(item_id)
        feat = self.item_features.get(sid)
        if feat is not None:
            return feat
        return ItemFeature(
            item_id=sid,
            item_v=deterministic_unit_vector(f"item::{sid}", self.emb_dim),
            item_n=stable_float_01(f"n::{sid}"),
            item_c=stable_float_01(f"c::{sid}"),
            item_pop=stable_float_01(f"pop::{sid}"),
        )

    def _load_user_sequences(self) -> None:
        ratings_path = self.dataset_dir / "ratings.dat"
        if not ratings_path.exists():
            raise FileNotFoundError(f"ratings.dat not found: {ratings_path}")

        df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
            usecols=[0, 1, 3],
        )
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)

        df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
        grouped = df.groupby("user_id")["item_id"].apply(list)

        users = []
        for user_id, item_seq in grouped.items():
            seq = [str(x) for x in item_seq if str(x)]
            if len(seq) < self.min_seq_len:
                continue
            if len(seq) > self.max_steps:
                seq = seq[: self.max_steps]
            if len(seq) >= self.min_seq_len:
                users.append((str(user_id), seq))

        if not users:
            raise RuntimeError("No user trajectory left after filtering")

        if len(users) > self.max_users:
            idx = self.rng.choice(len(users), size=self.max_users, replace=False)
            idx = sorted([int(i) for i in idx.tolist()])
            users = [users[i] for i in idx]

        self.user_sequences = users

    def _build_expert_intent_states(self) -> None:
        out: dict[str, np.ndarray] = {}
        for user_id, seq in self.user_sequences:
            engine = self._new_engine_for_user(user_id)
            states: list[np.ndarray] = []
            for item_id in seq:
                feat = self._get_item_feature(item_id)
                engine.consume_item(feat.item_v, feat.item_n, feat.item_c, feat.item_pop)
                states.append(state_dict_to_vec(engine.export_state()))
            out[user_id] = np.stack(states, axis=0).astype(np.float32)
        self.expert_states = out

    def _build_ugw_regression_features(
        self,
        *,
        user_id: str,
        item_vecs: np.ndarray,
    ) -> np.ndarray:
        vecs = np.array(item_vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.emb_dim:
            raise ValueError(f"item_vecs shape mismatch: expected [N,{self.emb_dim}], got {vecs.shape}")

        m = int(vecs.shape[0])
        feat_dim = self.ugw_init_user_dim + (2 * self.emb_dim) + 1
        if m <= 0:
            return np.zeros((0, feat_dim), dtype=np.float32)

        user_vec = deterministic_unit_vector(f"user::{user_id}", self.ugw_init_user_dim).astype(np.float32)
        user_tile = np.repeat(user_vec[None, :], m, axis=0)

        prefix_mean = np.zeros_like(vecs, dtype=np.float32)
        if m > 1:
            csum = np.cumsum(vecs, axis=0, dtype=np.float32)
            denom = np.arange(1, m, dtype=np.float32)[:, None]
            prefix_mean[1:] = csum[:-1] / np.maximum(denom, 1.0)

        rel_pos = np.arange(m, dtype=np.float32)
        if m > 1:
            rel_pos = rel_pos / float(m - 1)
        else:
            rel_pos[:] = 0.0

        return np.concatenate(
            [user_tile, prefix_mean, vecs, rel_pos[:, None]],
            axis=1,
        ).astype(np.float32)

    def _train_ugw_init_regressor(self) -> None:
        feature_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []

        for user_id, seq in self.user_sequences:
            gt_states = self.expert_states.get(user_id)
            if gt_states is None:
                continue
            gt = np.array(gt_states, dtype=np.float32)
            if gt.ndim != 2 or gt.shape[1] != 4 or gt.shape[0] == 0:
                continue

            m = min(len(seq), int(gt.shape[0]))
            if m <= 0:
                continue
            item_vecs = self._expert_item_vectors(seq[:m])
            if item_vecs.shape[0] != m:
                continue

            feats = self._build_ugw_regression_features(
                user_id=user_id,
                item_vecs=item_vecs,
            )
            feature_rows.append(feats)
            target_rows.append(gt[:m].astype(np.float32))

        if not feature_rows:
            raise RuntimeError("UGW init regression dataset is empty")

        x_np = np.concatenate(feature_rows, axis=0).astype(np.float32)
        y_np = np.concatenate(target_rows, axis=0).astype(np.float32)
        n_samples, in_dim = x_np.shape
        self.ugw_init_feature_dim = int(in_dim)

        device = torch.device(self.device)
        model = UGWInitRegressor(input_dim=in_dim, hidden_dim=self.ugw_init_hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.ugw_init_reg_lr, weight_decay=1e-5)

        history: list[dict[str, float]] = []
        for epoch in range(1, self.ugw_init_reg_epochs + 1):
            order = self.rng.permutation(n_samples)
            total_loss = 0.0
            batch_count = 0
            model.train()
            for start in range(0, n_samples, self.ugw_init_reg_batch_size):
                idx = order[start : start + self.ugw_init_reg_batch_size]
                xb = torch.tensor(x_np[idx], dtype=torch.float32, device=device)
                yb = torch.tensor(y_np[idx], dtype=torch.float32, device=device)
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
                batch_count += 1

            rec = {
                "epoch": float(epoch),
                "mse": total_loss / max(1, batch_count),
            }
            history.append(rec)
            print(
                "[UGW-Init] "
                f"epoch={epoch}/{self.ugw_init_reg_epochs} "
                f"mse={rec['mse']:.6f}"
            )

        model.eval()
        mae_sum = 0.0
        mae_count = 0
        with torch.no_grad():
            for start in range(0, n_samples, self.ugw_init_reg_batch_size):
                xb = torch.tensor(
                    x_np[start : start + self.ugw_init_reg_batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                yb = torch.tensor(
                    y_np[start : start + self.ugw_init_reg_batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                pred = model(xb)
                mae_sum += float(torch.sum(torch.abs(pred - yb)).item())
                mae_count += int(yb.numel())

        self.ugw_init_regressor = model
        self.ugw_init_regression_stats = {
            "enabled": True,
            "samples": int(n_samples),
            "feature_dim": int(in_dim),
            "hidden_dim": int(self.ugw_init_hidden_dim),
            "epochs": int(self.ugw_init_reg_epochs),
            "batch_size": int(self.ugw_init_reg_batch_size),
            "lr": float(self.ugw_init_reg_lr),
            "train_mae": float(mae_sum / max(1, mae_count)),
            "history_tail": history[-3:] if len(history) > 3 else history,
        }

    def _predict_regressed_expert_states(
        self,
        *,
        user_id: str,
        expert_item_seq: list[str],
    ) -> np.ndarray:
        item_vecs = self._expert_item_vectors(expert_item_seq)
        m = int(item_vecs.shape[0])
        if m <= 0:
            return np.zeros((0, 4), dtype=np.float32)

        if self.ugw_init_regressor is None:
            gt = self.expert_states.get(user_id)
            if isinstance(gt, np.ndarray) and gt.ndim == 2 and gt.shape[1] == 4 and gt.shape[0] >= m:
                return np.array(gt[:m], dtype=np.float32)
            return np.zeros((m, 4), dtype=np.float32)

        feats = self._build_ugw_regression_features(
            user_id=user_id,
            item_vecs=item_vecs,
        )
        device = torch.device(self.device)
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            pred = self.ugw_init_regressor(x).detach().cpu().numpy().astype(np.float32)
        return pred

    def _build_ugw_init_cost(
        self,
        *,
        agent_states: np.ndarray,
        pseudo_expert_states: np.ndarray,
    ) -> np.ndarray:
        a = np.array(agent_states, dtype=np.float32)
        b = np.array(pseudo_expert_states, dtype=np.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 4 or b.shape[1] != 4:
            raise ValueError(f"UGW init states shape invalid: agent={a.shape}, expert={b.shape}")

        dist_state = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        da = np.diff(a, axis=0, prepend=a[0:1])
        db = np.diff(b, axis=0, prepend=b[0:1])
        dist_delta = np.linalg.norm(da[:, None, :] - db[None, :, :], axis=2)
        dist_time = cross_normalized_time_distance(a.shape[0], b.shape[0])

        return (
            dist_state
            + self.ugw_init_delta_weight * dist_delta
            + self.ugw_init_time_weight * dist_time
        ).astype(np.float32)

    def _sample_icvf_batch(
        self,
        trajectories: list[np.ndarray],
        *,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        s_list: list[np.ndarray] = []
        sp_list: list[np.ndarray] = []
        sz_list: list[np.ndarray] = []
        splus_list: list[np.ndarray] = []
        k_list: list[float] = []
        reward_list: list[float] = []

        for _ in range(batch_size):
            traj = trajectories[int(self.rng.integers(0, len(trajectories)))]
            length = int(traj.shape[0])
            if length < 3:
                continue

            t = int(self.rng.integers(0, length - 1))
            future_start = t + 1
            future_end = min(length - 1, t + self.icvf_future_window)
            if future_start > future_end:
                future_start = length - 1
                future_end = length - 1
            tz = int(self.rng.integers(future_start, future_end + 1))
            tplus = int(self.rng.integers(future_start, future_end + 1))

            s = traj[t]
            s_prime = traj[min(t + 1, length - 1)]
            s_z = traj[tz]
            s_plus = traj[tplus]
            k = float(max(1, tz - t))

            s_list.append(s.astype(np.float32))
            sp_list.append(s_prime.astype(np.float32))
            sz_list.append(s_z.astype(np.float32))
            splus_list.append(s_plus.astype(np.float32))
            k_list.append(k)
            # In continuous state spaces, exact equality is rare, so this is almost always -1.
            reward_list.append(-1.0 if not np.allclose(s, s_z, atol=1e-6) else 0.0)

        if not s_list:
            raise RuntimeError("ICVF batch sampling failed: no valid sample")

        def _to_tensor(xs: list[np.ndarray]) -> torch.Tensor:
            return torch.tensor(np.stack(xs, axis=0), dtype=torch.float32, device=device)

        return {
            "s": _to_tensor(s_list),
            "s_prime": _to_tensor(sp_list),
            "s_z": _to_tensor(sz_list),
            "s_plus": _to_tensor(splus_list),
            "k": torch.tensor(k_list, dtype=torch.float32, device=device),
            "reward": torch.tensor(reward_list, dtype=torch.float32, device=device),
        }

    def _polyak_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p_tgt, p_src in zip(target.parameters(), source.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p_src.data)

    def _train_icvf_intent_mapper(self) -> None:
        if not self.icvf_enabled:
            print("[ICVF] disabled by config, OT will use raw 4D states.")
            self.expert_intents = {k: np.array(v, dtype=np.float32) for k, v in self.expert_states.items()}
            self.icvf_training_stats = {
                "enabled": False,
                "reason": "disabled_by_flag",
                "intent_dim": 4,
            }
            return

        trajectories = [
            np.array(states, dtype=np.float32)
            for _, states in self.expert_states.items()
            if isinstance(states, np.ndarray) and states.ndim == 2 and states.shape[0] >= 3 and states.shape[1] == 4
        ]
        if not trajectories:
            raise RuntimeError("ICVF needs at least one trajectory with length >= 3 and dim=4")

        device = torch.device(self.device)
        model = ICVFIntentModel(
            state_dim=4,
            intent_dim=self.icvf_intent_dim,
            hidden_dim=self.icvf_hidden_dim,
        ).to(device)
        target_model = copy.deepcopy(model).to(device)
        target_model.eval()
        for p in target_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.icvf_lr,
            weight_decay=self.icvf_weight_decay,
        )

        hist: list[dict[str, float]] = []
        for epoch in range(1, self.icvf_epochs + 1):
            model.train()
            epoch_exp = 0.0
            epoch_lin = 0.0
            epoch_norm = 0.0
            epoch_total = 0.0
            n_steps = 0

            for _ in range(self.icvf_steps_per_epoch):
                batch = self._sample_icvf_batch(
                    trajectories,
                    batch_size=self.icvf_batch_size,
                    device=device,
                )
                s = batch["s"]
                s_prime = batch["s_prime"]
                s_z = batch["s_z"]
                s_plus = batch["s_plus"]
                k = batch["k"]
                reward = batch["reward"]

                z = model.encode_intent(s_z)
                v = model.value(s, s_plus, z)

                with torch.no_grad():
                    z_tgt = target_model.encode_intent(s_z)
                    v_next = target_model.value(s_prime, s_plus, z_tgt)

                delta = reward + self.gamma * v_next - v
                w = torch.where(delta < 0.0, 1.0 - self.icvf_expectile, self.icvf_expectile)
                loss_exp = torch.mean(w * delta.pow(2))

                z_s = model.encode_intent(s)
                z_sz = model.encode_intent(s_z)
                dist_sq = torch.sum((z_sz - z_s).pow(2), dim=-1)
                target_dist_sq = self.icvf_lin_alpha * k
                loss_lin = torch.mean((dist_sq - target_dist_sq).pow(2))

                loss_norm = torch.mean((torch.linalg.norm(z_s, dim=-1) - 1.0).pow(2))

                loss = loss_exp + self.icvf_lin_coef * loss_lin + self.icvf_norm_coef * loss_norm

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                self._polyak_update(target_model, model, self.icvf_target_tau)

                n_steps += 1
                epoch_exp += float(loss_exp.item())
                epoch_lin += float(loss_lin.item())
                epoch_norm += float(loss_norm.item())
                epoch_total += float(loss.item())

            rec = {
                "epoch": float(epoch),
                "steps": float(n_steps),
                "loss_exp": epoch_exp / max(1, n_steps),
                "loss_lin": epoch_lin / max(1, n_steps),
                "loss_norm": epoch_norm / max(1, n_steps),
                "loss_total": epoch_total / max(1, n_steps),
            }
            hist.append(rec)
            print(
                "[ICVF] "
                f"epoch={epoch}/{self.icvf_epochs} "
                f"loss_total={rec['loss_total']:.6f} "
                f"loss_exp={rec['loss_exp']:.6f} "
                f"loss_lin={rec['loss_lin']:.6f} "
                f"loss_norm={rec['loss_norm']:.6f}"
            )

        model.eval()
        self.icvf_model = model

        # Save encoder bundle for downstream reuse.
        self.intent_encoder_out.parent.mkdir(parents=True, exist_ok=True)
        encoder_bundle = {
            "state_dict": model.psi_net.state_dict(),
            "metadata": {
                "state_dim": 4,
                "intent_dim": self.icvf_intent_dim,
                "hidden_dim": self.icvf_hidden_dim,
                "trained_utc": datetime.now(timezone.utc).isoformat(),
                "gamma": self.gamma,
                "expectile": self.icvf_expectile,
                "future_window": self.icvf_future_window,
            },
        }
        torch.save(encoder_bundle, self.intent_encoder_out)

        self.expert_intents = {
            user_id: self._map_states_to_intents(states)
            for user_id, states in self.expert_states.items()
        }
        tail = hist[-3:] if len(hist) > 3 else hist
        self.icvf_training_stats = {
            "enabled": True,
            "intent_dim": self.icvf_intent_dim,
            "hidden_dim": self.icvf_hidden_dim,
            "epochs": self.icvf_epochs,
            "steps_per_epoch": self.icvf_steps_per_epoch,
            "batch_size": self.icvf_batch_size,
            "lr": self.icvf_lr,
            "weight_decay": self.icvf_weight_decay,
            "expectile": self.icvf_expectile,
            "future_window": self.icvf_future_window,
            "lin_alpha": self.icvf_lin_alpha,
            "lin_coef": self.icvf_lin_coef,
            "norm_coef": self.icvf_norm_coef,
            "target_tau": self.icvf_target_tau,
            "history_tail": tail,
            "encoder_out": str(self.intent_encoder_out),
        }
        self.intent_stats_out.parent.mkdir(parents=True, exist_ok=True)
        self.intent_stats_out.write_text(
            json.dumps(self.icvf_training_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _map_states_to_intents(self, states: np.ndarray) -> np.ndarray:
        arr = np.array(states, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"states must be 2D, got {arr.shape}")
        if arr.shape[1] != 4:
            return arr.astype(np.float32)
        if not self.icvf_enabled or self.icvf_model is None:
            return arr.astype(np.float32)

        device = torch.device(self.device)
        with torch.no_grad():
            s = torch.tensor(arr, dtype=torch.float32, device=device)
            z = self.icvf_model.encode_intent(s)
            out = z.detach().cpu().numpy().astype(np.float32)
        return out

    def _sample_negative_items(
        self,
        *,
        target_item: str,
        k: int,
        rng: np.random.Generator,
    ) -> list[str]:
        exclude = {str(target_item)}
        out: list[str] = []
        max_try = max(100, k * 20)
        tries = 0
        while len(out) < k and tries < max_try:
            tries += 1
            item_id = str(self.item_pool[int(rng.integers(0, len(self.item_pool)))])
            if item_id in exclude:
                continue
            exclude.add(item_id)
            out.append(item_id)

        if len(out) < k:
            # deterministic fallback if random draw exhausted
            for item_id in self.item_pool:
                sid = str(item_id)
                if sid in exclude:
                    continue
                exclude.add(sid)
                out.append(sid)
                if len(out) >= k:
                    break

        return out[:k]

    @staticmethod
    def _stable_index(key: str, modulo: int) -> int:
        if modulo <= 0:
            return 0
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % modulo

    def _sample_negative_items_deterministic(
        self,
        *,
        target_item: str,
        k: int,
        user_id: str,
        step_idx: int,
    ) -> list[str]:
        target = str(target_item)
        pool_size = len(self.item_pool)
        if k <= 0 or pool_size <= 1:
            return []

        start = self._stable_index(
            f"tmpl-start::{user_id}::{step_idx}::{target}",
            pool_size,
        )
        if pool_size == 2:
            stride = 1
        else:
            stride = 1 + self._stable_index(
                f"tmpl-stride::{user_id}::{step_idx}::{target}",
                pool_size - 1,
            )
            if pool_size % 2 == 0 and stride % 2 == 0:
                stride += 1
            stride = stride % pool_size
            if stride == 0:
                stride = 1

        out: list[str] = []
        seen = {target}
        for i in range(pool_size * 2):
            sid = str(self.item_pool[(start + i * stride) % pool_size])
            if sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
            if len(out) >= k:
                return out[:k]

        for sid in self.item_pool:
            item_id = str(sid)
            if item_id in seen:
                continue
            seen.add(item_id)
            out.append(item_id)
            if len(out) >= k:
                break
        return out[:k]

    def _build_slate(
        self,
        target_item: str,
        rng: np.random.Generator,
        *,
        user_id: str | None = None,
        step_idx: int | None = None,
    ) -> list[str]:
        target = str(target_item)
        if self.ablation_wo_template_random_selection:
            uid = user_id or "global"
            step = 0 if step_idx is None else int(step_idx)
            negatives = self._sample_negative_items_deterministic(
                target_item=target,
                k=self.slate_size - 1,
                user_id=uid,
                step_idx=step,
            )
            insert_pos = self._stable_index(
                f"tmpl-pos::{uid}::{step}::{target}",
                self.slate_size,
            )
        else:
            negatives = self._sample_negative_items(
                target_item=target,
                k=self.slate_size - 1,
                rng=rng,
            )
            insert_pos = int(rng.integers(0, self.slate_size))

        slate = list(negatives)
        slate.insert(insert_pos, target)
        return slate

    def _simulate_user_rollout(
        self,
        *,
        model: FastPolicyNet,
        user_id: str,
        item_seq: list[str],
        rng: np.random.Generator,
        train_mode: bool,
    ) -> dict[str, Any]:
        device = torch.device(self.device)
        engine = self._new_engine_for_user(user_id)

        step_entries: list[dict[str, Any]] = []
        agent_states: list[np.ndarray] = []
        click_count = 0
        collect_step_debug = train_mode and (self.agent_step_logs_out is not None)

        for step_idx, item_id in enumerate(item_seq):
            slate_ids = self._build_slate(
                item_id,
                rng,
                user_id=user_id,
                step_idx=step_idx,
            )
            target_idx = slate_ids.index(str(item_id))
            candidates: list[dict[str, Any]] = []
            for sid in slate_ids:
                feat = self._get_item_feature(sid)
                candidates.append(
                    {
                        "item_id": feat.item_id,
                        "title": f"item_{feat.item_id}",
                        "v": feat.item_v,
                        "n": feat.item_n,
                        "c": feat.item_c,
                        "pop": feat.item_pop,
                    }
                )

            state_before = state_dict_to_vec(engine.export_state())
            action_mat = simulate_action_matrix(
                engine,
                candidates,
                include_skip=True,
                recovery_boost=self.skip_recovery_boost,
            )
            actions = action_mat["actions"]
            deltas = np.array(action_mat["deltas"], dtype=np.float32)
            pareto = pareto_front_indices(deltas)
            valid_indices = [int(i) for i in pareto if 0 <= int(i) < len(actions)]
            valid_indices, skip_allowed, _ = apply_skip_guard_to_candidates(
                state_before,
                actions,
                deltas,
                valid_indices,
            )
            if not valid_indices:
                valid_indices = [
                    idx for idx, action in enumerate(actions) if str(action.get("action") or "") == "click"
                ]
                if not valid_indices:
                    valid_indices = [0]

            feats = build_action_features(state_before, actions, deltas)
            x = torch.tensor(feats, dtype=torch.float32, device=device)
            logits = model(x).squeeze(-1)

            mask = torch.full_like(logits, -1e9)
            for idx in valid_indices:
                if 0 <= int(idx) < logits.shape[0]:
                    mask[int(idx)] = 0.0
            # Always allow the ground-truth target item during training so
            # imitation loss and policy action space stay consistent even
            # when the target click is Pareto-dominated by other actions.
            if 0 <= target_idx < mask.shape[0]:
                mask[target_idx] = 0.0
            logits_masked = logits + mask
            probs = torch.softmax(logits_masked, dim=0)

            if self.stochastic:
                action_idx = int(torch.multinomial(probs, num_samples=1).item())
            else:
                action_idx = int(torch.argmax(probs).item())

            selected = actions[action_idx]
            selected_action = str(selected.get("action"))

            if selected_action == "click":
                click_count += 1
                engine.consume_item(selected["v"], selected["n"], selected["c"], selected["pop"])
            else:
                apply_skip_to_engine(engine, recovery_boost=self.skip_recovery_boost)

            state_after = state_dict_to_vec(engine.export_state())
            agent_states.append(state_after)

            if train_mode:
                log_prob = torch.log(probs[action_idx] + 1e-12)
                entropy = -(probs * torch.log(probs + 1e-12)).sum()
                entry: dict[str, Any] = {
                    "log_prob": log_prob,
                    "entropy": entropy,
                    "logits": logits,
                    "target_idx": int(target_idx),
                    "step_idx": int(step_idx),
                    "action_idx": int(action_idx),
                    "selected_action": selected_action,
                    "pareto_indices": [int(i) for i in pareto],
                    "valid_indices": [int(i) for i in valid_indices],
                    "skip_allowed": bool(skip_allowed),
                    "slate_ids": [str(sid) for sid in slate_ids],
                }
                if collect_step_debug:
                    entry.update(
                        {
                            "state_before": state_before.astype(np.float32),
                            "state_after": state_after.astype(np.float32),
                            "logits_np": logits.detach().cpu().numpy().astype(np.float32),
                            "probs_np": probs.detach().cpu().numpy().astype(np.float32),
                        }
                    )
                step_entries.append(entry)

        states_np = np.stack(agent_states, axis=0).astype(np.float32) if agent_states else np.zeros((0, 4), dtype=np.float32)
        return {
            "states": states_np,
            "steps": step_entries,
            "click_count": click_count,
            "step_count": int(len(agent_states)),
        }

    def _compute_ot_rewards(
        self,
        *,
        agent_states: np.ndarray,
        expert_states: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if agent_states.size == 0 or expert_states.size == 0:
            return np.zeros((0,), dtype=np.float32), 0.0

        agent_repr = self._map_states_to_intents(agent_states)
        expert_repr = self._map_states_to_intents(expert_states)
        cost = build_cost_matrix(
            agent_repr,
            expert_repr,
            lookahead_lambda=self.lookahead_lambda,
        )
        pi = sinkhorn_transport(
            cost,
            epsilon=self.sinkhorn_eps,
            n_iter=self.sinkhorn_iters,
        )
        rewards = dense_rewards_from_transport(pi, cost)
        align_cost = float(np.mean(cost))
        return rewards, align_cost

    def _expert_item_vectors(self, item_seq: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for item_id in item_seq:
            feat = self._get_item_feature(item_id)
            rows.append(np.array(feat.item_v, dtype=np.float32))
        if not rows:
            return np.zeros((0, self.emb_dim), dtype=np.float32)
        return np.stack(rows, axis=0).astype(np.float32)

    def _compute_ugw_transport(
        self,
        *,
        user_id: str,
        agent_states: np.ndarray,
        expert_item_seq: list[str],
    ) -> tuple[np.ndarray, float]:
        a_states = np.array(agent_states, dtype=np.float32)
        item_vecs = self._expert_item_vectors(expert_item_seq)
        if a_states.size == 0 or item_vecs.size == 0:
            return np.zeros((a_states.shape[0], item_vecs.shape[0]), dtype=np.float64), 0.0

        agent_dist = build_agent_structure_distance(
            a_states,
            delta_weight=self.agent_delta_weight,
            time_weight=self.agent_time_weight,
        )
        expert_dist = build_expert_structure_distance(
            item_vecs,
            time_weight=self.expert_time_weight,
        )

        pseudo_states = self._predict_regressed_expert_states(
            user_id=user_id,
            expert_item_seq=expert_item_seq,
        )
        init_cost = self._build_ugw_init_cost(
            agent_states=a_states,
            pseudo_expert_states=pseudo_states,
        )
        init_plan = sinkhorn_transport(
            init_cost,
            epsilon=self.ugw_init_temperature,
            n_iter=self.ugw_init_sinkhorn_iters,
        )

        pi, struct_cost = ugw_transport(
            agent_dist,
            expert_dist,
            epsilon=self.ugw_eps,
            mass_reg=self.ugw_mass_reg,
            outer_iters=self.ugw_outer_iters,
            inner_iters=self.ugw_inner_iters,
            tol=self.ugw_tol,
            initial_plan=init_plan,
        )
        denom = float(pi.sum())
        align_cost = float((pi * struct_cost).sum() / (denom + 1e-12))
        return pi, align_cost

    def _build_action_soft_targets(
        self,
        *,
        transport: np.ndarray,
        expert_item_seq: list[str],
        step_entries: list[dict[str, Any]],
    ) -> list[np.ndarray]:
        pi = np.array(transport, dtype=np.float64)
        m = len(expert_item_seq)
        targets: list[np.ndarray] = []

        for step_idx, entry in enumerate(step_entries):
            slate_ids = [str(x) for x in entry.get("slate_ids", [])]
            action_dim = len(slate_ids) + 1  # + skip
            if m <= 0:
                targets.append(
                    np.full((max(1, action_dim),), 1.0 / max(1, action_dim), dtype=np.float32)
                )
                continue

            if step_idx < pi.shape[0]:
                row = np.array(pi[step_idx], dtype=np.float64)
            else:
                row = np.full((m,), 1.0 / m, dtype=np.float64)
            row_sum = float(row.sum())
            if row_sum <= 1e-12 or not np.isfinite(row_sum):
                row = np.full((m,), 1.0 / m, dtype=np.float64)
            else:
                row = row / row_sum

            item_dist: dict[str, float] = {}
            for j, item_id in enumerate(expert_item_seq):
                sid = str(item_id)
                item_dist[sid] = item_dist.get(sid, 0.0) + float(row[j])

            action_probs: list[float] = []
            assigned = 0.0
            for sid in slate_ids:
                p = float(item_dist.get(str(sid), 0.0))
                action_probs.append(p)
                assigned += p

            skip_p = max(0.0, 1.0 - assigned)
            action_probs.append(skip_p)

            out = np.array(action_probs, dtype=np.float32)
            skip_allowed = bool(entry.get("skip_allowed", True))
            if (not skip_allowed) and out.size >= 2:
                out[-1] = 0.0
                click_mass = float(np.sum(out[:-1]))
                if click_mass > 1e-12 and np.isfinite(click_mass):
                    out[:-1] = out[:-1] / click_mass
                else:
                    out[:-1] = 1.0 / float(max(1, out.size - 1))
            z = float(out.sum())
            if z <= 1e-12 or not np.isfinite(z):
                out = np.full((max(1, out.size),), 1.0 / max(1, out.size), dtype=np.float32)
            else:
                out = out / z
            targets.append(out)

        return targets

    def _fit(self) -> tuple[FastPolicyNet, list[dict[str, Any]]]:
        device = torch.device(self.device)
        model = FastPolicyNet(feature_dim=FEATURE_DIM, hidden_dim=self.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        history: list[dict[str, Any]] = []
        step_debug_logs: list[dict[str, Any]] = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            order = list(range(len(self.user_sequences)))
            self.rng.shuffle(order)
            imitation_coef_epoch = float(
                self.imitation_coef * (self.imitation_decay ** max(0, epoch - 1))
            )

            epoch_loss = 0.0
            epoch_primary_loss = 0.0
            epoch_imitation_loss = 0.0
            epoch_entropy = 0.0
            epoch_reward = 0.0
            epoch_align_cost = 0.0
            epoch_steps = 0
            epoch_click_rate_sum = 0.0
            used_users = 0
            softlabel_count = 0
            softlabel_entropy_sum = 0.0
            softlabel_skip_mass_sum = 0.0
            softlabel_clicked_mass_sum = 0.0

            for idx in order:
                user_id, seq = self.user_sequences[idx]

                rollout = self._simulate_user_rollout(
                    model=model,
                    user_id=user_id,
                    item_seq=seq,
                    rng=self.rng,
                    train_mode=True,
                )
                step_entries = rollout["steps"]
                if not step_entries:
                    continue

                agent_states = np.array(rollout["states"], dtype=np.float32)
                entropies: list[torch.Tensor] = []
                imitation_terms: list[torch.Tensor] = []

                if self.alignment_mode == "legacy_rl":
                    expert_states = (
                        self.expert_intents.get(user_id)
                        if self.expert_intents
                        else self.expert_states.get(user_id)
                    )
                    if expert_states is None or len(expert_states) == 0:
                        continue

                    rewards, align_cost = self._compute_ot_rewards(
                        agent_states=agent_states,
                        expert_states=expert_states[: len(agent_states)],
                    )
                    if rewards.size == 0:
                        continue

                    returns = discounted_returns(rewards, self.gamma)
                    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
                    if returns_t.numel() > 1:
                        std = returns_t.std(unbiased=False)
                        if float(std) > 1e-3:
                            returns_t = (returns_t - returns_t.mean()) / (std + 1e-6)

                    primary_terms: list[torch.Tensor] = []
                    for t, entry in enumerate(step_entries):
                        log_prob = entry["log_prob"]
                        entropy = entry["entropy"]
                        logits_raw = entry["logits"].unsqueeze(0)
                        target = torch.tensor([entry["target_idx"]], dtype=torch.long, device=device)
                        primary_terms.append(-returns_t[t] * log_prob)
                        imitation_terms.append(F.cross_entropy(logits_raw, target))
                        entropies.append(entropy)

                        if self.agent_step_logs_out is not None:
                            logits_np = entry.get("logits_np")
                            probs_np = entry.get("probs_np")
                            logits_out = (
                                [float(x) for x in logits_np.tolist()]
                                if isinstance(logits_np, np.ndarray)
                                else [float(x) for x in entry["logits"].detach().cpu().numpy().tolist()]
                            )
                            probs_out = (
                                [float(x) for x in probs_np.tolist()]
                                if isinstance(probs_np, np.ndarray)
                                else []
                            )
                            step_debug_logs.append(
                                {
                                    "epoch": int(epoch),
                                    "mode": self.alignment_mode,
                                    "user_id": str(user_id),
                                    "step_idx": int(entry.get("step_idx", t)),
                                    "target_idx": int(entry["target_idx"]),
                                    "action_idx": int(entry.get("action_idx", -1)),
                                    "selected_action": str(entry.get("selected_action", "")),
                                    "pareto_indices": [int(i) for i in entry.get("pareto_indices", [])],
                                    "slate_ids": [str(s) for s in entry.get("slate_ids", [])],
                                    "reward": float(rewards[t]),
                                    "return": float(returns[t]),
                                    "entropy": float(entropy.detach().item()),
                                    "logits": logits_out,
                                    "probs": probs_out,
                                }
                            )
                    primary_loss = torch.stack(primary_terms).mean()
                    epoch_reward += float(np.mean(rewards))
                else:
                    plan, align_cost = self._compute_ugw_transport(
                        user_id=user_id,
                        agent_states=agent_states,
                        expert_item_seq=seq[: len(agent_states)],
                    )
                    soft_targets = self._build_action_soft_targets(
                        transport=plan,
                        expert_item_seq=seq[: len(agent_states)],
                        step_entries=step_entries,
                    )
                    primary_terms = []
                    for t, entry in enumerate(step_entries):
                        entropy = entry["entropy"]
                        logits = entry["logits"]
                        target = torch.tensor([entry["target_idx"]], dtype=torch.long, device=device)
                        soft = torch.tensor(
                            soft_targets[t],
                            dtype=torch.float32,
                            device=device,
                        )
                        logp = F.log_softmax(logits, dim=0)
                        primary_terms.append(-(soft * logp).sum())
                        imitation_terms.append(F.cross_entropy(logits.unsqueeze(0), target))
                        entropies.append(entropy)
                        soft_np = np.array(soft_targets[t], dtype=np.float64)
                        soft_np = np.clip(soft_np, 1e-12, 1.0)
                        softlabel_entropy_sum += float(-np.sum(soft_np * np.log(soft_np)))
                        softlabel_skip_mass_sum += float(soft_targets[t][-1])
                        target_idx = int(entry["target_idx"])
                        if 0 <= target_idx < len(soft_targets[t]):
                            softlabel_clicked_mass_sum += float(soft_targets[t][target_idx])
                        softlabel_count += 1

                        if self.agent_step_logs_out is not None:
                            logits_np = entry.get("logits_np")
                            probs_np = entry.get("probs_np")
                            logits_out = (
                                [float(x) for x in logits_np.tolist()]
                                if isinstance(logits_np, np.ndarray)
                                else [float(x) for x in entry["logits"].detach().cpu().numpy().tolist()]
                            )
                            probs_out = (
                                [float(x) for x in probs_np.tolist()]
                                if isinstance(probs_np, np.ndarray)
                                else []
                            )
                            step_debug_logs.append(
                                {
                                    "epoch": int(epoch),
                                    "mode": self.alignment_mode,
                                    "user_id": str(user_id),
                                    "step_idx": int(entry.get("step_idx", t)),
                                    "target_idx": int(entry["target_idx"]),
                                    "action_idx": int(entry.get("action_idx", -1)),
                                    "selected_action": str(entry.get("selected_action", "")),
                                    "pareto_indices": [int(i) for i in entry.get("pareto_indices", [])],
                                    "slate_ids": [str(s) for s in entry.get("slate_ids", [])],
                                    "soft_target": [float(x) for x in soft_targets[t].tolist()],
                                    "entropy": float(entropy.detach().item()),
                                    "logits": logits_out,
                                    "probs": probs_out,
                                }
                            )
                    primary_loss = torch.stack(primary_terms).mean()

                if not imitation_terms:
                    continue
                imitation_loss = torch.stack(imitation_terms).mean()
                entropy_bonus = torch.stack(entropies).mean()
                loss = primary_loss + imitation_coef_epoch * imitation_loss - self.entropy_coef * entropy_bonus

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                used_users += 1
                epoch_steps += int(rollout["step_count"])
                epoch_loss += float(loss.item())
                epoch_primary_loss += float(primary_loss.item())
                epoch_imitation_loss += float(imitation_loss.item())
                epoch_entropy += float(entropy_bonus.item())
                epoch_align_cost += float(align_cost)
                step_count = max(1, int(rollout["step_count"]))
                epoch_click_rate_sum += float(rollout["click_count"]) / step_count

            if used_users == 0:
                raise RuntimeError("No valid user trajectory consumed during training")

            avg_primary = epoch_primary_loss / used_users
            avg_align = epoch_align_cost / used_users
            rec = {
                "epoch": epoch,
                "alignment_mode": self.alignment_mode,
                "users": used_users,
                "avg_loss": epoch_loss / used_users,
                "avg_primary_loss": avg_primary,
                "avg_imitation_loss": epoch_imitation_loss / used_users,
                "avg_entropy": epoch_entropy / used_users,
                "imitation_coef_epoch": imitation_coef_epoch,
                "avg_reward": epoch_reward / used_users,
                "avg_alignment_cost": avg_align,
                "avg_objective_reward_minus_cost": (
                    (epoch_reward / used_users) - avg_align
                ),
                "avg_click_rate": epoch_click_rate_sum / used_users,
                "total_steps": epoch_steps,
            }
            if self.alignment_mode == "softlabel_ugw" and softlabel_count > 0:
                rec["avg_soft_target_entropy"] = softlabel_entropy_sum / softlabel_count
                rec["avg_soft_target_skip_mass"] = softlabel_skip_mass_sum / softlabel_count
                rec["avg_soft_target_clicked_item_mass"] = (
                    softlabel_clicked_mass_sum / softlabel_count
                )
            history.append(rec)
            if self.alignment_mode == "legacy_rl":
                print(
                    "[Epoch] "
                    f"{epoch}/{self.epochs} "
                    f"mode={self.alignment_mode} "
                    f"loss={rec['avg_loss']:.4f} "
                    f"policy_loss={rec['avg_primary_loss']:.4f} "
                    f"imitation_loss={rec['avg_imitation_loss']:.4f} "
                    f"entropy={rec['avg_entropy']:.4f} "
                    f"imitation_coef={rec['imitation_coef_epoch']:.4f} "
                    f"reward={rec['avg_reward']:.4f} "
                    f"align_cost={rec['avg_alignment_cost']:.4f} "
                    f"objective={rec['avg_objective_reward_minus_cost']:.4f} "
                    f"click_rate={rec['avg_click_rate']:.4f}"
                )
            else:
                print(
                    "[Epoch] "
                    f"{epoch}/{self.epochs} "
                    f"mode={self.alignment_mode} "
                    f"loss={rec['avg_loss']:.4f} "
                    f"soft_ce={rec['avg_primary_loss']:.4f} "
                    f"imitation_loss={rec['avg_imitation_loss']:.4f} "
                    f"entropy={rec['avg_entropy']:.4f} "
                    f"target_entropy={rec.get('avg_soft_target_entropy', 0.0):.4f} "
                    f"target_skip={rec.get('avg_soft_target_skip_mass', 0.0):.4f} "
                    f"target_click_mass={rec.get('avg_soft_target_clicked_item_mass', 0.0):.4f} "
                    f"align_cost={rec['avg_alignment_cost']:.4f} "
                    f"click_rate={rec['avg_click_rate']:.4f}"
                )

        self.agent_step_logs = step_debug_logs
        model.eval()
        return model, history

    def _write_traj_preview(self) -> None:
        preview: dict[str, Any] = {}
        for user_id, states in list(self.expert_states.items())[: min(30, len(self.expert_states))]:
            preview[user_id] = {
                "length": int(states.shape[0]),
                "states_head": states[:5].tolist(),
                "states_tail": states[-5:].tolist(),
            }

        self.traj_preview_out.parent.mkdir(parents=True, exist_ok=True)
        self.traj_preview_out.write_text(
            json.dumps(preview, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_doc_alignment_checklist(self) -> dict[str, Any]:
        """Checklist against:
        1) 对齐实施步骤.pdf
        2) 心理学引擎建模.pdf
        """
        return {
            "alignment_steps_pdf": [
                {
                    "id": "step1_implicit_trajectory_construction",
                    "implemented": True,
                    "evidence": "_load_user_sequences(): ratings.dat -> (user,item,timestamp) sorted trajectories",
                },
                {
                    "id": "step2_expert_intent_conversion",
                    "implemented": True,
                    "evidence": "_build_expert_intent_states(): per-item consume -> 4D [P,C,E,S]",
                },
                {
                    "id": "step2_intent_mapping_icvf_pretraining",
                    "implemented": False,
                    "evidence": "Removed from stage-02 pipeline; OT alignment now uses raw 4D psychological trajectories directly.",
                },
                {
                    "id": "step2_agent_slate_5d_action",
                    "implemented": True,
                    "evidence": "_simulate_user_rollout(): Click1..4 + Skip with Pareto-filtered policy",
                },
                {
                    "id": "step3_ot_alignment_reward",
                    "implemented": True,
                    "evidence": (
                        "_train_ugw_init_regressor + _build_ugw_init_cost + "
                        "build_agent_structure_distance + build_expert_structure_distance + "
                        "ugw_transport(initial_plan=...)"
                        if self.alignment_mode == "softlabel_ugw"
                        else "build_cost_matrix + sinkhorn_transport + dense_rewards_from_transport"
                    ),
                },
                {
                    "id": "step4_offline_policy_update",
                    "implemented": True,
                    "evidence": (
                        "_fit(): OT soft-label cross entropy + imitation + entropy regularization"
                        if self.alignment_mode == "softlabel_ugw"
                        else "_fit(): policy gradient + imitation + entropy regularization"
                    ),
                },
                {
                    "id": "step4_online_fast_inference",
                    "implemented": True,
                    "evidence": "export fast_inference_policy.pt for web_sim matrixized online path",
                },
            ],
            "psy_engine_modeling_pdf": [
                {
                    "id": "state_vector_PCES",
                    "implemented": True,
                    "evidence": "PsyBerEngine export_state()/consume_item() with P,C,E,S",
                },
                {
                    "id": "input_features_v_n_c_pop",
                    "implemented": True,
                    "evidence": "ItemFeature uses item_v/item_n/item_c/item_pop from embeddings",
                },
                {
                    "id": "pareto_multi_objective_decision",
                    "implemented": True,
                    "evidence": "pareto_front_indices + softmax action selection",
                },
            ],
        }

    def run(self) -> None:
        print("[Load] Psy-engine")
        self.psy_engine_cls = self._load_psy_engine_class()

        print(f"[Load] item features from {self.embeddings_file}")
        self._load_item_features()

        print(f"[Load] user trajectories from {self.dataset_dir / 'ratings.dat'}")
        self._load_user_sequences()
        print(f"[Load] kept users={len(self.user_sequences)}")

        print(f"[Load] profiles from {self.profiles_file}")
        self._load_profiles()

        print("[Build] expert psychological trajectories")
        self._build_expert_intent_states()
        self._write_traj_preview()

        self.icvf_training_stats = {
            "enabled": False,
            "reason": "icvf_pretraining_removed",
        }

        if self.alignment_mode == "softlabel_ugw":
            print("[Train] UGW regression initializer")
            self._train_ugw_init_regressor()
        else:
            self.ugw_init_regression_stats = {
                "enabled": False,
                "reason": "alignment_mode_not_softlabel_ugw",
            }

        print(f"[Train] fast policy (mode={self.alignment_mode})")
        model, history = self._fit()

        ot_representation_dim = 4
        if self.expert_intents:
            first_intent = next(iter(self.expert_intents.values()))
            if isinstance(first_intent, np.ndarray) and first_intent.ndim == 2 and first_intent.shape[1] > 0:
                ot_representation_dim = int(first_intent.shape[1])

        metadata = {
            "trained_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_dir": str(self.dataset_dir),
            "embeddings_file": str(self.embeddings_file),
            "users": len(self.user_sequences),
            "max_steps": self.max_steps,
            "min_seq_len": self.min_seq_len,
            "slate_size": self.slate_size,
            "epochs": self.epochs,
            "alignment_mode": self.alignment_mode,
            "feature_dim": FEATURE_DIM,
            "hidden_dim": self.hidden_dim,
            "mid_dim": self.hidden_dim,
            "stochastic": bool(self.stochastic),
            "lookahead_lambda": self.lookahead_lambda,
            "sinkhorn_eps": self.sinkhorn_eps,
            "sinkhorn_iters": self.sinkhorn_iters,
            "ugw_eps": self.ugw_eps,
            "ugw_mass_reg": self.ugw_mass_reg,
            "ugw_outer_iters": self.ugw_outer_iters,
            "ugw_inner_iters": self.ugw_inner_iters,
            "ugw_tol": self.ugw_tol,
            "expert_time_weight": self.expert_time_weight,
            "agent_delta_weight": self.agent_delta_weight,
            "agent_time_weight": self.agent_time_weight,
            "ugw_init_temperature": self.ugw_init_temperature,
            "ugw_init_delta_weight": self.ugw_init_delta_weight,
            "ugw_init_time_weight": self.ugw_init_time_weight,
            "ugw_init_sinkhorn_iters": self.ugw_init_sinkhorn_iters,
            "ugw_init_reg_epochs": self.ugw_init_reg_epochs,
            "ugw_init_reg_batch_size": self.ugw_init_reg_batch_size,
            "ugw_init_reg_lr": self.ugw_init_reg_lr,
            "ugw_init_hidden_dim": self.ugw_init_hidden_dim,
            "ugw_init_user_dim": self.ugw_init_user_dim,
            "skip_recovery_boost": self.skip_recovery_boost,
            "icvf_enabled": bool(self.icvf_enabled),
            "icvf_intent_dim": int(self.icvf_intent_dim),
            "ot_representation_dim": int(ot_representation_dim),
            "intent_encoder_out": str(self.intent_encoder_out),
            "intent_stats_out": str(self.intent_stats_out),
            "agent_step_logs_out": str(self.agent_step_logs_out) if self.agent_step_logs_out else "",
            "action_space": ["click_1", "click_2", "click_3", "click_4", "skip"],
            "ablation_wo_template_random_selection": bool(
                self.ablation_wo_template_random_selection
            ),
            "template_selection_strategy": (
                "deterministic_hash"
                if self.ablation_wo_template_random_selection
                else "random"
            ),
            "history_tail": history[-3:],
        }

        save_fast_policy_bundle(self.policy_out, model, metadata)
        # Verify saved bundle can be loaded.
        _ = load_fast_policy_bundle(self.policy_out, device=self.device)

        report = {
            "config": {
                "dataset_dir": str(self.dataset_dir),
                "embeddings_file": str(self.embeddings_file),
                "profiles_file": str(self.profiles_file),
                "policy_out": str(self.policy_out),
                "report_out": str(self.report_out),
                "traj_preview_out": str(self.traj_preview_out),
                "agent_step_logs_out": (
                    str(self.agent_step_logs_out) if self.agent_step_logs_out else ""
                ),
                "max_users": self.max_users,
                "max_steps": self.max_steps,
                "min_seq_len": self.min_seq_len,
                "slate_size": self.slate_size,
                "epochs": self.epochs,
                "alignment_mode": self.alignment_mode,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "gamma": self.gamma,
                "entropy_coef": self.entropy_coef,
                "imitation_coef": self.imitation_coef,
                "imitation_decay": self.imitation_decay,
                "sinkhorn_eps": self.sinkhorn_eps,
                "sinkhorn_iters": self.sinkhorn_iters,
                "ugw_eps": self.ugw_eps,
                "ugw_mass_reg": self.ugw_mass_reg,
                "ugw_outer_iters": self.ugw_outer_iters,
                "ugw_inner_iters": self.ugw_inner_iters,
                "ugw_tol": self.ugw_tol,
                "expert_time_weight": self.expert_time_weight,
                "agent_delta_weight": self.agent_delta_weight,
                "agent_time_weight": self.agent_time_weight,
                "ugw_init_temperature": self.ugw_init_temperature,
                "ugw_init_delta_weight": self.ugw_init_delta_weight,
                "ugw_init_time_weight": self.ugw_init_time_weight,
                "ugw_init_sinkhorn_iters": self.ugw_init_sinkhorn_iters,
                "ugw_init_reg_epochs": self.ugw_init_reg_epochs,
                "ugw_init_reg_batch_size": self.ugw_init_reg_batch_size,
                "ugw_init_reg_lr": self.ugw_init_reg_lr,
                "ugw_init_hidden_dim": self.ugw_init_hidden_dim,
                "ugw_init_user_dim": self.ugw_init_user_dim,
                "lookahead_lambda": self.lookahead_lambda,
                "skip_recovery_boost": self.skip_recovery_boost,
                "hidden_dim": self.hidden_dim,
                "stochastic": self.stochastic,
                "ablation_wo_template_random_selection": self.ablation_wo_template_random_selection,
                "icvf_enabled": self.icvf_enabled,
                "icvf_intent_dim": self.icvf_intent_dim,
                "icvf_hidden_dim": self.icvf_hidden_dim,
                "icvf_epochs": self.icvf_epochs,
                "icvf_steps_per_epoch": self.icvf_steps_per_epoch,
                "icvf_batch_size": self.icvf_batch_size,
                "icvf_lr": self.icvf_lr,
                "icvf_weight_decay": self.icvf_weight_decay,
                "icvf_expectile": self.icvf_expectile,
                "icvf_future_window": self.icvf_future_window,
                "icvf_lin_alpha": self.icvf_lin_alpha,
                "icvf_lin_coef": self.icvf_lin_coef,
                "icvf_norm_coef": self.icvf_norm_coef,
                "icvf_target_tau": self.icvf_target_tau,
                "intent_encoder_out": str(self.intent_encoder_out),
                "intent_stats_out": str(self.intent_stats_out),
                "seed": self.seed,
                "device": self.device,
            },
            "stats": {
                "item_count": len(self.item_pool),
                "embedding_dim": self.emb_dim,
                "user_count": len(self.user_sequences),
                "expert_state_dim": 4,
                "ot_representation_dim": int(ot_representation_dim),
            },
            "history": history,
            "icvf_training": self.icvf_training_stats,
            "ugw_init_regression": self.ugw_init_regression_stats,
            "policy_metadata": metadata,
            "doc_alignment_checklist": self._build_doc_alignment_checklist(),
        }

        self.report_out.parent.mkdir(parents=True, exist_ok=True)
        self.report_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if self.agent_step_logs_out is not None:
            self.agent_step_logs_out.parent.mkdir(parents=True, exist_ok=True)
            self.agent_step_logs_out.write_text(
                json.dumps(self.agent_step_logs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(f"[Save] policy={self.policy_out}")
        print(f"[Save] report={self.report_out}")
        print(f"[Save] trajectory_preview={self.traj_preview_out}")
        if self.agent_step_logs_out is not None:
            print(f"[Save] agent_step_logs={self.agent_step_logs_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alignment pretraining agent")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--embeddings-file", type=Path, default=DEFAULT_EMBEDDINGS_FILE)
    parser.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES_FILE)
    parser.add_argument("--policy-out", type=Path, default=DEFAULT_POLICY_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--traj-preview-out", type=Path, default=DEFAULT_TRAJ_PREVIEW_OUT)
    parser.add_argument(
        "--agent-step-logs-out",
        type=Path,
        default=None,
        help="Optional JSON file to persist per-step training rollout logs.",
    )
    parser.add_argument("--intent-encoder-out", type=Path, default=DEFAULT_INTENT_ENCODER_OUT)
    parser.add_argument("--intent-stats-out", type=Path, default=DEFAULT_INTENT_STATS_OUT)

    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--min-seq-len", type=int, default=20)
    parser.add_argument("--slate-size", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--imitation-coef", type=float, default=0.05)
    parser.add_argument(
        "--imitation-decay",
        type=float,
        default=1.0,
        help="Per-epoch multiplicative decay for imitation coefficient (1.0 = no decay).",
    )
    parser.add_argument(
        "--alignment-mode",
        type=str,
        default="softlabel_ugw",
        choices=["softlabel_ugw", "legacy_rl"],
        help=(
            "softlabel_ugw: use regression-initialized UGW structural alignment + soft-label supervision; "
            "legacy_rl: keep historical RL-reward objective."
        ),
    )

    parser.add_argument("--sinkhorn-eps", type=float, default=0.10)
    parser.add_argument("--sinkhorn-iters", type=int, default=80)
    parser.add_argument("--ugw-eps", type=float, default=0.08)
    parser.add_argument("--ugw-mass-reg", type=float, default=0.30)
    parser.add_argument("--ugw-outer-iters", type=int, default=8)
    parser.add_argument("--ugw-inner-iters", type=int, default=40)
    parser.add_argument("--ugw-tol", type=float, default=1e-4)
    parser.add_argument("--expert-time-weight", type=float, default=0.10)
    parser.add_argument("--agent-delta-weight", type=float, default=0.50)
    parser.add_argument("--agent-time-weight", type=float, default=0.10)
    parser.add_argument(
        "--ugw-init-temperature",
        type=float,
        default=0.10,
        help="Temperature for converting regression init cost into initial transport plan.",
    )
    parser.add_argument(
        "--ugw-init-delta-weight",
        type=float,
        default=0.30,
        help="Weight for pseudo-state delta consistency in UGW initialization cost.",
    )
    parser.add_argument(
        "--ugw-init-time-weight",
        type=float,
        default=0.20,
        help="Weight for relative-time prior in UGW initialization cost.",
    )
    parser.add_argument(
        "--ugw-init-sinkhorn-iters",
        type=int,
        default=50,
        help="Sinkhorn iterations for normalizing UGW initialization plan.",
    )
    parser.add_argument(
        "--ugw-init-reg-epochs",
        type=int,
        default=6,
        help="Epochs for training regression-based UGW initializer.",
    )
    parser.add_argument(
        "--ugw-init-reg-batch-size",
        type=int,
        default=512,
        help="Batch size for regression-based UGW initializer.",
    )
    parser.add_argument(
        "--ugw-init-reg-lr",
        type=float,
        default=1e-3,
        help="Learning rate for regression-based UGW initializer.",
    )
    parser.add_argument(
        "--ugw-init-hidden-dim",
        type=int,
        default=128,
        help="Hidden dim of regression model used for UGW initialization.",
    )
    parser.add_argument(
        "--ugw-init-user-dim",
        type=int,
        default=16,
        help="User id embedding dim for regression-based UGW initializer.",
    )
    parser.add_argument("--lookahead-lambda", type=float, default=0.10)
    parser.add_argument("--skip-recovery-boost", type=float, default=0.3)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--disable-icvf",
        action="store_true",
        help="Deprecated no-op; ICVF pretraining has been removed from stage-02 runtime pipeline.",
    )
    parser.add_argument("--icvf-intent-dim", type=int, default=32)
    parser.add_argument("--icvf-hidden-dim", type=int, default=128)
    parser.add_argument("--icvf-epochs", type=int, default=20)
    parser.add_argument("--icvf-steps-per-epoch", type=int, default=100)
    parser.add_argument("--icvf-batch-size", type=int, default=256)
    parser.add_argument("--icvf-lr", type=float, default=1e-3)
    parser.add_argument("--icvf-weight-decay", type=float, default=1e-5)
    parser.add_argument("--icvf-expectile", type=float, default=0.8)
    parser.add_argument("--icvf-future-window", type=int, default=20)
    parser.add_argument("--icvf-lin-alpha", type=float, default=0.1)
    parser.add_argument("--icvf-lin-coef", type=float, default=0.1)
    parser.add_argument("--icvf-norm-coef", type=float, default=0.01)
    parser.add_argument("--icvf-target-tau", type=float, default=0.01)
    parser.add_argument(
        "--deterministic-action",
        action="store_true",
        help="Use greedy action selection instead of sampling during training rollouts.",
    )
    parser.add_argument(
        "--ablation-wo-template-random-selection",
        action="store_true",
        help=(
            "Table-6 style ablation: disable random slate/template selection and "
            "use deterministic hash-based selection."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto/cpu/cuda",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = AlignmentPretrainAgent(
        dataset_dir=args.dataset_dir,
        embeddings_file=args.embeddings_file,
        profiles_file=args.profiles_file,
        policy_out=args.policy_out,
        report_out=args.report_out,
        traj_preview_out=args.traj_preview_out,
        agent_step_logs_out=args.agent_step_logs_out,
        intent_encoder_out=args.intent_encoder_out,
        intent_stats_out=args.intent_stats_out,
        max_users=args.max_users,
        max_steps=args.max_steps,
        min_seq_len=args.min_seq_len,
        slate_size=args.slate_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        imitation_coef=args.imitation_coef,
        imitation_decay=args.imitation_decay,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        lookahead_lambda=args.lookahead_lambda,
        skip_recovery_boost=args.skip_recovery_boost,
        alignment_mode=args.alignment_mode,
        ugw_eps=args.ugw_eps,
        ugw_mass_reg=args.ugw_mass_reg,
        ugw_outer_iters=args.ugw_outer_iters,
        ugw_inner_iters=args.ugw_inner_iters,
        ugw_tol=args.ugw_tol,
        expert_time_weight=args.expert_time_weight,
        agent_delta_weight=args.agent_delta_weight,
        agent_time_weight=args.agent_time_weight,
        ugw_init_temperature=args.ugw_init_temperature,
        ugw_init_delta_weight=args.ugw_init_delta_weight,
        ugw_init_time_weight=args.ugw_init_time_weight,
        ugw_init_sinkhorn_iters=args.ugw_init_sinkhorn_iters,
        ugw_init_reg_epochs=args.ugw_init_reg_epochs,
        ugw_init_reg_batch_size=args.ugw_init_reg_batch_size,
        ugw_init_reg_lr=args.ugw_init_reg_lr,
        ugw_init_hidden_dim=args.ugw_init_hidden_dim,
        ugw_init_user_dim=args.ugw_init_user_dim,
        hidden_dim=args.hidden_dim,
        stochastic=not args.deterministic_action,
        ablation_wo_template_random_selection=args.ablation_wo_template_random_selection,
        seed=args.seed,
        device=args.device,
        disable_icvf=args.disable_icvf,
        icvf_intent_dim=args.icvf_intent_dim,
        icvf_hidden_dim=args.icvf_hidden_dim,
        icvf_epochs=args.icvf_epochs,
        icvf_steps_per_epoch=args.icvf_steps_per_epoch,
        icvf_batch_size=args.icvf_batch_size,
        icvf_lr=args.icvf_lr,
        icvf_weight_decay=args.icvf_weight_decay,
        icvf_expectile=args.icvf_expectile,
        icvf_future_window=args.icvf_future_window,
        icvf_lin_alpha=args.icvf_lin_alpha,
        icvf_lin_coef=args.icvf_lin_coef,
        icvf_norm_coef=args.icvf_norm_coef,
        icvf_target_tau=args.icvf_target_tau,
    )
    agent.run()


if __name__ == "__main__":
    main()
