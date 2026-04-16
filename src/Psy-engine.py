import json
import os
import time
from pathlib import Path

import numpy as np


def cosine_sim(v1, v2):
    """计算余弦相似度，范围映射到 [-1, 1]"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class PsyBerEngine:
    """Psy engine v3: 3D core state [C, E, S] with a 4D compatibility shell.

    Notes:
    - The new design only optimizes Cognitive reserve (C), Emotional satisfaction (E),
      and Self-congruence (S).
    - `P` is retained as a neutral compatibility slot so the existing training/inference
      stack can still consume `export_state()` without immediate refactors.
    """

    COMPAT_P_DEFAULT = 0.5
    OBJECTIVE_KEYS = ("C", "E", "S")
    COMPAT_STATE_KEYS = ("P", "C", "E", "S")

    def __init__(
        self,
        emb_dim=16,
        persona: dict | None = None,
        initial_memory: list[list[float]] | None = None,
        decision_weights: list[float] | None = None,
        stochastic: bool = False,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.persona = persona or {}
        self.emb_dim = int(emb_dim)
        self.stochastic = bool(stochastic)

        def _safe_norm(vec):
            arr = np.array(vec, dtype=np.float32)
            n = np.linalg.norm(arr)
            return arr if n <= 1e-12 else arr / n

        # Compatibility slot P is kept positive but is no longer part of the core state.
        init_state = self.persona.get("initial_state", {})
        self.P = float(np.clip(init_state.get("P", self.COMPAT_P_DEFAULT), 0.0, 1.0))
        self.C = float(np.clip(init_state.get("C", 0.55), 0.0, 1.0))
        self.E = float(np.clip(init_state.get("E", 0.55), 0.0, 1.0))
        self.S = float(np.clip(init_state.get("S", 0.55), 0.0, 1.0))
        self.Surf = 1

        self.u_base = _safe_norm(self.persona.get("u_base", self.rng.normal(size=self.emb_dim)))
        self.u_ideal = _safe_norm(self.persona.get("u_ideal", self.rng.normal(size=self.emb_dim)))
        self.u_real = _safe_norm(self.persona.get("u_real", self.rng.normal(size=self.emb_dim)))

        self.memory = []
        self.max_memory_size = int(self.persona.get("max_memory_size", 50))
        self.short_term_window = int(self.persona.get("short_term_window", 5))
        if initial_memory:
            for v in initial_memory[-self.max_memory_size :]:
                self.memory.append(_safe_norm(v))

        # Parameter override priority:
        # 1) persona["psy_params"]
        # 2) env PSY_PARAMS_JSON
        psy_params = {}
        persona_psy = self.persona.get("psy_params", {})
        if isinstance(persona_psy, dict):
            psy_params.update(persona_psy)

        env_json = os.getenv("PSY_PARAMS_JSON", "").strip()
        if env_json:
            try:
                parsed = json.loads(env_json)
                if isinstance(parsed, dict):
                    psy_params.update(parsed)
            except Exception:
                pass

        self.tau = float(np.clip(psy_params.get("tau", 0.35), 0.0, 1.0))
        self.rho = float(np.clip(psy_params.get("rho", psy_params.get("recovery_boost", 0.25)), 0.0, 1.0))
        self.recovery_boost = self.rho

        # Accept both 3D weights [C,E,S] and legacy 4D weights [P,C,E,S].
        if decision_weights is None:
            decision_weights = self.persona.get("decision_weights", [1.0, 1.0, 1.0])
        self.decision_weights = self._normalize_decision_weights(decision_weights)

    @staticmethod
    def _normalize_decision_weights(weights) -> np.ndarray:
        arr = np.array(weights, dtype=np.float32).reshape(-1)
        if arr.size >= 4:
            arr = arr[-3:]
        elif arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size), constant_values=1.0)
        if not np.any(np.isfinite(arr)):
            arr = np.ones((3,), dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, 1.0).astype(np.float32)
        return arr

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    def _normalize_item_vec(self, item_v) -> np.ndarray:
        arr = np.array(item_v, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(arr)
        if n <= 1e-12:
            return arr
        return arr / n

    def export_state(self):
        return {
            "P": float(self.P),
            "C": float(self.C),
            "E": float(self.E),
            "S": float(self.S),
            "Surf": int(self.Surf),
            "memory_size": int(len(self.memory)),
            "state_dim": 3,
        }

    def apply_recovery_if_needed(self):
        """If the previous step failed Surf check, recover depleted C/E/S by rho."""
        recovered = []
        if self.Surf != 0:
            return recovered

        for key in self.OBJECTIVE_KEYS:
            val = float(getattr(self, key))
            if val <= 0.0:
                setattr(self, key, self._clip01(val + self.rho))
                recovered.append(key)

        if min(self.C, self.E, self.S) > 0.0:
            self.Surf = 1
        return recovered

    def _compute_drive_metrics(self, item_v, item_n, item_c, item_pop):
        v = self._normalize_item_vec(item_v)
        item_n = self._clip01(item_n)
        item_c = self._clip01(item_c)
        item_pop = self._clip01(item_pop)

        if self.memory:
            sims = [cosine_sim(v, m) for m in self.memory]
            Nov_t = self._clip01(1.0 - max(sims))
            recent = sims[-self.short_term_window :]
            Sat_t = self._clip01(float(np.mean([max(0.0, s) for s in recent])))
        else:
            Nov_t = 1.0
            Sat_t = 0.0

        Bel_t = self._clip01(max(0.0, cosine_sim(v, self.u_base)))
        Ser_t = self._clip01(Nov_t * Bel_t)
        ET_t = self._clip01((Bel_t + Ser_t + item_pop) / 3.0)

        Guilt_t = self._clip01(
            max(0.0, cosine_sim(v, self.u_real) - cosine_sim(v, self.u_ideal))
        )
        SAS_t = self._clip01(Bel_t - Guilt_t)

        Stim_t = self._clip01((Nov_t + item_n) / 2.0)
        Drain_t = self._clip01((item_c + Sat_t) / 2.0)

        return {
            "item_v_norm": v,
            "item_n": item_n,
            "item_c": item_c,
            "item_pop": item_pop,
            "Nov_t": Nov_t,
            "Sat_t": Sat_t,
            "Stim_t": Stim_t,
            "Drain_t": Drain_t,
            "Bel_t": Bel_t,
            "Ser_t": Ser_t,
            "ET_t": ET_t,
            "Guilt_t": Guilt_t,
            "SAS_t": SAS_t,
        }

    def _calculate_transitions(self, item_v, item_n, item_c, item_pop):
        """Pure transition function for the new 3D state core.

        Returns a 4D compatibility tuple/delta: [P, C, E, S], where P is unchanged.
        """
        tmp_P = float(self.P)
        tmp_C = float(self.C)
        tmp_E = float(self.E)
        tmp_S = float(self.S)
        recovered = []

        if self.Surf == 0:
            for key in self.OBJECTIVE_KEYS:
                val = float(locals()[f"tmp_{key}"])
                if val <= 0.0:
                    val = self._clip01(val + self.rho)
                    locals()[f"tmp_{key}"] = val
                    recovered.append(key)
            # Rebind locals explicitly for clarity.
            if "C" in recovered:
                tmp_C = self._clip01(self.C + self.rho)
            if "E" in recovered:
                tmp_E = self._clip01(self.E + self.rho)
            if "S" in recovered:
                tmp_S = self._clip01(self.S + self.rho)

        m = self._compute_drive_metrics(item_v, item_n, item_c, item_pop)

        delta_C = float(m["Stim_t"] - m["Drain_t"])
        new_C = self._clip01(tmp_C + self.tau * delta_C)
        new_E = self._clip01((1.0 - self.tau) * tmp_E + self.tau * m["ET_t"])
        new_S = self._clip01((1.0 - self.tau) * tmp_S + self.tau * m["SAS_t"])

        new_P = tmp_P
        deltas = np.array(
            [0.0, new_C - tmp_C, new_E - tmp_E, new_S - tmp_S],
            dtype=np.float32,
        )

        metrics = {
            "Nov_t": float(m["Nov_t"]),
            "Sat_t": float(m["Sat_t"]),
            "Stim_t": float(m["Stim_t"]),
            "Drain_t": float(m["Drain_t"]),
            "Bel_t": float(m["Bel_t"]),
            "Ser_t": float(m["Ser_t"]),
            "ET_t": float(m["ET_t"]),
            "Guilt_t": float(m["Guilt_t"]),
            "SAS_t": float(m["SAS_t"]),
            "delta_C": float(delta_C),
        }
        return (new_P, new_C, new_E, new_S), deltas, recovered, metrics

    def select_best_item(self, candidates):
        """基于新 3D 目标 [C, E, S] 的多目标帕累托最优进行决策。"""
        n_cand = len(candidates)
        if n_cand <= 0:
            raise ValueError("candidates must not be empty")

        full_deltas = []
        objective_deltas = []
        for cand in candidates:
            _, deltas, _, _ = self._calculate_transitions(
                cand["v"], cand["n"], cand["c"], cand["pop"]
            )
            full_deltas.append(deltas)
            objective_deltas.append(deltas[1:4])

        full_deltas = np.array(full_deltas, dtype=np.float32)
        objective_deltas = np.array(objective_deltas, dtype=np.float32)
        is_dominated = np.zeros(n_cand, dtype=bool)

        for i in range(n_cand):
            for j in range(n_cand):
                if i == j:
                    continue
                if np.all(objective_deltas[j] >= objective_deltas[i]) and np.any(
                    objective_deltas[j] > objective_deltas[i]
                ):
                    is_dominated[i] = True
                    break

        pareto_front_indices = np.where(~is_dominated)[0]
        scores = np.array(
            [
                float(np.dot(self.decision_weights, objective_deltas[idx]))
                for idx in pareto_front_indices
            ],
            dtype=np.float32,
        )
        scores = scores - float(np.max(scores))
        exp_scores = np.exp(scores)
        z = float(np.sum(exp_scores))
        probs = exp_scores / z if z > 1e-12 else np.full_like(exp_scores, 1.0 / max(1, len(exp_scores)))

        if self.stochastic:
            chosen_pareto_idx = int(self.rng.choice(len(pareto_front_indices), p=probs))
            chosen_idx = int(pareto_front_indices[chosen_pareto_idx])
        else:
            chosen_idx = int(pareto_front_indices[int(np.argmax(scores))])

        return chosen_idx, pareto_front_indices.tolist()

    def consume_item(self, item_v, item_n, item_c, item_pop):
        """实际消费被选中的内容并固化状态。"""
        new_states, deltas, recovered, metrics = self._calculate_transitions(
            item_v, item_n, item_c, item_pop
        )

        self.P, self.C, self.E, self.S = [float(x) for x in new_states]

        norm_v = self._normalize_item_vec(item_v)
        self.memory.append(norm_v)
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

        self.Surf = 0 if min(self.C, self.E, self.S) <= 0.0 else 1
        return deltas, recovered, metrics


# ================= 运行模拟测试 =================
if __name__ == "__main__":
    np.random.seed(42)
    emb_dim = 16
    agent = PsyBerEngine(emb_dim=emb_dim)
    output_dir = Path(os.getenv("PSY_TEST_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "outputs" / "psy_engine"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "psy_engine_simulation.log"
    plot_path = output_dir / "psy_engine_state_trends.png"

    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    c_history = [float(agent.C)]
    e_history = [float(agent.E)]
    s_history = [float(agent.S)]

    log("=" * 85)
    log(f"【初始状态】 C(认知): {agent.C:.2f}, E(情感): {agent.E:.2f}, S(自我): {agent.S:.2f}")
    log("=" * 85)

    num_rounds = int(os.getenv("PSY_TEST_NUM_ROUNDS", "10000"))
    num_candidates = int(os.getenv("PSY_TEST_NUM_CANDIDATES", "8"))  # 每轮推荐系统下发的候选数

    for t in range(1, num_rounds + 1):
        start_time = time.perf_counter()
        recovered_pre = agent.apply_recovery_if_needed()

        candidates = []
        for _ in range(num_candidates):
            candidates.append(
                {
                    "v": np.random.randn(emb_dim),
                    "n": np.random.uniform(0, 1),
                    "c": np.random.uniform(0, 1) if t != 3 else 5.0,
                    "pop": np.random.uniform(0, 1),
                }
            )

        log(f"\n🔄 第 {t} 轮：接收候选集数量 = {num_candidates}")

        chosen_idx, pareto_front = agent.select_best_item(candidates)
        best_item = candidates[chosen_idx]
        deltas, recovered, m = agent.consume_item(
            best_item["v"], best_item["n"], best_item["c"], best_item["pop"]
        )
        c_history.append(float(agent.C))
        e_history.append(float(agent.E))
        s_history.append(float(agent.S))

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        if recovered_pre:
            log(
                f"   [👼 回血机制触发] 上回合跳过浏览。本回合基础值恢复: "
                f"{', '.join(recovered_pre)} (各 +{agent.recovery_boost})"
            )

        log("   [🧠 帕累托决策引擎]")
        log(f"      ├─ 筛选出帕累托最优解 (序号): {pareto_front}")
        log(
            f"      └─ Softmax 最终采纳项: 候选 {chosen_idx} "
            f"(争议度n={best_item['n']:.2f}, 负荷c={best_item['c']:.2f})"
        )

        log("   [内部心理变量推演 (被采纳项)]")
        log(
            f"      ├─ 认知通量 : 新奇感={m['Nov_t']:.3f} | 语义饱和={m['Sat_t']:.3f} "
            f"| 刺激项={m['Stim_t']:.3f} | 消耗项={m['Drain_t']:.3f}"
        )
        log(
            f"      ├─ 情感目标 : 归属感={m['Bel_t']:.3f} | 偶然惊喜={m['Ser_t']:.3f} "
            f"| ET={m['ET_t']:.3f}"
        )
        log(
            f"      └─ 自我协调 : 负罪感={m['Guilt_t']:.3f} | SAS={m['SAS_t']:.3f}"
        )

        surf_status_str = "浏览正常" if agent.Surf == 1 else "浏览跳过"

        log("   [宏观状态转移与决策]")
        log(f"      ► 认知(C) : {agent.C:.3f} (Δ {deltas[1]:+.3f})")
        log(f"      ► 情感(E) : {agent.E:.3f} (Δ {deltas[2]:+.3f})")
        log(f"      ► 自我(S) : {agent.S:.3f} (Δ {deltas[3]:+.3f})")
        log(f"      ⚙️ 本轮计算耗时 : {elapsed_ms:.4f} ms")
        log(f"      🚀 最终状态 : 【{surf_status_str}】 (Surf = {agent.Surf})")
        log("-" * 85)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = np.arange(len(c_history), dtype=np.int32)
        plt.figure(figsize=(11, 6))
        plt.plot(rounds, c_history, label="Cognitive Reserve (C)", linewidth=1.6)
        plt.plot(rounds, e_history, label="Emotional Satisfaction (E)", linewidth=1.6)
        plt.plot(rounds, s_history, label="Self-Congruence (S)", linewidth=1.6)
        plt.xlabel("Round")
        plt.ylabel("State Value")
        plt.title("Psy Engine Simulation Trends")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()
        log(f"📈 三个心理量变化趋势图已保存: {plot_path}")
    except Exception as err:
        log(f"⚠️ 趋势图生成失败: {err}")

    log(f"📝 模拟日志已保存: {log_path}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
