import json
import os
import time

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cosine_sim(v1, v2):
    """计算余弦相似度，范围映射到 [-1, 1]"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

class PsyBerEngine:
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
        # 1. 初始内部状态：保持中性但避免 P/C 过高起步导致快速饱和
        self.P, self.C, self.E, self.S = 0.45, 0.45, 0.55, 0.55
        self.Surf = 1 
        
        # 2. 画像与记忆
        self.persona = persona or {}
        self.u_base = np.array(self.persona.get("u_base", self.rng.normal(size=emb_dim)), dtype=np.float32)
        self.u_ideal = np.array(self.persona.get("u_ideal", self.rng.normal(size=emb_dim)), dtype=np.float32)
        self.u_real = np.array(self.persona.get("u_real", self.rng.normal(size=emb_dim)), dtype=np.float32)

        def _safe_norm(vec):
            n = np.linalg.norm(vec)
            return vec if n <= 1e-12 else vec / n

        self.u_base = _safe_norm(self.u_base)
        self.u_ideal = _safe_norm(self.u_ideal)
        self.u_real = _safe_norm(self.u_real)
        self.memory = []
        self.max_memory_size = 50
        if initial_memory:
            for v in initial_memory[-self.max_memory_size :]:
                arr = np.array(v, dtype=np.float32)
                self.memory.append(_safe_norm(arr))
        
        # 3. 超参数设定
        # 基于当前批次统计的稳态校准：
        # - C 下降过快（长期贴近 0）-> 降低认知惩罚、提高认知恢复项；
        # - S 持续单调下滑 -> 降低惩罚并提高 Bel 正反馈；
        # - E 过快衰减 -> 提高社会反馈占比，减轻中后段坍塌。
        # 覆盖优先级：
        # 1) legacy 预设（仅当 PSY_PARAM_PRESET=legacy|old|v1 时启用）
        # 2) persona["psy_params"]
        # 3) 环境变量 PSY_PARAMS_JSON（JSON dict，最高优先级）
        legacy_params = {
            "alpha_1": 0.53,
            "beta_1": 0.24,
            "gamma_1": 0.17,
            "delta_1": 0.07,
            "w_load": 0.29,
            "w_sat": 0.22,
            "w_prim": 0.10,
            "w_nov": 0.06,
            "alpha_3": 0.58,
            "lambda_1": 0.42,
            "lambda_2": 0.26,
            "lambda_3": 0.32,
            "mu_1": 0.20,
            "mu_2": 0.18,
            "mu_3": 0.22,
            "mu_4": 0.28,
            "theta": 0.12,
        }

        preset = os.getenv("PSY_PARAM_PRESET", "").strip().lower()
        if preset in {"legacy", "old", "v1"}:
            psy_params = dict(legacy_params)
        else:
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
        self.alpha_1 = float(psy_params.get("alpha_1", 0.56))
        self.beta_1 = float(psy_params.get("beta_1", 0.22))
        self.gamma_1 = float(psy_params.get("gamma_1", 0.15))
        self.delta_1 = float(psy_params.get("delta_1", 0.07))

        self.w_load = float(psy_params.get("w_load", 0.21))
        self.w_sat = float(psy_params.get("w_sat", 0.17))
        self.w_prim = float(psy_params.get("w_prim", 0.15))
        self.w_nov = float(psy_params.get("w_nov", 0.09))

        self.alpha_3 = float(psy_params.get("alpha_3", 0.62))
        self.lambda_1 = float(psy_params.get("lambda_1", 0.34))
        self.lambda_2 = float(psy_params.get("lambda_2", 0.24))
        self.lambda_3 = float(psy_params.get("lambda_3", 0.42))

        self.mu_1 = float(psy_params.get("mu_1", 0.14))
        self.mu_2 = float(psy_params.get("mu_2", 0.12))
        self.mu_3 = float(psy_params.get("mu_3", 0.15))
        self.mu_4 = float(psy_params.get("mu_4", 0.34))
        self.theta = float(psy_params.get("theta", 0.08))
        # When Surf=0, next round auto-recovery for depleted dimensions.
        self.recovery_boost = 0.25
        
        # 帕累托决策权重 (P, C, E, S 的偏好权重，可调整)
        if decision_weights is None:
            decision_weights = self.persona.get("decision_weights", [1.0, 1.0, 1.0, 1.0])
        self.decision_weights = np.array(decision_weights, dtype=np.float32)
        self.stochastic = stochastic

    def export_state(self):
        return {
            "P": float(self.P),
            "C": float(self.C),
            "E": float(self.E),
            "S": float(self.S),
            "Surf": int(self.Surf),
            "memory_size": int(len(self.memory)),
        }

    def apply_recovery_if_needed(self):
        """Next-round recovery: if Surf=0, depleted dimensions recover by recovery_boost."""
        recovered = []
        if self.Surf != 0:
            return recovered
        for key in ("P", "C", "E", "S"):
            val = float(getattr(self, key))
            if val <= 0.0:
                setattr(self, key, float(np.clip(val + self.recovery_boost, 0, 1)))
                recovered.append(key)
        # If all dimensions recovered above 0, allow normal browsing again.
        if self.P > 0 and self.C > 0 and self.E > 0 and self.S > 0:
            self.Surf = 1
        return recovered

    def _calculate_transitions(self, item_v, item_n, item_c, item_pop):
        """纯净的状态推演函数：计算如果消费该内容，状态会如何变化（不直接改变属性）"""
        tmp_P, tmp_C, tmp_E, tmp_S = self.P, self.C, self.E, self.S
        recovered = []
        # 当上一轮已进入 Surf=0 时，先做一轮局部恢复，再进行本轮状态推演。
        # 这样 recovery_boost 在点击路径和 skip 路径都能一致生效。
        if self.Surf == 0:
            if tmp_P <= 0.0:
                tmp_P = float(np.clip(tmp_P + self.recovery_boost, 0, 1))
                recovered.append("P")
            if tmp_C <= 0.0:
                tmp_C = float(np.clip(tmp_C + self.recovery_boost, 0, 1))
                recovered.append("C")
            if tmp_E <= 0.0:
                tmp_E = float(np.clip(tmp_E + self.recovery_boost, 0, 1))
                recovered.append("E")
            if tmp_S <= 0.0:
                tmp_S = float(np.clip(tmp_S + self.recovery_boost, 0, 1))
                recovered.append("S")

        # 计算中间变量
        if len(self.memory) > 0:
            sims = [cosine_sim(item_v, m) for m in self.memory]
            Nov_t = 1.0 - max(sims)
            Sat_t = np.mean([max(0, s) for s in sims[-5:]]) 
            Prim_t = max(0, cosine_sim(item_v, self.memory[-1])) 
            mem_var = np.mean(np.var(np.array(self.memory), axis=0)) 
        else:
            Nov_t = 1.0; Sat_t = 0.0; Prim_t = 0.0; mem_var = 1.0
            
        R_t = sigmoid(2.0 * Nov_t) 
        Bel_t = max(0, cosine_sim(item_v, self.u_base)) 
        Ser_t = Nov_t * Bel_t 
        
        Echo_t = np.exp(-mem_var) * Bel_t 
        Offense_t = max(0, self.theta - cosine_sim(item_v, self.u_base)) 
        Guilt_t = max(0, cosine_sim(item_v, self.u_real) - cosine_sim(item_v, self.u_ideal)) 

        # 计算新状态与 Delta
        new_P = np.clip(self.alpha_1 * tmp_P + self.beta_1 * Nov_t + self.gamma_1 * item_n + self.delta_1 * R_t, 0, 1)
        new_C = np.clip(tmp_C - self.w_load * item_c - self.w_sat * Sat_t + self.w_prim * Prim_t + self.w_nov * new_P, 0, 1)
        new_E = np.clip(self.alpha_3 * tmp_E + (1 - self.alpha_3) * (self.lambda_1 * Bel_t + self.lambda_2 * Ser_t + self.lambda_3 * item_pop), 0, 1)
        new_S = np.clip(tmp_S - self.mu_1 * Echo_t - self.mu_2 * Offense_t - self.mu_3 * Guilt_t + self.mu_4 * Bel_t, 0, 1)

        deltas = np.array([new_P - tmp_P, new_C - tmp_C, new_E - tmp_E, new_S - tmp_S])
        
        metrics = {"Nov_t": Nov_t, "Sat_t": Sat_t, "Prim_t": Prim_t, "R_t": R_t,
                   "Bel_t": Bel_t, "Ser_t": Ser_t, "Echo_t": Echo_t, 
                   "Offense_t": Offense_t, "Guilt_t": Guilt_t}
                   
        return (new_P, new_C, new_E, new_S), deltas, recovered, metrics

    def select_best_item(self, candidates):
        """基于多目标帕累托最优进行决策"""
        n_cand = len(candidates)
        all_deltas = []
        
        # 1. 状态模拟: 获取所有候选动作的预期 Delta H
        for cand in candidates:
            _, deltas, _, _ = self._calculate_transitions(cand['v'], cand['n'], cand['c'], cand['pop'])
            all_deltas.append(deltas)
            
        all_deltas = np.array(all_deltas)
        is_dominated = np.zeros(n_cand, dtype=bool)
        
        # 2. 帕累托支配判定
        for i in range(n_cand):
            for j in range(n_cand):
                if i != j:
                    # 如果 j 在所有维度都大于等于 i，且至少在一个维度严格大于 i，则 j 支配 i
                    if np.all(all_deltas[j] >= all_deltas[i]) and np.any(all_deltas[j] > all_deltas[i]):
                        is_dominated[i] = True
                        break
                        
        # 3. 提取帕累托前沿
        pareto_front_indices = np.where(~is_dominated)[0]
        
        # 4. Softmax 概率采样输出
        scores = np.array([np.dot(self.decision_weights, all_deltas[idx]) for idx in pareto_front_indices])
        exp_scores = np.exp(scores - np.max(scores)) # 减去最大值防止指数爆炸
        probs = exp_scores / np.sum(exp_scores)
        
        if self.stochastic:
            chosen_pareto_idx = self.rng.choice(len(pareto_front_indices), p=probs)
            chosen_idx = pareto_front_indices[chosen_pareto_idx]
        else:
            best_local_idx = int(np.argmax(scores))
            chosen_idx = int(pareto_front_indices[best_local_idx])
        
        return chosen_idx, pareto_front_indices.tolist()

    def consume_item(self, item_v, item_n, item_c, item_pop):
        """实际消费被选中的内容并固化状态"""
        # 利用纯函数获取最终确定的推演结果
        new_states, deltas, recovered, metrics = self._calculate_transitions(item_v, item_n, item_c, item_pop)
        
        # 将新状态覆写进自身属性
        self.P, self.C, self.E, self.S = new_states
        
        # 存入记忆并重置 Surf 判定
        self.memory.append(item_v)
        if len(self.memory) > self.max_memory_size: self.memory.pop(0)
            
        if self.P <= 0 or self.C <= 0 or self.E <= 0 or self.S <= 0:
            self.Surf = 0
        else:
            self.Surf = 1
            
        return deltas, recovered, metrics

# ================= 运行模拟测试 =================
if __name__ == "__main__":
    np.random.seed(42) 
    emb_dim = 16
    agent = PsyBerEngine(emb_dim=emb_dim)
    
    print("=" * 85)
    print(f"【初始状态】 P(感知): {agent.P:.2f}, C(认知): {agent.C:.2f}, E(情感): {agent.E:.2f}, S(自我): {agent.S:.2f}")
    print("=" * 85)
    
    num_rounds = 10000
    num_candidates = 8 # 每轮推荐系统下发 8 个商品/内容供 Agent 选择
    
    for t in range(1, num_rounds + 1):
        start_time = time.perf_counter()
        recovered_pre = agent.apply_recovery_if_needed()
        
        # 1. 模拟推荐系统生成 8 个候选集
        candidates = []
        for _ in range(num_candidates):
            candidates.append({
                'v': np.random.randn(emb_dim),
                'n': np.random.uniform(0, 1),
                'c': np.random.uniform(0, 1) if t != 3 else 5.0, # 第3轮集体塞入高认知负荷，强制击穿
                'pop': np.random.uniform(0, 1)
            })
            
        print(f"\n🔄 第 {t} 轮：接收候选集数量 = {num_candidates}")
        
        # 2. Agent 内部帕累托决策
        chosen_idx, pareto_front = agent.select_best_item(candidates)
        best_item = candidates[chosen_idx]
        
        # 3. Agent 实际消费决策胜出的内容
        deltas, recovered, m = agent.consume_item(best_item['v'], best_item['n'], best_item['c'], best_item['pop'])
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000 
        
        # ---- 打印输出 ----
        if recovered_pre:
            print(f"   [👼 回血机制触发] 上回合跳过浏览。本回合基础值恢复: {', '.join(recovered_pre)} (各 +{agent.recovery_boost})")
            
        print(f"   [🧠 帕累托决策引擎]")
        print(f"      ├─ 筛选出帕累托最优解 (序号): {pareto_front}")
        print(f"      └─ Softmax 最终采纳项: 候选 {chosen_idx} (争议度n={best_item['n']:.2f}, 负荷c={best_item['c']:.2f})")

        print(f"   [内部心理变量推演 (被采纳项)]")
        print(f"      ├─ 感知激发 : 新奇感={m['Nov_t']:.3f} | 内部奖励={m['R_t']:.3f}")
        print(f"      ├─ 认知磨损 : 语义饱和={m['Sat_t']:.3f} | 启动效应={m['Prim_t']:.3f}")
        print(f"      ├─ 情感反馈 : 归属感={m['Bel_t']:.3f} | 偶然惊喜={m['Ser_t']:.3f}")
        print(f"      └─ 自我审查 : 茧房激化={m['Echo_t']:.3f} | 冒犯感={m['Offense_t']:.3f} | 负罪感={m['Guilt_t']:.3f}")
        
        surf_status_str = "浏览正常" if agent.Surf == 1 else "浏览跳过"
        
        print(f"   [宏观状态转移与决策]")
        print(f"      ► 感知(P) : {agent.P:.3f} (Δ {deltas[0]:+.3f})")
        print(f"      ► 认知(C) : {agent.C:.3f} (Δ {deltas[1]:+.3f})")
        print(f"      ► 情感(E) : {agent.E:.3f} (Δ {deltas[2]:+.3f})")
        print(f"      ► 自我(S) : {agent.S:.3f} (Δ {deltas[3]:+.3f})")
        print(f"      ⚙️ 本轮计算耗时 : {elapsed_ms:.4f} ms")
        print(f"      🚀 最终状态 : 【{surf_status_str}】 (Surf = {agent.Surf})")
        print("-" * 85)
