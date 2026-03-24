import numpy as np
import time

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
        # 1. 初始内部状态
        self.P, self.C, self.E, self.S = 0.5, 0.8, 0.5, 0.5
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
        self.alpha_1, self.beta_1, self.gamma_1, self.delta_1 = 0.5, 0.3, 0.2, 0.1
        self.w_load, self.w_sat, self.w_prim, self.w_nov = 0.3, 0.2, 0.1, 0.1
        self.alpha_3, self.lambda_1, self.lambda_2, self.lambda_3 = 0.6, 0.4, 0.3, 0.3
        self.mu_1, self.mu_2, self.mu_3, self.mu_4 = 0.2, 0.3, 0.4, 0.2
        self.theta = 0.2  
        # When Surf=0, next round auto-recovery for depleted dimensions.
        self.recovery_boost = 0.2
        
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
