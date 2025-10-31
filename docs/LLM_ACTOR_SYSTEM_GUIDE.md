# LLM-Actor 决策系统

## 概述

这是一个创新的多智能体决策架构，结合了**大语言模型（LLM）的语言理解能力**和**强化学习（PPO）的优化能力**。

### 核心思路

```
LLM生成候选 → Actor选择 → 执行 → 奖励反馈 → PPO训练Actor
```

### 关键优势

1. **Token效率**: 一次LLM调用生成N个候选，Actor选择无额外成本
2. **可训练**: Actor网络通过PPO学习何时选择哪个候选
3. **可解释**: 所有决策都是自然语言，完全透明
4. **鲁棒性**: 拒绝机制 + 多版本prompt + 默认策略兜底

---

## 架构组件

### 1. LLMCandidateGenerator
- **功能**: 调用LLM生成多个候选动作
- **输入**: 系统状态、历史、角色
- **输出**: N个自然语言候选 + Token消耗

### 2. SemanticEmbedder
- **功能**: 将自然语言转换为固定维度向量
- **方法**: sentence-transformers (可选) 或 mock嵌入
- **输出**: [N, 384] 的嵌入矩阵

### 3. CandidateSelector (Actor网络)
- **功能**: 从候选中选择最优动作，或拒绝所有
- **架构**: 状态编码器 + 候选编码器 + 注意力机制
- **输出**: 选择索引 [0, N] (N=拒绝)

### 4. NaturalLanguageActionParser
- **功能**: 将选择的文本解析为控制向量
- **方法**: 关键词匹配 + 强度提取
- **输出**: 17维控制向量

### 5. LLMActorDecisionSystem
- **功能**: 集成以上组件的完整决策系统
- **流程**: 生成→嵌入→选择→解析→执行

---

## 使用方法

### 基础使用

```python
from src.hospital_governance.agents.llm_actor_system import LLMActorDecisionSystem

# 初始化系统
system = LLMActorDecisionSystem(
    llm_provider="mock",  # 或 "openai"
    n_candidates=5,
    state_dim=16
)

# 获取决策
state = np.random.uniform(0.3, 0.9, 16)
result = system.get_action(
    role='doctors',
    state=state,
    deterministic=False
)

print(f"选择的动作: {result.selected_action}")
print(f"Token消耗: {result.tokens_used}")
print(f"控制向量: {result.action_vector}")
```

### 多智能体协同

```python
roles = ['doctors', 'interns', 'patients', 'accountants', 'government']

for role in roles:
    result = system.get_action(role=role, state=state)
    print(f"{role}: {result.selected_action}")
```

### PPO训练（即将实现）

```python
# 收集经验
for episode in range(n_episodes):
    for step in range(max_steps):
        result = system.get_action(role, state)
        next_state, reward = env.step(result.action_vector)
        
        # 奖励 = 改善 - Token成本 - 拒绝惩罚
        reward = improvement - result.tokens_used * 0.001
        
        buffer.add(state, result.log_prob, reward, ...)

# PPO更新
system.selector.train(buffer)
```

---

## 奖励机制

```python
reward = (
    system_improvement * 1.0 +      # 系统状态改善
    action_quality * 0.3 -          # 动作语义质量
    tokens_used * 0.001 -           # Token成本惩罚
    rejection_penalty * 0.1         # 拒绝惩罚
)
```

### 权衡

- **Token成本因子** 太高 → 永不调用LLM
- **Token成本因子** 太低 → 每步都调用LLM
- **最优策略** → 只在关键时刻调用

---

## 测试

```bash
# 运行测试套件
python test/test_llm_actor_system.py

# 运行交互式演示
python demo_llm_actor.py
```

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm_provider` | "mock" | LLM提供商 (mock/openai) |
| `n_candidates` | 5 | 生成候选数量 |
| `state_dim` | 16 | 系统状态维度 |
| `embedding_dim` | 384 | 语义嵌入维度 |
| `max_retries` | 3 | 拒绝后最大重试次数 |
| `temperature` | 0.8 | LLM生成温度（多样性）|

---

## 扩展方向

### 1. 真实LLM集成
```python
# 替换mock为真实API
generator = LLMCandidateGenerator(llm_provider="openai")
```

### 2. 更精确的解析
```python
# 用小型LLM提取参数
def parse_with_llm(nl_action):
    prompt = f"提取以下决策的参数（强度、优先级）：{nl_action}"
    params = small_llm(prompt)
    return params_to_vector(params)
```

### 3. 多模态输入
```python
# 除了文本，还可以输入图表、数据可视化
result = system.get_action(
    role='doctors',
    state=state,
    visual_context=state_chart  # 医院运营图表
)
```

### 4. 层次化决策
```python
# 战略层（LLM）+ 战术层（Actor）
strategic_plan = llm.generate_plan(state)  # 长期规划
tactical_action = actor.select(candidates, strategic_plan)  # 短期行动
```

---

## 性能指标

### Token效率对比

| 方法 | 每步Token | 说明 |
|------|----------|------|
| 独立调用5次LLM | ~1250 | 5个智能体各调用1次 |
| 候选生成+选择 | ~300 | 一次生成5个候选 |
| **节省比率** | **~76%** | 显著降低成本 |

### 预期学习曲线

```
Episode 0-100:   随机探索，高拒绝率
Episode 100-500: 学习何时调用LLM
Episode 500+:    稳定策略，只在关键时刻调用
```

---

## 相关文件

- `src/hospital_governance/agents/llm_actor_system.py` - 核心实现
- `test/test_llm_actor_system.py` - 测试套件
- `demo_llm_actor.py` - 交互式演示
- `docs/MADDPG_LLM_COLLABORATION_ARCHITECTURE.md` - 架构文档

---

## 常见问题

### Q: 为什么不直接用LLM做所有决策？
A: Token成本高昂。通过Actor学习"何时需要LLM"，可以节省~76%成本。

### Q: 如果所有候选都被拒绝怎么办？
A: 系统会：
1. 修改prompt版本（更具体/更激进）
2. 重新生成候选
3. 最多重试3次
4. 最终使用默认策略兜底

### Q: 如何处理LLM的随机性？
A: 
1. 生成多个候选增加覆盖面
2. Actor学习稳定的选择策略
3. 可设置`deterministic=True`使用argmax

### Q: 能否用于其他领域？
A: 完全可以！只需：
1. 修改状态描述函数
2. 调整角色和动作映射
3. 重新训练Actor网络

---

## 下一步计划

- [x] 核心组件实现
- [x] 测试套件
- [ ] 集成到Simulator
- [ ] PPO训练循环
- [ ] 真实LLM接入（OpenAI API）
- [ ] 可视化仪表板
- [ ] 性能基准测试

---

## 引用

如果使用本系统，请引用：

```bibtex
@software{kallipolis_llm_actor,
  title = {LLM-Actor Decision System for Multi-Agent Medical Governance},
  author = {Kallipolis Medical Republic Team},
  year = {2025},
  note = {Combining LLM generation with RL optimization}
}
```
