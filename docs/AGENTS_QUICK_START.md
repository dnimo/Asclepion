# Agents模组快速入门

本指南帮助您快速上手Kallipolis医疗共和国治理系统的智能体模组。

## 🚀 5分钟快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install httpx requests

# 配置API密钥（可选，未配置时使用Mock模式）
export OPENAI_API_KEY="sk-your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 2. 基础使用

```python
from src.hospital_governance.agents import create_agent_registry
import numpy as np

# 创建智能体注册中心
registry = create_agent_registry(llm_provider="mock")  # 或 "openai", "anthropic"

# 注册所有智能体
agents = registry.register_all_agents()
print(f"✅ 已注册 {len(agents)} 个智能体: {list(agents.keys())}")

# 获取医生智能体
doctor = registry.get_agent('doctors')

# 生成观测并进行决策
observation = np.random.uniform(0.3, 0.7, 8)
action = doctor.select_action_with_llm(observation, use_llm=True)
print(f"医生动作: {action}")  # 输出: [0.1, 0.3, -0.2, 0.8]
```

### 3. 测试系统

```bash
# 测试智能体注册和LLM集成
python3 env_config_example.py test

# 交互式演示
python3 env_config_example.py interactive

# 详细动作测试
python3 detailed_action_test.py
```

## 🤖 智能体类型

| 角色 | 动作维度 | 主要职责 | 示例动作 |
|------|----------|----------|----------|
| 医生 | 4维 | 医疗质量、患者安全 | [质量改进, 资源申请, 负荷调整, 安全措施] |
| 实习生 | 3维 | 教育培训、职业发展 | [培训需求, 工作调整, 发展计划] |
| 患者 | 3维 | 患者权益、服务质量 | [服务改善, 可及性优化, 安全关注] |
| 会计 | 3维 | 财务健康、成本控制 | [成本控制, 效率提升, 预算优化] |
| 政府 | 3维 | 监管合规、公共利益 | [监管措施, 政策调整, 协调行动] |

## 🧠 LLM集成

### 支持的提供者

```python
# OpenAI GPT
registry = create_agent_registry(llm_provider="openai")

# Anthropic Claude  
registry = create_agent_registry(llm_provider="anthropic")

# 本地模型 (Ollama)
registry = create_agent_registry(llm_provider="local")

# Mock模式 (测试)
registry = create_agent_registry(llm_provider="mock")
```

### 动作生成示例

```python
# LLM生成动作
llm_action = doctor.generate_llm_action(
    observation=observation,
    holy_code_state={'active_rules': []},
    context={'context_type': 'crisis'}
)

# 策略生成动作
policy_action = doctor.sample_action(observation)

# 混合决策（LLM优先，失败时回退策略）
final_action = doctor.select_action_with_llm(
    observation=observation,
    use_llm=True  # 尝试LLM，失败时自动回退
)
```

## 📊 性能监控

```python
# 获取智能体性能指标
metrics = doctor.get_performance_metrics()
print(f"性能评分: {metrics['performance_score']:.2f}")
print(f"平均奖励: {metrics['mean_reward']:.2f}")
print(f"累积奖励: {metrics['cumulative_reward']:.2f}")

# 获取注册中心状态
status = registry.get_registry_status()
print(f"LLM提供者: {status['llm_provider']}")
print(f"API状态: {status['api_status']}")

# 测试LLM生成功能
test_results = registry.test_llm_generation()
for role, result in test_results.items():
    print(f"{role}: {result['status']} - 动作维度: {result.get('action_shape', 'N/A')}")
```

## 🔧 配置选项

### 环境变量

```bash
# LLM配置
export HOSPITAL_LLM_PROVIDER="openai"     # 提供者类型
export HOSPITAL_LLM_PRESET="openai_gpt4"  # 预设配置  
export HOSPITAL_ENABLE_LLM="true"         # 启用LLM
export HOSPITAL_FALLBACK_MOCK="true"      # 失败回退

# API密钥
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
```

### 代码配置

```python
from src.hospital_governance.agents import AgentRegistryConfig, LLMProviderType

# 自定义配置
config = AgentRegistryConfig(
    llm_provider=LLMProviderType.OPENAI,
    llm_preset="openai_gpt4",
    enable_llm_generation=True,
    fallback_to_mock=True
)

registry = AgentRegistry(config)
```

## 🛠️ 故障排除

### 常见问题

1. **API密钥未配置**
   ```
   ⚠️ openai API密钥未配置 (环境变量: OPENAI_API_KEY)
   ```
   **解决**: `export OPENAI_API_KEY="your-key"`

2. **模块导入错误**
   ```
   ModuleNotFoundError: No module named 'httpx'
   ```
   **解决**: `pip install httpx requests`

3. **LLM生成失败**
   ```
   ❌ doctors: failed - 错误: LLM generator not available
   ```
   **解决**: 检查网络连接和API密钥，或使用Mock模式

### 调试技巧

```python
# 检查API连接
registry.test_llm_generation('doctors')

# 切换提供者
registry.update_llm_config(LLMProviderType.MOCK)

# 导出配置诊断
registry.export_config("debug_config.json")

# 查看详细日志
import logging
logging.basicConfig(level=logging.INFO)
```

## 📈 高级用法

### 自定义智能体配置

```python
from src.hospital_governance.agents import AgentConfig

custom_configs = {
    'doctors': AgentConfig(
        role='doctors',
        action_dim=4,
        observation_dim=8,
        learning_rate=0.002,
        alpha=0.4,  # 全局效用权重
        beta=0.4,   # 局部价值权重  
        gamma=0.2   # 理想状态偏差权重
    )
}

agents = registry.register_all_agents(custom_configs)
```

### 批量操作

```python
# 批量决策
observations = {role: np.random.uniform(0.3, 0.7, 8) for role in agents.keys()}
actions = {}

for role, obs in observations.items():
    agent = registry.get_agent(role)
    actions[role] = agent.select_action_with_llm(obs, use_llm=True)

print("所有智能体行动:", actions)
```

### 性能分析

```python
# 生成统计报告
performance_summary = registry.get_performance_summary()

for role, metrics in performance_summary.items():
    print(f"\n{role} 智能体:")
    print(f"  性能评分: {metrics['performance_score']:.3f}")
    print(f"  策略参数范数: {metrics['policy_norm']:.3f}")
    print(f"  总行动次数: {metrics['total_actions']}")
```

## 🎯 下一步

- 📖 阅读 [完整的Agents模组指南](AGENTS_MODULE_GUIDE.md)
- 🧪 运行 `python3 env_config_example.py interactive` 进行交互式体验
- 🎮 启动完整的仿真系统: `python3 websocket_server.py`
- 📊 访问Web界面: http://localhost:8080/frontend/websocket_demo.html

---

**🏥 开始您的智能医疗治理之旅！**