# Agents模组详细文档

## 概述

Agents模组是Kallipolis医疗共和国治理系统的核心智能体架构，实现了基于多智能体博弈论的医院治理决策系统。该模组集成了深度强化学习、大语言模型（LLM）和分布式控制理论，为医院的各个利益相关者提供智能化的决策支持。

## 系统架构

### 核心组件

```
src/hospital_governance/agents/
├── __init__.py                 # 模组统一导出接口
├── agent_registry.py           # 智能体注册中心（NEW）
├── role_agents.py              # 角色智能体核心实现
├── behavior_models.py          # 行为模型系统
├── learning_models.py          # 深度学习模型
├── llm_action_generator.py     # LLM动作生成器
├── llm_providers.py            # LLM服务提供者（NEW）
├── interaction_engine.py       # 交互引擎
├── multi_agent_coordinator.py  # 多智能体协调器
└── role_agents_old.py          # 遗留实现（兼容性）
```

### 架构设计原则

1. **模块化设计**：各组件职责分明，支持独立替换和升级
2. **可扩展性**：支持新角色类型和行为模型的动态添加
3. **LLM集成**：统一的LLM接口，支持多种提供者
4. **环境驱动配置**：通过环境变量管理API密钥和配置
5. **优雅降级**：API失败时自动回退到Mock模式

## 智能体角色系统

### 支持的角色类型

| 角色 | 类名 | 动作维度 | 关注重点 | 决策导向 |
|------|------|----------|----------|----------|
| 医生 | `DoctorAgent` | 4维 | 医疗质量、患者安全 | 专业医疗标准 |
| 实习医生 | `InternAgent` | 3维 | 教育培训、职业发展 | 学习成长 |
| 患者代表 | `PatientAgent` | 3维 | 患者权益、服务质量 | 患者体验 |
| 会计 | `AccountantAgent` | 3维 | 财务健康、成本控制 | 经济效益 |
| 政府监管 | `GovernmentAgent` | 3维 | 监管合规、公共利益 | 系统稳定 |

### 动作空间设计

#### 医生动作向量 (4维)
```python
[质量改进强度, 资源申请强度, 工作负荷调整, 安全措施强度]
```
- 数值范围：`[-1.0, 1.0]`
- `1.0`：最大正向行动（增加、提高、加强）
- `0.0`：中性行动（保持现状）
- `-1.0`：最大负向行动（减少、限制、谨慎）

#### 实习医生动作向量 (3维)
```python
[培训需求强度, 工作负荷调整, 发展规划强度]
```

#### 患者代表动作向量 (3维)
```python
[服务改善需求, 可及性优化, 安全关注强度]
```

#### 会计动作向量 (3维)
```python
[成本控制强度, 效率提升, 预算优化]
```

#### 政府监管动作向量 (3维)
```python
[监管措施强度, 政策调整, 协调行动]
```

## 智能体注册中心

### AgentRegistry设计

智能体注册中心是新引入的核心组件，负责统一管理智能体的创建、配置和LLM服务集成。

```python
from src.hospital_governance.agents import create_agent_registry

# 创建注册中心
registry = create_agent_registry(
    llm_provider="openai",      # 或 "anthropic", "local", "mock"
    enable_llm=True,
    fallback_to_mock=True
)

# 批量注册所有角色
agents = registry.register_all_agents()

# 获取特定角色智能体
doctor_agent = registry.get_agent('doctors')
```

### 环境变量配置

系统支持通过环境变量管理API密钥和配置：

```bash
# LLM API密钥
export OPENAI_API_KEY="sk-your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# 系统配置（可选）
export HOSPITAL_LLM_PROVIDER="openai"     # openai, anthropic, local, mock
export HOSPITAL_LLM_PRESET="openai_gpt4"  # 预设配置
export HOSPITAL_ENABLE_LLM="true"         # 是否启用LLM
export HOSPITAL_FALLBACK_MOCK="true"      # API失败时回退到mock
```

### LLM提供者支持

| 提供者 | 类名 | 环境变量 | 支持模型 |
|--------|------|----------|----------|
| OpenAI | `OpenAIProvider` | `OPENAI_API_KEY` | GPT-4, GPT-3.5-turbo |
| Anthropic | `AnthropicProvider` | `ANTHROPIC_API_KEY` | Claude-3-sonnet |
| 本地模型 | `LocalModelProvider` | 无 | Ollama, llama2:7b |
| Mock模拟 | `MockLLMProvider` | 无 | 测试和开发用 |

## 数学模型基础

### 智能体策略模型

基于参数化随机策略的数学框架：

```
π_i(a_i | o_i; θ_i) = exp(φ_i(o_i, a_i)^T θ_i) / Σ exp(φ_i(o_i, a_j)^T θ_i)
```

其中：
- `π_i`：智能体i的策略函数
- `o_i`：局部观测向量
- `a_i`：动作向量
- `θ_i`：策略参数
- `φ_i`：特征映射函数

### 收益函数设计

多目标收益函数：

```
R_i(x, a_i, a_{-i}) = α_i U(x) + β_i V_i(x, a_i) - γ_i D_i(x, x*)
```

其中：
- `α_i`：全局效用权重（默认0.3）
- `β_i`：局部价值权重（默认0.5）
- `γ_i`：理想状态偏差权重（默认0.2）
- `U(x)`：全局资源效用函数
- `V_i(x, a_i)`：角色特异性价值函数
- `D_i(x, x*)`：到理想状态的偏差

### 策略梯度更新

```
θ_i(t+1) = θ_i(t) + η ∇_{θ_i} J_i(θ)
```

其中：
- `η`：学习率（默认0.001）
- `J_i(θ)`：策略目标函数
- `∇_{θ_i} J_i(θ)`：策略梯度

## LLM集成架构

### LLMActionGenerator

LLM动作生成器是连接自然语言推理和数值决策的桥梁：

```python
# 异步生成
action = await generator.generate_action_async(
    role="doctors",
    observation=obs_vector,
    holy_code_state=ethics_rules,
    context={"context_type": "crisis"}
)

# 同步生成
action = generator.generate_action_sync(
    role="doctors", 
    observation=obs_vector,
    holy_code_state=ethics_rules,
    context={}
)
```

### 角色特定提示模板

每个角色都有定制化的提示模板：

```python
# 医生角色提示示例
"作为医生，考虑当前医疗质量指标{quality}，患者安全指标{safety}，
建议采取的医疗质量改进行动：
...
请返回4维向量：[质量改进, 资源申请, 工作负荷调整, 安全措施]"
```

### 响应解析机制

1. **向量格式解析**：优先提取 `[0.5, -0.2, 0.8, 0.1]` 格式
2. **关键词推断**：基于文本关键词推断动作强度
3. **默认回退**：解析失败时使用数学模型计算默认动作

## 行为模型系统

### 支持的行为类型

```python
from src.hospital_governance.agents.behavior_models import BehaviorType

# 可用行为类型
BehaviorType.RATIONAL          # 理性决策
BehaviorType.BOUNDED_RATIONAL  # 有界理性
BehaviorType.EMOTIONAL         # 情感驱动
BehaviorType.SOCIAL            # 社交影响
BehaviorType.ADAPTIVE          # 自适应学习
```

### 行为模型配置

```python
from src.hospital_governance.agents.behavior_models import BehaviorModelFactory

# 创建理性行为模型
rational_model = BehaviorModelFactory.create_model(
    BehaviorType.RATIONAL,
    {
        'risk_tolerance': 0.3,
        'optimization_method': 'gradient_descent',
        'convergence_threshold': 0.01
    }
)

# 创建情感行为模型
emotional_model = BehaviorModelFactory.create_model(
    BehaviorType.EMOTIONAL,
    {
        'stress_sensitivity': 0.7,
        'mood_influence': 0.5,
        'emotion_decay_rate': 0.1
    }
)
```

## 深度学习集成

### MADDPG支持

多智能体深度确定性策略梯度算法：

```python
from src.hospital_governance.agents.learning_models import MADDPGModel

# 创建MADDPG模型
maddpg = MADDPGModel(
    observation_dims={'doctors': 8, 'interns': 8, 'patients': 8},
    action_dims={'doctors': 4, 'interns': 3, 'patients': 3},
    hidden_dims=[128, 64],
    learning_rate=0.001
)

# 训练更新
maddpg.update(experiences)
```

### DQN支持

深度Q网络用于离散动作空间：

```python
from src.hospital_governance.agents.learning_models import DQNModel

dqn = DQNModel(
    state_dim=16,
    action_dim=5,
    hidden_dims=[256, 128],
    learning_rate=0.0001
)
```

## 使用示例

### 基础使用

```python
from src.hospital_governance.agents import create_agent_registry

# 1. 创建注册中心
registry = create_agent_registry(llm_provider="mock")

# 2. 注册所有智能体
agents = registry.register_all_agents()

# 3. 获取医生智能体
doctor = registry.get_agent('doctors')

# 4. 生成观测
import numpy as np
observation = np.random.uniform(0.3, 0.7, 8)

# 5. 智能体决策
action = doctor.select_action_with_llm(
    observation=observation,
    holy_code_guidance={'priority_boost': 1.2},
    use_llm=True
)

print(f"医生动作: {action}")  # 输出: [0.1, 0.3, -0.2, 0.8]
```

### 高级配置

```python
from src.hospital_governance.agents import (
    AgentRegistry, AgentRegistryConfig, LLMProviderType,
    AgentConfig
)

# 自定义配置
config = AgentRegistryConfig(
    llm_provider=LLMProviderType.OPENAI,
    llm_preset="openai_gpt4",
    enable_llm_generation=True,
    fallback_to_mock=True
)

registry = AgentRegistry(config)

# 自定义智能体配置
custom_configs = {
    'doctors': AgentConfig(
        role='doctors',
        action_dim=4,
        observation_dim=8,
        learning_rate=0.002,
        alpha=0.4,  # 更高的全局效用权重
        beta=0.4,
        gamma=0.2
    )
}

agents = registry.register_all_agents(custom_configs)
```

### 测试和调试

```python
# 检查注册中心状态
status = registry.get_registry_status()
print("注册中心状态:", status)

# 测试LLM生成功能
test_results = registry.test_llm_generation()
for role, result in test_results.items():
    print(f"{role}: {result}")

# 切换LLM提供者
registry.update_llm_config(LLMProviderType.ANTHROPIC, "anthropic_claude")

# 导出配置
registry.export_config("my_config.json")
```

## 性能指标

### 智能体性能监控

每个智能体提供详细的性能指标：

```python
metrics = agent.get_performance_metrics()
# {
#     'performance_score': 0.75,
#     'mean_reward': 0.23,
#     'std_reward': 0.15,
#     'cumulative_reward': 45.7,
#     'policy_norm': 2.34,
#     'baseline_value': 0.21,
#     'total_actions': 150
# }
```

### LLM生成统计

```python
llm_stats = generator.get_generation_stats()
# {
#     'total_generations': 100,
#     'success_rate': 0.95,
#     'error_rate': 0.05,
#     'average_response_time': 1.2
# }
```

## 配置文件

### 智能体配置 (agent_registry_config.json)

```json
{
  "llm_provider": "mock",
  "llm_preset": "mock",
  "enable_llm_generation": true,
  "api_key_env_vars": {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY"
  },
  "registry_status": {
    "total_agents": 5,
    "registered_roles": ["doctors", "interns", "patients", "accountants", "government"],
    "llm_provider": "mock",
    "api_status": {
      "openai": false,
      "anthropic": false
    }
  }
}
```

## 故障排除

### 常见问题

1. **API密钥未配置**
   ```
   ⚠️ openai API密钥未配置 (环境变量: OPENAI_API_KEY)
   ```
   解决：设置对应的环境变量

2. **LLM生成失败**
   ```
   ❌ doctors: failed - 错误: LLM generator not available
   ```
   解决：检查网络连接和API密钥有效性

3. **模块导入错误**
   ```
   ModuleNotFoundError: No module named 'httpx'
   ```
   解决：安装缺失依赖 `pip install httpx requests`

### 调试工具

```python
# 调试Mock LLM响应
from src.hospital_governance.agents.llm_action_generator import MockLLMProvider, LLMConfig

config = LLMConfig(model_name="mock")
provider = MockLLMProvider(config)
response = provider.generate_text_sync("测试提示", {'role': 'doctors'})
print(response)

# 测试智能体注册
python3 env_config_example.py test

# 交互式测试
python3 env_config_example.py interactive
```

## 扩展开发

### 添加新角色

1. 创建新的角色类：
```python
class NurseAgent(RoleAgent):
    def observe(self, environment):
        # 实现护士特定的观测逻辑
        pass
    
    def compute_local_value(self, system_state, action):
        # 实现护士的价值函数
        pass
```

2. 注册到工厂：
```python
registry.agent_classes['nurses'] = NurseAgent
```

### 添加新LLM提供者

1. 继承BaseLLMProvider：
```python
class CustomLLMProvider(BaseLLMProvider):
    async def generate_text(self, prompt, context=None):
        # 实现自定义LLM接口
        pass
```

2. 注册到工厂：
```python
providers['custom'] = CustomLLMProvider
```

## 许可证

本模组遵循MIT许可证，详见项目根目录的LICENSE文件。

## 贡献指南

欢迎提交问题报告和功能请求到项目的GitHub仓库。贡献代码请遵循项目的代码规范和测试要求。