# 行为模型组件文档

## 概述

`behavior_models.py` 是医院治理系统中的核心组件，实现了基于心理学、经济学和博弈论原理的智能体行为建模。该组件为不同角色的智能体提供了丰富的行为模式，使得系统能够更真实地模拟医院环境中的复杂人际交互和决策过程。

## 主要特性

### 1. 多种行为类型

- **理性行为模型 (RationalBehaviorModel)**：基于效用最大化的经典理性决策
- **有限理性行为模型 (BoundedRationalBehaviorModel)**：使用启发式规则的简化决策
- **情感行为模型 (EmotionalBehaviorModel)**：考虑情感状态的决策模型
- **社会性行为模型 (SocialBehaviorModel)**：强调合作和社会影响的行为
- **适应性行为模型 (AdaptiveBehaviorModel)**：基于学习和环境变化的动态适应

### 2. 核心数据结构

#### BehaviorParameters
配置行为模型的关键参数：
- `rationality_level`: 理性程度 [0,1]
- `emotional_weight`: 情感权重 [0,1]
- `social_influence`: 社会影响 [0,1]
- `risk_tolerance`: 风险容忍度 [0,1]
- `adaptation_rate`: 适应速率 [0,1]
- `cooperation_tendency`: 合作倾向 [0,1]
- `fairness_concern`: 公平关注 [0,1]

#### BehaviorState
跟踪智能体的动态行为状态：
- `current_mood`: 当前情绪状态
- `stress_level`: 压力水平
- `confidence`: 信心水平
- `trust_levels`: 对其他角色的信任度
- `reputation`: 声誉值
- `experience_count`: 经验计数

### 3. 行为模型详解

#### 理性行为模型
- 使用效用函数进行决策
- 考虑即时奖励、未来奖励和风险调整
- 支持社会效用计算
- 理性程度决定选择的确定性

#### 有限理性行为模型
- 实现四种启发式规则：
  - 满足化启发式：选择第一个满足条件的行动
  - 模仿启发式：模仿成功的其他智能体
  - 阈值启发式：基于简单阈值规则
  - 默认启发式：保守的平均行动
- 根据认知负荷选择决策策略

#### 情感行为模型
- 三维情感空间：效价、唤醒度、控制感
- 情感状态影响行动选择
- 动态情感更新机制
- 情感衰减和记忆效应

#### 社会性行为模型
- 社会规范符合度评估
- 声誉系统和信任机制
- 合作与公平性评估
- 群体压力和从众效应

#### 适应性行为模型
- 四种适应策略：探索、利用、模仿、梯度上升
- 策略权重动态更新
- 基于性能的策略选择
- 适应历史记录

### 4. 角色特定配置

系统为不同角色预设了合适的行为模型：

- **医生 (Doctors)**: 理性行为，高理性水平，风险厌恶
- **实习生 (Interns)**: 适应性行为，高学习率，易受社会影响
- **患者 (Patients)**: 情感行为，高情感权重，风险厌恶
- **会计 (Accountants)**: 理性行为，极高理性，中等社会影响
- **政府 (Government)**: 社会性行为，高公平关注，强调合作

## 使用方法

### 基本使用

```python
from behavior_models import BehaviorModelFactory, BehaviorParameters, BehaviorType

# 创建理性行为模型
params = BehaviorParameters(rationality_level=0.8, emotional_weight=0.2)
model = BehaviorModelFactory.create_behavior_model(BehaviorType.RATIONAL, params)

# 计算行动概率
observation = np.array([0.7, 0.5, 0.8, 0.6])
available_actions = np.array([[0.5, 0.3], [0.8, 0.1], [0.2, 0.9]])
context = {'reward_weights': np.array([1.0, 0.8])}

probabilities = model.compute_action_probabilities(observation, available_actions, context)
selected_action = available_actions[np.argmax(probabilities)]

# 更新行为状态
reward = 0.7
model.update_behavior_state(observation, selected_action, reward, context)
```

### 使用角色特定模型

```python
# 为医生角色创建专门的行为模型
doctor_model = BehaviorModelFactory.create_role_specific_model('doctors')

# 为所有角色创建行为模型管理器
manager = BehaviorModelManager()
manager.create_all_role_models()

# 批量更新所有模型
observations = {'doctors': obs1, 'patients': obs2, ...}
actions = {'doctors': act1, 'patients': act2, ...}
rewards = {'doctors': rew1, 'patients': rew2, ...}
manager.update_all_models(observations, actions, rewards, context)
```

### 行为分析

```python
# 获取单个模型的行为指标
metrics = model.get_behavior_metrics()
print(f"情绪: {metrics['mood']}, 压力: {metrics['stress']}")

# 获取集体行为指标
collective_metrics = manager.get_collective_behavior_metrics()
print(f"平均情绪: {collective_metrics['collective']['avg_mood']}")

# 分析行为模式
patterns = manager.analyze_behavioral_patterns()
for role, pattern in patterns.items():
    print(f"{role}: 奖励趋势 {pattern['avg_reward_trend']}")
```

## 高级特性

### 1. 情感动力学
- 基于奖励的效价更新
- 基于环境变化的唤醒度调节
- 基于成功率的控制感调整
- 自动情感衰减机制

### 2. 社会交互机制
- 信任水平动态更新
- 声誉系统
- 合作行为评估
- 公平性计算（基尼系数）

### 3. 学习与适应
- 多策略权重优化
- 经验记录和回放
- 梯度上升学习
- 策略性能跟踪

### 4. 启发式决策
- 认知负荷评估
- 满足化决策
- 社会学习
- 阈值触发机制

## 集成指南

该组件设计为与现有的智能体系统无缝集成：

1. **与role_agents.py集成**：为RoleAgent类提供行为决策支持
2. **与learning_models.py协作**：行为模型可以指导强化学习的探索策略
3. **与interaction_engine.py配合**：提供丰富的社会交互建模
4. **支持实时监控**：提供详细的行为指标用于系统监控

## 扩展性

- **新行为类型**：继承BaseBehaviorModel轻松添加新的行为模式
- **自定义参数**：灵活的参数配置系统
- **插件式设计**：模块化的组件架构便于扩展
- **多层次建模**：支持个体、群体和系统层面的行为建模

## 性能考虑

- 使用numpy进行高效数值计算
- 智能的状态更新机制避免不必要计算
- 适应历史的自动内存管理
- 可配置的精度和性能平衡

## 示例应用

运行 `examples/behavior_models_demo.py` 查看完整的使用示例，包括：
- 单个行为模型演示
- 角色特定模型配置
- 多轮交互模拟
- 集体行为分析
- 行为模式识别

该组件为医院治理系统提供了强大的行为建模能力，使得智能体能够展现出更加真实和复杂的行为模式，从而提高整个系统的仿真质量和决策效果。