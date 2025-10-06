"""
行为模型使用示例

演示如何使用behavior_models.py中的各种行为模型
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.hospital_governance.agents.behavior_models import (
    BehaviorType, BehaviorParameters, BehaviorModelFactory, 
    BehaviorModelManager, RationalBehaviorModel, EmotionalBehaviorModel
)

def demo_individual_behavior_models():
    """演示单个行为模型的使用"""
    print("=== 行为模型演示 ===\n")
    
    # 1. 理性行为模型演示
    print("1. 理性行为模型演示")
    rational_params = BehaviorParameters(
        rationality_level=0.9,
        emotional_weight=0.1,
        risk_tolerance=0.3
    )
    rational_model = RationalBehaviorModel(rational_params)
    
    # 模拟观测和可用行动
    observation = np.array([0.7, 0.5, 0.8, 0.6, 0.9, 0.4, 0.3, 0.8])
    available_actions = np.array([
        [0.5, 0.3, 0.7, 0.2],
        [0.8, 0.1, 0.4, 0.6],
        [0.2, 0.9, 0.3, 0.5]
    ])
    
    context = {
        'reward_weights': np.array([1.0, 0.8, 1.2, 0.6]),
        'other_actions': {
            'accountants': np.array([0.6, 0.4, 0.5, 0.3])
        }
    }
    
    action_probs = rational_model.compute_action_probabilities(
        observation, available_actions, context
    )
    print(f"行动概率: {action_probs}")
    print(f"选择的行动索引: {np.argmax(action_probs)}")
    print(f"选择的行动: {available_actions[np.argmax(action_probs)]}")
    
    # 更新模型状态
    selected_action = available_actions[np.argmax(action_probs)]
    reward = 0.7
    rational_model.update_behavior_state(observation, selected_action, reward, context)
    print(f"行为指标: {rational_model.get_behavior_metrics()}\n")
    
    # 2. 情感行为模型演示
    print("2. 情感行为模型演示")
    emotional_params = BehaviorParameters(
        rationality_level=0.6,
        emotional_weight=0.6,
        risk_tolerance=0.4
    )
    emotional_model = EmotionalBehaviorModel(emotional_params)
    
    # 模拟负面体验
    emotional_model.update_behavior_state(observation, selected_action, -0.3, context)
    print(f"负面体验后的情感维度: {emotional_model.emotion_dimensions}")
    
    action_probs_emotional = emotional_model.compute_action_probabilities(
        observation, available_actions, context
    )
    print(f"情感模型行动概率: {action_probs_emotional}")
    print(f"行为指标: {emotional_model.get_behavior_metrics()}\n")

def demo_role_specific_models():
    """演示角色特定的行为模型"""
    print("=== 角色特定行为模型演示 ===\n")
    
    roles = ['doctors', 'interns', 'patients', 'accountants', 'government']
    
    # 为每个角色创建模型
    models = {}
    for role in roles:
        models[role] = BehaviorModelFactory.create_role_specific_model(role)
        print(f"{role}角色行为模型类型: {models[role].behavior_type}")
        print(f"参数配置: 理性水平={models[role].parameters.rationality_level:.2f}, "
              f"情感权重={models[role].parameters.emotional_weight:.2f}, "
              f"社会影响={models[role].parameters.social_influence:.2f}")
    
    print()
    
    # 模拟多轮交互
    observation = np.array([0.6, 0.7, 0.5, 0.8, 0.4, 0.9, 0.3, 0.7])
    available_actions = np.array([
        [0.5, 0.3, 0.7, 0.2],
        [0.8, 0.1, 0.4, 0.6],
        [0.2, 0.9, 0.3, 0.5],
        [0.6, 0.5, 0.6, 0.4]
    ])
    
    print("多轮交互模拟:")
    for round_num in range(3):
        print(f"\n第 {round_num + 1} 轮:")
        
        # 所有角色选择行动
        all_actions = {}
        for role in roles:
            context = {
                'other_actions': {r: np.random.uniform(-0.5, 0.5, 4) for r in roles if r != role},
                'ethics_compliance': np.random.uniform(0.6, 0.9)
            }
            
            action_probs = models[role].compute_action_probabilities(
                observation, available_actions, context
            )
            selected_idx = np.argmax(action_probs)
            selected_action = available_actions[selected_idx]
            all_actions[role] = selected_action
            
            print(f"  {role}: 选择行动 {selected_idx}, 行动值 {selected_action}")
        
        # 模拟奖励并更新模型
        for role in roles:
            reward = np.random.uniform(-0.5, 1.0)  # 随机奖励
            context = {
                'other_actions': {r: all_actions[r] for r in roles if r != role},
                'interaction_outcomes': {r: np.random.uniform(-0.2, 0.3) for r in roles if r != role}
            }
            models[role].update_behavior_state(observation, all_actions[role], reward, context)
    
    # 显示最终状态
    print("\n最终行为状态:")
    for role in roles:
        metrics = models[role].get_behavior_metrics()
        print(f"{role}: 情绪={metrics['mood']:.2f}, 压力={metrics['stress']:.2f}, "
              f"信心={metrics['confidence']:.2f}, 声誉={metrics['reputation']:.2f}")

def demo_behavior_manager():
    """演示行为模型管理器"""
    print("\n=== 行为模型管理器演示 ===\n")
    
    # 创建管理器并初始化所有角色模型
    manager = BehaviorModelManager()
    manager.create_all_role_models()
    
    print("已创建的行为模型:")
    for role, model in manager.models.items():
        print(f"  {role}: {model.behavior_type.value}")
    
    # 模拟多步交互
    print("\n模拟系统运行:")
    
    for step in range(5):
        # 生成模拟数据
        observations = {
            role: np.random.uniform(0, 1, 8) for role in manager.models.keys()
        }
        
        actions = {
            role: np.random.uniform(-1, 1, 4) for role in manager.models.keys()
        }
        
        rewards = {
            role: np.random.uniform(-0.5, 1.0) for role in manager.models.keys()
        }
        
        context = {
            'system_stability': np.random.uniform(0.6, 0.9),
            'ethics_compliance': np.random.uniform(0.7, 0.95),
            'resource_distribution': np.random.uniform(0.3, 0.8, 5),
            'time_pressure': np.random.uniform(0.1, 0.6)
        }
        
        # 更新所有模型
        manager.update_all_models(observations, actions, rewards, context)
        
        print(f"步骤 {step + 1}: 已更新所有模型状态")
    
    # 获取集体行为指标
    collective_metrics = manager.get_collective_behavior_metrics()
    print(f"\n集体行为指标:")
    print(f"  平均情绪: {collective_metrics['collective']['avg_mood']:.3f}")
    print(f"  平均压力: {collective_metrics['collective']['avg_stress']:.3f}")
    print(f"  平均信心: {collective_metrics['collective']['avg_confidence']:.3f}")
    print(f"  情绪方差: {collective_metrics['collective']['mood_variance']:.3f}")
    print(f"  交互次数: {collective_metrics['collective']['interaction_count']}")
    
    # 分析行为模式
    patterns = manager.analyze_behavioral_patterns()
    print(f"\n行为模式分析:")
    for role, pattern in patterns.items():
        if role != 'collective' and isinstance(pattern, dict):
            print(f"  {role}:")
            print(f"    奖励趋势: {pattern.get('avg_reward_trend', 0):.3f}")
            print(f"    奖励稳定性: {pattern.get('reward_stability', 0):.3f}")
            print(f"    行动一致性: {pattern.get('action_consistency', 0):.3f}")

if __name__ == "__main__":
    # 运行所有演示
    demo_individual_behavior_models()
    demo_role_specific_models()
    demo_behavior_manager()
    
    print("\n=== 演示完成 ===")
    print("行为模型组件已成功完善，支持以下功能:")
    print("1. 多种行为类型：理性、有限理性、情感、社会性、适应性")
    print("2. 角色特定的行为模型配置")
    print("3. 动态行为状态更新")
    print("4. 社会交互和信任机制")
    print("5. 情感动力学建模")
    print("6. 适应性学习算法")
    print("7. 集体行为分析工具")