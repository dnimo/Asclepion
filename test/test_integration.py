"""
行为模型与角色智能体集成测试
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_integration_with_role_agents():
    """测试行为模型与角色智能体的集成"""
    try:
        from src.hospital_governance.agents.role_agents import DoctorAgent, AgentConfig
        from src.hospital_governance.agents.behavior_models import BehaviorModelFactory
        
        # 创建医生智能体
        config = AgentConfig(
            role='doctors',
            action_dim=4,
            observation_dim=8,
            learning_rate=0.001
        )
        
        doctor_agent = DoctorAgent(config)
        print(f"✓ 创建医生智能体: {doctor_agent.role}")
        
        # 为智能体配置行为模型
        behavior_model = BehaviorModelFactory.create_role_specific_model('doctors')
        doctor_agent.set_behavior_model(behavior_model)
        print(f"✓ 为智能体设置行为模型: {behavior_model.behavior_type}")
        
        # 测试集成决策流程
        environment = {
            'medical_quality': 0.7,
            'patient_safety': 0.8,
            'resource_adequacy': 0.6,
            'staff_satisfaction': 0.7,
            'operational_efficiency': 0.75,
            'waiting_times': 0.3,
            'crisis_severity': 0.2,
            'ethics_compliance': 0.9
        }
        
        # 智能体观察环境
        observation = doctor_agent.observe(environment)
        print(f"✓ 智能体观察环境: {observation.shape}")
        
        # 使用行为模型影响决策
        if doctor_agent.behavior_model:
            available_actions = np.array([
                [0.5, 0.3, 0.7, 0.2],
                [0.8, 0.1, 0.4, 0.6],
                [0.2, 0.9, 0.3, 0.5]
            ])
            
            context = {
                'reward_weights': np.ones(4),
                'ethics_compliance': environment['ethics_compliance']
            }
            
            action_probs = behavior_model.compute_action_probabilities(
                observation, available_actions, context
            )
            
            # 选择行动
            selected_action_idx = np.argmax(action_probs)
            selected_action = available_actions[selected_action_idx]
            
            print(f"✓ 行为模型选择行动: {selected_action}")
            
            # 更新行为状态
            reward = 0.8  # 模拟奖励
            behavior_model.update_behavior_state(observation, selected_action, reward, context)
            
            # 获取行为指标
            metrics = behavior_model.get_behavior_metrics()
            print(f"✓ 行为指标更新: 情绪={metrics['mood']:.2f}, 信心={metrics['confidence']:.2f}")
        
        print("\n🎉 行为模型与角色智能体集成测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_agent_behavior_interaction():
    """测试多智能体行为交互"""
    try:
        from src.hospital_governance.agents.behavior_models import BehaviorModelManager
        
        # 创建行为模型管理器
        manager = BehaviorModelManager()
        manager.create_all_role_models()
        
        print(f"✓ 创建多智能体行为管理器，包含 {len(manager.models)} 个角色")
        
        # 模拟多智能体交互场景
        roles = list(manager.models.keys())
        
        # 生成模拟环境状态
        observations = {}
        actions = {}
        rewards = {}
        
        for role in roles:
            observations[role] = np.random.uniform(0, 1, 8)
            actions[role] = np.random.uniform(-1, 1, 4)
            rewards[role] = np.random.uniform(0, 1)
        
        # 构造交互上下文
        context = {
            'other_actions': {role: actions[role] for role in roles},
            'ethics_compliance': 0.85,
            'system_stability': 0.75,
            'resource_distribution': np.array([0.6, 0.7, 0.5, 0.8, 0.4]),
            'interaction_outcomes': {role: np.random.uniform(-0.2, 0.3) for role in roles}
        }
        
        # 更新所有智能体的行为状态
        manager.update_all_models(observations, actions, rewards, context)
        print("✓ 成功更新所有智能体行为状态")
        
        # 获取集体行为指标
        collective_metrics = manager.get_collective_behavior_metrics()
        print(f"✓ 集体行为指标: 平均情绪={collective_metrics['collective']['avg_mood']:.2f}")
        
        print("\n🎉 多智能体行为交互测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 多智能体交互测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 行为模型集成测试 ===\n")
    
    success1 = test_integration_with_role_agents()
    print()
    success2 = test_multi_agent_behavior_interaction()
    
    if success1 and success2:
        print("\n✅ 所有集成测试通过！行为模型组件已成功完善并与现有系统集成。")
    else:
        print("\n❌ 部分测试失败，需要进一步调试。")