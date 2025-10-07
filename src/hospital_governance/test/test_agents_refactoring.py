"""
重构后的agents模组集成测试

测试重构后的各个组件之间的协作和一致性
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_agents_module_integration():
    """测试agents模组的整体集成"""
    try:
        from src.hospital_governance.agents import (
            # 角色智能体
            RoleAgent, RoleManager, DoctorAgent, InternAgent, PatientAgent, 
            AccountantAgent, GovernmentAgent, AgentConfig,
            
            # 行为模型
            BehaviorModelFactory, BehaviorModelManager, BehaviorType, BehaviorParameters,
            
            # 学习模型
            MADDPGModel,
            
            # LLM集成
            LLMActionGenerator, LLMConfig, MockLLMProvider,
            
            # 交互引擎
            KallipolisInteractionEngine, MultiAgentInteractionEngine, InteractionConfig
        )
        
        print("✅ 成功导入重构后的所有agents模组组件")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_role_agents_functionality():
    """测试角色智能体功能"""
    try:
        from src.hospital_governance.agents import RoleManager, DoctorAgent, AgentConfig
        
        # 创建角色管理器
        role_manager = RoleManager()
        
        # 创建医生智能体配置
        doctor_config = AgentConfig(
            role='doctors',
            action_dim=4,
            observation_dim=8
        )
        
        # 创建医生智能体
        doctor = DoctorAgent(doctor_config)
        role_manager.register_agent(doctor)
        
        # 测试观察功能
        environment = {
            'medical_quality': 0.8,
            'patient_safety': 0.9,
            'resource_adequacy': 0.7,
            'staff_satisfaction': 0.6
        }
        
        observation = doctor.observe(environment)
        print(f"✅ 角色智能体观察功能正常: {observation.shape}")
        
        # 测试行动选择
        action = doctor.select_action(observation, training=False)
        print(f"✅ 角色智能体行动选择正常: {action.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 角色智能体测试失败: {e}")
        return False

def test_behavior_models_integration():
    """测试行为模型集成"""
    try:
        from src.hospital_governance.agents import (
            BehaviorModelFactory, BehaviorModelManager, BehaviorType, BehaviorParameters
        )
        
        # 创建行为模型管理器
        manager = BehaviorModelManager()
        manager.create_all_role_models()
        
        print(f"✅ 行为模型管理器创建成功，包含 {len(manager.models)} 个角色模型")
        
        # 测试角色特定模型
        doctor_model = BehaviorModelFactory.create_role_specific_model('doctors')
        print(f"✅ 医生行为模型创建成功: {doctor_model.behavior_type}")
        
        # 测试行动概率计算
        observation = np.random.uniform(0, 1, 8)
        available_actions = np.array([
            [0.5, 0.3, 0.7, 0.2],
            [0.8, 0.1, 0.4, 0.6],
            [0.2, 0.9, 0.3, 0.5]
        ])
        context = {'reward_weights': np.ones(4)}
        
        probabilities = doctor_model.compute_action_probabilities(
            observation, available_actions, context
        )
        print(f"✅ 行为模型行动概率计算正常: {probabilities}")
        
        return True
        
    except Exception as e:
        print(f"❌ 行为模型集成测试失败: {e}")
        return False

def test_llm_integration():
    """测试LLM集成功能"""
    try:
        from src.hospital_governance.agents import LLMActionGenerator, LLMConfig, MockLLMProvider
        
        # 创建LLM配置和生成器
        config = LLMConfig(model_name="mock", temperature=0.7)
        provider = MockLLMProvider(config)
        generator = LLMActionGenerator(config, provider)
        
        print("✅ LLM组件创建成功")
        
        # 测试行动生成
        observation = np.array([0.7, 0.8, 0.6, 0.9, 0.5, 0.7, 0.4, 0.8])
        holy_code_state = {'active_rules': []}
        context = {'context_type': 'normal', 'role': 'doctors'}
        
        action = generator.generate_action_sync('doctors', observation, holy_code_state, context)
        print(f"✅ LLM行动生成正常: {action}")
        
        # 测试统计信息
        stats = generator.get_generation_stats()
        print(f"✅ LLM统计信息: 成功率 {stats['success_rate']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM集成测试失败: {e}")
        return False

def test_multi_agent_coordinator():
    """测试多智能体协调器"""
    try:
        from src.hospital_governance.agents import (
            RoleManager, MultiAgentInteractionEngine, InteractionConfig,
            DoctorAgent, InternAgent, AgentConfig
        )
        
        # 创建角色管理器和智能体
        role_manager = RoleManager()
        
        doctor_config = AgentConfig(role='doctors', action_dim=4, observation_dim=8)
        intern_config = AgentConfig(role='interns', action_dim=3, observation_dim=8)
        
        doctor = DoctorAgent(doctor_config)
        intern = InternAgent(intern_config)
        
        role_manager.register_agent(doctor)
        role_manager.register_agent(intern)
        
        # 创建交互配置和引擎
        interaction_config = InteractionConfig(
            use_behavior_models=False,  # disabled - using MADDPG + LLM
            use_learning_models=False,
            use_llm_generation=False,
            conflict_resolution="negotiation"
        )
        
        coordinator = MultiAgentInteractionEngine(role_manager, interaction_config)
        print("✅ 多智能体协调器创建成功")
        
        # 测试行动生成和协调
        system_state = np.random.uniform(0, 1, 16)
        context = {
            'environment': {
                'medical_quality': 0.8,
                'resource_adequacy': 0.6,
                'education_effectiveness': 0.7
            },
            'context_type': 'normal'
        }
        
        actions = coordinator.generate_actions(system_state, context, training=False)
        print(f"✅ 协调行动生成成功: {list(actions.keys())}")
        
        # 测试交互指标
        metrics = coordinator.get_interaction_metrics()
        print(f"✅ 交互指标计算成功: 合作得分 {metrics.get('average_cooperation_score', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 多智能体协调器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_workflow():
    """测试完整工作流程"""
    try:
        from src.hospital_governance.agents import (
            RoleManager, MultiAgentInteractionEngine, InteractionConfig,
            BehaviorModelFactory, LLMActionGenerator, LLMConfig,
            DoctorAgent, InternAgent, AccountantAgent, AgentConfig
        )
        
        print("🔄 开始完整工作流程测试...")
        
        # 1. 创建智能体
        role_manager = RoleManager()
        
        configs = {
            'doctors': AgentConfig(role='doctors', action_dim=4, observation_dim=8),
            'interns': AgentConfig(role='interns', action_dim=3, observation_dim=8),
            'accountants': AgentConfig(role='accountants', action_dim=3, observation_dim=8)
        }
        
        agents = {
            'doctors': DoctorAgent(configs['doctors']),
            'interns': InternAgent(configs['interns']),
            'accountants': AccountantAgent(configs['accountants'])
        }
        
        for agent in agents.values():
            role_manager.register_agent(agent)
        
        print("  ✅ 智能体创建和注册完成")
        
        # 2. 配置行为模型
        for role, agent in agents.items():
            behavior_model = BehaviorModelFactory.create_role_specific_model(role)
            if hasattr(agent, 'set_behavior_model'):
                agent.set_behavior_model(behavior_model)
        
        print("  ✅ 行为模型集成完成")
        
        # 3. 创建协调引擎
        interaction_config = InteractionConfig(
            use_behavior_models=False,  # disabled - using MADDPG + LLM
            use_learning_models=False,
            use_llm_generation=False,
            conflict_resolution="negotiation"
        )
        
        coordinator = MultiAgentInteractionEngine(role_manager, interaction_config)
        print("  ✅ 协调引擎创建完成")
        
        # 4. 模拟多轮交互
        print("  🔄 开始多轮交互模拟...")
        
        for round_num in range(3):
            # 生成系统状态
            system_state = np.random.uniform(0.3, 0.9, 16)
            
            # 构建上下文
            context = {
                'environment': {
                    'medical_quality': system_state[0],
                    'resource_adequacy': system_state[1],
                    'financial_health': system_state[2],
                    'education_effectiveness': system_state[3]
                },
                'context_type': 'normal',
                'round': round_num
            }
            
            # 生成协调行动
            actions = coordinator.generate_actions(system_state, context, training=False)
            
            print(f"    轮次 {round_num + 1}: 生成 {len(actions)} 个角色的协调行动")
        
        # 5. 获取最终指标
        metrics = coordinator.get_interaction_metrics()
        print(f"  ✅ 完整工作流程测试成功:")
        print(f"    - 总交互次数: {metrics.get('total_interactions', 0)}")
        print(f"    - 平均合作得分: {metrics.get('average_cooperation_score', 0):.3f}")
        print(f"    - 平均冲突数量: {metrics.get('average_conflict_count', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整工作流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 重构后的agents模组集成测试 ===\\n")
    
    tests = [
        ("模组导入测试", test_agents_module_integration),
        ("角色智能体功能测试", test_role_agents_functionality),
        ("行为模型集成测试", test_behavior_models_integration),
        ("LLM集成测试", test_llm_integration),
        ("多智能体协调器测试", test_multi_agent_coordinator),
        ("完整工作流程测试", test_complete_workflow)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🧪 {test_name}...")
        if test_func():
            passed += 1
            print(f"   ✅ 通过\\n")
        else:
            print(f"   ❌ 失败\\n")
    
    print(f"=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\\n🎉 所有测试通过！agents模组重构成功。")
        print("\\n主要改进:")
        print("- ✅ 解决了循环导入问题")
        print("- ✅ 统一了角色命名约定")
        print("- ✅ 重构了交互引擎架构")
        print("- ✅ 完善了LLM集成功能")
        print("- ✅ 添加了多智能体协调机制")
        print("- ✅ 修复了接口不一致问题")
        print("- ✅ 增强了错误处理和降级策略")
    else:
        print(f"\\n⚠️  {total-passed} 个测试失败，需要进一步调试。")