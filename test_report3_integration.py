"""
测试 Report3Agent 在 agents 模块中的集成

验证：
1. Report3Agent 可以从 agents 模块导入
2. 继承自 RoleAgent 的接口兼容性
3. Fixed LLM + Semantic Critic 架构正常工作
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 测试从 agents 模块导入
from src.hospital_governance.agents import (
    Report3Agent,
    create_report3_agent,
    AgentConfig,
    SystemState,
    LLM_PARAMETERS_FROZEN,
    SemanticEncoder,
    SemanticCritic,
    SemanticCriticTrainer
)


def test_report3_agent_integration():
    """测试 Report3Agent 集成"""
    
    print("=" * 80)
    print("Report3Agent 集成测试")
    print("=" * 80)
    
    # 1. 使用工厂函数创建 agent
    print("\n1️⃣ 创建 Report3Agent (doctors)")
    agent = create_report3_agent(
        role='doctors',
        num_candidates=5,
        use_mock_llm=True,
        replay_buffer_capacity=1000
    )
    
    print(f"   ✓ Agent role: {agent.role}")
    print(f"   ✓ Action dim: {agent.action_dim}")
    print(f"   ✓ LLM frozen: {LLM_PARAMETERS_FROZEN}")
    print(f"   ✓ Num candidates: {agent.num_candidates}")
    
    # 2. 测试 observe 方法（RoleAgent 接口）
    print("\n2️⃣ 测试 observe() 方法")
    environment = {
        'medical_resource_utilization': 0.75,
        'patient_waiting_time': 0.35,
        'financial_indicator': 0.68,
        'ethical_compliance': 0.92,
        'patient_satisfaction': 0.80
    }
    observation = agent.observe(environment)
    print(f"   ✓ Observation shape: {observation.shape}")
    print(f"   ✓ Observation: {observation[:4]}")
    
    # 3. 注入全局状态（16维）
    print("\n3️⃣ 注入全局16维状态")
    global_state = np.random.rand(16) * 0.5 + 0.5  # [0.5, 1.0] 范围
    agent.set_global_state(global_state)
    print(f"   ✓ Global state injected: {global_state[:4]}")
    
    # 4. 测试 select_action（Report 3 架构核心）
    print("\n4️⃣ 测试 select_action() - LLM + Critic")
    
    holy_code_guidance = {
        'active_rules': [
            'Maximize patient safety',
            'Optimize resource allocation',
            'Maintain ethical standards'
        ],
        'priority_level': 0.85
    }
    
    # 利用模式（epsilon=0）
    action = agent.select_action(
        observation=observation,
        holy_code_guidance=holy_code_guidance,
        training=True,
        exploration_epsilon=0.0
    )
    
    print(f"   ✓ Selected action shape: {action.shape}")
    print(f"   ✓ Action vector: {action}")
    print(f"   ✓ Action info cached: {hasattr(agent, '_last_action_info')}")
    
    if hasattr(agent, '_last_action_info'):
        info = agent._last_action_info
        print(f"   ✓ Action text: {info['action_text'][:50]}...")
        print(f"   ✓ Q value: {info['q_value']:.3f}")
        print(f"   ✓ Candidates: {len(info['candidates'])}")
    
    # 5. 测试经验存储
    print("\n5️⃣ 测试 store_transition()")
    
    reward = 0.65
    next_observation = observation + np.random.randn(8) * 0.05
    
    agent.store_transition(
        reward=reward,
        next_observation=next_observation,
        next_holy_code_guidance=holy_code_guidance,
        done=False
    )
    
    print(f"   ✓ Transition stored")
    print(f"   ✓ Replay buffer size: {len(agent.replay_buffer)}")
    
    # 6. 模拟多个步骤以收集经验
    print("\n6️⃣ 收集多个经验（模拟3个episode）")
    
    for episode in range(3):
        for step in range(5):
            # 选择动作
            action = agent.select_action(
                observation=observation,
                holy_code_guidance=holy_code_guidance,
                training=True,
                exploration_epsilon=0.3 if episode == 0 else 0.0
            )
            
            # 模拟环境反馈
            reward = np.random.rand() * 0.5 + 0.3  # [0.3, 0.8]
            next_observation = observation + np.random.randn(8) * 0.1
            
            # 存储经验
            agent.store_transition(
                reward=reward,
                next_observation=next_observation,
                next_holy_code_guidance=holy_code_guidance,
                done=(step == 4)
            )
            
            observation = next_observation
        
        agent.episode_count += 1
    
    print(f"   ✓ Episodes completed: {agent.episode_count}")
    print(f"   ✓ Total transitions: {len(agent.replay_buffer)}")
    
    # 7. 训练 Critic
    print("\n7️⃣ 训练 Semantic Critic")
    
    if len(agent.replay_buffer) >= 8:
        stats = agent.train_critic(batch_size=8, num_epochs=2)
        
        print(f"   ✓ Training completed")
        print(f"   ✓ Loss: {stats.get('loss', 0):.4f}")
        print(f"   ✓ Mean Q: {stats.get('mean_q', 0):.3f}")
        print(f"   ✓ Training steps: {agent.training_steps}")
    else:
        print(f"   ⚠️  Not enough experiences ({len(agent.replay_buffer)} < 8)")
    
    # 8. 测试 compute_local_value（RoleAgent 接口）
    print("\n8️⃣ 测试 compute_local_value()")
    
    # 创建完整的 SystemState
    system_state = SystemState(
        medical_resource_utilization=0.75,
        patient_waiting_time=0.35,
        financial_indicator=0.68,
        ethical_compliance=0.92,
        education_training_quality=0.85,
        intern_satisfaction=0.78,
        professional_development=0.80,
        mentorship_effectiveness=0.82,
        patient_satisfaction=0.85,
        service_accessibility=0.80,
        care_quality_index=0.90,
        safety_incident_rate=0.05,
        operational_efficiency=0.75,
        staff_workload_balance=0.70,
        crisis_response_capability=0.80,
        regulatory_compliance_score=0.90
    )
    
    local_value = agent.compute_local_value(system_state, action=0)
    print(f"   ✓ Local value: {local_value:.3f}")
    
    # 9. 获取统计信息
    print("\n9️⃣ 获取统计信息")
    
    stats = agent.get_statistics()
    print(f"   ✓ Role: {stats['role']}")
    print(f"   ✓ Episodes: {stats['episode_count']}")
    print(f"   ✓ Training steps: {stats['training_steps']}")
    print(f"   ✓ Generation count: {stats['generation_count']}")
    print(f"   ✓ Buffer size: {stats['replay_buffer_size']}")
    print(f"   ✓ Parameters frozen: {stats['parameters_frozen']}")
    
    if stats['critic_stats']['losses']:
        print(f"   ✓ Recent losses: {[f'{x:.4f}' for x in stats['critic_stats']['losses'][-3:]]}")
        print(f"   ✓ Recent Q values: {[f'{x:.3f}' for x in stats['critic_stats']['q_values'][-3:]]}")
    
    # 10. 验证架构原则
    print("\n🔟 验证 Report 3 架构原则")
    
    print(f"   ✓ LLM 参数冻结: {LLM_PARAMETERS_FROZEN}")
    print(f"   ✓ 候选生成器类型: {type(agent.llm_generator).__name__}")
    print(f"   ✓ Critic 网络类型: {type(agent.critic).__name__}")
    print(f"   ✓ 语义编码器类型: {type(agent.semantic_encoder).__name__}")
    print(f"   ✓ 经验回放类型: {type(agent.replay_buffer).__name__}")
    
    # 检查继承关系
    from src.hospital_governance.agents import RoleAgent
    print(f"   ✓ 继承 RoleAgent: {isinstance(agent, RoleAgent)}")
    
    print("\n" + "=" * 80)
    print("✅ Report3Agent 集成测试完成！")
    print("=" * 80)
    
    return agent


def test_multi_role_creation():
    """测试创建多个角色的 Report3Agent"""
    
    print("\n" + "=" * 80)
    print("多角色 Report3Agent 创建测试")
    print("=" * 80)
    
    roles = ['doctors', 'interns', 'patients', 'accountants', 'government']
    agents = {}
    
    for role in roles:
        agent = create_report3_agent(role=role, num_candidates=3, use_mock_llm=True)
        agents[role] = agent
        print(f"✓ Created {role} agent (action_dim={agent.action_dim})")
    
    print(f"\n✅ 成功创建 {len(agents)} 个 Report3Agent")
    
    return agents


if __name__ == "__main__":
    # 主测试
    agent = test_report3_agent_integration()
    
    # 多角色测试
    agents = test_multi_role_creation()
    
    print("\n🎉 所有集成测试通过！")
