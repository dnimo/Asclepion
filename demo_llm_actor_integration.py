#!/usr/bin/env python3
"""
LLM-Actor与现有Agent系统集成演示

展示如何将LLM-Actor决策系统集成到基于价值函数的策略梯度架构中
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.hospital_governance.agents.role_agents import (
    RoleManager, SystemState, AgentConfig
)
from src.hospital_governance.agents.llm_actor_system import LLMActorDecisionSystem


def demo_llm_actor_integration():
    """演示LLM-Actor集成"""
    print("=" * 80)
    print("🎯 LLM-Actor与价值函数架构集成演示")
    print("=" * 80)
    
    # 1. 创建传统的Agent系统
    print("\n📦 步骤1: 创建传统Agent系统（策略梯度 + 价值函数）")
    print("-" * 80)
    
    role_manager = RoleManager()
    role_manager.create_all_agents()
    
    print(f"✅ 创建了 {role_manager.get_agent_count()} 个智能体:")
    for role in role_manager.agents.keys():
        agent = role_manager.get_agent(role)
        print(f"  • {role}: observation_dim={agent.state_dim}, action_dim={agent.action_dim}")
        print(f"    收益权重: α={agent.alpha:.2f}, β={agent.beta:.2f}, γ={agent.gamma:.2f}")
    
    # 2. 创建LLM-Actor决策系统
    print("\n🤖 步骤2: 创建LLM-Actor决策系统")
    print("-" * 80)
    
    llm_actor_system = LLMActorDecisionSystem(
        llm_provider="mock",
        n_candidates=5,
        state_dim=16,  # 完整的系统状态维度
        device='cpu'
    )
    
    print("✅ LLM-Actor系统已创建")
    print(f"  • 候选生成器: 每次生成 {llm_actor_system.n_candidates} 个候选")
    print(f"  • 语义嵌入器: {llm_actor_system.embedder.embedding_dim}维向量")
    print(f"  • Actor选择器: 状态维度={llm_actor_system.selector.state_dim}")
    
    # 3. 为所有Agent启用LLM-Actor
    print("\n🔗 步骤3: 启用LLM-Actor模式")
    print("-" * 80)
    
    role_manager.enable_llm_actor_for_all(llm_actor_system)
    
    print("✅ 所有智能体已切换到LLM-Actor模式")
    print("  决策流程: LLM生成候选 → Actor选择 → 解析为控制向量")
    
    # 4. 模拟环境并生成决策
    print("\n🎮 步骤4: 模拟多轮决策")
    print("-" * 80)
    
    # 创建模拟环境
    env_state = {
        'medical_resource_utilization': 0.82,
        'patient_waiting_time': 0.35,
        'financial_indicator': 0.68,
        'ethical_compliance': 0.91,
        'education_training_quality': 0.75,
        'intern_satisfaction': 0.70,
        'patient_satisfaction': 0.83,
        'service_accessibility': 0.78,
        'care_quality_index': 0.88,
        'safety_incident_rate': 0.08,
        'operational_efficiency': 0.72,
        'staff_workload_balance': 0.65,
        'crisis_response_capability': 0.80,
        'regulatory_compliance_score': 0.89
    }
    
    system_state = SystemState(
        medical_resource_utilization=env_state['medical_resource_utilization'],
        patient_waiting_time=env_state['patient_waiting_time'],
        financial_indicator=env_state['financial_indicator'],
        ethical_compliance=env_state['ethical_compliance'],
        education_training_quality=env_state['education_training_quality'],
        intern_satisfaction=env_state['intern_satisfaction'],
        professional_development=0.68,
        mentorship_effectiveness=0.76,
        patient_satisfaction=env_state['patient_satisfaction'],
        service_accessibility=env_state['service_accessibility'],
        care_quality_index=env_state['care_quality_index'],
        safety_incident_rate=env_state['safety_incident_rate'],
        operational_efficiency=env_state['operational_efficiency'],
        staff_workload_balance=env_state['staff_workload_balance'],
        crisis_response_capability=env_state['crisis_response_capability'],
        regulatory_compliance_score=env_state['regulatory_compliance_score']
    )
    # 将全局16维状态注入到所有agent，确保价值网络评估是环境级
    for agent in role_manager.agents.values():
        try:
            agent.set_global_state(system_state.to_vector())
        except Exception:
            pass
    
    ideal_state = SystemState.from_vector(np.full(16, 0.9))  # 理想状态
    
    # 执行3轮决策
    n_steps = 3
    for step in range(n_steps):
        print(f"\n--- 第 {step+1}/{n_steps} 轮决策 ---")
        
        step_tokens = 0
        step_actions = {}
        
        for role, agent in role_manager.agents.items():
            # 获取观测
            observation = agent.observe(env_state)
            
            # 选择动作（使用LLM-Actor）
            action = agent.select_action(observation, training=True)
            step_actions[role] = action
            
            # 显示决策信息
            if agent._last_llm_result:
                result = agent._last_llm_result
                print(f"\n{role}:")
                print(f"  候选数量: {len(result.candidates)}")
                print(f"  选择的动作: {result.selected_action}")
                print(f"  Token消耗: {result.tokens_used}")
                print(f"  动作向量: {result.action_vector[:3]}... (前3维)")
                step_tokens += result.tokens_used
            else:
                print(f"\n{role}: [策略梯度模式]")
        
        print(f"\n本轮总Token消耗: {step_tokens}")
    
    # 5. 展示统计信息
    print("\n📊 步骤5: LLM使用统计")
    print("-" * 80)
    
    llm_stats = role_manager.get_llm_statistics_summary()
    
    for role, stats in llm_stats.items():
        if role == '_aggregate':
            print(f"\n📈 汇总统计:")
            print(f"  • 总Token消耗: {stats['total_tokens_all_agents']}")
            print(f"  • 总调用次数: {stats['total_calls_all_agents']}")
            print(f"  • 平均每次Token: {stats['avg_tokens_per_call']:.1f}")
        else:
            print(f"\n{role}:")
            print(f"  • 调用次数: {stats['total_calls']}")
            print(f"  • Token消耗: {stats['total_tokens']}")
            print(f"  • 平均Token: {stats['avg_tokens_per_call']:.1f}")
    
    # 6. 演示奖励计算（包含Token成本）
    print("\n💰 步骤6: 奖励计算（包含Token成本）")
    print("-" * 80)
    
    token_cost_factor = 0.001  # 每个token的成本系数
    rejection_penalty_factor = 0.1  # 拒绝惩罚系数
    
    for role, agent in role_manager.agents.items():
        if agent._last_llm_result:
            result = agent._last_llm_result
            
            # 计算Token成本
            token_cost = result.tokens_used * token_cost_factor
            
            # 计算拒绝惩罚
            rejection_penalty = rejection_penalty_factor if result.was_rejected else 0.0
            
            # 计算基础奖励（假设）
            base_reward = agent.compute_reward(
                system_state=system_state,
                action=0,  # 简化，实际需要从action_vector解析
                global_utility=0.8,
                ideal_state=ideal_state,
                token_cost=token_cost,
                rejection_penalty=rejection_penalty
            )
            
            print(f"\n{role}:")
            print(f"  基础收益: α*U + β*V - γ*D = {base_reward + token_cost + rejection_penalty:.3f}")
            print(f"  Token成本: -{token_cost:.4f} ({result.tokens_used} tokens)")
            print(f"  拒绝惩罚: -{rejection_penalty:.4f}")
            print(f"  最终收益: {base_reward:.3f}")
    
    # 7. 对比模式切换
    print("\n🔄 步骤7: 演示模式切换")
    print("-" * 80)
    
    print("\n禁用LLM-Actor，切换回策略梯度...")
    role_manager.disable_llm_actor_for_all()
    
    # 使用策略梯度模式
    doctor = role_manager.get_agent('doctors')
    observation = doctor.observe(env_state)
    action_pg = doctor.select_action(observation, training=True)
    
    print(f"✅ doctors使用策略梯度模式: {action_pg[:3]}... (前3维)")
    
    print("\n重新启用LLM-Actor...")
    role_manager.enable_llm_actor_for_all(llm_actor_system)
    
    action_llm = doctor.select_action(observation, training=True)
    print(f"✅ doctors使用LLM-Actor模式: {action_llm[:3]}... (前3维)")
    
    # 8. 性能指标对比
    print("\n📈 步骤8: 性能指标")
    print("-" * 80)
    
    perf_summary = role_manager.get_performance_summary()
    
    for role, metrics in perf_summary.items():
        print(f"\n{role}:")
        print(f"  性能分数: {metrics.get('performance_score', 0.0):.3f}")
        print(f"  累积奖励: {metrics.get('cumulative_reward', 0.0):.3f}")
        print(f"  策略范数: {metrics.get('policy_norm', 0.0):.3f}")
        
        if 'llm_enabled' in metrics:
            print(f"  LLM启用: {metrics['llm_enabled']}")
            print(f"  LLM调用: {metrics.get('total_calls', 0)}")
            print(f"  Token总计: {metrics.get('total_tokens', 0)}")
    
    print("\n" + "=" * 80)
    print("✅ LLM-Actor集成演示完成！")
    print("=" * 80)
    
    print("\n💡 关键要点:")
    print("  1. ✅ LLM-Actor与价值函数架构完全兼容")
    print("  2. ✅ 可以动态切换决策模式（LLM vs 策略梯度）")
    print("  3. ✅ 奖励函数扩展支持Token成本和拒绝惩罚")
    print("  4. ✅ 保留原有的策略更新和价值估计机制")
    print("  5. ✅ 完整的统计和监控功能")
    
    return role_manager, llm_actor_system


def demo_hybrid_training():
    """演示混合训练模式"""
    print("\n" + "=" * 80)
    print("🔬 混合训练模式演示")
    print("=" * 80)
    
    print("\n混合训练策略:")
    print("  1. 初期（Episode 0-100）: 纯策略梯度，快速探索")
    print("  2. 中期（Episode 100-500）: LLM辅助，生成高质量候选")
    print("  3. 后期（Episode 500+）: 选择性使用LLM（关键时刻）")
    
    # 创建系统
    role_manager = RoleManager()
    role_manager.create_all_agents()
    
    llm_actor_system = LLMActorDecisionSystem(
        llm_provider="mock",
        n_candidates=5,
        state_dim=16
    )
    
    # 模拟训练阶段
    for episode in [50, 150, 300, 600]:
        print(f"\n--- Episode {episode} ---")
        
        if episode < 100:
            print("🎯 策略: 纯策略梯度（探索）")
            role_manager.disable_llm_actor_for_all()
            mode = "Policy Gradient"
        elif episode < 500:
            print("🤖 策略: LLM-Actor（学习高质量动作）")
            role_manager.enable_llm_actor_for_all(llm_actor_system)
            mode = "LLM-Actor"
        else:
            # 选择性使用：只在关键情况下调用LLM
            print("⚡ 策略: 选择性LLM（仅关键时刻）")
            # 这里可以根据状态特征决定是否启用
            role_manager.enable_llm_actor_for_all(llm_actor_system)
            mode = "Selective LLM"
        
        print(f"  当前模式: {mode}")
    
    print("\n✅ 混合训练演示完成")


if __name__ == "__main__":
    # 主集成演示
    role_manager, llm_system = demo_llm_actor_integration()
    
    # 混合训练演示
    demo_hybrid_training()
    
    print("\n🎊 所有演示完成！")
    print("\n📚 下一步:")
    print("  1. 在simulator中集成interaction_engine")
    print("  2. 实现完整的训练循环")
    print("  3. 添加可视化和监控")
    print("  4. 接入真实LLM API（OpenAI/Claude）")
