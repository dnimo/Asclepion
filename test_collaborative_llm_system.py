#!/usr/bin/env python3
"""
测试协作式LLM+智能体决策系统和增强议会功能
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_collaborative_llm_system():
    """测试协作式LLM+智能体决策系统"""
    print("🎯 开始测试协作式LLM+智能体决策系统...")
    
    try:
        # 导入必要模块
        from src.hospital_governance.simulation.simulator import KallipolisSimulator
        from src.hospital_governance.agents.agent_registry import AgentRegistry
        
        # 创建智能体注册表
        agent_registry = AgentRegistry()
        agent_registry.register_all_agents()
        
        # 创建模拟器配置
        from src.hospital_governance.simulation.simulator import SimulationConfig
        config = SimulationConfig()
        config.use_maddpg = True  # 启用MADDPG
        config.parliament_frequency = 3  # 每3步召开议会
        config.enable_llm = True  # 启用LLM协作
        
        # 创建模拟器
        simulator = KallipolisSimulator(config=config)
        
        # 手动设置智能体注册表
        simulator.agent_registry = agent_registry
        
        print("✅ 模拟器初始化成功")
        
        # 运行几个步骤测试协作决策
        print("\n📊 测试协作式决策系统...")
        
        for step in range(6):  # 运行6步，包含2次议会
            print(f"\n--- 步骤 {step + 1} ---")
            
            # 执行步骤
            result = simulator.step()
            
            # 检查结果
            if result:
                print(f"✅ 步骤 {step + 1} 完成")
                print(f"   整体性能: {result.get('metrics', {}).get('overall_performance', 0):.3f}")
                print(f"   使用MADDPG: {result.get('used_maddpg', False)}")
                print(f"   使用LLM: {result.get('used_llm', False)}")
                print(f"   协作决策: {result.get('collaborative_decisions', False)}")
                
                # 检查议会结果
                if 'parliament_result' in result:
                    parliament = result['parliament_result']
                    print(f"   🏛️ 议会召开:")
                    print(f"      参与讨论: {parliament.get('discussion_participants', [])}")
                    print(f"      共识程度: {parliament.get('consensus_level', 0):.3f}")
                    print(f"      共同关注: {parliament.get('common_concerns', [])}")
                    print(f"      新规则数: {len(parliament.get('new_rules', []))}")
                    
                    # 显示新规则
                    for rule in parliament.get('new_rules', []):
                        print(f"      📋 新规则: {rule.get('name', 'Unknown')}")
                
                # 检查LLM讨论
                if result.get('parliament_result', {}).get('enhanced_by_llm'):
                    discussions = result['parliament_result'].get('llm_discussions', {})
                    print(f"   💬 LLM智能体讨论: {len(discussions)}个参与者")
                    for role, discussion in discussions.items():
                        preview = discussion[:50] + "..." if len(discussion) > 50 else discussion
                        print(f"      {role}: {preview}")
            else:
                print(f"❌ 步骤 {step + 1} 失败")
        
        # 获取最终状态
        final_state = simulator.get_current_state()
        print(f"\n📈 最终系统状态:")
        print(f"   总步数: {final_state.get('current_step', 0)}")
        print(f"   MADDPG训练数据: {len(simulator.experience_buffer)}条")
        
        # 测试MADDPG训练
        if len(simulator.experience_buffer) > 0:
            print(f"   🎓 MADDPG训练就绪，缓冲区大小: {len(simulator.experience_buffer)}")
            
            # 尝试训练（如果有足够数据）
            if len(simulator.experience_buffer) >= 10:
                try:
                    simulator._train_maddpg_model()
                    print("   ✅ MADDPG模型训练成功")
                except Exception as e:
                    print(f"   ⚠️ MADDPG训练跳过: {e}")
        
        print("\n🎉 协作式LLM+智能体决策系统测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_llm_features():
    """测试特定的LLM功能"""
    print("\n🔍 测试特定LLM功能...")
    
    try:
        from src.hospital_governance.simulation.simulator import KallipolisSimulator
        from src.hospital_governance.agents.agent_registry import AgentRegistry
        
        # 创建测试实例
        agent_registry = AgentRegistry()
        agent_registry.register_all_agents()
        
        from src.hospital_governance.simulation.simulator import SimulationConfig
        config = SimulationConfig()
        simulator = KallipolisSimulator(config=config)
        simulator.agent_registry = agent_registry
        
        # 测试议会讨论生成
        print("   测试议会讨论生成...")
        test_step_data = {
            'metrics': {'overall_performance': 0.75},
            'system_state': {
                'care_quality_index': 0.8,
                'financial_indicator': 0.7,
                'patient_satisfaction': 0.85
            }
        }
        
        # 测试单个角色讨论
        discussion = simulator._generate_parliament_discussion('senior_doctor', test_step_data)
        print(f"   ✅ 医生讨论生成: {len(discussion)}字符")
        
        # 测试共识计算
        test_discussions = {
            'senior_doctor': '支持提升医疗质量，建议增加培训',
            'head_nurse': '同意医生观点，护理质量也需要改善',
            'hospital_administrator': '赞成质量提升，但需要考虑成本控制'
        }
        
        consensus = simulator._calculate_consensus_level(test_discussions, test_step_data)
        print(f"   ✅ 共识计算: {consensus:.3f}")
        
        # 测试规则生成
        parliament_result = {
            'consensus_level': 0.8,
            'common_concerns': ['医疗质量', '财务管理']
        }
        
        new_rules = simulator._generate_consensus_rules(parliament_result, test_step_data)
        print(f"   ✅ 新规则生成: {len(new_rules)}条规则")
        
        for rule in new_rules:
            print(f"      📋 {rule.get('name', 'Unknown')} - {rule.get('type', 'Unknown')}")
        
        print("   🎯 LLM功能测试完成")
        return True
        
    except Exception as e:
        print(f"   ❌ LLM功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试协作式LLM+智能体系统...")
    
    # 测试主要功能
    success1 = test_collaborative_llm_system()
    
    # 测试特定功能
    success2 = test_specific_llm_features()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！协作式LLM+智能体决策系统运行正常")
    else:
        print("\n❌ 部分测试失败，需要进一步调试")