"""
测试奖励控制系统角色名称映射修复
Test Reward Control System Role Name Mapping Fix
"""

import sys
import os
import logging

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_reward_control_integration():
    """测试奖励控制系统集成是否修复"""
    print("🧪 测试奖励控制系统角色名称映射修复...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        # 创建启用奖励控制的配置
        config = SimulationConfig(
            max_steps=3,
            enable_llm_integration=True,     # 启用LLM
            enable_reward_control=True,      # 启用奖励控制 - 关键测试点
            enable_holy_code=True,           # 启用神圣法典
            enable_crises=False,             # 简化测试
            llm_provider="mock"
        )
        
        print("✅ 配置创建成功")
        
        # 创建仿真器
        simulator = KallipolisSimulator(config)
        
        # 检查组件状态
        component_status = simulator._get_component_status()
        print(f"📊 组件状态: {component_status}")
        
        # 检查奖励控制系统是否正确初始化
        if simulator.reward_control_system:
            controllers = simulator.reward_control_system.controllers
            print(f"🎛️ 奖励控制器数量: {len(controllers)}")
            print("🔧 已注册的控制器:")
            for role in controllers.keys():
                print(f"   ✅ {role}")
                
            # 检查是否所有预期的单数角色都被注册
            expected_roles = ['doctor', 'intern', 'patient', 'accountant', 'government']
            missing_roles = []
            for role in expected_roles:
                if role not in controllers:
                    missing_roles.append(role)
            
            if missing_roles:
                print(f"❌ 缺失的角色控制器: {missing_roles}")
                return False
            else:
                print("✅ 所有角色控制器都已正确注册")
        else:
            print("❌ 奖励控制系统未初始化")
            return False
        
        # 执行一步仿真测试奖励分发
        print("\n🚀 测试奖励分发...")
        step_data = simulator.step(training=False)
        
        rewards = step_data.get('rewards', {})
        print(f"💰 奖励分发结果: {len(rewards)}个角色")
        for role, reward in rewards.items():
            print(f"   💰 {role}: {reward:.4f}")
        
        # 验证奖励是否使用了正确的角色名称（复数形式）
        expected_registry_roles = ['doctors', 'interns', 'patients', 'accountants', 'government']
        reward_roles = set(rewards.keys())
        expected_roles_set = set(expected_registry_roles)
        
        if reward_roles == expected_roles_set:
            print("✅ 奖励角色名称映射正确")
            return True
        else:
            print(f"❌ 奖励角色名称不匹配")
            print(f"   期望: {expected_roles_set}")
            print(f"   实际: {reward_roles}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 奖励控制系统角色名称映射修复测试")
    print("=" * 60)
    
    result = test_reward_control_integration()
    
    if result:
        print("\n🎉 修复成功！奖励控制系统角色名称映射正常工作")
        print("📋 修复内容:")
        print("  ✅ 智能体注册中心角色名称: doctors, interns, patients, accountants, government")
        print("  ✅ 奖励控制系统角色名称: doctor, intern, patient, accountant, government")
        print("  ✅ 自动映射转换: 复数 ↔ 单数")
        print("  ✅ 奖励分发使用正确的角色名称")
    else:
        print("\n❌ 修复失败，需要进一步调试")
    
    print("\n🔧 下一步测试: 完整仿真验证")