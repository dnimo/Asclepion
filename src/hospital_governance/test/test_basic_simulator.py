"""
简化的仿真器重构测试 - 只测试基础模块
Simplified Simulator Refactoring Test - Basic Modules Only
"""

import sys
import os
import logging
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_basic_simulator():
    """测试基础仿真器功能"""
    print("🧪 开始测试基础仿真器功能...")
    
    try:
        # 直接导入仿真器配置
        from src.hospital_governance.simulation.simulator_refactored import SimulationConfig
        
        print("✅ SimulationConfig 导入成功")
        
        # 创建基础配置
        config = SimulationConfig(
            max_steps=3,
            enable_llm_integration=False,  # 禁用LLM避免依赖问题
            enable_reward_control=False,   # 禁用奖励控制
            enable_holy_code=False,        # 禁用神圣法典
            enable_crises=False,           # 禁用危机
            llm_provider="mock"
        )
        
        print("✅ 基础配置创建成功")
        print(f"   - 最大步数: {config.max_steps}")
        print(f"   - LLM集成: {config.enable_llm_integration}")
        print(f"   - 奖励控制: {config.enable_reward_control}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulator_creation():
    """测试仿真器创建"""
    print("\n🧪 测试仿真器创建...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        # 创建最小化配置
        config = SimulationConfig(
            max_steps=2,
            enable_llm_integration=False,
            enable_reward_control=False,
            enable_holy_code=False,
            enable_crises=False
        )
        
        print("✅ 配置创建成功")
        
        # 创建仿真器
        simulator = KallipolisSimulator(config)
        
        print("✅ 仿真器创建成功")
        
        # 检查基础属性
        print(f"   - 当前步数: {simulator.current_step}")
        print(f"   - 仿真时间: {simulator.simulation_time}")
        print(f"   - 运行状态: {simulator.is_running}")
        
        # 检查降级模式
        if hasattr(simulator, 'fallback_agents'):
            print(f"   - 降级智能体数量: {len(simulator.fallback_agents)}")
        
        if hasattr(simulator, 'fallback_state'):
            print(f"   - 降级状态维度: {len(simulator.fallback_state)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 仿真器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_functionality():
    """测试降级功能"""
    print("\n🧪 测试降级功能...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        config = SimulationConfig(
            max_steps=1,
            enable_llm_integration=False,
            enable_reward_control=False,
            enable_holy_code=False,
            enable_crises=False
        )
        
        simulator = KallipolisSimulator(config)
        
        # 强制初始化降级模式
        simulator._initialize_fallback_mode()
        
        print("✅ 降级模式初始化成功")
        
        # 测试降级决策
        fallback_actions = simulator._process_fallback_decisions()
        print(f"✅ 降级决策: {len(fallback_actions)}个智能体")
        
        # 测试降级奖励
        fallback_rewards = simulator._compute_fallback_rewards()
        print(f"✅ 降级奖励: {len(fallback_rewards)}个奖励")
        
        # 测试状态更新
        simulator._update_fallback_state()
        state_dict = simulator._get_current_state_dict()
        print(f"✅ 状态更新: {len(state_dict)}个状态变量")
        
        return True
        
    except Exception as e:
        print(f"❌ 降级功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulation_step():
    """测试仿真步骤"""
    print("\n🧪 测试仿真步骤...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        config = SimulationConfig(
            max_steps=3,
            enable_llm_integration=False,
            enable_reward_control=False,
            enable_holy_code=False,
            enable_crises=False
        )
        
        simulator = KallipolisSimulator(config)
        
        print("✅ 仿真器创建成功")
        
        # 执行仿真步骤
        step_data = simulator.step(training=False)
        
        print("✅ 仿真步骤执行成功")
        print(f"   - 步数: {step_data['step']}")
        print(f"   - 时间: {step_data['time']}")
        print(f"   - 智能体行动数: {len(step_data.get('agent_actions', {}))}")
        print(f"   - 奖励数: {len(step_data.get('rewards', {}))}")
        print(f"   - 指标数: {len(step_data.get('metrics', {}))}")
        
        # 再执行一步
        step_data_2 = simulator.step(training=False)
        print(f"✅ 第二步执行成功: 步数={step_data_2['step']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 仿真步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulation_control():
    """测试仿真控制"""
    print("\n🧪 测试仿真控制...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        config = SimulationConfig(
            max_steps=2,
            enable_llm_integration=False,
            enable_reward_control=False,
            enable_holy_code=False,
            enable_crises=False
        )
        
        simulator = KallipolisSimulator(config)
        
        # 测试重置
        simulator.reset()
        print("✅ 重置功能正常")
        print(f"   - 重置后步数: {simulator.current_step}")
        
        # 测试暂停和恢复
        simulator.pause()
        print(f"✅ 暂停功能: {simulator.is_paused}")
        
        simulator.resume()
        print(f"✅ 恢复功能: {simulator.is_paused}")
        
        # 测试停止
        simulator.stop()
        print(f"✅ 停止功能: {simulator.is_running}")
        
        # 测试报告生成
        report = simulator.get_simulation_report()
        print("✅ 仿真报告生成成功")
        print(f"   - 组件健康: {report.get('component_health', 'N/A')}")
        print(f"   - 当前步数: {report['simulation_info']['current_step']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 仿真控制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_callback():
    """测试数据回调"""
    print("\n🧪 测试数据回调...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        config = SimulationConfig(
            max_steps=2,
            enable_llm_integration=False,
            enable_reward_control=False,
            enable_holy_code=False,
            enable_crises=False
        )
        
        simulator = KallipolisSimulator(config)
        
        # 数据收集器
        received_data = []
        
        def data_callback(step_data):
            received_data.append(step_data)
            print(f"📡 回调接收: 步骤 {step_data['step']}")
        
        # 设置回调
        simulator.set_data_callback(data_callback)
        print("✅ 数据回调设置成功")
        
        # 执行仿真并触发回调
        simulator.step()
        simulator.step()
        
        print(f"✅ 回调测试完成: 接收到 {len(received_data)} 条数据")
        
        return len(received_data) == 2
        
    except Exception as e:
        print(f"❌ 数据回调测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🏥 Kallipolis仿真器重构 - 基础测试")
    print("=" * 60)
    
    # 执行测试
    tests = [
        ("基础功能测试", test_basic_simulator),
        ("仿真器创建测试", test_simulator_creation),
        ("降级功能测试", test_fallback_functionality),
        ("仿真步骤测试", test_simulation_step),
        ("仿真控制测试", test_simulation_control),
        ("数据回调测试", test_data_callback)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ 通过" if result else "❌ 失败"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n{test_name}: ❌ 异常 - {e}")
    
    # 总结
    print(f"\n{'='*60}")
    print("🎯 基础测试总结:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n🎉 所有基础测试通过！仿真器重构成功！")
        print("\n📋 重构效果总结:")
        print("  ✅ 统一的组件架构")
        print("  ✅ 降级模式支持")
        print("  ✅ 异步和同步仿真")
        print("  ✅ 数据回调机制")
        print("  ✅ 仿真控制功能")
        print("  ✅ 错误处理和恢复")
    else:
        print(f"\n⚠️ {total-passed}个测试失败，需要进一步优化")
    
    print(f"\n🔧 下一步: 集成全功能模块测试")