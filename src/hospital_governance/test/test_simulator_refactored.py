"""
测试重构后的仿真器
Test the refactored KallipolisSimulator
"""

import sys
import os
import asyncio
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_simulator_refactored():
    """测试重构后的仿真器基础功能"""
    print("🧪 开始测试重构后的仿真器...")
    
    try:
        # 导入重构后的仿真器
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        print("✅ 仿真器模块导入成功")
        
        # 创建配置
        config = SimulationConfig(
            max_steps=5,
            enable_llm_integration=True,
            llm_provider="mock",  # 使用Mock模式进行测试
            enable_reward_control=True,
            enable_holy_code=True,
            enable_crises=True
        )
        
        print("✅ 仿真配置创建成功")
        
        # 初始化仿真器
        simulator = KallipolisSimulator(config)
        
        print("✅ 仿真器初始化完成")
        
        # 检查组件状态
        component_status = simulator._get_component_status()
        print(f"📊 组件状态: {component_status}")
        
        # 获取初始报告
        initial_report = simulator.get_simulation_report()
        print(f"📋 初始报告: {initial_report['component_health']}")
        
        # 执行几步仿真
        print("\n🚀 开始执行仿真步骤...")
        results = []
        
        for i in range(3):
            step_data = simulator.step(training=False)
            results.append(step_data)
            
            print(f"Step {step_data['step']}: "
                  f"性能={step_data['metrics']['overall_performance']:.3f}, "
                  f"智能体行动数={len(step_data['agent_actions'])}, "
                  f"组件状态={sum(step_data['component_status'].values())}/6")
        
        print("\n✅ 仿真步骤执行完成")
        
        # 测试议会会议
        print("\n🏛️ 测试议会会议...")
        simulator.current_step = 7  # 设置为议会会议步骤
        parliament_step = simulator.step()
        
        if parliament_step['parliament_meeting']:
            print(f"✅ 议会会议执行成功: {parliament_step.get('parliament_result', {})}")
        else:
            print("⚠️ 未触发议会会议")
        
        # 获取最终报告
        final_report = simulator.get_simulation_report()
        print(f"\n📊 最终报告:")
        print(f"  - 总步数: {final_report['simulation_info']['current_step']}")
        print(f"  - 组件健康: {final_report['component_health']}")
        print(f"  - 智能体注册状态: {final_report.get('agent_registry_status', 'N/A')}")
        print(f"  - 奖励控制状态: {final_report['reward_control_status']}")
        
        # 重置测试
        print("\n🔄 测试重置功能...")
        simulator.reset()
        reset_report = simulator.get_simulation_report()
        print(f"✅ 重置后步数: {reset_report['simulation_info']['current_step']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 这可能是正常的，因为依赖模块可能不存在")
        return False
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_async_simulation():
    """测试异步仿真功能"""
    print("\n🧪 测试异步仿真功能...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        config = SimulationConfig(
            max_steps=3,
            llm_provider="mock"
        )
        
        simulator = KallipolisSimulator(config)
        
        # 异步运行测试
        async def async_test():
            await simulator.run_async(steps=3, training=False)
            return simulator.get_simulation_report()
        
        # 运行异步测试
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(async_test())
            print(f"✅ 异步仿真完成: {result['simulation_info']['current_step']}步")
            return True
        finally:
            loop.close()
        
    except Exception as e:
        print(f"❌ 异步测试失败: {e}")
        return False

def test_fallback_mode():
    """测试降级模式"""
    print("\n🧪 测试降级模式...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        # 创建一个可能触发降级模式的配置
        config = SimulationConfig(
            max_steps=2,
            enable_llm_integration=False,  # 禁用LLM
            enable_reward_control=False,   # 禁用奖励控制
            enable_holy_code=False         # 禁用神圣法典
        )
        
        simulator = KallipolisSimulator(config)
        
        # 强制进入降级模式
        simulator._initialize_fallback_mode()
        
        # 测试降级模式的决策
        fallback_actions = simulator._process_fallback_decisions()
        print(f"✅ 降级模式决策: {len(fallback_actions)}个智能体")
        
        # 测试降级模式的奖励
        fallback_rewards = simulator._compute_fallback_rewards()
        print(f"✅ 降级模式奖励: {len(fallback_rewards)}个奖励")
        
        # 执行降级模式步骤
        step_data = simulator.step()
        print(f"✅ 降级模式步骤完成: 性能={step_data['metrics']['overall_performance']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 降级模式测试失败: {e}")
        return False

def test_data_callback():
    """测试数据回调功能"""
    print("\n🧪 测试数据回调功能...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        config = SimulationConfig(max_steps=2, llm_provider="mock")
        simulator = KallipolisSimulator(config)
        
        # 数据收集器
        received_data = []
        
        def data_callback(step_data):
            received_data.append(step_data)
            print(f"📡 接收到步骤 {step_data['step']} 的数据")
        
        # 设置回调
        simulator.set_data_callback(data_callback)
        
        # 执行仿真
        simulator.step()
        simulator.step()
        
        print(f"✅ 回调测试完成: 接收到 {len(received_data)} 条数据")
        return len(received_data) == 2
        
    except Exception as e:
        print(f"❌ 数据回调测试失败: {e}")
        return False

def test_configuration_flexibility():
    """测试配置灵活性"""
    print("\n🧪 测试配置灵活性...")
    
    try:
        from src.hospital_governance.simulation.simulator_refactored import (
            KallipolisSimulator, SimulationConfig
        )
        
        # 测试不同的配置组合
        configs = [
            SimulationConfig(llm_provider="mock", enable_llm_integration=True),
            SimulationConfig(llm_provider="openai", enable_llm_integration=False),
            SimulationConfig(enable_crises=False, enable_holy_code=False),
            SimulationConfig(meeting_interval=3, crisis_probability=0.1)
        ]
        
        for i, config in enumerate(configs):
            simulator = KallipolisSimulator(config)
            report = simulator.get_simulation_report()
            print(f"✅ 配置 {i+1}: {report['component_health']}")
        
        print("✅ 配置灵活性测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 配置灵活性测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🏥 Kallipolis仿真器重构测试")
    print("=" * 50)
    
    # 执行所有测试
    tests = [
        ("基础功能测试", test_simulator_refactored),
        ("异步仿真测试", test_async_simulation),
        ("降级模式测试", test_fallback_mode),
        ("数据回调测试", test_data_callback),
        ("配置灵活性测试", test_configuration_flexibility)
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
    print(f"\n{'='*50}")
    print("🎯 测试总结:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n🎉 所有测试通过！重构成功！")
    else:
        print(f"\n⚠️ {total-passed}个测试失败，需要进一步调试")