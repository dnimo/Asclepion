"""
详细测试智能体动作生成
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.hospital_governance.agents import create_agent_registry

def detailed_action_test():
    print("🔍 详细测试智能体动作生成")
    print("=" * 50)
    
    # 创建注册中心
    registry = create_agent_registry(llm_provider="mock")
    
    # 注册一个医生智能体进行详细测试
    agent = registry.register_agent('doctors')
    llm_generator = registry.get_llm_generator('doctors')
    
    print(f"✅ 已注册医生智能体，LLM生成器: {llm_generator is not None}")
    
    # 生成测试观测
    test_observation = np.random.uniform(0.3, 0.7, 8)
    print(f"📊 测试观测值: {test_observation}")
    print(f"📊 观测平均值: {np.mean(test_observation):.3f}")
    
    # 计算预期的默认动作值
    base_value = np.mean(test_observation) - 0.5
    expected_action_value = base_value * 0.5
    print(f"📊 预期动作值: {expected_action_value:.3f}")
    
    # 测试LLM生成
    context = {'role': 'doctors', 'context_type': 'test'}
    action = llm_generator.generate_action_sync('doctors', test_observation, {}, context)
    
    print(f"🎯 生成的动作: {action}")
    print(f"🎯 动作维度: {action.shape}")
    print(f"🎯 数值范围: [{action.min():.3f}, {action.max():.3f}]")
    print(f"🎯 所有值相同: {np.allclose(action, action[0])}")
    
    # 测试不同观测值
    print("\\n🧪 测试不同观测值的影响:")
    test_cases = [
        ("低值观测", np.full(8, 0.2)),
        ("中值观测", np.full(8, 0.5)), 
        ("高值观测", np.full(8, 0.8)),
        ("混合观测", np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.6, 0.4]))
    ]
    
    for case_name, obs in test_cases:
        action = llm_generator.generate_action_sync('doctors', obs, {}, context)
        obs_mean = np.mean(obs)
        expected = (obs_mean - 0.5) * 0.5
        print(f"  {case_name}: 观测均值={obs_mean:.2f}, 预期动作={expected:.3f}, 实际动作={action[0]:.3f}")

if __name__ == "__main__":
    detailed_action_test()