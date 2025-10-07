"""
奖励-状态联动集成测试
Reward-State Coupling Integration Test

验证奖励调节与状态转移方程的联动效果
"""

import numpy as np
import asyncio
from src.hospital_governance.core.state_space import StateSpace
from src.hospital_governance.core.system_dynamics import SystemDynamics
from src.hospital_governance.control.distributed_reward_control import DistributedRewardControlSystem, DistributedRewardControlConfig
from src.hospital_governance.control.role_specific_reward_controllers import DoctorRewardController, InternRewardController, PatientRewardController, AccountantRewardController, GovernmentRewardController
from src.hospital_governance.agents.role_agents import AgentConfig
from src.hospital_governance.control.reward_based_controller import RewardControlConfig

# 构造系统矩阵（简化为单位阵和零阵，实际应从system_matrices.yaml加载）
system_matrices = {
    'A': np.eye(16),
    'B': np.zeros((16, 17)),
    'E': np.zeros((16, 6)),
    'C': np.eye(16),
    'D': np.zeros((16, 17))
}

dynamics = SystemDynamics(system_matrices)

# 初始化状态空间
initial_state = np.random.uniform(0.3, 0.7, 16)
state_space = StateSpace(initial_state)

# 创建奖励控制系统
reward_config = DistributedRewardControlConfig()
reward_system = DistributedRewardControlSystem(reward_config)

# 创建并注册智能体及控制器
roles = ['doctor', 'intern', 'patient', 'accountant', 'government']
controller_classes = [DoctorRewardController, InternRewardController, PatientRewardController, AccountantRewardController, GovernmentRewardController]
agents = {}
for role, ctrl_cls in zip(roles, controller_classes):
    config = AgentConfig(role=role, action_dim=8, observation_dim=16)
    # mock agent: 只需有role属性
    class MockAgent:
        def __init__(self, config):
            self.role = config.role
            self.config = config
    agent = MockAgent(config)
    controller = ctrl_cls(RewardControlConfig(role=role), agent)
    reward_system.controllers[role] = controller
    reward_system.agents[role] = agent

# 测试循环：奖励调节与状态转移联动
async def test_reward_state_coupling(steps=5):
    x_t = state_space.get_state_vector()
    print("初始状态:", x_t)
    for t in range(steps):
        # 1. 构造基础奖励
        base_rewards = {role: np.random.uniform(0.3, 0.7) for role in roles}
        global_utility = np.mean(x_t)
        control_context = {role: {} for role in roles}
        # 2. 计算奖励调节
        final_rewards = await reward_system.compute_distributed_rewards(base_rewards, global_utility, control_context)
        print(f"\nStep {t+1} 奖励调节: {final_rewards}")
        # 3. 奖励转为控制输入（简化：直接用奖励填充u_t）
        u_t = np.zeros(17)
        for i, role in enumerate(roles):
            u_t[i] = final_rewards[role]
        # 4. 状态转移
        d_t = np.zeros(6)  # 无扰动
        x_next = dynamics.state_transition(x_t, u_t, d_t)
        print(f"Step {t+1} 状态转移后: {x_next}")
        # 5. 更新状态空间
        state_space.update_state(x_next)
        x_t = x_next
    print("\n最终状态:", x_t)

if __name__ == "__main__":
    asyncio.run(test_reward_state_coupling(steps=5))
