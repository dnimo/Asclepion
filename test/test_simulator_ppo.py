"""
测试 KallipolisSimulator + CTDE PPO 训练集成
"""
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.hospital_governance.simulation.simulator import KallipolisSimulator, SimulationConfig
from src.hospital_governance.agents.learning_models import CTDEPPOModel, RolloutBuffer

# 初始化仿真器和 PPO 模型
sim_config = SimulationConfig(max_steps=10)
simulator = KallipolisSimulator(config=sim_config)

# 假定有3个智能体，每个观测维度为16，动作为2（可根据实际调整）
n_agents = 3
obs_dim = 16
n_actions = 2
ppo_model = CTDEPPOModel(obs_dim=obs_dim, n_agents=n_agents, n_actions=n_actions, device='cpu')

# 仿真一个回合，收集经验
simulator.reset()
for _ in range(sim_config.max_steps):
    step_data = simulator.step()
    # 经验已自动收集到 simulator.rollout_buffer

# 回合结束后，进行 PPO 训练
if simulator.rollout_buffer:
    print(f"收集到 {len(simulator.rollout_buffer.buffers[0])} 步经验，开始训练...")
    ppo_model.train(simulator.rollout_buffer)
    print("PPO训练完成！")
else:
    print("未收集到经验数据，检查仿真器配置和 agent_actions 逻辑。")
