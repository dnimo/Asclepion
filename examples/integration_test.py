"""
Kallipolis系统集成测试脚本
"""
import os
import time
from src.hospital_governance.simulation.simulator import KallipolisSimulator, SimulationConfig
from src.hospital_governance.simulation.data_logger import DataLogger
from src.hospital_governance.simulation.scenario_runner import ScenarioRunner
from src.hospital_governance.simulation.real_time_monitor import create_monitoring_system

# 1. 初始化配置和组件
config = SimulationConfig()
simulator = KallipolisSimulator(config)
data_logger = DataLogger(log_dir="logs/test_session")
scenario_runner = ScenarioRunner(simulator)
monitor = create_monitoring_system(simulator, data_logger)

# 2. 启动数据记录会话
session_name = "integration_test_session"
data_logger.start_session(session_name, config.__dict__)

# 3. 运行一个预设场景
presets = scenario_runner.create_preset_scenarios()
scenario = presets['moderate_pandemic']
result = scenario_runner.run_scenario(scenario, steps=120)

# 4. 实时监控（模拟10步）
monitor.start_monitoring()
for _ in range(10):
    monitor.update()
    time.sleep(monitor.config.update_interval)
monitor.stop_monitoring()

# 5. 输出仿真摘要和监控摘要
print("\n仿真摘要:")
print(simulator.get_simulation_summary())

print("\n监控摘要:")
print(monitor.get_monitoring_summary())

print("\n场景统计:")
print(scenario_runner.get_scenario_statistics())

# 6. 结束数据记录会话
data_logger.end_session()

# 7. 保存场景结果
scenario_runner.save_scenario_results("logs/test_session/scenario_results.json")

print("\n集成测试完成！")
