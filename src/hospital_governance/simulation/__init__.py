"""
Kallipolis Simulation Module
统一入口，标准化接口，导出所有核心组件和工厂方法
"""

from .simulator import KallipolisSimulator, SimulationConfig
from .scenario_runner import ScenarioRunner, CrisisScenario
from .data_logger import DataLogger, SimulationMetrics
from .real_time_monitor import RealTimeMonitor, Dashboard, create_monitoring_system

def create_simulation_system(
    config: SimulationConfig = None,
    log_dir: str = "logs",
    monitor_config=None
) -> dict:
    """
    创建并组装仿真系统所有核心组件
    返回 {'simulator', 'data_logger', 'scenario_runner', 'monitor'}
    """
    simulator = KallipolisSimulator(config or SimulationConfig())
    data_logger = DataLogger(log_dir=log_dir)
    scenario_runner = ScenarioRunner(simulator)
    monitor = create_monitoring_system(simulator, data_logger) if monitor_config is None else RealTimeMonitor(simulator, data_logger, monitor_config)
    return {
        'simulator': simulator,
        'data_logger': data_logger,
        'scenario_runner': scenario_runner,
        'monitor': monitor
    }

__all__ = [
    'KallipolisSimulator',
    'SimulationConfig',
    'ScenarioRunner', 
    'CrisisScenario',
    'DataLogger',
    'SimulationMetrics',
    'RealTimeMonitor',
    'Dashboard',
    'create_monitoring_system',
    'create_simulation_system'
]