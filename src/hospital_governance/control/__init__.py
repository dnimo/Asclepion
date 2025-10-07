"""
医院治理控制系统 - 奖励驱动控制架构
Hospital Governance Control System - Reward-Driven Control Architecture

基于智能体奖励逻辑的新一代医院治理控制系统
支持传统控制系统的向后兼容
"""

# 新的奖励驱动控制系统
from .reward_based_controller import (
    RewardBasedController, 
    RewardControlConfig,
    RewardControllerFactory
)
from .role_specific_reward_controllers import (
    DoctorRewardController,
    InternRewardController, 
    PatientRewardController,
    AccountantRewardController,
    GovernmentRewardController
)
from .distributed_reward_control import (
    DistributedRewardControlSystem,
    DistributedRewardControlConfig,
    get_global_reward_control_system
)
from .reward_control_adapter import (
    RewardControlAdapter,
    LegacyControlMapping
)

# 传统控制系统（向后兼容）
# 主要导出接口
__all__ = [
    # 新的奖励驱动控制系统
    'RewardBasedController',
    'RewardControlConfig', 
    'RewardControllerFactory',
    'DoctorRewardController',
    'InternRewardController',
    'PatientRewardController', 
    'AccountantRewardController',
    'GovernmentRewardController',
    'DistributedRewardControlSystem',
    'DistributedRewardControlConfig',
    'get_global_reward_control_system',
    'RewardControlAdapter',
    'LegacyControlMapping',
]

# 推荐使用的控制系统
RECOMMENDED_CONTROL_SYSTEM = 'DistributedRewardControlSystem'
LEGACY_CONTROL_SYSTEM = None

