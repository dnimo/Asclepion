"""
奖励控制适配器 - 与现有系统的集成接口
Reward Control Adapter - Integration interface with existing systems

提供向后兼容性和平滑迁移路径
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

from .distributed_reward_control import DistributedRewardControlSystem, DistributedRewardControlConfig
from .reward_based_controller import RewardBasedController
from ..core.kallipolis_mathematical_core import SystemState
from ..agents.role_agents import RoleAgent

logger = logging.getLogger(__name__)


@dataclass
class LegacyControlMapping:
    """传统控制映射配置"""
    control_signal_to_reward_scale: float = 2.0
    reward_to_control_signal_scale: float = 0.5
    compatibility_mode: bool = True


class RewardControlAdapter:
    """奖励控制适配器
    
    功能：
    1. 将传统控制信号转换为奖励调节
    2. 将奖励输出转换为控制信号（向后兼容）
    3. 提供渐进式迁移支持
    4. 桥接新旧控制架构
    """
    
    def __init__(self, 
                 reward_control_system: DistributedRewardControlSystem,
                 legacy_mapping: Optional[LegacyControlMapping] = None):
        
        self.reward_system = reward_control_system
        self.legacy_mapping = legacy_mapping or LegacyControlMapping()
        
        # 兼容性映射
        self.role_mapping = {
            'doctor': 'primary_controller',
            'intern': 'observer_controller', 
            'patient': 'patient_controller',
            'accountant': 'constraint_controller',
            'government': 'government_controller'
        }
        
        # 控制信号缓存
        self.last_control_signals: Dict[str, np.ndarray] = {}
        self.last_reward_adjustments: Dict[str, float] = {}
        
        logger.info("Initialized reward control adapter")
    
    def convert_legacy_control_to_rewards(self, 
                                        legacy_control_signals: Dict[str, np.ndarray],
                                        system_state: SystemState) -> Dict[str, float]:
        """将传统控制信号转换为奖励调节
        
        Args:
            legacy_control_signals: 传统控制器输出的控制信号
            system_state: 当前系统状态
            
        Returns:
            reward_adjustments: 转换后的奖励调节值
        """
        
        reward_adjustments = {}
        
        for controller_name, control_signal in legacy_control_signals.items():
            # 找到对应的角色
            role = self._map_controller_to_role(controller_name)
            if not role:
                continue
            
            # 转换控制信号到奖励调节
            reward_adjustment = self._control_signal_to_reward(
                control_signal, role, system_state
            )
            
            reward_adjustments[role] = reward_adjustment
        
        logger.debug(f"Converted {len(legacy_control_signals)} control signals to rewards")
        
        return reward_adjustments
    
    def convert_rewards_to_legacy_control(self, 
                                        reward_adjustments: Dict[str, float],
                                        system_state: SystemState) -> Dict[str, np.ndarray]:
        """将奖励调节转换为传统控制信号
        
        Args:
            reward_adjustments: 奖励调节值
            system_state: 当前系统状态
            
        Returns:
            legacy_control_signals: 转换后的传统控制信号
        """
        
        legacy_control_signals = {}
        
        for role, reward_adjustment in reward_adjustments.items():
            # 找到对应的控制器
            controller_name = self._map_role_to_controller(role)
            if not controller_name:
                continue
            
            # 转换奖励到控制信号
            control_signal = self._reward_to_control_signal(
                reward_adjustment, role, system_state
            )
            
            legacy_control_signals[controller_name] = control_signal
        
        logger.debug(f"Converted {len(reward_adjustments)} rewards to control signals")
        
        return legacy_control_signals
    
    def _map_controller_to_role(self, controller_name: str) -> Optional[str]:
        """映射控制器名称到角色"""
        for role, controller in self.role_mapping.items():
            if controller_name.lower().startswith(controller.split('_')[0]):
                return role
        return None
    
    def _map_role_to_controller(self, role: str) -> Optional[str]:
        """映射角色到控制器名称"""
        return self.role_mapping.get(role)
    
    def _control_signal_to_reward(self, 
                                control_signal: np.ndarray,
                                role: str,
                                system_state: SystemState) -> float:
        """将控制信号转换为奖励调节"""
        
        # 计算控制信号的强度
        signal_magnitude = np.linalg.norm(control_signal)
        
        # 基于角色的转换系数
        role_scaling = {
            'doctor': 1.2,      # 医生控制信号权重高
            'intern': 0.8,      # 实习生权重适中
            'patient': 1.0,     # 患者权重标准
            'accountant': 1.1,  # 会计权重较高
            'government': 0.9   # 政府权重适中
        }
        
        scaling_factor = role_scaling.get(role, 1.0)
        
        # 转换公式：控制信号强度 -> 奖励调节
        reward_adjustment = (signal_magnitude * 
                           self.legacy_mapping.control_signal_to_reward_scale * 
                           scaling_factor)
        
        # 根据控制信号方向调整奖励符号
        if len(control_signal) > 0:
            # 正向控制信号 -> 正向奖励
            # 负向控制信号 -> 惩罚（负奖励）
            primary_signal = control_signal[0]
            if primary_signal < -0.1:
                reward_adjustment = -reward_adjustment
        
        # 限制奖励调节范围
        reward_adjustment = np.clip(reward_adjustment, -2.0, 2.0)
        
        return reward_adjustment
    
    def _reward_to_control_signal(self, 
                                reward_adjustment: float,
                                role: str,
                                system_state: SystemState) -> np.ndarray:
        """将奖励调节转换为控制信号"""
        
        # 基于角色确定控制信号维度
        control_dimensions = {
            'doctor': 4,        # 医生有4个控制输入
            'intern': 4,        # 实习生有4个控制输入
            'patient': 3,       # 患者有3个控制输入
            'accountant': 3,    # 会计有3个控制输入
            'government': 2     # 政府有2个控制输入
        }
        
        dim = control_dimensions.get(role, 3)
        
        # 转换奖励到控制信号强度
        signal_magnitude = (abs(reward_adjustment) * 
                          self.legacy_mapping.reward_to_control_signal_scale)
        
        # 生成控制信号向量
        if reward_adjustment > 0:
            # 正向奖励 -> 正向控制信号
            control_signal = np.ones(dim) * signal_magnitude
        else:
            # 负向奖励 -> 负向控制信号
            control_signal = -np.ones(dim) * signal_magnitude
        
        # 基于角色和系统状态调整控制信号分布
        control_signal = self._adjust_control_signal_distribution(
            control_signal, role, system_state
        )
        
        # 限制控制信号范围
        control_signal = np.clip(control_signal, -1.0, 1.0)
        
        return control_signal
    
    def _adjust_control_signal_distribution(self, 
                                          control_signal: np.ndarray,
                                          role: str,
                                          system_state: SystemState) -> np.ndarray:
        """调整控制信号分布以匹配角色特性"""
        
        adjusted_signal = control_signal.copy()
        
        if role == 'doctor':
            # 医生：重点关注质量和安全
            if len(adjusted_signal) >= 4:
                adjusted_signal[3] *= 1.5  # 质量控制权重增加
                adjusted_signal[0] *= 1.2  # 资源分配权重增加
                
        elif role == 'intern':
            # 实习生：重点关注学习和培训
            if len(adjusted_signal) >= 4:
                adjusted_signal[0] *= 1.3  # 培训权重增加
                adjusted_signal[1] *= 0.8  # 其他权重降低
                
        elif role == 'patient':
            # 患者：重点关注满意度和可及性
            if len(adjusted_signal) >= 3:
                adjusted_signal[0] *= 1.4  # 满意度权重增加
                adjusted_signal[1] *= 1.2  # 可及性权重增加
                
        elif role == 'accountant':
            # 会计：重点关注效率和成本
            if len(adjusted_signal) >= 3:
                adjusted_signal[1] *= 1.5  # 效率权重增加
                adjusted_signal[2] *= 1.3  # 成本控制权重增加
                
        elif role == 'government':
            # 政府：重点关注合规和公平
            if len(adjusted_signal) >= 2:
                adjusted_signal[0] *= 1.4  # 合规权重增加
                adjusted_signal[1] *= 1.2  # 公平权重增加
        
        return adjusted_signal
    
    async def hybrid_control_step(self, 
                                legacy_control_signals: Dict[str, np.ndarray],
                                base_rewards: Dict[str, float],
                                global_utility: float,
                                system_state: SystemState,
                                target_state: SystemState,
                                control_context: Dict[str, Any]) -> Dict[str, Any]:
        """混合控制步骤 - 同时支持传统控制和奖励控制
        
        Returns:
            control_outputs: 包含两种控制输出的字典
        """
        
        # 1. 将传统控制信号转换为奖励调节
        legacy_reward_adjustments = self.convert_legacy_control_to_rewards(
            legacy_control_signals, system_state
        )
        
        # 2. 合并基础奖励和传统控制转换的奖励
        combined_rewards = {}
        for role in set(list(base_rewards.keys()) + list(legacy_reward_adjustments.keys())):
            base = base_rewards.get(role, 0.0)
            legacy = legacy_reward_adjustments.get(role, 0.0)
            combined_rewards[role] = base + legacy * 0.5  # 50%权重融合
        
        # 3. 执行分布式奖励控制
        self.reward_system.update_system_states(system_state, target_state)
        final_rewards = await self.reward_system.compute_distributed_rewards(
            combined_rewards, global_utility, control_context
        )
        
        # 4. 转换奖励回传统控制信号（向后兼容）
        converted_control_signals = self.convert_rewards_to_legacy_control(
            final_rewards, system_state
        )
        
        # 5. 缓存结果
        self.last_control_signals = converted_control_signals
        self.last_reward_adjustments = final_rewards
        
        return {
            'reward_adjustments': final_rewards,
            'legacy_control_signals': converted_control_signals,
            'hybrid_metrics': self._compute_hybrid_metrics(),
            'system_metrics': self.reward_system.get_system_metrics()
        }
    
    def _compute_hybrid_metrics(self) -> Dict[str, float]:
        """计算混合控制指标"""
        
        metrics = {}
        
        # 控制信号与奖励的一致性
        if self.last_control_signals and self.last_reward_adjustments:
            consistency_scores = []
            
            for role, reward in self.last_reward_adjustments.items():
                controller_name = self._map_role_to_controller(role)
                if controller_name and controller_name in self.last_control_signals:
                    control_signal = self.last_control_signals[controller_name]
                    # 计算一致性（简化指标）
                    signal_sign = np.sign(np.mean(control_signal))
                    reward_sign = np.sign(reward)
                    consistency = 1.0 if signal_sign == reward_sign else 0.0
                    consistency_scores.append(consistency)
            
            metrics['control_reward_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # 系统响应性
        if hasattr(self.reward_system, 'convergence_history') and self.reward_system.convergence_history:
            metrics['system_responsiveness'] = 1.0 / (np.mean(self.reward_system.convergence_history[-5:]) + 1e-8)
        else:
            metrics['system_responsiveness'] = 0.0
        
        return metrics
    
    def enable_legacy_mode(self, enabled: bool = True):
        """启用/禁用传统兼容模式"""
        self.legacy_mapping.compatibility_mode = enabled
        logger.info(f"Legacy compatibility mode: {'enabled' if enabled else 'disabled'}")
    
    def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态信息"""
        
        # 统计已注册的奖励控制器
        registered_controllers = len(self.reward_system.controllers)
        expected_controllers = 5  # 5个角色
        
        migration_progress = registered_controllers / expected_controllers
        
        return {
            'migration_progress': migration_progress,
            'registered_controllers': list(self.reward_system.controllers.keys()),
            'legacy_mode_enabled': self.legacy_mapping.compatibility_mode,
            'adapter_ready': migration_progress >= 1.0,
            'control_system_metrics': self.reward_system.get_system_metrics()
        }