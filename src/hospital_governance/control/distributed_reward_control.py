"""
分布式奖励控制系统 - 替代原有的distributed_control.py
Distributed Reward Control System

基于智能体奖励逻辑的分布式控制架构，替代传统控制理论方法
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .reward_based_controller import RewardBasedController, RewardControllerFactory
from .role_specific_reward_controllers import (
    DoctorRewardController, InternRewardController, PatientRewardController,
    AccountantRewardController, GovernmentRewardController
)
from ..core.kallipolis_mathematical_core import SystemState
from ..agents.role_agents import RoleAgent
from ..holy_code.holy_code_manager import HolyCodeManager

logger = logging.getLogger(__name__)


@dataclass
class DistributedRewardControlConfig:
    """分布式奖励控制配置"""
    global_coordination_weight: float = 0.3
    local_autonomy_weight: float = 0.7
    consensus_threshold: float = 0.8
    max_iterations: int = 100
    convergence_tolerance: float = 1e-4
    
    # 角色权重配置
    role_influence_weights: Dict[str, float] = field(default_factory=lambda: {
        'doctor': 0.25,
        'intern': 0.15,
        'patient': 0.25,
        'accountant': 0.2,
        'government': 0.15
    })
    
    # 奖励同步配置
    reward_sync_interval: float = 1.0  # 秒
    global_reward_scaling: float = 1.0
    adaptive_coordination: bool = True


class DistributedRewardControlSystem:
    """分布式奖励控制系统
    
    核心功能：
    1. 协调多个智能体的奖励调节
    2. 实现全局最优与局部自主的平衡
    3. 处理奖励冲突和协调问题
    4. 动态调整控制策略
    """
    
    def __init__(self, config: DistributedRewardControlConfig):
        self.config = config
        
        # 控制器管理
        self.controllers: Dict[str, RewardBasedController] = {}
        self.agents: Dict[str, RoleAgent] = {}
        
        # 系统状态
        self.current_state: Optional[SystemState] = None
        self.target_state: Optional[SystemState] = None
        self.ideal_state: Optional[SystemState] = None
        
        # 奖励协调
        self.global_reward_signal: float = 0.0
        self.role_reward_signals: Dict[str, float] = {}
        self.reward_history: List[Dict[str, float]] = []
        
        # 控制协调
        self.coordination_matrix: np.ndarray = np.eye(5)  # 5个角色的协调矩阵
        self.consensus_reached: bool = False
        self.last_consensus_time: float = 0.0
        
        # 神圣法典管理器（伦理约束与治理接口）
        self.holy_code_manager: Optional[HolyCodeManager] = None
        
        # 性能监控
        self.control_metrics: Dict[str, Any] = {}
        self.convergence_history: List[float] = []
        
        logger.info("Initialized distributed reward control system")
    
    def register_agent(self, role: str, agent: RoleAgent, 
                      controller_config: Optional[Dict[str, Any]] = None):
        """注册智能体及其奖励控制器"""
        # 创建角色特定的奖励控制器
        controller = RewardControllerFactory.create_controller(
            role, agent, controller_config
        )
        
        self.controllers[role] = controller
        self.agents[role] = agent
        self.role_reward_signals[role] = 0.0
        
        logger.info(f"Registered {role} agent with reward controller")
    
    def set_holy_code_manager(self, holy_code_manager: HolyCodeManager):
        """设置神圣法典管理器（伦理约束与治理接口）"""
        self.holy_code_manager = holy_code_manager
    
    def update_system_states(self, current: SystemState, target: SystemState, 
                           ideal: Optional[SystemState] = None):
        """更新系统状态"""
        self.current_state = current
        self.target_state = target
        if ideal:
            self.ideal_state = ideal
        
        logger.debug("Updated system states for reward control")
    
    async def compute_distributed_rewards(self, 
                                        base_rewards: Dict[str, float],
                                        global_utility: float,
                                        control_context: Dict[str, Any]) -> Dict[str, float]:
        """
        计算分布式奖励调节
        
        核心算法：
        1. 并行计算各角色的局部奖励调节
        2. 计算全局协调信号
        3. 解决奖励冲突
        4. 达成奖励共识
        """
        if not self.current_state or not self.target_state:
            logger.warning("System states not set, returning base rewards")
            return base_rewards
        
        # 1. 并行计算局部奖励调节
        local_adjustments = await self._compute_local_reward_adjustments(
            base_rewards, global_utility, control_context
        )
        
        # 2. 计算全局协调信号
        global_coordination = self._compute_global_coordination_signal(
            local_adjustments, global_utility
        )
        
        # 3. 解决奖励冲突
        conflict_resolved_rewards = self._resolve_reward_conflicts(
            local_adjustments, global_coordination
        )
        
        # 4. 应用伦理约束
        if self.ethical_system:
            ethical_constrained_rewards = self._apply_ethical_constraints(
                conflict_resolved_rewards, control_context
            )
        else:
            ethical_constrained_rewards = conflict_resolved_rewards
        
        # 5. 达成共识
        final_rewards = await self._achieve_reward_consensus(
            ethical_constrained_rewards, control_context
        )
        
        # 6. 更新系统状态
        self._update_control_state(final_rewards, local_adjustments)
        
        return final_rewards
    
    async def _compute_local_reward_adjustments(self, 
                                              base_rewards: Dict[str, float],
                                              global_utility: float,
                                              context: Dict[str, Any]) -> Dict[str, float]:
        """并行计算局部奖励调节"""
        
        async def compute_role_adjustment(role: str, base_reward: float):
            """计算单个角色的奖励调节"""
            controller = self.controllers[role]
            try:
                adjusted_reward = controller.compute_reward_adjustment(
                    self.current_state, self.target_state, base_reward,
                    global_utility, context.get(role, {})
                )
                return role, adjusted_reward
            except Exception as e:
                logger.error(f"Error computing reward for {role}: {e}")
                return role, base_reward
        
        # 并行执行
        tasks = [
            compute_role_adjustment(role, base_rewards.get(role, 0.0))
            for role in self.controllers.keys()
        ]
        
        results = await asyncio.gather(*tasks)
        return dict(results)
    
    def _compute_global_coordination_signal(self, 
                                          local_adjustments: Dict[str, float],
                                          global_utility: float) -> float:
        """计算全局协调信号"""
        
        # 计算奖励差异程度
        rewards = list(local_adjustments.values())
        reward_variance = np.var(rewards) if len(rewards) > 1 else 0.0
        mean_reward = np.mean(rewards)
        
        # 基于全局效用的协调信号
        utility_signal = global_utility * self.config.global_reward_scaling
        
        # 基于奖励方差的协调信号（减少冲突）
        variance_signal = -reward_variance * 0.5
        
        # 基于历史趋势的协调信号
        trend_signal = 0.0
        if len(self.reward_history) > 3:
            recent_means = [np.mean(list(h.values())) for h in self.reward_history[-3:]]
            trend = np.polyfit(range(3), recent_means, 1)[0]
            trend_signal = trend * 0.3
        
        global_signal = utility_signal + variance_signal + trend_signal
        
        # 更新全局奖励信号
        self.global_reward_signal = (0.7 * self.global_reward_signal + 
                                   0.3 * global_signal)
        
        logger.debug(f"Global coordination signal: {global_signal:.3f}")
        
        return global_signal
    
    def _resolve_reward_conflicts(self, 
                                local_adjustments: Dict[str, float],
                                global_coordination: float) -> Dict[str, float]:
        """解决奖励冲突"""
        
        resolved_rewards = {}
        
        # 计算角色影响权重
        influence_weights = self.config.role_influence_weights
        
        for role, local_reward in local_adjustments.items():
            # 获取角色权重
            role_weight = influence_weights.get(role, 0.2)
            
            # 计算全局协调影响
            global_influence = global_coordination * self.config.global_coordination_weight
            local_influence = local_reward * self.config.local_autonomy_weight
            
            # 加权融合
            resolved_reward = (role_weight * local_influence + 
                             (1 - role_weight) * global_influence)
            
            resolved_rewards[role] = resolved_reward
        
        # 检查冲突情况
        self._detect_and_log_conflicts(local_adjustments, resolved_rewards)
        
        return resolved_rewards
    
    def _detect_and_log_conflicts(self, 
                                local_rewards: Dict[str, float],
                                resolved_rewards: Dict[str, float]):
        """检测并记录奖励冲突"""
        
        conflicts = []
        for role in local_rewards:
            local = local_rewards[role]
            resolved = resolved_rewards[role]
            
            if abs(local - resolved) > 0.2:  # 显著差异
                conflicts.append(f"{role}: {local:.3f} -> {resolved:.3f}")
        
        if conflicts:
            logger.warning(f"Reward conflicts detected: {', '.join(conflicts)}")
    
    def _apply_ethical_constraints(self, rewards: Dict[str, float], context: Dict[str, Any]) -> Dict[str, float]:
        """应用伦理约束（通过HolyCodeManager接口）"""
        if not self.holy_code_manager:
            return rewards
        # 通过HolyCodeManager获取决策建议和约束
        system_status = self.holy_code_manager.get_system_status() if hasattr(self.holy_code_manager, 'get_system_status') else {}
        # 可根据system_status或决策建议调整奖励
        # 示例：如果处于危机模式，降低部分角色奖励
        constrained_rewards = rewards.copy()
        if system_status.get('system_state', {}).get('active_crisis', False):
            for role in constrained_rewards:
                constrained_rewards[role] *= 0.8  # 危机时收紧激励
        # 可扩展：根据HolyCodeManager的更多接口和建议动态调整
        return constrained_rewards
    
    async def _achieve_reward_consensus(self, 
                                      rewards: Dict[str, float],
                                      context: Dict[str, Any]) -> Dict[str, float]:
        """达成奖励共识"""
        
        current_rewards = rewards.copy()
        iterations = 0
        
        while iterations < self.config.max_iterations:
            # 计算共识程度
            consensus_score = self._compute_consensus_score(current_rewards)
            
            if consensus_score >= self.config.consensus_threshold:
                self.consensus_reached = True
                break
            
            # 更新奖励以提高共识
            current_rewards = self._update_rewards_for_consensus(current_rewards)
            iterations += 1
        
        if not self.consensus_reached:
            logger.warning(f"Consensus not reached after {iterations} iterations")
        
        return current_rewards
    
    def _compute_consensus_score(self, rewards: Dict[str, float]) -> float:
        """计算共识程度"""
        if len(rewards) < 2:
            return 1.0
        
        values = list(rewards.values())
        mean_reward = np.mean(values)
        std_reward = np.std(values)
        
        # 基于方差的共识分数（方差越小，共识越高）
        if std_reward == 0:
            return 1.0
        
        consensus = 1.0 / (1.0 + std_reward)
        return consensus
    
    def _update_rewards_for_consensus(self, 
                                    rewards: Dict[str, float]) -> Dict[str, float]:
        """更新奖励以提高共识"""
        
        mean_reward = np.mean(list(rewards.values()))
        updated_rewards = {}
        
        for role, reward in rewards.items():
            # 向平均值靠拢，但保持角色特异性
            role_weight = self.config.role_influence_weights.get(role, 0.2)
            consensus_factor = 0.1  # 共识调节因子
            
            updated_reward = (reward * (1 - consensus_factor) + 
                            mean_reward * consensus_factor * (1 - role_weight))
            
            updated_rewards[role] = updated_reward
        
        return updated_rewards
    
    def _update_control_state(self, 
                            final_rewards: Dict[str, float],
                            local_adjustments: Dict[str, float]):
        """更新控制系统状态"""
        
        # 更新角色奖励信号
        self.role_reward_signals = final_rewards.copy()
        
        # 更新奖励历史
        self.reward_history.append(final_rewards.copy())
        if len(self.reward_history) > 100:  # 保持最近100个记录
            self.reward_history.pop(0)
        
        # 计算收敛指标
        if len(self.reward_history) > 1:
            prev_rewards = self.reward_history[-2]
            convergence = self._compute_convergence_metric(prev_rewards, final_rewards)
            self.convergence_history.append(convergence)
            if len(self.convergence_history) > 50:
                self.convergence_history.pop(0)
        
        # 更新控制指标
        self._update_control_metrics(final_rewards, local_adjustments)
    
    def _compute_convergence_metric(self, 
                                  prev_rewards: Dict[str, float],
                                  current_rewards: Dict[str, float]) -> float:
        """计算收敛指标"""
        
        total_change = 0.0
        count = 0
        
        for role in current_rewards:
            if role in prev_rewards:
                change = abs(current_rewards[role] - prev_rewards[role])
                total_change += change
                count += 1
        
        return total_change / count if count > 0 else 0.0
    
    def _update_control_metrics(self, 
                              final_rewards: Dict[str, float],
                              local_adjustments: Dict[str, float]):
        """更新控制指标"""
        
        self.control_metrics = {
            'global_reward_signal': self.global_reward_signal,
            'consensus_reached': self.consensus_reached,
            'reward_variance': np.var(list(final_rewards.values())),
            'mean_reward': np.mean(list(final_rewards.values())),
            'convergence_rate': np.mean(self.convergence_history[-10:]) if self.convergence_history else 0.0,
            'control_effectiveness': self._compute_control_effectiveness(final_rewards, local_adjustments)
        }
    
    def _compute_control_effectiveness(self, 
                                     final_rewards: Dict[str, float],
                                     local_adjustments: Dict[str, float]) -> float:
        """计算控制有效性"""
        
        if not self.current_state or not self.target_state:
            return 0.0
        
        # 计算状态改善程度
        current_vec = self.current_state.to_vector()
        target_vec = self.target_state.to_vector()
        
        state_error = np.linalg.norm(target_vec - current_vec)
        max_possible_error = np.linalg.norm(target_vec)
        
        if max_possible_error > 0:
            effectiveness = 1.0 - (state_error / max_possible_error)
        else:
            effectiveness = 1.0
        
        return max(0.0, min(1.0, effectiveness))
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        
        controller_metrics = {}
        for role, controller in self.controllers.items():
            controller_metrics[role] = controller.get_control_metrics()
        
        return {
            'distributed_control_metrics': self.control_metrics,
            'individual_controller_metrics': controller_metrics,
            'role_reward_signals': self.role_reward_signals,
            'global_reward_signal': self.global_reward_signal,
            'convergence_history': self.convergence_history[-10:],  # 最近10个
            'reward_history_length': len(self.reward_history)
        }
    
    def reset_system(self):
        """重置整个控制系统"""
        
        # 重置所有控制器
        for controller in self.controllers.values():
            controller.reset()
        
        # 重置系统状态
        self.global_reward_signal = 0.0
        self.role_reward_signals.clear()
        self.reward_history.clear()
        self.convergence_history.clear()
        self.consensus_reached = False
        self.control_metrics.clear()
        
        logger.info("Reset distributed reward control system")


# 全局实例（可选）
_global_reward_control_system: Optional[DistributedRewardControlSystem] = None

def get_global_reward_control_system() -> DistributedRewardControlSystem:
    """获取全局奖励控制系统实例"""
    global _global_reward_control_system
    if _global_reward_control_system is None:
        config = DistributedRewardControlConfig()
        _global_reward_control_system = DistributedRewardControlSystem(config)
    return _global_reward_control_system