"""
奖励驱动控制系统 - 基于智能体奖励逻辑重构的新控制架构
Reward-Based Control System - Control architecture refactored based on agent reward logic

Author: Asclepion Development Team
Created: 2024-12-19
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from ..core.kallipolis_mathematical_core import SystemState, Agent
from ..agents.role_agents import RoleAgent

logger = logging.getLogger(__name__)


@dataclass
class RewardControlConfig:
    """奖励控制配置"""
    role: str
    reward_scaling_factor: float = 1.0
    reward_history_window: int = 50
    exploration_rate: float = 0.1
    exploitation_threshold: float = 0.8
    reward_normalization: bool = True
    adaptive_learning_rate: bool = True
    baseline_alpha: float = 0.95  # EMA系数


class RewardBasedController(ABC):
    """基于奖励的控制器基类
    
    核心思想：将传统控制信号转换为智能体奖励调节机制
    - 不再直接控制系统状态，而是通过调节agent的奖励来引导行为
    - 奖励调节基于系统偏差和控制目标
    - 实现分布式奖励优化
    """
    
    def __init__(self, config: RewardControlConfig, agent: RoleAgent):
        self.config = config
        self.agent = agent
        self.role = config.role
        
        # 奖励历史和基线
        self.reward_history: List[float] = []
        self.baseline_reward: float = 0.0
        self.reward_variance: float = 0.0
        
        # 控制状态
        self.last_reward_adjustment: float = 0.0
        self.performance_trend: float = 0.0
        
        # 自适应参数
        self.current_learning_rate: float = 0.01
        self.exploration_bonus: float = 0.0
        
        logger.info(f"Initialized reward-based controller for {self.role}")
    
    def compute_reward_adjustment(self, 
                                current_state: SystemState,
                                target_state: SystemState,
                                base_reward: float,
                                global_utility: float,
                                control_context: Dict[str, Any]) -> float:
        """
        计算奖励调节量 - 核心控制逻辑
        
        Args:
            current_state: 当前系统状态
            target_state: 目标状态
            base_reward: 智能体的基础奖励
            global_utility: 全局效用
            control_context: 控制上下文信息
            
        Returns:
            adjusted_reward: 调节后的奖励值
        """
        # 计算状态偏差
        state_error = self._compute_state_error(current_state, target_state)
        
        # 计算基于偏差的奖励调节
        error_adjustment = self._compute_error_based_adjustment(state_error)
        
        # 计算基于性能趋势的调节
        trend_adjustment = self._compute_trend_based_adjustment()
        
        # 计算基于角色特异性的调节
        role_adjustment = self._compute_role_specific_adjustment(
            current_state, target_state, control_context
        )
        
        # 计算探索奖励
        exploration_bonus = self._compute_exploration_bonus(control_context)
        
        # 综合调节
        total_adjustment = (error_adjustment + 
                          trend_adjustment + 
                          role_adjustment + 
                          exploration_bonus)
        
        # 应用缩放因子
        scaled_adjustment = total_adjustment * self.config.reward_scaling_factor
        
        # 计算最终奖励
        adjusted_reward = base_reward + scaled_adjustment
        
        # 更新内部状态
        self._update_internal_state(adjusted_reward, state_error)
        
        # 奖励归一化
        if self.config.reward_normalization:
            adjusted_reward = self._normalize_reward(adjusted_reward)
        
        logger.debug(f"{self.role} reward adjustment: {base_reward:.3f} -> {adjusted_reward:.3f}")
        
        return adjusted_reward
    
    def _compute_state_error(self, current: SystemState, target: SystemState) -> np.ndarray:
        """计算状态偏差向量"""
        current_vec = current.to_vector()
        target_vec = target.to_vector()
        return target_vec - current_vec
    
    @abstractmethod
    def _compute_role_specific_adjustment(self, 
                                        current_state: SystemState,
                                        target_state: SystemState,
                                        context: Dict[str, Any]) -> float:
        """计算基于角色特异性的奖励调节 - 子类实现"""
        pass
    
    def _compute_error_based_adjustment(self, state_error: np.ndarray) -> float:
        """基于状态偏差的奖励调节"""
        # 计算加权误差
        error_magnitude = np.linalg.norm(state_error)
        
        # 误差越大，奖励调节越显著
        if error_magnitude > 0.1:
            # 系统偏离目标较大，增加奖励以激励纠正行为
            adjustment = min(0.5, error_magnitude * 2.0)
        elif error_magnitude < 0.05:
            # 系统接近目标，给予正向奖励
            adjustment = 0.2 * (0.05 - error_magnitude) / 0.05
        else:
            # 中等偏差，小幅调节
            adjustment = -0.1 * error_magnitude
        
        return adjustment
    
    def _compute_trend_based_adjustment(self) -> float:
        """基于性能趋势的奖励调节"""
        if len(self.reward_history) < 3:
            return 0.0
        
        # 计算近期奖励趋势
        recent_rewards = self.reward_history[-3:]
        trend = np.polyfit(range(3), recent_rewards, 1)[0]  # 线性趋势
        
        # 更新性能趋势
        self.performance_trend = 0.7 * self.performance_trend + 0.3 * trend
        
        # 基于趋势给予奖励调节
        if self.performance_trend > 0.02:
            # 性能改善，正向激励
            return 0.1
        elif self.performance_trend < -0.02:
            # 性能下降，增加激励
            return 0.3
        else:
            return 0.0
    
    def _compute_exploration_bonus(self, context: Dict[str, Any]) -> float:
        """计算探索奖励"""
        if 'action_diversity' in context:
            diversity = context['action_diversity']
            if diversity < 0.3:  # 行为过于保守
                self.exploration_bonus = self.config.exploration_rate
            else:
                self.exploration_bonus = 0.0
        
        return self.exploration_bonus
    
    def _update_internal_state(self, reward: float, state_error: np.ndarray):
        """更新控制器内部状态"""
        # 更新奖励历史
        self.reward_history.append(reward)
        if len(self.reward_history) > self.config.reward_history_window:
            self.reward_history.pop(0)
        
        # 更新基线奖励（指数移动平均）
        if len(self.reward_history) > 1:
            self.baseline_reward = (self.config.baseline_alpha * self.baseline_reward + 
                                  (1 - self.config.baseline_alpha) * reward)
        else:
            self.baseline_reward = reward
        
        # 更新奖励方差
        if len(self.reward_history) > 5:
            self.reward_variance = np.var(self.reward_history[-10:])
        
        # 自适应学习率
        if self.config.adaptive_learning_rate:
            self._update_learning_rate(state_error)
        
        self.last_reward_adjustment = reward - self.baseline_reward
    
    def _update_learning_rate(self, state_error: np.ndarray):
        """自适应更新学习率"""
        error_magnitude = np.linalg.norm(state_error)
        
        if error_magnitude > 0.2:
            # 大偏差时提高学习率
            self.current_learning_rate = min(0.05, self.current_learning_rate * 1.1)
        elif error_magnitude < 0.05:
            # 接近目标时降低学习率
            self.current_learning_rate = max(0.001, self.current_learning_rate * 0.95)
    
    def _normalize_reward(self, reward: float) -> float:
        """奖励归一化"""
        if self.reward_variance > 0:
            # Z-score归一化
            normalized = (reward - self.baseline_reward) / np.sqrt(self.reward_variance + 1e-8)
            # 映射到[-1, 1]范围
            return np.tanh(normalized)
        else:
            return np.tanh(reward - self.baseline_reward)
    
    def get_control_metrics(self) -> Dict[str, float]:
        """获取控制指标"""
        return {
            'baseline_reward': self.baseline_reward,
            'reward_variance': self.reward_variance,
            'performance_trend': self.performance_trend,
            'current_learning_rate': self.current_learning_rate,
            'last_adjustment': self.last_reward_adjustment,
            'exploration_bonus': self.exploration_bonus,
            'reward_history_length': len(self.reward_history)
        }
    
    def reset(self):
        """重置控制器状态"""
        self.reward_history.clear()
        self.baseline_reward = 0.0
        self.reward_variance = 0.0
        self.last_reward_adjustment = 0.0
        self.performance_trend = 0.0
        self.current_learning_rate = 0.01
        self.exploration_bonus = 0.0
        
        logger.info(f"Reset reward-based controller for {self.role}")


class RewardControllerFactory:
    """奖励控制器工厂"""
    
    @staticmethod
    def create_controller(role: str, agent: RoleAgent, 
                         config_overrides: Optional[Dict[str, Any]] = None) -> RewardBasedController:
        """创建角色特定的奖励控制器"""
        # 基础配置
        base_config = RewardControlConfig(role=role)
        
        # 应用覆盖配置
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
        
        # 动态导入角色特定控制器类以避免循环导入
        try:
            from .role_specific_reward_controllers import (
                DoctorRewardController, InternRewardController, PatientRewardController,
                AccountantRewardController, GovernmentRewardController
            )
            
            # 根据角色创建特定控制器
            if role == 'doctor':
                return DoctorRewardController(base_config, agent)
            elif role == 'intern':
                return InternRewardController(base_config, agent)
            elif role == 'patient':
                return PatientRewardController(base_config, agent)
            elif role == 'accountant':
                return AccountantRewardController(base_config, agent)
            elif role == 'government':
                return GovernmentRewardController(base_config, agent)
            else:
                raise ValueError(f"Unknown role: {role}")
                
        except ImportError as e:
            logger.warning(f"⚠️ 无法导入角色特定控制器: {e}")
            # 返回基础控制器作为降级
            return BaseRewardController(base_config, agent)


class BaseRewardController(RewardBasedController):
    """基础奖励控制器，用作降级选项"""
    
    def _compute_role_specific_adjustment(self, 
                                        current_state: SystemState,
                                        target_state: SystemState,
                                        context: Dict[str, Any]) -> float:
        """基础奖励调节实现"""
        # 简单的性能差异调节
        if hasattr(current_state, 'to_vector') and hasattr(target_state, 'to_vector'):
            current_vec = current_state.to_vector()
            target_vec = target_state.to_vector()
            error = np.mean(target_vec - current_vec)
            return error * 0.1
        return 0.0