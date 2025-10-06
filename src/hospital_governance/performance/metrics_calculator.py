import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from src.hospital_governance.control.distributed_control import DistributedControlSystem
from src.hospital_governance.control.primary_controller import PrimaryStabilizingController
from src.hospital_governance.holy_code.rule_engine import RuleEngine

@dataclass
class ControlMetrics:
    """控制性能指标"""
    settling_time: float
    overshoot: float
    steady_state_error: float
    rise_time: float
    control_effort: float
    robustness: float

@dataclass
class LearningMetrics:
    """学习性能指标"""
    convergence_rate: float
    sample_efficiency: float
    transfer_learning: float
    exploration_efficiency: float

class MetricsCalculator:
    """综合指标计算器"""
    
    def __init__(self, state_space, distributed_controller: DistributedControlSystem,
                 primary_controller: PrimaryStabilizingController, rule_engine: RuleEngine):
        self.state_space = state_space
        self.distributed_controller = distributed_controller
        self.primary_controller = primary_controller
        self.rule_engine = rule_engine
        
        # 指标历史
        self.control_metrics_history: List[ControlMetrics] = []
        self.learning_metrics_history: List[LearningMetrics] = []
    
    def compute_control_metrics(self, reference: np.ndarray, 
                               response: np.ndarray,
                               control_inputs: List[np.ndarray]) -> ControlMetrics:
        """计算控制性能指标"""
        error = reference - response
        
        # 稳态误差（最后10%的数据）
        steady_state_start = int(0.9 * len(error))
        steady_state_error = np.mean(np.abs(error[steady_state_start:]))
        
        # 超调量
        overshoot = self._compute_overshoot(response, reference)
        
        # 稳定时间
        settling_time = self._compute_settling_time(error)
        
        # 上升时间
        rise_time = self._compute_rise_time(response, reference)
        
        # 控制能量
        control_effort = np.sum([np.linalg.norm(u) for u in control_inputs])
        
        # 鲁棒性（基于灵敏度）
        robustness = self._estimate_robustness(response, control_inputs)
        
        metrics = ControlMetrics(
            settling_time=settling_time,
            overshoot=overshoot,
            steady_state_error=steady_state_error,
            rise_time=rise_time,
            control_effort=control_effort,
            robustness=robustness
        )
        
        self.control_metrics_history.append(metrics)
        return metrics
    
    def _compute_overshoot(self, response: np.ndarray, reference: np.ndarray) -> float:
        """计算超调量"""
        if len(response) == 0:
            return 0.0
        
        max_response = np.max(response)
        steady_state = np.mean(response[-len(response)//10:])  # 最后10%
        
        if abs(steady_state) < 1e-6:
            return 0.0
        
        overshoot = (max_response - steady_state) / abs(steady_state)
        return max(0.0, overshoot)
    
    def _compute_settling_time(self, error: np.ndarray, threshold: float = 0.02) -> float:
        """计算稳定时间"""
        if len(error) == 0:
            return 0.0
        
        absolute_error = np.abs(error)
        settling_index = np.where(absolute_error < threshold)[0]
        
        if len(settling_index) == 0:
            return len(error)  # 返回总时间
        
        first_settling = settling_index[0]
        return float(first_settling)
    
    def _compute_rise_time(self, response: np.ndarray, reference: np.ndarray) -> float:
        """计算上升时间"""
        if len(response) == 0:
            return 0.0
        
        target = 0.9 * reference[0] if len(reference) > 0 else 0.9
        rise_indices = np.where(response >= target)[0]
        
        if len(rise_indices) == 0:
            return len(response)  # 返回总时间
        
        first_rise = rise_indices[0]
        return float(first_rise)
    
    def _estimate_robustness(self, response: np.ndarray, 
                           control_inputs: List[np.ndarray]) -> float:
        """估计鲁棒性"""
        if len(response) < 2:
            return 0.5
        
        # 基于响应曲线的平滑度估计鲁棒性
        response_diff = np.diff(response)
        response_variability = np.std(response_diff)
        
        # 基于控制输入的平滑度
        control_variability = np.std([np.linalg.norm(u) for u in control_inputs])
        
        # 综合鲁棒性评分
        robustness = 1.0 / (1.0 + response_variability + control_variability)
        return float(np.clip(robustness, 0.0, 1.0))
    
    def compute_learning_metrics(self, reward_history: List[float],
                               exploration_rates: Dict[str, float]) -> LearningMetrics:
        """计算学习性能指标"""
        if len(reward_history) < 2:
            return LearningMetrics(0.5, 0.5, 0.5, 0.5)
        
        # 收敛速率
        convergence_rate = self._compute_convergence_rate(reward_history)
        
        # 样本效率
        sample_efficiency = self._compute_sample_efficiency(reward_history)
        
        # 迁移学习（简化）
        transfer_learning = self._estimate_transfer_learning()
        
        # 探索效率
        exploration_efficiency = self._compute_exploration_efficiency(exploration_rates)
        
        metrics = LearningMetrics(
            convergence_rate=convergence_rate,
            sample_efficiency=sample_efficiency,
            transfer_learning=transfer_learning,
            exploration_efficiency=exploration_efficiency
        )
        
        self.learning_metrics_history.append(metrics)
        return metrics
    
    def _compute_convergence_rate(self, reward_history: List[float]) -> float:
        """计算收敛速率"""
        if len(reward_history) < 10:
            return 0.5
        
        # 使用最近50个样本计算趋势
        recent_rewards = reward_history[-50:]
        x = np.arange(len(recent_rewards))
        slope, _ = np.polyfit(x, recent_rewards, 1)
        
        # 归一化到[0,1]
        convergence = (slope + 0.1) / 0.2  # 假设最大斜率为0.1
        return float(np.clip(convergence, 0.0, 1.0))
    
    def _compute_sample_efficiency(self, reward_history: List[float]) -> float:
        """计算样本效率"""
        if len(reward_history) < 2:
            return 0.5
        
        # 达到90%最大奖励所需的步数
        max_reward = np.max(reward_history)
        target_reward = 0.9 * max_reward
        
        efficient_steps = np.where(np.array(reward_history) >= target_reward)[0]
        if len(efficient_steps) == 0:
            return 0.0
        
        first_efficient = efficient_steps[0]
        efficiency = 1.0 - (first_efficient / len(reward_history))
        return float(np.clip(efficiency, 0.0, 1.0))
    
    def _estimate_transfer_learning(self) -> float:
        """估计迁移学习能力"""
        # 简化实现 - 基于规则引擎的适应性
        rule_stats = self.rule_engine.get_rule_statistics()
        total_activations = rule_stats.get('total_activations', 0)
        total_rules = rule_stats.get('total_rules', 1)
        
        # 规则使用频率越高，迁移学习能力越强
        transfer_learning = min(1.0, total_activations / (total_rules * 10))
        return transfer_learning
    
    def _compute_exploration_efficiency(self, exploration_rates: Dict[str, float]) -> float:
        """计算探索效率"""
        if not exploration_rates:
            return 0.5
        
        # 理想的探索率在0.1-0.3之间
        optimal_range = (0.1, 0.3)
        efficiencies = []
        
        for rate in exploration_rates.values():
            if rate < optimal_range[0]:
                efficiency = rate / optimal_range[0]
            elif rate > optimal_range[1]:
                efficiency = optimal_range[1] / rate
            else:
                efficiency = 1.0
            efficiencies.append(efficiency)
        
        return float(np.mean(efficiencies))
    
    def compute_system_efficiency(self, control_metrics: ControlMetrics,
                                learning_metrics: LearningMetrics) -> float:
        """计算系统综合效率"""
        # 控制效率分量
        control_efficiency = (1.0 - control_metrics.overshoot) * \
                           (1.0 - control_metrics.steady_state_error) * \
                           control_metrics.robustness
        
        # 学习效率分量
        learning_efficiency = learning_metrics.convergence_rate * \
                            learning_metrics.sample_efficiency * \
                            learning_metrics.exploration_efficiency
        
        # 综合效率
        system_efficiency = (control_efficiency + learning_efficiency) / 2.0
        return float(np.clip(system_efficiency, 0.0, 1.0))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.control_metrics_history or not self.learning_metrics_history:
            return {}
        
        recent_control = self.control_metrics_history[-20:]
        recent_learning = self.learning_metrics_history[-20:]
        
        control_means = {}
        for field in ControlMetrics.__dataclass_fields__:
            values = [getattr(metrics, field) for metrics in recent_control]
            control_means[field] = np.mean(values)
        
        learning_means = {}
        for field in LearningMetrics.__dataclass_fields__:
            values = [getattr(metrics, field) for metrics in recent_learning]
            learning_means[field] = np.mean(values)
        
        # 计算系统效率
        system_efficiency = self.compute_system_efficiency(
            recent_control[-1], recent_learning[-1]
        )
        
        return {
            'control_metrics': control_means,
            'learning_metrics': learning_means,
            'system_efficiency': system_efficiency,
            'total_control_evaluations': len(self.control_metrics_history),
            'total_learning_evaluations': len(self.learning_metrics_history)
        }
    
    def reset(self):
        """重置指标计算器"""
        self.control_metrics_history.clear()
        self.learning_metrics_history.clear()