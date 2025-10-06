import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from src.hospital_governance.core.state_space import StateSpace

@dataclass
class PerformanceWeights:
    """性能指标权重配置"""
    medical_quality: float = 0.25
    patient_safety: float = 0.20
    financial_health: float = 0.15
    education_quality: float = 0.10
    system_stability: float = 0.15
    ethics_compliance: float = 0.08
    public_trust: float = 0.07

@dataclass
class PerformanceThresholds:
    """性能阈值配置"""
    excellent: float = 0.9
    good: float = 0.7
    fair: float = 0.5
    poor: float = 0.3

class PerformanceIndex:
    """性能指标计算器"""
    
    def __init__(self, state_space: StateSpace, 
                 weights: PerformanceWeights = None,
                 thresholds: PerformanceThresholds = None):
        self.state_space = state_space
        self.weights = weights or PerformanceWeights()
        self.thresholds = thresholds or PerformanceThresholds()
        
        # 性能历史
        self.performance_history: List[Dict[str, float]] = []
        self.overall_scores: List[float] = []
    
    def compute_performance(self, state: np.ndarray = None) -> Dict[str, float]:
        """计算综合性能指标"""
        if state is None:
            state = self.state_space.current_state
        
        # 提取关键性能指标
        performance_metrics = self._extract_performance_metrics(state)
        
        # 计算加权总分
        overall_score = self._compute_weighted_score(performance_metrics)
        
        # 性能等级
        performance_grade = self._evaluate_performance_grade(overall_score)
        
        result = {
            'overall_score': overall_score,
            'performance_grade': performance_grade,
            'timestamp': len(self.performance_history),
            **performance_metrics
        }
        
        self.performance_history.append(result)
        self.overall_scores.append(overall_score)
        
        return result
    
    def _extract_performance_metrics(self, state: np.ndarray) -> Dict[str, float]:
        """从状态向量提取性能指标"""
        metrics = {}
        
        # 映射状态变量到性能指标
        variable_mapping = {
            'medical_quality': 'medical_quality',
            'patient_safety': 'patient_safety', 
            'financial_health': 'financial_health',
            'education_quality': 'education_quality',
            'system_stability': 'system_stability',
            'ethics_compliance': 'ethics_compliance',
            'public_trust': 'public_trust'
        }
        
        for metric_name, var_name in variable_mapping.items():
            if var_name in self.state_space.variable_names:
                index = self.state_space.variable_names.index(var_name)
                metrics[metric_name] = state[index]
            else:
                metrics[metric_name] = 0.5  # 默认值
        
        return metrics
    
    def _compute_weighted_score(self, metrics: Dict[str, float]) -> float:
        """计算加权性能分数"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric_name, value in metrics.items():
            weight = getattr(self.weights, metric_name, 0.0)
            weighted_sum += weight * value
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_performance_grade(self, score: float) -> str:
        """评估性能等级"""
        if score >= self.thresholds.excellent:
            return 'excellent'
        elif score >= self.thresholds.good:
            return 'good'
        elif score >= self.thresholds.fair:
            return 'fair'
        else:
            return 'poor'
    
    def compute_trend_analysis(self, window_size: int = 50) -> Dict[str, Any]:
        """计算性能趋势分析"""
        if len(self.overall_scores) < 2:
            return {}
        
        recent_scores = self.overall_scores[-window_size:]
        
        # 计算趋势
        x = np.arange(len(recent_scores))
        slope, intercept = np.polyfit(x, recent_scores, 1)
        
        # 计算波动性
        volatility = np.std(recent_scores)
        
        # 预测下一个时间步
        next_score = slope * len(recent_scores) + intercept
        
        return {
            'trend_slope': float(slope),
            'current_score': recent_scores[-1],
            'average_score': float(np.mean(recent_scores)),
            'volatility': float(volatility),
            'predicted_next_score': float(next_score),
            'trend_direction': 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {}
        
        recent_performance = self.performance_history[-100:]  # 最近100个样本
        
        # 计算各指标平均值
        metric_means = {}
        for metric in self.weights.__dataclass_fields__:
            values = [p[metric] for p in recent_performance if metric in p]
            if values:
                metric_means[metric] = np.mean(values)
        
        # 总体统计
        overall_scores = [p['overall_score'] for p in recent_performance]
        
        return {
            'current_performance': self.performance_history[-1],
            'metric_averages': metric_means,
            'overall_score_mean': float(np.mean(overall_scores)),
            'overall_score_std': float(np.std(overall_scores)),
            'performance_trend': self.compute_trend_analysis(),
            'total_evaluations': len(self.performance_history)
        }
    
    def reset(self):
        """重置性能评估器"""
        self.performance_history.clear()
        self.overall_scores.clear()