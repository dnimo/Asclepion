import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import linalg, optimize
from .frequency_analysis import FrequencyAnalyzer

@dataclass
class RobustnessConfig:
    """鲁棒性分析配置"""
    uncertainty_types: List[str]          # 不确定性类型
    uncertainty_bounds: Dict[str, float]  # 不确定性边界
    performance_requirements: Dict[str, float]  # 性能要求
    monte_carlo_samples: int = 1000       # 蒙特卡洛样本数

class RobustnessAnalyzer:
    """鲁棒性分析器"""
    
    def __init__(self, system_dynamics, frequency_analyzer: FrequencyAnalyzer,
                 config: RobustnessConfig = None):
        self.system_dynamics = system_dynamics
        self.frequency_analyzer = frequency_analyzer
        self.config = config or RobustnessConfig(
            uncertainty_types=['parametric', 'dynamic', 'structural'],
            uncertainty_bounds={'parametric': 0.1, 'dynamic': 0.2, 'structural': 0.05},
            performance_requirements={'stability_margin': 0.1, 'performance_degradation': 0.2}
        )
        
        # 分析结果
        self.robustness_metrics: Dict[str, float] = {}
        self.worst_case_scenarios: List[Dict[str, Any]] = []
        self.monte_carlo_results: List[Dict[str, Any]] = []
    
    def analyze_parametric_robustness(self, parameter_variations: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """分析参数鲁棒性"""
        robustness_scores = {}
        worst_case_performance = {}
        
        for param_name, (min_val, max_val) in parameter_variations.items():
            # 测试参数变化对性能的影响
            performance_variation = self._test_parameter_variation(param_name, min_val, max_val)
            
            # 计算鲁棒性评分
            robustness = self._compute_parametric_robustness(performance_variation)
            robustness_scores[param_name] = robustness
            
            # 记录最坏情况
            worst_idx = np.argmin([p['performance'] for p in performance_variation])
            worst_case_performance[param_name] = performance_variation[worst_idx]
        
        return {
            'robustness_scores': robustness_scores,
            'worst_case_performance': worst_case_performance,
            'most_sensitive_parameter': min(robustness_scores, key=robustness_scores.get),
            'least_sensitive_parameter': max(robustness_scores, key=robustness_scores.get)
        }
    
    def _test_parameter_variation(self, param_name: str, min_val: float, max_val: float, 
                                 num_points: int = 10) -> List[Dict[str, Any]]:
        """测试参数变化"""
        results = []
        
        for param_val in np.linspace(min_val, max_val, num_points):
            # 应用参数变化（简化实现）
            # 在实际系统中，这里需要修改系统参数并重新计算性能
            
            # 模拟性能计算
            performance = self._compute_system_performance({param_name: param_val})
            
            results.append({
                'parameter': param_name,
                'value': param_val,
                'performance': performance,
                'stability_margin': performance.get('stability_margin', 0.5)
            })
        
        return results
    
    def _compute_parametric_robustness(self, performance_variation: List[Dict[str, Any]]) -> float:
        """计算参数鲁棒性"""
        performances = [p['performance']['overall'] for p in performance_variation]
        stability_margins = [p['stability_margin'] for p in performance_variation]
        
        # 性能变化程度
        performance_variability = np.std(performances) / (np.mean(performances) + 1e-6)
        
        # 稳定性保证
        stability_robustness = min(stability_margins) / self.config.performance_requirements['stability_margin']
        
        # 综合鲁棒性评分
        robustness = 1.0 / (1.0 + performance_variability) * min(1.0, stability_robustness)
        return float(robustness)
    
    def analyze_structural_robustness(self, component_failures: List[str]) -> Dict[str, Any]:
        """分析结构鲁棒性"""
        robustness_scores = {}
        failure_impacts = {}
        
        for component in component_failures:
            # 模拟组件失效
            impact = self._simulate_component_failure(component)
            
            # 计算鲁棒性评分
            robustness = self._compute_structural_robustness(impact)
            robustness_scores[component] = robustness
            failure_impacts[component] = impact
        
        return {
            'robustness_scores': robustness_scores,
            'failure_impacts': failure_impacts,
            'critical_components': [c for c, s in robustness_scores.items() if s < 0.5],
            'system_redundancy': self._assess_system_redundancy(robustness_scores)
        }
    
    def _simulate_component_failure(self, component: str) -> Dict[str, float]:
        """模拟组件失效"""
        # 简化实现 - 在实际系统中需要具体建模组件失效的影响
        base_performance = self._compute_system_performance({})
        
        # 模拟性能下降
        performance_degradation = np.random.uniform(0.1, 0.8)
        degraded_performance = {k: v * (1 - performance_degradation) 
                              for k, v in base_performance.items()}
        
        return {
            'component': component,
            'performance_degradation': performance_degradation,
            'degraded_performance': degraded_performance,
            'recovery_time': np.random.exponential(10.0),  # 恢复时间
            'cascade_effect': np.random.uniform(0.0, 0.3)  # 级联效应
        }
    
    def _compute_structural_robustness(self, impact: Dict[str, float]) -> float:
        """计算结构鲁棒性"""
        performance_degradation = impact['performance_degradation']
        recovery_time = impact['recovery_time']
        cascade_effect = impact['cascade_effect']
        
        # 鲁棒性评分基于性能下降、恢复时间和级联效应
        robustness = (1.0 - performance_degradation) * \
                    (1.0 / (1.0 + recovery_time / 10.0)) * \
                    (1.0 - cascade_effect)
        
        return float(max(0.0, min(1.0, robustness)))
    
    def _assess_system_redundancy(self, robustness_scores: Dict[str, float]) -> float:
        """评估系统冗余度"""
        # 基于关键组件的鲁棒性评估冗余度
        critical_components = [s for s in robustness_scores.values() if s < 0.7]
        
        if not critical_components:
            return 1.0
        
        redundancy = 1.0 - (len(critical_components) / len(robustness_scores))
        return float(max(0.0, min(1.0, redundancy)))
    
    def monte_carlo_robustness_analysis(self, num_samples: int = None) -> Dict[str, Any]:
        """蒙特卡洛鲁棒性分析"""
        if num_samples is None:
            num_samples = self.config.monte_carlo_samples
        
        results = []
        robustness_scores = []
        
        for i in range(num_samples):
            # 生成随机不确定性
            uncertain_system = self._apply_random_uncertainty()
            
            # 计算系统性能
            performance = self._compute_system_performance(uncertain_system)
            
            # 计算鲁棒性评分
            robustness = self._compute_overall_robustness(performance)
            
            result = {
                'sample': i,
                'performance': performance,
                'robustness': robustness,
                'uncertainty_level': uncertain_system.get('uncertainty_level', 0.0)
            }
            results.append(result)
            robustness_scores.append(robustness)
        
        self.monte_carlo_results = results
        
        return {
            'monte_carlo_samples': num_samples,
            'average_robustness': float(np.mean(robustness_scores)),
            'robustness_std': float(np.std(robustness_scores)),
            'min_robustness': float(np.min(robustness_scores)),
            'max_robustness': float(np.max(robustness_scores)),
            'robustness_percentile_95': float(np.percentile(robustness_scores, 95)),
            'robustness_percentile_5': float(np.percentile(robustness_scores, 5)),
            'success_rate': float(np.mean([r > 0.5 for r in robustness_scores]))
        }
    
    def _apply_random_uncertainty(self) -> Dict[str, Any]:
        """应用随机不确定性"""
        uncertainty_types = self.config.uncertainty_types
        selected_type = np.random.choice(uncertainty_types)
        uncertainty_level = np.random.uniform(0.0, self.config.uncertainty_bounds.get(selected_type, 0.1))
        
        return {
            'uncertainty_type': selected_type,
            'uncertainty_level': uncertainty_level,
            'affected_parameters': self._generate_uncertain_parameters(selected_type, uncertainty_level)
        }
    
    def _generate_uncertain_parameters(self, uncertainty_type: str, level: float) -> Dict[str, float]:
        """生成不确定参数"""
        # 简化实现
        n_params = self.system_dynamics.state_space.dimensions
        perturbations = np.random.uniform(-level, level, n_params)
        
        return {f'param_{i}': pert for i, pert in enumerate(perturbations)}
    
    def _compute_system_performance(self, uncertain_system: Dict[str, Any]) -> Dict[str, float]:
        """计算系统性能（简化实现）"""
        # 在实际系统中，这里需要基于不确定系统计算真实性能
        base_performance = {
            'stability_margin': 0.8,
            'tracking_error': 0.1,
            'response_time': 5.0,
            'overshoot': 0.05,
            'settling_time': 10.0,
            'control_effort': 0.3
        }
        
        # 应用不确定性影响
        uncertainty_level = uncertain_system.get('uncertainty_level', 0.0)
        performance_degradation = 1.0 - uncertainty_level * 0.5  # 线性退化
        
        degraded_performance = {k: v * performance_degradation for k, v in base_performance.items()}
        degraded_performance['overall'] = np.mean(list(degraded_performance.values()))
        
        return degraded_performance
    
    def _compute_overall_robustness(self, performance: Dict[str, float]) -> float:
        """计算总体鲁棒性"""
        stability_margin = performance.get('stability_margin', 0.0)
        tracking_error = performance.get('tracking_error', 1.0)
        overall_performance = performance.get('overall', 0.5)
        
        # 稳定性鲁棒性
        stability_robustness = min(1.0, stability_margin / self.config.performance_requirements['stability_margin'])
        
        # 性能鲁棒性
        performance_robustness = 1.0 / (1.0 + tracking_error)
        
        # 综合鲁棒性
        overall_robustness = (stability_robustness + performance_robustness + overall_performance) / 3.0
        
        return float(max(0.0, min(1.0, overall_robustness)))
    
    def find_worst_case_scenario(self, num_candidates: int = 10) -> List[Dict[str, Any]]:
        """寻找最坏情况场景"""
        if not self.monte_carlo_results:
            self.monte_carlo_robustness_analysis()
        
        # 按鲁棒性排序
        sorted_results = sorted(self.monte_carlo_results, key=lambda x: x['robustness'])
        worst_cases = sorted_results[:num_candidates]
        
        self.worst_case_scenarios = worst_cases
        
        return worst_cases
    
    def compute_robustness_metrics(self) -> Dict[str, float]:
        """计算鲁棒性指标"""
        if not self.monte_carlo_results:
            self.monte_carlo_robustness_analysis()
        
        mc_results = self.monte_carlo_robustness_analysis()
        
        # 频域鲁棒性
        freq_robustness = self.frequency_analyzer.compute_robustness_metrics()
        
        # 参数鲁棒性（简化）
        param_variations = {'gain': (0.5, 1.5), 'time_constant': (0.8, 1.2)}
        param_robustness = self.analyze_parametric_robustness(param_variations)
        
        # 结构鲁棒性
        components = ['sensor_1', 'actuator_1', 'controller_1']
        struct_robustness = self.analyze_structural_robustness(components)
        
        self.robustness_metrics = {
            'monte_carlo_robustness': mc_results['average_robustness'],
            'frequency_domain_robustness': freq_robustness['stability_robustness'],
            'parametric_robustness': np.mean(list(param_robustness['robustness_scores'].values())),
            'structural_robustness': struct_robustness['system_redundancy'],
            'overall_robustness': np.mean([
                mc_results['average_robustness'],
                freq_robustness['stability_robustness'],
                np.mean(list(param_robustness['robustness_scores'].values())),
                struct_robustness['system_redundancy']
            ])
        }
        
        return self.robustness_metrics
    
    def get_robustness_summary(self) -> Dict[str, Any]:
        """获取鲁棒性摘要"""
        robustness_metrics = self.compute_robustness_metrics()
        worst_cases = self.find_worst_case_scenario(5)
        
        return {
            'robustness_metrics': robustness_metrics,
            'worst_case_scenarios': worst_cases,
            'uncertainty_bounds': self.config.uncertainty_bounds,
            'performance_requirements': self.config.performance_requirements,
            'monte_carlo_samples': len(self.monte_carlo_results),
            'critical_failure_modes': self._identify_critical_failure_modes()
        }
    
    def _identify_critical_failure_modes(self) -> List[Dict[str, Any]]:
        """识别关键故障模式"""
        critical_modes = []
        
        for scenario in self.worst_case_scenarios[:3]:
            critical_modes.append({
                'scenario_id': scenario['sample'],
                'robustness': scenario['robustness'],
                'uncertainty_type': scenario.get('uncertainty_type', 'unknown'),
                'performance_impact': 1.0 - scenario['performance']['overall'],
                'recommendations': self._generate_robustness_recommendations(scenario)
            })
        
        return critical_modes
    
    def _generate_robustness_recommendations(self, scenario: Dict[str, Any]) -> List[str]:
        """生成鲁棒性改进建议"""
        recommendations = []
        robustness = scenario['robustness']
        
        if robustness < 0.3:
            recommendations.extend([
                "重新设计控制器以提高鲁棒性",
                "增加系统冗余度",
                "实施更严格的监控和诊断"
            ])
        elif robustness < 0.6:
            recommendations.extend([
                "优化控制器参数",
                "改进不确定性建模",
                "增强故障检测能力"
            ])
        else:
            recommendations.append("当前鲁棒性水平可接受，建议持续监控")
        
        return recommendations