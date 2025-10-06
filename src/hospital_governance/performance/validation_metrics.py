import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from ...core.state_space import StateSpace
from ...holy_code.rule_engine import RuleEngine

@dataclass
class ValidationCriteria:
    """验证标准"""
    stability_requirement: float = 0.8
    performance_requirement: float = 0.7
    robustness_requirement: float = 0.6
    safety_requirement: float = 0.9
    ethics_requirement: float = 0.8

class ValidationMetrics:
    """验证指标计算器"""
    
    def __init__(self, state_space: StateSpace, rule_engine: RuleEngine,
                 criteria: ValidationCriteria = None):
        self.state_space = state_space
        self.rule_engine = rule_engine
        self.criteria = criteria or ValidationCriteria()
        
        # 验证结果
        self.validation_results: List[Dict[str, Any]] = []
        self.compliance_scores: Dict[str, List[float]] = {}
    
    def validate_system_performance(self, performance_metrics: Dict[str, float],
                                  stability_metrics: Dict[str, float],
                                  robustness_metrics: Dict[str, float]) -> Dict[str, Any]:
        """验证系统性能"""
        validation_result = {}
        
        # 稳定性验证
        stability_validation = self._validate_stability(stability_metrics)
        validation_result['stability'] = stability_validation
        
        # 性能验证
        performance_validation = self._validate_performance(performance_metrics)
        validation_result['performance'] = performance_validation
        
        # 鲁棒性验证
        robustness_validation = self._validate_robustness(robustness_metrics)
        validation_result['robustness'] = robustness_validation
        
        # 安全性验证
        safety_validation = self._validate_safety()
        validation_result['safety'] = safety_validation
        
        # 伦理合规性验证
        ethics_validation = self._validate_ethics_compliance()
        validation_result['ethics'] = ethics_validation
        
        # 总体验证结果
        overall_validation = self._compute_overall_validation(validation_result)
        validation_result['overall'] = overall_validation
        
        self.validation_results.append(validation_result)
        return validation_result
    
    def _validate_stability(self, stability_metrics: Dict[str, float]) -> Dict[str, Any]:
        """验证稳定性"""
        stability_margin = stability_metrics.get('stability_margin', 0.0)
        is_stable = stability_metrics.get('is_stable', False)
        
        meets_requirement = stability_margin >= self.criteria.stability_requirement
        
        return {
            'score': stability_margin,
            'meets_requirement': meets_requirement,
            'requirement': self.criteria.stability_requirement,
            'is_stable': is_stable,
            'description': f"稳定性裕度: {stability_margin:.3f} (要求: {self.criteria.stability_requirement})"
        }
    
    def _validate_performance(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """验证性能"""
        overall_score = performance_metrics.get('overall_score', 0.0)
        meets_requirement = overall_score >= self.criteria.performance_requirement
        
        return {
            'score': overall_score,
            'meets_requirement': meets_requirement,
            'requirement': self.criteria.performance_requirement,
            'description': f"综合性能: {overall_score:.3f} (要求: {self.criteria.performance_requirement})"
        }
    
    def _validate_robustness(self, robustness_metrics: Dict[str, float]) -> Dict[str, Any]:
        """验证鲁棒性"""
        overall_robustness = robustness_metrics.get('overall_robustness', 0.0)
        meets_requirement = overall_robustness >= self.criteria.robustness_requirement
        
        return {
            'score': overall_robustness,
            'meets_requirement': meets_requirement,
            'requirement': self.criteria.robustness_requirement,
            'description': f"综合鲁棒性: {overall_robustness:.3f} (要求: {self.criteria.robustness_requirement})"
        }
    
    def _validate_safety(self) -> Dict[str, Any]:
        """验证安全性"""
        current_state = self.state_space.current_state
        
        # 检查关键安全指标
        safety_metrics = {
            'patient_safety': self._get_state_variable('patient_safety'),
            'system_stability': self._get_state_variable('system_stability'),
            'resource_adequacy': self._get_state_variable('resource_adequacy')
        }
        
        # 计算安全分数
        safety_score = np.mean(list(safety_metrics.values()))
        meets_requirement = safety_score >= self.criteria.safety_requirement
        
        return {
            'score': safety_score,
            'meets_requirement': meets_requirement,
            'requirement': self.criteria.safety_requirement,
            'metrics': safety_metrics,
            'description': f"安全分数: {safety_score:.3f} (要求: {self.criteria.safety_requirement})"
        }
    
    def _validate_ethics_compliance(self) -> Dict[str, Any]:
        """验证伦理合规性"""
        # 使用规则引擎评估伦理合规性
        evaluation_context = {
            'state': self._get_current_state_dict(),
            'context_type': 'validation',
            'timestamp': len(self.validation_results)
        }
        
        activated_rules = self.rule_engine.evaluate_rules(evaluation_context)
        
        # 计算合规分数
        if activated_rules:
            rule_scores = [rule['weight'] for rule in activated_rules]
            compliance_score = np.mean(rule_scores)
        else:
            compliance_score = 1.0  # 没有违反规则
        
        meets_requirement = compliance_score >= self.criteria.ethics_requirement
        
        return {
            'score': compliance_score,
            'meets_requirement': meets_requirement,
            'requirement': self.criteria.ethics_requirement,
            'activated_rules': len(activated_rules),
            'description': f"伦理合规性: {compliance_score:.3f} (要求: {self.criteria.ethics_requirement})"
        }
    
    def _get_state_variable(self, variable_name: str) -> float:
        """获取状态变量值"""
        if variable_name in self.state_space.variable_names:
            index = self.state_space.variable_names.index(variable_name)
            return self.state_space.current_state[index]
        return 0.5  # 默认值
    
    def _get_current_state_dict(self) -> Dict[str, float]:
        """获取当前状态字典"""
        state_dict = {}
        for i, var_name in enumerate(self.state_space.variable_names):
            state_dict[var_name] = self.state_space.current_state[i]
        return state_dict
    
    def _compute_overall_validation(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体验证结果"""
        category_results = []
        for category, result in validation_result.items():
            if category != 'overall':
                category_results.append(result['meets_requirement'])
        
        overall_passed = all(category_results)
        pass_rate = np.mean(category_results)
        
        return {
            'passed': overall_passed,
            'pass_rate': pass_rate,
            'passed_categories': sum(category_results),
            'total_categories': len(category_results),
            'description': f"总体验证: {'通过' if overall_passed else '未通过'} ({pass_rate:.1%})"
        }
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        if not self.validation_results:
            return {}
        
        recent_validations = self.validation_results[-10:]  # 最近10次验证
        
        # 计算通过率
        pass_rates = {}
        for category in ['stability', 'performance', 'robustness', 'safety', 'ethics']:
            category_results = [v[category]['meets_requirement'] for v in recent_validations]
            pass_rates[category] = np.mean(category_results)
        
        overall_results = [v['overall']['passed'] for v in recent_validations]
        overall_pass_rate = np.mean(overall_results)
        
        # 识别常见问题
        common_issues = self._identify_common_issues(recent_validations)
        
        return {
            'overall_pass_rate': overall_pass_rate,
            'category_pass_rates': pass_rates,
            'total_validations': len(self.validation_results),
            'recent_validations': len(recent_validations),
            'common_issues': common_issues,
            'system_status': 'VALID' if overall_pass_rate >= 0.8 else 'NEEDS_ATTENTION' if overall_pass_rate >= 0.6 else 'CRITICAL'
        }
    
    def _identify_common_issues(self, validations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别常见问题"""
        issue_counts = {}
        
        for validation in validations:
            for category, result in validation.items():
                if category != 'overall' and not result['meets_requirement']:
                    issue_counts[category] = issue_counts.get(category, 0) + 1
        
        common_issues = []
        for category, count in issue_counts.items():
            frequency = count / len(validations)
            if frequency > 0.3:  # 频率超过30%认为是常见问题
                common_issues.append({
                    'category': category,
                    'frequency': frequency,
                    'description': f"{category} 验证失败频率: {frequency:.1%}",
                    'suggestions': self._generate_issue_suggestions(category)
                })
        
        return common_issues
    
    def _generate_issue_suggestions(self, category: str) -> List[str]:
        """生成问题改进建议"""
        suggestions = {
            'stability': [
                "调整控制器增益以提高稳定性",
                "增加系统阻尼",
                "优化参考轨迹"
            ],
            'performance': [
                "优化性能指标权重",
                "改进控制策略",
                "调整系统参数"
            ],
            'robustness': [
                "增强系统鲁棒性设计",
                "增加不确定性建模",
                "改进故障检测机制"
            ],
            'safety': [
                "加强安全监控",
                "完善安全协议",
                "增加冗余备份"
            ],
            'ethics': [
                "重新评估伦理规则",
                "加强伦理合规培训",
                "优化决策流程"
            ]
        }
        
        return suggestions.get(category, ["检查系统配置和参数"])
    
    def reset(self):
        """重置验证器"""
        self.validation_results.clear()
        self.compliance_scores.clear()