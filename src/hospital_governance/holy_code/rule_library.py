import numpy as np
from typing import Dict, List, Any, Callable
from .rule_engine import Rule, RulePriority

class RuleLibrary:
    """规则库 - 预定义规则集合和函数管理"""
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self._condition_functions: Dict[str, Callable] = {}
        self._action_functions: Dict[str, Callable] = {}
        self._initialize_core_rules()
        self._register_functions()
    
    def _register_functions(self):
        """注册条件和动作函数"""
        # 注册条件函数
        self._condition_functions.update({
            "patient_safety_first": self._patient_safety_condition,
            "fair_resource_allocation": self._fair_allocation_condition,
            "medical_education_priority": self._education_priority_condition,
            "financial_sustainability": self._financial_sustainability_condition,
            "crisis_adaptation": self._crisis_adaptation_condition,
            "regulatory_compliance": self._regulatory_compliance_condition,
            "public_interest": self._public_interest_condition,
            "system_stability": self._system_stability_condition
        })
        
        # 注册动作函数
        self._action_functions.update({
            "patient_safety_first": self._patient_safety_action,
            "fair_resource_allocation": self._fair_allocation_action,
            "medical_education_priority": self._education_priority_action,
            "financial_sustainability": self._financial_sustainability_action,
            "crisis_adaptation": self._crisis_adaptation_action,
            "regulatory_compliance": self._regulatory_compliance_action,
            "public_interest": self._public_interest_action,
            "system_stability": self._system_stability_action
        })
    
    def get_condition_function(self, logic_function: str) -> Callable:
        """获取条件函数"""
        return self._condition_functions.get(logic_function, self._default_condition)
    
    def get_action_function(self, logic_function: str) -> Callable:
        """获取动作函数"""
        return self._action_functions.get(logic_function, self._default_action)
    
    def _initialize_core_rules(self):
        """初始化核心规则"""
        # 患者安全规则
        self.add_rule(Rule(
            rule_id="ETHIC_01",
            name="患者安全第一",
            condition=self._patient_safety_condition,
            action=self._patient_safety_action,
            priority=RulePriority.CRITICAL,
            weight=1.0,
            context=["all", "medical", "crisis"],
            description="确保所有决策优先考虑患者安全"
        ))
        
        # 公平资源分配规则
        self.add_rule(Rule(
            rule_id="ETHIC_02",
            name="公平资源分配", 
            condition=self._fair_allocation_condition,
            action=self._fair_allocation_action,
            priority=RulePriority.HIGH,
            weight=0.9,
            context=["resource_allocation", "budget"],
            description="资源分配必须公平合理"
        ))
        
        # 医学教育优先规则
        self.add_rule(Rule(
            rule_id="ETHIC_03",
            name="医学教育优先",
            condition=self._education_priority_condition,
            action=self._education_priority_action,
            priority=RulePriority.MEDIUM,
            weight=0.8,
            context=["education", "training"],
            description="保障医学教育和人才培养"
        ))
        
        # 财务可持续性规则
        self.add_rule(Rule(
            rule_id="ETHIC_04", 
            name="财务可持续性",
            condition=self._financial_sustainability_condition,
            action=self._financial_sustainability_action,
            priority=RulePriority.MEDIUM,
            weight=0.7,
            context=["financial", "budget"],
            description="确保医院财务健康可持续发展"
        ))
        
        # 危机适应能力规则
        self.add_rule(Rule(
            rule_id="ETHIC_05",
            name="危机适应能力", 
            condition=self._crisis_adaptation_condition,
            action=self._crisis_adaptation_action,
            priority=RulePriority.HIGH,
            weight=0.6,
            context=["crisis", "emergency"],
            description="提高系统对危机的适应和响应能力"
        ))
        
        # 政府监管规则
        self.add_rule(Rule(
            rule_id="GOV_01",
            name="法规合规性",
            condition=self._regulatory_compliance_condition,
            action=self._regulatory_compliance_action,
            priority=RulePriority.HIGH,
            weight=0.8,
            context=["regulatory", "compliance"],
            description="确保所有操作符合法规要求"
        ))
        
        self.add_rule(Rule(
            rule_id="GOV_02",
            name="公共利益",
            condition=self._public_interest_condition,
            action=self._public_interest_action,
            priority=RulePriority.CRITICAL,
            weight=0.9,
            context=["policy", "public"],
            description="决策必须服务于公共利益"
        ))
        
        self.add_rule(Rule(
            rule_id="GOV_03", 
            name="系统稳定性",
            condition=self._system_stability_condition,
            action=self._system_stability_action,
            priority=RulePriority.HIGH,
            weight=0.85,
            context=["all", "system"],
            description="维护整个医疗系统的稳定性"
        ))
    
    # 条件函数
    def _patient_safety_condition(self, context: Dict[str, Any]) -> bool:
        state = context.get('state', {})
        return state.get('patient_safety', 0.8) < 0.8 or state.get('medical_quality', 0.8) < 0.7
    
    def _fair_allocation_condition(self, context: Dict[str, Any]) -> bool:
        allocation = context.get('resource_allocation', {})
        if not allocation or len(allocation) < 2:
            return False
        
        values = list(allocation.values())
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        gini = sum((2 * (i + 1) - n - 1) * sorted_vals[i] for i in range(n)) / (n * sum(sorted_vals))
        return gini > 0.3
    
    def _education_priority_condition(self, context: Dict[str, Any]) -> bool:
        state = context.get('state', {})
        budget = context.get('budget_allocation', {})
        return (state.get('education_quality', 0.7) < 0.7 or 
                budget.get('education', 0.0) < 0.1)
    
    def _financial_sustainability_condition(self, context: Dict[str, Any]) -> bool:
        state = context.get('state', {})
        return state.get('financial_health', 0.7) < 0.6
    
    def _crisis_adaptation_condition(self, context: Dict[str, Any]) -> bool:
        crisis = context.get('crisis_context', {})
        return crisis.get('active', False)
    
    def _regulatory_compliance_condition(self, context: Dict[str, Any]) -> bool:
        state = context.get('state', {})
        return state.get('regulatory_compliance', 0.8) < 0.8
    
    def _public_interest_condition(self, context: Dict[str, Any]) -> bool:
        state = context.get('state', {})
        return (state.get('public_trust', 0.7) < 0.7 or 
                state.get('patient_satisfaction', 0.7) < 0.7)
    
    def _system_stability_condition(self, context: Dict[str, Any]) -> bool:
        state = context.get('state', {})
        return state.get('system_stability', 0.8) < 0.7
    
    # 动作函数  
    def _patient_safety_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'safety_enhancement',
            'priority_boost': 0.2,
            'recommendations': [
                '增加患者安全监控',
                '加强医疗质量检查',
                '优化医疗流程'
            ],
            'weight_adjustment': 1.2
        }
    
    def _fair_allocation_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'allocation_adjustment', 
            'priority_boost': 0.15,
            'recommendations': [
                '重新评估资源分配公平性',
                '考虑各部门实际需求',
                '提高分配过程透明度'
            ],
            'weight_adjustment': 1.1
        }
    
    def _education_priority_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'education_support',
            'priority_boost': 0.1,
            'recommendations': [
                '增加医学教育预算',
                '提供更多培训机会',
                '改善教学设施'
            ],
            'weight_adjustment': 1.05
        }
    
    def _financial_sustainability_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'financial_measures',
            'priority_boost': 0.15,
            'recommendations': [
                '实施成本控制措施',
                '优化运营效率', 
                '探索新的收入来源'
            ],
            'weight_adjustment': 1.1
        }
    
    def _crisis_adaptation_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        crisis_type = context.get('crisis_context', {}).get('type', '危机')
        return {
            'type': 'crisis_response',
            'priority_boost': 0.3,
            'recommendations': [
                f'激活{crisis_type}应急预案',
                '重新分配紧急资源',
                '加强内外沟通协调'
            ],
            'weight_adjustment': 1.5
        }
    
    def _regulatory_compliance_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'compliance_enhancement',
            'priority_boost': 0.2,
            'recommendations': [
                '组织法规培训',
                '进行合规性审计',
                '完善文档记录'
            ],
            'weight_adjustment': 1.2
        }
    
    def _public_interest_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'public_engagement', 
            'priority_boost': 0.15,
            'recommendations': [
                '提高决策透明度',
                '加强社区沟通',
                '改进服务质量'
            ],
            'weight_adjustment': 1.1
        }
    
    def _system_stability_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'stabilization_measures',
            'priority_boost': 0.25,
            'recommendations': [
                '进行系统性风险评估',
                '制定应急备份计划',
                '加强系统监控预警'
            ],
            'weight_adjustment': 1.3
        }
    
    def _default_condition(self, context: Dict[str, Any]) -> bool:
        """默认条件"""
        return False
    
    def _default_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """默认动作"""
        return {
            'type': 'general_recommendation',
            'priority_boost': 0.0,
            'recommendations': ['审查当前情况'],
            'weight_adjustment': 1.0
        }
    
    def add_rule(self, rule: Rule):
        """添加规则到库"""
        self.rules[rule.rule_id] = rule
    
    def get_rule(self, rule_id: str) -> Rule:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def get_all_rules(self) -> List[Rule]:
        """获取所有规则"""
        return list(self.rules.values())
    
    def get_rules_by_context(self, context: str) -> List[Rule]:
        """按上下文获取规则"""
        return [rule for rule in self.rules.values() 
                if context in rule.context or 'all' in rule.context]
    
    def get_rules_by_priority(self, priority: RulePriority) -> List[Rule]:
        """按优先级获取规则"""
        return [rule for rule in self.rules.values() if rule.priority == priority]