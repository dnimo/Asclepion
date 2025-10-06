import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import yaml

class RulePriority(Enum):
    """规则优先级"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class Rule:
    """规则定义"""
    rule_id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Any]
    priority: RulePriority
    weight: float
    context: List[str]
    description: str

class RuleEngine:
    """规则引擎 - 执行神圣法典规则"""
    
    def __init__(self, rules_file: Optional[str] = None):
        self.rules: Dict[str, Rule] = {}
        self.rule_history: List[Dict[str, Any]] = []
        self.activation_counts: Dict[str, int] = {}
        
        if rules_file:
            self.load_rules_from_file(rules_file)
    
    def load_rules_from_file(self, filepath: str) -> None:
        """从文件加载规则"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                rules_config = yaml.safe_load(f)
            
            self._create_rules_from_config(rules_config)
            print(f"从 {filepath} 加载了 {len(self.rules)} 条规则")
        except Exception as e:
            print(f"加载规则文件错误: {e}")
    
    def _create_rules_from_config(self, config: Dict[str, Any]) -> None:
        """从配置创建规则"""
        # 核心伦理规则
        core_rules = config.get('core_rules', [])
        for rule_config in core_rules:
            rule = self._create_rule_from_config(rule_config)
            if rule:
                self.add_rule(rule)
        
        # 政府规则
        gov_rules = config.get('government_rules', [])
        for rule_config in gov_rules:
            rule = self._create_rule_from_config(rule_config)
            if rule:
                self.add_rule(rule)
    
    def _create_rule_from_config(self, config: Dict[str, Any]) -> Optional[Rule]:
        """从单个配置创建规则"""
        try:
            rule_id = config['rule_id']
            name = config['name']
            logic_function = config['logic_function']
            weight = config.get('weight', 1.0)
            context = config.get('context', ['all'])
            priority_value = config.get('priority', 3)
            description = config.get('description', '')
            
            priority = RulePriority(priority_value)
            
            # 创建条件和动作函数
            condition = self._create_condition_function(logic_function)
            action = self._create_action_function(logic_function)
            
            return Rule(
                rule_id=rule_id,
                name=name,
                condition=condition,
                action=action,
                priority=priority,
                weight=weight,
                context=context if isinstance(context, list) else [context],
                description=description
            )
        except Exception as e:
            print(f"创建规则错误 {config.get('rule_id', 'unknown')}: {e}")
            return None
    
    def _create_condition_function(self, logic_function: str) -> Callable:
        """创建条件评估函数 - 委托给规则库"""
        from .rule_library import RuleLibrary
        
        if not hasattr(self, '_rule_library'):
            self._rule_library = RuleLibrary()
        
        return self._rule_library.get_condition_function(logic_function)
    
    def _create_action_function(self, logic_function: str) -> Callable:
        """创建规则动作函数 - 委托给规则库"""
        from .rule_library import RuleLibrary
        
        if not hasattr(self, '_rule_library'):
            self._rule_library = RuleLibrary()
        
        return self._rule_library.get_action_function(logic_function)
    
    def add_rule(self, rule: Rule) -> None:
        """添加规则，并保存到yaml文件"""
        self.rules[rule.rule_id] = rule
        self.activation_counts[rule.rule_id] = 0
        self.save_rules_to_file()
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除规则，并保存到yaml文件"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            if rule_id in self.activation_counts:
                del self.activation_counts[rule_id]
            self.save_rules_to_file()
            return True
        return False
    
    def evaluate_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """评估所有规则"""
        current_context = context.get('context_type', 'all')
        activated_rules = []
        
        for rule_id, rule in self.rules.items():
            if current_context in rule.context or 'all' in rule.context:
                if rule.condition(context):
                    # 执行规则动作
                    action_result = rule.action(context)
                    
                    # 记录规则激活
                    rule_record = {
                        'rule_id': rule_id,
                        'name': rule.name,
                        'priority': rule.priority.value,
                        'weight': rule.weight,
                        'action_result': action_result,
                        'timestamp': context.get('timestamp'),
                        'context': current_context
                    }
                    
                    activated_rules.append(rule_record)
                    self.rule_history.append(rule_record)
                    self.activation_counts[rule_id] = self.activation_counts.get(rule_id, 0) + 1
        
        # 按优先级和权重排序
        activated_rules.sort(key=lambda x: (x['priority'], x['weight']), reverse=True)
        
        return activated_rules
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计"""
        total_rules = len(self.rules)
        total_activations = sum(self.activation_counts.values())
        
        # 按优先级统计
        priority_counts = {}
        for rule in self.rules.values():
            priority = rule.priority.name
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # 最活跃的规则
        most_active = max(self.activation_counts.items(), key=lambda x: x[1]) if self.activation_counts else ('none', 0)
        
        return {
            'total_rules': total_rules,
            'total_activations': total_activations,
            'priority_distribution': priority_counts,
            'most_active_rule': most_active[0],
            'most_active_count': most_active[1],
            'average_activations_per_rule': total_activations / total_rules if total_rules > 0 else 0
        }
    
    def update_rule_weight(self, rule_id: str, new_weight: float) -> bool:
        """更新规则权重，并保存到yaml文件"""
        if rule_id in self.rules:
            self.rules[rule_id].weight = new_weight
            self.save_rules_to_file()
            return True
        return False
    def save_rules_to_file(self, filepath: str = 'config/holy_code_rules.yaml') -> None:
        """将所有规则保存到yaml文件"""
        rules_config = {'core_rules': [], 'government_rules': []}
        for rule in self.rules.values():
            rule_dict = {
                'rule_id': rule.rule_id,
                'name': rule.name,
                'logic_function': getattr(rule.condition, '__name__', ''),
                'weight': rule.weight,
                'context': rule.context,
                'priority': rule.priority.value,
                'description': rule.description
            }
            # 简单分类，实际可扩展
            if rule.priority in [RulePriority.CRITICAL, RulePriority.HIGH, RulePriority.MEDIUM]:
                rules_config['core_rules'].append(rule_dict)
            else:
                rules_config['government_rules'].append(rule_dict)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.safe_dump(rules_config, f, allow_unicode=True)
        except Exception as e:
            print(f"保存规则到文件失败: {e}")
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.activation_counts = {rule_id: 0 for rule_id in self.rules.keys()}
        self.rule_history.clear()