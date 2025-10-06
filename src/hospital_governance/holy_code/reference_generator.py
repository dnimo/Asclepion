import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import yaml

"""
参考值生成器 - 为医院治理系统生成动态参考值和目标设定

支持多种参考值类型：
- 设定点参考：固定目标值
- 轨迹参考：时间序列目标
- 自适应参考：基于系统状态的动态调整  
- 危机响应参考：紧急情况下的特殊目标
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

class ReferenceType(Enum):
    """参考值类型"""
    SETPOINT = "setpoint"           # 设定点参考
    TRAJECTORY = "trajectory"       # 轨迹参考  
    ADAPTIVE = "adaptive"           # 自适应参考
    CRISIS_RESPONSE = "crisis_response"  # 危机响应参考@dataclass
class ReferenceConfig:
    """参考生成器配置"""
    reference_type: ReferenceType
    setpoints: Dict[str, float]                    # 设定点值
    trajectory_parameters: Dict[str, Any]          # 轨迹参数
    adaptation_rules: List[Dict[str, Any]]         # 自适应规则
    crisis_response_profiles: Dict[str, Any]       # 危机响应配置
    update_frequency: int = 1                      # 更新频率

class ReferenceGenerator:
    """参考信号生成器 - 动态生成理想状态"""
    
    def __init__(self, state_space, config: ReferenceConfig):
        self.state_space = state_space
        self.config = config
        self.current_reference = self._initialize_reference()
        self.reference_history: List[np.ndarray] = [self.current_reference.copy()]
        self.adaptation_step = 0
        self.crisis_mode = False
        self.active_crisis: Optional[str] = None
        
    def _initialize_reference(self) -> np.ndarray:
        """初始化参考信号"""
        reference = np.zeros(self.state_space.dimensions)
        
        # 基于设定点初始化
        for i, var_name in enumerate(self.state_space.variable_names):
            if var_name in self.config.setpoints:
                reference[i] = self.config.setpoints[var_name]
            else:
                reference[i] = 0.7  # 默认设定点
        
        return reference
    
    def generate_reference(self, current_state: np.ndarray, 
                          context: Dict[str, Any] = None) -> np.ndarray:
        """生成参考信号"""
        if context is None:
            context = {}
        
        if self.config.reference_type == ReferenceType.SETPOINT:
            return self._generate_setpoint_reference()
        elif self.config.reference_type == ReferenceType.TRAJECTORY:
            return self._generate_trajectory_reference()
        elif self.config.reference_type == ReferenceType.ADAPTIVE:
            return self._generate_adaptive_reference(current_state, context)
        elif self.config.reference_type == ReferenceType.CRISIS_RESPONSE:
            return self._generate_crisis_reference(current_state, context)
        else:
            return self.current_reference
    
    def _generate_setpoint_reference(self) -> np.ndarray:
        """生成固定设定点参考"""
        return self.current_reference
    
    def _generate_trajectory_reference(self) -> np.ndarray:
        """生成轨迹跟踪参考"""
        step = len(self.reference_history)
        reference = self.current_reference.copy()
        
        # 简单的正弦轨迹示例
        for i, var_name in enumerate(self.state_space.variable_names):
            if var_name in self.config.trajectory_parameters:
                params = self.config.trajectory_parameters[var_name]
                amplitude = params.get('amplitude', 0.1)
                frequency = params.get('frequency', 0.01)
                phase = params.get('phase', 0.0)
                
                base_value = self.config.setpoints.get(var_name, 0.7)
                oscillation = amplitude * np.sin(2 * np.pi * frequency * step + phase)
                reference[i] = base_value + oscillation
        
        return np.clip(reference, 0.1, 1.0)
    
    def _generate_adaptive_reference(self, current_state: np.ndarray, 
                                   context: Dict[str, Any]) -> np.ndarray:
        """生成自适应参考"""
        reference = self.current_reference.copy()
        self.adaptation_step += 1
        
        # 应用自适应规则
        for rule in self.config.adaptation_rules:
            if self._evaluate_rule_condition(rule, current_state, context):
                reference = self._apply_rule_action(rule, reference, current_state)
        
        # 缓慢适应当前状态
        adaptation_rate = 0.01
        reference = (1 - adaptation_rate) * reference + adaptation_rate * current_state
        
        return np.clip(reference, 0.1, 1.0)
    
    def _generate_crisis_reference(self, current_state: np.ndarray,
                                 context: Dict[str, Any]) -> np.ndarray:
        """生成危机响应参考"""
        crisis_type = context.get('crisis_type')
        crisis_severity = context.get('crisis_severity', 0.0)
        
        if crisis_type and crisis_severity > 0.3:
            self.crisis_mode = True
            self.active_crisis = crisis_type
            
            if crisis_type in self.config.crisis_response_profiles:
                profile = self.config.crisis_response_profiles[crisis_type]
                return self._apply_crisis_profile(profile, crisis_severity)
        
        # 无危机或危机结束，恢复正常参考
        if self.crisis_mode and crisis_severity <= 0.1:
            self.crisis_mode = False
            self.active_crisis = None
        
        return self._generate_adaptive_reference(current_state, context)
    
    def _apply_crisis_profile(self, profile: Dict[str, Any], 
                            severity: float) -> np.ndarray:
        """应用危机响应配置"""
        reference = self.current_reference.copy()
        
        for var_name, adjustment in profile.get('adjustments', {}).items():
            if var_name in self.state_space.variable_names:
                index = self.state_space.variable_names.index(var_name)
                base_value = self.config.setpoints.get(var_name, 0.7)
                
                # 根据严重程度调整参考值
                if adjustment.get('type') == 'reduction':
                    reduction = adjustment.get('amount', 0.2) * severity
                    reference[index] = base_value * (1 - reduction)
                elif adjustment.get('type') == 'increase':
                    increase = adjustment.get('amount', 0.1) * severity
                    reference[index] = base_value * (1 + increase)
        
        return np.clip(reference, 0.1, 1.0)
    
    def _evaluate_rule_condition(self, rule: Dict[str, Any], 
                               current_state: np.ndarray,
                               context: Dict[str, Any]) -> bool:
        """评估规则条件"""
        condition_type = rule.get('condition_type', 'state_threshold')
        
        if condition_type == 'state_threshold':
            return self._evaluate_state_threshold(rule, current_state)
        elif condition_type == 'performance_degradation':
            return self._evaluate_performance_degradation(rule, context)
        elif condition_type == 'external_event':
            return self._evaluate_external_event(rule, context)
        else:
            return False
    
    def _evaluate_state_threshold(self, rule: Dict[str, Any], 
                                current_state: np.ndarray) -> bool:
        """评估状态阈值条件"""
        variable = rule.get('variable')
        threshold = rule.get('threshold', 0.5)
        comparison = rule.get('comparison', 'lt')  # less than
        
        if variable in self.state_space.variable_names:
            index = self.state_space.variable_names.index(variable)
            value = current_state[index]
            
            if comparison == 'lt':
                return value < threshold
            elif comparison == 'gt':
                return value > threshold
            elif comparison == 'eq':
                return abs(value - threshold) < 0.05
        
        return False
    
    def _evaluate_performance_degradation(self, rule: Dict[str, Any],
                                        context: Dict[str, Any]) -> bool:
        """评估性能退化条件"""
        performance = context.get('performance_metrics', {})
        metric = rule.get('metric')
        threshold = rule.get('threshold', 0.6)
        
        if metric in performance:
            return performance[metric] < threshold
        
        return False
    
    def _evaluate_external_event(self, rule: Dict[str, Any],
                               context: Dict[str, Any]) -> bool:
        """评估外部事件条件"""
        event_type = rule.get('event_type')
        current_events = context.get('external_events', [])
        
        return event_type in current_events
    
    def _apply_rule_action(self, rule: Dict[str, Any], 
                          reference: np.ndarray,
                          current_state: np.ndarray) -> np.ndarray:
        """应用规则动作"""
        action_type = rule.get('action_type', 'adjust_setpoint')
        updated_reference = reference.copy()
        
        if action_type == 'adjust_setpoint':
            variable = rule.get('variable')
            adjustment = rule.get('adjustment', 0.0)
            
            if variable in self.state_space.variable_names:
                index = self.state_space.variable_names.index(variable)
                updated_reference[index] += adjustment
        
        elif action_type == 'scale_reference':
            scale_factor = rule.get('scale_factor', 1.0)
            updated_reference *= scale_factor
        
        elif action_type == 'blend_with_current':
            blend_ratio = rule.get('blend_ratio', 0.1)
            updated_reference = (1 - blend_ratio) * updated_reference + blend_ratio * current_state
        
        return np.clip(updated_reference, 0.1, 1.0)
    
    def update_reference(self, new_reference: np.ndarray) -> None:
        """更新当前参考信号"""
        self.current_reference = np.clip(new_reference, 0.1, 1.0)
        self.reference_history.append(self.current_reference.copy())
    
    def get_reference_performance(self, actual_states: List[np.ndarray]) -> Dict[str, float]:
        """获取参考跟踪性能"""
        if len(actual_states) != len(self.reference_history):
            return {}
        
        tracking_errors = []
        for i, (actual, reference) in enumerate(zip(actual_states, self.reference_history)):
            error = np.linalg.norm(actual - reference)
            tracking_errors.append(error)
        
        errors_array = np.array(tracking_errors)
        
        return {
            'mean_tracking_error': float(np.mean(errors_array)),
            'max_tracking_error': float(np.max(errors_array)),
            'tracking_rmse': float(np.sqrt(np.mean(errors_array**2))),
            'reference_variability': float(np.std(self.reference_history, axis=0).mean()),
            'adaptation_count': self.adaptation_step
        }
    
    def reset(self) -> None:
        """重置参考生成器"""
        self.current_reference = self._initialize_reference()
        self.reference_history = [self.current_reference.copy()]
        self.adaptation_step = 0
        self.crisis_mode = False
        self.active_crisis = None