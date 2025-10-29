# 统一状态空间定义，供参考生成器等模块使用
class StateSpaceDefinition:
    def __init__(self):
        self.dimensions = 16
        self.state_names = [
            'medical_resource_utilization',
            'patient_waiting_time',
            'financial_indicator',
            'ethical_compliance',
            'education_training_quality',
            'intern_satisfaction',
            'professional_development',
            'mentorship_effectiveness',
            'patient_satisfaction',
            'service_accessibility',
            'care_quality_index',
            'safety_incident_rate',
            'operational_efficiency',
            'staff_workload_balance',
            'crisis_response_capability',
            'regulatory_compliance_score'
        ]
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass



# 直接在此定义 SystemState，其他模块统一从此处导入
from dataclasses import dataclass
import numpy as np

@dataclass
class SystemState:
    medical_resource_utilization: float = 0.0
    patient_waiting_time: float = 0.0
    financial_indicator: float = 0.0
    ethical_compliance: float = 0.0
    education_training_quality: float = 0.0
    intern_satisfaction: float = 0.0
    professional_development: float = 0.0
    mentorship_effectiveness: float = 0.0
    patient_satisfaction: float = 0.0
    service_accessibility: float = 0.0
    care_quality_index: float = 0.0
    safety_incident_rate: float = 0.0
    operational_efficiency: float = 0.0
    staff_workload_balance: float = 0.0
    crisis_response_capability: float = 0.0
    regulatory_compliance_score: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.medical_resource_utilization,
            self.patient_waiting_time,
            self.financial_indicator,
            self.ethical_compliance,
            self.education_training_quality,
            self.intern_satisfaction,
            self.professional_development,
            self.mentorship_effectiveness,
            self.patient_satisfaction,
            self.service_accessibility,
            self.care_quality_index,
            self.safety_incident_rate,
            self.operational_efficiency,
            self.staff_workload_balance,
            self.crisis_response_capability,
            self.regulatory_compliance_score
        ])

    @staticmethod
    def from_vector(vec: np.ndarray):
        return SystemState(*vec[:16])


class StateSpace:
    """状态空间管理类，统一使用 SystemState 结构"""
    def __init__(self, initial_state: SystemState = None):
        if initial_state is None:
            self.current_state = SystemState.from_vector(np.zeros(16))
        else:
            self.current_state = initial_state
        self.state_history = [self.current_state]

    def update_state(self, new_state: SystemState):
        """更新系统状态"""
        self.current_state = new_state
        self.state_history.append(new_state)

    def get_state_by_name(self, state_name: str) -> float:
        """通过名称获取状态值（根据 SystemState 属性名）"""
        if hasattr(self.current_state, state_name):
            return getattr(self.current_state, state_name)
        else:
            raise ValueError(f"未知状态名称: {state_name}")

    def get_state_vector(self) -> np.ndarray:
        """获取当前状态向量"""
        return self.current_state.to_vector()

    def get_state_history(self) -> List[SystemState]:
        """获取状态历史"""
        return self.state_history.copy()