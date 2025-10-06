import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class StateSpaceDefinition:
    """状态空间定义"""
    dimension: int = 16
    state_names: List[str] = None
    
    def __post_init__(self):
        if self.state_names is None:
            self.state_names = [
                # 物理资源状态 (x₁-x₄)
                'bed_occupancy_rate',           # 病床占用率
                'medical_equipment_utilization', # 医疗设备利用率  
                'staff_utilization_rate',       # 人员利用率
                'medication_inventory_level',   # 药品库存水平
                
                # 财务状态 (x₅-x₈)
                'cash_reserve_ratio',           # 现金储备率
                'operating_margin',             # 运营利润率
                'debt_to_asset_ratio',          # 资产负债率
                'cost_efficiency_index',        # 成本效率指数
                
                # 服务质量状态 (x₉-x₁₂)
                'patient_satisfaction_index',   # 患者满意度指数
                'treatment_success_rate',       # 治疗成功率
                'average_wait_time',            # 平均等待时间
                'medical_safety_index',         # 医疗安全指数
                
                # 教育伦理状态 (x₁₃-x₁₆)
                'ethical_compliance_score',     # 伦理合规得分
                'resource_allocation_fairness', # 资源分配公平性
                'intern_learning_efficiency',   # 实习生学习效率
                'knowledge_transfer_rate'       # 知识传递率
            ]

class StateSpace:
    """状态空间管理类"""
    
    def __init__(self, initial_state: np.ndarray = None):
        self.definition = StateSpaceDefinition()
        
        if initial_state is None:
            self.current_state = np.zeros(self.definition.dimension)
        else:
            self.current_state = initial_state
            
        self.state_history = [self.current_state.copy()]
    
    def update_state(self, new_state: np.ndarray):
        """更新系统状态"""
        self.current_state = new_state
        self.state_history.append(new_state.copy())
    
    def get_state_by_name(self, state_name: str) -> float:
        """通过名称获取状态值"""
        if state_name in self.definition.state_names:
            index = self.definition.state_names.index(state_name)
            return self.current_state[index]
        else:
            raise ValueError(f"未知状态名称: {state_name}")
    
    def get_state_vector(self) -> np.ndarray:
        """获取当前状态向量"""
        return self.current_state.copy()
    
    def get_state_history(self) -> List[np.ndarray]:
        """获取状态历史"""
        return self.state_history.copy()