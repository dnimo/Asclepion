import numpy as np
from typing import Dict, List
from .primary_controller import PrimaryStabilizingController
from .observer_controller import ObserverFeedforwardController
from .constraint_controller import ConstraintEnforcementController

class DistributedControlSystem:
    """分布式控制系统"""
    
    def __init__(self, controller_configs: Dict):
        self.controllers = self._initialize_controllers(controller_configs)
        self.control_history = []
        
    def _initialize_controllers(self, configs: Dict) -> Dict:
        """初始化各角色控制器"""
        from .patient_controller import PatientAdaptiveController
        from .government_controller import GovernmentPolicyController
        return {
            'doctors': PrimaryStabilizingController(configs['doctors']),
            'interns': ObserverFeedforwardController(configs['interns']),
            'patients': PatientAdaptiveController(configs['patients']),
            'accountants': ConstraintEnforcementController(configs['accountants']),
            'government': GovernmentPolicyController(configs['government'])
        }
    
    def compute_control(self, x_t: np.ndarray, x_ref: np.ndarray, 
                       d_hat: np.ndarray, holy_code_state: Dict) -> np.ndarray:
        """计算分布式控制信号"""
        control_signals = {}
        
        for role, controller in self.controllers.items():
            y_local = self._get_local_observation(x_t, role)
            u_role = controller.compute_control(y_local, x_ref, d_hat, holy_code_state, role)
            control_signals[role] = u_role
        
        # 合成全局控制向量
        u_global = self._synthesize_global_control(control_signals)
        
        # 记录控制历史
        self.control_history.append({
            'time_step': len(self.control_history),
            'control_signals': control_signals,
            'global_control': u_global
        })
        
        return u_global
    
    def _get_local_observation(self, x_t: np.ndarray, role: str) -> np.ndarray:
        """获取局部观测"""
        observation_masks = {
            'doctors': np.ones(16, dtype=bool),  # 医生可观测全部状态
            'interns': np.array([1]*8 + [0]*4 + [1]*4, dtype=bool),  # 实习生关注资源和教育
            'patients': np.array([0]*4 + [1]*8 + [0]*4, dtype=bool),  # 患者关注财务和质量
            'accountants': np.array([0]*8 + [1]*4 + [0]*4, dtype=bool),  # 会计关注财务
            'government': np.array([1]*4 + [0]*8 + [1]*4, dtype=bool)  # 政府关注资源和伦理
        }
        return x_t[observation_masks[role]]
    
    def _synthesize_global_control(self, control_signals: Dict) -> np.ndarray:
        """合成全局控制向量"""
        u_global = np.zeros(17)
        
        # 医生控制 (u₁-u₄)
        if 'doctors' in control_signals:
            u_global[0:4] = control_signals['doctors'][:4]
            
        # 实习生控制 (u₅-u₈)
        if 'interns' in control_signals:
            u_global[4:8] = control_signals['interns'][:4]
            
        # 患者控制 (u₉-u₁₁)
        if 'patients' in control_signals:
            u_global[8:11] = control_signals['patients'][:3]
            
        # 会计控制 (u₁₂-u₁₄)
        if 'accountants' in control_signals:
            u_global[11:14] = control_signals['accountants'][:3]
            
        # 政府控制 (u₁₅-u₁₇)
        if 'government' in control_signals:
            u_global[14:17] = control_signals['government'][:3]
            
        return u_global
    
    def get_control_history(self) -> List[Dict]:
        """获取控制历史"""
        return self.control_history.copy()