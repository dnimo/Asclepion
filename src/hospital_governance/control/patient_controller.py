import numpy as np
from typing import Dict

class PatientAdaptiveController:
    """患者自适应控制器"""
    def __init__(self, config: Dict):
        self.Kp = config.get('Kp', 0.5)
        self.Ka = config.get('Ka', 0.2)  # 适应性增益
        self.control_limits = config.get('control_limits', [-1.0, 1.0])
        self.last_state = None
    
    def compute_control(self, y_local: np.ndarray, x_ref: np.ndarray, d_hat: np.ndarray, holy_code_state: Dict, role: str) -> np.ndarray:
        """
        患者控制: u_patient = Kp(y_local - x_ref) + Ka * adapt(y_local, d_hat) + 伦理/危机约束
        """
        error = y_local - x_ref
        adapt_term = self.Ka * (y_local - d_hat)
        u_patient = self.Kp * error + adapt_term
        # 神圣法典约束
        u_constrained = self._apply_holy_code_constraints(u_patient, holy_code_state, role)
        # 限幅
        u_saturated = np.clip(u_constrained, self.control_limits[0], self.control_limits[1])
        self.last_state = y_local.copy()
        return u_saturated
    
    def _apply_holy_code_constraints(self, u: np.ndarray, holy_code_state: Dict, role: str) -> np.ndarray:
        u_constrained = u.copy()
        ethical_constraints = holy_code_state.get('ethical_constraints', {})
        # 生命权优先约束
        if 'min_health_level' in ethical_constraints:
            min_health = ethical_constraints['min_health_level']
            u_constrained[0] = max(u_constrained[0], min_health)
        return u_constrained
    
    def reset(self):
        self.last_state = None
