import numpy as np
from typing import Dict

class GovernmentPolicyController:
    """政府政策控制器"""
    def __init__(self, config: Dict):
        self.P = config.get('policy_matrix', np.eye(2))
        self.policy_limits = config.get('policy_limits', [-1.0, 1.0])
        self.last_state = None
    
    def compute_control(self, y_local: np.ndarray, x_ref: np.ndarray, d_hat: np.ndarray, holy_code_state: Dict, role: str) -> np.ndarray:
        """
        政府控制: u_gov = P(y_local - x_ref) + 政策约束 + 伦理/危机约束
        """
        error = y_local - x_ref
        u_policy = np.dot(self.P, error)
        # 神圣法典约束
        u_constrained = self._apply_holy_code_constraints(u_policy, holy_code_state, role)
        # 限幅
        u_saturated = np.clip(u_constrained, self.policy_limits[0], self.policy_limits[1])
        self.last_state = y_local.copy()
        return u_saturated
    
    def _apply_holy_code_constraints(self, u: np.ndarray, holy_code_state: Dict, role: str) -> np.ndarray:
        u_constrained = u.copy()
        ethical_constraints = holy_code_state.get('ethical_constraints', {})
        # 公平性约束
        if 'min_equity_level' in ethical_constraints:
            min_equity = ethical_constraints['min_equity_level']
            u_constrained[0] = max(u_constrained[0], min_equity)
        return u_constrained
    
    def reset(self):
        self.last_state = None
