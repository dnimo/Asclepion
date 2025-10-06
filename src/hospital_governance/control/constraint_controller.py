import numpy as np
from typing import Dict

class ConstraintEnforcementController:
    """约束强化控制器 - 会计群体"""
    def __init__(self, config: Dict):
        self.C = config['constraint_matrix']  # 约束矩阵
        self.budget_limit = config.get('budget_limit', 1.0)
        self.control_limits = config.get('control_limits', [-1.0, 1.0])
        self.last_state = None
    
    def compute_control(self, y_local: np.ndarray, x_ref: np.ndarray, d_hat: np.ndarray, holy_code_state: Dict, role: str) -> np.ndarray:
        """
        会计控制: u_accountant = C(y_local - x_ref) + 预算约束 + 伦理约束
        """
        error = y_local - x_ref
        u_constraint = np.dot(self.C, error)
        # 预算约束
        u_constraint = np.clip(u_constraint, -self.budget_limit, self.budget_limit)
        # 神圣法典约束
        u_constrained = self._apply_holy_code_constraints(u_constraint, holy_code_state, role)
        # 限幅
        u_saturated = np.clip(u_constrained, self.control_limits[0], self.control_limits[1])
        self.last_state = y_local.copy()
        return u_saturated
    
    def _apply_holy_code_constraints(self, u: np.ndarray, holy_code_state: Dict, role: str) -> np.ndarray:
        u_constrained = u.copy()
        ethical_constraints = holy_code_state.get('ethical_constraints', {})
        # 财务透明度约束
        if 'min_cost_efficiency' in ethical_constraints:
            min_eff = ethical_constraints['min_cost_efficiency']
            u_constrained[1] = max(u_constrained[1], min_eff)
        return u_constrained
    
    def reset(self):
        self.last_state = None
