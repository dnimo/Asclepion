import numpy as np
from typing import Dict

class ObserverFeedforwardController:
    """观测器前馈控制器 - 实习医生群体"""
    def __init__(self, config: Dict):
        self.L = config['observer_gain']  # 观测器增益矩阵
        self.feedforward_gain = config['feedforward_gain']
        self.control_limits = config.get('control_limits', [-1.0, 1.0])
        self.last_observation = None
    
    def compute_control(self, y_local: np.ndarray, x_ref: np.ndarray, d_hat: np.ndarray, holy_code_state: Dict, role: str) -> np.ndarray:
        """
        实习医生控制: u_intern = L(y_local - x_ref) + u_ff + 伦理约束
        """
        # 误差观测
        obs_error = y_local - x_ref
        u_obs = np.dot(self.L, obs_error)
        # 前馈项
        u_ff = self.feedforward_gain * d_hat[:4]  # 只取前4个扰动分量
        # 神圣法典约束
        u_constrained = self._apply_holy_code_constraints(u_obs + u_ff, holy_code_state, role)
        # 限幅
        u_saturated = np.clip(u_constrained, self.control_limits[0], self.control_limits[1])
        self.last_observation = y_local.copy()
        return u_saturated
    
    def _apply_holy_code_constraints(self, u: np.ndarray, holy_code_state: Dict, role: str) -> np.ndarray:
        u_constrained = u.copy()
        ethical_constraints = holy_code_state.get('ethical_constraints', {})
        # 教育公平约束
        if 'min_training_hours' in ethical_constraints:
            min_training = ethical_constraints['min_training_hours']
            u_constrained[0] = max(u_constrained[0], min_training)
        return u_constrained
    
    def reset(self):
        self.last_observation = None
