import numpy as np
from typing import Dict

class PrimaryStabilizingController:
    """主稳定控制器 - 医生群体"""
    
    def __init__(self, config: Dict):
        self.K = config['feedback_gain']  # 反馈增益矩阵
        self.integrator_gain = config['integrator_gain']  # 积分增益
        self.control_limits = config.get('control_limits', [-1.0, 1.0])
        
        # 积分误差初始化
        self.integral_error = None
        
    def compute_control(self, y_local: np.ndarray, x_ref: np.ndarray, 
                       d_hat: np.ndarray, holy_code_state: Dict, role: str) -> np.ndarray:
        """
        主控制器: u_doctor = -K e + u_integral + u_ff
        """
        
        # 误差计算
        e = y_local - x_ref
        
        # 初始化积分误差
        if self.integral_error is None:
            self.integral_error = np.zeros_like(e)
        
        # 比例反馈
        u_feedback = -np.dot(self.K, e)
        
        # 积分项（消除稳态误差）
        self.integral_error += e
        u_integral = -self.integrator_gain * self.integral_error
        
        # 前馈补偿（基于扰动预测）
        u_feedforward = self._compute_feedforward(d_hat)
        
        # 神圣法典约束
        u_constrained = self._apply_holy_code_constraints(
            u_feedback + u_integral + u_feedforward, 
            holy_code_state, role
        )
        
        # 输出限幅
        u_saturated = np.clip(u_constrained, 
                             self.control_limits[0], 
                             self.control_limits[1])
        
        return u_saturated
    
    def _compute_feedforward(self, d_hat: np.ndarray) -> np.ndarray:
        """计算前馈补偿"""
        # 简化的前馈补偿，基于扰动预测
        # 在实际系统中，这需要更复杂的设计
        u_ff = np.zeros(4)  # 医生有4个控制输入
        
        # 基于疫情扰动的补偿
        if d_hat[0] > 0.5:  # 严重疫情
            u_ff[0] = 0.3   # 增加资源分配
            u_ff[3] = 0.2   # 加强质量控制
            
        # 基于需求冲击的补偿
        if d_hat[3] > 0.6:  # 高需求冲击
            u_ff[2] = -0.1  # 调整工作负荷
            
        return u_ff
    
    def _apply_holy_code_constraints(self, u: np.ndarray, 
                                   holy_code_state: Dict, role: str) -> np.ndarray:
        """应用神圣法典约束"""
        u_constrained = u.copy()
        
        # 检查伦理约束
        ethical_constraints = holy_code_state.get('ethical_constraints', {})
        
        # 最小质量控制约束
        if 'min_quality_control' in ethical_constraints:
            min_quality = ethical_constraints['min_quality_control']
            u_constrained[3] = max(u_constrained[3], min_quality)
            
        # 最大工作负荷约束
        if 'max_workload' in ethical_constraints:
            max_workload = ethical_constraints['max_workload']
            u_constrained[2] = min(u_constrained[2], max_workload)
            
        return u_constrained
    
    def reset_integrator(self):
        """重置积分器"""
        self.integral_error = None