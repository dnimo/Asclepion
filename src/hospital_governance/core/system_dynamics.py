import numpy as np
from typing import Dict, Any, Tuple
from .state_space import StateSpace

class SystemDynamics:
    """系统动力学模型"""
    
    def __init__(self, system_matrices: Dict[str, np.ndarray]):
        # 验证系统矩阵维度
        self._validate_matrices(system_matrices)
        
        # 系统矩阵
        self.A = system_matrices['A']  # 状态转移矩阵 16x16
        self.B = system_matrices['B']  # 控制输入矩阵 16x17
        self.E = system_matrices['E']  # 扰动输入矩阵 16x6
        self.C = system_matrices['C']  # 输出矩阵 16x16
        self.D = system_matrices['D']  # 直通矩阵 16x17
        
        # 非线性参数
        self.saturation_thresholds = {
            'bed_occupancy': 0.95,
            'staff_utilization': 0.85,
            'equipment_utilization': 0.9
        }
    
    def _validate_matrices(self, matrices: Dict[str, np.ndarray]):
        """验证系统矩阵维度"""
        assert matrices['A'].shape == (16, 16), "A矩阵维度错误"
        assert matrices['B'].shape == (16, 17), "B矩阵维度错误" 
        assert matrices['E'].shape == (16, 6), "E矩阵维度错误"
        assert matrices['C'].shape == (16, 16), "C矩阵维度错误"
        assert matrices['D'].shape == (16, 17), "D矩阵维度错误"
    
    def state_transition(self, x_t: np.ndarray, u_t: np.ndarray, 
                        d_t: np.ndarray) -> np.ndarray:
        """
        状态转移方程: x(t+1) = A x(t) + B u(t) + E d(t) + f(x,u,d)
        """
        # 线性部分
        linear_dynamics = (
            np.dot(self.A, x_t) + 
            np.dot(self.B, u_t) + 
            np.dot(self.E, d_t)
        )
        
        # 非线性部分
        nonlinear_effects = self._compute_nonlinear_effects(x_t, u_t, d_t)
        
        # 合成下一状态
        next_state = linear_dynamics + nonlinear_effects
        
        # 应用系统约束
        constrained_state = self._apply_system_constraints(next_state)
        
        return constrained_state
    
    def _compute_nonlinear_effects(self, x_t: np.ndarray, u_t: np.ndarray, 
                                 d_t: np.ndarray) -> np.ndarray:
        """计算非线性效应"""
        nonlinear = np.zeros_like(x_t)
        
        # 1. 饱和非线性
        nonlinear += self._saturation_nonlinearity(x_t)
        
        # 2. 博弈交互非线性
        nonlinear += self._game_theoretic_nonlinearity(x_t, u_t)
        
        # 3. 约束非线性
        nonlinear += self._constraint_nonlinearity(x_t, u_t)
        
        return nonlinear
    
    def _saturation_nonlinearity(self, x_t: np.ndarray) -> np.ndarray:
        """饱和非线性"""
        effects = np.zeros_like(x_t)
        
        # 病床占用率饱和
        if x_t[0] > self.saturation_thresholds['bed_occupancy']:
            effects[0] = -0.1 * (x_t[0] - self.saturation_thresholds['bed_occupancy'])
            
        # 人员利用率饱和
        if x_t[2] > self.saturation_thresholds['staff_utilization']:
            effects[2] = -0.15 * (x_t[2] - self.saturation_thresholds['staff_utilization'])
            effects[9] = -0.05  # 影响患者满意度
            
        return effects
    
    def _game_theoretic_nonlinearity(self, x_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        """博弈论非线性"""
        effects = np.zeros_like(x_t)
        
        # 提取角色控制输入
        doctor_control = u_t[0:4]      # u₁-u₄
        intern_control = u_t[4:8]      # u₅-u₈
        accountant_control = u_t[11:14] # u₁₂-u₁₄
        
        # 医生-实习生资源博弈
        resource_tension = np.linalg.norm(doctor_control[:2]) - np.linalg.norm(intern_control[:2])
        if abs(resource_tension) > 0.5:
            effects[2] = 0.02 * np.sin(resource_tension)  # 人员利用振荡
            effects[14] = -0.01 * resource_tension  # 影响学习效率
            
        # 医生-会计成本博弈
        cost_tension = np.dot(doctor_control[1:3], accountant_control[:2])
        if abs(cost_tension) > 0.4:
            effects[5] = -0.03 * cost_tension  # 影响利润率
            effects[9] = -0.02 * cost_tension  # 影响患者满意度
            
        return effects
    
    def _constraint_nonlinearity(self, x_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        """约束非线性"""
        effects = np.zeros_like(x_t)
        
        # 伦理约束
        if x_t[12] < 0.3:  # 伦理合规度过低
            effects[12] = 0.1 * (0.5 - x_t[12])  # 恢复力
            
        # 安全约束
        if x_t[11] < 0.4:  # 安全指数过低
            effects[11] = 0.15 * (0.6 - x_t[11])
            effects[9] = -0.1  # 影响患者满意度
            
        return effects
    
    def _apply_system_constraints(self, state: np.ndarray) -> np.ndarray:
        """应用系统约束"""
        # 物理约束
        state[0] = np.clip(state[0], 0, 1)    # 病床占用率 [0,1]
        state[1] = np.clip(state[1], 0, 1)    # 设备利用率 [0,1]
        state[2] = np.clip(state[2], 0, 1)    # 人员利用率 [0,1]
        state[3] = np.clip(state[3], 0, 1)    # 库存水平 [0,1]
        
        # 财务约束
        state[5] = np.clip(state[5], -1, 1)   # 利润率 [-1,1]
        state[6] = np.clip(state[6], 0, 1)    # 负债率 [0,1]
        
        # 质量约束
        state[9] = np.clip(state[9], 0, 1)    # 患者满意度 [0,1]
        state[10] = np.clip(state[10], 0, 1)  # 治疗成功率 [0,1]
        state[11] = np.clip(state[11], 0, 1)  # 安全指数 [0,1]
        
        return state
    
    def output_equation(self, x_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        """输出方程"""
        return np.dot(self.C, x_t) + np.dot(self.D, u_t)