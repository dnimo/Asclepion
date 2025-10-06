import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import linalg
from src.hospital_governance.core.state_space import StateSpace
from src.hospital_governance.core.system_dynamics import SystemDynamics

@dataclass
class LyapunovConfig:
    """李雅普诺夫分析配置"""
    stability_threshold: float = 0.1
    convergence_rate_threshold: float = 0.05
    max_iterations: int = 1000
    tolerance: float = 1e-6
    adaptation_rate: float = 0.01

class LyapunovAnalyzer:
    """李雅普诺夫稳定性分析器"""
    
    def __init__(self, state_space: StateSpace, system_dynamics: SystemDynamics,
                 config: LyapunovConfig = None):
        self.state_space = state_space
        self.system_dynamics = system_dynamics
        self.config = config or LyapunovConfig()
        
        # 李雅普诺夫矩阵
        self.P = np.eye(state_space.definition.dimension)
        self.P_history: List[np.ndarray] = [self.P.copy()]
        
        # 稳定性历史
        self.stability_history: List[Dict[str, float]] = []
        self.convergence_rates: List[float] = []
    
    def compute_lyapunov_function(self, state: np.ndarray, 
                                 reference: np.ndarray = None) -> float:
        """计算李雅普诺夫函数值"""
        if reference is None:
            reference = np.zeros(self.state_space.definition.dimension)
        
        error = state - reference
        V = error.T @ self.P @ error
        return float(V)
    
    def compute_lyapunov_derivative(self, state: np.ndarray, 
                                   control_input: np.ndarray,
                                   reference: np.ndarray = None,
                                   disturbance: np.ndarray = None) -> float:
        """计算李雅普诺夫函数导数"""
        if reference is None:
            reference = np.zeros(self.state_space.definition.dimension)
        
        error = state - reference
        
        # 线性化系统
        linearized = self.system_dynamics.linearize(state)
        A_lin = linearized['A_linearized']
        B_lin = linearized['B_linearized']
        
        # 计算误差导数
        error_derivative = A_lin @ error + B_lin @ control_input
        
        # 计算李雅普诺夫导数
        V_dot = error.T @ (A_lin.T @ self.P + self.P @ A_lin) @ error + \
                2 * error.T @ self.P @ B_lin @ control_input
        
        return float(V_dot)
    
    def update_lyapunov_matrix(self, state: np.ndarray, 
                              adaptation_rate: float = None) -> np.ndarray:
        """更新李雅普诺夫矩阵"""
        if adaptation_rate is None:
            adaptation_rate = self.config.adaptation_rate
        
        # 线性化系统
        linearized = self.system_dynamics.linearize(state)
        A_lin = linearized['A_linearized']
        
        # 解李雅普诺夫方程
        Q = np.eye(self.state_space.definition.dimension)
        try:
            P_new = linalg.solve_continuous_lyapunov(A_lin.T, -Q)
            
            # 确保P是正定的
            P_new = self._ensure_positive_definite(P_new)
            
            # 自适应更新
            self.P = (1 - adaptation_rate) * self.P + adaptation_rate * P_new
            self.P = self._ensure_positive_definite(self.P)
            
        except (ValueError, linalg.LinAlgError):
            # 如果无法求解，保持原矩阵
            pass
        
        self.P_history.append(self.P.copy())
        return self.P
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """确保矩阵正定"""
        # 检查是否正定
        try:
            linalg.cholesky(matrix)
            return matrix
        except linalg.LinAlgError:
            # 如果不是正定，添加小的正定扰动
            n = matrix.shape[0]
            return matrix + np.eye(n) * 1e-6
    
    def check_stability(self, state: np.ndarray, reference: np.ndarray = None,
                       control_input: np.ndarray = None) -> Dict[str, Any]:
        """检查系统稳定性"""
        if reference is None:
            reference = np.zeros(self.state_space.definition.dimension)
        
        if control_input is None:
            control_input = np.zeros(self.system_dynamics.config.B_matrix.shape[1])
        
        # 计算李雅普诺夫函数值
        V = self.compute_lyapunov_function(state, reference)
        
        # 计算李雅普诺夫导数
        V_dot = self.compute_lyapunov_derivative(state, control_input, reference)
        
        # 更新李雅普诺夫矩阵
        self.update_lyapunov_matrix(state)
        
        # 判断稳定性
        is_stable = V_dot < -self.config.stability_threshold
        is_marginally_stable = abs(V_dot) < self.config.stability_threshold
        is_unstable = V_dot > self.config.stability_threshold
        
        # 计算稳定裕度
        stability_margin = -V_dot if V_dot < 0 else 0.0
        
        # 计算收敛速率估计
        convergence_rate = self._estimate_convergence_rate(V, V_dot)
        
        # 记录稳定性历史
        stability_record = {
            'V': V,
            'V_dot': V_dot,
            'is_stable': is_stable,
            'stability_margin': stability_margin,
            'convergence_rate': convergence_rate,
            'timestamp': len(self.stability_history)
        }
        self.stability_history.append(stability_record)
        self.convergence_rates.append(convergence_rate)
        
        return stability_record
    
    def _estimate_convergence_rate(self, V: float, V_dot: float) -> float:
        """估计收敛速率"""
        if V <= 0 or V_dot >= 0:
            return 0.0
        
        # 简化的收敛速率估计
        convergence_rate = -V_dot / V
        return min(1.0, max(0.0, convergence_rate))
    
    def compute_region_of_attraction(self, equilibrium: np.ndarray,
                                   max_radius: float = 1.0,
                                   num_directions: int = 100) -> Dict[str, Any]:
        """计算吸引域"""
        directions = self._generate_directions(num_directions)
        attraction_radii = []
        
        for direction in directions:
            radius = self._find_attraction_radius(equilibrium, direction, max_radius)
            attraction_radii.append(radius)
        
        attraction_radii = np.array(attraction_radii)
        
        return {
            'equilibrium': equilibrium,
            'average_radius': float(np.mean(attraction_radii)),
            'min_radius': float(np.min(attraction_radii)),
            'max_radius': float(np.max(attraction_radii)),
            'volume_estimate': float(self._estimate_volume(attraction_radii)),
            'directions_tested': num_directions
        }
    
    def _generate_directions(self, num_directions: int) -> List[np.ndarray]:
        """生成测试方向"""
        directions = []
        n = self.state_space.definition.dimension
        
        # 生成均匀分布的方向
        for _ in range(num_directions):
            direction = np.random.randn(n)
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)
        
        return directions
    
    def _find_attraction_radius(self, equilibrium: np.ndarray,
                               direction: np.ndarray,
                               max_radius: float) -> float:
        """在给定方向找到吸引半径"""
        low, high = 0.0, max_radius
        
        for _ in range(self.config.max_iterations):
            radius = (low + high) / 2
            test_state = equilibrium + radius * direction
            
            # 检查稳定性
            stability = self.check_stability(test_state, equilibrium)
            
            if stability['is_stable'] and stability['convergence_rate'] > self.config.convergence_rate_threshold:
                low = radius  # 可以扩大半径
            else:
                high = radius  # 需要缩小半径
            
            if high - low < self.config.tolerance:
                break
        
        return (low + high) / 2
    
    def _estimate_volume(self, radii: np.ndarray) -> float:
        """估计吸引域体积"""
        n = len(radii)
        if n == 0:
            return 0.0
        
        # 使用球坐标体积公式的简化估计
        avg_radius = np.mean(radii)
        volume = (np.pi ** (n/2)) / (np.math.gamma(n/2 + 1)) * (avg_radius ** n)
        return volume
    
    def analyze_robust_stability(self, uncertainty_bound: float,
                               num_samples: int = 100) -> Dict[str, Any]:
        """分析鲁棒稳定性"""
        stability_margins = []
        convergence_rates = []
        
        for _ in range(num_samples):
            # 添加随机不确定性
            uncertain_state = self._add_uncertainty(uncertainty_bound)
            
            # 检查稳定性
            stability = self.check_stability(uncertain_state)
            
            stability_margins.append(stability['stability_margin'])
            convergence_rates.append(stability['convergence_rate'])
        
        return {
            'uncertainty_bound': uncertainty_bound,
            'average_stability_margin': float(np.mean(stability_margins)),
            'min_stability_margin': float(np.min(stability_margins)),
            'robust_stability_probability': float(np.mean([m > 0 for m in stability_margins])),
            'average_convergence_rate': float(np.mean(convergence_rates)),
            'samples_tested': num_samples
        }
    
    def _add_uncertainty(self, bound: float) -> np.ndarray:
        """添加状态不确定性"""
        current_state = self.state_space.current_state
        uncertainty = np.random.uniform(-bound, bound, current_state.shape)
        return current_state + uncertainty
    
    def get_stability_summary(self) -> Dict[str, Any]:
        """获取稳定性摘要"""
        if not self.stability_history:
            return {}
        
        recent_stability = self.stability_history[-50:]  # 最近50个样本
        
        stability_margins = [s['stability_margin'] for s in recent_stability]
        convergence_rates = [s['convergence_rate'] for s in recent_stability]
        V_values = [s['V'] for s in recent_stability]
        
        return {
            'current_stability_margin': self.stability_history[-1]['stability_margin'],
            'average_stability_margin': float(np.mean(stability_margins)),
            'stability_margin_std': float(np.std(stability_margins)),
            'average_convergence_rate': float(np.mean(convergence_rates)),
            'min_convergence_rate': float(np.min(convergence_rates)),
            'lyapunov_function_mean': float(np.mean(V_values)),
            'lyapunov_function_trend': self._compute_trend(V_values),
            'total_stability_checks': len(self.stability_history),
            'stable_percentage': float(np.mean([s['is_stable'] for s in recent_stability]))
        }
    
    def _compute_trend(self, values: List[float]) -> float:
        """计算趋势（斜率）"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)
    
    def reset(self):
        """重置分析器"""
        self.P = np.eye(self.state_space.definition.dimension)
        self.P_history = [self.P.copy()]
        self.stability_history.clear()
        self.convergence_rates.clear()