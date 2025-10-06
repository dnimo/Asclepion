import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import signal, linalg
import matplotlib.pyplot as plt

@dataclass
class FrequencyAnalysisConfig:
    """频域分析配置"""
    frequency_range: Tuple[float, float] = (0.01, 10.0)  # 频率范围 (rad/s)
    num_points: int = 1000           # 频率点数
    stability_margins: Dict[str, float] = None  # 稳定裕度要求
    
    def __post_init__(self):
        if self.stability_margins is None:
            self.stability_margins = {
                'gain_margin_db': 6.0,     # 增益裕度 (dB)
                'phase_margin_deg': 45.0,  # 相位裕度 (度)
            }

class FrequencyAnalyzer:
    """频域分析器"""
    
    def __init__(self, system_dynamics, config: FrequencyAnalysisConfig = None):
        self.system_dynamics = system_dynamics
        self.config = config or FrequencyAnalysisConfig()
        
        # 分析结果
        self.frequency_response: Dict[str, np.ndarray] = {}
        self.stability_margins: Dict[str, float] = {}
        self.bode_data: Dict[str, np.ndarray] = {}
    
    def compute_frequency_response(self, operating_point: np.ndarray = None,
                                 linearized_system: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """计算频率响应"""
        if linearized_system is None:
            if operating_point is None:
                operating_point = self.system_dynamics.state_space.current_state
            linearized_system = self.system_dynamics.linearize(operating_point)
        
        A, B, C, D = (linearized_system['A_linearized'], 
                      linearized_system['B_linearized'], 
                      linearized_system['C_linearized'],
                      np.zeros((C.shape[0], B.shape[1])))  # 假设D=0
        
        # 创建状态空间系统
        sys = signal.StateSpace(A, B, C, D)
        
        # 计算频率响应
        w = np.logspace(np.log10(self.config.frequency_range[0]), 
                       np.log10(self.config.frequency_range[1]), 
                       self.config.num_points)
        
        # 频率响应计算
        frequency_response = signal.freqresp(sys, w)
        
        self.frequency_response = {
            'frequencies': frequency_response[0],
            'magnitude': np.abs(frequency_response[1]),
            'phase': np.angle(frequency_response[1], deg=True),
            'real': np.real(frequency_response[1]),
            'imaginary': np.imag(frequency_response[1])
        }
        
        return self.frequency_response
    
    def compute_stability_margins(self) -> Dict[str, float]:
        """计算稳定裕度"""
        if not self.frequency_response:
            self.compute_frequency_response()
        
        # 寻找增益穿越频率和相位穿越频率
        gain_margin, phase_margin, gain_crossover, phase_crossover = \
            self._find_stability_margins()
        
        self.stability_margins = {
            'gain_margin_db': gain_margin,
            'phase_margin_deg': phase_margin,
            'gain_crossover_freq': gain_crossover,
            'phase_crossover_freq': phase_crossover,
            'is_stable': gain_margin > 0 and phase_margin > 0
        }
        
        return self.stability_margins
    
    def _find_stability_margins(self) -> Tuple[float, float, float, float]:
        """寻找稳定裕度"""
        mag = self.frequency_response['magnitude']
        phase = self.frequency_response['phase']
        w = self.frequency_response['frequencies']
        
        # 寻找相位穿越频率 (相位 = -180度)
        phase_crossover_indices = np.where(np.diff(np.sign(phase + 180)))[0]
        
        if len(phase_crossover_indices) > 0:
            idx = phase_crossover_indices[0]
            phase_crossover_freq = w[idx]
            gain_margin_db = -20 * np.log10(mag[idx]) if mag[idx] > 0 else np.inf
        else:
            phase_crossover_freq = np.inf
            gain_margin_db = np.inf
        
        # 寻找增益穿越频率 (增益 = 0 dB)
        mag_db = 20 * np.log10(mag)
        gain_crossover_indices = np.where(np.diff(np.sign(mag_db)))[0]
        
        if len(gain_crossover_indices) > 0:
            idx = gain_crossover_indices[0]
            gain_crossover_freq = w[idx]
            phase_margin_deg = phase[idx] + 180
        else:
            gain_crossover_freq = np.inf
            phase_margin_deg = np.inf
        
        return gain_margin_db, phase_margin_deg, gain_crossover_freq, phase_crossover_freq
    
    def compute_sensitivity_functions(self, controller: Any) -> Dict[str, np.ndarray]:
        """计算灵敏度函数"""
        if not self.frequency_response:
            self.compute_frequency_response()
        
        # 简化灵敏度函数计算
        # S = 1/(1 + GK), T = GK/(1 + GK)
        G = self.frequency_response['magnitude'] * np.exp(1j * np.radians(self.frequency_response['phase']))
        
        # 假设控制器是比例控制器
        K = 1.0  # 简化假设
        
        S = 1 / (1 + G * K)  # 灵敏度函数
        T = G * K / (1 + G * K)  # 互补灵敏度函数
        
        return {
            'sensitivity': np.abs(S),
            'complementary_sensitivity': np.abs(T),
            'frequencies': self.frequency_response['frequencies']
        }
    
    def compute_robustness_metrics(self, uncertainty_model: Any = None) -> Dict[str, float]:
        """计算鲁棒性指标"""
        stability_margins = self.compute_stability_margins()
        sensitivity_functions = self.compute_sensitivity_functions(None)
        
        # H∞范数估计 (最大奇异值)
        h_inf_norm = np.max(self.frequency_response['magnitude'])
        
        # 峰值灵敏度
        peak_sensitivity = np.max(sensitivity_functions['sensitivity'])
        
        # 带宽估计
        bandwidth = self._estimate_bandwidth()
        
        return {
            'h_inf_norm': float(h_inf_norm),
            'peak_sensitivity': float(peak_sensitivity),
            'bandwidth': float(bandwidth),
            'gain_margin_db': stability_margins['gain_margin_db'],
            'phase_margin_deg': stability_margins['phase_margin_deg'],
            'stability_robustness': self._compute_stability_robustness(stability_margins),
            'performance_robustness': self._compute_performance_robustness(peak_sensitivity)
        }
    
    def _estimate_bandwidth(self) -> float:
        """估计带宽"""
        if not self.frequency_response:
            return 0.0
        
        mag = self.frequency_response['magnitude']
        w = self.frequency_response['frequencies']
        
        # 寻找-3dB点
        mag_db = 20 * np.log10(mag)
        dc_gain = mag_db[0] if len(mag_db) > 0 else 0
        
        bandwidth_indices = np.where(mag_db <= dc_gain - 3)[0]
        
        if len(bandwidth_indices) > 0:
            return w[bandwidth_indices[0]]
        else:
            return w[-1]  # 返回最大测试频率
    
    def _compute_stability_robustness(self, margins: Dict[str, float]) -> float:
        """计算稳定性鲁棒性"""
        required_gain_margin = self.config.stability_margins['gain_margin_db']
        required_phase_margin = self.config.stability_margins['phase_margin_deg']
        
        actual_gain_margin = margins['gain_margin_db']
        actual_phase_margin = margins['phase_margin_deg']
        
        # 计算裕度比
        gain_ratio = actual_gain_margin / required_gain_margin if required_gain_margin > 0 else 1.0
        phase_ratio = actual_phase_margin / required_phase_margin if required_phase_margin > 0 else 1.0
        
        # 综合鲁棒性评分
        robustness = min(gain_ratio, phase_ratio)
        return float(max(0.0, min(1.0, robustness)))
    
    def _compute_performance_robustness(self, peak_sensitivity: float) -> float:
        """计算性能鲁棒性"""
        # 峰值灵敏度越小，鲁棒性越好
        # 典型目标: Ms < 2 (6dB)
        target_peak_sensitivity = 2.0
        robustness = target_peak_sensitivity / peak_sensitivity if peak_sensitivity > 0 else 1.0
        return float(max(0.0, min(1.0, robustness)))
    
    def plot_bode_diagram(self, save_path: str = None) -> plt.Figure:
        """绘制Bode图"""
        if not self.frequency_response:
            self.compute_frequency_response()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 幅频特性
        ax1.semilogx(self.frequency_response['frequencies'], 
                    20 * np.log10(self.frequency_response['magnitude']))
        ax1.set_ylabel('Magnitude [dB]')
        ax1.set_title('Bode Diagram')
        ax1.grid(True)
        
        # 相频特性
        ax2.semilogx(self.frequency_response['frequencies'], 
                    self.frequency_response['phase'])
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.grid(True)
        
        # 添加稳定裕度标注
        margins = self.compute_stability_margins()
        if np.isfinite(margins['gain_crossover_freq']):
            ax2.axhline(y=-180, color='r', linestyle='--', alpha=0.7)
            ax1.axvline(x=margins['gain_crossover_freq'], color='r', linestyle='--', alpha=0.7)
        
        if np.isfinite(margins['phase_crossover_freq']):
            ax1.axhline(y=0, color='g', linestyle='--', alpha=0.7)
            ax2.axvline(x=margins['phase_crossover_freq'], color='g', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_nyquist_diagram(self, save_path: str = None) -> plt.Figure:
        """绘制Nyquist图"""
        if not self.frequency_response:
            self.compute_frequency_response()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        real = self.frequency_response['real']
        imag = self.frequency_response['imaginary']
        
        ax.plot(real, imag)
        ax.plot(real, -imag, '--', alpha=0.5)  # 对称部分
        
        # 添加单位圆和(-1,0)点
        circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.5)
        ax.add_artist(circle)
        ax.plot(-1, 0, 'ro', markersize=8)
        
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title('Nyquist Diagram')
        ax.grid(True)
        ax.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        if not self.frequency_response:
            return {}
        
        robustness_metrics = self.compute_robustness_metrics()
        stability_margins = self.compute_stability_margins()
        
        return {
            'frequency_range': self.config.frequency_range,
            'stability_margins': stability_margins,
            'robustness_metrics': robustness_metrics,
            'system_order': self.system_dynamics.state_space.dimensions,
            'bandwidth': self._estimate_bandwidth(),
            'resonant_peak': np.max(self.frequency_response['magnitude']),
            'dc_gain': self.frequency_response['magnitude'][0] if len(self.frequency_response['magnitude']) > 0 else 0
        }