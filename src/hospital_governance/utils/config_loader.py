import yaml
import numpy as np
from typing import Dict, Any

class SystemMatricesConfig:
    """系统矩阵配置对象，支持从yaml加载A, B, C, D, E矩阵"""
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.state_dim = 16
        self.input_dim = 17
        self.disturbance_dim = 6
        self._parse_state_transition(cfg.get('state_transition_matrix', {}))
        self._parse_control_input(cfg.get('control_input_matrix', {}))
        self._parse_disturbance(cfg.get('disturbance_matrix', []))
        self._parse_output(cfg.get('output_matrix', {}))
        self._parse_feedthrough(cfg.get('feedthrough_matrix', {}))

    def _parse_state_transition(self, conf: Dict[str, Any]):
        self.A = np.eye(self.state_dim) * conf.get('diagonal', 1.0)
        for i, j, v in conf.get('interactions', []):
            self.A[i, j] = v

    def _parse_control_input(self, conf: Dict[str, Any]):
        self.B = np.zeros((self.state_dim, self.input_dim))
        for group in ['doctor_controls', 'intern_controls']:
            for i, j, v in conf.get(group, []):
                self.B[i, j] = v

    def _parse_disturbance(self, conf):
        self.E = np.zeros((self.state_dim, self.disturbance_dim))
        for i, j, v in conf:
            self.E[i, j] = v

    def _parse_output(self, conf: Dict[str, Any]):
        if conf.get('type') == 'identity':
            self.C = np.eye(self.state_dim)
        else:
            self.C = np.zeros((self.state_dim, self.state_dim))

    def _parse_feedthrough(self, conf: Dict[str, Any]):
        if conf.get('type') == 'zero':
            self.D = np.zeros((self.state_dim, self.input_dim))
        else:
            self.D = np.eye(self.state_dim, self.input_dim)

    def as_dict(self):
        return {
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'D': self.D,
            'E': self.E
        }

# 用法示例：
# config = SystemMatricesConfig('config/system_matrices.yaml')
# matrices = config.as_dict()
