import numpy as np
from typing import Dict, Optional, Any
from pathlib import Path
import yaml

class SystemMatrixGenerator:
    """系统矩阵生成器"""
    
    @staticmethod
    def generate_nominal_matrices() -> Dict[str, np.ndarray]:
        """生成标称系统矩阵"""
        n, m, p = 16, 17, 6  # 状态、控制、扰动维度
        
        # 状态转移矩阵 A (16×16)
        A = np.zeros((n, n))
        np.fill_diagonal(A, 0.8)  # 状态惯性
        
        # 资源状态交互
        A[0, 2] = 0.1   # 人员利用影响病床占用
        A[1, 0] = 0.05  # 病床占用影响设备利用
        A[2, 3] = -0.1  # 库存影响人员效率
        A[3, 1] = 0.08  # 设备利用影响库存消耗
        
        # 财务状态交互  
        A[4, 0] = 0.15  # 病床占用影响现金储备
        A[5, 0] = 0.2   # 病床占用影响利润
        A[5, 9] = 0.15  # 患者满意度影响利润
        A[6, 5] = 0.3   # 利润率影响负债率
        A[7, 2] = -0.1  # 人员利用影响成本效率
        
        # 服务质量状态交互
        A[8, 0] = -0.1  # 病床占用影响等待时间
        A[9, 2] = -0.2  # 人员利用影响满意度
        A[9, 8] = -0.3  # 等待时间影响满意度
        A[9, 10] = 0.4  # 治疗成功率影响满意度
        A[10, 1] = 0.2  # 设备利用影响治疗成功率
        A[11, 10] = 0.3 # 治疗成功率影响安全指数
        
        # 教育伦理状态交互
        A[12, 9] = 0.2  # 患者满意度影响伦理合规
        A[13, 5] = 0.15 # 利润率影响分配公平性
        A[14, 2] = 0.25 # 人员利用影响学习效率
        A[15, 14] = 0.3 # 学习效率影响知识传递
        
        # 控制输入矩阵 B (16×17)
        B = np.zeros((n, m))
        
        # 医生控制影响 (u₁-u₄)
        B[0, 0] = 0.3   # 资源分配影响病床占用
        B[1, 1] = 0.25  # 治疗方案影响设备利用
        B[2, 2] = 0.2   # 工作负荷管理影响人员利用
        B[9, 3] = 0.4   # 质量控制影响患者满意度
        B[10, 1] = 0.3  # 治疗方案影响治疗成功率
        
        # 实习生控制影响 (u₅-u₈)
        B[2, 5] = 0.3   # 临床参与影响人员利用
        B[9, 6] = 0.2   # 反馈生成影响患者满意度
        B[14, 4] = 0.5  # 学习时间分配影响学习效率
        B[15, 7] = 0.3  # 扰动预测影响知识传递
        
        # 患者控制影响 (u₉-u₁₁)
        B[9, 8] = 0.6   # 满意度反馈影响满意度
        B[8, 9] = 0.4   # 需求倡导影响等待时间
        B[11, 10] = 0.3 # 质量监控影响安全指数
        
        # 会计控制影响 (u₁₂-u₁₄)
        B[4, 11] = 0.5  # 预算控制影响现金储备
        B[5, 12] = 0.4  # 成本优化影响利润率
        B[6, 13] = 0.3  # 风险缓解影响负债率
        
        # 政府控制影响 (u₁₅-u₁₇)
        B[5, 14] = 0.4  # 政策指导影响利润率
        B[12, 15] = 0.6 # 资金调整影响伦理合规
        B[13, 16] = 0.5 # 法规合规影响分配公平性
        
        # 扰动输入矩阵 E (16×6)
        E = np.zeros((n, p))
        
        E[0, 0] = 0.3   # 疫情影响病床占用
        E[0, 3] = 0.5   # 需求冲击影响病床占用
        E[1, 4] = 0.4   # 供应中断影响设备利用
        E[2, 5] = 0.4   # 人员变化影响人员利用
        E[4, 1] = 0.3   # 政策变化影响现金储备
        E[5, 2] = 0.2   # 经济波动影响利润率
        E[8, 3] = 0.6   # 需求冲击影响等待时间
        E[9, 0] = 0.4   # 疫情影响患者满意度
        
        # 输出矩阵 C (假设完全观测)
        C = np.eye(n)
        
        # 直通矩阵 D (通常为零)
        D = np.zeros((n, m))
        
        return {
            'A': A, 'B': B, 'E': E, 
            'C': C, 'D': D
        }
    
    @staticmethod
    def add_uncertainty(nominal_matrices: Dict[str, np.ndarray], 
                       uncertainty_level: float = 0.1) -> Dict[str, np.ndarray]:
        """添加模型不确定性"""
        uncertain_matrices = {}
        
        for key, matrix in nominal_matrices.items():
            if key in ['A', 'B']:  # 只为A和B矩阵添加不确定性
                uncertainty = np.random.normal(0, uncertainty_level, matrix.shape)
                uncertain_matrices[key] = matrix + uncertainty
            else:
                uncertain_matrices[key] = matrix.copy()
                
        return uncertain_matrices

    @staticmethod
    def load_from_yaml(
        yaml_path: str = 'config/system_matrices.yaml',
        scenario: Optional[str] = None,
        n: int = 16,
        m: int = 17,
        p: int = 6
    ) -> Dict[str, np.ndarray]:
        """从YAML配置加载系统矩阵

        参数:
        - yaml_path: 配置文件路径
        - scenario: 可选场景名（若YAML中按场景组织）
        - n, m, p: 状态/控制/扰动维度（默认16/17/6）

        返回: 包含 A, B, E, C, D 的字典
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"System matrices YAML not found: {yaml_path}")

        with path.open('r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}

        # 支持两种结构：
        # 1) 直接扁平定义（当前仓库默认）
        # 2) scenarios: { name: { ...上述键... } }
        if scenario and isinstance(cfg.get('scenarios'), dict):
            if scenario not in cfg['scenarios']:
                raise KeyError(f"Scenario '{scenario}' not found in YAML")
            cfg = cfg['scenarios'][scenario]

        # 构造A
        A = np.zeros((n, n), dtype=float)
        stm = cfg.get('state_transition_matrix', {}) or {}
        diag_val = stm.get('diagonal')
        if diag_val is not None:
            np.fill_diagonal(A, float(diag_val))
        interactions = stm.get('interactions', []) or []
        for triplet in interactions:
            i, j, val = SystemMatrixGenerator._parse_entry(triplet, 3, 'A.interactions')
            SystemMatrixGenerator._safe_assign(A, i, j, val, (n, n), 'A')

        # 构造B
        B = np.zeros((n, m), dtype=float)
        cim = cfg.get('control_input_matrix', {}) or {}
        # 允许如下两种：
        # - 直接列表 entries: [[r,c,v], ...]
        # - 分组键：doctor_controls/intern_controls/... 每个键下面是 [[r,c,v], ...]
        if isinstance(cim, dict):
            # entries 汇总
            if isinstance(cim.get('entries'), list):
                for triplet in cim['entries']:
                    i, j, val = SystemMatrixGenerator._parse_entry(triplet, 3, 'B.entries')
                    SystemMatrixGenerator._safe_assign(B, i, j, val, (n, m), 'B')
            # 分组汇总
            for group, items in cim.items():
                if group == 'entries':
                    continue
                if isinstance(items, list):
                    for triplet in items:
                        i, j, val = SystemMatrixGenerator._parse_entry(triplet, 3, f'B.{group}')
                        SystemMatrixGenerator._safe_assign(B, i, j, val, (n, m), 'B')

        # 构造E
        E = np.zeros((n, p), dtype=float)
        dm = cfg.get('disturbance_matrix', []) or []
        if isinstance(dm, list):
            for triplet in dm:
                i, j, val = SystemMatrixGenerator._parse_entry(triplet, 3, 'E')
                SystemMatrixGenerator._safe_assign(E, i, j, val, (n, p), 'E')

        # 构造C
        C = np.eye(n, dtype=float)
        om = cfg.get('output_matrix', {}) or {}
        if isinstance(om, dict):
            om_type = (om.get('type') or 'identity').lower()
            if om_type == 'identity':
                C = np.eye(n, dtype=float)
            elif om_type == 'zero':
                C = np.zeros((n, n), dtype=float)
            elif om_type == 'custom':
                # 支持 entries: [[i,j,val], ...]
                C = np.zeros((n, n), dtype=float)
                for triplet in (om.get('entries') or []):
                    i, j, val = SystemMatrixGenerator._parse_entry(triplet, 3, 'C.entries')
                    SystemMatrixGenerator._safe_assign(C, i, j, val, (n, n), 'C')

        # 构造D
        D = np.zeros((n, m), dtype=float)
        fm = cfg.get('feedthrough_matrix', {}) or {}
        if isinstance(fm, dict):
            fm_type = (fm.get('type') or 'zero').lower()
            if fm_type == 'zero':
                D = np.zeros((n, m), dtype=float)
            elif fm_type == 'identity':
                # 取 min(n,m) 作为对角线长度
                D = np.zeros((n, m), dtype=float)
                for k in range(min(n, m)):
                    D[k, k] = 1.0
            elif fm_type == 'custom':
                D = np.zeros((n, m), dtype=float)
                for triplet in (fm.get('entries') or []):
                    i, j, val = SystemMatrixGenerator._parse_entry(triplet, 3, 'D.entries')
                    SystemMatrixGenerator._safe_assign(D, i, j, val, (n, m), 'D')

        # 最终返回
        return {'A': A, 'B': B, 'E': E, 'C': C, 'D': D}

    @staticmethod
    def _parse_entry(entry: Any, expected_len: int, where: str):
        if not isinstance(entry, (list, tuple)) or len(entry) != expected_len:
            raise ValueError(f"Invalid entry at {where}: expected list/tuple of len {expected_len}, got {entry}")
        i, j, val = entry
        return int(i), int(j), float(val)

    @staticmethod
    def _safe_assign(mat: np.ndarray, i: int, j: int, val: float, shape: tuple, name: str):
        nrows, ncols = shape
        if not (0 <= i < nrows and 0 <= j < ncols):
            raise IndexError(f"{name}[{i},{j}] out of bounds for shape {shape}")
        mat[i, j] = val