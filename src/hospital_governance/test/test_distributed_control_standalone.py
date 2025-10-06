#!/usr/bin/env python3
"""
分布式控制系统独立测试
避免复杂依赖，直接测试控制器逻辑
"""

import sys
import os
import numpy as np

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_individual_controllers():
    """测试各个控制器"""
    print("=== 测试各个控制器 ===")
    
    # 测试医生主控制器
    from hospital_governance.control.primary_controller import PrimaryStabilizingController
    
    doctor_config = {
        'feedback_gain': np.eye(16),
        'integrator_gain': 0.1,
        'control_limits': [-1.0, 1.0]
    }
    
    doctor_controller = PrimaryStabilizingController(doctor_config)
    
    # 模拟输入
    y_local = np.random.rand(16) * 0.5
    x_ref = np.zeros(16)
    d_hat = np.random.rand(4) * 0.3
    holy_code_state = {
        'ethical_constraints': {
            'min_quality_control': 0.1,
            'max_workload': 0.9
        }
    }
    
    u_doctor = doctor_controller.compute_control(y_local, x_ref, d_hat, holy_code_state, 'doctors')
    print(f"医生控制输出: {u_doctor[:4]}")  # 只显示前4个
    assert u_doctor.shape == (4,), f"医生控制输出维度错误: {u_doctor.shape}"
    
    # 测试实习医生控制器
    from hospital_governance.control.observer_controller import ObserverFeedforwardController
    
    intern_config = {
        'observer_gain': np.eye(12),  # 实习医生观测12个状态
        'feedforward_gain': 0.05,
        'control_limits': [-1.0, 1.0]
    }
    
    intern_controller = ObserverFeedforwardController(intern_config)
    y_local_intern = np.random.rand(12) * 0.5
    x_ref_intern = np.zeros(12)
    
    u_intern = intern_controller.compute_control(y_local_intern, x_ref_intern, d_hat, holy_code_state, 'interns')
    print(f"实习医生控制输出: {u_intern[:4]}")
    assert len(u_intern) == 4, f"实习医生控制输出维度错误: {len(u_intern)}"
    
    # 测试患者控制器
    from hospital_governance.control.patient_controller import PatientAdaptiveController
    
    patient_config = {
        'Kp': 0.5,
        'Ka': 0.2,
        'control_limits': [-1.0, 1.0]
    }
    
    patient_controller = PatientAdaptiveController(patient_config)
    y_local_patient = np.random.rand(8) * 0.5  # 患者观测8个财务和质量状态
    x_ref_patient = np.zeros(8)
    
    u_patient = patient_controller.compute_control(y_local_patient, x_ref_patient, d_hat, holy_code_state, 'patients')
    print(f"患者控制输出: {u_patient[:3]}")
    assert len(u_patient) == 3, f"患者控制输出维度错误: {len(u_patient)}"
    
    # 测试会计控制器
    from hospital_governance.control.constraint_controller import ConstraintEnforcementController
    
    accountant_config = {
        'constraint_matrix': np.eye(4),
        'budget_limit': 0.8,
        'control_limits': [-1.0, 1.0]
    }
    
    accountant_controller = ConstraintEnforcementController(accountant_config)
    y_local_accountant = np.random.rand(4) * 0.5  # 会计观测4个财务状态
    x_ref_accountant = np.zeros(4)
    
    u_accountant = accountant_controller.compute_control(y_local_accountant, x_ref_accountant, d_hat, holy_code_state, 'accountants')
    print(f"会计控制输出: {u_accountant[:3]}")
    assert len(u_accountant) == 3, f"会计控制输出维度错误: {len(u_accountant)}"
    
    # 测试政府控制器
    from hospital_governance.control.government_controller import GovernmentPolicyController
    
    government_config = {
        'policy_matrix': np.eye(8),  # 政府观测8个资源和伦理状态
        'policy_limits': [-1.0, 1.0]
    }
    
    government_controller = GovernmentPolicyController(government_config)
    y_local_government = np.random.rand(8) * 0.5
    x_ref_government = np.zeros(8)
    
    u_government = government_controller.compute_control(y_local_government, x_ref_government, d_hat, holy_code_state, 'government')
    print(f"政府控制输出: {u_government[:3]}")
    assert len(u_government) == 3, f"政府控制输出维度错误: {len(u_government)}"
    
    print("✓ 所有控制器测试通过")

def test_distributed_system():
    """测试分布式控制系统集成"""
    print("\n=== 测试分布式控制系统 ===")
    
    # 创建简化的分布式控制系统配置
    controller_configs = {
        'doctors': {
            'feedback_gain': np.eye(16),
            'integrator_gain': 0.1,
            'control_limits': [-1.0, 1.0]
        },
        'interns': {
            'observer_gain': np.eye(12),
            'feedforward_gain': 0.05,
            'control_limits': [-1.0, 1.0]
        },
        'patients': {
            'Kp': 0.5,
            'Ka': 0.2,
            'control_limits': [-1.0, 1.0]
        },
        'accountants': {
            'constraint_matrix': np.eye(4),
            'budget_limit': 0.8,
            'control_limits': [-1.0, 1.0]
        },
        'government': {
            'policy_matrix': np.eye(8),
            'policy_limits': [-1.0, 1.0]
        }
    }
    
    # 手动创建分布式控制系统（避免导入问题）
    from hospital_governance.control.primary_controller import PrimaryStabilizingController
    from hospital_governance.control.observer_controller import ObserverFeedforwardController
    from hospital_governance.control.patient_controller import PatientAdaptiveController
    from hospital_governance.control.constraint_controller import ConstraintEnforcementController
    from hospital_governance.control.government_controller import GovernmentPolicyController
    
    controllers = {
        'doctors': PrimaryStabilizingController(controller_configs['doctors']),
        'interns': ObserverFeedforwardController(controller_configs['interns']),
        'patients': PatientAdaptiveController(controller_configs['patients']),
        'accountants': ConstraintEnforcementController(controller_configs['accountants']),
        'government': GovernmentPolicyController(controller_configs['government'])
    }
    
    # 模拟系统状态
    x_t = np.random.rand(16) * 0.5  # 16维系统状态
    x_ref = np.zeros(16)  # 参考状态
    d_hat = np.random.rand(4) * 0.3  # 扰动预测
    
    holy_code_state = {
        'ethical_constraints': {
            'min_quality_control': 0.1,
            'max_workload': 0.9,
            'min_training_hours': 0.2,
            'min_cost_efficiency': 0.3,
            'min_health_level': 0.4,
            'min_equity_level': 0.5
        }
    }
    
    # 局部观测掩码
    observation_masks = {
        'doctors': np.ones(16, dtype=bool),  # 医生可观测全部状态
        'interns': np.array([1]*8 + [0]*4 + [1]*4, dtype=bool),  # 实习生关注资源和教育
        'patients': np.array([0]*4 + [1]*8 + [0]*4, dtype=bool),  # 患者关注财务和质量
        'accountants': np.array([0]*8 + [1]*4 + [0]*4, dtype=bool),  # 会计关注财务
        'government': np.array([1]*4 + [0]*8 + [1]*4, dtype=bool)  # 政府关注资源和伦理
    }
    
    # 计算各角色控制信号
    control_signals = {}
    
    for role, controller in controllers.items():
        y_local = x_t[observation_masks[role]]
        u_role = controller.compute_control(y_local, x_ref[observation_masks[role]], d_hat, holy_code_state, role)
        control_signals[role] = u_role
        print(f"{role} 控制信号维度: {len(u_role)}, 前3个值: {u_role[:3]}")
    
    # 合成全局控制向量
    u_global = np.zeros(17)
    
    # 医生控制 (u₁-u₄)
    if 'doctors' in control_signals:
        u_global[0:4] = control_signals['doctors'][:4]
        
    # 实习生控制 (u₅-u₈)
    if 'interns' in control_signals:
        u_global[4:8] = control_signals['interns'][:4]
        
    # 患者控制 (u₉-u₁₁)
    if 'patients' in control_signals:
        u_global[8:11] = control_signals['patients'][:3]
        
    # 会计控制 (u₁₂-u₁₄)
    if 'accountants' in control_signals:
        u_global[11:14] = control_signals['accountants'][:3]
        
    # 政府控制 (u₁₅-u₁₇)
    if 'government' in control_signals:
        u_global[14:17] = control_signals['government'][:3]
    
    print(f"\n全局控制向量维度: {u_global.shape}")
    print(f"全局控制向量: {u_global}")
    print(f"控制向量范围: [{u_global.min():.3f}, {u_global.max():.3f}]")
    
    # 验证输出约束
    assert u_global.shape == (17,), f"全局控制向量维度错误: {u_global.shape}"
    assert np.all(u_global >= -1.0) and np.all(u_global <= 1.0), "控制信号超出限制范围"
    
    print("✓ 分布式控制系统集成测试通过")

def test_holy_code_constraints():
    """测试神圣法典约束功能"""
    print("\n=== 测试神圣法典约束 ===")
    
    from hospital_governance.control.primary_controller import PrimaryStabilizingController
    
    config = {
        'feedback_gain': np.eye(16),
        'integrator_gain': 0.1,
        'control_limits': [-1.0, 1.0]
    }
    
    controller = PrimaryStabilizingController(config)
    
    # 测试约束场景
    y_local = np.random.rand(16) * 0.5
    x_ref = np.zeros(16)
    d_hat = np.random.rand(4) * 0.3
    
    # 无约束情况
    holy_code_no_constraint = {'ethical_constraints': {}}
    u_no_constraint = controller.compute_control(y_local, x_ref, d_hat, holy_code_no_constraint, 'doctors')
    
    # 有约束情况
    holy_code_with_constraint = {
        'ethical_constraints': {
            'min_quality_control': 0.5,  # 强制最小质量控制
            'max_workload': 0.2  # 限制最大工作负荷
        }
    }
    u_with_constraint = controller.compute_control(y_local, x_ref, d_hat, holy_code_with_constraint, 'doctors')
    
    print(f"无约束控制输出: {u_no_constraint}")
    print(f"有约束控制输出: {u_with_constraint}")
    
    # 验证约束生效
    assert u_with_constraint[3] >= 0.5, f"质量控制约束未生效: {u_with_constraint[3]}"
    assert u_with_constraint[2] <= 0.2, f"工作负荷约束未生效: {u_with_constraint[2]}"
    
    print("✓ 神圣法典约束测试通过")

def main():
    """主测试函数"""
    print("开始分布式控制系统测试...")
    
    try:
        test_individual_controllers()
        test_distributed_system()
        test_holy_code_constraints()
        
        print("\n🎉 所有测试通过！分布式控制系统运行正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())