import numpy as np
from hospital_governance.control.distributed_control import DistributedControlSystem

def get_dummy_configs():
    return {
        'doctors': {
            'feedback_gain': np.eye(16),
            'integrator_gain': 0.1,
            'control_limits': [-1.0, 1.0]
        },
        'interns': {
            'observer_gain': np.eye(16),
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
            'policy_matrix': np.eye(3),
            'policy_limits': [-1.0, 1.0]
        }
    }

def test_distributed_control():
    configs = get_dummy_configs()
    dcs = DistributedControlSystem(configs)
    x_t = np.random.rand(16)
    x_ref = np.zeros(16)
    d_hat = np.random.rand(4)
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
    u_global = dcs.compute_control(x_t, x_ref, d_hat, holy_code_state)
    print('Global control vector:', u_global)
    assert u_global.shape == (17,)
    history = dcs.get_control_history()
    assert len(history) == 1
    print('Control history:', history)

if __name__ == '__main__':
    test_distributed_control()
