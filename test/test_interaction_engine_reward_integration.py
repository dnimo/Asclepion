import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from hospital_governance.agents.interaction_engine import KallipolisInteractionEngine
from hospital_governance.core.state_space import SystemState

class TestInteractionEngineRewardIntegration(unittest.TestCase):
    def setUp(self):
        # 初始化系统状态
        from hospital_governance.agents.role_agents import create_default_agent_system
        from test.mock_parliament import MockParliament
        self.system_state = SystemState(
            medical_resource_utilization=0.5,
            patient_waiting_time=0.2,
            financial_indicator=0.9,
            ethical_compliance=0.8,
            education_training_quality=0.6,
            intern_satisfaction=0.7,
            professional_development=0.65,
            mentorship_effectiveness=0.75,
            patient_satisfaction=0.85,
            service_accessibility=0.8,
            care_quality_index=0.9,
            safety_incident_rate=0.05,
            operational_efficiency=0.7,
            staff_workload_balance=0.6,
            crisis_response_capability=0.8,
            regulatory_compliance_score=0.95
        )
        self.role_manager = create_default_agent_system()
        self.parliament = MockParliament()
        self.engine = KallipolisInteractionEngine(self.role_manager, self.parliament, self.system_state)
        self.engine.active_crises = []

    def test_reward_controller_invocation(self):
        actions = {
            'doctors': np.array([1.0, 0.5]),
            'interns': np.array([0.8, 0.2]),
            'patients': np.array([0.6]),
            'accountants': np.array([0.9]),
            'government': np.array([0.7])
        }
        parliament_decisions = {
            'policy1': {'approved': True, 'proposer': 'doctors', 'approval_ratio': 0.95},
            'policy2': {'approved': False, 'proposer': 'government', 'approval_ratio': 0.5}
        }
        rewards = self.engine._execute_actions(actions, parliament_decisions)
        # 检查所有角色都获得了奖励
        self.assertTrue(all(role in rewards for role in actions))
        # 检查奖励为浮点数
        for reward in rewards.values():
            self.assertIsInstance(reward, float)
        # 检查奖励值合理（非负）
        for reward in rewards.values():
            self.assertGreaterEqual(reward, 0.0)

if __name__ == '__main__':
    unittest.main()
