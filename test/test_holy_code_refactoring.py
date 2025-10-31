"""
测试重构后的Holy Code组件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接导入模块进行测试
try:
    from src.hospital_governance.holy_code.holy_code_manager import HolyCodeManager, HolyCodeConfig
    from src.hospital_governance.holy_code.parliament import ParliamentConfig
    from src.hospital_governance.holy_code.reference_generator import ReferenceConfig, ReferenceType
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试修复导入路径...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from hospital_governance.holy_code.holy_code_manager import HolyCodeManager, HolyCodeConfig
    from hospital_governance.holy_code.parliament import ParliamentConfig
    from hospital_governance.holy_code.reference_generator import ReferenceConfig, ReferenceType

class TestHolyCodeRefactoring:
    """测试Holy Code重构"""
    
    def setup_method(self):
        """设置测试"""
        self.config = HolyCodeConfig(
            enable_rule_learning=True,
            enable_adaptive_references=True,
            crisis_threshold=0.6
        )
        self.holy_code_manager = HolyCodeManager(self.config)
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        assert self.holy_code_manager is not None
        assert self.holy_code_manager.rule_engine is not None
        assert self.holy_code_manager.rule_library is not None
        assert self.holy_code_manager.parliament is not None
        assert self.holy_code_manager.reference_generator is not None
    
    def test_rule_deduplication(self):
        """测试规则去重"""
        # 验证规则引擎不再包含重复的条件/动作函数
        rule_engine = self.holy_code_manager.rule_engine
        
        # 检查规则引擎是否正确委托给规则库
        assert hasattr(rule_engine, '_rule_library')
        
        # 验证规则库包含所有预定义规则
        rule_library = self.holy_code_manager.rule_library
        rules = rule_library.get_all_rules()
        assert len(rules) > 0
        
        # 验证有核心伦理规则
        ethic_rules = [r for r in rules if r.rule_id.startswith('ETHIC_')]
        assert len(ethic_rules) >= 5
        
        # 验证有政府规则
        gov_rules = [r for r in rules if r.rule_id.startswith('GOV_')]
        assert len(gov_rules) >= 3
    
    def test_agent_decision_request_normal(self):
        """测试正常情况下的agent决策请求"""
        decision_context = {
            'decision_type': 'routine_operation',
            'agent_id': 'test_doctor',
            'state': {
                'patient_safety': 0.8,
                'medical_quality': 0.75,
                'system_stability': 0.8,
                'financial_health': 0.7
            },
            'proposed_action': {
                'type': 'treatment_plan',
                'patient_id': 'P001'
            }
        }
        
        guidance = self.holy_code_manager.process_agent_decision_request(
            'test_doctor', decision_context
        )
        
        assert 'rule_recommendations' in guidance
        assert 'reference_targets' in guidance
        assert 'priority_adjustments' in guidance
        assert 'crisis_mode' in guidance
        assert not guidance['crisis_mode']  # 正常情况不应触发危机模式
    
    def test_agent_decision_request_crisis(self):
        """测试危机情况下的agent决策请求"""
        decision_context = {
            'decision_type': 'crisis_response',
            'agent_id': 'chief_doctor',
            'state': {
                'patient_safety': 0.6,  # 低于阈值
                'medical_quality': 0.5,  # 低于阈值
                'system_stability': 0.5,  # 低于阈值
                'financial_health': 0.4   # 低于阈值
            },
            'emergency_situation': True,
            'proposed_action': {
                'type': 'emergency_protocol',
                'severity': 'high'
            }
        }
        
        guidance = self.holy_code_manager.process_agent_decision_request(
            'chief_doctor', decision_context
        )
        
        assert guidance['crisis_mode']  # 应触发危机模式
        assert guidance['priority_boost'] > 0.0  # 应有优先级提升
        assert self.holy_code_manager.system_state['active_crisis']
        assert '危机应对协议' in guidance['rule_recommendations']
    
    def test_collective_decision_required(self):
        """测试需要集体决策的情况"""
        decision_context = {
            'decision_type': 'resource_allocation',
            'agent_id': 'administrator',
            'impact_scope': 'system_wide',
            'state': {
                'patient_safety': 0.8,
                'financial_health': 0.7
            },
            'proposed_action': {
                'type': 'budget_reallocation',
                'departments': ['emergency', 'surgery', 'icu']
            }
        }
        
        guidance = self.holy_code_manager.process_agent_decision_request(
            'administrator', decision_context
        )
        
        assert 'collective_approval' in guidance
        assert guidance['collective_approval'] is not None
        assert 'approved' in guidance['collective_approval']
    
    def test_reference_generation(self):
        """测试参考值生成"""
        current_state = {
            'patient_safety': 0.75,
            'medical_quality': 0.8,
            'financial_health': 0.7
        }
        
        # 测试不同类型的参考值生成
        setpoint_ref = self.holy_code_manager.reference_generator.generate_reference(
            ReferenceType.SETPOINT, current_state, 10
        )
        assert 'reference_values' in setpoint_ref
        
        trajectory_ref = self.holy_code_manager.reference_generator.generate_reference(
            ReferenceType.TRAJECTORY, current_state, 10
        )
        assert 'trajectory' in trajectory_ref
        
        adaptive_ref = self.holy_code_manager.reference_generator.generate_reference(
            ReferenceType.ADAPTIVE, current_state, 10
        )
        assert 'adaptive_targets' in adaptive_ref
    
    def test_parliament_voting_simulation(self):
        """测试议会投票模拟"""
        proposal = {
            'context': 'policy_change',
            'current_state': {
                'patient_safety': 0.7,
                'public_trust': 0.6
            },
            'proposed_action': {
                'type': 'policy_update',
                'area': 'patient_care'
            }
        }
        
        proposal_id = self.holy_code_manager.parliament.submit_proposal(
            proposal, 'test_proposer'
        )
        
        # 模拟投票
        voters = ['chief_doctor', 'doctors', 'nurses', 'administrators']
        for voter in voters:
            self.holy_code_manager.parliament.cast_vote(
                proposal_id, voter, True, f"Support from {voter}"
            )
        
        approved, approval_rate, voter_analysis = self.holy_code_manager.parliament.tally_votes(proposal_id)
        
        assert isinstance(approved, bool)
        assert 0.0 <= approval_rate <= 1.0
        assert len(voter_analysis) == len(voters)
    
    def test_crisis_mode_activation_deactivation(self):
        """测试危机模式的激活和解除"""
        # 激活危机模式
        crisis_context = {
            'state': {
                'patient_safety': 0.5,
                'system_stability': 0.4,
                'financial_health': 0.3
            },
            'emergency_situation': True
        }
        
        self.holy_code_manager._activate_crisis_mode(crisis_context)
        assert self.holy_code_manager.system_state['active_crisis']
        assert self.holy_code_manager.system_state['crisis_type'] is not None
        
        # 解除危机模式
        improved_metrics = {
            'patient_safety': 0.9,
            'system_stability': 0.85,
            'financial_health': 0.8
        }
        
        self.holy_code_manager.update_performance_metrics(improved_metrics)
        assert not self.holy_code_manager.system_state['active_crisis']
    
    def test_integration_interface(self):
        """测试与agents模块的集成接口"""
        interface = self.holy_code_manager.get_integration_interface()
        
        assert 'decision_request_handler' in interface
        assert 'status_query_handler' in interface
        assert 'performance_update_handler' in interface
        assert 'supported_decision_types' in interface
        assert 'required_context_fields' in interface
        
        # 验证接口函数可调用
        assert callable(interface['decision_request_handler'])
        assert callable(interface['status_query_handler'])
        assert callable(interface['performance_update_handler'])
    
    def test_system_status_reporting(self):
        """测试系统状态报告"""
        status = self.holy_code_manager.get_system_status()
        
        assert 'system_state' in status
        assert 'rule_engine_stats' in status
        assert 'parliament_metrics' in status
        assert 'reference_generator_status' in status
        assert 'total_decisions' in status
        assert 'crisis_decisions' in status
    
    def test_performance_tracking(self):
        """测试性能跟踪"""
        initial_decisions = len(self.holy_code_manager.performance_history)
        
        # 执行一些决策
        for i in range(3):
            decision_context = {
                'decision_type': 'routine_operation',
                'agent_id': f'agent_{i}',
                'state': {'patient_safety': 0.8}
            }
            
            self.holy_code_manager.process_agent_decision_request(
                f'agent_{i}', decision_context
            )
        
        final_decisions = len(self.holy_code_manager.performance_history)
        assert final_decisions == initial_decisions + 3
        
        # 验证决策记录结构
        if self.holy_code_manager.performance_history:
            record = self.holy_code_manager.performance_history[-1]
            assert 'timestamp' in record
            assert 'agent_id' in record
            assert 'decision_type' in record
            assert 'guidance' in record
            assert 'crisis_mode' in record

def test_holy_code_refactoring():
    """主测试函数"""
    test_suite = TestHolyCodeRefactoring()
    test_suite.setup_method()
    
    try:
        print("开始测试Holy Code重构...")
        
        print("✓ 测试管理器初始化")
        test_suite.test_manager_initialization()
        
        print("✓ 测试规则去重")
        test_suite.test_rule_deduplication()
        
        print("✓ 测试正常决策请求")
        test_suite.test_agent_decision_request_normal()
        
        print("✓ 测试危机决策请求")
        test_suite.test_agent_decision_request_crisis()
        
        print("✓ 测试集体决策")
        test_suite.test_collective_decision_required()
        
        print("✓ 测试参考值生成")
        test_suite.test_reference_generation()
        
        print("✓ 测试议会投票")
        test_suite.test_parliament_voting_simulation()
        
        print("✓ 测试危机模式")
        test_suite.test_crisis_mode_activation_deactivation()
        
        print("✓ 测试集成接口")
        test_suite.test_integration_interface()
        
        print("✓ 测试状态报告")
        test_suite.test_system_status_reporting()
        
        print("✓ 测试性能跟踪")
        test_suite.test_performance_tracking()
        
        print("\n🎉 所有Holy Code重构测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_holy_code_refactoring()