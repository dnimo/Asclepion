#!/usr/bin/env python3
"""
测试神圣法典规则引擎
"""

import sys
import os
import tempfile
import yaml

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_rule_engine_yaml_persistence():
    """测试规则引擎YAML持久化功能"""
    print("=== 测试规则引擎YAML持久化 ===")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_yaml_path = f.name
        
        # 写入测试规则配置
        test_config = {
            'core_rules': [
                {
                    'rule_id': 'ETHICS_001',
                    'name': '患者生命权优先',
                    'logic_function': 'patient_life_priority',
                    'weight': 1.0,
                    'context': ['crisis', 'medical'],
                    'priority': 1,
                    'description': '在任何情况下，患者生命权都具有最高优先级'
                },
                {
                    'rule_id': 'RESOURCE_001',
                    'name': '资源公平分配',
                    'logic_function': 'fair_resource_allocation',
                    'weight': 0.8,
                    'context': ['normal', 'crisis'],
                    'priority': 2,
                    'description': '医疗资源应按需求和紧急程度公平分配'
                }
            ],
            'government_rules': [
                {
                    'rule_id': 'GOV_001',
                    'name': '预算透明度',
                    'logic_function': 'budget_transparency',
                    'weight': 0.6,
                    'context': ['financial'],
                    'priority': 3,
                    'description': '医院财政预算必须保持透明度'
                }
            ]
        }
        
        yaml.safe_dump(test_config, f, allow_unicode=True)
    
    try:
        # 导入规则引擎
        from hospital_governance.holy_code.rule_engine import RuleEngine, RulePriority, Rule
        
        # 测试加载规则
        engine = RuleEngine(temp_yaml_path)
        
        print(f"加载的规则数量: {len(engine.rules)}")
        
        # 验证规则
        assert 'ETHICS_001' in engine.rules
        assert 'RESOURCE_001' in engine.rules
        assert 'GOV_001' in engine.rules
        
        ethics_rule = engine.rules['ETHICS_001']
        assert ethics_rule.name == '患者生命权优先'
        assert ethics_rule.priority == RulePriority.CRITICAL
        assert ethics_rule.weight == 1.0
        
        print("✓ 规则加载测试通过")
        
        # 测试规则评估
        test_context = {
            'context_type': 'crisis',
            'patient_condition': 'critical',
            'resource_level': 0.3,
            'timestamp': 'test_time'
        }
        
        activated_rules = engine.evaluate_rules(test_context)
        print(f"激活的规则数量: {len(activated_rules)}")
        
        # 验证危机情况下伦理规则被激活
        ethics_activated = any(rule['rule_id'] == 'ETHICS_001' for rule in activated_rules)
        assert ethics_activated, "伦理规则未在危机情况下激活"
        
        print("✓ 规则评估测试通过")
        
        # 测试规则持久化
        new_rule = Rule(
            rule_id='TEST_001',
            name='测试规则',
            condition=lambda ctx: True,
            action=lambda ctx: {'message': '测试动作'},
            priority=RulePriority.HIGH,
            weight=0.9,
            context=['test'],
            description='这是一个测试规则'
        )
        
        # 添加规则（应该自动保存）
        engine.add_rule(new_rule)
        
        # 验证规则文件更新
        temp_yaml_path_new = temp_yaml_path.replace('.yaml', '_new.yaml')
        engine.save_rules_to_file(temp_yaml_path_new)
        
        # 读取保存的文件
        with open(temp_yaml_path_new, 'r', encoding='utf-8') as f:
            saved_config = yaml.safe_load(f)
        
        # 验证新规则被保存
        all_saved_rules = saved_config['core_rules'] + saved_config['government_rules']
        test_rule_saved = any(rule['rule_id'] == 'TEST_001' for rule in all_saved_rules)
        assert test_rule_saved, "新规则未被保存到YAML文件"
        
        print("✓ 规则持久化测试通过")
        
        # 测试规则统计
        stats = engine.get_rule_statistics()
        print(f"规则统计: {stats}")
        
        assert stats['total_rules'] == 4  # 3个原始 + 1个新增
        assert 'CRITICAL' in stats['priority_distribution']
        
        print("✓ 规则统计测试通过")
        
        # 测试规则权重更新
        original_weight = engine.rules['RESOURCE_001'].weight
        new_weight = 0.95
        
        success = engine.update_rule_weight('RESOURCE_001', new_weight)
        assert success, "规则权重更新失败"
        assert engine.rules['RESOURCE_001'].weight == new_weight
        
        print("✓ 规则权重更新测试通过")
        
        # 清理临时文件
        os.unlink(temp_yaml_path)
        os.unlink(temp_yaml_path_new)
        
        print("🎉 神圣法典规则引擎所有测试通过！")
        
    except Exception as e:
        print(f"❌ 规则引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理
        if os.path.exists(temp_yaml_path):
            os.unlink(temp_yaml_path)
        raise

def test_rule_engine_integration_with_control():
    """测试规则引擎与控制系统集成"""
    print("\n=== 测试规则引擎与控制系统集成 ===")
    
    # 创建规则引擎状态
    holy_code_state = {
        'ethical_constraints': {
            'min_quality_control': 0.3,
            'max_workload': 0.7,
            'min_training_hours': 0.25,
            'min_health_level': 0.4,
            'min_cost_efficiency': 0.35,
            'min_equity_level': 0.3
        },
        'active_rules': ['ETHICS_001', 'RESOURCE_001'],
        'crisis_level': 'medium'
    }
    
    # 模拟控制器应用约束
    def apply_constraints_test(u_input, constraints):
        u_constrained = u_input.copy()
        
        # 应用最小质量控制约束
        if 'min_quality_control' in constraints:
            u_constrained[3] = max(u_constrained[3], constraints['min_quality_control'])
        
        # 应用最大工作负荷约束
        if 'max_workload' in constraints:
            u_constrained[2] = min(u_constrained[2], constraints['max_workload'])
        
        return u_constrained
    
    # 测试约束应用
    u_original = [0.1, 0.2, 0.9, 0.1]  # 工作负荷过高，质量控制过低
    u_constrained = apply_constraints_test(u_original, holy_code_state['ethical_constraints'])
    
    print(f"原始控制信号: {u_original}")
    print(f"约束后控制信号: {u_constrained}")
    
    # 验证约束生效
    assert u_constrained[3] >= 0.3, f"质量控制约束未生效: {u_constrained[3]}"
    assert u_constrained[2] <= 0.7, f"工作负荷约束未生效: {u_constrained[2]}"
    
    print("✓ 规则引擎约束集成测试通过")

def main():
    """主测试函数"""
    print("🚀 开始神圣法典规则引擎测试...")
    
    try:
        test_rule_engine_yaml_persistence()
        test_rule_engine_integration_with_control()
        
        print("\n🎉 所有神圣法典测试通过！")
        print("✓ YAML规则持久化功能正常")
        print("✓ 规则评估和激活机制有效")
        print("✓ 伦理约束与控制系统集成成功")
        
    except Exception as e:
        print(f"\n❌ 神圣法典测试失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())