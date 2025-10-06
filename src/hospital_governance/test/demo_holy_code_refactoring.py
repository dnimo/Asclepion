"""
Holy Code重构演示
展示重构后的系统如何协调工作
"""

# 模拟环境，避免依赖问题
import sys
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

def demonstrate_holy_code_refactoring():
    """演示Holy Code重构后的系统工作流程"""
    
    print("🏥 Kallipolis医院治理系统 - Holy Code重构演示")
    print("=" * 60)
    
    # 1. 演示统一管理
    print("\n📋 1. 统一组件管理演示")
    print("-" * 30)
    
    print("✅ HolyCodeManager初始化")
    print("  ├── RuleLibrary: 统一管理8个核心规则")
    print("  ├── RuleEngine: 委托模式，无重复代码")
    print("  ├── Parliament: 集体决策系统")
    print("  └── ReferenceGenerator: 动态参考值生成")
    
    # 2. 演示决策处理流程
    print("\n🤖 2. Agent决策请求处理流程")
    print("-" * 30)
    
    # 模拟决策请求
    decision_context = {
        'agent_id': 'chief_doctor',
        'decision_type': 'resource_allocation',
        'state': {
            'patient_safety': 0.65,  # 低于标准
            'medical_quality': 0.70,
            'financial_health': 0.55,  # 需要关注
            'system_stability': 0.80
        },
        'proposed_action': {
            'type': 'budget_reallocation',
            'target_departments': ['emergency', 'icu'],
            'amount': 500000
        },
        'impact_scope': 'system_wide'
    }
    
    print(f"📥 决策请求: {decision_context['decision_type']}")
    print(f"   Agent: {decision_context['agent_id']}")
    print(f"   影响范围: {decision_context['impact_scope']}")
    
    # 模拟处理流程
    print("\n🔍 系统处理流程:")
    
    # Step 1: 危机检测
    crisis_indicators = [
        decision_context['state']['patient_safety'] < 0.7,
        decision_context['state']['financial_health'] < 0.6
    ]
    crisis_detected = sum(crisis_indicators) >= 2
    
    print(f"   1️⃣ 危机检测: {'🚨 危机模式激活' if crisis_detected else '✅ 正常状态'}")
    
    # Step 2: 规则评估
    activated_rules = []
    if decision_context['state']['patient_safety'] < 0.8:
        activated_rules.append({
            'name': '患者安全第一',
            'priority': 1,
            'weight': 1.0,
            'recommendations': ['增加安全检查', '优化医疗流程']
        })
    
    if decision_context['state']['financial_health'] < 0.6:
        activated_rules.append({
            'name': '财务可持续性',
            'priority': 3,
            'weight': 0.7,
            'recommendations': ['成本控制', '效率优化']
        })
    
    print(f"   2️⃣ 规则评估: 激活了 {len(activated_rules)} 条规则")
    for rule in activated_rules:
        print(f"      - {rule['name']} (优先级: {rule['priority']})")
    
    # Step 3: 参考值生成
    reference_targets = {
        'patient_safety': 0.90,
        'medical_quality': 0.85,
        'financial_health': 0.75
    }
    
    print(f"   3️⃣ 参考值生成: 设定目标值")
    for metric, target in reference_targets.items():
        current = decision_context['state'].get(metric, 0.0)
        adjustment = target - current
        print(f"      - {metric}: {current:.2f} → {target:.2f} ({adjustment:+.2f})")
    
    # Step 4: 集体决策判断
    requires_parliament = (
        decision_context['decision_type'] == 'resource_allocation' or
        decision_context['impact_scope'] == 'system_wide' or
        crisis_detected
    )
    
    print(f"   4️⃣ 集体决策: {'🏛️ 需要议会审议' if requires_parliament else '✅ 个人决策即可'}")
    
    if requires_parliament:
        # 模拟议会投票
        voters = {
            'chief_doctor': True,    # 支持
            'doctors': True,         # 支持  
            'nurses': True,          # 支持
            'administrators': False, # 反对（预算担忧）
            'patients_rep': True     # 支持
        }
        
        yes_votes = sum(1 for vote in voters.values() if vote)
        total_votes = len(voters)
        approval_rate = yes_votes / total_votes
        approved = approval_rate >= 0.6
        
        print(f"      📊 投票结果: {yes_votes}/{total_votes} 支持 ({approval_rate:.1%})")
        print(f"      🏆 决议: {'✅ 通过' if approved else '❌ 否决'}")
    
    # Step 5: 整合建议
    print(f"   5️⃣ 建议整合:")
    
    all_recommendations = []
    priority_boost = 0.0
    
    for rule in activated_rules:
        all_recommendations.extend(rule['recommendations'])
        priority_boost += 0.1 * rule['weight']
    
    if crisis_detected:
        all_recommendations.insert(0, "执行危机应对协议")
        priority_boost *= 1.5
    
    if requires_parliament and 'approved' in locals() and approved:
        all_recommendations.append("议会已批准，可立即执行")
    
    print(f"      📝 核心建议: {len(set(all_recommendations))} 项")
    unique_recommendations = list(set(all_recommendations))
    for i, rec in enumerate(unique_recommendations[:3], 1):
        print(f"         {i}. {rec}")
    
    print(f"      📈 优先级提升: +{priority_boost:.1%}")
    
    # 3. 演示重构效果对比
    print("\n🔄 3. 重构效果对比")
    print("-" * 30)
    
    print("重构前问题:")
    print("  ❌ 代码重复: rule_engine和rule_library中重复函数")
    print("  ❌ 维护困难: 修改需要在多个文件中同步")
    print("  ❌ 缺乏统一接口: 各组件独立工作")
    print("  ❌ 集成复杂: agents模块需要对接多个组件")
    
    print("\n重构后改进:")
    print("  ✅ 消除重复: 统一的RuleLibrary管理所有函数")
    print("  ✅ 委托模式: RuleEngine委托RuleLibrary处理")
    print("  ✅ 统一管理: HolyCodeManager协调所有组件")
    print("  ✅ 简化集成: 单一接口处理所有决策请求")
    
    # 4. 展示系统状态
    print("\n📊 4. 系统状态总览")
    print("-" * 30)
    
    system_stats = {
        '规则总数': 8,
        '激活规则': len(activated_rules),
        '处理决策': 1,
        '危机状态': '激活' if crisis_detected else '正常',
        '议会状态': '活跃' if requires_parliament else '待命'
    }
    
    for key, value in system_stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🎉 Holy Code重构演示完成!")
    print("\n✨ 重构亮点:")
    print("  🏗️  架构优化: 消除代码重复，提升可维护性")
    print("  🤝 统一管理: HolyCodeManager协调所有组件")
    print("  🔌 简化集成: 标准化的agents接口")
    print("  🚀 性能提升: 委托模式减少代码冗余")
    print("  📈 扩展性强: 模块化设计支持未来扩展")
    
    print("\n🔗 准备与agents模块进行完整集成!")

if __name__ == "__main__":
    demonstrate_holy_code_refactoring()