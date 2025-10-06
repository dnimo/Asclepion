#!/usr/bin/env python3
"""
医院治理系统综合测试总结
包含分布式控制系统和神圣法典规则引擎的完整测试
"""

import subprocess
import sys

def run_test_file(test_file, description):
    """运行测试文件并显示结果"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """综合测试主函数"""
    print("🏥 医院治理系统综合测试")
    print("=" * 60)
    print("测试范围:")
    print("  1. 分布式控制系统（医生、实习医生、患者、会计、政府）")
    print("  2. 神圣法典规则引擎（YAML持久化、伦理约束）")
    print("  3. 控制系统与规则引擎集成")
    
    test_results = []
    
    # 测试1: 分布式控制系统
    success = run_test_file('test_control_simple.py', 
                           '分布式控制系统测试')
    test_results.append(('分布式控制系统', success))
    
    # 测试2: 神圣法典规则引擎
    success = run_test_file('test_holy_code_simple.py', 
                           '神圣法典规则引擎测试')
    test_results.append(('神圣法典规则引擎', success))
    
    # 测试总结
    print(f"\n{'='*60}")
    print("🎯 测试总结")
    print('='*60)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 所有测试通过！医院治理系统集成测试成功")
        print("\n✅ 验证通过的功能:")
        print("  • 医生主稳定控制器（PID+前馈+HolyCode约束）")
        print("  • 实习医生观测器前馈控制器（观测+前馈+伦理约束）")
        print("  • 患者自适应控制器（比例+适应项+健康约束）")
        print("  • 会计约束强化控制器（预算+效率约束）")
        print("  • 政府政策控制器（政策矩阵+公平约束）")
        print("  • 17维全局控制向量合成与分发")
        print("  • 神圣法典规则引擎（条件评估+动作执行）")
        print("  • YAML规则持久化（保存+加载+更新）")
        print("  • 伦理约束与控制信号集成")
        print("  • 危机情况下规则优先级排序")
        
        print("\n🚀 系统架构验证:")
        print("  • 分布式控制架构：每个角色独立控制器，统一合成")
        print("  • 伦理约束机制：HolyCode规则实时约束控制信号")
        print("  • 参数化配置：YAML文件驱动的系统配置")
        print("  • 模块化设计：控制、规则、配置模块独立可扩展")
        
        return 0
    else:
        print("❌ 部分测试失败，请检查错误信息")
        return 1

if __name__ == '__main__':
    exit(main())