"""
行为模型组件的基本验证测试
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.hospital_governance.agents.behavior_models import (
        BehaviorType, BehaviorParameters, BehaviorModelFactory, 
        BehaviorModelManager, BaseBehaviorModel
    )
    print("✓ 成功导入所有行为模型组件")
    
    # 测试基本功能
    params = BehaviorParameters()
    print(f"✓ 创建行为参数: {params}")
    
    # 测试工厂模式
    rational_model = BehaviorModelFactory.create_behavior_model(
        BehaviorType.RATIONAL, params
    )
    print(f"✓ 创建理性行为模型: {rational_model.behavior_type}")
    
    # 测试角色特定模型
    doctor_model = BehaviorModelFactory.create_role_specific_model('doctors')
    print(f"✓ 创建医生行为模型: {doctor_model.behavior_type}")
    
    # 测试管理器
    manager = BehaviorModelManager()
    manager.create_all_role_models()
    print(f"✓ 创建行为模型管理器，包含 {len(manager.models)} 个角色模型")
    
    print("\n🎉 行为模型组件验证通过！所有核心功能正常工作。")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
except Exception as e:
    print(f"❌ 运行错误: {e}")
    import traceback
    traceback.print_exc()