"""
调试Mock LLM响应
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.hospital_governance.agents.llm_action_generator import MockLLMProvider, LLMConfig

def debug_mock_response():
    print("🔍 调试Mock LLM响应")
    print("=" * 50)
    
    # 创建Mock提供者
    config = LLMConfig(model_name="mock")
    mock_provider = MockLLMProvider(config)
    
    # 模拟不同的提示
    test_prompts = [
        "作为doctors角色，基于当前观测[0.5, 0.6, 0.4, 0.7]，考虑神圣法典规则，建议采取的行动：",
        "提高医疗质量标准",
        "申请更多资源", 
        "基于当前情况考虑最佳决策"
    ]
    
    contexts = [
        {'role': 'doctors'},
        {'role': 'interns'},
        {'role': 'patients'}
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\\n🧪 测试提示 {i+1}: {prompt[:50]}...")
        for context in contexts:
            response = mock_provider.generate_text_sync(prompt, context)
            print(f"  角色 {context['role']}: {response}")

if __name__ == "__main__":
    debug_mock_response()