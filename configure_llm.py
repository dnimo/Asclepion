#!/usr/bin/env python3
"""
LLM API配置和测试工具
支持配置和测试不同的LLM提供者
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional

def load_env_config():
    """加载环境配置"""
    config = {}
    
    # 尝试从.env文件加载
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    
    # 从环境变量覆盖
    config.update(os.environ)
    
    return config

def create_llm_config(provider: str, model: str = None, api_key: str = None) -> Dict[str, Any]:
    """创建LLM配置"""
    env_config = load_env_config()
    
    configs = {
        'openai': {
            'provider_type': 'openai',
            'model_name': model or env_config.get('DEFAULT_MODEL', 'gpt-4'),
            'api_key': api_key or env_config.get('OPENAI_API_KEY'),
            'base_url': env_config.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'temperature': 0.7,
            'max_tokens': 800,
            'timeout': 30.0
        },
        'anthropic': {
            'provider_type': 'anthropic',
            'model_name': model or 'claude-3-sonnet-20240229',
            'api_key': api_key or env_config.get('ANTHROPIC_API_KEY'),
            'base_url': env_config.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com/v1'),
            'temperature': 0.7,
            'max_tokens': 800,
            'timeout': 30.0
        },
        'local': {
            'provider_type': 'local',
            'model_name': model or 'llama2:7b',
            'base_url': env_config.get('LOCAL_MODEL_URL', 'http://localhost:11434'),
            'temperature': 0.7,
            'max_tokens': 800,
            'timeout': 60.0
        },
        'mock': {
            'provider_type': 'mock',
            'model_name': 'mock-model',
            'temperature': 0.0,
            'max_tokens': 500,
            'timeout': 1.0
        }
    }
    
    if provider not in configs:
        raise ValueError(f"Unsupported provider: {provider}. Available: {list(configs.keys())}")
    
    return configs[provider]

async def test_llm_provider(provider: str, model: str = None, api_key: str = None):
    """测试LLM提供者"""
    print(f"🧪 测试 {provider} LLM提供者...")
    
    try:
        config = create_llm_config(provider, model, api_key)
        
        if provider == 'mock':
            # 测试模拟提供者
            from hospital_simulation_complete import SimpleLLMProvider
            llm = SimpleLLMProvider('mock')
            
            test_obs = [0.3, 0.7, 0.2, 0.8]
            test_constraints = {'min_quality_control': 0.5}
            
            result = llm.generate_action('doctors', test_obs, test_constraints)
            print(f"✅ 模拟LLM测试成功")
            print(f"   输入观测: {test_obs}")
            print(f"   输出动作: {result.tolist()}")
            return True
            
        elif provider == 'openai':
            # 测试OpenAI API
            if not config['api_key'] or config['api_key'] == 'your_openai_api_key_here':
                print("❌ OpenAI API密钥未配置")
                return False
            
            import httpx
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": config['model_name'],
                "messages": [
                    {"role": "user", "content": "请返回一个简单的测试响应"}
                ],
                "max_tokens": 50
            }
            
            async with httpx.AsyncClient(timeout=config['timeout']) as client:
                response = await client.post(
                    f"{config['base_url']}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ OpenAI API测试成功")
                    print(f"   模型: {config['model_name']}")
                    print(f"   响应: {result['choices'][0]['message']['content'][:100]}...")
                    return True
                else:
                    print(f"❌ OpenAI API测试失败: {response.status_code}")
                    print(f"   错误: {response.text}")
                    return False
        
        elif provider == 'anthropic':
            # 测试Anthropic API
            if not config['api_key'] or config['api_key'] == 'your_anthropic_api_key_here':
                print("❌ Anthropic API密钥未配置")
                return False
            
            import httpx
            headers = {
                "x-api-key": config['api_key'],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": config['model_name'],
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "请返回一个简单的测试响应"}
                ]
            }
            
            async with httpx.AsyncClient(timeout=config['timeout']) as client:
                response = await client.post(
                    f"{config['base_url']}/messages",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Anthropic API测试成功")
                    print(f"   模型: {config['model_name']}")
                    print(f"   响应: {result['content'][0]['text'][:100]}...")
                    return True
                else:
                    print(f"❌ Anthropic API测试失败: {response.status_code}")
                    print(f"   错误: {response.text}")
                    return False
        
        elif provider == 'local':
            # 测试本地模型
            import httpx
            payload = {
                "model": config['model_name'],
                "prompt": "请返回一个简单的测试响应",
                "options": {"num_predict": 50}
            }
            
            try:
                async with httpx.AsyncClient(timeout=config['timeout']) as client:
                    response = await client.post(
                        f"{config['base_url']}/api/generate",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print("✅ 本地模型测试成功")
                        print(f"   模型: {config['model_name']}")
                        print(f"   响应: {result['response'][:100]}...")
                        return True
                    else:
                        print(f"❌ 本地模型测试失败: {response.status_code}")
                        return False
            except httpx.ConnectError:
                print("❌ 无法连接到本地模型服务")
                print("   请确保Ollama等本地服务正在运行")
                return False
        
    except Exception as e:
        print(f"❌ {provider} 测试失败: {e}")
        return False

def interactive_setup():
    """交互式设置LLM API"""
    print("🔧 医院治理系统LLM配置向导")
    print("=" * 50)
    
    providers = ['mock', 'openai', 'anthropic', 'local']
    
    print("可用的LLM提供者:")
    for i, provider in enumerate(providers, 1):
        descriptions = {
            'mock': '模拟LLM（无需API，用于测试）',
            'openai': 'OpenAI GPT模型（需要API密钥）',
            'anthropic': 'Anthropic Claude模型（需要API密钥）',
            'local': '本地模型（如Ollama，需要本地服务）'
        }
        print(f"  {i}. {provider} - {descriptions[provider]}")
    
    while True:
        try:
            choice = input("\\n请选择LLM提供者 (1-4): ").strip()
            provider_idx = int(choice) - 1
            if 0 <= provider_idx < len(providers):
                selected_provider = providers[provider_idx]
                break
            else:
                print("无效选择，请输入1-4之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    print(f"\\n选择的提供者: {selected_provider}")
    
    # 根据选择的提供者获取配置
    api_key = None
    model = None
    
    if selected_provider == 'openai':
        api_key = input("请输入OpenAI API密钥 (或按回车使用环境变量): ").strip()
        model = input("请输入模型名称 (默认: gpt-4): ").strip() or 'gpt-4'
    elif selected_provider == 'anthropic':
        api_key = input("请输入Anthropic API密钥 (或按回车使用环境变量): ").strip()
        model = input("请输入模型名称 (默认: claude-3-sonnet-20240229): ").strip() or 'claude-3-sonnet-20240229'
    elif selected_provider == 'local':
        model = input("请输入本地模型名称 (默认: llama2:7b): ").strip() or 'llama2:7b'
    
    return selected_provider, model, api_key or None

async def main():
    """主函数"""
    print("🏥 医院治理系统LLM API配置和测试")
    print("=" * 60)
    
    # 检查现有配置
    env_config = load_env_config()
    print("\\n当前环境配置:")
    for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'LOCAL_MODEL_URL']:
        value = env_config.get(key, '未设置')
        if 'API_KEY' in key and value and value != '未设置':
            value = value[:8] + '***' if len(value) > 8 else '***'
        print(f"  {key}: {value}")
    
    # 测试所有可用提供者
    print("\\n🧪 测试所有LLM提供者:")
    print("-" * 30)
    
    test_results = {}
    
    # 测试Mock（总是可用）
    test_results['mock'] = await test_llm_provider('mock')
    
    # 测试其他提供者
    for provider in ['openai', 'anthropic', 'local']:
        test_results[provider] = await test_llm_provider(provider)
    
    # 显示结果摘要
    print("\\n📋 测试结果摘要:")
    print("-" * 30)
    available_providers = []
    for provider, success in test_results.items():
        status = "✅ 可用" if success else "❌ 不可用"
        print(f"  {provider}: {status}")
        if success:
            available_providers.append(provider)
    
    print(f"\\n可用提供者: {', '.join(available_providers)}")
    
    # 交互式配置（可选）
    setup_choice = input("\\n是否需要交互式配置? (y/n): ").strip().lower()
    if setup_choice == 'y':
        provider, model, api_key = interactive_setup()
        print(f"\\n配置完成: {provider}")
        if model:
            print(f"模型: {model}")
        
        # 测试新配置
        print("\\n测试新配置...")
        success = await test_llm_provider(provider, model, api_key)
        if success:
            print("✅ 配置测试成功！可以开始仿真了")
            
            # 保存配置示例
            config_example = create_llm_config(provider, model, api_key)
            print("\\n示例配置:")
            print(json.dumps(config_example, indent=2))
        else:
            print("❌ 配置测试失败，请检查设置")
    
    print("\\n🚀 LLM配置完成！")
    print("现在可以运行 'python hospital_simulation_complete.py' 开始仿真")

if __name__ == '__main__':
    asyncio.run(main())