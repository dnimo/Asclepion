#!/usr/bin/env python3
"""
医院治理系统 - 真实LLM API配置指南
帮助用户配置OpenAI或Anthropic API进行完整仿真
"""

import os
import asyncio
import json

def setup_api_keys():
    """交互式API密钥配置"""
    print("🔧 LLM API配置向导")
    print("=" * 50)
    
    print("支持的LLM提供者:")
    print("1. OpenAI GPT (推荐: gpt-4, gpt-3.5-turbo)")
    print("2. Anthropic Claude (推荐: claude-3-sonnet)")
    print("3. 模拟LLM (无需API，用于测试)")
    
    choice = input("\\n请选择 (1-3): ").strip()
    
    if choice == '1':
        return setup_openai()
    elif choice == '2':
        return setup_anthropic()
    elif choice == '3':
        print("✅ 将使用模拟LLM进行仿真")
        return 'mock', None
    else:
        print("❌ 无效选择")
        return None, None

def setup_openai():
    """配置OpenAI API"""
    print("\\n🔑 OpenAI API配置")
    print("-" * 30)
    print("1. 访问 https://platform.openai.com/api-keys")
    print("2. 创建新的API密钥")
    print("3. 将密钥粘贴到下方")
    
    api_key = input("\\n请输入OpenAI API密钥: ").strip()
    
    if not api_key:
        print("❌ 未输入API密钥")
        return None, None
    
    # 验证API密钥格式
    if not api_key.startswith('sk-'):
        print("⚠️ 警告: OpenAI API密钥通常以 'sk-' 开头")
    
    model = input("请输入模型名称 (默认: gpt-4): ").strip() or 'gpt-4'
    
    print(f"\\n✅ OpenAI配置完成")
    print(f"   模型: {model}")
    print(f"   API密钥: {api_key[:8]}...")
    
    return 'openai', {'api_key': api_key, 'model': model}

def setup_anthropic():
    """配置Anthropic API"""
    print("\\n🔑 Anthropic API配置")
    print("-" * 30)
    print("1. 访问 https://console.anthropic.com/")
    print("2. 创建新的API密钥")
    print("3. 将密钥粘贴到下方")
    
    api_key = input("\\n请输入Anthropic API密钥: ").strip()
    
    if not api_key:
        print("❌ 未输入API密钥")
        return None, None
    
    model = input("请输入模型名称 (默认: claude-3-sonnet-20240229): ").strip() or 'claude-3-sonnet-20240229'
    
    print(f"\\n✅ Anthropic配置完成")
    print(f"   模型: {model}")
    print(f"   API密钥: {api_key[:8]}...")
    
    return 'anthropic', {'api_key': api_key, 'model': model}

async def test_api_connection(provider, config):
    """测试API连接"""
    if provider == 'mock':
        print("🧪 模拟LLM无需测试连接")
        return True
    
    print(f"\\n🧪 测试 {provider} API连接...")
    
    try:
        if provider == 'openai':
            import httpx
            
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": config['model'],
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    print("✅ OpenAI API连接成功")
                    return True
                else:
                    print(f"❌ OpenAI API错误: {response.status_code}")
                    print(f"   错误信息: {response.text[:200]}")
                    return False
                    
        elif provider == 'anthropic':
            import httpx
            
            headers = {
                "x-api-key": config['api_key'],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": config['model'],
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}]
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    print("✅ Anthropic API连接成功")
                    return True
                else:
                    print(f"❌ Anthropic API错误: {response.status_code}")
                    print(f"   错误信息: {response.text[:200]}")
                    return False
                    
    except ImportError:
        print("❌ 缺少httpx依赖，请运行: pip install httpx")
        return False
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

def save_config(provider, config):
    """保存配置到环境文件"""
    env_file = '.env'
    
    config_lines = []
    if provider == 'openai':
        config_lines.append(f"OPENAI_API_KEY={config['api_key']}")
        config_lines.append(f"DEFAULT_MODEL={config['model']}")
    elif provider == 'anthropic':
        config_lines.append(f"ANTHROPIC_API_KEY={config['api_key']}")
        config_lines.append(f"DEFAULT_MODEL={config['model']}")
    
    if config_lines:
        with open(env_file, 'w') as f:
            f.write("# 医院治理系统LLM API配置\\n")
            f.write("# 自动生成\\n\\n")
            for line in config_lines:
                f.write(line + "\\n")
        
        print(f"\\n💾 配置已保存到 {env_file}")
        print("   下次运行时将自动加载此配置")

async def run_test_simulation(provider, config):
    """运行测试仿真"""
    print("\\n🚀 运行测试仿真...")
    
    # 设置环境变量
    if provider == 'openai' and config:
        os.environ['OPENAI_API_KEY'] = config['api_key']
    elif provider == 'anthropic' and config:
        os.environ['ANTHROPIC_API_KEY'] = config['api_key']
    
    # 动态导入仿真模块
    try:
        from hospital_simulation_llm import AdvancedHospitalSimulation
        
        # 创建短期测试仿真
        simulation = AdvancedHospitalSimulation(
            llm_provider=provider,
            api_key=config['api_key'] if config else None,
            duration=3  # 只运行3步进行测试
        )
        
        # 运行仿真
        summary = await simulation.run_simulation_async()
        
        print("\\n✅ 测试仿真成功！")
        print(f"   稳定性: {summary['final_stability']:.3f}")
        print(f"   规则激活: {summary['total_rule_activations']} 次")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ 测试仿真失败: {e}")
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\\n📚 使用示例:")
    print("=" * 50)
    
    print("1. 运行简化仿真 (无需API):")
    print("   python3 hospital_simulation_simple.py")
    
    print("\\n2. 运行高级LLM仿真:")
    print("   python3 hospital_simulation_llm.py")
    
    print("\\n3. 完整功能仿真 (需要matplotlib):")
    print("   python3 hospital_simulation_complete.py")
    
    print("\\n4. 环境变量方式配置:")
    print("   export OPENAI_API_KEY='your-key-here'")
    print("   python3 hospital_simulation_llm.py")
    
    print("\\n📊 仿真特性对比:")
    print("   简化版: 基础多智能体决策，无外部依赖")
    print("   LLM版: 真实LLM驱动决策，支持API调用")
    print("   完整版: 图表可视化，数据导出，完整分析")

async def main():
    """主函数"""
    print("🏥 医院治理系统 - LLM API配置与测试")
    print("=" * 60)
    
    # 检查现有配置
    current_openai = os.getenv('OPENAI_API_KEY')
    current_anthropic = os.getenv('ANTHROPIC_API_KEY')
    
    print("\\n当前API状态:")
    print(f"  OpenAI: {'✅ 已配置' if current_openai else '❌ 未配置'}")
    print(f"  Anthropic: {'✅ 已配置' if current_anthropic else '❌ 未配置'}")
    
    if current_openai or current_anthropic:
        use_existing = input("\\n是否使用现有配置? (y/n): ").strip().lower()
        if use_existing == 'y':
            if current_openai:
                provider, config = 'openai', {'api_key': current_openai, 'model': 'gpt-4'}
            else:
                provider, config = 'anthropic', {'api_key': current_anthropic, 'model': 'claude-3-sonnet-20240229'}
        else:
            provider, config = setup_api_keys()
    else:
        provider, config = setup_api_keys()
    
    if not provider:
        print("❌ 配置失败")
        return
    
    # 测试API连接（如果需要）
    if provider != 'mock':
        connection_ok = await test_api_connection(provider, config)
        if not connection_ok:
            print("⚠️ API连接测试失败，但仍可运行模拟版本")
    
    # 询问是否保存配置
    if provider != 'mock' and config:
        save_choice = input("\\n是否保存配置到 .env 文件? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_config(provider, config)
    
    # 询问是否运行测试仿真
    test_choice = input("\\n是否运行测试仿真? (y/n): ").strip().lower()
    if test_choice == 'y':
        await run_test_simulation(provider, config)
    
    # 显示使用指南
    show_usage_examples()
    
    print("\\n🎉 配置完成！")
    print("现在可以运行完整的医院治理系统仿真了")

if __name__ == '__main__':
    asyncio.run(main())