"""
环境变量配置示例
Environment Variables Configuration Example

展示如何使用环境变量管理API keys和配置智能体注册中心
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.hospital_governance.agents import (
    AgentRegistry, AgentRegistryConfig, LLMProviderType,
    create_agent_registry, get_global_agent_registry
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_environment_variables_example():
    """设置环境变量示例"""
    print("🔧 环境变量配置示例")
    print("=" * 50)
    
    # 示例环境变量设置
    example_env_vars = {
        'OPENAI_API_KEY': 'sk-your-openai-api-key-here',
        'ANTHROPIC_API_KEY': 'your-anthropic-api-key-here',
        'HOSPITAL_LLM_PROVIDER': 'mock',  # 可选: openai, anthropic, local, mock
        'HOSPITAL_LLM_PRESET': 'mock',    # 可选: openai_gpt4, anthropic_claude, mock
        'HOSPITAL_ENABLE_LLM': 'true',    # 是否启用LLM
        'HOSPITAL_FALLBACK_MOCK': 'true'  # API失败时是否回退到mock
    }
    
    print("📝 推荐的环境变量配置:")
    for var, value in example_env_vars.items():
        if 'api-key' in value:
            print(f"export {var}=\"{value}\"")
        else:
            print(f"export {var}={value}")
    
    print("\\n💡 使用方式:")
    print("1. 将上述环境变量添加到 ~/.bashrc 或 ~/.zshrc")
    print("2. 重新启动终端或执行 source ~/.bashrc")
    print("3. 运行 python3 env_config_example.py test")

def test_agent_registry_with_env():
    """测试环境变量驱动的智能体注册"""
    print("\\n🧪 测试智能体注册中心")
    print("=" * 50)
    
    # 从环境变量读取配置
    llm_provider = os.getenv('HOSPITAL_LLM_PROVIDER', 'mock')
    enable_llm = os.getenv('HOSPITAL_ENABLE_LLM', 'true').lower() == 'true'
    fallback_mock = os.getenv('HOSPITAL_FALLBACK_MOCK', 'true').lower() == 'true'
    
    print(f"🔧 配置: provider={llm_provider}, llm_enabled={enable_llm}, fallback={fallback_mock}")
    
    try:
        # 创建注册中心
        registry = create_agent_registry(
            llm_provider=llm_provider,
            enable_llm=enable_llm,
            fallback_to_mock=fallback_mock
        )
        
        # 显示注册中心状态
        status = registry.get_registry_status()
        print("\\n📊 注册中心状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # 注册所有智能体
        print("\\n🤖 注册智能体...")
        agents = registry.register_all_agents()
        
        print(f"✅ 成功注册 {len(agents)} 个智能体:")
        for role in agents.keys():
            print(f"  - {role}")
        
        # 测试LLM生成
        print("\\n🧠 测试LLM生成功能...")
        test_results = registry.test_llm_generation()
        
        for role, result in test_results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"  {status_icon} {role}: {result['status']}")
            if result['status'] == 'success':
                print(f"     - 动作维度: {result['action_shape']}")
                print(f"     - 数值范围: {result['action_range']}")
                print(f"     - 提供者: {result['provider']}")
            else:
                print(f"     - 错误: {result.get('error', 'unknown')}")
        
        # 导出配置
        config_file = "agent_registry_config.json"
        registry.export_config(config_file)
        print(f"\\n📁 配置已导出到: {config_file}")
        
        return registry
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None

def test_llm_providers():
    """测试不同LLM提供者"""
    print("\\n🔄 测试不同LLM提供者")
    print("=" * 50)
    
    providers_to_test = ['mock', 'openai', 'anthropic']
    
    for provider in providers_to_test:
        print(f"\\n🧪 测试提供者: {provider}")
        
        try:
            registry = create_agent_registry(
                llm_provider=provider,
                enable_llm=True,
                fallback_to_mock=True
            )
            
            # 只注册一个智能体进行快速测试
            agent = registry.register_agent('doctors')
            
            # 测试生成
            test_result = registry.test_llm_generation('doctors')
            result = test_result['doctors']
            
            if result['status'] == 'success':
                print(f"  ✅ {provider} 提供者正常工作")
                print(f"     - 生成动作: {result['action_shape']}")
            else:
                print(f"  ⚠️  {provider} 提供者失败: {result.get('error')}")
                
        except Exception as e:
            print(f"  ❌ {provider} 提供者异常: {e}")

def demonstrate_api_key_management():
    """演示API key管理"""
    print("\\n🔑 API Key管理演示")
    print("=" * 50)
    
    # 显示当前API key状态
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    print("📋 当前API Key状态:")
    print(f"  OPENAI_API_KEY: {'✅ 已配置' if openai_key else '❌ 未配置'}")
    print(f"  ANTHROPIC_API_KEY: {'✅ 已配置' if anthropic_key else '❌ 未配置'}")
    
    if not openai_key and not anthropic_key:
        print("\\n💡 要使用真实LLM服务，请配置对应的API key:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        print("  export ANTHROPIC_API_KEY='your-anthropic-api-key'")
        print("\\n🔄 当前将使用Mock提供者进行演示")
    
    # 创建注册中心并显示API状态
    registry = create_agent_registry()
    status = registry.get_registry_status()
    
    print("\\n🔍 API可用性检查:")
    for provider, available in status['api_status'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {provider}: {'可用' if available else '不可用'}")

def interactive_agent_demo():
    """交互式智能体演示"""
    print("\\n🎮 交互式智能体演示")
    print("=" * 50)
    
    # 获取全局注册中心
    registry = get_global_agent_registry()
    
    # 如果没有注册智能体，先注册
    if not registry.get_all_agents():
        print("🚀 初始化智能体...")
        registry.register_all_agents()
    
    agents = registry.get_all_agents()
    print(f"\\n🤖 可用的智能体角色: {list(agents.keys())}")
    
    while True:
        print("\\n" + "="*30)
        print("1. 查看智能体状态")
        print("2. 测试LLM生成")
        print("3. 切换LLM提供者")
        print("4. 导出配置")
        print("5. 退出")
        
        choice = input("\\n请选择操作 (1-5): ").strip()
        
        if choice == '1':
            status = registry.get_registry_status()
            print("\\n📊 注册中心状态:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        
        elif choice == '2':
            role = input("输入要测试的角色 (doctors/interns/patients/accountants/government): ").strip()
            if role in agents:
                result = registry.test_llm_generation(role)
                print(f"\\n🧠 {role} LLM测试结果: {result[role]}")
            else:
                print(f"❌ 角色 {role} 不存在")
        
        elif choice == '3':
            provider = input("输入新的LLM提供者 (mock/openai/anthropic/local): ").strip()
            try:
                new_provider = LLMProviderType(provider)
                registry.update_llm_config(new_provider)
                print(f"✅ 已切换到 {provider} 提供者")
            except ValueError:
                print(f"❌ 不支持的提供者: {provider}")
        
        elif choice == '4':
            filename = input("输入配置文件名 (默认: config.json): ").strip() or "config.json"
            registry.export_config(filename)
            print(f"✅ 配置已导出到 {filename}")
        
        elif choice == '5':
            print("👋 再见!")
            break
        
        else:
            print("❌ 无效选择，请重试")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'setup':
            setup_environment_variables_example()
        elif command == 'test':
            test_agent_registry_with_env()
        elif command == 'providers':
            test_llm_providers()
        elif command == 'api':
            demonstrate_api_key_management()
        elif command == 'interactive':
            interactive_agent_demo()
        else:
            print(f"❌ 未知命令: {command}")
            print("💡 可用命令: setup, test, providers, api, interactive")
    else:
        # 默认运行完整演示
        print("🏥 医院治理智能体系统 - 环境变量配置演示")
        print("=" * 60)
        
        setup_environment_variables_example()
        demonstrate_api_key_management()
        test_agent_registry_with_env()
        
        print("\\n✅ 演示完成! 使用以下命令进行更多测试:")
        print("  python3 env_config_example.py interactive  # 交互式演示")
        print("  python3 env_config_example.py providers    # 测试不同提供者")

if __name__ == "__main__":
    main()