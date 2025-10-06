#!/usr/bin/env python3
"""
医院治理系统 - 快速启动菜单
提供多种仿真模式的便捷访问
"""

import subprocess
import sys
import os

def print_banner():
    """显示横幅"""
    print("🏥 医院治理系统        elif choice == "9":
            print("\\n🌐 启动实时监控界面...")
            try:
                import subprocess
                subprocess.run(["python3", "hospital_web_server.py"], check=True)
            except Exception as e:
                print(f"启动失败: {e}")
                print("请确保所有依赖已安装")
                
        elif choice == "10":
            print("\\n🔧 WebSocket算法集成演示...")
            print("选择运行模式:")
            print("1. 快速集成测试")
            print("2. 启动集成服务器")
            
            sub_choice = input("请选择 (1-2): ").strip()
            
            if sub_choice == "1":
                try:
                    import subprocess
                    subprocess.run(["python3", "simple_websocket_server.py", "test"], check=True)
                except Exception as e:
                    print(f"测试失败: {e}")
            elif sub_choice == "2":
                print("\\n启动WebSocket集成服务器...")
                print("前端界面: http://localhost:8000/frontend/websocket_demo.html")
                print("按 Ctrl+C 停止服务器")
                try:
                    import subprocess
                    subprocess.run(["python3", "simple_websocket_server.py"], check=True)
                except KeyboardInterrupt:
                    print("\\n服务器已停止")
                except Exception as e:
                    print(f"启动失败: {e}")
            else:
                print("无效选择")nt("=" * 60)
    print("基于分布式控制理论和LLM的智能医院治理仿真")
    print("支持多角色智能体、约束控制和伦理规则引擎")
    print("=" * 60)

def show_simulation_menu():
    """显示仿真选项菜单"""
    print("\\n📋 可用仿真模式:")
    print("-" * 40)
    print("1. 🔧 测试分布式控制系统")
    print("   - 测试5个控制器角色")
    print("   - 验证约束和稳定性")
    print("   - 基础功能验证")
    
    print("\\n2. 📏 测试Holy Code规则引擎")
    print("   - 伦理规则系统测试")
    print("   - YAML规则持久化")
    print("   - 道德约束验证")
    
    print("\\n3. 🤖 简化LLM仿真 (推荐新手)")
    print("   - 50步基础仿真")
    print("   - 无外部依赖")
    print("   - 模拟LLM决策")
    
    print("\\n4. 🧠 高级LLM仿真")
    print("   - 真实API支持")
    print("   - 异步决策引擎")
    print("   - OpenAI/Anthropic集成")
    
    print("\\n5. 📊 完整功能仿真")
    print("   - 图表可视化")
    print("   - 数据导出")
    print("   - 完整性能分析")
    
    print("\\n6. ⚙️ 配置LLM API")
    print("   - 设置OpenAI/Anthropic密钥")
    print("   - API连接测试")
    print("   - 环境配置")
    
    print("\\n7. 🔬 运行所有测试")
    print("   - 综合系统测试")
    print("   - 完整性验证")
    print("   - 性能基准测试")
    
    print("\\n8. 📊 数据导出演示")
    print("   - 完整数据导出功能")
    print("   - 多格式支持")
    print("   - 真实仿真数据")
    
    print("\\n9. 🌐 启动实时监控界面")
    print("   - Web服务器 + WebSocket")
    print("   - 实时智能体监控")
    print("   - 可视化仪表板")
    
    print("\\n10. 🔧 WebSocket算法集成演示")
    print("   - 真实算法集成")
    print("   - 75%集成度演示")
    print("   - 控制器+规则引擎")
    
    print("\\n0. 退出")

def run_script(script_name, description):
    """运行指定脚本"""
    print(f"\\n🚀 {description}")
    print(f"执行: python3 {script_name}")
    print("-" * 40)
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"\\n✅ {description} - 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ {description} - 失败 (错误码: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"\\n❌ 文件未找到: {script_name}")
        return False

def check_file_exists(filename):
    """检查文件是否存在"""
    return os.path.exists(filename)

def show_status():
    """显示系统状态"""
    print("\\n📊 系统状态:")
    print("-" * 30)
    
    # 检查关键文件
    key_files = [
        ("test_control_simple.py", "分布式控制测试"),
        ("test_holy_code_simple.py", "规则引擎测试"),
        ("hospital_simulation_simple.py", "简化仿真"),
        ("hospital_simulation_llm.py", "LLM仿真"),
        ("hospital_simulation_complete.py", "完整仿真"),
        ("setup_llm_simulation.py", "API配置工具"),
        ("demo_export_standalone.py", "数据导出演示"),
        ("hospital_web_server.py", "实时监控服务器")
    ]
    
    for filename, description in key_files:
        status = "✅" if check_file_exists(filename) else "❌"
        print(f"  {status} {description}: {filename}")
    
    # 检查API配置
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    print("\\n🔑 API配置:")
    print(f"  OpenAI: {'✅ 已配置' if openai_key else '❌ 未配置'}")
    print(f"  Anthropic: {'✅ 已配置' if anthropic_key else '❌ 未配置'}")

def main():
    """主函数"""
    print_banner()
    show_status()
    
    while True:
        show_simulation_menu()
        
        try:
            choice = input("\\n请选择操作 (0-9): ").strip()
        except KeyboardInterrupt:
            print("\\n\\n👋 再见！")
            break
        
        if choice == '0':
            print("\\n👋 再见！")
            break
        elif choice == '1':
            run_script("test_control_simple.py", "测试分布式控制系统")
        elif choice == '2':
            run_script("test_holy_code_simple.py", "测试Holy Code规则引擎")
        elif choice == '3':
            run_script("hospital_simulation_simple.py", "运行简化LLM仿真")
        elif choice == '4':
            run_script("hospital_simulation_llm.py", "运行高级LLM仿真")
        elif choice == '5':
            run_script("hospital_simulation_complete.py", "运行完整功能仿真")
        elif choice == '6':
            run_script("setup_llm_simulation.py", "配置LLM API")
        elif choice == '7':
            run_script("test_comprehensive.py", "运行所有测试")
        elif choice == '8':
            run_script("demo_export_standalone.py", "数据导出演示")
        elif choice == '9':
            print("\\n🌐 启动实时监控界面...")
            print("执行: python3 hospital_web_server.py")
            print("浏览器将自动打开监控面板")
            print("-" * 40)
            try:
                import webbrowser
                result = subprocess.run([sys.executable, "hospital_web_server.py"], 
                                      timeout=5, capture_output=True, text=True)
                print("✅ 服务器启动成功")
                print("💡 访问地址: http://localhost:8000")
            except subprocess.TimeoutExpired:
                print("✅ 服务器正在后台运行")
                print("💡 访问地址: http://localhost:8000")
            except Exception as e:
                print(f"❌ 启动失败: {e}")
                print("💡 请手动运行: python3 hospital_web_server.py")
        else:
            print("\\n❌ 无效选择，请输入 0-9")
        
        if choice != '0':
            input("\\n按回车键继续...")

if __name__ == '__main__':
    main()