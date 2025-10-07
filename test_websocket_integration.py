#!/usr/bin/env python3
"""
WebSocket集成测试脚本
测试前端与重构后的仿真器集成
"""

import asyncio
import websockets
import json
import time

async def test_websocket_integration():
    """测试WebSocket集成"""
    print("🔌 开始WebSocket集成测试...")
    
    try:
        uri = 'ws://localhost:8000'
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket连接成功")
            
            # 监听欢迎消息
            try:
                welcome_message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                welcome_data = json.loads(welcome_message)
                print(f"📩 收到欢迎消息: {welcome_data.get('type', 'unknown')}")
                if welcome_data.get('type') == 'welcome':
                    print(f"   🏥 系统: {welcome_data['server_info']['system_name']}")
                    print(f"   📊 架构: {welcome_data['server_info']['architecture']}")
                    print(f"   🔧 状态: {welcome_data['server_info']['integration_status']}")
            except asyncio.TimeoutError:
                print("⏰ 等待欢迎消息超时")
            
            # 发送开始仿真命令
            start_command = {'command': 'start'}
            await websocket.send(json.dumps(start_command))
            print("🚀 发送开始仿真命令")
            
            # 监听系统消息
            message_count = 0
            state_received = False
            agent_actions_received = False
            
            while message_count < 20:  # 最多监听20条消息
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    message_type = data.get('type', 'unknown')
                    message_count += 1
                    
                    print(f"📡 [{message_count}] 收到: {message_type}")
                    
                    if message_type == 'system_state':
                        state_received = True
                        state_data = data.get('state', {})
                        print("   📊 16维状态空间数据:")
                        
                        # 检查物理资源状态
                        physical_keys = ['bed_occupancy_rate', 'medical_equipment_utilization', 
                                       'staff_utilization_rate', 'medication_inventory_level']
                        print("      🏥 物理资源状态:")
                        for key in physical_keys:
                            if key in state_data:
                                print(f"         {key}: {state_data[key]:.3f}")
                        
                        # 检查财务状态
                        financial_keys = ['cash_reserve_ratio', 'operating_margin', 
                                        'debt_to_asset_ratio', 'cost_efficiency_index']
                        print("      💰 财务状态:")
                        for key in financial_keys:
                            if key in state_data:
                                print(f"         {key}: {state_data[key]:.3f}")
                    
                    elif message_type == 'agent_action':
                        agent_actions_received = True
                        agent_id = data.get('agent_id', 'unknown')
                        action = data.get('action', 'unknown')
                        decision_layer = data.get('decision_layer', 'unknown')
                        print(f"   🤖 智能体行动: {agent_id} -> {action}")
                        print(f"      🧠 决策层: {decision_layer}")
                    
                    elif message_type == 'simulation_step':
                        step = data.get('step', 0)
                        print(f"   🔄 仿真步骤: {step}")
                    
                    elif message_type == 'metrics':
                        print("   📈 性能指标更新")
                        
                    elif message_type == 'holy_code_rules':
                        rules_count = len(data.get('active_rules', []))
                        print(f"   ⚖️ 神圣法典规则: {rules_count} 条活跃规则")
                        
                except asyncio.TimeoutError:
                    print("⏰ 等待消息超时，结束监听")
                    break
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析错误: {e}")
            
            # 测试结果总结
            print("\n" + "="*60)
            print("🎯 集成测试结果:")
            print(f"✅ WebSocket连接: 成功")
            print(f"✅ 16维状态数据: {'成功' if state_received else '失败'}")
            print(f"✅ 智能体行动: {'成功' if agent_actions_received else '失败'}")
            print(f"📊 消息总数: {message_count}")
            
            if state_received and agent_actions_received:
                print("🎉 集成测试完全成功!")
            else:
                print("⚠️ 部分功能未能正常工作")
            
    except ConnectionRefusedError:
        print("❌ 无法连接到WebSocket服务器 (端口8000)")
        print("   请确保websocket_server.py正在运行")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    print("🏥 Kallipolis医疗共和国治理系统 - WebSocket集成测试")
    print("="*60)
    asyncio.run(test_websocket_integration())