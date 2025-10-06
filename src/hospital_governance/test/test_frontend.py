#!/usr/bin/env python3
"""
医院治理系统 - 前端测试脚本
用于测试前端界面功能
"""

import asyncio
import json
import websockets
import time

async def test_frontend():
    """测试前端WebSocket连接"""
    uri = "ws://localhost:8000/ws/hospital"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ 成功连接到WebSocket服务器")
            
            # 发送启动命令
            await websocket.send(json.dumps({"command": "start"}))
            print("🚀 已发送启动仿真命令")
            
            # 监听消息
            message_count = 0
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type', 'unknown')
                    
                    print(f"📨 收到消息 #{message_count}: {message_type}")
                    
                    if message_type == 'agent_action':
                        print(f"   🤖 {data['agent_id']}: {data['action']}")
                    elif message_type == 'rule_activation':
                        status = "激活" if data['activated'] else "停用"
                        print(f"   ⚖️ {data['rule_name']}: {status} (严重度: {data['severity']:.2f})")
                    elif message_type == 'metrics':
                        print(f"   📊 性能指标 - 稳定性: {data['stability']:.3f}, 安全性: {data['safety']:.3f}")
                    elif message_type == 'dialog':
                        participants = ', '.join(data['participants'])
                        print(f"   💬 对话 [{participants}]: {data['content'][:50]}...")
                    
                    message_count += 1
                    
                    # 测试10条消息后停止
                    if message_count >= 20:
                        print("\n🔄 测试暂停命令...")
                        await websocket.send(json.dumps({"command": "pause"}))
                        
                        # 等待几秒后重置
                        await asyncio.sleep(3)
                        print("🔄 测试重置命令...")
                        await websocket.send(json.dumps({"command": "reset"}))
                        break
                        
                except json.JSONDecodeError:
                    print(f"❌ 无法解析消息: {message}")
                    
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("请确保服务器正在运行: python3 hospital_web_server.py")

if __name__ == "__main__":
    print("🧪 医院治理系统前端测试")
    print("=" * 50)
    asyncio.run(test_frontend())