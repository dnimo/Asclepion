#!/usr/bin/env python3
"""
WebSocket客户端测试
测试生产版本WebSocket服务器的连接和功能
"""

import asyncio
import websockets
import json
from datetime import datetime

async def test_websocket_connection():
    """测试WebSocket连接"""
    uri = "ws://localhost:8000"
    
    try:
        print("🔗 连接到WebSocket服务器...")
        async with websockets.connect(uri) as websocket:
            print("✅ 连接成功!")
            
            # 等待欢迎消息
            welcome_message = await websocket.recv()
            welcome_data = json.loads(welcome_message)
            print(f"📨 收到欢迎消息: {welcome_data}")
            
            # 等待系统状态
            status_message = await websocket.recv() 
            status_data = json.loads(status_message)
            print(f"📊 收到系统状态: {status_data['type']}")
            
            # 发送启动仿真命令
            start_command = {
                'command': 'start',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(start_command))
            print("🚀 发送启动仿真命令")
            
            # 接收几条消息
            for i in range(5):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    data = json.loads(message)
                    print(f"📈 收到消息 {i+1}: {data['type']}")
                    
                    if data['type'] == 'simulation_step':
                        print(f"   仿真步骤: {data.get('step', 'N/A')}")
                    elif data['type'] == 'metrics':
                        print(f"   性能指标: 稳定性={data.get('stability', 'N/A')}")
                    elif data['type'] == 'agent_activity':
                        print(f"   智能体活动: {data.get('agent', 'N/A')}")
                except asyncio.TimeoutError:
                    print(f"⏱️  消息 {i+1} 超时")
                    break
            
            # 发送暂停命令
            pause_command = {
                'command': 'pause',
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(pause_command))
            print("⏸️  发送暂停仿真命令")
            
            # 最后一条消息
            try:
                final_message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                final_data = json.loads(final_message)
                print(f"🔚 最终消息: {final_data['type']}")
            except asyncio.TimeoutError:
                print("⏱️  未收到最终消息")
                
    except Exception as e:
        print(f"❌ 连接失败: {e}")

async def main():
    print("🧪 WebSocket服务器测试")
    print("=" * 50)
    await test_websocket_connection()
    print("=" * 50)
    print("✅ 测试完成")

if __name__ == "__main__":
    asyncio.run(main())