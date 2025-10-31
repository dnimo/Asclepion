#!/usr/bin/env python3
"""
WebSocket服务器集成测试脚本
测试KallipolisSimulator与WebSocket服务器的完整集成
"""

import asyncio
import websockets
import json
import time
from datetime import datetime

class IntegrationTester:
    """WebSocket + KallipolisSimulator集成测试客户端"""
    
    def __init__(self, uri="ws://localhost:8000"):
        self.uri = uri
        self.received_messages = []
        self.message_counts = {
            'welcome': 0,
            'simulation_step': 0,
            'system_state': 0,
            'agent_action': 0,
            'metrics': 0,
            'parliament_meeting': 0,
            'crisis': 0,
            'rule_activation': 0
        }
    
    async def test_complete_integration(self):
        """测试完整集成功能"""
        print("🚀 开始集成测试...")
        print(f"连接目标: {self.uri}")
        print("=" * 60)
        
        try:
            async with websockets.connect(self.uri) as websocket:
                print("✅ WebSocket连接成功")
                
                # 1. 接收欢迎消息
                await self.test_welcome_message(websocket)
                
                # 2. 获取系统状态
                await self.test_system_status(websocket)
                
                # 3. 启动仿真
                await self.test_start_simulation(websocket)
                
                # 4. 监听仿真数据流 - 这是集成测试的核心
                await self.test_simulation_data_stream(websocket)
                
                # 5. 测试控制功能
                await self.test_simulation_controls(websocket)
                
                # 6. 生成测试报告
                self.generate_test_report()
                
        except Exception as e:
            print(f"❌ 集成测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_welcome_message(self, websocket):
        """测试欢迎消息"""
        print("\n📨 测试欢迎消息...")
        
        try:
            welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            welcome_data = json.loads(welcome_msg)
            
            if welcome_data.get('type') == 'welcome':
                self.message_counts['welcome'] += 1
                print(f"✅ 收到欢迎消息: {welcome_data['message']}")
                print(f"   架构信息: {welcome_data['server_info']['architecture']}")
                print(f"   集成状态: {welcome_data['server_info']['integration_status']}")
            else:
                print(f"❓ 意外消息: {welcome_data}")
                
        except asyncio.TimeoutError:
            print("❌ 欢迎消息超时")
    
    async def test_system_status(self, websocket):
        """测试系统状态"""
        print("\n📊 测试系统状态...")
        
        command = {"command": "get_status"}
        await websocket.send(json.dumps(command))
        
        # 可能收到多个响应消息
        for _ in range(3):  # 最多等待3个消息
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                
                if data.get('type') == 'status':
                    print(f"✅ 状态响应: 运行={data.get('simulation_running')}, 步数={data.get('current_step')}")
                elif data.get('type') == 'system_status':
                    print(f"✅ 系统状态: 架构={data.get('architecture')}")
                    print(f"   性能指标数量: {len(data.get('performance_metrics', {}))}")
                
            except asyncio.TimeoutError:
                break
    
    async def test_start_simulation(self, websocket):
        """测试启动仿真"""
        print("\n🚀 测试启动仿真...")
        
        command = {"command": "start"}
        await websocket.send(json.dumps(command))
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            
            if data.get('type') == 'simulation_control' and data.get('action') == 'started':
                print("✅ 仿真启动成功")
            else:
                print(f"❓ 收到响应: {data.get('type')} - {data.get('action')}")
                
        except asyncio.TimeoutError:
            print("❌ 启动响应超时")
    
    async def test_simulation_data_stream(self, websocket):
        """测试仿真数据流 - 这是集成测试的核心部分"""
        print("\n📡 测试仿真数据流集成...")
        print("正在监听KallipolisSimulator推送的数据...")
        
        start_time = time.time()
        test_duration = 20  # 监听20秒
        
        while time.time() - start_time < test_duration:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(response)
                msg_type = data.get('type')
                
                # 统计消息类型
                if msg_type in self.message_counts:
                    self.message_counts[msg_type] += 1
                
                # 详细分析不同类型的数据
                if msg_type == 'simulation_step':
                    step = data.get('step')
                    time_val = data.get('time', 0)
                    print(f"📈 仿真步骤: #{step} (时间: {time_val:.1f})")
                    
                elif msg_type == 'system_state':
                    state = data.get('state', {})
                    performance_count = len([k for k, v in state.items() if isinstance(v, (int, float))])
                    print(f"🏥 系统状态更新: {performance_count}个指标")
                    
                elif msg_type == 'agent_action':
                    agent_id = data.get('agent_id')
                    action = data.get('action')
                    confidence = data.get('confidence', 0)
                    print(f"🤖 智能体行动: {agent_id} - {action} (置信度: {confidence:.2f})")
                    
                elif msg_type == 'metrics':
                    metrics_count = len(data) - 2  # 减去type和timestamp
                    print(f"📊 性能指标: {metrics_count}个指标更新")
                    
                elif msg_type == 'parliament_meeting':
                    print(f"🏛️ 议会会议: {data.get('description', '议会会议进行中')}")
                    self.message_counts['parliament_meeting'] += 1
                    
                elif msg_type == 'crisis':
                    crisis_type = data.get('crisis_type')
                    severity = data.get('severity', 0)
                    print(f"🚨 危机事件: {crisis_type} (严重程度: {severity:.2f})")
                    
                elif msg_type == 'rule_activation':
                    rule_name = data.get('rule_name')
                    activated = data.get('activated')
                    print(f"⚖️ 规则状态: {rule_name} - {'激活' if activated else '停用'}")
                
                # 记录消息
                self.received_messages.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': msg_type,
                    'data': data
                })
                
            except asyncio.TimeoutError:
                print("⏱️ 等待下一条消息...")
                continue
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析错误: {e}")
    
    async def test_simulation_controls(self, websocket):
        """测试仿真控制功能"""
        print("\n🎮 测试仿真控制...")
        
        # 测试暂停
        print("⏸️ 测试暂停功能...")
        await websocket.send(json.dumps({"command": "pause"}))
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            data = json.loads(response)
            if data.get('type') == 'simulation_control':
                print(f"✅ 暂停响应: {data.get('action')}")
        except asyncio.TimeoutError:
            print("❌ 暂停响应超时")
        
        await asyncio.sleep(2)
        
        # 测试恢复
        print("▶️ 测试恢复功能...")
        await websocket.send(json.dumps({"command": "pause"}))  # 再次切换
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            data = json.loads(response)
            if data.get('type') == 'simulation_control':
                print(f"✅ 恢复响应: {data.get('action')}")
        except asyncio.TimeoutError:
            print("❌ 恢复响应超时")
        
        # 测试重置
        print("🔄 测试重置功能...")
        await websocket.send(json.dumps({"command": "reset"}))
        
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            data = json.loads(response)
            if data.get('type') == 'simulation_control' and data.get('action') == 'reset':
                print("✅ 重置成功")
        except asyncio.TimeoutError:
            print("❌ 重置响应超时")
    
    def generate_test_report(self):
        """生成集成测试报告"""
        print("\n" + "="*60)
        print("📋 集成测试报告")
        print("="*60)
        
        print("📊 消息统计:")
        total_messages = sum(self.message_counts.values())
        for msg_type, count in self.message_counts.items():
            if count > 0:
                percentage = (count / total_messages * 100) if total_messages > 0 else 0
                print(f"   {msg_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n📈 总消息数: {total_messages}")
        print(f"📡 连接时长: ~20秒")
        print(f"🚀 消息频率: {total_messages/20:.1f} 消息/秒")
        
        # 集成健康评估
        print("\n🏥 集成健康评估:")
        
        # 基础连接
        basic_connection = self.message_counts['welcome'] > 0
        print(f"   基础连接: {'✅ 正常' if basic_connection else '❌ 异常'}")
        
        # 仿真数据流
        simulation_active = self.message_counts['simulation_step'] > 0
        print(f"   仿真数据流: {'✅ 正常' if simulation_active else '❌ 异常'}")
        
        # 系统状态推送
        state_updates = self.message_counts['system_state'] > 0
        print(f"   系统状态推送: {'✅ 正常' if state_updates else '❌ 异常'}")
        
        # 智能体活动
        agent_activity = self.message_counts['agent_action'] > 0
        print(f"   智能体活动: {'✅ 正常' if agent_activity else '❌ 异常'}")
        
        # 性能指标
        metrics_updates = self.message_counts['metrics'] > 0
        print(f"   性能指标: {'✅ 正常' if metrics_updates else '❌ 异常'}")
        
        # 整体评估
        healthy_components = sum([
            basic_connection, simulation_active, state_updates, 
            agent_activity, metrics_updates
        ])
        
        print(f"\n🎯 集成健康度: {healthy_components}/5 ({healthy_components*20}%)")
        
        if healthy_components >= 4:
            print("🎉 集成测试通过！KallipolisSimulator与WebSocket服务器集成正常")
        elif healthy_components >= 3:
            print("⚠️ 集成基本正常，部分功能可能需要优化")
        else:
            print("❌ 集成存在问题，需要进一步调试")

async def main():
    """主测试函数"""
    print("🧪 KallipolisSimulator + WebSocket 集成测试")
    print("请确保WebSocket服务器运行在 ws://localhost:8000")
    print("按 Ctrl+C 停止测试")
    print()
    
    tester = IntegrationTester()
    await tester.test_complete_integration()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 测试已手动停止")
    except Exception as e:
        print(f"\n❌ 测试错误: {e}")