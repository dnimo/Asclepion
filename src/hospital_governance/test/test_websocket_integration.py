#!/usr/bin/env python3
"""
WebSocket服务器集成测试
测试真实算法的集成效果
"""

import asyncio
import json
import websockets
import numpy as np
from datetime import datetime
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketIntegrationTester:
    """WebSocket集成测试器"""
    
    def __init__(self, uri="ws://localhost:8000"):
        self.uri = uri
        self.connected = False
        self.websocket = None
        self.messages_received = []
        
    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            logger.info(f"已连接到WebSocket服务器: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("已断开WebSocket连接")
    
    async def send_command(self, command):
        """发送命令到服务器"""
        if not self.connected:
            logger.error("未连接到服务器")
            return False
            
        try:
            message = json.dumps({"command": command})
            await self.websocket.send(message)
            logger.info(f"发送命令: {command}")
            return True
        except Exception as e:
            logger.error(f"发送命令失败: {e}")
            return False
    
    async def listen_for_messages(self, duration=30):
        """监听指定时间内的消息"""
        if not self.connected:
            return []
            
        start_time = time.time()
        messages = []
        
        try:
            while time.time() - start_time < duration:
                try:
                    # 等待消息，超时1秒
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=1.0
                    )
                    
                    data = json.loads(message)
                    messages.append(data)
                    
                    # 记录重要消息类型
                    if data.get('type') in ['agent_action', 'rule_activation', 'metrics', 'dialog']:
                        logger.info(f"收到{data['type']}消息: {data.get('agent_id', data.get('rule_name', 'system'))}")
                        
                except asyncio.TimeoutError:
                    continue
                except json.JSONDecodeError:
                    logger.warning("收到无效JSON消息")
                except Exception as e:
                    logger.error(f"接收消息错误: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"监听消息失败: {e}")
            
        return messages
    
    def analyze_messages(self, messages):
        """分析接收到的消息"""
        analysis = {
            'total_messages': len(messages),
            'message_types': {},
            'agent_activities': {},
            'rule_activations': {},
            'performance_data': [],
            'dialogs': [],
            'integration_check': {
                'has_real_agents': False,
                'has_rule_engine': False,
                'has_performance_metrics': False,
                'has_system_state': False
            }
        }
        
        for msg in messages:
            msg_type = msg.get('type', 'unknown')
            analysis['message_types'][msg_type] = analysis['message_types'].get(msg_type, 0) + 1
            
            # 分析智能体活动
            if msg_type == 'agent_action':
                agent_id = msg.get('agent_id', 'unknown')
                if agent_id not in analysis['agent_activities']:
                    analysis['agent_activities'][agent_id] = []
                
                analysis['agent_activities'][agent_id].append({
                    'action': msg.get('action'),
                    'reasoning': msg.get('reasoning'),
                    'confidence': msg.get('confidence'),
                    'timestamp': msg.get('timestamp')
                })
                
                # 检查是否使用真实智能体
                if any(role in agent_id for role in ['医生', '实习医生', '患者', '会计', '政府']):
                    analysis['integration_check']['has_real_agents'] = True
            
            # 分析规则激活
            elif msg_type == 'rule_activation':
                rule_name = msg.get('rule_name')
                if msg.get('activated'):
                    analysis['rule_activations'][rule_name] = {
                        'severity': msg.get('severity'),
                        'description': msg.get('description'),
                        'timestamp': msg.get('timestamp')
                    }
                    analysis['integration_check']['has_rule_engine'] = True
            
            # 分析性能数据
            elif msg_type == 'metrics':
                analysis['performance_data'].append({
                    'stability': msg.get('stability'),
                    'performance': msg.get('performance'),
                    'efficiency': msg.get('efficiency'),
                    'safety': msg.get('safety'),
                    'timestamp': msg.get('timestamp')
                })
                analysis['integration_check']['has_performance_metrics'] = True
            
            # 分析系统状态
            elif msg_type == 'system_state':
                if msg.get('state'):
                    analysis['integration_check']['has_system_state'] = True
            
            # 分析对话
            elif msg_type == 'dialog':
                analysis['dialogs'].append({
                    'participants': msg.get('participants'),
                    'content': msg.get('content'),
                    'timestamp': msg.get('timestamp')
                })
        
        return analysis
    
    def print_analysis_report(self, analysis):
        """打印分析报告"""
        print("\n" + "="*80)
        print("WebSocket集成测试分析报告")
        print("="*80)
        
        # 基本统计
        print(f"\n📊 基本统计:")
        print(f"  总消息数: {analysis['total_messages']}")
        print(f"  消息类型分布:")
        for msg_type, count in analysis['message_types'].items():
            print(f"    {msg_type}: {count}")
        
        # 集成检查
        print(f"\n🔧 集成状态检查:")
        checks = analysis['integration_check']
        print(f"  ✅ 真实智能体系统: {'是' if checks['has_real_agents'] else '否'}")
        print(f"  ✅ 规则引擎集成: {'是' if checks['has_rule_engine'] else '否'}")
        print(f"  ✅ 性能指标系统: {'是' if checks['has_performance_metrics'] else '否'}")
        print(f"  ✅ 系统状态更新: {'是' if checks['has_system_state'] else '否'}")
        
        # 智能体活动
        print(f"\n🤖 智能体活动分析:")
        if analysis['agent_activities']:
            for agent_id, activities in analysis['agent_activities'].items():
                print(f"  {agent_id}: {len(activities)}次活动")
                if activities:
                    latest = activities[-1]
                    print(f"    最新活动: {latest['action']}")
                    print(f"    置信度: {latest.get('confidence', 'N/A')}")
        else:
            print("  未检测到智能体活动")
        
        # 规则激活
        print(f"\n📋 规则激活分析:")
        if analysis['rule_activations']:
            for rule_name, rule_info in analysis['rule_activations'].items():
                print(f"  {rule_name}: 严重程度 {rule_info['severity']:.3f}")
        else:
            print("  未检测到规则激活")
        
        # 性能趋势
        print(f"\n📈 性能趋势分析:")
        if analysis['performance_data']:
            latest_metrics = analysis['performance_data'][-1]
            print(f"  最新性能指标:")
            print(f"    稳定性: {latest_metrics.get('stability', 'N/A'):.3f}")
            print(f"    性能: {latest_metrics.get('performance', 'N/A'):.3f}")
            print(f"    效率: {latest_metrics.get('efficiency', 'N/A'):.3f}")
            print(f"    安全性: {latest_metrics.get('safety', 'N/A'):.3f}")
        else:
            print("  未检测到性能数据")
        
        # 对话分析
        print(f"\n💬 智能体对话分析:")
        if analysis['dialogs']:
            print(f"  对话次数: {len(analysis['dialogs'])}")
            if analysis['dialogs']:
                latest_dialog = analysis['dialogs'][-1]
                print(f"  最新对话参与者: {', '.join(latest_dialog['participants'])}")
                print(f"  对话内容: {latest_dialog['content']}")
        else:
            print("  未检测到智能体对话")
        
        # 总体评估
        print(f"\n🎯 总体集成评估:")
        integration_score = sum(analysis['integration_check'].values()) / len(analysis['integration_check'])
        if integration_score >= 0.75:
            print(f"  ✅ 集成状态: 优秀 ({integration_score:.1%})")
        elif integration_score >= 0.5:
            print(f"  ⚠️  集成状态: 良好 ({integration_score:.1%})")
        else:
            print(f"  ❌ 集成状态: 需要改进 ({integration_score:.1%})")
        
        print("="*80)

async def run_integration_test():
    """运行集成测试"""
    print("🏥 医院治理系统 - WebSocket集成测试")
    print("="*60)
    
    tester = WebSocketIntegrationTester()
    
    # 连接到服务器
    if not await tester.connect():
        print("❌ 无法连接到WebSocket服务器")
        print("请确保服务器正在运行: python websocket_server.py")
        return
    
    try:
        # 发送启动命令
        print("\n📡 发送仿真启动命令...")
        await tester.send_command("start")
        
        # 监听消息
        print("🎧 监听仿真数据 (30秒)...")
        messages = await tester.listen_for_messages(duration=30)
        
        # 分析消息
        print("🔍 分析接收到的数据...")
        analysis = tester.analyze_messages(messages)
        
        # 打印报告
        tester.print_analysis_report(analysis)
        
    finally:
        await tester.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(run_integration_test())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试执行错误: {e}")