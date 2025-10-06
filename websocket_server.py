#!/usr/bin/env python3
"""
医院治理系统 - WebSocket实时监控服务器
专注于数据推送和订阅，仿真逻辑由KallipolisSimulator负责
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Any
import websockets
import numpy as np
from pathlib import Path
import sys
import yaml
import http.server
import socketserver
from threading import Thread

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查核心算法模块
try:
    from src.hospital_governance.simulation.simulator import KallipolisSimulator, SimulationConfig
    from src.hospital_governance.simulation.scenario_runner import ScenarioRunner
    HAS_CORE_ALGORITHMS = True
    logger.info("✅ 医院治理系统核心模块导入成功")
except ImportError as e:
    logger.warning(f"⚠️ 核心算法模块导入失败: {e}")
    logger.info("🔄 将使用模拟数据运行")
    HAS_CORE_ALGORITHMS = False

class HospitalSimulationServer:
    """医院仿真WebSocket服务器
    
    职责：
    1. WebSocket连接管理
    2. 数据推送和订阅
    3. 前端界面通信
    4. 仿真器生命周期管理
    """
    
    def __init__(self, host="localhost", port=8000):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # 仿真状态
        self.simulation_running = False
        self.simulation_paused = False
        self.current_step = 0
        self.start_time = None
        
        # 仿真器和场景运行器
        self.simulator = None
        self.scenario_runner = None
        
        # 16维系统状态指标（统一定义）
        self.performance_metrics = {
            'bed_utilization': 0.7,           # 床位占用率
            'equipment_utilization': 0.8,     # 设备利用率
            'staff_utilization': 0.6,         # 人员利用率
            'medication_level': 0.9,          # 药品库存水平
            'cash_reserves': 0.8,             # 现金储备
            'profit_margin': 0.1,             # 运营利润率
            'debt_ratio': 0.3,                # 资产负债率
            'cost_efficiency': 0.75,          # 成本效率
            'patient_satisfaction': 0.85,     # 患者满意度
            'treatment_success': 0.9,         # 治疗成功率
            'average_wait_time': 0.2,         # 平均等待时间
            'safety_index': 0.95,             # 医疗安全指数
            'ethics_score': 0.8,              # 伦理合规评分
            'fairness_index': 0.85,           # 资源公平性指数
            'learning_efficiency': 0.7,       # 学习效率
            'knowledge_transfer': 0.8         # 知识传递效率
        }
        
        # 基础规则系统（用于fallback）
        self.basic_rules = {
            'patient_safety_protocol': {
                'name': '患者安全协议',
                'description': '确保患者安全的基本协议',
                'activated': False,
                'severity': 0.0
            },
            'resource_allocation_rule': {
                'name': '资源分配规则',
                'description': '优化医疗资源分配',
                'activated': False,
                'severity': 0.0
            },
            'emergency_response_protocol': {
                'name': '紧急响应协议',
                'description': '紧急情况下的响应机制',
                'activated': False,
                'severity': 0.0
            },
            'quality_control_standard': {
                'name': '质量控制标准',
                'description': '医疗质量控制标准',
                'activated': False,
                'severity': 0.0
            },
            'financial_oversight_rule': {
                'name': '财务监督规则',
                'description': '财务运营监督规则',
                'activated': False,
                'severity': 0.0
            }
        }
        
        logger.info("🏥 WebSocket服务器初始化完成")

    async def register_client(self, websocket, path=None):
        """注册新客户端"""
        self.clients.add(websocket)
        logger.info(f"📱 客户端已连接: {websocket.remote_address}")
        
        # 发送初始状态
        await self.send_to_client(websocket, {
            'type': 'welcome',
            'message': '🏥 欢迎连接到Kallipolis医疗共和国治理系统',
            'server_info': {
                'system_name': 'Kallipolis Medical Republic',
                'version': '2.0.0 (重构版)',
                'architecture': 'WebSocket推送/订阅 + KallipolisSimulator仿真',
                'integration_status': 'production' if HAS_CORE_ALGORITHMS else 'simulation'
            },
            'timestamp': datetime.now().isoformat()
        })
        
        # 发送当前系统状态
        await self.send_system_status(websocket)
        
        try:
            # 处理客户端消息
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"📱 客户端已断开: {websocket.remote_address}")
    
    async def handle_client_message(self, websocket, message):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'start':
                await self.start_simulation()
            elif command == 'pause':
                await self.pause_simulation()
            elif command == 'reset':
                await self.reset_simulation()
            elif command == 'get_status':
                await self.send_status(websocket)
            else:
                logger.warning(f"❓ 未知命令: {command}")
                
        except json.JSONDecodeError:
            logger.error("❌ 无效的JSON消息")
        except Exception as e:
            logger.error(f"❌ 处理客户端消息时出错: {e}")
    
    async def start_simulation(self):
        """开始仿真"""
        if not self.simulation_running:
            self.simulation_running = True
            self.simulation_paused = False
            self.start_time = datetime.now()
            logger.info("🚀 仿真已启动")
            
            await self.broadcast({
                'type': 'simulation_control',
                'action': 'started',
                'timestamp': datetime.now().isoformat()
            })
            
            # 启动仿真循环
            asyncio.create_task(self.simulation_loop())
    
    async def pause_simulation(self):
        """暂停仿真"""
        if self.simulation_running:
            self.simulation_paused = not self.simulation_paused
            action = 'paused' if self.simulation_paused else 'resumed'
            logger.info(f"⏸️ 仿真已{action}")
            
            # 通知仿真器暂停/继续
            if self.simulator:
                if self.simulation_paused:
                    self.simulator.pause()
                else:
                    self.simulator.resume()
            
            await self.broadcast({
                'type': 'simulation_control',
                'action': action,
                'timestamp': datetime.now().isoformat()
            })
    
    async def reset_simulation(self):
        """重置仿真"""
        self.simulation_running = False
        self.simulation_paused = False
        self.current_step = 0
        self.start_time = None
        
        # 通知仿真器停止并重置
        if self.simulator:
            self.simulator.stop()
            self.simulator.reset()
        
        # 重置性能指标
        self.performance_metrics.update({
            'bed_utilization': 0.7,
            'equipment_utilization': 0.8,
            'staff_utilization': 0.6,
            'medication_level': 0.9,
            'cash_reserves': 0.8,
            'profit_margin': 0.1,
            'debt_ratio': 0.3,
            'cost_efficiency': 0.75,
            'patient_satisfaction': 0.85,
            'treatment_success': 0.9,
            'average_wait_time': 0.2,
            'safety_index': 0.95,
            'ethics_score': 0.8,
            'fairness_index': 0.85,
            'learning_efficiency': 0.7,
            'knowledge_transfer': 0.8
        })
        
        logger.info("🔄 仿真已重置")
        
        await self.broadcast({
            'type': 'simulation_control',
            'action': 'reset',
            'timestamp': datetime.now().isoformat()
        })
    
    async def send_status(self, websocket):
        """发送状态信息"""
        await self.send_to_client(websocket, {
            'type': 'status',
            'simulation_running': self.simulation_running,
            'simulation_paused': self.simulation_paused,
            'current_step': self.current_step,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'timestamp': datetime.now().isoformat()
        })
    
    async def send_system_status(self, websocket):
        """发送详细系统状态"""
        try:
            system_status = {
                'type': 'system_status',
                'simulation': {
                    'running': self.simulation_running,
                    'paused': self.simulation_paused,
                    'step': self.current_step,
                    'start_time': self.start_time.isoformat() if self.start_time else None
                },
                'performance_metrics': self.performance_metrics,
                'integration_status': 'production' if HAS_CORE_ALGORITHMS else 'simulation',
                'architecture': 'Separated WebSocket Server + KallipolisSimulator',
                'timestamp': datetime.now().isoformat()
            }
            
            await self.send_to_client(websocket, system_status)
        except Exception as e:
            logger.error(f"❌ 发送系统状态失败: {e}")
    
    async def simulation_loop(self):
        """启动仿真循环 - WebSocket服务器作为数据推送接口"""
        try:
            if HAS_CORE_ALGORITHMS:
                # 使用真实仿真器
                await self._start_real_simulation()
            else:
                # 使用模拟仿真器
                await self._start_mock_simulation()
                
        except Exception as e:
            logger.error(f"❌ 仿真循环启动失败: {e}")
            # 回退到模拟模式
            await self._start_mock_simulation()
    
    async def _start_real_simulation(self):
        """启动真实仿真循环"""
        logger.info("🔄 启动真实仿真循环...")
        
        # 创建仿真器实例
        config = SimulationConfig(
            max_steps=1000,
            enable_learning=True,
            enable_holy_code=True,
            enable_crises=True
        )
        self.simulator = KallipolisSimulator(config)
        
        # 设置数据推送回调
        self.simulator.set_data_callback(self.on_simulation_data)
        
        # 创建场景运行器
        self.scenario_runner = ScenarioRunner(self.simulator)
        try:
            self.scenario_runner.load_scenarios_from_yaml('config/simulation_scenarios.yaml')
        except Exception as e:
            logger.warning(f"⚠️ 场景文件加载失败: {e}")
        
        # 启动仿真循环（异步运行）
        await self.simulator.run_async(
            steps=1000, 
            scenario_runner=self.scenario_runner,
            training=False
        )
    
    async def _start_mock_simulation(self):
        """启动模拟仿真循环（回退模式）"""
        logger.info("🔄 使用模拟仿真循环（回退模式）")
        
        while self.simulation_running:
            if not self.simulation_paused:
                await self._mock_simulation_step()
                self.current_step += 1
            await asyncio.sleep(2)
    
    async def _mock_simulation_step(self):
        """执行模拟仿真步骤"""
        # 更新性能指标（添加随机变化）
        for key in self.performance_metrics:
            if key not in ['average_wait_time']:  # 等待时间是反向指标
                noise = np.random.normal(0, 0.02)
                self.performance_metrics[key] += noise
                self.performance_metrics[key] = np.clip(self.performance_metrics[key], 0.1, 1.0)
            else:
                noise = np.random.normal(0, 0.02)
                self.performance_metrics[key] += noise
                self.performance_metrics[key] = np.clip(self.performance_metrics[key], 0.0, 1.0)
        
        # 推送仿真步骤数据
        await self.broadcast({
            'type': 'simulation_step',
            'step': self.current_step,
            'timestamp': datetime.now().isoformat()
        })
        
        # 推送系统状态
        await self.broadcast({
            'type': 'system_state',
            'state': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # 推送性能指标
        await self.broadcast({
            'type': 'metrics',
            **self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # 模拟智能体活动
        if np.random.random() < 0.6:  # 60%概率生成活动
            await self._generate_mock_agent_activity()
        
        # 模拟规则检查
        await self._check_mock_rules()
    
    async def _generate_mock_agent_activity(self):
        """生成模拟智能体活动"""
        agents = ['doctors', 'interns', 'patients', 'accountants', 'government']
        agent_actions = {
            'doctors': ['诊断患者', '制定治疗方案', '紧急救治', '医疗会诊'],
            'interns': ['学习新技能', '协助诊疗', '参与培训', '临床实践'],
            'patients': ['就医咨询', '反馈意见', '参与治疗', '康复训练'],
            'accountants': ['成本分析', '预算规划', '财务审计', '资源优化'],
            'government': ['政策制定', '监管检查', '资源分配', '绩效评估']
        }
        
        agent_id = np.random.choice(agents)
        action = np.random.choice(agent_actions[agent_id])
        
        await self.broadcast({
            'type': 'agent_action',
            'agent_id': agent_id,
            'action': action,
            'reasoning': f"{agent_id} 基于当前系统状态执行决策",
            'confidence': 0.7 + np.random.random() * 0.3,
            'timestamp': datetime.now().isoformat()
        })
    
    async def _check_mock_rules(self):
        """检查模拟规则激活"""
        for rule_id, rule_info in self.basic_rules.items():
            if np.random.random() < 0.05:  # 5%概率触发规则
                rule_info['activated'] = not rule_info['activated']
                rule_info['severity'] = np.random.random()
                
                await self.broadcast({
                    'type': 'rule_activation',
                    'rule_name': rule_info['name'],
                    'activated': rule_info['activated'],
                    'severity': rule_info['severity'],
                    'description': rule_info['description'],
                    'timestamp': datetime.now().isoformat()
                })
    
    async def on_simulation_data(self, step_data: Dict[str, Any]):
        """处理来自仿真器的数据推送"""
        try:
            # 更新服务器状态
            self.current_step = step_data.get('step', self.current_step)
            
            # 推送仿真步骤数据
            await self.broadcast({
                'type': 'simulation_step',
                'step': step_data.get('step'),
                'time': step_data.get('time'),
                'timestamp': datetime.now().isoformat()
            })
            
            # 推送系统状态
            if 'system_state' in step_data:
                # 将系统状态映射到16维性能指标
                system_state = step_data['system_state']
                if isinstance(system_state, dict):
                    # 更新性能指标
                    mapping = {
                        'bed_utilization': system_state.get('resource_utilization', 0.7),
                        'equipment_utilization': system_state.get('resource_adequacy', 0.8),
                        'staff_utilization': system_state.get('workload', 0.6),
                        'patient_satisfaction': system_state.get('patient_satisfaction', 0.85),
                        'treatment_success': system_state.get('medical_quality', 0.9),
                        'safety_index': system_state.get('patient_safety', 0.95),
                        'cost_efficiency': system_state.get('cost_efficiency', 0.75),
                        'financial_health': system_state.get('financial_health', 0.8)
                    }
                    
                    for metric, value in mapping.items():
                        if metric in self.performance_metrics:
                            self.performance_metrics[metric] = float(value)
                
                await self.broadcast({
                    'type': 'system_state',
                    'state': self.performance_metrics,
                    'timestamp': datetime.now().isoformat()
                })
            
            # 推送智能体行动
            if 'actions' in step_data:
                for agent_id, action_data in step_data['actions'].items():
                    await self.broadcast({
                        'type': 'agent_action',
                        'agent_id': agent_id,
                        'action': action_data.get('action', 'Unknown action'),
                        'reasoning': action_data.get('reasoning', f"{agent_id} 基于当前状态执行决策"),
                        'confidence': action_data.get('confidence', 0.8),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # 推送性能指标
            if 'metrics' in step_data:
                metrics = step_data['metrics']
                await self.broadcast({
                    'type': 'metrics',
                    **self.performance_metrics,
                    'timestamp': datetime.now().isoformat()
                })
            
            # 推送危机事件
            if 'crises' in step_data and step_data['crises']:
                for crisis in step_data['crises']:
                    await self.broadcast({
                        'type': 'crisis',
                        'crisis_type': crisis.get('type', 'unknown'),
                        'severity': crisis.get('severity', 0.0),
                        'description': crisis.get('description', '未知危机'),
                        'timestamp': datetime.now().isoformat()
                    })
                
        except Exception as e:
            logger.error(f"❌ 处理仿真数据推送失败: {e}")
    
    async def broadcast(self, message):
        """向所有客户端广播消息"""
        if self.clients:
            await asyncio.gather(
                *[self.send_to_client(client, message) for client in self.clients],
                return_exceptions=True
            )
    
    async def send_to_client(self, websocket, message):
        """向单个客户端发送消息"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(websocket)
        except Exception as e:
            logger.error(f"❌ 发送消息失败: {e}")
    
    async def start_server(self):
        """启动WebSocket服务器和HTTP服务器"""
        logger.info(f"🌐 启动WebSocket服务器: {self.host}:{self.port}")
        
        # 启动HTTP服务器提供前端文件
        def start_http_server():
            """启动HTTP服务器"""
            class HTTPHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory="/Users/dnimo/Asclepion", **kwargs)
                    
                def end_headers(self):
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    super().end_headers()
            
            try:
                with socketserver.TCPServer(("", 8080), HTTPHandler) as httpd:
                    logger.info("✅ HTTP服务器启动在端口 8080")
                    logger.info("🌐 前端页面: http://localhost:8080/frontend/websocket_demo.html")
                    httpd.serve_forever()
            except Exception as e:
                logger.error(f"❌ HTTP服务器启动失败: {e}")
        
        # 在后台线程启动HTTP服务器
        http_thread = Thread(target=start_http_server, daemon=True)
        http_thread.start()
        
        # 启动WebSocket服务器
        async with websockets.serve(
            self.register_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        ):
            logger.info("✅ WebSocket服务器启动成功")
            logger.info("🎯 重构后的架构:")
            logger.info("   📡 WebSocket服务器: 数据推送/订阅")
            logger.info("   🧠 KallipolisSimulator: 仿真逻辑主体")
            logger.info("   📊 数据流: Simulator -> Callback -> WebSocket -> Frontend")
            
            # 保持服务器运行
            await asyncio.Future()

async def main():
    """主函数"""
    server = HospitalSimulationServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("🛑 服务器已停止")
    except Exception as e:
        logger.error(f"❌ 服务器错误: {e}")

if __name__ == "__main__":
    print("🏥 医院治理系统 - WebSocket实时监控服务器 (重构版)")
    print("=" * 70)
    print("📡 架构: 分离式 WebSocket推送/订阅 + KallipolisSimulator仿真")
    print("🌐 前端界面: http://localhost:8080/frontend/websocket_demo.html")
    print("🔌 WebSocket端点: ws://localhost:8000")
    print("📊 HTTP服务器: http://localhost:8080")
    print("按 Ctrl+C 停止服务器")
    print("=" * 70)
    
    asyncio.run(main())