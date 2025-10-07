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
    # 移除ScenarioRunner导入，直接使用Simulator内置功能
    HAS_CORE_ALGORITHMS = True
    logger.info("✅ 医院治理系统核心模块导入成功")
    logger.info("🎯 支持: MADDPG + LLM + 分布式控制 + 数学策略 + 模板")
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
        
        # 仿真器（集成完整的多层决策架构）
        self.simulator = None
        self.simulation_task = None  # 异步仿真任务
        
        # 16维系统状态指标（与state_space.py完全一致）
        self.performance_metrics = {
            # 物理资源状态 (x₁-x₄)
            'bed_occupancy_rate': 0.7,                    # 病床占用率
            'medical_equipment_utilization': 0.8,        # 医疗设备利用率
            'staff_utilization_rate': 0.6,               # 人员利用率
            'medication_inventory_level': 0.9,           # 药品库存水平
            
            # 财务状态 (x₅-x₈)
            'cash_reserve_ratio': 0.8,                   # 现金储备率
            'operating_margin': 0.1,                     # 运营利润率
            'debt_to_asset_ratio': 0.3,                  # 资产负债率
            'cost_efficiency_index': 0.75,               # 成本效率指数
            
            # 服务质量状态 (x₉-x₁₂)
            'patient_satisfaction_index': 0.85,          # 患者满意度指数
            'treatment_success_rate': 0.9,               # 治疗成功率
            'average_wait_time': 0.2,                    # 平均等待时间
            'medical_safety_index': 0.95,                # 医疗安全指数
            
            # 教育伦理状态 (x₁₃-x₁₆)
            'ethical_compliance_score': 0.8,             # 伦理合规得分
            'resource_allocation_fairness': 0.85,        # 资源分配公平性
            'intern_learning_efficiency': 0.7,           # 实习生学习效率
            'knowledge_transfer_rate': 0.8               # 知识传递率
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
        
        # 停止异步仿真任务
        if self.simulation_task and not self.simulation_task.done():
            self.simulation_task.cancel()
            try:
                await self.simulation_task
            except asyncio.CancelledError:
                logger.info("🛑 仿真任务已取消")
        
        # 重置仿真器
        self.simulator = None
        self.simulation_task = None
        
        # 重置性能指标（与state_space.py完全一致）
        self.performance_metrics.update({
            # 物理资源状态 (x₁-x₄)
            'bed_occupancy_rate': 0.7,                    # 病床占用率
            'medical_equipment_utilization': 0.8,        # 医疗设备利用率
            'staff_utilization_rate': 0.6,               # 人员利用率
            'medication_inventory_level': 0.9,           # 药品库存水平
            
            # 财务状态 (x₅-x₈)
            'cash_reserve_ratio': 0.8,                   # 现金储备率
            'operating_margin': 0.1,                     # 运营利润率
            'debt_to_asset_ratio': 0.3,                  # 资产负债率
            'cost_efficiency_index': 0.75,               # 成本效率指数
            
            # 服务质量状态 (x₉-x₁₂)
            'patient_satisfaction_index': 0.85,          # 患者满意度指数
            'treatment_success_rate': 0.9,               # 治疗成功率
            'average_wait_time': 0.2,                    # 平均等待时间
            'medical_safety_index': 0.95,                # 医疗安全指数
            
            # 教育伦理状态 (x₁₃-x₁₆)
            'ethical_compliance_score': 0.8,             # 伦理合规得分
            'resource_allocation_fairness': 0.85,        # 资源分配公平性
            'intern_learning_efficiency': 0.7,           # 实习生学习效率
            'knowledge_transfer_rate': 0.8               # 知识传递率
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
            agent_count = 5  # 默认5个智能体
            if hasattr(self, 'simulator') and self.simulator and hasattr(self.simulator, 'agents'):
                agent_count = len(self.simulator.agents)
            
            system_status = {
                'type': 'system_status',
                'simulation': {
                    'running': self.simulation_running,
                    'paused': self.simulation_paused,
                    'step': self.current_step,
                    'start_time': self.start_time.isoformat() if self.start_time else None
                },
                'agents_count': agent_count,
                'performance_metrics': self.performance_metrics,
                'integration_status': 'production' if HAS_CORE_ALGORITHMS else 'simulation',
                'architecture': 'Separated WebSocket Server + KallipolisSimulator',
                'timestamp': datetime.now().isoformat()
            }
            
            await self.send_to_client(websocket, system_status)
            
            # 发送初始规则数据
            await self._send_initial_rules(websocket)
            
            # 发送初始智能体状态
            await self._send_initial_agent_states(websocket)
        except Exception as e:
            logger.error(f"❌ 发送系统状态失败: {e}")
    
    async def _send_initial_agent_states(self, websocket):
        """发送初始智能体状态"""
        try:
            logger.info("🤖 开始发送初始智能体状态...")
            agent_configs = {
                'doctors': {'name': '医生群体', 'type': 'doctor'},
                'interns': {'name': '实习生群体', 'type': 'intern'},
                'patients': {'name': '患者代表', 'type': 'patient'},
                'accountants': {'name': '会计群体', 'type': 'accountant'},
                'government': {'name': '政府监管', 'type': 'government'}
            }
            
            for agent_id, config in agent_configs.items():
                await self.send_to_client(websocket, {
                    'type': 'agent_action',
                    'agent_id': agent_id,
                    'action': '系统初始化',
                    'reasoning': f'{config["name"]}已就绪，等待仿真开始',
                    'decision_layer': '基础模板',
                    'confidence': 1.0,
                    'agent_type': config['type'],
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"✅ 发送智能体状态: {agent_id} - {config['name']}")
                
        except Exception as e:
            logger.error(f"❌ 发送初始智能体状态失败: {e}")
    
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
        logger.info("🏗️ 多层决策架构: MADDPG → LLM → 控制器 → 数学策略 → 模板")
        
        # 创建仿真器实例
        config = SimulationConfig(
            max_steps=14,
            enable_learning=True,
            enable_llm_integration=True,
            enable_holy_code=True,
            enable_crises=True,
            enable_reward_control=True,
            meeting_interval=7  # 议会每7步召开一次
        )
        self.simulator = KallipolisSimulator(config)
        
        logger.info(f"✅ Simulator初始化完成 - 智能体数量: {len(self.simulator.agent_registry.get_all_agents()) if self.simulator.agent_registry else 0}")
        
        # 初始化完成后推送真实的神圣法典规则
        await self._push_real_holy_code_rules()
        
        # 启动仿真循环（异步运行）
        self.simulation_task = asyncio.create_task(self._run_simulation_steps())
    
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
    
    async def _send_initial_rules(self, websocket):
        """发送初始规则数据到客户端"""
        try:
            # 尝试从simulator获取规则数据
            if hasattr(self, 'simulator') and self.simulator and self.simulator.holy_code_manager:
                rules = self.simulator.holy_code_manager.rules
                active_rules = []
                all_rules = []
                
                for rule_id, rule_data in rules.items():
                    rule_info = {
                        'id': rule_id,
                        'name': rule_data.get('name', rule_id),
                        'description': rule_data.get('description', ''),
                        'priority': rule_data.get('priority', 1),
                        'context': rule_data.get('context', 'general'),
                        'active': rule_data.get('active', True)
                    }
                    all_rules.append(rule_info)
                    if rule_info['active']:
                        active_rules.append(rule_info)
                
                await self.send_to_client(websocket, {
                    'type': 'holy_code_rules',
                    'active_rules': active_rules,
                    'all_rules': all_rules,
                    'voting_results': [],
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # 发送模拟规则数据
                await self.send_to_client(websocket, {
                    'type': 'holy_code_rules', 
                    'active_rules': list(self.basic_rules.values())[:3],
                    'all_rules': list(self.basic_rules.values()),
                    'voting_results': [],
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"❌ 发送初始规则数据失败: {e}")
    
    async def _push_real_holy_code_rules(self):
        """推送真实的神圣法典规则"""
        try:
            if hasattr(self.simulator, 'holy_code_manager') and self.simulator.holy_code_manager:
                # HolyCodeManager的规则存储在rule_engine中
                if hasattr(self.simulator.holy_code_manager, 'rule_engine') and \
                   hasattr(self.simulator.holy_code_manager.rule_engine, 'rules'):
                    rules_dict = self.simulator.holy_code_manager.rule_engine.rules
                    
                    active_rules = []
                    all_rules = []
                    
                    for rule_id, rule_obj in rules_dict.items():
                        rule_info = {
                            'id': str(rule_obj.rule_id) if hasattr(rule_obj, 'rule_id') else str(rule_id),
                            'name': str(rule_obj.name) if hasattr(rule_obj, 'name') else str(rule_id),
                            'description': str(rule_obj.description) if hasattr(rule_obj, 'description') else '',
                            'priority': int(rule_obj.priority.value) if hasattr(rule_obj, 'priority') and hasattr(rule_obj.priority, 'value') else 3,
                            'context': rule_obj.context if hasattr(rule_obj, 'context') else ['general'],
                            'active': True,
                            'weight': float(rule_obj.weight) if hasattr(rule_obj, 'weight') else 1.0
                        }
                        all_rules.append(rule_info)
                        active_rules.append(rule_info)
                    
                    if all_rules:  # 只在有规则时推送
                        await self.broadcast({
                            'type': 'holy_code_rules',
                            'active_rules': active_rules,
                            'all_rules': all_rules,
                            'voting_results': [],
                            'timestamp': datetime.now().isoformat()
                        })
                        logger.info(f"✅ 推送了 {len(all_rules)} 条真实神圣法典规则")
                        return True
                    else:
                        logger.warning("⚠️ 规则字典为空")
                else:
                    logger.warning("⚠️ 未找到rule_engine.rules")
            else:
                logger.warning("⚠️ HolyCodeManager未初始化")
        except Exception as e:
            logger.error(f"❌ 推送真实规则失败: {e}")
        return False

    async def _run_simulation_steps(self):
        """执行仿真步骤循环"""
        try:
            step = 0
            while self.simulation_running and step < 14:
                if not self.simulation_paused:
                    # 执行单步仿真
                    step_result = self.simulator.step()
                    
                    # 处理仿真数据
                    await self.on_simulation_data(step_result)
                    
                    step += 1
                    
                    # 检查是否仿真完成
                    if step >= 14:
                        logger.info("🏁 仿真完成")
                        self.simulation_running = False
                        await self.broadcast({
                            'type': 'simulation_control',
                            'action': 'completed',
                            'timestamp': datetime.now().isoformat()
                        })
                        break
                
                # 等待间隔
                await asyncio.sleep(2)  # 每2秒执行一步
                
        except Exception as e:
            logger.error(f"❌ 仿真执行失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            
            # 回退到模拟模式
            logger.info("🔄 回退到模拟仿真模式")
            await self._start_mock_simulation()
    
    async def on_simulation_data(self, step_data: Dict[str, Any]):
        """处理来自仿真器的数据推送（多层决策架构）"""
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
            
            # 推送系统状态（16维）
            if 'system_state' in step_data:
                system_state = step_data['system_state']
                if isinstance(system_state, dict):
                    # 映射仿真器状态到16维状态空间
                    state_mapping = {
                        # 物理资源状态 (x₁-x₄)
                        'bed_occupancy_rate': system_state.get('medical_resource_utilization', 0.7),
                        'medical_equipment_utilization': system_state.get('operational_efficiency', 0.8),
                        'staff_utilization_rate': system_state.get('staff_workload_balance', 0.6),
                        'medication_inventory_level': system_state.get('crisis_response_capability', 0.9),
                        
                        # 财务状态 (x₅-x₈)
                        'cash_reserve_ratio': system_state.get('financial_indicator', 0.8),
                        'operating_margin': system_state.get('financial_indicator', 0.1),
                        'debt_to_asset_ratio': 0.3,  # 默认值，如果仿真器没有提供
                        'cost_efficiency_index': system_state.get('operational_efficiency', 0.75),
                        
                        # 服务质量状态 (x₉-x₁₂)
                        'patient_satisfaction_index': system_state.get('patient_satisfaction', 0.85),
                        'treatment_success_rate': system_state.get('care_quality_index', 0.9),
                        'average_wait_time': system_state.get('patient_waiting_time', 0.2),
                        'medical_safety_index': system_state.get('safety_incident_rate', 0.95),
                        
                        # 教育伦理状态 (x₁₃-x₁₆)
                        'ethical_compliance_score': system_state.get('ethical_compliance', 0.8),
                        'resource_allocation_fairness': system_state.get('regulatory_compliance_score', 0.85),
                        'intern_learning_efficiency': system_state.get('education_training_quality', 0.7),
                        'knowledge_transfer_rate': system_state.get('professional_development', 0.8)
                    }
                    
                    # 更新性能指标
                    for metric, value in state_mapping.items():
                        if metric in self.performance_metrics:
                            self.performance_metrics[metric] = float(value)
                
                await self.broadcast({
                    'type': 'system_state',
                    'state': state_mapping,
                    'timestamp': datetime.now().isoformat()
                })
            
            # 推送智能体行动（支持多层决策）
            if 'actions' in step_data:
                for agent_id, action_data in step_data['actions'].items():
                    # 检测决策层级
                    reasoning = action_data.get('reasoning', '')
                    decision_layer = 'Unknown'
                    if 'MADDPG' in reasoning:
                        decision_layer = 'MADDPG深度强化学习'
                    elif 'LLM' in reasoning:
                        decision_layer = 'LLM智能生成'
                    elif '控制' in reasoning:
                        decision_layer = '分布式控制器'
                    elif '数学' in reasoning:
                        decision_layer = '数学策略模板'
                    else:
                        decision_layer = '基础模板'
                    
                    await self.broadcast({
                        'type': 'agent_action',
                        'agent_id': agent_id,
                        'action': action_data.get('action', 'Unknown action'),
                        'reasoning': reasoning,
                        'decision_layer': decision_layer,
                        'confidence': action_data.get('confidence', 0.8),
                        'strategy_params': action_data.get('strategy_params', []),
                        'agent_type': action_data.get('agent_type', 'Unknown'),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # 推送性能指标
            if 'metrics' in step_data:
                metrics = step_data['metrics']
                combined_metrics = {
                    **metrics,
                    'overall_performance': metrics.get('overall_performance', 0.5),
                    'system_stability': metrics.get('system_stability', 0.8),
                    'crisis_count': metrics.get('crisis_count', 0),
                    'parliament_meetings': metrics.get('parliament_meetings', 0),
                    'consensus_efficiency': metrics.get('consensus_efficiency', 0.5)
                }
                
                await self.broadcast({
                    'type': 'metrics',
                    **combined_metrics,
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
            
            # 推送议会会议结果
            if step_data.get('parliament_meeting', False):
                await self.broadcast({
                    'type': 'parliament_meeting',
                    'parliament_result': {
                        'consensus': {
                            'consensus_level': step_data.get('metrics', {}).get('consensus_efficiency', 0.5),
                            'main_decision': '议会通过医院治理优化决议'
                        }
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            # 推送神圣法典规则（从holy_code_manager获取）
            if hasattr(self.simulator, 'holy_code_manager') and self.simulator.holy_code_manager:
                try:
                    # HolyCodeManager的规则存储在rule_engine中
                    if hasattr(self.simulator.holy_code_manager, 'rule_engine') and \
                       hasattr(self.simulator.holy_code_manager.rule_engine, 'rules'):
                        rules_dict = self.simulator.holy_code_manager.rule_engine.rules
                        
                        active_rules = []
                        all_rules = []
                        
                        for rule_id, rule_obj in rules_dict.items():
                            rule_info = {
                                'id': str(rule_obj.rule_id) if hasattr(rule_obj, 'rule_id') else str(rule_id),
                                'name': str(rule_obj.name) if hasattr(rule_obj, 'name') else str(rule_id),
                                'description': str(rule_obj.description) if hasattr(rule_obj, 'description') else '',
                                'priority': int(rule_obj.priority.value) if hasattr(rule_obj, 'priority') and hasattr(rule_obj.priority, 'value') else 3,
                                'context': rule_obj.context if hasattr(rule_obj, 'context') else ['general'],
                                'active': True,
                                'weight': float(rule_obj.weight) if hasattr(rule_obj, 'weight') else 1.0
                            }
                            all_rules.append(rule_info)
                            active_rules.append(rule_info)
                        
                        if all_rules:  # 只在有规则时推送
                            await self.broadcast({
                                'type': 'holy_code_rules',
                                'active_rules': active_rules,
                                'all_rules': all_rules,
                                'voting_results': [],
                                'timestamp': datetime.now().isoformat()
                            })
                            logger.info(f"✅ 推送了 {len(all_rules)} 条真实神圣法典规则")
                    else:
                        logger.warning("⚠️ 未找到rule_engine.rules")
                except Exception as rule_error:
                    logger.warning(f"⚠️ 规则数据处理失败: {rule_error}")
                    # 不影响其他数据推送
                
        except Exception as e:
            logger.error(f"❌ 处理仿真数据推送失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
    
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
            # 处理numpy数据类型和其他不可序列化的对象
            import json
            
            def convert_numpy(obj):
                if hasattr(obj, 'tolist'):  # numpy数组
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # numpy标量
                    return obj.item()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            cleaned_message = convert_numpy(message)
            await websocket.send(json.dumps(cleaned_message))
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(websocket)
        except Exception as e:
            logger.error(f"❌ 发送消息失败: {e}")
            # 尝试发送一个简化的错误消息
            try:
                error_msg = {
                    'type': 'error',
                    'message': f'数据发送失败: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send(json.dumps(error_msg))
            except:
                pass
    
    async def _send_initial_rules(self, websocket):
        """发送初始规则数据到客户端"""
        try:
            # 尝试从simulator获取规则数据
            if hasattr(self, 'simulator') and self.simulator and \
               hasattr(self.simulator, 'holy_code_manager') and self.simulator.holy_code_manager:
                
                # HolyCodeManager的规则存储在rule_engine中
                if hasattr(self.simulator.holy_code_manager, 'rule_engine') and \
                   hasattr(self.simulator.holy_code_manager.rule_engine, 'rules'):
                    rules_dict = self.simulator.holy_code_manager.rule_engine.rules
                    
                    active_rules = []
                    all_rules = []
                    
                    for rule_id, rule_obj in rules_dict.items():
                        rule_info = {
                            'id': str(rule_obj.rule_id) if hasattr(rule_obj, 'rule_id') else str(rule_id),
                            'name': str(rule_obj.name) if hasattr(rule_obj, 'name') else str(rule_id),
                            'description': str(rule_obj.description) if hasattr(rule_obj, 'description') else '',
                            'priority': int(rule_obj.priority.value) if hasattr(rule_obj, 'priority') and hasattr(rule_obj.priority, 'value') else 3,
                            'context': rule_obj.context if hasattr(rule_obj, 'context') else ['general'],
                            'active': True,
                            'weight': float(rule_obj.weight) if hasattr(rule_obj, 'weight') else 1.0
                        }
                        all_rules.append(rule_info)
                        active_rules.append(rule_info)
                    
                    await self.send_to_client(websocket, {
                        'type': 'holy_code_rules',
                        'active_rules': active_rules,
                        'all_rules': all_rules,
                        'voting_results': [],
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.info(f"✅ 发送了 {len(all_rules)} 条神圣法典规则")
                    return
                else:
                    logger.warning("⚠️ 未找到rule_engine.rules")
            
            # 发送模拟规则数据
            active_rules = [
                {
                    'id': 'mock_rule_1',
                    'name': '患者安全协议',
                    'description': '确保患者安全的基本协议',
                    'priority': 1,
                    'context': ['medical'],
                    'active': True,
                    'weight': 1.0
                },
                {
                    'id': 'mock_rule_2',
                    'name': '资源分配规则',
                    'description': '优化医疗资源分配',
                    'priority': 2,
                    'context': ['resource'],
                    'active': True,
                    'weight': 0.8
                }
            ]
            all_rules = active_rules + [
                {
                    'id': 'mock_rule_3',
                    'name': '质量控制标准',
                    'description': '医疗质量控制标准',
                    'priority': 3,
                    'context': ['quality'],
                    'active': False,
                    'weight': 0.6
                }
            ]
            
            await self.send_to_client(websocket, {
                'type': 'holy_code_rules', 
                'active_rules': active_rules,
                'all_rules': all_rules,
                'voting_results': [],
                'timestamp': datetime.now().isoformat()
            })
            logger.info("✅ 发送了模拟神圣法典规则")
            
        except Exception as e:
            logger.error(f"❌ 发送初始规则数据失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
    
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