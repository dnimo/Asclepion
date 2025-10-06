#!/usr/bin/env python3
"""
WebSocket演示服务器 - 集成真实算法
专门用于演示系统集成的简化版本
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Set
import websockets
import numpy as np
from pathlib import Path
import sys
import yaml

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 简化的导入 - 逐步检查可用性
HAS_HOSPITAL_SYSTEM = False
HAS_CONTROLLER = False
HAS_RULE_ENGINE = False
HAS_LLM = False

try:
    from src.hospital_governance.core.hospital_system import HospitalSystem
    HAS_HOSPITAL_SYSTEM = True
    logger.info("✅ 医院系统模块导入成功")
except ImportError as e:
    logger.warning(f"❌ 医院系统模块导入失败: {e}")

try:
    from src.hospital_governance.core.multi_agent_controller import MultiAgentController
    HAS_CONTROLLER = True
    logger.info("✅ 多智能体控制器导入成功")
except ImportError as e:
    logger.warning(f"❌ 多智能体控制器导入失败: {e}")

try:
    from src.hospital_governance.core.holy_code_engine import SimpleRuleEngine
    HAS_RULE_ENGINE = True
    logger.info("✅ 规则引擎导入成功")
except ImportError as e:
    logger.warning(f"❌ 规则引擎导入失败: {e}")

try:
    from src.hospital_governance.agents.llm_action_generator import MockLLMProvider, LLMConfig
    HAS_LLM = True
    logger.info("✅ LLM模块导入成功")
except ImportError as e:
    logger.warning(f"❌ LLM模块导入失败: {e}")

class HospitalDemoServer:
    """医院演示WebSocket服务器"""
    
    def __init__(self, host="localhost", port=8000):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.simulation_running = False
        self.simulation_paused = False
        self.current_step = 0
        self.start_time = None
        
        # 初始化可用的系统组件
        self.initialize_available_systems()
        
    def initialize_available_systems(self):
        """初始化可用的系统组件"""
        global HAS_HOSPITAL_SYSTEM, HAS_CONTROLLER, HAS_RULE_ENGINE, HAS_LLM
        integration_level = 0
        
        # 初始化医院系统
        if HAS_HOSPITAL_SYSTEM:
            try:
                self.hospital_system = HospitalSystem()
                self.system_state = self.hospital_system.get_state()
                integration_level += 1
                logger.info("✅ 医院系统初始化成功")
            except Exception as e:
                logger.error(f"医院系统初始化失败: {e}")
                HAS_HOSPITAL_SYSTEM = False
        
        if not HAS_HOSPITAL_SYSTEM:
            # 使用模拟状态
            self.system_state = np.array([0.7, 0.6, 0.65, 0.8, 0.9, 0.85, 0.2])
            logger.info("🔄 使用模拟医院系统状态")
        
        # 初始化控制器
        if HAS_CONTROLLER:
            try:
                self.controller = MultiAgentController()
                integration_level += 1
                logger.info("✅ 多智能体控制器初始化成功")
            except Exception as e:
                logger.error(f"控制器初始化失败: {e}")
        
        # 初始化规则引擎
        if HAS_RULE_ENGINE:
            try:
                self.rule_engine = SimpleRuleEngine()
                integration_level += 1
                logger.info("✅ 规则引擎初始化成功")
            except Exception as e:
                logger.error(f"规则引擎初始化失败: {e}")
        
        # 初始化LLM
        if HAS_LLM:
            try:
                self.llm_provider = MockLLMProvider(LLMConfig())
                integration_level += 1
                logger.info("✅ LLM提供者初始化成功")
            except Exception as e:
                logger.error(f"LLM提供者初始化失败: {e}")
        
        # 计算集成程度
        total_components = 4
        self.integration_level = integration_level / total_components
        logger.info(f"🎯 系统集成程度: {self.integration_level:.1%} ({integration_level}/{total_components})")
        
        # 初始化智能体配置
        self.initialize_agents()
        
        # 初始化性能指标
        self.performance_metrics = {
            'stability': 0.8,
            'performance': 0.75,
            'efficiency': 0.7,
            'safety': 0.9,
            'integration_level': self.integration_level
        }
    
    def initialize_agents(self):
        """初始化智能体配置"""
        if self.integration_level >= 0.5:
            # 高集成度 - 使用真实智能体角色
            self.agents = {
                'doctor': {
                    'name': '医生',
                    'role': 'doctor',
                    'last_action_time': 0,
                    'current_task': None,
                    'performance_score': 0.8,
                    'actions': ['诊断患者', '制定治疗方案', '医疗会诊', '手术决策', '药物调整']
                },
                'intern': {
                    'name': '实习医生',
                    'role': 'intern', 
                    'last_action_time': 0,
                    'current_task': None,
                    'performance_score': 0.6,
                    'actions': ['学习观摩', '辅助治疗', '记录病历', '基础检查', '跟随指导']
                },
                'patient': {
                    'name': '患者代表',
                    'role': 'patient',
                    'last_action_time': 0,
                    'current_task': None,
                    'performance_score': 0.7,
                    'actions': ['配合治疗', '反馈症状', '遵循医嘱', '康复训练', '投诉建议']
                },
                'accountant': {
                    'name': '会计',
                    'role': 'accountant',
                    'last_action_time': 0,
                    'current_task': None,
                    'performance_score': 0.75,
                    'actions': ['成本分析', '预算管理', '费用审核', '财务报告', '资源优化']
                },
                'government': {
                    'name': '政府监管',
                    'role': 'government',
                    'last_action_time': 0,
                    'current_task': None,
                    'performance_score': 0.85,
                    'actions': ['政策监管', '质量检查', '合规审计', '资格认证', '标准制定']
                }
            }
        else:
            # 低集成度 - 使用模拟智能体
            self.agents = {
                'senior_doctor': {
                    'name': '主治医生',
                    'actions': ['诊断患者', '制定治疗方案', '紧急救治', '医疗会诊', '监督实习医生'],
                    'last_action_time': 0
                },
                'head_nurse': {
                    'name': '护士长',
                    'actions': ['分配护理任务', '监控患者状态', '协调医护配合', '紧急响应', '质量控制'],
                    'last_action_time': 0
                }
            }
        
        # 初始化规则系统
        self.rules = {
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
            'quality_assurance_rule': {
                'name': '质量保证规则',
                'description': '医疗质量监控和保证',
                'activated': False,
                'severity': 0.0
            }
        }
    
    async def register_client(self, websocket, path):
        """注册新客户端"""
        self.clients.add(websocket)
        logger.info(f"客户端已连接: {websocket.remote_address}")
        
        # 发送初始状态和集成信息
        await self.send_to_client(websocket, {
            'type': 'init',
            'message': '欢迎连接医院治理系统监控服务器',
            'integration_level': self.integration_level,
            'components': {
                'hospital_system': HAS_HOSPITAL_SYSTEM,
                'controller': HAS_CONTROLLER,
                'rule_engine': HAS_RULE_ENGINE,
                'llm': HAS_LLM
            },
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            # 处理客户端消息
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"客户端已断开: {websocket.remote_address}")
    
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
                logger.warning(f"未知命令: {command}")
                
        except json.JSONDecodeError:
            logger.error("无效的JSON消息")
        except Exception as e:
            logger.error(f"处理客户端消息时出错: {e}")
    
    async def start_simulation(self):
        """开始仿真"""
        if not self.simulation_running:
            self.simulation_running = True
            self.simulation_paused = False
            self.start_time = datetime.now()
            logger.info("仿真已启动")
            
            await self.broadcast({
                'type': 'simulation_control',
                'action': 'started',
                'integration_level': self.integration_level,
                'timestamp': datetime.now().isoformat()
            })
            
            # 启动仿真循环
            asyncio.create_task(self.simulation_loop())
    
    async def pause_simulation(self):
        """暂停仿真"""
        if self.simulation_running:
            self.simulation_paused = not self.simulation_paused
            action = 'paused' if self.simulation_paused else 'resumed'
            logger.info(f"仿真已{action}")
            
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
        
        # 重置状态
        if HAS_HOSPITAL_SYSTEM and hasattr(self, 'hospital_system'):
            try:
                self.hospital_system.reset()
                self.system_state = self.hospital_system.get_state()
            except:
                self.system_state = np.array([0.7, 0.6, 0.65, 0.8, 0.9, 0.85, 0.2])
        else:
            self.system_state = np.array([0.7, 0.6, 0.65, 0.8, 0.9, 0.85, 0.2])
            
        # 重置规则
        for rule in self.rules.values():
            rule['activated'] = False
            rule['severity'] = 0.0
            
        logger.info("仿真已重置")
        
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
            'integration_level': self.integration_level,
            'timestamp': datetime.now().isoformat()
        })
    
    async def simulation_loop(self):
        """仿真主循环"""
        while self.simulation_running:
            if not self.simulation_paused:
                await self.simulation_step()
                self.current_step += 1
            
            await asyncio.sleep(2)  # 每2秒一步
    
    async def simulation_step(self):
        """执行一步仿真"""
        # 根据集成程度选择仿真方式
        if self.integration_level >= 0.5:
            await self.integrated_simulation_step()
        else:
            await self.mock_simulation_step()
        
        # 发送步骤信息
        await self.broadcast({
            'type': 'simulation_step',
            'step': self.current_step,
            'integration_level': self.integration_level,
            'timestamp': datetime.now().isoformat()
        })
    
    async def integrated_simulation_step(self):
        """集成算法的仿真步骤"""
        try:
            # 1. 获取当前状态
            if HAS_HOSPITAL_SYSTEM and hasattr(self, 'hospital_system'):
                current_state = self.hospital_system.get_state()
            else:
                current_state = self.system_state
            
            # 2. 智能体决策
            agent_decisions = {}
            
            for agent_id, agent_info in self.agents.items():
                # 生成智能体决策
                if HAS_LLM and hasattr(self, 'llm_provider'):
                    try:
                        decision_prompt = f"作为{agent_info['name']}，当前系统状态为{current_state.tolist()[:3]}，你需要做什么决策？"
                        llm_response = await self.llm_provider.generate_text(
                            decision_prompt, 
                            context={'role': agent_info.get('role', 'unknown')}
                        )
                        
                        action = self.parse_action_from_response(llm_response, agent_info.get('role', 'unknown'))
                        reasoning = llm_response[:100] + "..." if len(llm_response) > 100 else llm_response
                    except Exception as e:
                        logger.warning(f"LLM决策失败: {e}")
                        action = np.random.choice(agent_info.get('actions', ['执行任务']))
                        reasoning = "基于系统状态的标准决策"
                else:
                    action = np.random.choice(agent_info.get('actions', ['执行任务']))
                    reasoning = "基于系统状态的标准决策"
                
                decision = {
                    'action': action,
                    'reasoning': reasoning,
                    'confidence': 0.7 + np.random.random() * 0.3
                }
                
                agent_decisions[agent_id] = decision
                agent_info['current_task'] = action
                
                # 广播智能体活动
                await self.broadcast({
                    'type': 'agent_action',
                    'agent_id': agent_info['name'],
                    'action': action,
                    'reasoning': reasoning,
                    'confidence': decision['confidence'],
                    'integration_mode': 'real' if HAS_LLM else 'simulated',
                    'timestamp': datetime.now().isoformat()
                })
            
            # 3. 应用控制器
            if HAS_CONTROLLER and hasattr(self, 'controller'):
                try:
                    control_input = self.controller.compute_control(current_state, agent_decisions)
                except Exception as e:
                    logger.warning(f"控制计算失败: {e}")
                    control_input = np.zeros(len(current_state))
            else:
                # 模拟控制输入
                control_input = np.random.normal(0, 0.1, len(current_state))
            
            # 4. 更新系统状态
            if HAS_HOSPITAL_SYSTEM and hasattr(self, 'hospital_system'):
                try:
                    new_state = self.hospital_system.update(control_input)
                    self.system_state = new_state
                except Exception as e:
                    logger.warning(f"系统更新失败: {e}")
                    await self.update_mock_state()
            else:
                await self.update_mock_state()
            
            # 5. 检查规则激活
            await self.check_rule_activations()
            
            # 6. 更新性能指标
            self.update_performance_metrics()
            
            # 7. 广播更新
            await self.broadcast({
                'type': 'metrics',
                **self.performance_metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            await self.broadcast({
                'type': 'system_state',
                'state': self.system_state.tolist(),
                'timestamp': datetime.now().isoformat()
            })
            
            # 8. 生成智能体对话
            await self.generate_agent_dialogs(agent_decisions)
            
        except Exception as e:
            logger.error(f"集成仿真步骤执行失败: {e}")
            await self.mock_simulation_step()
    
    async def mock_simulation_step(self):
        """模拟仿真步骤"""
        # 更新系统状态
        await self.update_mock_state()
        
        # 生成智能体活动
        await self.generate_mock_agent_activities()
        
        # 检查规则激活
        await self.check_rule_activations()
        
        # 生成智能体对话
        await self.generate_mock_agent_dialogs()
    
    async def update_mock_state(self):
        """更新模拟系统状态"""
        # 添加随机波动
        noise = np.random.normal(0, 0.05, len(self.system_state))
        self.system_state += noise
        self.system_state = np.clip(self.system_state, 0, 1)
        
        # 更新性能指标
        self.update_performance_metrics()
        
        # 广播更新
        await self.broadcast({
            'type': 'metrics',
            **self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        await self.broadcast({
            'type': 'system_state',
            'state': self.system_state.tolist(),
            'timestamp': datetime.now().isoformat()
        })
    
    async def generate_mock_agent_activities(self):
        """生成模拟智能体活动"""
        current_time = time.time()
        
        for agent_id, agent_info in self.agents.items():
            if current_time - agent_info['last_action_time'] > np.random.exponential(5):
                action = np.random.choice(agent_info['actions'])
                
                await self.broadcast({
                    'type': 'agent_action',
                    'agent_id': agent_info['name'],
                    'action': action,
                    'reasoning': f"基于当前系统状态的决策",
                    'confidence': 0.7 + np.random.random() * 0.3,
                    'integration_mode': 'mock',
                    'timestamp': datetime.now().isoformat()
                })
                
                agent_info['last_action_time'] = current_time
    
    def parse_action_from_response(self, llm_response: str, role: str) -> str:
        """从LLM响应中解析行动"""
        role_actions = {
            'doctor': ['诊断患者', '制定治疗方案', '医疗会诊', '手术决策', '药物调整'],
            'intern': ['学习观摩', '辅助治疗', '记录病历', '基础检查', '跟随指导'],
            'patient': ['配合治疗', '反馈症状', '遵循医嘱', '康复训练', '投诉建议'],
            'accountant': ['成本分析', '预算管理', '费用审核', '财务报告', '资源优化'],
            'government': ['政策监管', '质量检查', '合规审计', '资格认证', '标准制定']
        }
        
        actions = role_actions.get(role, ['执行任务', '监控状态', '协调合作'])
        
        # 简单的关键词匹配
        for action in actions:
            if any(keyword in llm_response for keyword in action.split()):
                return action
        
        # 默认返回第一个行动
        return actions[0]
    
    def update_performance_metrics(self):
        """更新性能指标"""
        try:
            if len(self.system_state) >= 7:
                self.performance_metrics['stability'] = max(0.1, 1 - np.std(self.system_state))
                self.performance_metrics['performance'] = np.mean(self.system_state[:5])
                self.performance_metrics['efficiency'] = max(0.1, 1 - np.mean(self.system_state[1:3]))
                self.performance_metrics['safety'] = self.system_state[5]
            
            # 集成程度影响性能
            self.performance_metrics['integration_level'] = self.integration_level
            
        except Exception as e:
            logger.error(f"性能指标更新失败: {e}")
    
    async def check_rule_activations(self):
        """检查规则激活"""
        if HAS_RULE_ENGINE and hasattr(self, 'rule_engine'):
            try:
                # 使用真实规则引擎
                rule_results = self.rule_engine.apply_rules({
                    'system_state': self.system_state,
                    'performance_metrics': self.performance_metrics,
                    'current_step': self.current_step
                })
                
                for rule_name, result in rule_results.items():
                    if result.get('activated', False):
                        await self.broadcast({
                            'type': 'rule_activation',
                            'rule_name': rule_name,
                            'activated': True,
                            'severity': result.get('severity', 0.5),
                            'description': result.get('description', f'{rule_name}被激活'),
                            'integration_mode': 'real',
                            'timestamp': datetime.now().isoformat()
                        })
                        
            except Exception as e:
                logger.warning(f"规则引擎检查失败: {e}")
                await self.mock_rule_check()
        else:
            await self.mock_rule_check()
    
    async def mock_rule_check(self):
        """模拟规则检查"""
        # 患者安全协议
        if self.performance_metrics['safety'] < 0.7:
            self.rules['patient_safety_protocol']['activated'] = True
            self.rules['patient_safety_protocol']['severity'] = 1 - self.performance_metrics['safety']
            
            await self.broadcast({
                'type': 'rule_activation',
                'rule_name': '患者安全协议',
                'activated': True,
                'severity': self.rules['patient_safety_protocol']['severity'],
                'description': '患者安全指标低于阈值',
                'integration_mode': 'mock',
                'timestamp': datetime.now().isoformat()
            })
        
        # 资源分配规则
        if len(self.system_state) > 3 and self.system_state[3] < 0.3:
            self.rules['resource_allocation_rule']['activated'] = True
            self.rules['resource_allocation_rule']['severity'] = 0.3 - self.system_state[3]
            
            await self.broadcast({
                'type': 'rule_activation',
                'rule_name': '资源分配规则',
                'activated': True,
                'severity': self.rules['resource_allocation_rule']['severity'],
                'description': '医疗资源不足需要重新分配',
                'integration_mode': 'mock',
                'timestamp': datetime.now().isoformat()
            })
    
    async def generate_agent_dialogs(self, agent_decisions: Dict):
        """基于真实决策生成智能体对话"""
        if np.random.random() < 0.4:
            participating_agents = np.random.choice(
                list(agent_decisions.keys()),
                size=min(3, len(agent_decisions)),
                replace=False
            )
            
            actions = [agent_decisions[agent]['action'] for agent in participating_agents]
            participant_names = [self.agents[agent]['name'] for agent in participating_agents]
            
            dialog_content = f"讨论协调: {', '.join(actions[:2])}等医疗决策"
            
            await self.broadcast({
                'type': 'dialog',
                'participants': participant_names,
                'content': dialog_content,
                'integration_mode': 'real' if len(agent_decisions) > 0 else 'mock',
                'timestamp': datetime.now().isoformat()
            })
    
    async def generate_mock_agent_dialogs(self):
        """生成模拟智能体对话"""
        if np.random.random() < 0.3:
            participants = np.random.choice(
                list(self.agents.keys()), 
                size=np.random.randint(2, min(4, len(self.agents))), 
                replace=False
            )
            
            dialog_templates = [
                "讨论当前患者分流策略和资源分配方案",
                "协调紧急医疗响应，确保患者安全",
                "评估医疗质量指标，制定改进措施",
                "分析工作负荷分布，优化人员配置"
            ]
            
            content = np.random.choice(dialog_templates)
            
            await self.broadcast({
                'type': 'dialog',
                'participants': [self.agents[p]['name'] for p in participants],
                'content': content,
                'integration_mode': 'mock',
                'timestamp': datetime.now().isoformat()
            })
    
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
            logger.error(f"发送消息失败: {e}")
    
    async def start_server(self):
        """启动WebSocket服务器"""
        logger.info(f"启动WebSocket服务器: {self.host}:{self.port}")
        
        async with websockets.serve(
            self.register_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        ):
            logger.info("WebSocket服务器已启动")
            await asyncio.Future()  # 保持运行

async def main():
    """主函数"""
    server = HospitalDemoServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"服务器错误: {e}")

if __name__ == "__main__":
    print("🏥 医院治理系统 - 集成演示WebSocket服务器")
    print("=" * 70)
    print("正在检查系统组件...")
    
    # 组件状态预检
    components = {
        '医院系统': HAS_HOSPITAL_SYSTEM,
        '多智能体控制器': HAS_CONTROLLER, 
        '规则引擎': HAS_RULE_ENGINE,
        'LLM模块': HAS_LLM
    }
    
    print("\n📋 组件状态:")
    for component, status in components.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}: {'可用' if status else '不可用'}")
    
    integration_score = sum(components.values()) / len(components)
    print(f"\n🎯 预期集成程度: {integration_score:.1%}")
    
    if integration_score >= 0.75:
        print("🚀 系统准备就绪，使用高度集成模式")
    elif integration_score >= 0.5:
        print("⚡ 系统部分就绪，使用混合模式")
    else:
        print("🔄 系统使用模拟模式")
    
    print("\n🌐 服务器信息:")
    print("前端界面: http://localhost:8000/frontend/websocket_demo.html")
    print("WebSocket端点: ws://localhost:8000")
    print("按 Ctrl+C 停止服务器")
    print("=" * 70)
    
    asyncio.run(main())