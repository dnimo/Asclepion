import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
import time
import json
from dataclasses import dataclass

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """仿真配置"""
    max_steps: int = 1000
    time_scale: float = 1.0
    meeting_interval: int = 168  # 每168小时（一周）举行议会
    enable_learning: bool = True
    enable_holy_code: bool = True
    enable_crises: bool = True
    enable_behavior_models: bool = True
    enable_llm_integration: bool = False
    enable_scenario_runner: bool = True
    data_logging_interval: int = 10
    stability_check_interval: int = 50
    crisis_probability: float = 0.03
    
    # 集成配置
    holy_code_config: Optional[Any] = None
    interaction_config: Optional[Any] = None
    scenario_config: Optional[Any] = None

class KallipolisSimulator:
    """Kallipolis医疗共和国模拟器
    
    仿真循环的主体，负责：
    1. 系统状态更新
    2. 智能体决策
    3. 议会会议
    4. 危机处理
    5. 数据推送
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        
        # 仿真状态
        self.current_step = 0
        self.simulation_time = 0.0
        self.is_running = False
        self.is_paused = False
        
        # 数据回调机制
        self.data_callback: Optional[Callable] = None
        
        # 历史记录
        self.decision_history = []
        self.interaction_history = []
        self.crisis_history = []
        self.performance_history = []
        self.parliament_history = []
        
        # 核心组件（延迟初始化，避免导入错误）
        self.role_manager = None
        self.holy_code_manager = None
        self.learning_model = None
        self.interaction_engine = None
        self.coordinator = None
        
        # 系统状态 - 对应16维医院治理系统
        self.system_state = {
            # 医疗质量维度
            'medical_quality': 0.85,
            'patient_safety': 0.9,
            'care_quality': 0.8,
            
            # 资源管理维度
            'resource_adequacy': 0.7,
            'resource_utilization': 0.75,
            'resource_access': 0.8,
            
            # 教育培训维度
            'education_quality': 0.7,
            'training_hours': 30.0,
            'mentorship_availability': 0.6,
            'career_development': 0.65,
            
            # 财务健康维度
            'financial_health': 0.8,
            'cost_efficiency': 0.75,
            'revenue_growth': 0.7,
            
            # 患者服务维度
            'patient_satisfaction': 0.75,
            'accessibility': 0.8,
            'waiting_times': 0.3,  # 反向指标，越低越好
            
            # 员工福利维度
            'staff_satisfaction': 0.7,
            'workload': 0.6,  # 适中的工作负荷
            'salary_satisfaction': 0.65,
            
            # 系统治理维度
            'system_stability': 0.8,
            'ethics_compliance': 0.85,
            'regulatory_compliance': 0.9,
            'public_trust': 0.75,
            
            # 综合指标
            'overall_performance': 0.78,
            'crisis_severity': 0.0
        }
        
        # 智能体状态
        self.agents = {
            'doctors': {
                'name': '医生群体',
                'performance': 0.8,
                'last_decision': None,
                'payoff': 0.0,
                'strategy_params': np.random.uniform(0.3, 0.7, 3),
                'active': True
            },
            'interns': {
                'name': '实习生群体',
                'performance': 0.7,
                'last_decision': None,
                'payoff': 0.0,
                'strategy_params': np.random.uniform(0.2, 0.6, 3),
                'active': True
            },
            'patients': {
                'name': '患者代表',
                'performance': 0.75,
                'last_decision': None,
                'payoff': 0.0,
                'strategy_params': np.random.uniform(0.4, 0.8, 3),
                'active': True
            },
            'accountants': {
                'name': '会计群体',
                'performance': 0.8,
                'last_decision': None,
                'payoff': 0.0,
                'strategy_params': np.random.uniform(0.5, 0.9, 3),
                'active': True
            },
            'government': {
                'name': '政府监管',
                'performance': 0.75,
                'last_decision': None,
                'payoff': 0.0,
                'strategy_params': np.random.uniform(0.3, 0.7, 3),
                'active': True
            }
        }
        
        logger.info("🏥 KallipolisSimulator初始化完成")
    
    def set_data_callback(self, callback: Callable):
        """设置数据推送回调函数"""
        self.data_callback = callback
        logger.info("📡 数据回调已设置")
    
    def step(self, training: bool = False) -> Dict[str, Any]:
        """执行一个仿真步骤"""
        if not self.is_running:
            self.is_running = True
        
        self.current_step += 1
        self.simulation_time += self.config.time_scale
        
        # 初始化步骤数据
        step_data = {
            'step': self.current_step,
            'time': self.simulation_time,
            'system_state': self.system_state.copy(),
            'observations': {},
            'actions': {},
            'rewards': {},
            'decisions': {},
            'metrics': {},
            'crises': [],
            'parliament_meeting': False
        }
        
        try:
            # 1. 检查议会会议周期
            if self.current_step % self.config.meeting_interval == 0 and self.current_step > 0:
                parliament_result = self._run_parliament_meeting()
                step_data['parliament_meeting'] = True
                step_data['parliament_result'] = parliament_result
                logger.info(f"🏛️ 议会会议在第{self.current_step}步举行")
            
            # 2. 模拟系统动态变化
            self._simulate_system_dynamics()
            
            # 3. 智能体决策和行动
            agent_actions = self._simulate_agent_decisions()
            step_data['actions'] = agent_actions
            
            # 4. 处理危机事件
            if self.config.enable_crises:
                crises = self._handle_random_crises()
                step_data['crises'] = crises
            
            # 5. 计算性能指标
            metrics = self._calculate_performance_metrics()
            step_data['metrics'] = metrics
            self.performance_history.append(metrics)
            
            # 6. 更新智能体收益
            self._update_agent_payoffs(step_data)
            
        except Exception as e:
            logger.error(f"❌ 仿真步骤执行错误: {e}")
            import traceback
            traceback.print_exc()
        
        # 推送数据到回调
        if self.data_callback:
            try:
                if asyncio.iscoroutinefunction(self.data_callback):
                    # 异步回调
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self.data_callback(step_data))
                        else:
                            asyncio.run(self.data_callback(step_data))
                    except RuntimeError:
                        # 没有事件循环，同步调用
                        asyncio.run(self.data_callback(step_data))
                else:
                    # 同步回调
                    self.data_callback(step_data)
            except Exception as e:
                logger.error(f"❌ 数据回调执行失败: {e}")
        
        return step_data
    
    def _run_parliament_meeting(self) -> Dict[str, Any]:
        """运行议会会议"""
        try:
            # 1. 收集智能体提案
            proposals = {}
            for agent_id, agent_info in self.agents.items():
                if agent_info['active']:
                    proposal = self._generate_agent_proposal(agent_id, agent_info)
                    proposals[agent_id] = proposal
            
            # 2. 模拟议会投票和共识达成
            consensus_result = self._simulate_parliament_consensus(proposals)
            
            # 3. 将共识写入神圣法典（模拟）
            holy_code_update = self._update_holy_code(consensus_result)
            
            # 4. 计算本次收益
            meeting_rewards = self._calculate_meeting_rewards(consensus_result)
            
            # 5. 更新智能体网络（模拟actor-critic更新）
            self._update_agent_networks(meeting_rewards)
            
            # 记录议会历史
            parliament_record = {
                'step': self.current_step,
                'proposals': proposals,
                'consensus': consensus_result,
                'holy_code_update': holy_code_update,
                'rewards': meeting_rewards,
                'timestamp': time.time()
            }
            self.parliament_history.append(parliament_record)
            
            logger.info(f"🏛️ 议会会议完成，达成共识: {consensus_result['consensus_level']:.2f}")
            return parliament_record
            
        except Exception as e:
            logger.error(f"❌ 议会会议执行失败: {e}")
            return {'error': str(e)}
    
    def _generate_agent_proposal(self, agent_id: str, agent_info: Dict) -> Dict[str, Any]:
        """生成智能体提案"""
        proposal_types = {
            'doctors': ['提高医疗质量标准', '增加医生培训项目', '改善医患沟通'],
            'interns': ['扩大实习生培训规模', '提高实习津贴', '改善导师制度'],
            'patients': ['缩短等待时间', '提高服务质量', '降低医疗费用'],
            'accountants': ['优化成本结构', '提高财务透明度', '改善预算管理'],
            'government': ['加强监管措施', '提高合规标准', '促进公平竞争']
        }
        
        proposals = proposal_types.get(agent_id, ['维持现状'])
        selected_proposal = np.random.choice(proposals)
        
        return {
            'agent_id': agent_id,
            'proposal_text': selected_proposal,
            'priority': np.random.uniform(0.5, 1.0),
            'expected_benefit': np.random.uniform(0.3, 0.8),
            'implementation_cost': np.random.uniform(0.1, 0.5),
            'strategy_params': agent_info['strategy_params'].tolist()
        }
    
    def _simulate_parliament_consensus(self, proposals: Dict) -> Dict[str, Any]:
        """模拟议会共识达成过程"""
        # 简化的共识算法
        total_priority = sum(p['priority'] for p in proposals.values())
        consensus_level = min(0.9, total_priority / len(proposals) if proposals else 0.0)
        
        # 选择最高优先级的提案作为主要决策
        if proposals:
            best_proposal = max(proposals.values(), key=lambda x: x['priority'])
            main_decision = best_proposal['proposal_text']
        else:
            main_decision = '维持现状'
        
        return {
            'consensus_level': consensus_level,
            'main_decision': main_decision,
            'participating_agents': list(proposals.keys()),
            'decision_quality': consensus_level * 0.9,
            'implementation_probability': consensus_level
        }
    
    def _update_holy_code(self, consensus_result: Dict) -> Dict[str, Any]:
        """更新神圣法典（模拟）"""
        new_rule = {
            'rule_id': f"rule_{self.current_step}",
            'rule_text': consensus_result['main_decision'],
            'consensus_level': consensus_result['consensus_level'],
            'active': True,
            'creation_step': self.current_step,
            'priority': np.random.uniform(0.5, 1.0)
        }
        
        return {
            'new_rule': new_rule,
            'total_rules': len(self.parliament_history) + 1,
            'activation_success': consensus_result['consensus_level'] > 0.7
        }
    
    def _calculate_meeting_rewards(self, consensus_result: Dict) -> Dict[str, float]:
        """计算议会会议的收益"""
        base_reward = consensus_result['consensus_level'] * 0.5
        
        rewards = {}
        for agent_id in self.agents.keys():
            if self.agents[agent_id]['active']:
                # 基础收益 + 随机变动
                agent_reward = base_reward + np.random.normal(0, 0.1)
                rewards[agent_id] = np.clip(agent_reward, -0.5, 1.0)
        
        return rewards
    
    def _update_agent_networks(self, rewards: Dict[str, float]):
        """更新智能体的actor-critic网络（模拟）"""
        for agent_id, reward in rewards.items():
            if agent_id in self.agents:
                # 模拟网络参数更新
                self.agents[agent_id]['payoff'] += reward
                
                # 模拟策略参数调整
                learning_rate = 0.01
                param_update = np.random.normal(0, learning_rate, 3)
                self.agents[agent_id]['strategy_params'] += param_update
                self.agents[agent_id]['strategy_params'] = np.clip(
                    self.agents[agent_id]['strategy_params'], 0.0, 1.0
                )
                
                # 更新性能指标
                performance_change = reward * 0.1
                self.agents[agent_id]['performance'] += performance_change
                self.agents[agent_id]['performance'] = np.clip(
                    self.agents[agent_id]['performance'], 0.1, 1.0
                )
    
    def _simulate_system_dynamics(self):
        """模拟系统动态变化"""
        # 添加自然变化和随机噪音
        for key in self.system_state:
            if key not in ['training_hours', 'crisis_severity']:
                # 自然衰减/恢复
                if key in ['waiting_times', 'workload']:
                    # 反向指标：趋向增加
                    trend = 0.001
                else:
                    # 正向指标：趋向衰减
                    trend = -0.001
                
                # 添加随机噪音
                noise = np.random.normal(0, 0.01)
                self.system_state[key] += trend + noise
                
                # 限制在合理范围内
                if key in ['waiting_times', 'workload']:
                    self.system_state[key] = np.clip(self.system_state[key], 0.0, 1.0)
                else:
                    self.system_state[key] = np.clip(self.system_state[key], 0.1, 1.0)
    
    def _simulate_agent_decisions(self) -> Dict[str, Any]:
        """模拟智能体决策"""
        actions = {}
        
        for agent_id, agent_info in self.agents.items():
            if agent_info['active'] and np.random.random() < 0.7:  # 70%概率有行动
                action_templates = {
                    'doctors': ['诊断患者', '制定治疗方案', '紧急救治', '医疗会诊', '指导实习生'],
                    'interns': ['学习新技能', '协助诊疗', '参与培训', '临床实践', '请教导师'],
                    'patients': ['就医咨询', '反馈意见', '参与治疗', '康复训练', '投诉建议'],
                    'accountants': ['成本分析', '预算规划', '财务审计', '资源优化', '绩效评估'],
                    'government': ['政策制定', '监管检查', '资源分配', '绩效评估', '合规审查']
                }
                
                possible_actions = action_templates.get(agent_id, ['维持现状'])
                selected_action = np.random.choice(possible_actions)
                
                # 计算决策置信度
                performance = agent_info['performance']
                confidence = performance * (0.7 + np.random.random() * 0.3)
                
                actions[agent_id] = {
                    'action': selected_action,
                    'confidence': confidence,
                    'reasoning': f"{agent_info['name']}基于当前系统状态和个人策略执行{selected_action}",
                    'strategy_params': agent_info['strategy_params'].tolist()
                }
                
                # 更新智能体状态
                agent_info['last_decision'] = selected_action
        
        return actions
    
    def _handle_random_crises(self) -> List[Dict[str, Any]]:
        """处理随机危机事件"""
        crises = []
        
        if np.random.random() < self.config.crisis_probability:
            crisis_types = ['pandemic', 'funding_cut', 'staff_shortage', 'equipment_failure', 'cyber_attack']
            crisis_type = np.random.choice(crisis_types)
            severity = np.random.uniform(0.2, 0.8)
            duration = np.random.randint(5, 20)  # 持续5-20步
            
            crisis = {
                'type': crisis_type,
                'severity': severity,
                'duration': duration,
                'start_step': self.current_step,
                'description': f'{crisis_type}危机 (严重程度: {severity:.2f})',
                'affected_metrics': self._get_crisis_affected_metrics(crisis_type)
            }
            
            self._apply_crisis_effects(crisis)
            self.crisis_history.append(crisis)
            crises.append(crisis)
            
            logger.info(f"🚨 危机事件: {crisis['description']}")
        
        return crises
    
    def _get_crisis_affected_metrics(self, crisis_type: str) -> List[str]:
        """获取危机影响的指标"""
        crisis_effects = {
            'pandemic': ['medical_quality', 'patient_safety', 'resource_adequacy', 'staff_satisfaction'],
            'funding_cut': ['financial_health', 'resource_adequacy', 'education_quality', 'salary_satisfaction'],
            'staff_shortage': ['workload', 'staff_satisfaction', 'patient_satisfaction', 'medical_quality'],
            'equipment_failure': ['resource_utilization', 'medical_quality', 'cost_efficiency'],
            'cyber_attack': ['system_stability', 'patient_safety', 'regulatory_compliance']
        }
        return crisis_effects.get(crisis_type, ['overall_performance'])
    
    def _apply_crisis_effects(self, crisis: Dict[str, Any]):
        """应用危机影响"""
        crisis_type = crisis['type']
        severity = crisis['severity']
        affected_metrics = crisis['affected_metrics']
        
        for metric in affected_metrics:
            if metric in self.system_state:
                # 根据危机类型和严重程度调整系统状态
                if metric in ['waiting_times', 'workload']:
                    # 反向指标：危机会增加这些指标
                    self.system_state[metric] += severity * 0.3
                    self.system_state[metric] = min(self.system_state[metric], 1.0)
                else:
                    # 正向指标：危机会减少这些指标
                    self.system_state[metric] -= severity * 0.2
                    self.system_state[metric] = max(self.system_state[metric], 0.1)
        
        # 更新总体危机严重程度
        self.system_state['crisis_severity'] = max(
            self.system_state['crisis_severity'], 
            severity
        )
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        # 计算各个维度的平均表现
        medical_dimension = np.mean([
            self.system_state['medical_quality'],
            self.system_state['patient_safety'],
            self.system_state['care_quality']
        ])
        
        resource_dimension = np.mean([
            self.system_state['resource_adequacy'],
            self.system_state['resource_utilization'],
            self.system_state['resource_access']
        ])
        
        financial_dimension = np.mean([
            self.system_state['financial_health'],
            self.system_state['cost_efficiency'],
            self.system_state['revenue_growth']
        ])
        
        satisfaction_dimension = np.mean([
            self.system_state['patient_satisfaction'],
            self.system_state['staff_satisfaction']
        ])
        
        # 总体性能指标
        overall_performance = np.mean([
            medical_dimension,
            resource_dimension, 
            financial_dimension,
            satisfaction_dimension
        ])
        
        # 更新系统状态
        self.system_state['overall_performance'] = overall_performance
        
        metrics = {
            'medical_dimension': medical_dimension,
            'resource_dimension': resource_dimension,
            'financial_dimension': financial_dimension,
            'satisfaction_dimension': satisfaction_dimension,
            'overall_performance': overall_performance,
            'system_stability': self.system_state['system_stability'],
            'crisis_count': len(self.crisis_history),
            'active_crisis': self.system_state.get('crisis_severity', 0.0) > 0.1,
            'parliament_meetings': len(self.parliament_history),
            'consensus_efficiency': np.mean([p.get('consensus', {}).get('consensus_level', 0.5) 
                                          for p in self.parliament_history[-5:]]) if self.parliament_history else 0.5
        }
        
        return metrics
    
    def _update_agent_payoffs(self, step_data: Dict[str, Any]):
        """更新智能体收益"""
        performance_metrics = step_data.get('metrics', {})
        overall_performance = performance_metrics.get('overall_performance', 0.5)
        
        for agent_id, agent_info in self.agents.items():
            if agent_info['active']:
                # 基础收益基于系统整体表现
                base_payoff = overall_performance * 0.1
                
                # 角色特定的收益调整
                role_multiplier = {
                    'doctors': performance_metrics.get('medical_dimension', 0.5),
                    'interns': performance_metrics.get('medical_dimension', 0.5) * 0.8,
                    'patients': performance_metrics.get('satisfaction_dimension', 0.5),
                    'accountants': performance_metrics.get('financial_dimension', 0.5),
                    'government': performance_metrics.get('system_stability', 0.5)
                }
                
                role_bonus = role_multiplier.get(agent_id, 0.5) * 0.05
                
                # 随机变动
                noise = np.random.normal(0, 0.02)
                
                total_payoff = base_payoff + role_bonus + noise
                agent_info['payoff'] += total_payoff
    
    async def run_async(self, steps: int = None, scenario_runner=None, training: bool = False):
        """异步运行仿真"""
        if steps is None:
            steps = self.config.max_steps
        
        self.is_running = True
        logger.info(f"🚀 开始异步仿真运行: {steps}步")
        
        try:
            for step in range(steps):
                # 检查暂停状态
                while self.is_paused and self.is_running:
                    await asyncio.sleep(0.1)
                
                if not self.is_running:
                    break
                
                # 场景事件插入
                if scenario_runner is not None:
                    try:
                        scenario_runner.check_and_insert_event(self.current_step)
                    except Exception as e:
                        logger.warning(f"⚠️ 场景事件插入失败: {e}")
                
                # 执行仿真步
                step_data = self.step(training=training)
                
                # 异步等待
                await asyncio.sleep(2.0)  # 2秒间隔
                
                # 进度显示
                if step % 50 == 0:
                    logger.info(f"📊 异步进度: {step}/{steps}步, 性能: {step_data.get('metrics', {}).get('overall_performance', 0):.2f}")
        
        except Exception as e:
            logger.error(f"❌ 异步仿真运行错误: {e}")
        finally:
            self.is_running = False
            logger.info("✅ 异步仿真运行完成")
    
    def run(self, steps: int = 1000, scenario_runner=None, training: bool = False):
        """同步运行仿真"""
        self.is_running = True
        results = []
        
        logger.info(f"🚀 开始同步仿真运行: {steps}步")
        
        try:
            for step in range(steps):
                if not self.is_running:
                    break
                
                # 场景事件插入
                if scenario_runner is not None:
                    try:
                        scenario_runner.check_and_insert_event(self.current_step)
                    except Exception as e:
                        logger.warning(f"⚠️ 场景事件插入失败: {e}")
                
                # 仿真步
                step_data = self.step(training=training)
                results.append(step_data)
                
                if step % 100 == 0:
                    logger.info(f"📊 进度: {step}/{steps}步, 性能: {step_data.get('metrics', {}).get('overall_performance', 0):.2f}")
        
        except Exception as e:
            logger.error(f"❌ 同步仿真运行错误: {e}")
        finally:
            self.is_running = False
            logger.info("✅ 同步仿真运行完成")
        
        return results
    
    def pause(self):
        """暂停仿真"""
        self.is_paused = True
        logger.info("⏸️ 仿真已暂停")
    
    def resume(self):
        """恢复仿真"""
        self.is_paused = False
        logger.info("▶️ 仿真已恢复")
    
    def stop(self):
        """停止仿真"""
        self.is_running = False
        self.is_paused = False
        logger.info("⏹️ 仿真已停止")
    
    def reset(self):
        """重置仿真器"""
        self.current_step = 0
        self.simulation_time = 0.0
        self.is_running = False
        self.is_paused = False
        
        # 重置系统状态
        self.system_state.update({
            'medical_quality': 0.85,
            'patient_safety': 0.9,
            'financial_health': 0.8,
            'system_stability': 0.8,
            'crisis_severity': 0.0,
            'overall_performance': 0.78
        })
        
        # 重置智能体状态
        for agent_id, agent_info in self.agents.items():
            agent_info.update({
                'performance': 0.7 + np.random.random() * 0.2,
                'payoff': 0.0,
                'strategy_params': np.random.uniform(0.3, 0.7, 3),
                'last_decision': None,
                'active': True
            })
        
        # 清理历史记录
        self.performance_history.clear()
        self.crisis_history.clear()
        self.decision_history.clear()
        self.parliament_history.clear()
        
        logger.info("🔄 仿真器已重置")
    
    def get_simulation_report(self) -> Dict[str, Any]:
        """获取仿真报告"""
        try:
            metrics = self._calculate_performance_metrics()
            
            return {
                'simulation_info': {
                    'current_step': self.current_step,
                    'simulation_time': self.simulation_time,
                    'total_decisions': len(self.decision_history),
                    'total_crises': len(self.crisis_history),
                    'parliament_meetings': len(self.parliament_history),
                    'is_running': self.is_running,
                    'is_paused': self.is_paused
                },
                'system_state': self.system_state.copy(),
                'performance_metrics': metrics,
                'agent_status': {
                    agent_id: {
                        'name': info['name'],
                        'performance': info['performance'],
                        'payoff': info['payoff'],
                        'active': info['active'],
                        'last_decision': info['last_decision']
                    }
                    for agent_id, info in self.agents.items()
                },
                'recent_activity': {
                    'recent_crises': self.crisis_history[-3:] if self.crisis_history else [],
                    'recent_parliament': self.parliament_history[-1:] if self.parliament_history else [],
                    'performance_trend': self.performance_history[-10:] if self.performance_history else []
                }
            }
        except Exception as e:
            logger.error(f"❌ 生成仿真报告失败: {e}")
            return {
                'error': str(e),
                'current_step': self.current_step,
                'system_state': self.system_state
            }

# 导出的类和函数
__all__ = ['KallipolisSimulator', 'SimulationConfig']