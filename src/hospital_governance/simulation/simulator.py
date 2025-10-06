import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
import time
import json
from dataclasses import dataclass
import yaml
import os
import traceback

# 导入core数学模块
try:
    from ..core.kallipolis_mathematical_core import SystemState, KallipolisMedicalSystem
    from ..core.system_dynamics import SystemDynamics
    from ..core.system_matrices import SystemMatrixGenerator
    from ..core.state_space import StateSpace
    HAS_CORE_MATH = True
except ImportError as e:
    logger.warning(f"Core数学模块导入失败: {e}")
    HAS_CORE_MATH = False

# 导入agents包的详细功能
from ..agents.role_agents import RoleAgent, RoleManager, AgentConfig
from ..agents.multi_agent_coordinator import MultiAgentInteractionEngine, InteractionConfig
from ..agents.llm_action_generator import LLMActionGenerator, LLMConfig
from ..agents.role_agents_old import ParliamentMemberAgent
from ..agents.learning_models import MADDPGModel, LearningModel
from ..control.distributed_control import DistributedControlSystem

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """仿真配置"""
    max_steps: int = 14
    time_scale: float = 1.0
    meeting_interval: int = 7  # 每7步举行议会会议
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
        
        # 核心组件初始化
        self.role_manager = None
        self.holy_code_manager = None
        self.learning_model = None
        self.maddpg_model = None  # MADDPG多智能体学习模型
        self.experience_buffer = []  # 经验回放缓冲区
        self.distributed_controller = None  # 分布式控制系统
        self.interaction_engine = None
        self.coordinator = None
        self.llm_action_generator = None
        
        # 初始化核心数学系统
        if HAS_CORE_MATH:
            self.core_system = KallipolisMedicalSystem()
            self.system_state = self.core_system.current_state
            
            # 初始化系统动力学
            system_matrices = SystemMatrixGenerator.generate_nominal_matrices()
            self.system_matrices = system_matrices  # 保存为实例变量
            self.system_dynamics = SystemDynamics(system_matrices)
            
            # 初始化状态空间管理
            self.state_space = StateSpace(self.system_state.to_vector())
        else:
            # 回退到简化状态
            self._legacy_system_state = {
                'medical_quality': 0.85, 'patient_safety': 0.9, 'care_quality': 0.8,
                'resource_adequacy': 0.7, 'resource_utilization': 0.75, 'resource_access': 0.8,
                'education_quality': 0.7, 'training_hours': 30.0, 'mentorship_availability': 0.6,
                'career_development': 0.65, 'financial_health': 0.8, 'cost_efficiency': 0.75,
                'revenue_growth': 0.7, 'patient_satisfaction': 0.75, 'accessibility': 0.8,
                'waiting_times': 0.3, 'staff_satisfaction': 0.7, 'workload': 0.6,
                'salary_satisfaction': 0.65, 'system_stability': 0.8, 'ethics_compliance': 0.85,
                'regulatory_compliance': 0.9, 'public_trust': 0.75, 'overall_performance': 0.78,
                'crisis_severity': 0.0
            }
            self.system_state = None
        
        # 初始化真实的智能体对象
        self.agent_objects = {}  # 存储RoleAgent对象
        self.agents = {  # 保持兼容性的简化状态
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
        
        # 在agents定义之后初始化LLM和智能体系统
        self._initialize_agents_and_llm()
        
        logger.info("🏥 KallipolisSimulator初始化完成")
    
    def _convert_state_to_dict(self) -> Dict[str, float]:
        """将SystemState对象转换为字典格式，保持向后兼容性"""
        if HAS_CORE_MATH and hasattr(self.system_state, 'to_vector'):
            # 使用core模块的SystemState
            state_vector = self.system_state.to_vector()
            state_names = self.system_state.get_component_names()
            
            # 创建映射字典
            state_dict = dict(zip([
                'medical_resource_utilization', 'patient_waiting_time', 'financial_indicator', 'ethical_compliance',
                'education_training_quality', 'intern_satisfaction', 'professional_development', 'mentorship_effectiveness',
                'patient_satisfaction', 'service_accessibility', 'care_quality_index', 'safety_incident_rate',
                'operational_efficiency', 'staff_workload_balance', 'crisis_response_capability', 'regulatory_compliance_score'
            ], state_vector))
            
            # 添加兼容性映射
            legacy_mapping = {
                'medical_quality': state_dict['care_quality_index'],
                'patient_safety': 1.0 - state_dict['safety_incident_rate'],
                'care_quality': state_dict['care_quality_index'],
                'resource_adequacy': state_dict['medical_resource_utilization'],
                'resource_utilization': state_dict['medical_resource_utilization'], 
                'resource_access': state_dict['service_accessibility'],
                'education_quality': state_dict['education_training_quality'],
                'training_hours': 30.0 + state_dict['education_training_quality'] * 10,
                'mentorship_availability': state_dict['mentorship_effectiveness'],
                'career_development': state_dict['professional_development'],
                'financial_health': state_dict['financial_indicator'],
                'cost_efficiency': state_dict['operational_efficiency'],
                'revenue_growth': state_dict['financial_indicator'],
                'accessibility': state_dict['service_accessibility'],
                'waiting_times': state_dict['patient_waiting_time'],
                'staff_satisfaction': state_dict['intern_satisfaction'],
                'workload': state_dict['staff_workload_balance'],
                'salary_satisfaction': state_dict['intern_satisfaction'],
                'system_stability': state_dict['crisis_response_capability'],
                'ethics_compliance': state_dict['ethical_compliance'],
                'regulatory_compliance': state_dict['regulatory_compliance_score'],
                'public_trust': state_dict['regulatory_compliance_score'],
                'overall_performance': np.mean(list(state_dict.values())),
                'crisis_severity': 1.0 - state_dict['crisis_response_capability']
            }
            
            state_dict.update(legacy_mapping)
            return state_dict
        else:
            # 回退到原有字典格式
            return getattr(self, '_legacy_system_state', {
                'medical_quality': 0.85, 'patient_safety': 0.9, 'care_quality': 0.8,
                'resource_adequacy': 0.7, 'resource_utilization': 0.75, 'resource_access': 0.8,
                'education_quality': 0.7, 'training_hours': 30.0, 'mentorship_availability': 0.6,
                'career_development': 0.65, 'financial_health': 0.8, 'cost_efficiency': 0.75,
                'revenue_growth': 0.7, 'patient_satisfaction': 0.75, 'accessibility': 0.8,
                'waiting_times': 0.3, 'staff_satisfaction': 0.7, 'workload': 0.6,
                'salary_satisfaction': 0.65, 'system_stability': 0.8, 'ethics_compliance': 0.85,
                'regulatory_compliance': 0.9, 'public_trust': 0.75, 'overall_performance': 0.78,
                'crisis_severity': 0.0
            })
    
    def _initialize_agents_and_llm(self):
        """初始化智能体和LLM系统"""
        try:
            # 初始化LLM行动生成器
            from ..agents.llm_action_generator import LLMActionGenerator, LLMConfig
            llm_config = LLMConfig(
                model_name="gpt-4",
                temperature=0.7,
                use_async=True
            )
            self.llm_action_generator = LLMActionGenerator(llm_config)
            
            # 初始化角色智能体管理器
            from ..agents.role_agents import RoleAgent, RoleManager, AgentConfig
            from ..agents.multi_agent_coordinator import MultiAgentInteractionEngine, InteractionConfig
            from ..agents.role_agents_old import ParliamentMemberAgent
            
            # 初始化角色管理器（使用默认配置）
            self.role_manager = RoleManager()
            
            # 配置交互引擎
            interaction_config = InteractionConfig(
                use_behavior_models=True,
                use_learning_models=False,
                use_llm_generation=self.config.enable_llm_integration
            )
            self.interaction_engine = MultiAgentInteractionEngine(
                self.role_manager, interaction_config
            )
            
            # 为每个角色创建RoleAgent对象
            from ..agents.role_agents import DoctorAgent, InternAgent, PatientAgent, AccountantAgent, GovernmentAgent
            
            agent_classes = {
                'doctors': DoctorAgent,
                'interns': InternAgent,
                'patients': PatientAgent,
                'accountants': AccountantAgent,
                'government': GovernmentAgent
            }
            
            for agent_id, agent_data in self.agents.items():
                try:
                    # 创建智能体配置
                    agent_config = AgentConfig(
                        role=agent_id,
                        action_dim=4,  # 基于系统动力学的行动维度
                        observation_dim=16,  # 基于16维状态空间
                        learning_rate=0.001,
                        alpha=0.3,
                        beta=0.5,
                        gamma=0.2
                    )
                    
                    # 获取对应的智能体类
                    agent_class = agent_classes.get(agent_id, RoleAgent)
                    if agent_class == RoleAgent:
                        logger.warning(f"⚠️ 未找到 {agent_id} 的具体实现类，跳过创建")
                        continue
                    
                    # 创建具体的RoleAgent对象
                    role_agent = agent_class(agent_config)
                    role_agent.llm_generator = self.llm_action_generator
                    
                    # 存储到agent_objects中
                    self.agent_objects[agent_id] = role_agent
                    
                    logger.info(f"✅ 创建智能体对象: {agent_id} ({agent_data['name']}) -> {agent_class.__name__}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ 创建智能体 {agent_id} 失败: {e}")
                    
            logger.info(f"✅ 共创建 {len(self.agent_objects)} 个智能体对象")
            
            # 初始化MADDPG学习模型
            if self.config.enable_learning:
                self._initialize_maddpg_model()
            
            # 初始化分布式控制系统
            self._initialize_distributed_control()
            
            # 初始化神圣法典管理器
            from ..holy_code.holy_code_manager import HolyCodeManager, HolyCodeConfig
            holy_config = HolyCodeConfig(rule_config_path='config/holy_code_rules.yaml')
            self.holy_code_manager = HolyCodeManager(holy_config)
            
            logger.info("✅ 智能体和LLM系统初始化完成")
            
        except Exception as e:
            import traceback
            logger.warning(f"⚠️ 智能体系统初始化失败，使用模拟模式: {e}")
            logger.warning(f"详细错误信息: {traceback.format_exc()}")
            self.llm_action_generator = None
            self.role_manager = None
            self.interaction_engine = None
            self.holy_code_manager = None
    
    def _initialize_maddpg_model(self):
        """初始化MADDPG多智能体学习模型"""
        try:
            # 定义每个智能体的行动维度
            action_dims = {
                'doctors': 4,     # 医疗质量、诊断精度、治疗效率、患者沟通
                'interns': 4,     # 学习强度、技能训练、导师互动、临床实践
                'patients': 3,    # 反馈强度、合作程度、满意度表达
                'accountants': 4, # 成本控制、预算规划、财务监管、效率优化
                'government': 4   # 政策制定、监管强度、资源分配、合规要求
            }
            
            # 系统状态维度（16维状态空间）
            state_dim = 16
            
            # 初始化MADDPG模型
            self.maddpg_model = MADDPGModel(
                state_dim=state_dim,
                action_dims=action_dims,
                hidden_dim=128,
                actor_lr=0.001,
                critic_lr=0.002,
                tau=0.01,
                gamma=0.99
            )
            
            # 初始化经验缓冲区
            self.experience_buffer = []
            self.max_buffer_size = 10000
            self.batch_size = 64
            
            logger.info("✅ MADDPG多智能体学习模型初始化完成")
            
        except Exception as e:
            logger.warning(f"⚠️ MADDPG模型初始化失败: {e}")
            self.maddpg_model = None
    
    def _initialize_distributed_control(self):
        """初始化分布式控制系统"""
        try:
            # 定义控制器配置
            controller_configs = {
                'doctors': {
                    'feedback_gain': np.array([[0.5, 0.3, 0.2, 0.1],
                                              [0.3, 0.6, 0.1, 0.2],
                                              [0.2, 0.1, 0.5, 0.3],
                                              [0.1, 0.2, 0.3, 0.4]]),
                    'integrator_gain': 0.1,
                    'control_limits': [-1.0, 1.0]
                },
                'interns': {
                    'observer_gain': np.array([[0.4, 0.3, 0.2, 0.1],
                                              [0.3, 0.4, 0.1, 0.2],
                                              [0.2, 0.1, 0.4, 0.3],
                                              [0.1, 0.2, 0.3, 0.4]]),
                    'feedforward_gain': 0.2,
                    'control_limits': [-1.0, 1.0]
                },
                'patients': {
                    'Kp': 0.5,
                    'Ka': 0.2,
                    'control_limits': [-1.0, 1.0]
                },
                'accountants': {
                    'constraint_matrix': np.array([[0.6, 0.3, 0.1],
                                                  [0.3, 0.5, 0.2],
                                                  [0.1, 0.2, 0.7]]),
                    'budget_limit': 1.0,
                    'constraint_weights': np.array([0.6, 0.3, 0.1]),
                    'safety_margin': 0.1,
                    'control_limits': [-1.0, 1.0]
                },
                'government': {
                    'policy_matrix': np.array([[0.4, 0.3, 0.3],
                                              [0.3, 0.4, 0.3],
                                              [0.3, 0.3, 0.4]]),
                    'policy_limits': [-1.0, 1.0]
                }
            }
            
            # 创建分布式控制系统
            self.distributed_controller = DistributedControlSystem(controller_configs)
            
            logger.info("✅ 分布式控制系统初始化完成")
            
        except Exception as e:
            logger.warning(f"⚠️ 分布式控制系统初始化失败: {e}")
            self.distributed_controller = None
    
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
            'system_state': self._convert_state_to_dict(),  # 转换为字典格式用于兼容性
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
            
            # 7. 收集经验数据用于MADDPG训练
            if self.config.enable_learning:
                self._collect_experience(step_data)

            # 8. 使用holy_code模块处理规则激活
            if self.holy_code_manager:
                try:
                    decision_context = {
                        'decision_type': 'routine_operation',
                        'current_state': self._convert_state_to_dict(),
                        'step': self.current_step,
                        'agent_id': 'system'
                    }
                    
                    guidance = self.holy_code_manager.process_agent_decision_request(
                        'system', decision_context
                    )
                    
                    # 将指导信息发送到WebSocket
                    if self.data_callback and guidance:
                        rule_msg = {
                            'type': 'holy_code_update',
                            'guidance': guidance,
                            'timestamp': time.time()
                        }
                        if asyncio.iscoroutinefunction(self.data_callback):
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(self.data_callback(rule_msg))
                            else:
                                asyncio.run(self.data_callback(rule_msg))
                        else:
                            self.data_callback(rule_msg)
                            
                except Exception as e:
                    logger.warning(f"⚠️ Holy code决策处理失败: {e}")

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
        """运行议会会议 - 集成智能体提案生成"""
        try:
            if self.holy_code_manager:
                # 使用RoleAgent对象生成提案（如果可用）
                proposals = {}
                for agent_id, agent_info in self.agents.items():
                    if agent_info['active']:
                        proposal = self._generate_agent_proposal_advanced(agent_id, agent_info)
                        proposals[agent_id] = proposal
                
                # 使用holy_code模块的议会会议流程
                parliament_result = self.holy_code_manager.run_weekly_parliament_meeting(
                    self.agents, 
                    self.system_state
                )
                
                # 记录议会历史
                parliament_record = {
                    'step': self.current_step,
                    'parliament_result': parliament_result,
                    'proposals': proposals,
                    'timestamp': time.time()
                }
                self.parliament_history.append(parliament_record)
                
                # 获取共识水平
                consensus_level = getattr(parliament_result, 'consensus_level', 0.7)
                logger.info(f"🏛️ 议会会议完成，达成共识: {consensus_level:.2f}")
                logger.info(f"📋 本次收到 {len(proposals)} 项提案")
                return parliament_record
            else:
                # 回退到简化版本
                return self._run_parliament_meeting_fallback()
                
        except Exception as e:
            logger.error(f"❌ 议会会议执行失败: {e}")
            return {'error': str(e)}
    
    def _generate_agent_proposal_advanced(self, agent_id: str, agent_info: Dict) -> Dict[str, Any]:
        """生成智能体提案 - 集成RoleAgent对象方法"""
        try:
            # 优先使用RoleAgent对象生成提案
            if agent_id in self.agent_objects:
                role_agent = self.agent_objects[agent_id]
                
                # 构建提案上下文
                proposal_context = {
                    'system_state': self._convert_state_to_dict(),
                    'current_step': self.current_step,
                    'agent_performance': agent_info['performance'],
                    'parliament_history': self.parliament_history[-3:] if self.parliament_history else []  # 最近3次会议
                }
                
                # 如果RoleAgent有formulate_proposal方法，使用它
                if hasattr(role_agent, 'formulate_proposal'):
                    proposal = role_agent.formulate_proposal(proposal_context)
                    logger.info(f"✅ {agent_id} 使用RoleAgent生成提案")
                    return proposal
                    
                # 否则使用LLM生成提案
                elif self.llm_action_generator:
                    llm_proposal = self.llm_action_generator.generate_proposal_sync(
                        agent_id, proposal_context
                    )
                    return {
                        'agent_id': agent_id,
                        'proposal_text': llm_proposal.get('proposal', '维持现状'),
                        'priority': llm_proposal.get('priority', 0.5),
                        'expected_benefit': llm_proposal.get('benefit', 0.5),
                        'implementation_cost': llm_proposal.get('cost', 0.3),
                        'reasoning': llm_proposal.get('reasoning', 'LLM生成提案'),
                        'strategy_params': agent_info['strategy_params'].tolist()
                    }
            
            # 回退到模板提案
            return self._generate_agent_proposal(agent_id, agent_info)
            
        except Exception as e:
            logger.warning(f"⚠️ {agent_id} 高级提案生成失败，使用模板: {e}")
            return self._generate_agent_proposal(agent_id, agent_info)
    
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
    
    def _run_parliament_meeting_fallback(self) -> Dict[str, Any]:
        """回退的简化议会会议流程"""
        # 收集智能体提案
        proposals = {}
        for agent_id, agent_info in self.agents.items():
            if agent_info['active']:
                proposal = self._generate_agent_proposal(agent_id, agent_info)
                proposals[agent_id] = proposal
        
        # 简化的共识算法
        if proposals:
            total_priority = sum(p['priority'] for p in proposals.values())
            consensus_level = min(0.9, total_priority / len(proposals))
            best_proposal = max(proposals.values(), key=lambda x: x['priority'])
            main_decision = best_proposal['proposal_text']
        else:
            consensus_level = 0.5
            main_decision = '维持现状'
        
        return {
            'consensus_level': consensus_level,
            'main_decision': main_decision,
            'participating_agents': list(proposals.keys()),
            'proposals': proposals
        }
    
    def _update_holy_code_via_consensus(self, consensus_result: Dict) -> Dict[str, Any]:
        """通过holy_code模块更新神圣法典"""
        if self.holy_code_manager:
            # 使用holy_code模块的写入共识功能
            try:
                self.holy_code_manager.write_consensus(consensus_result)
                return {
                    'success': True,
                    'consensus_level': consensus_result.get('consensus_level', 0.5),
                    'operation': 'HOLY_CODE_UPDATE'
                }
            except Exception as e:
                logger.warning(f"⚠️ Holy code更新失败: {e}")
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': 'HolyCodeManager未初始化'}
    
    def _calculate_meeting_rewards(self, parliament_result: Dict) -> Dict[str, float]:
        """计算议会会议的收益"""
        if self.holy_code_manager:
            try:
                # 使用holy_code模块的收益计算
                rewards = self.holy_code_manager.calculate_rewards(
                    parliament_result, 
                    self.agents, 
                    self.system_state
                )
                return rewards
            except Exception as e:
                logger.warning(f"⚠️ Holy code收益计算失败，使用简化版本: {e}")
        
        # 回退到简化收益计算
        base_reward = parliament_result.get('consensus_level', 0.5) * 0.5
        rewards = {}
        for agent_id in self.agents.keys():
            if self.agents[agent_id]['active']:
                agent_reward = base_reward + np.random.normal(0, 0.1)
                rewards[agent_id] = np.clip(agent_reward, -0.5, 1.0)
        return rewards
    
    def _update_agent_networks(self, rewards: Dict[str, float]):
        """更新智能体的actor-critic网络 - 集成MADDPG训练"""
        # 使用MADDPG模型进行真正的网络训练
        if self.maddpg_model and self.config.enable_learning and len(self.experience_buffer) >= self.batch_size:
            try:
                # 从经验缓冲区采样批次数据
                batch = self._sample_experience_batch()
                
                # 训练MADDPG模型
                losses = self.maddpg_model.train(batch)
                
                # 记录训练损失
                total_loss = sum(losses.values()) if losses else 0
                logger.info(f"🧠 MADDPG训练完成，平均损失: {total_loss/len(losses) if losses else 0:.4f}")
                
            except Exception as e:
                logger.warning(f"⚠️ MADDPG训练失败: {e}")
        
        # 更新传统的智能体状态
        for agent_id, reward in rewards.items():
            if agent_id in self.agents:
                # 更新收益
                self.agents[agent_id]['payoff'] += reward
                
                # 如果没有MADDPG，使用简化的策略参数调整
                if not self.maddpg_model:
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
    
    def _collect_experience(self, step_data: Dict[str, Any]):
        """收集经验数据用于MADDPG训练"""
        if not self.maddpg_model or not self.config.enable_learning:
            return
        
        try:
            # 获取当前状态
            if self.system_state:
                current_state = self.system_state.to_vector()
            else:
                current_state = np.array(list(self._legacy_system_state.values())[:16])
            
            # 为每个有行动的智能体收集经验
            for agent_id, action_data in step_data.get('actions', {}).items():
                if agent_id in self.agents and 'last_action_vector' in self.agents[agent_id]:
                    
                    # 构建经验元组
                    experience = {
                        'role': agent_id,
                        'state': current_state.copy(),
                        'action': self.agents[agent_id]['last_action_vector'].copy(),
                        'reward': step_data.get('rewards', {}).get(agent_id, 0.0),
                        'next_state': current_state.copy(),  # 下一步会在下次更新
                        'done': False
                    }
                    
                    # 添加到经验缓冲区
                    self.experience_buffer.append(experience)
                    
                    # 限制缓冲区大小
                    if len(self.experience_buffer) > self.max_buffer_size:
                        self.experience_buffer.pop(0)
                    
                    # 清除临时存储的行动向量
                    del self.agents[agent_id]['last_action_vector']
            
        except Exception as e:
            logger.warning(f"⚠️ 经验收集失败: {e}")
    
    def _sample_experience_batch(self) -> List[Dict]:
        """从经验缓冲区采样训练批次"""
        import random
        
        if len(self.experience_buffer) < self.batch_size:
            return self.experience_buffer.copy()
        
        return random.sample(self.experience_buffer, self.batch_size)
    
    def get_maddpg_stats(self) -> Dict[str, Any]:
        """获取MADDPG训练统计信息"""
        if not self.maddpg_model:
            return {'status': 'disabled'}
        
        return {
            'status': 'enabled',
            'experience_buffer_size': len(self.experience_buffer),
            'max_buffer_size': self.max_buffer_size,
            'batch_size': self.batch_size,
            'ready_for_training': len(self.experience_buffer) >= self.batch_size,
            'agent_action_dims': {
                'doctors': 4, 'interns': 4, 'patients': 3, 
                'accountants': 4, 'government': 4
            }
        }
    
    def save_maddpg_model(self, filepath: str):
        """保存MADDPG模型"""
        if self.maddpg_model:
            try:
                self.maddpg_model.save_models(filepath)
                logger.info(f"✅ MADDPG模型已保存到: {filepath}")
            except Exception as e:
                logger.error(f"❌ MADDPG模型保存失败: {e}")
        else:
            logger.warning("⚠️ MADDPG模型未初始化，无法保存")
    
    def load_maddpg_model(self, filepath: str):
        """加载MADDPG模型"""
        if self.maddpg_model:
            try:
                self.maddpg_model.load_models(filepath)
                logger.info(f"✅ MADDPG模型已从{filepath}加载")
            except Exception as e:
                logger.error(f"❌ MADDPG模型加载失败: {e}")
        else:
            logger.warning("⚠️ MADDPG模型未初始化，无法加载")
    
    def _simulate_system_dynamics(self):
        """模拟系统动态变化 - 集成分布式控制系统"""
        try:
            # 获取当前状态向量
            current_state_vector = self.system_state.to_vector()
            
            # 使用分布式控制器生成控制输入
            if self.distributed_controller and hasattr(self, 'holy_code_manager'):
                try:
                    # 设定参考状态（理想状态）
                    x_ref = np.array([0.8, 0.3, 0.8, 0.85, 0.75, 0.8, 0.7, 0.8, 
                                     0.85, 0.8, 0.9, 0.1, 0.8, 0.7, 0.8, 0.9])
                    
                    # 扰动预测（简化版本）
                    d_hat = np.random.normal(0, 0.1, 6)
                    
                    # 获取神圣法典状态
                    holy_code_state = {}
                    if self.holy_code_manager:
                        system_status = self.holy_code_manager.get_system_status()
                        holy_code_state = {
                            'ethical_constraints': {
                                'min_quality_control': 0.7,
                                'max_workload': 0.8,
                                'min_health_level': 0.6,
                                'min_equity_level': 0.5
                            },
                            'rule_library': system_status.get('rule_library', {}),
                            'consensus_level': system_status.get('consensus_level', 0.7)
                        }
                    
                    # 使用分布式控制器计算控制输入
                    u_t = self.distributed_controller.compute_control(
                        current_state_vector, x_ref, d_hat, holy_code_state
                    )
                    
                    logger.debug(f"🎛️ 使用分布式控制器生成控制输入")
                    
                except Exception as e:
                    logger.warning(f"⚠️ 分布式控制器失败，使用随机控制: {e}")
                    # 回退到随机控制输入
                    u_t = np.random.normal(0, 0.1, 17)
            else:
                # 回退到随机控制输入（17维）
                u_t = np.random.normal(0, 0.1, 17)
            
            # 生成随机扰动（6维）
            d_t = np.random.normal(0, 0.05, 6)
            
            # 使用核心系统动力学
            next_state_vector = self.system_dynamics.state_transition(current_state_vector, u_t, d_t)
            
            # 更新系统状态
            self.system_state = SystemState.from_vector(next_state_vector)
            self.state_space.update_state(next_state_vector)
            
        except Exception as e:
            logger.warning(f"⚠️ 系统动力学更新失败，使用简化模式: {e}")
            # 回退到简化的随机游走
            state_vector = self.system_state.to_vector()
            noise = np.random.normal(0, 0.01, len(state_vector))
            new_state_vector = np.clip(state_vector + noise, 0, 1)
            self.system_state = SystemState.from_vector(new_state_vector)
    
    def _simulate_agent_decisions(self) -> Dict[str, Any]:
        """模拟智能体决策 - 集成RoleAgent和LLM生成"""
        actions = {}
        
        for agent_id, agent_info in self.agents.items():
            if agent_info['active'] and np.random.random() < 0.7:  # 70%概率有行动
                
                # 优先使用RoleAgent对象
                if agent_id in self.agent_objects:
                    try:
                        role_agent = self.agent_objects[agent_id]
                        
                        # 构建观测上下文
                        observation_context = {
                            'role': agent_id,
                            'system_state': self._convert_state_to_dict(),
                            'agent_performance': agent_info['performance'],
                            'current_step': self.current_step,
                            'system_matrices': self.system_matrices if hasattr(self, 'system_matrices') else None
                        }
                        
                        # 使用RoleAgent的智能决策
                        if self.llm_action_generator and self.config.enable_llm_integration:
                            # 使用LLM生成行动
                            # 准备观测向量
                            if self.system_state:
                                observation = self.system_state.to_vector()
                            else:
                                observation = np.array(list(self._legacy_system_state.values())[:16])
                            
                            # 准备holy_code状态
                            holy_code_state = {}
                            if hasattr(self, 'holy_code_manager') and self.holy_code_manager:
                                system_status = self.holy_code_manager.get_system_status()
                                holy_code_state = {
                                    'rules': system_status.get('rule_library', {}),
                                    'consensus': system_status.get('consensus_level', 0.7),
                                    'active_rules': len(system_status.get('rule_library', {}))
                                }
                            
                            llm_action_vector = self.llm_action_generator.generate_action_sync(
                                agent_id, observation, holy_code_state, observation_context
                            )
                            
                            # 将数值向量转换为描述性行动
                            selected_action = self._convert_action_vector_to_description(agent_id, llm_action_vector)
                            reasoning = f'{agent_id}基于LLM数值策略决策'
                            confidence = np.mean(np.abs(llm_action_vector))
                        else:
                            # 尝试使用MADDPG网络生成行动
                            if self.maddpg_model and self.config.enable_learning:
                                try:
                                    # 获取系统状态向量
                                    if self.system_state:
                                        state_vector = self.system_state.to_vector()
                                    else:
                                        state_vector = np.array(list(self._legacy_system_state.values())[:16])
                                    
                                    # 使用Actor网络生成行动
                                    observations = {agent_id: state_vector}
                                    actions_dict = self.maddpg_model.get_actions(observations, training=True)
                                    action_vector = actions_dict[agent_id]
                                    
                                    # 将行动向量转换为描述
                                    selected_action = self._convert_action_vector_to_description(agent_id, action_vector)
                                    reasoning = f'{agent_id}基于Actor网络决策'
                                    confidence = np.tanh(np.linalg.norm(action_vector))  # 行动强度转置信度
                                    
                                    # 存储用于训练的行动向量
                                    agent_info['last_action_vector'] = action_vector
                                    
                                except Exception as e:
                                    logger.warning(f"⚠️ {agent_id} Actor网络决策失败，使用数学策略: {e}")
                                    # 回退到RoleAgent数学策略
                                    sampled_action = role_agent.sample_action(self.system_state)
                                    selected_action = f"数学策略行动_{np.random.randint(1,5)}"
                                    reasoning = f'{agent_id}基于数学策略选择: {selected_action}'
                                    confidence = 0.7
                            else:
                                # 使用分布式控制器作为回退
                                if self.distributed_controller:
                                    try:
                                        # 获取当前状态和参考状态
                                        current_state = self.system_state.to_vector() if self.system_state else np.array(list(self._legacy_system_state.values())[:16])
                                        x_ref = np.array([0.8, 0.3, 0.8, 0.85, 0.75, 0.8, 0.7, 0.8, 
                                                         0.85, 0.8, 0.9, 0.1, 0.8, 0.7, 0.8, 0.9])
                                        d_hat = np.random.normal(0, 0.1, 6)
                                        
                                        # 获取神圣法典状态
                                        holy_code_state = {}
                                        if self.holy_code_manager:
                                            system_status = self.holy_code_manager.get_system_status()
                                            holy_code_state = {
                                                'ethical_constraints': {
                                                    'min_quality_control': 0.7,
                                                    'max_workload': 0.8,
                                                    'min_health_level': 0.6,
                                                    'min_equity_level': 0.5
                                                }
                                            }
                                        
                                        # 使用分布式控制器
                                        control_signal = self.distributed_controller.compute_control(
                                            current_state, x_ref, d_hat, holy_code_state
                                        )
                                        
                                        # 提取该智能体的控制信号部分
                                        agent_control_ranges = {
                                            'doctors': control_signal[0:4],
                                            'interns': control_signal[4:8], 
                                            'patients': control_signal[8:11],
                                            'accountants': control_signal[11:14],
                                            'government': control_signal[14:17]
                                        }
                                        
                                        if agent_id in agent_control_ranges:
                                            action_vector = agent_control_ranges[agent_id]
                                            selected_action = self._convert_action_vector_to_description(agent_id, action_vector)
                                            reasoning = f'{agent_id}基于分布式控制器决策'
                                            confidence = np.tanh(np.linalg.norm(action_vector))
                                            
                                            # 存储用于训练的行动向量
                                            agent_info['last_action_vector'] = action_vector
                                        else:
                                            selected_action = f"控制策略行动_{np.random.randint(1,5)}"
                                            reasoning = f'{agent_id}基于控制策略选择: {selected_action}'
                                            confidence = 0.7
                                            
                                    except Exception as e:
                                        logger.warning(f"⚠️ {agent_id} 分布式控制器失败，使用数学策略: {e}")
                                        selected_action = f"数学策略行动_{np.random.randint(1,5)}"
                                        reasoning = f'{agent_id}基于数学策略选择: {selected_action}'
                                        confidence = 0.7
                                else:
                                    # 使用RoleAgent的数学策略
                                    selected_action = f"数学策略行动_{np.random.randint(1,5)}"
                                    reasoning = f'{agent_id}基于数学策略选择: {selected_action}'
                                    confidence = 0.7
                        
                        # 更新智能体状态（注释掉不存在的方法）
                        # role_agent.update_performance(confidence)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ RoleAgent {agent_id} 决策失败，使用模板: {e}")
                        selected_action, reasoning, confidence = self._generate_template_action(agent_id)
                
                else:
                    # 回退到传统LLM生成或模板
                    if self.llm_action_generator and self.config.enable_llm_integration:
                        try:
                            observation_context = {
                                'role': agent_id,
                                'system_state': self._convert_state_to_dict(),
                                'agent_performance': agent_info['performance'],
                                'current_step': self.current_step
                            }
                            
                            llm_action = self.llm_action_generator.generate_action_sync(
                                agent_id, observation_context
                            )
                            
                            selected_action = llm_action.get('action', '维持现状')
                            reasoning = llm_action.get('reasoning', f'{agent_id}基于LLM决策')
                            confidence = llm_action.get('confidence', 0.8)
                            
                        except Exception as e:
                            logger.warning(f"⚠️ LLM生成失败，使用模板: {e}")
                            selected_action, reasoning, confidence = self._generate_template_action(agent_id)
                    else:
                        selected_action, reasoning, confidence = self._generate_template_action(agent_id)
                
                actions[agent_id] = {
                    'action': selected_action,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'strategy_params': agent_info['strategy_params'].tolist(),
                    'agent_type': 'RoleAgent' if agent_id in self.agent_objects else 'Template'
                }
                
                # 更新智能体状态
                agent_info['last_decision'] = selected_action
        
        return actions
    
    def _convert_action_vector_to_description(self, agent_id: str, action_vector: np.ndarray) -> str:
        """将数值行动向量转换为描述性行动"""
        # 基于行动向量的数值生成描述
        if len(action_vector) == 0:
            return "维持现状"
        
        # 计算行动强度
        action_magnitude = np.linalg.norm(action_vector)
        dominant_action_idx = np.argmax(np.abs(action_vector))
        
        # 角色特定的行动映射
        action_mappings = {
            'doctors': [
                '提高医疗诊断精度', '优化治疗方案', '加强患者沟通', '改进医疗流程'
            ],
            'interns': [
                '增加学习时间', '寻求导师指导', '参与病例讨论', '提高技能训练'
            ],
            'patients': [
                '提出服务改进建议', '反馈治疗体验', '要求缩短等待时间', '关注医疗质量'
            ],
            'accountants': [
                '优化成本控制', '改善财务流程', '提高预算精度', '加强财务监管'
            ],
            'government': [
                '制定新政策', '加强监管措施', '促进医疗公平', '提高服务标准'
            ]
        }
        
        actions = action_mappings.get(agent_id, ['维持现状', '评估状况', '制定计划', '执行改进'])
        base_action = actions[dominant_action_idx % len(actions)]
        
        # 根据行动强度添加修饰
        if action_magnitude > 0.7:
            return f"强力推进：{base_action}"
        elif action_magnitude > 0.4:
            return f"积极执行：{base_action}"
        else:
            return f"谨慎实施：{base_action}"
    
    def _generate_template_action(self, agent_id: str) -> Tuple[str, str, float]:
        """生成模板行动（回退方案）"""
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
        performance = self.agents[agent_id]['performance']
        confidence = performance * (0.7 + np.random.random() * 0.3)
        reasoning = f"{self.agents[agent_id]['name']}基于当前系统状态和个人策略执行{selected_action}"
        
        return selected_action, reasoning, confidence
    
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
        
        if HAS_CORE_MATH and hasattr(self.system_state, 'to_vector'):
            # 使用core模块的SystemState
            current_vector = self.system_state.to_vector()
            
            # 映射旧的指标名称到新的状态向量索引
            metric_mapping = {
                'medical_quality': [10],  # care_quality_index
                'patient_safety': [11],   # safety_incident_rate (反向)
                'resource_adequacy': [0], # medical_resource_utilization
                'staff_satisfaction': [5], # intern_satisfaction
                'patient_satisfaction': [8], # patient_satisfaction
                'medical_quality': [10],
                'resource_utilization': [0],
                'cost_efficiency': [12],  # operational_efficiency
                'system_stability': [14], # crisis_response_capability
                'regulatory_compliance': [15] # regulatory_compliance_score
            }
            
            for metric in affected_metrics:
                if metric in metric_mapping:
                    indices = metric_mapping[metric]
                    for idx in indices:
                        if metric in ['waiting_times', 'workload', 'safety_incident_rate']:
                            # 反向指标：危机会增加这些指标
                            current_vector[idx] = min(current_vector[idx] + severity * 0.3, 1.0)
                        else:
                            # 正向指标：危机会减少这些指标
                            current_vector[idx] = max(current_vector[idx] - severity * 0.2, 0.1)
            
            # 更新危机响应能力（表示当前危机严重程度）
            current_vector[14] = max(0.1, current_vector[14] - severity * 0.4)
            
            # 更新SystemState
            self.system_state = SystemState.from_vector(current_vector)
            
        else:
            # 回退到旧的字典处理方式
            state_dict = self._convert_state_to_dict()
            for metric in affected_metrics:
                if metric in state_dict:
                    if metric in ['waiting_times', 'workload']:
                        state_dict[metric] += severity * 0.3
                        state_dict[metric] = min(state_dict[metric], 1.0)
                    else:
                        state_dict[metric] -= severity * 0.2
                        state_dict[metric] = max(state_dict[metric], 0.1)
            
            state_dict['crisis_severity'] = max(
                state_dict.get('crisis_severity', 0), severity
            )
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        # 获取当前状态的字典表示
        state_dict = self._convert_state_to_dict()
        
        # 计算各个维度的平均表现
        medical_dimension = np.mean([
            state_dict.get('medical_quality', 0.5),
            state_dict.get('patient_safety', 0.5),
            state_dict.get('care_quality', 0.5)
        ])
        
        resource_dimension = np.mean([
            state_dict.get('resource_adequacy', 0.5),
            state_dict.get('resource_utilization', 0.5),
            state_dict.get('resource_access', 0.5)
        ])
        
        financial_dimension = np.mean([
            state_dict.get('financial_health', 0.5),
            state_dict.get('cost_efficiency', 0.5),
            state_dict.get('revenue_growth', 0.5)
        ])
        
        satisfaction_dimension = np.mean([
            state_dict.get('patient_satisfaction', 0.5),
            state_dict.get('staff_satisfaction', 0.5)
        ])
        
        # 总体性能指标
        overall_performance = np.mean([
            medical_dimension,
            resource_dimension, 
            financial_dimension,
            satisfaction_dimension
        ])
        
        metrics = {
            'medical_dimension': medical_dimension,
            'resource_dimension': resource_dimension,
            'financial_dimension': financial_dimension,
            'satisfaction_dimension': satisfaction_dimension,
            'overall_performance': overall_performance,
            'system_stability': state_dict.get('system_stability', 0.5),
            'crisis_count': len(self.crisis_history),
            'active_crisis': state_dict.get('crisis_severity', 0.0) > 0.1,
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
        
        # 重置系统状态 - 使用core模块的SystemState
        if HAS_CORE_MATH:
            from ..core.kallipolis_mathematical_core import SystemState
            self.system_state = SystemState(
                medical_resource_utilization=0.7,
                patient_waiting_time=0.3,
                financial_indicator=0.8,
                ethical_compliance=0.85,
                education_training_quality=0.7,
                intern_satisfaction=0.7,
                professional_development=0.6,
                mentorship_effectiveness=0.8,
                patient_satisfaction=0.85,
                service_accessibility=0.8,
                care_quality_index=0.9,
                safety_incident_rate=0.05,
                operational_efficiency=0.75,
                staff_workload_balance=0.7,
                crisis_response_capability=0.8,
                regulatory_compliance_score=0.9
            )
        
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
                'system_state': self._convert_state_to_dict(),
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
                },
                'learning_status': {
                    'maddpg_stats': self.get_maddpg_stats(),
                    'learning_enabled': self.config.enable_learning,
                    'llm_enabled': self.config.enable_llm_integration,
                    'agent_objects_count': len(self.agent_objects)
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