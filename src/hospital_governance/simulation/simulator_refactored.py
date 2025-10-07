"""
Kallipolis医疗共和国仿真器 - 重构版本
KallipolisSimulator - Refactored Version

基于新的智能体注册中心和奖励控制系统的统一仿真架构
"""

import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
import time
import json
from dataclasses import dataclass, field
import traceback

# 导入新的智能体注册中心
try:
    from ..agents.agent_registry import (
        AgentRegistry, AgentRegistryConfig, LLMProviderType,
        create_agent_registry, get_global_agent_registry
    )
    HAS_AGENT_REGISTRY = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"智能体注册中心导入失败: {e}")
    HAS_AGENT_REGISTRY = False

# 导入核心数学系统
try:
    from ..core.kallipolis_mathematical_core import SystemState, KallipolisMedicalSystem
    from ..core.system_dynamics import SystemDynamics
    from ..core.system_matrices import SystemMatrixGenerator
    from ..core.state_space import StateSpace
    HAS_CORE_MATH = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Core数学模块导入失败: {e}")
    HAS_CORE_MATH = False

# 导入控制系统
try:
    from ..control.distributed_reward_control import (
        DistributedRewardControlSystem, DistributedRewardControlConfig,
        get_global_reward_control_system
    )
    HAS_REWARD_CONTROL = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"奖励控制系统导入失败: {e}")
    HAS_REWARD_CONTROL = False

# 导入神圣法典管理器
try:
    from ..holy_code.holy_code_manager import HolyCodeManager, HolyCodeConfig
    HAS_HOLY_CODE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"神圣法典系统导入失败: {e}")
    HAS_HOLY_CODE = False

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """仿真配置"""
    # 基础仿真参数
    max_steps: int = 14
    time_scale: float = 1.0
    meeting_interval: int = 7
    
    # 功能开关
    enable_learning: bool = True
    enable_holy_code: bool = True
    enable_crises: bool = True
    enable_llm_integration: bool = True
    enable_reward_control: bool = True
    
    # LLM配置
    llm_provider: str = "mock"
    llm_fallback_to_mock: bool = True
    
    # 数据配置
    data_logging_interval: int = 10
    crisis_probability: float = 0.03
    
    # 高级配置
    holy_code_config_path: str = 'config/holy_code_rules.yaml'
    system_matrices_config: Optional[Dict] = None
    performance_weights: Optional[Dict] = None

class KallipolisSimulator:
    """Kallipolis医疗共和国仿真器 - 重构版本
    
    统一的仿真管理器，集成：
    1. 智能体注册中心 (AgentRegistry)
    2. 奖励控制系统 (RewardControlSystem)  
    3. 状态空间管理 (StateSpace)
    4. 神圣法典系统 (HolyCodeManager)
    5. 系统动力学 (SystemDynamics)
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
        self.history = {
            'decisions': [],
            'interactions': [],
            'crises': [],
            'performance': [],
            'parliament': [],
            'rewards': []
        }
        
        # 核心系统组件
        self.agent_registry: Optional[AgentRegistry] = None
        self.reward_control_system: Optional[DistributedRewardControlSystem] = None
        self.holy_code_manager: Optional[HolyCodeManager] = None
        self.core_system: Optional[KallipolisMedicalSystem] = None
        self.system_dynamics: Optional[SystemDynamics] = None
        self.state_space: Optional[StateSpace] = None
        
        # 初始化核心组件
        self._initialize_components()
        
        logger.info("🏥 KallipolisSimulator重构版本初始化完成")
    
    def _initialize_components(self):
        """初始化所有核心组件"""
        try:
            # 1. 初始化智能体注册中心
            self._initialize_agent_registry()
            
            # 2. 初始化核心数学系统
            self._initialize_core_math_system()
            
            # 3. 初始化奖励控制系统
            self._initialize_reward_control_system()
            
            # 4. 初始化神圣法典管理器
            self._initialize_holy_code_manager()
            
            # 5. 验证组件集成
            self._validate_component_integration()
            
            logger.info("✅ 所有核心组件初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            logger.error(traceback.format_exc())
            self._initialize_fallback_mode()
    
    def _initialize_agent_registry(self):
        """初始化智能体注册中心"""
        if not HAS_AGENT_REGISTRY:
            logger.warning("⚠️ 智能体注册中心模块不可用，跳过初始化")
            return
        
        try:
            # 创建智能体注册中心
            self.agent_registry = create_agent_registry(
                llm_provider=self.config.llm_provider,
                enable_llm=self.config.enable_llm_integration,
                fallback_to_mock=self.config.llm_fallback_to_mock
            )
            
            # 注册所有智能体
            agents = self.agent_registry.register_all_agents()
            
            logger.info(f"✅ 智能体注册中心初始化完成，注册了 {len(agents)} 个智能体")
            
            # 测试LLM功能
            if self.config.enable_llm_integration:
                test_results = self.agent_registry.test_llm_generation()
                success_count = sum(1 for r in test_results.values() if r['status'] == 'success')
                logger.info(f"🧠 LLM测试完成: {success_count}/{len(test_results)} 成功")
            
        except Exception as e:
            logger.error(f"❌ 智能体注册中心初始化失败: {e}")
            self.agent_registry = None
    
    def _initialize_core_math_system(self):
        """初始化核心数学系统"""
        if not HAS_CORE_MATH:
            logger.warning("⚠️ 核心数学模块不可用，跳过初始化")
            return
        
        try:
            # 初始化核心医疗系统
            self.core_system = KallipolisMedicalSystem()
            
            # 获取系统矩阵
            system_matrices = SystemMatrixGenerator.generate_nominal_matrices()
            
            # 初始化系统动力学
            self.system_dynamics = SystemDynamics(system_matrices)
            
            # 初始化状态空间
            initial_state = self.core_system.current_state.to_vector()
            self.state_space = StateSpace(initial_state)
            
            logger.info("✅ 核心数学系统初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 核心数学系统初始化失败: {e}")
            self.core_system = None
            self.system_dynamics = None
            self.state_space = None
    
    def _initialize_reward_control_system(self):
        """初始化奖励控制系统"""
        if not HAS_REWARD_CONTROL or not self.agent_registry:
            logger.warning("⚠️ 奖励控制系统不可用或智能体注册失败，跳过初始化")
            return
        
        try:
            # 获取全局奖励控制系统
            self.reward_control_system = get_global_reward_control_system()
            
            # 角色名称映射：智能体注册中心使用复数，奖励控制使用单数
            role_mapping = {
                'doctors': 'doctor',
                'interns': 'intern', 
                'patients': 'patient',
                'accountants': 'accountant',
                'government': 'government'  # 这个保持不变
            }
            
            # 将智能体注册到奖励控制系统
            agents = self.agent_registry.get_all_agents()
            for registry_role, agent in agents.items():
                try:
                    # 使用映射获取奖励控制系统的角色名称
                    control_role = role_mapping.get(registry_role, registry_role)
                    
                    # 注册智能体到奖励控制系统
                    self.reward_control_system.register_agent(
                        role=control_role,
                        agent=agent,
                        controller_config=None  # 使用默认配置
                    )
                    
                    logger.info(f"✅ 智能体 {registry_role} -> {control_role} 已集成到奖励控制系统")
                    
                except Exception as e:
                    logger.warning(f"⚠️ 智能体 {registry_role} 集成失败: {e}")
            
            logger.info("✅ 奖励控制系统初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 奖励控制系统初始化失败: {e}")
            self.reward_control_system = None
    
    def _initialize_holy_code_manager(self):
        """初始化神圣法典管理器"""
        if not HAS_HOLY_CODE:
            logger.warning("⚠️ 神圣法典模块不可用，跳过初始化")
            return
        
        try:
            holy_config = HolyCodeConfig(
                rule_config_path=self.config.holy_code_config_path
            )
            self.holy_code_manager = HolyCodeManager(holy_config)
            
            logger.info("✅ 神圣法典管理器初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 神圣法典管理器初始化失败: {e}")
            self.holy_code_manager = None
    
    def _validate_component_integration(self):
        """验证组件集成状态"""
        status = {
            'agent_registry': self.agent_registry is not None,
            'core_math_system': self.core_system is not None,
            'reward_control': self.reward_control_system is not None,
            'holy_code': self.holy_code_manager is not None,
            'system_dynamics': self.system_dynamics is not None
        }
        
        total_components = len(status)
        active_components = sum(status.values())
        
        logger.info(f"📊 组件集成状态: {active_components}/{total_components}")
        for component, active in status.items():
            status_icon = "✅" if active else "❌"
            logger.info(f"  {status_icon} {component}")
        
        if active_components < total_components // 2:
            logger.warning("⚠️ 过多组件初始化失败，启动降级模式")
            self._initialize_fallback_mode()
    
    def _initialize_fallback_mode(self):
        """初始化降级模式"""
        logger.info("🔄 启动仿真器降级模式")
        
        # 创建简化的智能体状态
        self.fallback_agents = {
            'doctors': {'name': '医生群体', 'performance': 0.8, 'active': True, 'payoff': 0.0},
            'interns': {'name': '实习生群体', 'performance': 0.7, 'active': True, 'payoff': 0.0},
            'patients': {'name': '患者代表', 'performance': 0.75, 'active': True, 'payoff': 0.0},
            'accountants': {'name': '会计群体', 'performance': 0.8, 'active': True, 'payoff': 0.0},
            'government': {'name': '政府监管', 'performance': 0.75, 'active': True, 'payoff': 0.0}
        }
        
        # 创建简化的系统状态
        self.fallback_state = {
            'medical_quality': 0.85, 'patient_safety': 0.9, 'care_quality': 0.8,
            'resource_adequacy': 0.7, 'financial_health': 0.8, 'patient_satisfaction': 0.75,
            'staff_satisfaction': 0.7, 'system_stability': 0.8, 'overall_performance': 0.78
        }
        
        logger.info("✅ 降级模式初始化完成")
    
    def set_data_callback(self, callback: Callable):
        """设置数据推送回调函数"""
        self.data_callback = callback
        logger.info("📡 数据回调已设置")
    
    def step(self, training: bool = False) -> Dict[str, Any]:
        """执行一个仿真步骤"""
        self.current_step += 1
        self.simulation_time += self.config.time_scale
        
        # 初始化步骤数据
        step_data = {
            'step': self.current_step,
            'time': self.simulation_time,
            'system_state': {},
            'agent_actions': {},
            'rewards': {},
            'metrics': {},
            'crises': [],
            'parliament_meeting': False,
            'component_status': self._get_component_status()
        }
        
        try:
            # 1. 更新系统状态
            self._update_system_state()
            step_data['system_state'] = self._get_current_state_dict()
            
            # 2. 智能体决策
            if self.agent_registry:
                step_data['agent_actions'] = self._process_agent_decisions()
            else:
                step_data['agent_actions'] = self._process_fallback_decisions()
            
            # 3. 奖励计算和分发
            if self.reward_control_system:
                step_data['rewards'] = self._compute_and_distribute_rewards(step_data)
            else:
                step_data['rewards'] = self._compute_fallback_rewards()
            
            # 4. 处理议会会议
            if self.current_step % self.config.meeting_interval == 0:
                step_data['parliament_meeting'] = True
                step_data['parliament_result'] = self._run_parliament_meeting(step_data)
            
            # 5. 处理危机事件
            if self.config.enable_crises:
                step_data['crises'] = self._handle_crisis_events()
            
            # 6. 计算性能指标
            step_data['metrics'] = self._calculate_performance_metrics(step_data)
            
            # 7. 记录历史数据
            self._record_step_history(step_data)
            
            # 8. 推送数据
            self._push_data_callback(step_data)
            
        except Exception as e:
            logger.error(f"❌ 仿真步骤 {self.current_step} 执行失败: {e}")
            step_data['error'] = str(e)
        
        return step_data
    
    def _update_system_state(self):
        """更新系统状态"""
        if self.system_dynamics and self.state_space:
            try:
                # 使用系统动力学更新状态
                current_state = self.state_space.get_state_vector()
                
                # 生成控制输入（简化版本）
                u_t = np.random.normal(0, 0.1, 17)
                d_t = np.random.normal(0, 0.05, 6)
                
                # 状态转移
                next_state = self.system_dynamics.state_transition(current_state, u_t, d_t)
                
                # 更新状态空间
                self.state_space.update_state(next_state)
                
                # 更新核心系统状态
                if self.core_system:
                    self.core_system.current_state = SystemState.from_vector(next_state)
                
            except Exception as e:
                logger.warning(f"⚠️ 系统动力学更新失败: {e}")
                self._update_fallback_state()
        else:
            self._update_fallback_state()
    
    def _update_fallback_state(self):
        """更新降级模式的系统状态"""
        if hasattr(self, 'fallback_state'):
            # 简单的随机游走
            for key in self.fallback_state:
                if key != 'overall_performance':
                    noise = np.random.normal(0, 0.01)
                    self.fallback_state[key] = np.clip(
                        self.fallback_state[key] + noise, 0.1, 1.0
                    )
            
            # 重新计算总体性能
            self.fallback_state['overall_performance'] = np.mean([
                self.fallback_state['medical_quality'],
                self.fallback_state['financial_health'],
                self.fallback_state['patient_satisfaction'],
                self.fallback_state['system_stability']
            ])
    
    def _get_current_state_dict(self) -> Dict[str, float]:
        """获取当前状态的字典表示"""
        if self.core_system and hasattr(self.core_system, 'current_state'):
            try:
                state_vector = self.core_system.current_state.to_vector()
                return {
                    'medical_resource_utilization': state_vector[0],
                    'patient_waiting_time': state_vector[1],
                    'financial_indicator': state_vector[2],
                    'ethical_compliance': state_vector[3],
                    'education_training_quality': state_vector[4],
                    'intern_satisfaction': state_vector[5],
                    'professional_development': state_vector[6],
                    'mentorship_effectiveness': state_vector[7],
                    'patient_satisfaction': state_vector[8],
                    'service_accessibility': state_vector[9],
                    'care_quality_index': state_vector[10],
                    'safety_incident_rate': state_vector[11],
                    'operational_efficiency': state_vector[12],
                    'staff_workload_balance': state_vector[13],
                    'crisis_response_capability': state_vector[14],
                    'regulatory_compliance_score': state_vector[15],
                    'overall_performance': np.mean(state_vector)
                }
            except Exception as e:
                logger.warning(f"⚠️ 状态转换失败: {e}")
        
        # 降级到简化状态
        return getattr(self, 'fallback_state', {
            'medical_quality': 0.8, 'financial_health': 0.7, 'patient_satisfaction': 0.75,
            'system_stability': 0.8, 'overall_performance': 0.77
        })
    
    def _process_agent_decisions(self) -> Dict[str, Any]:
        """处理智能体决策（使用新的注册中心）"""
        actions = {}
        
        try:
            agents = self.agent_registry.get_all_agents()
            
            for role, agent in agents.items():
                try:
                    # 生成观测
                    observation = self._generate_observation_for_agent(role)
                    
                    # 智能体决策
                    if hasattr(agent, 'select_action_with_llm'):
                        action = agent.select_action_with_llm(
                            observation=observation,
                            use_llm=self.config.enable_llm_integration
                        )
                    else:
                        action = agent.sample_action(observation)
                    
                    # 记录行动
                    actions[role] = {
                        'action_vector': action.tolist() if hasattr(action, 'tolist') else action,
                        'agent_type': 'RoleAgent',
                        'confidence': 0.8,
                        'reasoning': f'{role}基于新架构决策'
                    }
                    
                except Exception as e:
                    logger.warning(f"⚠️ 智能体 {role} 决策失败: {e}")
                    actions[role] = {
                        'action_vector': [0.0] * 3,
                        'agent_type': 'Fallback',
                        'confidence': 0.5,
                        'reasoning': f'{role}使用默认行动'
                    }
            
        except Exception as e:
            logger.error(f"❌ 智能体决策处理失败: {e}")
            return self._process_fallback_decisions()
        
        return actions
    
    def _generate_observation_for_agent(self, role: str) -> np.ndarray:
        """为智能体生成观测"""
        if self.state_space:
            # 使用完整的16维状态空间
            return self.state_space.get_state_vector()
        else:
            # 降级到简化观测
            state_dict = self._get_current_state_dict()
            return np.array([
                state_dict.get('medical_quality', 0.8),
                state_dict.get('financial_health', 0.7),
                state_dict.get('patient_satisfaction', 0.75),
                state_dict.get('system_stability', 0.8),
                state_dict.get('overall_performance', 0.77),
                0.0, 0.0, 0.0  # 填充到8维
            ])
    
    def _process_fallback_decisions(self) -> Dict[str, Any]:
        """处理降级模式的智能体决策"""
        actions = {}
        
        if hasattr(self, 'fallback_agents'):
            for role, agent_info in self.fallback_agents.items():
                if agent_info['active'] and np.random.random() < 0.7:
                    actions[role] = {
                        'action_vector': np.random.uniform(-0.5, 0.5, 3).tolist(),
                        'agent_type': 'Fallback',
                        'confidence': agent_info['performance'],
                        'reasoning': f'{agent_info["name"]}基于简化逻辑决策'
                    }
        
        return actions
    
    def _compute_and_distribute_rewards(self, step_data: Dict[str, Any]) -> Dict[str, float]:
        """计算和分发奖励（使用奖励控制系统）"""
        try:
            if not self.reward_control_system:
                return self._compute_fallback_rewards()
            
            # 角色名称映射
            role_mapping = {
                'doctors': 'doctor',
                'interns': 'intern', 
                'patients': 'patient',
                'accountants': 'accountant',
                'government': 'government'
            }
            
            # 获取基础奖励（使用智能体注册中心的角色名称）
            base_rewards = {}
            for registry_role in step_data['agent_actions'].keys():
                performance = step_data['metrics'].get('overall_performance', 0.5)
                base_rewards[registry_role] = performance + np.random.normal(0, 0.1)
            
            # 转换为奖励控制系统的角色名称
            control_base_rewards = {}
            for registry_role, reward in base_rewards.items():
                control_role = role_mapping.get(registry_role, registry_role)
                control_base_rewards[control_role] = reward
            
            
            # 使用分布式奖励控制系统
            global_utility = step_data['metrics'].get('overall_performance', 0.5)
            control_context = {role: {} for role in control_base_rewards.keys()}
            
            # 异步调用奖励计算
            try:
                control_final_rewards = asyncio.run(
                    self.reward_control_system.compute_distributed_rewards(
                        control_base_rewards, global_utility, control_context
                    )
                )
                
                # 转换回智能体注册中心的角色名称
                final_rewards = {}
                reverse_mapping = {v: k for k, v in role_mapping.items()}
                for control_role, reward in control_final_rewards.items():
                    registry_role = reverse_mapping.get(control_role, control_role)
                    final_rewards[registry_role] = reward
                    
                return final_rewards
                
            except Exception as e:
                logger.warning(f"⚠️ 异步奖励计算失败，使用同步方式: {e}")
                # 转换回智能体注册中心的角色名称
                final_rewards = {}
                reverse_mapping = {v: k for k, v in role_mapping.items()}
                for control_role, reward in control_base_rewards.items():
                    registry_role = reverse_mapping.get(control_role, control_role)
                    final_rewards[registry_role] = reward
                return final_rewards
            
        except Exception as e:
            logger.warning(f"⚠️ 奖励计算失败: {e}")
            return self._compute_fallback_rewards()
    
    def _compute_fallback_rewards(self) -> Dict[str, float]:
        """计算降级模式的奖励"""
        rewards = {}
        
        if hasattr(self, 'fallback_agents'):
            for role, agent_info in self.fallback_agents.items():
                if agent_info['active']:
                    base_reward = agent_info['performance'] * 0.1
                    noise = np.random.normal(0, 0.02)
                    rewards[role] = np.clip(base_reward + noise, -0.1, 0.2)
        
        return rewards
    
    def _run_parliament_meeting(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行议会会议"""
        try:
            if self.holy_code_manager and self.agent_registry:
                # 使用神圣法典管理器运行议会
                agents_dict = {}
                for role, agent in self.agent_registry.get_all_agents().items():
                    agents_dict[role] = {
                        'name': f'{role}群体',
                        'performance': step_data['metrics'].get('overall_performance', 0.5),
                        'active': True
                    }
                
                parliament_result = self.holy_code_manager.run_weekly_parliament_meeting(
                    agents_dict, step_data['system_state']
                )
                
                # 记录议会历史
                self.history['parliament'].append({
                    'step': self.current_step,
                    'result': parliament_result,
                    'timestamp': time.time()
                })
                
                return parliament_result
            else:
                return self._run_fallback_parliament_meeting(step_data)
        
        except Exception as e:
            logger.error(f"❌ 议会会议失败: {e}")
            return {'error': str(e)}
    
    def _run_fallback_parliament_meeting(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行降级模式的议会会议"""
        # 简化的议会流程
        consensus_level = np.random.uniform(0.5, 0.9)
        decision = "维持当前政策"
        
        return {
            'consensus_level': consensus_level,
            'main_decision': decision,
            'participating_agents': list(step_data['agent_actions'].keys())
        }
    
    def _handle_crisis_events(self) -> List[Dict[str, Any]]:
        """处理危机事件"""
        crises = []
        
        if np.random.random() < self.config.crisis_probability:
            crisis = {
                'type': np.random.choice(['pandemic', 'funding_cut', 'staff_shortage']),
                'severity': np.random.uniform(0.2, 0.8),
                'duration': np.random.randint(5, 15),
                'start_step': self.current_step
            }
            
            self.history['crises'].append(crisis)
            crises.append(crisis)
            
            logger.info(f"🚨 危机事件: {crisis['type']} (严重程度: {crisis['severity']:.2f})")
        
        return crises
    
    def _calculate_performance_metrics(self, step_data: Dict[str, Any]) -> Dict[str, float]:
        """计算性能指标"""
        state_dict = step_data['system_state']
        
        # 基础指标
        medical_performance = state_dict.get('care_quality_index', 0.8)
        financial_performance = state_dict.get('financial_indicator', 0.7)
        satisfaction_performance = state_dict.get('patient_satisfaction', 0.75)
        stability_performance = state_dict.get('crisis_response_capability', 0.8)
        
        # 综合指标
        overall_performance = np.mean([
            medical_performance, financial_performance, 
            satisfaction_performance, stability_performance
        ])
        
        metrics = {
            'medical_performance': medical_performance,
            'financial_performance': financial_performance,
            'satisfaction_performance': satisfaction_performance,
            'stability_performance': stability_performance,
            'overall_performance': overall_performance,
            'crisis_count': len(self.history['crises']),
            'parliament_meetings': len(self.history['parliament']),
            'agent_actions_count': len(step_data['agent_actions'])
        }
        
        return metrics
    
    def _record_step_history(self, step_data: Dict[str, Any]):
        """记录步骤历史"""
        # 记录性能历史
        self.history['performance'].append(step_data['metrics'])
        
        # 记录奖励历史
        if step_data['rewards']:
            self.history['rewards'].append({
                'step': self.current_step,
                'rewards': step_data['rewards']
            })
        
        # 限制历史长度
        max_history = 1000
        for key in self.history:
            if len(self.history[key]) > max_history:
                self.history[key] = self.history[key][-max_history:]
    
    def _push_data_callback(self, step_data: Dict[str, Any]):
        """推送数据到回调函数"""
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
                        asyncio.run(self.data_callback(step_data))
                else:
                    # 同步回调
                    self.data_callback(step_data)
            except Exception as e:
                logger.error(f"❌ 数据回调执行失败: {e}")
    
    def _get_component_status(self) -> Dict[str, bool]:
        """获取组件状态"""
        return {
            'agent_registry': self.agent_registry is not None,
            'reward_control': self.reward_control_system is not None,
            'holy_code': self.holy_code_manager is not None,
            'core_math': self.core_system is not None,
            'system_dynamics': self.system_dynamics is not None,
            'state_space': self.state_space is not None
        }
    
    # 仿真控制方法
    async def run_async(self, steps: int = None, training: bool = False):
        """异步运行仿真"""
        if steps is None:
            steps = self.config.max_steps
        
        self.is_running = True
        logger.info(f"🚀 开始异步仿真: {steps}步")
        
        try:
            for step in range(steps):
                if not self.is_running:
                    break
                
                while self.is_paused:
                    await asyncio.sleep(0.1)
                
                step_data = self.step(training=training)
                await asyncio.sleep(1.0)  # 1秒间隔
                
                if step % 50 == 0:
                    perf = step_data['metrics']['overall_performance']
                    logger.info(f"📊 进度: {step}/{steps}, 性能: {perf:.3f}")
        
        except Exception as e:
            logger.error(f"❌ 异步仿真失败: {e}")
        finally:
            self.is_running = False
            logger.info("✅ 异步仿真完成")
    
    def run(self, steps: int = None, training: bool = False):
        """同步运行仿真"""
        if steps is None:
            steps = self.config.max_steps
        
        self.is_running = True
        results = []
        
        logger.info(f"🚀 开始同步仿真: {steps}步")
        
        try:
            for step in range(steps):
                if not self.is_running:
                    break
                
                step_data = self.step(training=training)
                results.append(step_data)
                
                if step % 50 == 0:
                    perf = step_data['metrics']['overall_performance']
                    logger.info(f"📊 进度: {step}/{steps}, 性能: {perf:.3f}")
        
        except Exception as e:
            logger.error(f"❌ 同步仿真失败: {e}")
        finally:
            self.is_running = False
            logger.info("✅ 同步仿真完成")
        
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
        
        # 清理历史记录
        for key in self.history:
            self.history[key].clear()
        
        # 重置组件状态
        if self.state_space and self.core_system:
            try:
                initial_state = self.core_system.current_state.to_vector()
                # 直接更新状态而不是调用reset方法
                if hasattr(self.state_space, 'update_state'):
                    self.state_space.update_state(initial_state)
                elif hasattr(self.state_space, '_current_state'):
                    self.state_space._current_state = initial_state
            except Exception as e:
                logger.warning(f"⚠️ 状态空间重置失败: {e}")
        
        logger.info("🔄 仿真器已重置")
    
    def get_simulation_report(self) -> Dict[str, Any]:
        """获取仿真报告"""
        try:
            component_status = self._get_component_status()
            active_components = sum(component_status.values())
            
            return {
                'simulation_info': {
                    'current_step': self.current_step,
                    'simulation_time': self.simulation_time,
                    'is_running': self.is_running,
                    'is_paused': self.is_paused,
                    'version': 'refactored'
                },
                'component_status': component_status,
                'component_health': f"{active_components}/{len(component_status)}",
                'system_state': self._get_current_state_dict(),
                'performance_summary': {
                    'recent_performance': self.history['performance'][-10:] if self.history['performance'] else [],
                    'crisis_count': len(self.history['crises']),
                    'parliament_meetings': len(self.history['parliament']),
                    'total_decisions': len(self.history['decisions'])
                },
                'agent_registry_status': self.agent_registry.get_registry_status() if self.agent_registry else None,
                'reward_control_status': 'active' if self.reward_control_system else 'inactive',
                'config': {
                    'max_steps': self.config.max_steps,
                    'enable_llm': self.config.enable_llm_integration,
                    'enable_reward_control': self.config.enable_reward_control,
                    'llm_provider': self.config.llm_provider
                }
            }
        except Exception as e:
            logger.error(f"❌ 生成仿真报告失败: {e}")
            return {
                'error': str(e),
                'current_step': self.current_step,
                'component_status': self._get_component_status()
            }

# 导出
__all__ = ['KallipolisSimulator', 'SimulationConfig']