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

# 导入场景运行器
try:
    from .scenario_runner import ScenarioRunner, CrisisScenario, ScenarioType
    HAS_SCENARIO_RUNNER = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"场景运行器导入失败: {e}")
    HAS_SCENARIO_RUNNER = False

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

# 导入PPO学习模型
try:
    from ..agents.learning_models import RolloutBuffer, AgentStep
    HAS_PPO_MODELS = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"PPO模型导入失败: {e}")
    HAS_PPO_MODELS = False
    # 创建占位类
    RolloutBuffer = None
    AgentStep = None

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
    
    # PPO训练配置（如有需要可扩展）
    ppo_training_episodes: int = 100
    ppo_batch_size: int = 32
    ppo_model_save_path: str = 'models/ppo'
    ppo_buffer_size: int = 10000

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
        self.scenario_runner: Optional[ScenarioRunner] = None
        # PPO经验回放缓冲区
        self.rollout_buffer: Optional[RolloutBuffer] = None
        self.parliament_waiting: bool = False
        self.last_parliament_step: int = 0
        # PPO经验存储
        self.experience_buffer: List[Dict[str, Any]] = []
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
            
            # 6. 初始化场景运行器
            self._initialize_scenario_runner()
            
            # 7. 初始化PPO学习系统
            self._initialize_ppo_learning_system()
            
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
            
            # 获取系统矩阵（优先从YAML加载，失败则回退到标称矩阵）
            try:
                matrices_cfg = self.config.system_matrices_config or {}
                yaml_path = matrices_cfg.get('path', 'config/system_matrices.yaml')
                scenario = matrices_cfg.get('scenario') if isinstance(matrices_cfg, dict) else None
                system_matrices = SystemMatrixGenerator.load_from_yaml(
                    yaml_path=yaml_path,
                    scenario=scenario,
                    n=16, m=17, p=6
                )
                logger.info(f"✅ 已从YAML加载系统矩阵: {yaml_path} (scenario={scenario})")
            except Exception as load_err:
                logger.warning(f"⚠️ 从YAML加载系统矩阵失败，使用标称矩阵: {load_err}")
                system_matrices = SystemMatrixGenerator.generate_nominal_matrices()
            
            # 初始化系统动力学
            self.system_dynamics = SystemDynamics(system_matrices)
            
            # 初始化状态空间（传入SystemState对象，而非向量）
            initial_state = self.core_system.current_state
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
    
    def _initialize_scenario_runner(self):
        """初始化场景运行器"""
        if not HAS_SCENARIO_RUNNER:
            logger.warning("⚠️ ScenarioRunner模块未可用")
            return
            
        try:
            from pathlib import Path
            self.scenario_runner = ScenarioRunner(self)
            
            # 尝试加载默认场景配置
            scenario_config_path = Path(__file__).parent.parent.parent.parent / "config" / "simulation_scenarios.yaml"
            if scenario_config_path.exists():
                self.scenario_runner.load_scenarios_from_yaml(str(scenario_config_path))
                logger.info(f"✅ 从 {scenario_config_path} 加载场景配置")
            else:
                logger.info("📋 使用内置预设场景")
                # 创建一些默认场景
                presets = self.scenario_runner.create_preset_scenarios()
                self.scenario_runner.scenarios = list(presets.values())
            
            logger.info("✅ 场景运行器初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 场景运行器初始化失败: {e}")
            self.scenario_runner = None
    
    def _initialize_ppo_learning_system(self):
        """初始化PPO学习系统"""
        if not HAS_PPO_MODELS:
            logger.warning("⚠️ PPO学习模型模块不可用，跳过初始化")
            return
        
        try:
            # PPO模型将在需要时（训练模式）动态初始化
            # 这里只是验证PPO组件可用性并准备RolloutBuffer
            if self.config.enable_learning:
                logger.info("✅ PPO学习系统准备就绪（RolloutBuffer将在收集经验时初始化）")
                # RolloutBuffer会在_collect_experience_data中根据智能体数量动态初始化
            else:
                logger.info("ℹ️ 学习模式未启用，PPO系统待命")
                
        except Exception as e:
            logger.error(f"❌ PPO学习系统初始化失败: {e}")
    
    def _validate_component_integration(self):
        """验证组件集成状态"""
        status = {
            'agent_registry': self.agent_registry is not None,
            'core_math_system': self.core_system is not None,
            'reward_control': self.reward_control_system is not None,
            'holy_code': self.holy_code_manager is not None,
            'system_dynamics': self.system_dynamics is not None,
            'scenario_runner': self.scenario_runner is not None,
            'ppo_learning': HAS_PPO_MODELS and self.config.enable_learning
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
            
            # 2. 智能体协作决策（LLM+角色智能体）
            llm_decisions = None
            if self.agent_registry and self.config.enable_llm_integration:
                llm_decisions = self._process_agent_decisions()
            step_data['agent_actions'] = llm_decisions if llm_decisions else self._process_fallback_decisions()
            
            # 3. 奖励计算和分发
            if self.reward_control_system:
                step_data['rewards'] = self._compute_and_distribute_rewards(step_data)
            else:
                step_data['rewards'] = self._compute_fallback_rewards()
            
            # 4. 处理议会会议
            if self._should_hold_parliament():
                step_data['parliament_meeting'] = True
                step_data['parliament_result'] = self._run_parliament_meeting(step_data)
            
            # 5. 处理危机事件
            if self.config.enable_crises:
                step_data['crises'] = self._handle_crisis_events()
            
            # 6. 计算性能指标
            step_data['metrics'] = self._calculate_performance_metrics(step_data)
            
            # 7. 记录历史数据
            self._record_step_history(step_data)
            
            # 8. 收集PPO经验数据
            if self.config.enable_learning:
                self._collect_experience_data(step_data)
            
            # 9. 推送数据
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
                next_state_vector = self.system_dynamics.state_transition(current_state, u_t, d_t)
                
                # 转换为SystemState对象
                next_state = SystemState.from_vector(next_state_vector)
                
                # 更新状态空间
                self.state_space.update_state(next_state)
                
                # 更新核心系统状态
                if self.core_system:
                    self.core_system.current_state = next_state
                
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
        """运行增强的议会会议（包含LLM智能体讨论和共识生成）"""
        try:
            if self.holy_code_manager and self.agent_registry:
                logger.info("🏛️ 启动LLM增强议会会议...")
                
                # 1. 准备议会参与者信息
                agents_dict = {}
                agent_discussions = {}
                
                for role, agent in self.agent_registry.get_all_agents().items():
                    agents_dict[role] = {
                        'name': f'{role}群体',
                        'performance': step_data['metrics'].get('overall_performance', 0.5),
                        'active': True
                    }
                    
                    # 2. LLM智能体生成议会发言
                    if hasattr(agent, 'llm_generator') and agent.llm_generator:
                        discussion_input = self._generate_parliament_discussion(role, step_data)
                        agent_discussions[role] = discussion_input
                
                # 3. 运行传统议会流程
                base_parliament_result = self.holy_code_manager.run_weekly_parliament_meeting(
                    agents_dict, step_data['system_state']
                )
                
                # 4. 进行LLM智能体讨论和共识达成
                enhanced_result = self._conduct_llm_parliament_discussion(
                    agent_discussions, base_parliament_result, step_data
                )
                
                # 5. 生成新规则（如果达成共识）
                new_rules = self._generate_consensus_rules(enhanced_result, step_data)
                if new_rules:
                    enhanced_result['new_rules_generated'] = new_rules
                    logger.info(f"📜 议会生成了 {len(new_rules)} 条新规则")
                
                # 记录议会历史
                self.history['parliament'].append({
                    'step': self.current_step,
                    'result': enhanced_result,
                    'agent_discussions': agent_discussions,
                    'timestamp': time.time()
                })
                
                return enhanced_result
            else:
                return self._run_fallback_parliament_meeting(step_data)
        
        except Exception as e:
            logger.error(f"❌ 增强议会会议失败: {e}")
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
    
    def _should_hold_parliament(self) -> bool:
        """判断是否应该召开议会会议"""
        if not self.config.enable_holy_code:
            return False
        # 每meeting_interval步召开一次
        return self.current_step % self.config.meeting_interval == 0 and self.current_step > 0
    
    def _handle_crisis_events(self) -> List[Dict[str, Any]]:
        """处理危机事件"""
        crises = []
        
        # 优先使用ScenarioRunner检查是否有预定义的危机事件
        if self.scenario_runner:
            try:
                self.scenario_runner.check_and_insert_event(self.current_step)
            except Exception as e:
                logger.warning(f"⚠️ 场景运行器检查事件失败: {e}")
        
        # 随机生成危机事件
        if np.random.random() < self.config.crisis_probability:
            crisis_types = {
                'pandemic': '疫情爆发，医院面临巨大压力',
                'funding_cut': '资金削减，需要优化资源分配',
                'staff_shortage': '人员短缺，影响医疗服务质量'
            }
            
            crisis_type = np.random.choice(list(crisis_types.keys()))
            crisis = {
                'type': crisis_type,
                'severity': np.random.uniform(0.2, 0.8),
                'duration': np.random.randint(5, 15),
                'start_step': self.current_step,
                'description': crisis_types[crisis_type]  # 添加描述字段
            }
            
            self.history['crises'].append(crisis)
            crises.append(crisis)
            
            logger.info(f"🚨 危机事件: {crisis['type']} - {crisis['description']} (严重程度: {crisis['severity']:.2f})")
        
        return crises
    
    def _apply_crisis_effects(self, crisis_data: Dict[str, Any]):
        """应用危机效果到系统状态"""
        try:
            crisis_type = crisis_data.get('type', 'unknown')
            severity = crisis_data.get('severity', 0.0)
            affected_metrics = crisis_data.get('affected_metrics', [])
            
            logger.info(f"🚨 应用危机效果: {crisis_type} (严重程度: {severity:.2f})")
            
            # 根据危机类型调整系统状态
            if hasattr(self, 'core_system') and self.core_system:
                current_state = self.core_system.get_current_state()
                
                # 应用危机影响
                for metric in affected_metrics:
                    if hasattr(current_state, metric):
                        current_value = getattr(current_state, metric)
                        impact = severity * 0.3  # 最多影响30%
                        new_value = max(0.1, current_value - impact)
                        setattr(current_state, metric, new_value)
                        logger.debug(f"  📉 {metric}: {current_value:.3f} → {new_value:.3f}")
                
                # 更新系统状态
                self.core_system.update_state(current_state)
            
        except Exception as e:
            logger.error(f"❌ 应用危机效果失败: {e}")
    
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
            'state_space': self.state_space is not None,
            'scenario_runner': self.scenario_runner is not None,
            'ppo_learning': HAS_PPO_MODELS and self.config.enable_learning,
            'rollout_buffer': self.rollout_buffer is not None
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
                # 获取初始状态（SystemState对象）
                initial_state = self.core_system.current_state
                # 更新状态空间
                self.state_space.update_state(initial_state)
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
                    'version': 'refactored',
                    'parliament_waiting': self.parliament_waiting
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
                'ppo_status': {
                    'buffer_size': len(self.experience_buffer) if hasattr(self, 'experience_buffer') else 0,
                    'rollout_buffer_initialized': self.rollout_buffer is not None
                },
                'config': {
                    'max_steps': self.config.max_steps,
                    'enable_llm': self.config.enable_llm_integration,
                    'enable_reward_control': self.config.enable_reward_control,
                    'llm_provider': self.config.llm_provider,
                    'ppo_training_episodes': getattr(self.config, 'ppo_training_episodes', 100)
                }
            }
        except Exception as e:
            logger.error(f"❌ 生成仿真报告失败: {e}")
            return {
                'error': str(e),
                'current_step': self.current_step,
                'component_status': self._get_component_status()
            }
    
    
    def _collect_experience_data(self, step_data: Dict[str, Any]):
        """收集经验数据用于PPO训练"""
        if not HAS_PPO_MODELS:
            logger.warning("⚠️ PPO模型未导入，跳过经验收集")
            return
        
        try:
            # 初始化RolloutBuffer（第一次收集时）
            if self.rollout_buffer is None and step_data['agent_actions']:
                n_agents = len(step_data['agent_actions'])
                self.rollout_buffer = RolloutBuffer(n_agents, device='cpu')
                logger.info(f"📊 初始化RolloutBuffer，智能体数量: {n_agents}")
            
            if self.rollout_buffer is None:
                return
            
            current_state = self._get_current_state_dict()
            per_agent_steps = []
            
            # 为每个智能体收集经验
            for role_idx, (role, action_data) in enumerate(step_data['agent_actions'].items()):
                if isinstance(action_data, dict) and 'action_vector' in action_data:
                    obs = self._get_observation_for_role(role, current_state)
                    action = action_data['action_vector']
                    
                    if isinstance(action, list):
                        action = np.array(action, dtype=np.float32)
                    
                    # PPO假设离散动作，将连续动作转换为离散索引
                    if hasattr(action, 'shape') and len(action.shape) > 0 and action.shape[0] > 1:
                        # 多维动作，取最大值的索引
                        action = int(np.argmax(action))
                    elif hasattr(action, 'shape') and action.shape == ():
                        action = int(action)
                    else:
                        action = 0  # 默认动作
                    
                    reward = float(step_data['rewards'].get(role, 0.0))
                    done = step_data.get('done', False) or (self.current_step >= self.config.max_steps - 1)
                    global_state = np.array(list(current_state.values()), dtype=np.float32)
                    logp = float(action_data.get('logp', 0.0))  # 可选，实际PPO需采集
                    
                    per_agent_steps.append(AgentStep(
                        obs=obs,
                        action=action,
                        logp=logp,
                        reward=reward,
                        global_state=global_state,
                        done=done
                    ))
            
            # 添加到RolloutBuffer
            if per_agent_steps:
                self.rollout_buffer.add(per_agent_steps)
                logger.debug(f"✅ 收集了 {len(per_agent_steps)} 个智能体的经验数据")
            
        except Exception as e:
            logger.warning(f"⚠️ 收集经验数据失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _get_observation_for_role(self, role: str, current_state: Dict[str, float]) -> np.ndarray:
        """为智能体生成观测"""
        if self.state_space:
            # 使用完整的16维状态空间
            return self.state_space.get_state_vector()
        else:
            # 降级到简化观测
            state_dict = current_state or self._get_current_state_dict()
            return np.array([
                state_dict.get('medical_quality', 0.8),
                state_dict.get('financial_health', 0.7),
                state_dict.get('patient_satisfaction', 0.75),
                state_dict.get('system_stability', 0.8),
                state_dict.get('overall_performance', 0.77),
                0.0, 0.0, 0.0  # 填充到8维
            ])
    
    def _parse_llm_response(self, llm_response: str, role: str) -> Tuple[List[float], str]:
        """解析LLM响应，提取动作向量和推理"""
        try:
            import re
            
            # 尝试提取向量格式 [x, y, z] 或 (x, y, z)
            vector_pattern = r'[\[\(]([\d\.-]+(?:,\s*[\d\.-]+)*)[\]\)]'
            vector_match = re.search(vector_pattern, llm_response)
            
            if vector_match:
                vector_str = vector_match.group(1)
                action_vector = [float(x.strip()) for x in vector_str.split(',')]
                
                # 规范化到[-1, 1]区间
                action_vector = [max(-1.0, min(1.0, x)) for x in action_vector]
                
                # 提取推理部分
                reasoning_parts = llm_response.split('\n')
                reasoning = next((part.strip() for part in reasoning_parts 
                               if part.strip() and not vector_match.group(0) in part), 
                               f"{role}的LLM决策")
                
                return action_vector, reasoning
            
            else:
                # 如果没有找到向量，基于关键词推断
                action_vector = self._infer_action_from_text(llm_response, role)
                return action_vector, llm_response[:100] + '...'
                
        except Exception as e:
            logger.warning(f"⚠️ 解析LLM响应失败: {e}")
            # 返回默认动作
            return [0.1, 0.1, 0.1], f"{role}默认动作"
    
    def _infer_action_from_text(self, text: str, role: str) -> List[float]:
        """从文本推断动作向量"""
        text_lower = text.lower()
        
        # 角色特定的关键词映射
        role_keywords = {
            'doctors': {
                '提高质量|治疗|诊断': [0.8, 0.2, 0.3, 0.4],
                '节约成本|效率': [0.3, 0.8, 0.2, 0.1],
                '安全|防范': [0.5, 0.1, 0.9, 0.2],
                '培训|教学': [0.2, 0.3, 0.1, 0.8]
            },
            'patients': {
                '满意|服务': [0.8, 0.4, 0.2],
                '投诉|不满': [-0.5, 0.1, 0.7],
                '等待|延误': [0.2, -0.3, 0.5]
            },
            'government': {
                '监管|检查': [0.6, 0.8, 0.4],
                '资金|支持': [0.4, 0.9, 0.2],
                '政策|规定': [0.8, 0.3, 0.7]
            }
        }
        
        if role in role_keywords:
            for keywords, action in role_keywords[role].items():
                if any(keyword in text_lower for keyword in keywords.split('|')):
                    return action
        
        # 默认中性动作
        default_dims = {'doctors': 4, 'interns': 3, 'patients': 3, 'accountants': 3, 'government': 3}
        dim = default_dims.get(role, 3)
        return [0.1] * dim
    
    def _generate_parliament_discussion(self, role: str, step_data: Dict[str, Any]) -> str:
        """生成议会讨论内容"""
        try:
            agent = self.agent_registry.get_agent(role)
            if not (hasattr(agent, 'llm_generator') and agent.llm_generator):
                return f"{role}未参与讨论"
            
            # 构建议会讨论提示
            discussion_prompt = f"""
            作为{role}的代表，在本次议会上，请针对当前医院运营情况发表意见：
            
            当前系统状态：
            - 整体绩效：{step_data['metrics'].get('overall_performance', 0.5):.2f}
            - 医疗质量：{step_data['system_state'].get('care_quality_index', 0.8):.2f}
            - 财务状况：{step_data['system_state'].get('financial_indicator', 0.7):.2f}
            - 患者满意度：{step_data['system_state'].get('patient_satisfaction', 0.75):.2f}
            
            请提出：
            1. 你的角色对当前情况的看法
            2. 你认为需要改进的问题
            3. 具体的改进建议
            4. 你支持制定哪些新规则
            
            请用150字左右表达你的观点：
            """
            
            # 获取LLM响应
            holy_code_state = step_data.get('holy_code_state', {})
            discussion = agent.llm_generator.generate_action_sync(
                role=role,
                observation=np.array([0.5] * 8),  # 议会上下文
                holy_code_state=holy_code_state,
                context={'prompt': discussion_prompt, 'type': 'parliament_discussion', 'system_state': step_data['system_state']}
            )
            
            return discussion
            
        except Exception as e:
            logger.warning(f"⚠️ 生成{role}议会讨论失败: {e}")
            return f"{role}：支持现有政策，建议维持稳定。"
    
    def _conduct_llm_parliament_discussion(self, agent_discussions: Dict[str, str], 
                                         base_result: Dict[str, Any], 
                                         step_data: Dict[str, Any]) -> Dict[str, Any]:
        """进行LLM智能体议会讨论和共识达成"""
        enhanced_result = base_result.copy()
        
        try:
            # 整合所有参与者的观点
            all_discussions = "\n\n".join([
                f"**{role}代表的发言**:\n{discussion}"
                for role, discussion in agent_discussions.items()
            ])
            
            # 分析共同关注点
            common_concerns = self._extract_common_concerns(agent_discussions)
            
            # 评估共识程度
            consensus_level = self._calculate_consensus_level(agent_discussions, step_data)
            
            # 增强结果
            enhanced_result.update({
                'llm_discussions': agent_discussions,
                'all_discussions_summary': all_discussions,
                'common_concerns': common_concerns,
                'consensus_level': consensus_level,
                'discussion_participants': list(agent_discussions.keys()),
                'enhanced_by_llm': True
            })
            
            logger.info(f"💬 议会讨论完成，共识程度: {consensus_level:.2f}")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ 议会讨论失败: {e}")
            enhanced_result['llm_discussion_error'] = str(e)
            return enhanced_result
    
    def _extract_common_concerns(self, discussions: Dict[str, str]) -> List[str]:
        """提取共同关注点"""
        # 关键词分析
        common_keywords = {
            '医疗质量': ['质量', '治疗', '医疗', '诊断'],
            '财务管理': ['成本', '费用', '财务', '预算'],
            '患者服务': ['患者', '服务', '满意', '体验'],
            '人员管理': ['医生', '护士', '人员', '培训'],
            '安全管理': ['安全', '风险', '防范', '事故']
        }
        
        concerns_count = {concern: 0 for concern in common_keywords.keys()}
        
        # 统计关键词出现频率
        for discussion in discussions.values():
            for concern, keywords in common_keywords.items():
                if any(keyword in discussion for keyword in keywords):
                    concerns_count[concern] += 1
        
        # 返回被多数人关注的问题
        threshold = len(discussions) * 0.5  # 超过50%的参与者关注
        common_concerns = [concern for concern, count in concerns_count.items() 
                          if count >= threshold]
        
        return common_concerns
    
    def _calculate_consensus_level(self, discussions: Dict[str, str], step_data: Dict[str, Any]) -> float:
        """计算共识程度"""
        try:
            # 基于关键词一致性和情感分析
            positive_keywords = ['支持', '赞成', '同意', '好', '优秀', '满意']
            negative_keywords = ['反对', '不同意', '问题', '不满', '抗议', '糟糕']
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for discussion in discussions.values():
                pos_score = sum(1 for keyword in positive_keywords if keyword in discussion)
                neg_score = sum(1 for keyword in negative_keywords if keyword in discussion)
                
                if pos_score > neg_score:
                    positive_count += 1
                elif neg_score > pos_score:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            total = len(discussions)
            if total == 0:
                return 0.5
            
            # 计算共识度 (0-1)
            consensus = (positive_count + neutral_count * 0.5) / total
            
            # 考虑系统整体状态
            system_performance = step_data['metrics'].get('overall_performance', 0.5)
            adjusted_consensus = (consensus + system_performance) / 2
            
            return min(1.0, max(0.0, adjusted_consensus))
            
        except Exception as e:
            logger.warning(f"⚠️ 计算共识度失败: {e}")
            return 0.5
    
    def _generate_consensus_rules(self, parliament_result: Dict[str, Any], step_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于共识生成新规则"""
        new_rules = []
        
        try:
            consensus_level = parliament_result.get('consensus_level', 0.5)
            common_concerns = parliament_result.get('common_concerns', [])
            
            # 只有在达成较高共识时才生成新规则
            if consensus_level < 0.7:
                logger.info(f"📊 共识程度较低({consensus_level:.2f})，不生成新规则")
                return new_rules
            
            # 基于共同关注点生成规则
            current_performance = step_data['metrics'].get('overall_performance', 0.5)
            
            for concern in common_concerns:
                rule = self._create_rule_for_concern(concern, current_performance, consensus_level)
                if rule:
                    new_rules.append(rule)
            
            # 添加到神圣法典管理器（如果可能）
            if new_rules and self.holy_code_manager:
                try:
                    for rule in new_rules:
                        # 尝试添加到规则库
                        if hasattr(self.holy_code_manager, 'rule_engine'):
                            # 这里可以添加具体的规则添加逻辑
                            logger.info(f"📜 尝试添加新规则: {rule['name']}")
                except Exception as e:
                    logger.warning(f"⚠️ 添加新规则失败: {e}")
            
            return new_rules
            
        except Exception as e:
            logger.error(f"❌ 生成共识规则失败: {e}")
            return []
    
    def _create_rule_for_concern(self, concern: str, performance: float, consensus: float) -> Dict[str, Any]:
        """为特定关注点创建规则"""
        rule_templates = {
            '医疗质量': {
                'name': f'医疗质量提升规则_{self.current_step}',
                'description': '基于议会共识制定的医疗质量改进措施',
                'type': 'quality_improvement',
                'target_metric': 'care_quality_index',
                'improvement_target': min(0.95, performance + 0.1),
                'consensus_level': consensus
            },
            '财务管理': {
                'name': f'财务优化规则_{self.current_step}',
                'description': '基于议会共识的成本控制和资源优化措施',
                'type': 'financial_optimization',
                'target_metric': 'financial_indicator',
                'improvement_target': min(0.9, performance + 0.08),
                'consensus_level': consensus
            },
            '患者服务': {
                'name': f'患者服务提升规则_{self.current_step}',
                'description': '基于议会共识的患者体验改善措施',
                'type': 'patient_service',
                'target_metric': 'patient_satisfaction',
                'improvement_target': min(0.95, performance + 0.12),
                'consensus_level': consensus
            },
            '人员管理': {
                'name': f'人力资源优化规则_{self.current_step}',
                'description': '基于议会共识的人员管理和培训改善',
                'type': 'hr_management',
                'target_metric': 'staff_workload_balance',
                'improvement_target': min(0.9, performance + 0.1),
                'consensus_level': consensus
            },
            '安全管理': {
                'name': f'安全管理强化规则_{self.current_step}',
                'description': '基于议会共识的安全风险防控措施',
                'type': 'safety_management',
                'target_metric': 'safety_incident_rate',
                'improvement_target': max(0.05, performance - 0.1),  # 事故率越低越好
                'consensus_level': consensus
            }
        }
        
        if concern in rule_templates:
            rule = rule_templates[concern].copy()
            rule['created_at'] = self.current_step
            rule['created_by'] = 'parliament_consensus'
            return rule
        
        return None
    
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
                    'version': 'refactored',
                    'parliament_waiting': self.parliament_waiting
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
                'ppo_status': {
                    'buffer_size': len(self.experience_buffer)
                },
                'config': {
                    'max_steps': self.config.max_steps,
                    'enable_llm': self.config.enable_llm_integration,
                    'enable_reward_control': self.config.enable_reward_control,
                    'llm_provider': self.config.llm_provider,
                    'ppo_training_episodes': getattr(self.config, 'ppo_training_episodes', 100)
                }
            }
        except Exception as e:
            logger.error(f"❌ 生成仿真报告失败: {e}")
            return {
                'error': str(e),
                'current_step': self.current_step,
                'component_status': self._get_component_status()
            }