"""
多智能体系统模块 - Kallipolis医疗共和国版本
基于控制理论和哲学理念的虚拟医院理事会治理系统
"""

from .role_agents import (RoleAgent, RoleManager, 
                         DoctorAgent, InternAgent,
                         PatientAgent, AccountantAgent, GovernmentAgent)
# behavior_models removed - using MADDPG + LLM architecture instead
from .learning_models import (LearningModel, MADDPGModel, DQNModel,
                             BaseNetwork, Actor, Critic)
from .llm_action_generator import (
    LLMConfig, BaseLLMProvider, MockLLMProvider, LLMActionGenerator
)
from .interaction_engine import (ExperienceReplay,
                                KallipolisInteractionEngine, CrisisScenario)
from .multi_agent_coordinator import (MultiAgentInteractionEngine, InteractionConfig)
from .agent_registry import (
    AgentRegistry, AgentRegistryConfig, LLMProviderType,
    create_agent_registry, get_global_agent_registry, reset_global_registry
)

__all__ = [
    # 角色智能体
    'RoleAgent', 'RoleManager', 
    'DoctorAgent', 'InternAgent', 'PatientAgent', 'AccountantAgent', 'GovernmentAgent',
    
    # Behavior models system (removed - using MADDPG + LLM instead)
    # 'BehaviorType', 'BehaviorParameters', 'BehaviorState', 'BaseBehaviorModel',
    # 'RationalBehaviorModel', 'BoundedRationalBehaviorModel', 'EmotionalBehaviorModel',
    # 'SocialBehaviorModel', 'AdaptiveBehaviorModel', 'BehaviorModelFactory', 'BehaviorModelManager',
    
    # 学习模型
    'LearningModel', 'MADDPGModel', 'DQNModel', 'BaseNetwork',
    'Actor', 'Critic',
    
    # LLM集成
    'LLMConfig', 'BaseLLMProvider', 'MockLLMProvider', 'LLMActionGenerator',
    
    # 交互引擎
    'ExperienceReplay', 'KallipolisInteractionEngine',
    'CrisisScenario', 'MultiAgentInteractionEngine', 'InteractionConfig',
    
    # 智能体注册中心
    'AgentRegistry', 'AgentRegistryConfig', 'LLMProviderType',
    'create_agent_registry', 'get_global_agent_registry', 'reset_global_registry',
]