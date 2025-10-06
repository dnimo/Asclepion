"""
多智能体系统模块 - Kallipolis医疗共和国版本
基于控制理论和哲学理念的虚拟医院理事会治理系统
"""

from .role_agents import (RoleAgent, RoleManager, 
                         ParliamentMemberAgent, DoctorAgent, InternAgent,
                         PatientAgent, AccountantAgent, GovernmentAgent)
from .behavior_models import (
    # 新的行为模型系统
    BehaviorType, BehaviorParameters, BehaviorState, BaseBehaviorModel,
    RationalBehaviorModel, BoundedRationalBehaviorModel, EmotionalBehaviorModel,
    SocialBehaviorModel, AdaptiveBehaviorModel, BehaviorModelFactory, BehaviorModelManager,
    # 保持向后兼容的旧接口（如果存在）
    # BehaviorModel, RuleBasedBehavior, LearningBasedBehavior, CompositeBehavior,
    # avoidance_rule, attraction_rule
)
from .learning_models import (LearningModel, MADDPGModel, DQNModel,
                             BaseNetwork, Actor, Critic)
from .llm_action_generator import (
    LLMConfig, BaseLLMProvider, MockLLMProvider, LLMActionGenerator
)
from .interaction_engine import (ExperienceReplay,
                                KallipolisInteractionEngine, CrisisScenario)
from .multi_agent_coordinator import (MultiAgentInteractionEngine, InteractionConfig)

__all__ = [
    # 角色智能体
    'RoleAgent', 'RoleManager', 'ParliamentMemberAgent', 
    'DoctorAgent', 'InternAgent', 'PatientAgent', 'AccountantAgent', 'GovernmentAgent',
    
    # 行为模型系统
    'BehaviorType', 'BehaviorParameters', 'BehaviorState', 'BaseBehaviorModel',
    'RationalBehaviorModel', 'BoundedRationalBehaviorModel', 'EmotionalBehaviorModel',
    'SocialBehaviorModel', 'AdaptiveBehaviorModel', 'BehaviorModelFactory', 'BehaviorModelManager',
    
    # 学习模型
    'LearningModel', 'MADDPGModel', 'DQNModel', 'BaseNetwork',
    'Actor', 'Critic',
    
    # LLM集成
    'LLMConfig', 'BaseLLMProvider', 'MockLLMProvider', 'LLMActionGenerator',
    
    # 交互引擎
    'ExperienceReplay', 'KallipolisInteractionEngine',
    'CrisisScenario', 'MultiAgentInteractionEngine', 'InteractionConfig',
    
]