"""
多智能体系统模块 - Kallipolis医疗共和国版本
基于控制理论和哲学理念的虚拟医院理事会治理系统
"""

# Agents模块
from .agents.role_agents import (RoleAgent, RoleManager, 
                                ParliamentMemberAgent, DoctorAgent, InternAgent,
                                PatientAgent, AccountantAgent, GovernmentAgent)
from .agents.behavior_models import (BaseBehaviorModel, RationalBehaviorModel,
                                   BoundedRationalBehaviorModel, EmotionalBehaviorModel,
                                   SocialBehaviorModel, AdaptiveBehaviorModel,
                                   BehaviorModelFactory, BehaviorModelManager)
from .agents.learning_models import (LearningModel, MADDPGModel, DQNModel,
                                   BaseNetwork, Actor, Critic)
from .agents.llm_action_generator import LLMActionGenerator
from .agents.interaction_engine import KallipolisInteractionEngine
from .agents.multi_agent_coordinator import MultiAgentInteractionEngine

# Holy Code模块
from .holy_code import (HolyCodeManager, RuleEngine, RuleLibrary, 
                       Parliament, ReferenceGenerator, VoteType, 
                       ParliamentConfig, ReferenceConfig, ReferenceType)

__all__ = [
    # 角色智能体
    'RoleAgent', 'RoleManager', 'ParliamentMemberAgent',
    'DoctorAgent', 'InternAgent', 'PatientAgent', 'AccountantAgent', 'GovernmentAgent',
    'ParliamentMemberAgent',
    
    # 行为模型
    'BaseBehaviorModel', 'RationalBehaviorModel', 'BoundedRationalBehaviorModel',
    'EmotionalBehaviorModel', 'SocialBehaviorModel', 'AdaptiveBehaviorModel',
    'BehaviorModelFactory', 'BehaviorModelManager',
    
    # 学习模型
    'LearningModel', 'MADDPGModel', 'DQNModel', 'BaseNetwork',
    'Actor', 'Critic',
    
    # LLM集成
    'LLMActionGenerator',
    
    # 交互引擎
    'KallipolisInteractionEngine', 'MultiAgentInteractionEngine',
    
    # Holy Code系统
    'HolyCodeManager', 'RuleEngine', 'RuleLibrary', 'Parliament', 
    'ReferenceGenerator', 'VoteType', 'ParliamentConfig', 'ReferenceConfig',
    'ReferenceType'
]