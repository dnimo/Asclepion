"""
🏥 Kallipolis医疗共和国治理系统 (Asclepion)

基于多智能体博弈论的医院治理实时监控仿真平台，融合了深度强化学习（MADDPG）、
大语言模型（LLM）、分布式控制系统和数学策略模板等前沿技术。

多智能体系统模块 - Kallipolis医疗共和国版本
基于控制理论和哲学理念的虚拟医院理事会治理系统
"""

__version__ = "1.0.0"
__author__ = "dnimo"
__email__ = "dnimo@example.com"
__license__ = "MIT"
__description__ = "Kallipolis医疗共和国治理系统 - 基于多智能体博弈论的医院治理仿真平台"

# 版本信息
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "pre_release": None,
    "build": None
}

# 系统架构信息
SYSTEM_INFO = {
    "name": "Kallipolis Medical Republic Governance System",
    "codename": "Asclepion",
    "architecture": "Multi-Agent + WebSocket + Real-time Monitoring",
    "ai_stack": "MADDPG + LLM + Distributed Control + Mathematical Strategy",
    "dimensions": 16,
    "agents": 5,
    "languages": ["Python", "JavaScript", "HTML", "CSS"],
    "frameworks": ["asyncio", "websockets", "Chart.js", "PyTorch"]
}

# Agents模块
from .agents.role_agents import (RoleAgent, RoleManager, 
                                DoctorAgent, InternAgent,
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

def get_version():
    """获取版本字符串"""
    return __version__

def get_system_info():
    """获取系统信息"""
    return SYSTEM_INFO

def print_banner():
    """打印系统横幅"""
    banner = f"""
🏥 {SYSTEM_INFO['name']} ({SYSTEM_INFO['codename']})
{'='*80}
📊 架构: {SYSTEM_INFO['architecture']}
🧠 AI技术栈: {SYSTEM_INFO['ai_stack']}
📈 状态维度: {SYSTEM_INFO['dimensions']}维
🤖 智能体数量: {SYSTEM_INFO['agents']}个
⚡ 版本: {__version__}
🏛️ 让医疗治理更智能、更民主、更高效！
{'='*80}
"""
    print(banner)

__all__ = [
    # 版本和信息
    '__version__', 'get_version', 'get_system_info', 'print_banner',
    'VERSION_INFO', 'SYSTEM_INFO',
    
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