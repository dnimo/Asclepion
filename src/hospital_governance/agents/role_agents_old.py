import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """智能体配置 - 基于数理推导的严格参数化"""
    role: str
    action_dim: int
    observation_dim: int
    learning_rate: float = 0.001  # η in θ_i(t+1) = θ_i(t) + η ∇J_i(θ)
    hidden_dims: List[int] = None
    # 收益函数权重 R_i(x, a_i, a_{-i}) = α_i U(x) + β_i V_i(x, a_i) - γ_i D_i(x, x*)
    alpha: float = 0.3  # 全局资源效用权重
    beta: float = 0.5   # 局部价值权重
    gamma: float = 0.2  # 理想状态偏差权重
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]

@dataclass
class AgentState:
    """智能体状态"""
    position: np.ndarray
    velocity: np.ndarray
    resources: Dict[str, float]
    beliefs: Dict[str, Any]
    goals: List[str]

class RoleAgent(ABC):
    """角色智能体基类 - 基于数理推导的严格实现
    
    实现参数化随机策略: π_i(a_i | o_i; θ_i) = exp(φ_i(o_i, a_i)^T θ_i) / Σ exp(...)
    和策略梯度更新: θ_i(t+1) = θ_i(t) + η ∇_θ_i J_i(θ)
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.role = config.role
        self.state_dim = config.observation_dim
        self.action_dim = config.action_dim
        
        # 策略参数 θ_i ∈ ℝ^d
        self.theta = np.random.normal(0, 0.1, self.action_dim)
        
        # 收益函数权重系数
        self.alpha = config.alpha  # 全局效用权重
        self.beta = config.beta    # 局部价值权重  
        self.gamma = config.gamma  # 理想状态偏差权重
        
        # 智能体状态
        self.state = AgentState(
            position=np.zeros(2),
            velocity=np.zeros(2), 
            resources={},
            beliefs={},
            goals=[],
            policy_params=self.theta.copy()
        )
        
        # 行为模型和学习组件
        self.behavior_model = None
        self.learning_model = None
        self.llm_generator = None
        
        # 历史记录
        self.action_history = []
        self.state_history = []
        self.reward_history = []
        self.q_value_history = []  # Q值历史
        
        # 特征映射缓存
        self._feature_cache = {}
        
        logger.debug(f"Initialized {self.role} agent with θ shape: {self.theta.shape}")
    def set_behavior_model(self, behavior_model):
        self.behavior_model = behavior_model
    def set_learning_model(self, learning_model):
        self.learning_model = learning_model
    def set_llm_generator(self, llm_generator):
        self.llm_generator = llm_generator
    @abstractmethod
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        pass
    @abstractmethod
    def select_action(self, observation: np.ndarray, 
                     holy_code_guidance: Optional[Dict[str, Any]] = None,
                     training: bool = False) -> np.ndarray:
        """选择动作 - 基类方法"""
        # 如果有Holy Code指导，使用专门的方法
        if holy_code_guidance:
            return self.select_action_with_holycode(observation, holy_code_guidance, training)
        
        # 默认实现
        return np.random.normal(0, 0.1, self.action_dim)
    def select_action_with_holycode(self, observation: np.ndarray,
                                    holycode_guidance: Optional[Dict[str, Any]] = None,
                                    training: bool = False) -> np.ndarray:
        """议会成员智能体行动选择，集成HolyCode指导"""
        action = np.zeros(self.action_dim)
        if holycode_guidance:
            boost = holycode_guidance.get('priority_boost', 1.0)
            recommendations = holycode_guidance.get('rule_recommendations', [])
            if boost > 1.0:
                action += boost * 0.1
            for rec in recommendations:
                if '投票支持' in rec:
                    action[0] = max(action[0], 0.8)
                if '提案调整' in rec:
                    action[1] = max(action[1], 0.7)
        action = np.clip(action + np.random.normal(0, 0.1, self.action_dim), -1, 1)
        return action
    def update_state(self, new_state: Dict[str, Any]):
        for key, value in new_state.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def update_policy(self, observation: np.ndarray, action: np.ndarray, reward: float):
        """更新策略参数 - 策略梯度上升（简化版本）"""
        # 如果有学习模型，使用学习模型更新
        if self.learning_model:
            self.learning_model.update(observation, action, reward)
        else:
            # 简化的策略梯度更新
            if hasattr(self, 'theta'):
                learning_rate = getattr(self.config, 'learning_rate', 0.01)
                gradient = 0.01 * reward * np.random.normal(0, 0.1, len(self.theta))
                self.theta += learning_rate * gradient
            
            # 更新性能分数
            if hasattr(self, 'performance_score'):
                self.performance_score = 0.9 * self.performance_score + 0.1 * reward
    
    def add_experience(self, state: np.ndarray, action: np.ndarray,
                      reward: float, next_state: np.ndarray, done: bool):
        experience = {
            'role': self.role,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.action_history.append(action)
        self.state_history.append(state)
        self.reward_history.append(reward)
        return experience
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """智能体配置"""
    role: str
    action_dim: int
    observation_dim: int
    learning_rate: float = 0.001
    hidden_dims: List[int] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]

@dataclass
class AgentState:
    """智能体状态"""
    position: np.ndarray
    velocity: np.ndarray
    resources: Dict[str, float]
    beliefs: Dict[str, Any]
    goals: List[str]

class RoleAgent(ABC):
    """角色智能体基类"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.role = config.role
        self.state_dim = config.observation_dim
        self.action_dim = config.action_dim
        
        # 智能体状态
        self.state = AgentState(
            position=np.zeros(2),
            velocity=np.zeros(2),
            resources={},
            beliefs={},
            goals=[]
        )
        
        # 行为模型和学习模型
        self.behavior_model = None
        self.learning_model = None
        self.llm_generator = None
        
        # 历史记录
        self.action_history = []
        self.state_history = []
        self.reward_history = []
    
    def set_behavior_model(self, behavior_model):
        """设置行为模型"""
        self.behavior_model = behavior_model
    
    def set_learning_model(self, learning_model):
        """设置学习模型"""
        self.learning_model = learning_model
    
    def set_llm_generator(self, llm_generator):
        """设置LLM动作生成器"""
        self.llm_generator = llm_generator
    
    @abstractmethod
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境"""
        pass
    
    @abstractmethod
    def select_action(self, observation: np.ndarray, 
                     holy_code_guidance: Optional[Dict[str, Any]] = None,
                     training: bool = False) -> np.ndarray:
        """选择行动"""
        pass
    
    def update_state(self, new_state: Dict[str, Any]):
        """更新状态"""
        for key, value in new_state.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def add_experience(self, state: np.ndarray, action: np.ndarray, 
                      reward: float, next_state: np.ndarray, done: bool):
        """添加经验到历史"""
        experience = {
            'role': self.role,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.action_history.append(action)
        self.state_history.append(state)
        self.reward_history.append(reward)
        
        return experience
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        if not self.reward_history:
            return {}
            
        recent_rewards = self.reward_history[-100:]  # 最近100步
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'total_actions': len(self.action_history)
        }

class DoctorAgent(RoleAgent):
    """医生智能体"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注医疗质量和资源"""
        observation = np.zeros(self.state_dim)
        
        # 医疗质量相关指标
        if 'medical_quality' in environment:
            observation[0] = environment['medical_quality']
        if 'patient_safety' in environment:
            observation[1] = environment['patient_safety']
        if 'resource_adequacy' in environment:
            observation[2] = environment['resource_adequacy']
        if 'staff_satisfaction' in environment:
            observation[3] = environment['staff_satisfaction']
        
        # 系统效率指标
        if 'operational_efficiency' in environment:
            observation[4] = environment['operational_efficiency']
        if 'waiting_times' in environment:
            observation[5] = 1.0 - environment['waiting_times']  # 反转，越高越好
        
        # 危机状态
        if 'crisis_severity' in environment:
            observation[6] = environment['crisis_severity']
        
        # 伦理合规性
        if 'ethics_compliance' in environment:
            observation[7] = environment['ethics_compliance']
        
        return observation
    
    def select_action(self, observation: np.ndarray, 
                     holy_code_guidance: Optional[Dict[str, Any]] = None,
                     training: bool = False) -> np.ndarray:
        """选择医疗相关行动，集成HolyCode指导"""
        if self.learning_model and training:
            action = self.learning_model.get_actions(
                {self.role: observation}, training
            )[self.role]
        elif self.llm_generator:
            action = self.llm_generator.generate_medical_action(
                observation, self.state
            )
        else:
            action = np.zeros(self.action_dim)
            
        # 集成Holy Code指导
        if holy_code_guidance:
            priority_boost = holy_code_guidance.get('priority_boost', 1.0)
            action *= priority_boost
            
            # 应用规则建议
            recommendations = holy_code_guidance.get('rule_recommendations', [])
            for rec in recommendations:
                if '医疗' in rec or '患者' in rec or '治疗' in rec:
                    action[0] *= 1.2  # 增强医疗决策权重
                    
        return action

class InternAgent(RoleAgent):
    """实习医生智能体"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注教育和资源"""
        observation = np.zeros(self.state_dim)
        
        # 教育相关指标
        if 'education_quality' in environment:
            observation[0] = environment['education_quality']
        if 'training_hours' in environment:
            observation[1] = environment['training_hours'] / 40.0  # 归一化
        if 'mentorship_availability' in environment:
            observation[2] = environment['mentorship_availability']
        
        # 工作条件
        if 'workload' in environment:
            observation[3] = 1.0 - environment['workload']  # 反转
        if 'resource_access' in environment:
            observation[4] = environment['resource_access']
        
        # 职业发展
        if 'career_development' in environment:
            observation[5] = environment['career_development']
        if 'salary_satisfaction' in environment:
            observation[6] = environment['salary_satisfaction']
        
        return observation
    
    def select_action(self, observation: np.ndarray,
                     holy_code_guidance: Optional[Dict[str, Any]] = None,
                     training: bool = False) -> np.ndarray:
        """选择教育和发展相关行动，集成HolyCode指导"""
        if self.learning_model and training:
            action = self.learning_model.get_actions(
                {self.role: observation}, training
            )[self.role]
        elif self.llm_generator:
            action = self.llm_generator.generate_education_action(
                observation, self.state
            )
        else:
            action = np.zeros(self.action_dim)
            
        # 集成Holy Code指导
        if holy_code_guidance:
            boost = holy_code_guidance.get('priority_boost', 1.0)
            recommendations = holy_code_guidance.get('rule_recommendations', [])
            if boost > 1.0:
                action += boost * 0.1
            for rec in recommendations:
                if '培训请求' in rec:
                    action[0] = max(action[0], 0.8)
                if '工作负荷调整' in rec:
                    action[1] = max(action[1], 0.6)
                    
        return action

class PatientAgent(RoleAgent):
    """患者智能体"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注服务质量"""
        observation = np.zeros(self.state_dim)
        
        # 服务质量指标
        if 'patient_satisfaction' in environment:
            observation[0] = environment['patient_satisfaction']
        if 'care_quality' in environment:
            observation[1] = environment['care_quality']
        if 'accessibility' in environment:
            observation[2] = environment['accessibility']
        if 'waiting_times' in environment:
            observation[3] = 1.0 - environment['waiting_times']  # 反转
        
        return observation
    
    def select_action(self, observation: np.ndarray,
                     holy_code_guidance: Optional[Dict[str, Any]] = None,
                     training: bool = False) -> np.ndarray:
        """选择患者权益相关行动，集成HolyCode指导"""
        if self.learning_model and training:
            action = self.learning_model.get_actions(
                {self.role: observation}, training
            )[self.role]
        elif self.llm_generator:
            action = self.llm_generator.generate_patient_action(
                observation, self.state
            )
        else:
            action = np.zeros(self.action_dim)
            
            # Holy Code指导
            if holy_code_guidance:
                boost = holy_code_guidance.get('priority_boost', 1.0)
                recommendations = holy_code_guidance.get('rule_recommendations', [])
                if boost > 1.0:
                    action += boost * 0.1
                for rec in recommendations:
                    if '满意度改进' in rec:
                        action[0] = max(action[0], 0.8)
                    if '可及性改善' in rec:
                        action[1] = max(action[1], 0.7)
                    if '等待时间优化' in rec:
                        action[2] = max(action[2], 0.6)
                        
            # 基于观察的自动调整
            if observation[0] < 0.7:
                action[0] = max(action[0], 0.8)
            if observation[2] < 0.6:
                action[1] = max(action[1], 0.7)
            if observation[3] < 0.5:
                action[2] = max(action[2], 0.6)
                
        return np.clip(action + np.random.normal(0, 0.1, self.action_dim), -1, 1)

class AccountantAgent(RoleAgent):
    """会计智能体"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注财务状态"""
        observation = np.zeros(self.state_dim)
        
        # 财务指标
        if 'financial_health' in environment:
            observation[0] = environment['financial_health']
        if 'resource_utilization' in environment:
            observation[1] = environment['resource_utilization']
        if 'cost_efficiency' in environment:
            observation[2] = environment['cost_efficiency']
        if 'revenue_growth' in environment:
            observation[3] = environment['revenue_growth']
        
        return observation
    
    def select_action(self, observation: np.ndarray,
                     holy_code_guidance: Optional[Dict[str, Any]] = None,
                     training: bool = False) -> np.ndarray:
        """选择财务相关行动，集成HolyCode指导"""
        if self.learning_model and training:
            action = self.learning_model.get_actions(
                {self.role: observation}, training
            )[self.role]
        elif self.llm_generator:
            action = self.llm_generator.generate_financial_action(
                observation, self.state
            )
        else:
            action = np.zeros(self.action_dim)
            
            # Holy Code指导
            if holy_code_guidance:
                boost = holy_code_guidance.get('priority_boost', 1.0)
                recommendations = holy_code_guidance.get('rule_recommendations', [])
                if boost > 1.0:
                    action += boost * 0.1
                for rec in recommendations:
                    if '成本控制' in rec:
                        action[0] = max(action[0], 0.8)
                    if '资源优化' in rec:
                        action[1] = max(action[1], 0.6)
                    if '效率提升' in rec:
                        action[2] = max(action[2], 0.7)
                        
            # 基于观察的自动调整
            if observation[0] < 0.6:
                action[0] = max(action[0], 0.8)
            if observation[1] < 0.7:
                action[1] = max(action[1], 0.6)
            if observation[2] < 0.7:
                action[2] = max(action[2], 0.7)
                
        return np.clip(action + np.random.normal(0, 0.1, self.action_dim), -1, 1)

class GovernmentAgent(RoleAgent):
    """政府智能体"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注系统和伦理状态"""
        observation = np.zeros(self.state_dim)
        
        # 系统整体状态
        if 'system_stability' in environment:
            observation[0] = environment['system_stability']
        if 'overall_performance' in environment:
            observation[1] = environment['overall_performance']
        
        # 伦理合规性
        if 'ethics_compliance' in environment:
            observation[2] = environment['ethics_compliance']
        if 'regulatory_compliance' in environment:
            observation[3] = environment['regulatory_compliance']
        
        # 公共利益
        if 'public_trust' in environment:
            observation[4] = environment['public_trust']
        
        return observation
    
    def select_action(self, observation: np.ndarray,
                     holy_code_guidance: Optional[Dict[str, Any]] = None,
                     training: bool = False) -> np.ndarray:
        """选择政策和监管相关行动，集成HolyCode指导"""
        if self.learning_model and training:
            action = self.learning_model.get_actions(
                {self.role: observation}, training
            )[self.role]
        elif self.llm_generator:
            action = self.llm_generator.generate_government_action(
                observation, self.state
            )
        else:
            action = np.zeros(self.action_dim)
            
            # Holy Code指导
            if holy_code_guidance:
                boost = holy_code_guidance.get('priority_boost', 1.0)
                recommendations = holy_code_guidance.get('rule_recommendations', [])
                if boost > 1.0:
                    action += boost * 0.1
                for rec in recommendations:
                    if '系统稳定措施' in rec:
                        action[0] = max(action[0], 0.8)
                    if '监管加强' in rec:
                        action[1] = max(action[1], 0.7)
                    if '透明度提升' in rec:
                        action[2] = max(action[2], 0.6)
                        
            # 基于观察的自动调整
            if observation[0] < 0.7:
                action[0] = max(action[0], 0.8)
            if observation[2] < 0.7:
                action[1] = max(action[1], 0.7)
            if observation[4] < 0.6:
                action[2] = max(action[2], 0.6)
                
        return np.clip(action + np.random.normal(0, 0.1, self.action_dim), -1, 1)

class ParliamentMemberAgent(RoleAgent):
    """议会成员智能体基类（扩展原有功能）"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.voting_strategy = "utility_based"
        self.proposal_history: List[Dict] = []
        self.voting_history: List[Dict] = []
    
    def set_parliament(self, parliament):
        """设置议会引用"""
        self.parliament = parliament
    
    def formulate_proposal(self, observation: np.ndarray, 
                          context: str) -> Optional[Dict[str, Any]]:
        """制定提案"""
        # 基于角色特性制定提案
        if self.role == 'doctors':
            return self._formulate_medical_proposal(observation, context)
        elif self.role == 'interns':
            return self._formulate_education_proposal(observation, context)
        elif self.role == 'accountants':
            return self._formulate_financial_proposal(observation, context)
        elif self.role == 'patients':
            return self._formulate_patient_proposal(observation, context)
        elif self.role == 'government':
            return self._formulate_government_proposal(observation, context)
        return None
    
    def vote_on_proposal(self, proposal: Dict[str, Any], 
                        proposal_id: str) -> Tuple[bool, str]:
        """对提案进行投票"""
        utility = self._calculate_proposal_utility(proposal)
        
        # 使用神圣法典评估合规性
        holy_code_compliance = 1.0
        if hasattr(self, 'parliament') and self.parliament:
            holy_code_compliance = self.parliament.holy_code.evaluate_decision(
                proposal, proposal.get('context', 'general')
            )
        
        # 综合效用和合规性进行投票决策
        vote_score = 0.6 * utility + 0.4 * holy_code_compliance
        vote = vote_score > 0.5
        rationale = f"Utility: {utility:.2f}, Compliance: {holy_code_compliance:.2f}"
        
        # 记录投票历史
        self.voting_history.append({
            'proposal_id': proposal_id,
            'vote': vote,
            'rationale': rationale,
            'timestamp': np.datetime64('now')
        })
        
        return vote, rationale
    
    def _calculate_proposal_utility(self, proposal: Dict[str, Any]) -> float:
        """计算提案对当前角色的效用"""
        if self.role == 'doctors':
            return self._calculate_doctor_utility(proposal)
        elif self.role == 'interns':
            return self._calculate_intern_utility(proposal)
        elif self.role == 'accountants':
            return self._calculate_accountant_utility(proposal)
        elif self.role == 'patients':
            return self._calculate_patient_utility(proposal)
        elif self.role == 'government':
            return self._calculate_government_utility(proposal)
        return 0.5
    
    def _calculate_doctor_utility(self, proposal: Dict[str, Any]) -> float:
        """医生效用计算"""
        medical_quality = proposal.get('medical_quality_index', 0.5)
        resource_adequacy = proposal.get('resource_adequacy', 0.5)
        return 0.6 * medical_quality + 0.4 * resource_adequacy
    
    def _calculate_intern_utility(self, proposal: Dict[str, Any]) -> float:
        """实习医生效用计算"""
        education_opportunity = proposal.get('education_budget_ratio', 0.1) / 0.15
        working_conditions = proposal.get('working_conditions_index', 0.5)
        return 0.7 * education_opportunity + 0.3 * working_conditions
    
    def _calculate_accountant_utility(self, proposal: Dict[str, Any]) -> float:
        """会计效用计算"""
        financial_health = proposal.get('financial_health_index', 0.5)
        efficiency = proposal.get('operational_efficiency', 0.5)
        return 0.8 * financial_health + 0.2 * efficiency
    
    def _calculate_patient_utility(self, proposal: Dict[str, Any]) -> float:
        """患者代表效用计算"""
        accessibility = proposal.get('accessibility_index', 0.5)
        safety = proposal.get('patient_safety_index', 0.5)
        satisfaction = proposal.get('patient_satisfaction', 0.5)
        return 0.4 * accessibility + 0.4 * safety + 0.2 * satisfaction
    
    def _calculate_government_utility(self, proposal: Dict[str, Any]) -> float:
        """政府效用计算"""
        system_stability = proposal.get('system_stability', 0.5)
        ethics_compliance = proposal.get('ethics_compliance', 0.5)
        public_trust = proposal.get('public_trust', 0.5)
        return 0.4 * system_stability + 0.3 * ethics_compliance + 0.3 * public_trust
    
    def _formulate_medical_proposal(self, observation: np.ndarray, context: str) -> Dict[str, Any]:
        """制定医疗相关提案"""
        return {
            'type': 'medical_improvement',
            'context': context,
            'medical_quality_index': 0.8,
            'resource_adequacy': 0.7,
            'proposed_budget_change': 0.1,
            'rationale': 'Improve medical equipment and training'
        }
    
    def _formulate_education_proposal(self, observation: np.ndarray, context: str) -> Dict[str, Any]:
        """制定教育相关提案"""
        return {
            'type': 'education_enhancement',
            'context': context,
            'education_budget_ratio': 0.12,
            'working_conditions_index': 0.75,
            'training_hours': 40,
            'rationale': 'Enhance intern training program'
        }
    
    def _formulate_financial_proposal(self, observation: np.ndarray, context: str) -> Dict[str, Any]:
        """制定财务相关提案"""
        return {
            'type': 'financial_optimization',
            'context': context,
            'financial_health_index': 0.85,
            'operational_efficiency': 0.8,
            'cost_reduction_target': 0.15,
            'rationale': 'Optimize resource allocation and reduce waste'
        }
    
    def _formulate_patient_proposal(self, observation: np.ndarray, context: str) -> Dict[str, Any]:
        """制定患者相关提案"""
        return {
            'type': 'patient_centered_improvement',
            'context': context,
            'accessibility_index': 0.9,
            'patient_safety_index': 0.95,
            'patient_satisfaction': 0.88,
            'rationale': 'Improve patient experience and safety measures'
        }
    
    def _formulate_government_proposal(self, observation: np.ndarray, context: str) -> Dict[str, Any]:
        """制定政府相关提案"""
        return {
            'type': 'regulatory_framework',
            'context': context,
            'system_stability': 0.9,
            'ethics_compliance': 0.95,
            'public_trust': 0.85,
            'rationale': 'Strengthen regulatory framework and ethical guidelines'
        }

class RoleManager:
    """角色管理器"""
    
    def __init__(self):
        self.agents: Dict[str, RoleAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
    
    def register_agent(self, agent: RoleAgent):
        """注册智能体"""
        self.agents[agent.role] = agent
        print(f"Registered agent: {agent.role}")
    
    def register_agent_config(self, role: str, config: AgentConfig):
        """注册智能体配置"""
        self.agent_configs[role] = config
    
    def create_agent(self, role: str, agent_class, config: AgentConfig = None) -> RoleAgent:
        """创建智能体"""
        if config is None:
            config = self.agent_configs.get(role)
            if config is None:
                raise ValueError(f"No config found for role: {role}")
        
        agent = agent_class(config)
        self.register_agent(agent)
        return agent
    
    def create_all_agents(self, configs: Dict[str, AgentConfig]):
        """创建所有智能体"""
        agent_classes = {
            'doctors': DoctorAgent,
            'interns': InternAgent,
            'patients': PatientAgent,
            'accountants': AccountantAgent,
            'government': GovernmentAgent
        }
        
        for role, config in configs.items():
            if role in agent_classes:
                self.create_agent(role, agent_classes[role], config)
            else:
                print(f"Warning: Unknown role {role}")
    
    def get_agent(self, role: str) -> Optional[RoleAgent]:
        """获取智能体"""
        return self.agents.get(role)
    
    def get_all_agents(self) -> List[RoleAgent]:
        """获取所有智能体"""
        return list(self.agents.values())
    
    def remove_agent(self, role: str):
        """移除智能体"""
        if role in self.agents:
            del self.agents[role]
            print(f"Removed agent: {role}")

class LyapunovStabilityController:
    """李雅普诺夫稳定性控制器"""
    
    def __init__(self, system_dim: int):
        self.system_dim = system_dim
        self.P = np.eye(system_dim)  # 李雅普诺夫矩阵
        self.stability_threshold = 0.1
    
    def check_stability(self, current_state: np.ndarray, 
                       desired_state: np.ndarray) -> Tuple[bool, float]:
        """检查系统稳定性"""
        error = current_state - desired_state
        V = error.T @ self.P @ error  # 李雅普诺夫函数
        
        # 简化的稳定性检查
        is_stable = V < self.stability_threshold
        stability_margin = self.stability_threshold - V
        
        return is_stable, stability_margin
    
    def update_lyapunov_matrix(self, system_dynamics: np.ndarray):
        """更新李雅普诺夫矩阵"""
        # 基于系统动力学更新P矩阵
        A = system_dynamics
        try:
            # 解李雅普诺夫方程: A^T P + P A = -I
            # 简化实现
            self.P = np.linalg.inv(A.T + A) @ (-np.eye(self.system_dim))
        except:
            # 如果无法求解，保持原矩阵
            pass