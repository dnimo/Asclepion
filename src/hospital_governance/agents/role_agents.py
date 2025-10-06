#!/usr/bin/env python3
"""
角色智能体重构版本 - 基于数理推导的严格实现
实现了完整的多智能体医院治理系统数学模型

基于以下数理推导：
1. 参数化随机策略: π_i(a_i | o_i; θ_i) = exp(φ_i(o_i, a_i)^T θ_i) / Σ exp(...)
2. 收益函数: R_i(x, a_i, a_{-i}) = α_i U(x) + β_i V_i(x, a_i) - γ_i D_i(x, x*)
3. 策略梯度更新: θ_i(t+1) = θ_i(t) + η ∇_{θ_i} J_i(θ)
4. 李雅普诺夫稳定性分析
5. 神圣法典动态生成理想状态
"""

import numpy as np
import scipy.linalg as la
from typing import Dict, List, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging

from .role_agents_old import ParliamentMemberAgent

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """智能体配置 - 基于数理推导的严格参数化"""
    role: str
    action_dim: int
    observation_dim: int
    learning_rate: float = 0.001  # η in θ_i(t+1) = θ_i(t) + η ∇J_i(θ)
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    
    # 收益函数权重 R_i(x, a_i, a_{-i}) = α_i U(x) + β_i V_i(x, a_i) - γ_i D_i(x, x*)
    alpha: float = 0.3  # 全局资源效用权重
    beta: float = 0.5   # 局部价值权重
    gamma: float = 0.2  # 理想状态偏差权重
    
    # 特征映射维度
    feature_dim: int = None
    
    def __post_init__(self):
        if self.feature_dim is None:
            self.feature_dim = self.observation_dim + self.action_dim

@dataclass
class AgentState:
    """智能体状态 - 基于16维状态空间的局部观测"""
    position: np.ndarray
    velocity: np.ndarray
    resources: Dict[str, float]
    beliefs: Dict[str, Any]
    goals: List[str]
    
    # 策略参数 θ_i
    policy_params: np.ndarray = None
    
    # 性能历史
    performance_score: float = 0.5
    cumulative_reward: float = 0.0
    
    # Q值估计缓存
    q_value_cache: Dict[str, float] = field(default_factory=dict)
    
    # 上次观测和动作
    last_observation: np.ndarray = None
    last_action: int = None

@dataclass 
class SystemState:
    """系统状态向量 x(t) ∈ ℝ^16 - 扩展的16维状态空间"""
    # 核心医疗指标 (x_1 到 x_4)
    medical_resource_utilization: float  # x₁: 医疗资源利用率
    patient_waiting_time: float         # x₂: 患者等待时间  
    financial_indicator: float          # x₃: 财务健康指标
    ethical_compliance: float           # x₄: 伦理合规度
    
    # 教育和培训指标 (x_5 到 x_8)
    education_training_quality: float   # x₅: 教育培训质量
    intern_satisfaction: float          # x₆: 实习生满意度
    professional_development: float     # x₇: 职业发展指数
    mentorship_effectiveness: float     # x₈: 指导效果
    
    # 患者服务指标 (x_9 到 x_12)
    patient_satisfaction: float         # x₉: 患者满意度
    service_accessibility: float        # x₁₀: 服务可及性
    care_quality_index: float          # x₁₁: 护理质量指数
    safety_incident_rate: float        # x₁₂: 安全事故率(反向)
    
    # 系统运营指标 (x_13 到 x_16)
    operational_efficiency: float       # x₁₃: 运营效率
    staff_workload_balance: float      # x₁₄: 员工工作负荷平衡
    crisis_response_capability: float   # x₁₅: 危机响应能力
    regulatory_compliance_score: float  # x₁₆: 监管合规分数
    
    def to_vector(self) -> np.ndarray:
        """转换为16维状态向量"""
        return np.array([
            self.medical_resource_utilization,
            self.patient_waiting_time,
            self.financial_indicator,
            self.ethical_compliance,
            self.education_training_quality,
            self.intern_satisfaction,
            self.professional_development,
            self.mentorship_effectiveness,
            self.patient_satisfaction,
            self.service_accessibility,
            self.care_quality_index,
            self.safety_incident_rate,
            self.operational_efficiency,
            self.staff_workload_balance,
            self.crisis_response_capability,
            self.regulatory_compliance_score
        ])
    
    @classmethod
    def from_vector(cls, x: np.ndarray) -> 'SystemState':
        """从16维向量构造系统状态"""
        return cls(
            medical_resource_utilization=x[0],
            patient_waiting_time=x[1],
            financial_indicator=x[2],
            ethical_compliance=x[3],
            education_training_quality=x[4],
            intern_satisfaction=x[5],
            professional_development=x[6],
            mentorship_effectiveness=x[7],
            patient_satisfaction=x[8],
            service_accessibility=x[9],
            care_quality_index=x[10],
            safety_incident_rate=x[11],
            operational_efficiency=x[12],
            staff_workload_balance=x[13],
            crisis_response_capability=x[14],
            regulatory_compliance_score=x[15]
        )

class RoleAgent(ABC):
    """角色智能体基类 - 基于数理推导的严格实现
    
    实现参数化随机策略: π_i(a_i | o_i; θ_i) = exp(φ_i(o_i, a_i)^T θ_i) / Σ exp(...)
    和策略梯度更新: θ_i(t+1) = θ_i(t) + η ∇_{θ_i} J_i(θ)
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.role = config.role
        self.state_dim = config.observation_dim
        self.action_dim = config.action_dim
        
        # 策略参数 θ_i ∈ ℝ^d
        self.theta = np.random.normal(0, 0.1, config.feature_dim)
        
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
        
        # 基线值估计(用于方差减少)
        self.baseline = 0.0
        self.baseline_lr = 0.1
        
        logger.debug(f"Initialized {self.role} agent with θ shape: {self.theta.shape}")
    
    def feature_mapping(self, observation: np.ndarray, action: int) -> np.ndarray:
        """特征映射 φ_i(o_i, a_i): O_i × A_i → ℝ^d"""
        cache_key = f"{hash(observation.tobytes())}_{action}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # 观测特征归一化
        obs_features = observation / (np.linalg.norm(observation) + 1e-8)
        
        # 动作独热编码
        action_features = np.zeros(self.action_dim)
        if 0 <= action < self.action_dim:
            action_features[action] = 1.0
        
        # 组合特征
        features = np.concatenate([obs_features, action_features])
        
        # 调整到配置的特征维度
        if len(features) > self.config.feature_dim:
            features = features[:self.config.feature_dim]
        elif len(features) < self.config.feature_dim:
            padding = np.zeros(self.config.feature_dim - len(features))
            features = np.concatenate([features, padding])
        
        self._feature_cache[cache_key] = features
        return features
    
    def compute_policy_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """计算策略概率分布 π_i(a_i | o_i; θ_i)"""
        logits = np.zeros(self.action_dim)
        
        for action in range(self.action_dim):
            phi = self.feature_mapping(observation, action)
            logits[action] = np.dot(phi, self.theta)
        
        # Softmax with numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)
        
        return probabilities
    
    def sample_action(self, observation: np.ndarray) -> int:
        """从策略分布中采样动作"""
        probabilities = self.compute_policy_probabilities(observation)
        action = np.random.choice(self.action_dim, p=probabilities)
        
        # 更新状态
        self.state.last_observation = observation.copy()
        self.state.last_action = action
        
        return action
    
    @abstractmethod
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，返回局部观测 o_i"""
        pass
    
    @abstractmethod
    def compute_local_value(self, system_state: SystemState, action: int) -> float:
        """计算局部价值函数 V_i(x, a_i) - 基于角色特异性"""
        pass
    
    def compute_reward(self, system_state: SystemState, action: int, 
                      global_utility: float, ideal_state: SystemState) -> float:
        """计算收益函数 R_i(x, a_i, a_{-i}) = α_i U(x) + β_i V_i(x, a_i) - γ_i D_i(x, x*)"""
        
        # 局部价值函数 V_i
        local_value = self.compute_local_value(system_state, action)
        
        # 到理想状态的偏差 D_i
        state_vec = system_state.to_vector()
        ideal_vec = ideal_state.to_vector()
        deviation = np.linalg.norm(state_vec - ideal_vec)
        
        # 组合收益
        reward = (self.alpha * global_utility + 
                 self.beta * local_value - 
                 self.gamma * deviation)
        
        return reward
    
    def update_policy(self, observation: np.ndarray, action: int, 
                     q_value: float, next_observation: Optional[np.ndarray] = None):
        """策略梯度更新 θ_i(t+1) = θ_i(t) + η ∇_{θ_i} J_i(θ)"""
        
        # 更新基线估计
        self.baseline = (1 - self.baseline_lr) * self.baseline + self.baseline_lr * q_value
        
        # 计算优势函数
        advantage = q_value - self.baseline
        
        # 计算策略梯度 ∇log π_i(a_i|o_i)
        probabilities = self.compute_policy_probabilities(observation)
        
        # 当前动作的特征
        phi_current = self.feature_mapping(observation, action)
        
        # 期望特征（所有动作的加权平均）
        phi_expected = np.zeros_like(phi_current)
        for a in range(self.action_dim):
            phi_a = self.feature_mapping(observation, a)
            phi_expected += probabilities[a] * phi_a
        
        # 策略梯度
        grad_log_pi = phi_current - phi_expected
        
        # 参数更新
        self.theta += self.config.learning_rate * advantage * grad_log_pi
        
        # 更新性能分数
        self.state.performance_score = 0.9 * self.state.performance_score + 0.1 * q_value
        self.state.cumulative_reward += q_value
        
        # 记录历史
        self.reward_history.append(q_value)
        self.q_value_history.append(q_value)
        
        logger.debug(f"{self.role} policy updated: advantage={advantage:.3f}, ||grad||={np.linalg.norm(grad_log_pi):.3f}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        if not self.reward_history:
            return {'performance_score': self.state.performance_score}
        
        recent_rewards = self.reward_history[-100:]  # 最近100步
        return {
            'performance_score': self.state.performance_score,
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'cumulative_reward': self.state.cumulative_reward,
            'policy_norm': np.linalg.norm(self.theta),
            'baseline_value': self.baseline,
            'total_actions': len(self.action_history)
        }
    
    # 继承类需要实现的抽象方法保持原有接口
    def select_action(self, observation: np.ndarray, 
                     holy_code_guidance: Optional[Dict[str, Any]] = None,
                     training: bool = False) -> np.ndarray:
        """选择动作 - 兼容原有接口"""
        discrete_action = self.sample_action(observation)
        
        # 转换为连续动作空间（如果需要）
        continuous_action = np.zeros(self.action_dim)
        continuous_action[discrete_action] = 1.0
        
        # 应用Holy Code指导
        if holy_code_guidance:
            priority_boost = holy_code_guidance.get('priority_boost', 1.0)
            continuous_action *= priority_boost
            
            # 应用规则建议
            recommendations = holy_code_guidance.get('rule_recommendations', [])
            continuous_action = self._apply_holy_code_recommendations(
                continuous_action, recommendations
            )
        
        return continuous_action
    
    @abstractmethod
    def _apply_holy_code_recommendations(self, action: np.ndarray, 
                                       recommendations: List[str]) -> np.ndarray:
        """应用神圣法典建议 - 各角色实现不同逻辑"""
        pass
    
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

class DoctorAgent(RoleAgent):
    """医生智能体 - 关注医疗质量和患者安全"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注医疗质量相关指标"""
        observation = np.zeros(self.state_dim)
        
        # 核心医疗指标 (索引 0-3)
        observation[0] = environment.get('medical_resource_utilization', 0.7)
        observation[1] = environment.get('patient_waiting_time', 0.3)
        observation[2] = environment.get('care_quality_index', 0.8)
        observation[3] = environment.get('safety_incident_rate', 0.1)
        
        # 患者服务指标 (索引 4-7)
        observation[4] = environment.get('patient_satisfaction', 0.8)
        observation[5] = environment.get('service_accessibility', 0.7)
        observation[6] = environment.get('ethical_compliance', 0.9)
        observation[7] = environment.get('crisis_response_capability', 0.8)
        
        return observation
    
    def compute_local_value(self, system_state: SystemState, action: int) -> float:
        """医生的局部价值函数 - 重视医疗质量和患者安全"""
        return (0.3 * system_state.medical_resource_utilization +
                0.25 * system_state.care_quality_index +
                0.25 * system_state.patient_satisfaction +
                0.2 * (1.0 - system_state.safety_incident_rate))  # 安全事故率越低越好
    
    def _apply_holy_code_recommendations(self, action: np.ndarray, 
                                       recommendations: List[str]) -> np.ndarray:
        """应用医生相关的神圣法典建议"""
        for rec in recommendations:
            if '医疗质量' in rec or '患者安全' in rec:
                action[0] = max(action[0], 0.8)  # 增强医疗决策权重
            elif '资源配置' in rec:
                action[1] = max(action[1], 0.7)
            elif '紧急响应' in rec:
                action[2] = max(action[2], 0.9)
        
        return np.clip(action, 0, 1)

class InternAgent(RoleAgent):
    """实习医生智能体 - 关注教育培训和职业发展"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注教育和发展指标"""
        observation = np.zeros(self.state_dim)
        
        # 教育培训指标 (索引 0-3)
        observation[0] = environment.get('education_training_quality', 0.7)
        observation[1] = environment.get('intern_satisfaction', 0.6)
        observation[2] = environment.get('professional_development', 0.5)
        observation[3] = environment.get('mentorship_effectiveness', 0.7)
        
        # 工作环境指标 (索引 4-7)
        observation[4] = environment.get('staff_workload_balance', 0.6)
        observation[5] = environment.get('medical_resource_utilization', 0.7)
        observation[6] = environment.get('operational_efficiency', 0.7)
        observation[7] = environment.get('ethical_compliance', 0.9)
        
        return observation
    
    def compute_local_value(self, system_state: SystemState, action: int) -> float:
        """实习医生的局部价值函数 - 重视教育和发展机会"""
        return (0.35 * system_state.education_training_quality +
                0.25 * system_state.intern_satisfaction +
                0.2 * system_state.professional_development +
                0.2 * system_state.staff_workload_balance)
    
    def _apply_holy_code_recommendations(self, action: np.ndarray, 
                                       recommendations: List[str]) -> np.ndarray:
        """应用实习医生相关的神圣法典建议"""
        for rec in recommendations:
            if '教育培训' in rec or '培训请求' in rec:
                action[0] = max(action[0], 0.8)
            elif '工作负荷' in rec or '工作负荷调整' in rec:
                action[1] = max(action[1], 0.7)
            elif '职业发展' in rec:
                action[2] = max(action[2], 0.6)
        
        return np.clip(action, 0, 1)

class PatientAgent(RoleAgent):
    """患者代表智能体 - 关注患者权益和服务质量"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注患者服务质量"""
        observation = np.zeros(self.state_dim)
        
        # 患者服务指标 (索引 0-3)
        observation[0] = environment.get('patient_satisfaction', 0.8)
        observation[1] = environment.get('service_accessibility', 0.7)
        observation[2] = environment.get('care_quality_index', 0.8)
        observation[3] = environment.get('patient_waiting_time', 0.3)
        
        # 系统服务质量 (索引 4-7)
        observation[4] = environment.get('safety_incident_rate', 0.1)
        observation[5] = environment.get('ethical_compliance', 0.9)
        observation[6] = environment.get('operational_efficiency', 0.7)
        observation[7] = environment.get('crisis_response_capability', 0.8)
        
        return observation
    
    def compute_local_value(self, system_state: SystemState, action: int) -> float:
        """患者代表的局部价值函数 - 重视患者体验和安全"""
        return (0.3 * system_state.patient_satisfaction +
                0.25 * system_state.service_accessibility +
                0.25 * system_state.care_quality_index +
                0.2 * (1.0 - system_state.safety_incident_rate))
    
    def _apply_holy_code_recommendations(self, action: np.ndarray, 
                                       recommendations: List[str]) -> np.ndarray:
        """应用患者相关的神圣法典建议"""
        for rec in recommendations:
            if '患者满意度' in rec or '满意度改进' in rec:
                action[0] = max(action[0], 0.8)
            elif '服务可及性' in rec or '可及性改善' in rec:
                action[1] = max(action[1], 0.7)
            elif '等待时间' in rec or '等待时间优化' in rec:
                action[2] = max(action[2], 0.6)
        
        return np.clip(action, 0, 1)

class AccountantAgent(RoleAgent):
    """会计智能体 - 关注财务健康和运营效率"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注财务和运营指标"""
        observation = np.zeros(self.state_dim)
        
        # 财务指标 (索引 0-3)
        observation[0] = environment.get('financial_indicator', 0.7)
        observation[1] = environment.get('operational_efficiency', 0.7)
        observation[2] = environment.get('medical_resource_utilization', 0.7)
        observation[3] = environment.get('staff_workload_balance', 0.6)
        
        # 系统效率指标 (索引 4-7)
        observation[4] = environment.get('patient_waiting_time', 0.3)
        observation[5] = environment.get('service_accessibility', 0.7)
        observation[6] = environment.get('regulatory_compliance_score', 0.8)
        observation[7] = environment.get('crisis_response_capability', 0.8)
        
        return observation
    
    def compute_local_value(self, system_state: SystemState, action: int) -> float:
        """会计的局部价值函数 - 重视财务健康和效率"""
        return (0.4 * system_state.financial_indicator +
                0.3 * system_state.operational_efficiency +
                0.2 * system_state.medical_resource_utilization +
                0.1 * system_state.regulatory_compliance_score)
    
    def _apply_holy_code_recommendations(self, action: np.ndarray, 
                                       recommendations: List[str]) -> np.ndarray:
        """应用会计相关的神圣法典建议"""
        for rec in recommendations:
            if '成本控制' in rec or '财务优化' in rec:
                action[0] = max(action[0], 0.8)
            elif '资源配置' in rec or '资源优化' in rec:
                action[1] = max(action[1], 0.7)
            elif '效率提升' in rec or '运营效率' in rec:
                action[2] = max(action[2], 0.7)
        
        return np.clip(action, 0, 1)

class GovernmentAgent(RoleAgent):
    """政府代理智能体 - 关注监管合规和公共利益"""
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """观察环境，关注监管和系统整体状态"""
        observation = np.zeros(self.state_dim)
        
        # 监管合规指标 (索引 0-3)
        observation[0] = environment.get('regulatory_compliance_score', 0.8)
        observation[1] = environment.get('ethical_compliance', 0.9)
        observation[2] = environment.get('crisis_response_capability', 0.8)
        observation[3] = environment.get('operational_efficiency', 0.7)
        
        # 公共利益指标 (索引 4-7)
        observation[4] = environment.get('patient_satisfaction', 0.8)
        observation[5] = environment.get('service_accessibility', 0.7)
        observation[6] = environment.get('safety_incident_rate', 0.1)
        observation[7] = environment.get('financial_indicator', 0.7)
        
        return observation
    
    def compute_local_value(self, system_state: SystemState, action: int) -> float:
        """政府代理的局部价值函数 - 重视整体系统稳定和公共利益"""
        return (0.25 * system_state.regulatory_compliance_score +
                0.25 * system_state.ethical_compliance +
                0.2 * system_state.crisis_response_capability +
                0.15 * system_state.patient_satisfaction +
                0.15 * system_state.service_accessibility)
    
    def _apply_holy_code_recommendations(self, action: np.ndarray, 
                                       recommendations: List[str]) -> np.ndarray:
        """应用政府相关的神圣法典建议"""
        for rec in recommendations:
            if '监管合规' in rec or '合规检查' in rec:
                action[0] = max(action[0], 0.9)
            elif '系统稳定' in rec or '稳定措施' in rec:
                action[1] = max(action[1], 0.8)
            elif '公共利益' in rec or '透明度提升' in rec:
                action[2] = max(action[2], 0.7)
            elif '危机响应' in rec:
                action[3] = max(action[3], 0.9)
        
        return np.clip(action, 0, 1)

class RoleManager:
    """角色管理器 - 统一管理所有智能体"""
    
    def __init__(self):
        self.agents: Dict[str, RoleAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """设置默认配置"""
        roles = ['doctors', 'interns', 'patients', 'accountants', 'government']
        
        for role in roles:
            config = AgentConfig(
                role=role,
                action_dim=5,  # 5个离散动作
                observation_dim=8,  # 8维局部观测
                learning_rate=0.001,
                alpha=0.3, beta=0.5, gamma=0.2  # 收益函数权重
            )
            self.agent_configs[role] = config
    
    def create_all_agents(self, custom_configs: Optional[Dict[str, AgentConfig]] = None):
        """创建所有智能体"""
        if custom_configs:
            self.agent_configs.update(custom_configs)
        
        agent_classes = {
            'doctors': DoctorAgent,
            'interns': InternAgent,
            'patients': PatientAgent,
            'accountants': AccountantAgent,
            'government': GovernmentAgent  # 新增政府代理
        }
        
        for role, agent_class in agent_classes.items():
            config = self.agent_configs[role]
            agent = agent_class(config)
            self.agents[role] = agent
            logger.info(f"Created {role} agent with config: {config}")
    
    def get_agent(self, role: str) -> Optional[RoleAgent]:
        """获取指定角色的智能体"""
        return self.agents.get(role)
    
    def get_all_agents(self) -> List[RoleAgent]:
        """获取所有智能体"""
        return list(self.agents.values())
    
    def get_agent_count(self) -> int:
        """获取智能体数量"""
        return len(self.agents)
    
    def update_all_policies(self, observations: Dict[str, np.ndarray], 
                           actions: Dict[str, int], q_values: Dict[str, float]):
        """批量更新所有智能体策略"""
        for role, agent in self.agents.items():
            if role in observations and role in actions and role in q_values:
                agent.update_policy(observations[role], actions[role], q_values[role])
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """获取所有智能体的性能摘要"""
        summary = {}
        for role, agent in self.agents.items():
            summary[role] = agent.get_performance_metrics()
        return summary

def create_default_agent_system() -> RoleManager:
    """创建默认的智能体系统"""
    manager = RoleManager()
    manager.create_all_agents()
    
    logger.info(f"Created agent system with {manager.get_agent_count()} agents:")
    for role in manager.agents.keys():
        logger.info(f"  - {role}")
    
    return manager

# 测试验证函数
def test_agent_system():
    """测试智能体系统"""
    print("🧪 测试多智能体系统")
    print("=" * 50)
    
    # 创建管理器
    manager = create_default_agent_system()
    
    # 模拟环境状态
    env_state = {
        'medical_resource_utilization': 0.8,
        'patient_waiting_time': 0.3,
        'financial_indicator': 0.7,
        'ethical_compliance': 0.9,
        'education_training_quality': 0.8,
        'intern_satisfaction': 0.7,
        'patient_satisfaction': 0.85,
        'service_accessibility': 0.8,
        'care_quality_index': 0.9,
        'safety_incident_rate': 0.05,
        'operational_efficiency': 0.75,
        'staff_workload_balance': 0.7,
        'crisis_response_capability': 0.8,
        'regulatory_compliance_score': 0.9
    }
    
    # 测试每个智能体
    for role, agent in manager.agents.items():
        print(f"\n测试 {role} 智能体:")
        
        # 观测
        observation = agent.observe(env_state)
        print(f"  观测维度: {observation.shape}")
        print(f"  观测样本: {observation[:4]}")  # 显示前4个值
        
        # 采样动作
        action = agent.sample_action(observation)
        print(f"  采样动作: {action}")
        
        # 计算策略概率
        probs = agent.compute_policy_probabilities(observation)
        print(f"  策略概率: {probs}")
        print(f"  概率和: {np.sum(probs):.3f}")
        
        # 模拟系统状态
        system_state = SystemState.from_vector(np.random.rand(16))
        ideal_state = SystemState.from_vector(np.random.rand(16))
        
        # 计算局部价值
        local_value = agent.compute_local_value(system_state, action)
        print(f"  局部价值: {local_value:.3f}")
        
        # 计算收益
        reward = agent.compute_reward(system_state, action, 0.8, ideal_state)
        print(f"  收益: {reward:.3f}")
        
        # 策略更新
        agent.update_policy(observation, action, reward)
        print(f"  策略参数更新完成")
    
    print(f"\n✅ 智能体系统测试完成")
    print(f"总计 {manager.get_agent_count()} 个智能体正常工作")
    
    return manager

if __name__ == "__main__":
    test_agent_system()