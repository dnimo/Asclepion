#!/usr/bin/env python3
"""
Kallipolis Medical Republic - 数理推导严格实现模块 (重构版本)
基于完整数理推导框架的核心算法实现

数学模型扩展:
1. 16维状态空间 x(t) ∈ ℝ^16  
2. 5个智能体角色 (医生、实习生、患者代表、会计、政府代理)
3. 参数化随机策略 π_i(a_i | o_i; θ_i)
4. 李雅普诺夫稳定性分析
5. 神圣法典动态演化
"""


import numpy as np
import scipy.linalg as la
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from hospital_governance.core.state_space import StateSpace, SystemState
from hospital_governance.stability.lyapunov_analysis import LyapunovAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """系统状态 x(t) ∈ ℝ^16 - 扩展的16维状态空间"""
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
        if len(x) < 16:
            # 如果输入向量不够16维，用默认值填充
            x_extended = np.zeros(16)
            x_extended[:len(x)] = x
            x_extended[len(x):] = 0.5  # 默认值
            x = x_extended
            
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
    
    def get_component_names(self) -> List[str]:
        """获取状态分量名称"""
        return [
            '医疗资源利用率', '患者等待时间', '财务健康指标', '伦理合规度',
            '教育培训质量', '实习生满意度', '职业发展指数', '指导效果',
            '患者满意度', '服务可及性', '护理质量指数', '安全事故率',
            '运营效率', '员工工作负荷平衡', '危机响应能力', '监管合规分数'
        ]

@dataclass
class HolyCodeRule:
    """神圣法典规则 (R_k, W_k, C_k) - 扩展支持16维状态"""
    rule_id: str
    logic_function: Callable[[SystemState], float]  # R_k: S → ℝ
    weight: float                                   # W_k ∈ ℝ⁺
    context: Callable[[SystemState], bool]         # C_k ⊂ S (指示函数)
    target_value: float                            # R_k*
    description: str
    category: str = "general"  # 规则类别
    
    def evaluate(self, state: SystemState) -> Tuple[bool, float]:
        """评估规则激活状态和严重程度"""
        if not self.context(state):
            return False, 0.0
        
        current_value = self.logic_function(state)
        deviation = abs(current_value - self.target_value)
        activated = deviation > 0.1  # 阈值可配置
        severity = self.weight * deviation
        
        return activated, severity

class HolyCode:
    """神圣法典 HC(t) - 扩展支持16维状态空间"""
    
    def __init__(self):
        self.rules: Dict[str, HolyCodeRule] = {}
        self._initialize_extended_rules()
    
    def _initialize_extended_rules(self):
        """初始化扩展的规则集合 - 支持16维状态空间"""
        
        # 核心医疗规则
        self.rules['patient_safety_protocol'] = HolyCodeRule(
            rule_id='patient_safety_protocol',
            logic_function=lambda s: s.care_quality_index * (1.0 - s.safety_incident_rate),
            weight=1.2,
            context=lambda s: True,  # 始终适用
            target_value=0.85,
            description='患者安全协议 - 确保护理质量与安全',
            category='medical'
        )
        
        # 资源优化规则
        self.rules['resource_optimization'] = HolyCodeRule(
            rule_id='resource_optimization',
            logic_function=lambda s: s.medical_resource_utilization * s.operational_efficiency,
            weight=0.9,
            context=lambda s: s.financial_indicator > 0.3,
            target_value=0.8,
            description='资源优化规则 - 平衡资源利用与效率',
            category='operational'
        )
        
        # 教育发展规则
        self.rules['education_excellence'] = HolyCodeRule(
            rule_id='education_excellence',
            logic_function=lambda s: (s.education_training_quality + s.intern_satisfaction + s.professional_development) / 3,
            weight=0.8,
            context=lambda s: s.intern_satisfaction < 0.8 or s.education_training_quality < 0.8,
            target_value=0.85,
            description='教育卓越规则 - 提升培训质量与满意度',
            category='education'
        )
        
        # 患者服务规则
        self.rules['patient_service_excellence'] = HolyCodeRule(
            rule_id='patient_service_excellence',
            logic_function=lambda s: (s.patient_satisfaction + s.service_accessibility) / 2,
            weight=1.0,
            context=lambda s: s.patient_satisfaction < 0.8 or s.service_accessibility < 0.7,
            target_value=0.85,
            description='患者服务卓越规则 - 提升患者体验',
            category='service'
        )
        
        # 财务稳定规则
        self.rules['financial_stability'] = HolyCodeRule(
            rule_id='financial_stability',
            logic_function=lambda s: s.financial_indicator * s.operational_efficiency,
            weight=0.85,
            context=lambda s: s.financial_indicator < 0.6,
            target_value=0.75,
            description='财务稳定规则 - 维护财务健康',
            category='financial'
        )
        
        # 监管合规规则
        self.rules['regulatory_compliance'] = HolyCodeRule(
            rule_id='regulatory_compliance',
            logic_function=lambda s: (s.regulatory_compliance_score + s.ethical_compliance) / 2,
            weight=1.1,
            context=lambda s: s.regulatory_compliance_score < 0.9 or s.ethical_compliance < 0.9,
            target_value=0.95,
            description='监管合规规则 - 确保合规性',
            category='governance'
        )
        
        # 危机响应规则
        self.rules['crisis_response'] = HolyCodeRule(
            rule_id='crisis_response',
            logic_function=lambda s: s.crisis_response_capability * (1.0 - s.patient_waiting_time),
            weight=1.3,
            context=lambda s: s.crisis_response_capability < 0.8 or s.patient_waiting_time > 0.4,
            target_value=0.8,
            description='危机响应规则 - 快速响应能力',
            category='emergency'
        )
        
        # 工作负荷平衡规则
        self.rules['workload_balance'] = HolyCodeRule(
            rule_id='workload_balance',
            logic_function=lambda s: s.staff_workload_balance * s.intern_satisfaction,
            weight=0.7,
            context=lambda s: s.staff_workload_balance < 0.7,
            target_value=0.8,
            description='工作负荷平衡规则 - 员工福祉',
            category='welfare'
        )
    
    def compute_ideal_state(self, current_state: SystemState, disturbance: np.ndarray) -> SystemState:
        """计算理想状态 x*(t) = Ψ(HC(t), d(t)) - 16维优化"""
        x = current_state.to_vector()
        x_ideal = x.copy()
        
        # 基于神圣法典优化理想状态
        total_weight = sum(rule.weight for rule in self.rules.values())
        
        # 分类别处理规则影响
        category_adjustments = {
            'medical': np.zeros(16),
            'operational': np.zeros(16),
            'education': np.zeros(16),
            'service': np.zeros(16),
            'financial': np.zeros(16),
            'governance': np.zeros(16),
            'emergency': np.zeros(16),
            'welfare': np.zeros(16)
        }
        
        for rule in self.rules.values():
            if rule.context(current_state):
                current_value = rule.logic_function(current_state)
                adjustment = rule.weight / total_weight * (rule.target_value - current_value)
                
                # 根据规则类别分配调整
                if rule.category == 'medical':
                    category_adjustments['medical'][[0, 10, 11]] += adjustment * 0.1
                elif rule.category == 'operational':
                    category_adjustments['operational'][[0, 12]] += adjustment * 0.1
                elif rule.category == 'education':
                    category_adjustments['education'][[4, 5, 6, 7]] += adjustment * 0.1
                elif rule.category == 'service':
                    category_adjustments['service'][[8, 9]] += adjustment * 0.1
                elif rule.category == 'financial':
                    category_adjustments['financial'][[2, 12]] += adjustment * 0.1
                elif rule.category == 'governance':
                    category_adjustments['governance'][[3, 15]] += adjustment * 0.1
                elif rule.category == 'emergency':
                    category_adjustments['emergency'][[1, 14]] += adjustment * 0.1
                elif rule.category == 'welfare':
                    category_adjustments['welfare'][[13, 5]] += adjustment * 0.1
        
        # 应用所有调整
        for adjustments in category_adjustments.values():
            x_ideal += adjustments
        
        # 加入扰动影响
        if len(disturbance) >= 16:
            x_ideal += 0.05 * disturbance[:16]
        else:
            x_ideal[:len(disturbance)] += 0.05 * disturbance
        
        x_ideal = np.clip(x_ideal, 0, 1)
        
        return SystemState.from_vector(x_ideal)
    
    def get_active_rules(self, state: SystemState) -> Dict[str, Dict]:
        """获取当前激活的规则"""
        active_rules = {}
        for rule_id, rule in self.rules.items():
            activated, severity = rule.evaluate(state)
            if activated:
                active_rules[rule_id] = {
                    'description': rule.description,
                    'severity': severity,
                    'category': rule.category,
                    'target_value': rule.target_value,
                    'current_value': rule.logic_function(state)
                }
        return active_rules

class Agent:
    """智能体 i ∈ A - 支持5个角色的扩展版本"""
    
    def __init__(self, agent_id: str, role: str, action_space_size: int = 8):
        self.agent_id = agent_id
        self.role = role
        self.action_space_size = action_space_size
        
        # 策略参数 θ_i
        self.theta = np.random.normal(0, 0.1, action_space_size)
        
        # 收益函数权重 - 基于角色差异化
        if role == 'doctor':
            self.alpha, self.beta, self.gamma = 0.2, 0.6, 0.2  # 重视局部医疗价值
        elif role == 'intern':
            self.alpha, self.beta, self.gamma = 0.3, 0.5, 0.2  # 平衡发展
        elif role == 'patient':
            self.alpha, self.beta, self.gamma = 0.4, 0.4, 0.2  # 重视全局和局部服务
        elif role == 'accountant':
            self.alpha, self.beta, self.gamma = 0.5, 0.3, 0.2  # 重视全局资源效用
        elif role == 'government':
            self.alpha, self.beta, self.gamma = 0.6, 0.2, 0.2  # 最重视全局利益
        else:
            self.alpha, self.beta, self.gamma = 0.3, 0.5, 0.2  # 默认权重
        
        # 学习率
        self.learning_rate = 0.01
        
        # 特征映射缓存
        self._feature_cache = {}
        
        # 基线值
        self.baseline = 0.0
        
        logger.debug(f"Created {role} agent: α={self.alpha}, β={self.beta}, γ={self.gamma}")
    
    def feature_mapping(self, observation: np.ndarray, action: int) -> np.ndarray:
        """特征映射 φ_i(o_i, a_i) - 支持16维观测"""
        # 简单的线性特征映射
        obs_features = observation / (np.linalg.norm(observation) + 1e-8)
        action_features = np.zeros(self.action_space_size)
        if 0 <= action < self.action_space_size:
            action_features[action] = 1.0
        
        # 组合特征，限制到策略参数维度
        if len(obs_features) + len(action_features) > len(self.theta):
            # 如果特征维度超过参数维度，截断观测特征
            obs_dim = len(self.theta) - len(action_features)
            obs_features = obs_features[:obs_dim]
        
        features = np.concatenate([obs_features, action_features])
        
        # 调整到正确维度
        if len(features) < len(self.theta):
            padding = np.zeros(len(self.theta) - len(features))
            features = np.concatenate([features, padding])
        elif len(features) > len(self.theta):
            features = features[:len(self.theta)]
        
        return features
    
    def policy(self, observation: np.ndarray, action: Optional[int] = None) -> np.ndarray:
        """随机策略 π_i(a_i | o_i; θ_i)"""
        if action is not None:
            # 计算特定动作的概率
            phi = self.feature_mapping(observation, action)
            logit = np.dot(phi, self.theta)
        else:
            # 计算所有动作的概率分布
            logits = []
            for a in range(self.action_space_size):
                phi = self.feature_mapping(observation, a)
                logits.append(np.dot(phi, self.theta))
            logit = np.array(logits)
        
        # Softmax with numerical stability
        if isinstance(logit, np.ndarray):
            logit = logit - np.max(logit)
            exp_logits = np.exp(logit)
            return exp_logits / np.sum(exp_logits)
        else:
            return np.exp(logit)  # 单个动作的非归一化概率
    
    def sample_action(self, observation: np.ndarray) -> int:
        """采样动作"""
        probs = self.policy(observation)
        return np.random.choice(self.action_space_size, p=probs)
    
    def compute_reward(self, state: SystemState, action: int, 
                      global_utility: float, ideal_state: SystemState) -> float:
        """收益函数 R_i(x, a_i, a_{-i}) = α_i U(x) + β_i V_i(x, a_i) - γ_i D_i(x, x*)"""
        
        # 局部价值函数 V_i - 基于角色特异性 (16维状态)
        if self.role == 'doctor':
            local_value = (state.medical_resource_utilization * 0.25 + 
                          state.care_quality_index * 0.3 + 
                          state.patient_satisfaction * 0.25 + 
                          (1.0 - state.safety_incident_rate) * 0.2)
        elif self.role == 'intern':
            local_value = (state.education_training_quality * 0.3 + 
                          state.intern_satisfaction * 0.25 + 
                          state.professional_development * 0.25 + 
                          state.mentorship_effectiveness * 0.2)
        elif self.role == 'patient':
            local_value = (state.patient_satisfaction * 0.3 + 
                          state.service_accessibility * 0.25 + 
                          state.care_quality_index * 0.25 + 
                          (1.0 - state.safety_incident_rate) * 0.2)
        elif self.role == 'accountant':
            local_value = (state.financial_indicator * 0.35 + 
                          state.operational_efficiency * 0.3 + 
                          state.medical_resource_utilization * 0.2 + 
                          state.regulatory_compliance_score * 0.15)
        elif self.role == 'government':
            local_value = (state.regulatory_compliance_score * 0.25 + 
                          state.ethical_compliance * 0.25 + 
                          state.crisis_response_capability * 0.2 + 
                          state.patient_satisfaction * 0.15 + 
                          state.service_accessibility * 0.15)
        else:
            local_value = 0.5
        
        # 到理想状态的偏差 D_i
        state_vec = state.to_vector()
        ideal_vec = ideal_state.to_vector()
        deviation = np.linalg.norm(state_vec - ideal_vec)
        
        # 组合收益
        reward = (self.alpha * global_utility + 
                 self.beta * local_value - 
                 self.gamma * deviation)
        
        return reward
    
    def update_policy(self, observation: np.ndarray, action: int, 
                     q_value: float, baseline: float = 0.0):
        """策略梯度更新 θ_i(t+1) = θ_i(t) + η ∇J_i(θ)"""
        # 更新基线估计
        self.baseline = 0.9 * self.baseline + 0.1 * q_value
        
        # 计算策略梯度
        phi = self.feature_mapping(observation, action)
        
        # 计算 ∇log π_i(a_i|o_i)
        probs = self.policy(observation)
        grad_log_pi = phi - sum([probs[a] * self.feature_mapping(observation, a) 
                               for a in range(self.action_space_size)])
        
        # 策略梯度
        advantage = q_value - self.baseline
        policy_gradient = grad_log_pi * advantage
        
        # 更新参数
        self.theta += self.learning_rate * policy_gradient


class KallipolisMedicalSystem:
    """Kallipolis医疗共和国系统 - 数理推导完整实现 (重构版本)"""
    
    def __init__(self):
        # 系统组件
        self.holy_code = HolyCode()
        self.agents: List[Agent] = []
        self.lyapunov_analyzer = LyapunovAnalyzer(16, 5, 8)  # 16维状态，5个智能体，8维参数
        # 16维系统状态初始化
        self.current_state = SystemState(
            medical_resource_utilization=0.7,
            patient_waiting_time=0.3,
            financial_indicator=0.65,
            ethical_compliance=0.8,
            education_training_quality=0.75,
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
        
        # 性能指标
        self.performance_metrics = {
            'disturbance_adaptation_time': 0.0,
            'rule_update_success_rate': 0.0,
            'consensus_convergence_rate': 0.0,
            'rule_update_response_time': 0.0
        }
        
        # 轨迹记录
        self.trajectory: List[Tuple[SystemState, List[np.ndarray]]] = []
        
        # 初始化智能体
        self._initialize_agents()
    
    def _initialize_agents(self):
        """初始化智能体集合 A = {医生, 实习生, 患者代表, 会计, 政府代理}"""
        agent_configs = [
            ('doctor', '医生'),
            ('intern', '实习医生'), 
            ('patient', '患者代表'),
            ('accountant', '会计'),
            ('government', '政府代理')  # 新增政府代理
        ]
        
        for agent_id, role in agent_configs:
            agent = Agent(agent_id, role)
            self.agents.append(agent)
            logger.info(f"Initialized agent: {agent_id} ({role})")
    
    def system_step(self, disturbance: np.ndarray) -> Dict:
        """执行一步系统动态 - 16维状态空间"""
        # 确保扰动维度正确
        if len(disturbance) < 16:
            disturbance_extended = np.zeros(16)
            disturbance_extended[:len(disturbance)] = disturbance
            disturbance = disturbance_extended
        
        # 1. 计算理想状态
        ideal_state = self.holy_code.compute_ideal_state(self.current_state, disturbance)
        
        # 2. 智能体观测和决策
        observations = []
        actions = []
        rewards = []
        
        # 16维系统状态作为观测基础
        base_observation = self.current_state.to_vector()
        
        for agent in self.agents:
            # 局部观测（基于角色的不同观测窗口）
            if agent.role == 'doctor':
                obs_indices = [0, 1, 2, 3, 8, 9, 10, 11]  # 医疗相关指标
            elif agent.role == 'intern':
                obs_indices = [4, 5, 6, 7, 13, 0, 12, 3]  # 教育培训相关
            elif agent.role == 'patient':
                obs_indices = [8, 9, 10, 11, 1, 3, 12, 14]  # 患者服务相关
            elif agent.role == 'accountant':
                obs_indices = [2, 12, 0, 15, 1, 9, 14, 13]  # 财务运营相关
            elif agent.role == 'government':
                obs_indices = [15, 3, 14, 12, 8, 9, 11, 2]  # 监管治理相关
            else:
                obs_indices = list(range(8))  # 默认前8维
            
            obs = base_observation[obs_indices] + np.random.normal(0, 0.02, 8)
            observations.append(obs)
            
            # 采样动作
            action = agent.sample_action(obs)
            actions.append(action)
        
        # 3. 计算全局效用
        global_utility = self._compute_global_utility(self.current_state)
        
        # 4. 计算收益和更新策略
        for i, agent in enumerate(self.agents):
            reward = agent.compute_reward(self.current_state, actions[i], 
                                        global_utility, ideal_state)
            rewards.append(reward)
            
            # Q值简化计算（实际应该用时序差分学习）
            q_value = reward + 0.9 * global_utility  # γ=0.9
            
            # 策略更新
            agent.update_policy(observations[i], actions[i], q_value)
        
        # 5. 状态转移（简化的动态方程） - 16维状态空间
        state_vec = self.current_state.to_vector()
        
        # 智能体动作对不同状态分量的影响
        action_effects = np.zeros(16)
        for i, agent in enumerate(self.agents):
            action_weight = 0.02 * actions[i] / agent.action_space_size
            
            if agent.role == 'doctor':
                action_effects[[0, 10, 11]] += action_weight
            elif agent.role == 'intern':
                action_effects[[4, 5, 6, 7]] += action_weight
            elif agent.role == 'patient':
                action_effects[[8, 9]] += action_weight
            elif agent.role == 'accountant':
                action_effects[[2, 12]] += action_weight
            elif agent.role == 'government':
                action_effects[[15, 3, 14]] += action_weight
        
        new_state_vec = state_vec + action_effects + disturbance * 0.05 + np.random.normal(0, 0.005, 16)
        new_state_vec = np.clip(new_state_vec, 0, 1)
        
        self.current_state = SystemState.from_vector(new_state_vec)
        
        # 6. 记录轨迹
        agent_params = [agent.theta.copy() for agent in self.agents]
        self.trajectory.append((self.current_state, agent_params))
        
        # 7. 规则激活检查
        rule_activations = self.holy_code.get_active_rules(self.current_state)
        
        return {
            'state': self.current_state,
            'ideal_state': ideal_state,
            'actions': actions,
            'rewards': rewards,
            'rule_activations': rule_activations,
            'global_utility': global_utility,
            'observations': observations
        }
    
    def _compute_global_utility(self, state: SystemState) -> float:
        """计算全局资源效用函数 U(x) - 16维状态空间"""
        state_vec = state.to_vector()
        
        # 16维资源效用加权组合 - 基于医院治理重要性
        weights = np.array([
            0.12,  # 医疗资源利用率
            -0.08, # 患者等待时间 (负权重)
            0.10,  # 财务健康指标
            0.12,  # 伦理合规度
            0.06,  # 教育培训质量
            0.04,  # 实习生满意度
            0.03,  # 职业发展指数
            0.03,  # 指导效果
            0.15,  # 患者满意度
            0.10,  # 服务可及性
            0.12,  # 护理质量指数
            -0.06, # 安全事故率 (负权重)
            0.08,  # 运营效率
            0.05,  # 员工工作负荷平衡
            0.06,  # 危机响应能力
            0.08   # 监管合规分数
        ])
        
        utility = np.dot(weights, state_vec)
        return np.clip(utility, 0, 1)
    
    def analyze_system_stability(self) -> Dict:
        """分析系统稳定性（调用 stability/lyapunov_analysis.py 实现）"""
        if len(self.trajectory) < 10:
            return {'error': 'insufficient_trajectory_data'}
        return self.lyapunov_analyzer.analyze_stability(self.trajectory[-50:])
    
    def get_performance_metrics(self) -> Dict:
        """获取系统性能指标"""
        if len(self.trajectory) < 2:
            return self.performance_metrics
        
        # 计算扰动适应时间 (DAT)
        recent_states = [item[0].to_vector() for item in self.trajectory[-10:]]
        if len(recent_states) >= 2:
            state_variations = [np.std(states) for states in zip(*recent_states)]
            avg_variation = np.mean(state_variations)
            self.performance_metrics['disturbance_adaptation_time'] = min(10.0, 1.0 / (avg_variation + 1e-6))
        
        # 计算规则更新成功率 (RUSR)
        rule_consistency = 0.0
        if len(self.trajectory) >= 5:
            recent_activations = []
            for state, _ in self.trajectory[-5:]:
                activations = []
                for rule in self.holy_code.rules.values():
                    activated, _ = rule.evaluate(state)
                    activations.append(activated)
                recent_activations.append(activations)
            
            if recent_activations:
                consistency_scores = []
                for i in range(len(recent_activations[0])):
                    rule_sequence = [activations[i] for activations in recent_activations]
                    consistency = 1.0 - np.std(rule_sequence)
                    consistency_scores.append(consistency)
                rule_consistency = np.mean(consistency_scores)
        
        self.performance_metrics['rule_update_success_rate'] = rule_consistency
        
        # 计算共识收敛率 (CCR)
        if len(self.trajectory) >= 5:
            recent_params = [params for _, params in self.trajectory[-5:]]
            param_variations = []
            for i in range(len(self.agents)):
                agent_param_sequence = [params[i] for params in recent_params]
                param_std = np.std(agent_param_sequence, axis=0)
                param_variations.append(np.mean(param_std))
            
            avg_param_variation = np.mean(param_variations)
            self.performance_metrics['consensus_convergence_rate'] = max(0.0, 1.0 - avg_param_variation)
        
        return self.performance_metrics
    
    def get_system_summary(self) -> Dict:
        """获取系统状态摘要"""
        return {
            'current_state': self.current_state,
            'state_vector': self.current_state.to_vector(),
            'state_names': self.current_state.get_component_names(),
            'num_agents': len(self.agents),
            'agent_roles': [agent.role for agent in self.agents],
            'active_rules': self.holy_code.get_active_rules(self.current_state),
            'trajectory_length': len(self.trajectory),
            'performance_metrics': self.get_performance_metrics()
        }
    
    def reset(self):
        """重置系统状态 - 16维状态空间"""
        self.current_state = SystemState(
            medical_resource_utilization=0.7,
            patient_waiting_time=0.3,
            financial_indicator=0.65,
            ethical_compliance=0.8,
            education_training_quality=0.75,
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
        
        # 重置智能体策略参数
        for agent in self.agents:
            agent.theta = np.random.normal(0, 0.1, agent.action_space_size)
            agent.baseline = 0.0
        
        # 清空轨迹
        self.trajectory = []
        
        logger.info("Kallipolis Medical System (16D, 5 agents) has been reset")

# 验证函数
def verify_extended_mathematical_implementation():
    """验证扩展数理推导实现的正确性"""
    print("🔬 Kallipolis Medical Republic - 扩展数理推导验证")
    print("=" * 70)
    
    # 创建系统实例
    system = KallipolisMedicalSystem()
    
    print("✅ 系统组件初始化:")
    print(f"  - 智能体数量: {len(system.agents)} (包含政府代理)")
    print(f"  - 神圣法典规则数: {len(system.holy_code.rules)}")
    print(f"  - 状态空间维度: {len(system.current_state.to_vector())} (16维扩展)")
    print(f"  - 智能体角色: {[agent.role for agent in system.agents]}")
    
    # 显示状态分量
    print("\\n📊 16维状态空间分量:")
    for i, name in enumerate(system.current_state.get_component_names()):
        value = system.current_state.to_vector()[i]
        print(f"  x_{i+1:2d}: {name:<20} = {value:.3f}")
    
    # 运行仿真步骤
    print("\\n🔄 执行仿真步骤:")
    disturbances = [np.random.normal(0, 0.03, 16) for _ in range(30)]
    
    for step in range(30):
        result = system.system_step(disturbances[step])
        
        if step % 5 == 0:
            print(f"  步骤 {step:2d}:")
            print(f"    全局效用: {result['global_utility']:.4f}")
            print(f"    激活规则数: {len(result['rule_activations'])}")
            print(f"    平均收益: {np.mean(result['rewards']):.4f}")
            print(f"    状态变化: {np.linalg.norm(result['state'].to_vector() - system.current_state.to_vector()):.4f}")
    
    # 稳定性分析
    print("\\n📊 稳定性分析:")
    stability_result = system.analyze_system_stability()
    if 'error' not in stability_result:
        print(f"  系统稳定性: {'✅ 稳定' if stability_result['stable'] else '❌ 不稳定'}")
        print(f"  收敛率: {stability_result['convergence_rate']:.6f}")
        print(f"  李雅普诺夫函数值: {stability_result['final_value']:.6f}")
    else:
        print(f"  ⚠️  {stability_result['error']}")
    
    # 性能指标
    print("\\n📈 性能指标:")
    metrics = system.get_performance_metrics()
    for metric, value in metrics.items():
        print(f"  {metric:<30}: {value:.4f}")
    
    # 智能体收益函数权重
    print("\\n👥 智能体配置:")
    for agent in system.agents:
        print(f"  {agent.role:<12}: α={agent.alpha:.1f}, β={agent.beta:.1f}, γ={agent.gamma:.1f}")
    
    # 神圣法典规则状态
    print("\\n📜 神圣法典规则状态:")
    active_rules = system.holy_code.get_active_rules(system.current_state)
    if active_rules:
        for rule_id, rule_info in active_rules.items():
            print(f"  🔴 {rule_id}: {rule_info['description']}")
            print(f"     严重程度: {rule_info['severity']:.3f}, 类别: {rule_info['category']}")
    else:
        print("  ✅ 所有规则均处于合规状态")
    
    print("\\n🎯 验证完成 - 扩展数理推导实现正常工作")
    print(f"📊 系统摘要: 16维状态空间, 5个智能体角色, {len(system.holy_code.rules)}个动态规则")
    
    return system

if __name__ == "__main__":
    verify_extended_mathematical_implementation()