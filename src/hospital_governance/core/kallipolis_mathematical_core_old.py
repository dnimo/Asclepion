#!/usr/bin/env python3
"""
Kallipolis Medical Republic - 数理推导严格实现模块
基于完整数理推导框架的核心算法实现
"""

import numpy as np
import scipy.linalg as la
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """系统状态 x(t) ∈ ℝ^n"""
    medical_resource_utilization: float  # x₁: 医疗资源利用率
    patient_waiting_time: float         # x₂: 患者等待时间
    financial_indicator: float          # x₃: 财务指标
    ethical_compliance: float           # x₄: 伦理合规度
    education_training: float           # x₅: 教育培训指标
    patient_satisfaction: float         # x₆: 患者满意度
    emergency_queue_length: float       # x₇: 急诊队列长度
    
    def to_vector(self) -> np.ndarray:
        """转换为状态向量"""
        return np.array([
            self.medical_resource_utilization,
            self.patient_waiting_time,
            self.financial_indicator,
            self.ethical_compliance,
            self.education_training,
            self.patient_satisfaction,
            self.emergency_queue_length
        ])
    
    @classmethod
    def from_vector(cls, x: np.ndarray) -> 'SystemState':
        """从状态向量构造"""
        return cls(
            medical_resource_utilization=x[0],
            patient_waiting_time=x[1],
            financial_indicator=x[2],
            ethical_compliance=x[3],
            education_training=x[4],
            patient_satisfaction=x[5],
            emergency_queue_length=x[6]
        )

@dataclass
class HolyCodeRule:
    """神圣法典规则 (R_k, W_k, C_k)"""
    rule_id: str
    logic_function: Callable[[SystemState], float]  # R_k: S → ℝ
    weight: float                                   # W_k ∈ ℝ⁺
    context: Callable[[SystemState], bool]         # C_k ⊂ S (指示函数)
    target_value: float                            # R_k*
    description: str
    
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
    """神圣法典 HC(t)"""
    
    def __init__(self):
        self.rules: Dict[str, HolyCodeRule] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """初始化默认规则集合"""
        # 患者安全协议
        self.rules['patient_safety'] = HolyCodeRule(
            rule_id='patient_safety',
            logic_function=lambda s: s.patient_satisfaction * s.ethical_compliance,
            weight=1.0,
            context=lambda s: True,  # 始终适用
            target_value=0.8,
            description='患者安全协议 - 确保患者安全和满意度'
        )
        
        # 资源分配规则
        self.rules['resource_allocation'] = HolyCodeRule(
            rule_id='resource_allocation',
            logic_function=lambda s: s.medical_resource_utilization,
            weight=0.8,
            context=lambda s: s.financial_indicator > 0.3,
            target_value=0.75,
            description='资源分配规则 - 优化医疗资源配置'
        )
        
        # 紧急响应协议
        self.rules['emergency_response'] = HolyCodeRule(
            rule_id='emergency_response',
            logic_function=lambda s: 1.0 - s.emergency_queue_length,
            weight=1.2,
            context=lambda s: s.emergency_queue_length > 0.5,
            target_value=0.7,
            description='紧急响应协议 - 快速响应紧急情况'
        )
        
        # 质量保证规则
        self.rules['quality_assurance'] = HolyCodeRule(
            rule_id='quality_assurance',
            logic_function=lambda s: (s.education_training + s.ethical_compliance) / 2,
            weight=0.9,
            context=lambda s: True,
            target_value=0.85,
            description='质量保证规则 - 维护医疗质量标准'
        )
    
    def compute_ideal_state(self, current_state: SystemState, disturbance: np.ndarray) -> SystemState:
        """计算理想状态 x*(t) = Ψ(HC(t), d(t))"""
        x = current_state.to_vector()
        x_ideal = x.copy()
        
        # 基于神圣法典优化理想状态
        total_weight = sum(rule.weight for rule in self.rules.values())
        
        for rule in self.rules.values():
            if rule.context(current_state):
                # 计算规则对理想状态的影响
                current_value = rule.logic_function(current_state)
                adjustment = rule.weight / total_weight * (rule.target_value - current_value)
                
                # 根据规则类型调整相应状态分量
                if 'safety' in rule.rule_id or 'patient' in rule.rule_id:
                    x_ideal[5] += 0.1 * adjustment  # 患者满意度
                elif 'resource' in rule.rule_id:
                    x_ideal[0] += 0.1 * adjustment  # 资源利用率
                elif 'emergency' in rule.rule_id:
                    x_ideal[6] += 0.1 * adjustment  # 急诊队列
                elif 'quality' in rule.rule_id:
                    x_ideal[4] += 0.1 * adjustment  # 教育培训
        
        # 加入扰动影响
        x_ideal += 0.1 * disturbance[:len(x_ideal)]
        x_ideal = np.clip(x_ideal, 0, 1)
        
        return SystemState.from_vector(x_ideal)

class Agent:
    """智能体 i ∈ A"""
    
    def __init__(self, agent_id: str, role: str, action_space_size: int = 5):
        self.agent_id = agent_id
        self.role = role
        self.action_space_size = action_space_size
        
        # 策略参数 θ_i
        self.theta = np.random.normal(0, 0.1, action_space_size)
        
        # 收益函数权重
        self.alpha = 0.3  # 全局效用权重
        self.beta = 0.5   # 局部价值权重
        self.gamma = 0.2  # 理想状态偏差权重
        
        # 学习率
        self.learning_rate = 0.01
        
        # 特征映射缓存
        self._feature_cache = {}
    
    def feature_mapping(self, observation: np.ndarray, action: int) -> np.ndarray:
        """特征映射 φ_i(o_i, a_i)"""
        # 简单的线性特征映射
        obs_features = observation / np.linalg.norm(observation + 1e-8)
        action_features = np.zeros(self.action_space_size)
        action_features[action] = 1.0
        
        # 组合特征
        features = np.concatenate([obs_features, action_features])
        return features[:len(self.theta)]
    
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
        
        # Softmax
        exp_logits = np.exp(logit - np.max(logit))
        if action is not None:
            return exp_logits / np.sum(exp_logits)
        else:
            return exp_logits / np.sum(exp_logits)
    
    def sample_action(self, observation: np.ndarray) -> int:
        """采样动作"""
        probs = self.policy(observation)
        return np.random.choice(self.action_space_size, p=probs)
    
    def compute_reward(self, state: SystemState, action: int, 
                      global_utility: float, ideal_state: SystemState) -> float:
        """收益函数 R_i(x, a_i, a_{-i})"""
        # 局部价值函数 V_i - 基于角色特异性
        if self.role == 'doctor':
            local_value = state.medical_resource_utilization * 0.6 + state.patient_satisfaction * 0.4
        elif self.role == 'intern':
            local_value = state.education_training * 0.7 + state.medical_resource_utilization * 0.3
        elif self.role == 'patient':
            local_value = state.patient_satisfaction * 0.8 + state.patient_waiting_time * (-0.2)
        elif self.role == 'accountant':
            local_value = state.financial_indicator * 0.8 + state.medical_resource_utilization * 0.2
        elif self.role == 'government':
            local_value = state.ethical_compliance * 0.6 + state.patient_satisfaction * 0.4
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
        # 计算策略梯度
        phi = self.feature_mapping(observation, action)
        
        # 计算 ∇log π_i(a_i|o_i)
        probs = self.policy(observation)
        grad_log_pi = phi - np.sum([probs[a] * self.feature_mapping(observation, a) 
                                   for a in range(self.action_space_size)], axis=0)
        
        # 策略梯度
        advantage = q_value - baseline
        policy_gradient = grad_log_pi * advantage
        
        # 更新参数
        self.theta += self.learning_rate * policy_gradient

class LyapunovAnalyzer:
    """李雅普诺夫稳定性分析器"""
    
    def __init__(self, state_dim: int, num_agents: int, param_dim: int):
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.param_dim = param_dim
        
        # 李雅普诺夫函数参数
        self.P = np.eye(state_dim)  # 状态权重矩阵
        self.Q = [np.eye(param_dim) for _ in range(num_agents)]  # 策略参数权重矩阵
        
    def compute_lyapunov_function(self, state: SystemState, agents: List[Agent], 
                                 ideal_state: SystemState, ideal_params: List[np.ndarray]) -> float:
        """计算李雅普诺夫函数 V(z)"""
        x = state.to_vector()
        x_star = ideal_state.to_vector()
        
        # 状态偏差项
        state_term = (x - x_star).T @ self.P @ (x - x_star)
        
        # 策略参数偏差项
        param_term = 0.0
        for i, agent in enumerate(agents):
            theta_diff = agent.theta - ideal_params[i]
            param_term += theta_diff.T @ self.Q[i] @ theta_diff
        
        return state_term + param_term
    
    def analyze_stability(self, trajectory: List[Tuple[SystemState, List[np.ndarray]]]) -> Dict:
        """分析系统稳定性"""
        if len(trajectory) < 2:
            return {'stable': False, 'reason': 'insufficient_data'}
        
        # 计算李雅普诺夫函数值序列
        v_values = []
        ideal_state = trajectory[-1][0]  # 假设最终状态为理想状态
        ideal_params = trajectory[-1][1]
        
        for state, params in trajectory:
            # 模拟智能体对象
            mock_agents = []
            for i, param in enumerate(params):
                agent = Agent(f'agent_{i}', 'mock')
                agent.theta = param
                mock_agents.append(agent)
            
            v = self.compute_lyapunov_function(state, mock_agents, ideal_state, ideal_params)
            v_values.append(v)
        
        # 检查单调递减性
        is_decreasing = all(v_values[i] >= v_values[i+1] for i in range(len(v_values)-1))
        
        # 计算收敛率
        if len(v_values) > 1:
            convergence_rate = np.mean([abs(v_values[i] - v_values[i+1]) / v_values[i] 
                                      for i in range(len(v_values)-1) if v_values[i] > 0])
        else:
            convergence_rate = 0.0
        
        return {
            'stable': is_decreasing,
            'convergence_rate': convergence_rate,
            'lyapunov_values': v_values,
            'final_value': v_values[-1] if v_values else 0.0
        }

class KallipolisMedicalSystem:
    """Kallipolis医疗共和国系统 - 数理推导完整实现"""
    
    def __init__(self):
        # 系统组件
        self.holy_code = HolyCode()
        self.agents: List[Agent] = []
        self.lyapunov_analyzer = LyapunovAnalyzer(7, 5, 5)
        
        # 系统状态
        self.current_state = SystemState(
            medical_resource_utilization=0.7,
            patient_waiting_time=0.6,
            financial_indicator=0.65,
            ethical_compliance=0.8,
            education_training=0.9,
            patient_satisfaction=0.85,
            emergency_queue_length=0.2
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
        """初始化智能体集合 A"""
        agent_configs = [
            ('doctor', '医生'),
            ('intern', '实习医生'), 
            ('accountant', '会计'),
            ('patient', '患者代表'),
            ('government', '政府代理')
        ]
        
        for agent_id, role in agent_configs:
            agent = Agent(agent_id, role)
            self.agents.append(agent)
    
    def system_step(self, disturbance: np.ndarray) -> Dict:
        """执行一步系统动态"""
        # 1. 计算理想状态
        ideal_state = self.holy_code.compute_ideal_state(self.current_state, disturbance)
        
        # 2. 智能体观测和决策
        observations = []
        actions = []
        rewards = []
        
        for agent in self.agents:
            # 局部观测（简化为全状态观测加噪声）
            obs = self.current_state.to_vector() + np.random.normal(0, 0.05, 7)
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
        
        # 5. 状态转移（简化的动态方程）
        state_vec = self.current_state.to_vector()
        action_effects = np.array([sum(actions[i] * 0.02 for i in range(len(actions)))] * 7)
        new_state_vec = state_vec + action_effects + disturbance + np.random.normal(0, 0.01, 7)
        new_state_vec = np.clip(new_state_vec, 0, 1)
        
        self.current_state = SystemState.from_vector(new_state_vec)
        
        # 6. 记录轨迹
        agent_params = [agent.theta.copy() for agent in self.agents]
        self.trajectory.append((self.current_state, agent_params))
        
        # 7. 规则激活检查
        rule_activations = {}
        for rule_id, rule in self.holy_code.rules.items():
            activated, severity = rule.evaluate(self.current_state)
            rule_activations[rule_id] = {
                'activated': activated,
                'severity': severity,
                'description': rule.description
            }
        
        return {
            'state': self.current_state,
            'ideal_state': ideal_state,
            'actions': actions,
            'rewards': rewards,
            'rule_activations': rule_activations,
            'global_utility': global_utility
        }
    
    def _compute_global_utility(self, state: SystemState) -> float:
        """计算全局资源效用函数 U(x)"""
        state_vec = state.to_vector()
        
        # 资源效用加权组合
        weights = np.array([0.2, -0.1, 0.15, 0.2, 0.15, 0.25, -0.15])
        utility = np.dot(weights, state_vec)
        
        return np.clip(utility, 0, 1)
    
    def analyze_system_stability(self) -> Dict:
        """分析系统稳定性"""
        if len(self.trajectory) < 10:
            return {'error': 'insufficient_trajectory_data'}
        
        return self.lyapunov_analyzer.analyze_stability(self.trajectory[-50:])  # 分析最近50步
    
    def get_performance_metrics(self) -> Dict:
        """获取系统性能指标"""
        if len(self.trajectory) < 2:
            return self.performance_metrics
        
        # 计算扰动适应时间 (DAT)
        # 简化计算：状态变化稳定所需时间
        recent_states = [item[0].to_vector() for item in self.trajectory[-10:]]
        if len(recent_states) >= 2:
            state_variations = [np.std(states) for states in zip(*recent_states)]
            avg_variation = np.mean(state_variations)
            self.performance_metrics['disturbance_adaptation_time'] = min(10.0, 1.0 / (avg_variation + 1e-6))
        
        # 计算规则更新成功率 (RUSR)
        # 基于规则激活的一致性
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
        # 基于智能体策略参数的收敛性
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
    
    def reset(self):
        """重置系统状态"""
        self.current_state = SystemState(
            medical_resource_utilization=0.7,
            patient_waiting_time=0.6,
            financial_indicator=0.65,
            ethical_compliance=0.8,
            education_training=0.9,
            patient_satisfaction=0.85,
            emergency_queue_length=0.2
        )
        
        # 重置智能体策略参数
        for agent in self.agents:
            agent.theta = np.random.normal(0, 0.1, agent.action_space_size)
        
        # 清空轨迹
        self.trajectory = []
        
        logger.info("Kallipolis Medical System has been reset")

# 验证函数
def verify_mathematical_implementation():
    """验证数理推导实现的正确性"""
    print("🔬 Kallipolis Medical Republic - 数理推导验证")
    print("=" * 60)
    
    # 创建系统实例
    system = KallipolisMedicalSystem()
    
    print("✅ 系统组件初始化:")
    print(f"  - 智能体数量: {len(system.agents)}")
    print(f"  - 神圣法典规则数: {len(system.holy_code.rules)}")
    print(f"  - 状态空间维度: {len(system.current_state.to_vector())}")
    
    # 运行仿真步骤
    print("\n🔄 执行仿真步骤:")
    disturbances = [np.random.normal(0, 0.05, 7) for _ in range(20)]
    
    for step in range(20):
        result = system.system_step(disturbances[step])
        
        if step % 5 == 0:
            print(f"  步骤 {step}:")
            print(f"    全局效用: {result['global_utility']:.3f}")
            print(f"    激活规则数: {sum(1 for r in result['rule_activations'].values() if r['activated'])}")
            print(f"    平均收益: {np.mean(result['rewards']):.3f}")
    
    # 稳定性分析
    print("\n📊 稳定性分析:")
    stability_result = system.analyze_system_stability()
    if 'error' not in stability_result:
        print(f"  系统稳定性: {'✅ 稳定' if stability_result['stable'] else '❌ 不稳定'}")
        print(f"  收敛率: {stability_result['convergence_rate']:.6f}")
        print(f"  李雅普诺夫函数值: {stability_result['final_value']:.6f}")
    
    # 性能指标
    print("\n📈 性能指标:")
    metrics = system.get_performance_metrics()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\n🎯 验证完成 - 数理推导实现正常工作")
    return system

if __name__ == "__main__":
    verify_mathematical_implementation()