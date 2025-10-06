"""
医院治理智能体行为模型

这个模块定义了不同类型的行为模型，用于指导智能体的决策行为。
行为模型基于心理学、经济学和博弈论原理，模拟真实世界中不同角色的行为模式。
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
from enum import Enum

class BehaviorType(Enum):
    """行为类型枚举"""
    RATIONAL = "rational"           # 理性行为
    BOUNDED_RATIONAL = "bounded_rational"  # 有限理性
    EMOTIONAL = "emotional"         # 情感驱动
    SOCIAL = "social"              # 社会性行为
    ADAPTIVE = "adaptive"          # 适应性行为
    HABITUAL = "habitual"          # 习惯性行为
    RISK_AVERSE = "risk_averse"    # 风险厌恶
    RISK_SEEKING = "risk_seeking"   # 风险寻求

@dataclass
class BehaviorParameters:
    """行为参数配置"""
    rationality_level: float = 0.8        # 理性程度 [0,1]
    emotional_weight: float = 0.2         # 情感权重 [0,1]
    social_influence: float = 0.3         # 社会影响 [0,1]
    risk_tolerance: float = 0.5           # 风险容忍度 [0,1]
    adaptation_rate: float = 0.1          # 适应速率 [0,1]
    habit_strength: float = 0.4           # 习惯强度 [0,1]
    cooperation_tendency: float = 0.6     # 合作倾向 [0,1]
    fairness_concern: float = 0.5         # 公平关注 [0,1]
    time_horizon: int = 5                 # 决策时间跨度
    
    def __post_init__(self):
        """验证参数范围"""
        for attr_name in ['rationality_level', 'emotional_weight', 'social_influence', 
                         'risk_tolerance', 'adaptation_rate', 'habit_strength',
                         'cooperation_tendency', 'fairness_concern']:
            value = getattr(self, attr_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{attr_name} must be in [0,1], got {value}")

@dataclass
class BehaviorState:
    """行为状态"""
    current_mood: float = 0.5              # 当前情绪 [-1,1]
    stress_level: float = 0.3              # 压力水平 [0,1]
    fatigue_level: float = 0.2             # 疲劳程度 [0,1]
    confidence: float = 0.7                # 信心水平 [0,1]
    trust_levels: Dict[str, float] = None  # 对其他角色的信任度
    reputation: float = 0.8                # 声誉值 [0,1]
    experience_count: int = 0              # 经验计数
    last_rewards: List[float] = None       # 最近奖励历史
    
    def __post_init__(self):
        if self.trust_levels is None:
            self.trust_levels = {
                'doctors': 0.8, 'interns': 0.7, 'patients': 0.9,
                'accountants': 0.6, 'government': 0.5
            }
        if self.last_rewards is None:
            self.last_rewards = []

class BaseBehaviorModel(ABC):
    """行为模型基类"""
    
    def __init__(self, behavior_type: BehaviorType, parameters: BehaviorParameters):
        self.behavior_type = behavior_type
        self.parameters = parameters
        self.state = BehaviorState()
        self.action_history: List[np.ndarray] = []
        self.decision_weights: Dict[str, float] = {}
        
    @abstractmethod
    def compute_action_probabilities(self, observation: np.ndarray, 
                                   available_actions: np.ndarray,
                                   context: Dict[str, Any]) -> np.ndarray:
        """计算行动概率分布"""
        pass
    
    @abstractmethod
    def update_behavior_state(self, observation: np.ndarray, action: np.ndarray,
                            reward: float, context: Dict[str, Any]):
        """更新行为状态"""
        pass
    
    def add_experience(self, reward: float):
        """添加经验"""
        self.state.last_rewards.append(reward)
        if len(self.state.last_rewards) > 20:  # 保持最近20个奖励
            self.state.last_rewards.pop(0)
        self.state.experience_count += 1
    
    def get_behavior_metrics(self) -> Dict[str, float]:
        """获取行为指标"""
        return {
            'mood': self.state.current_mood,
            'stress': self.state.stress_level,
            'confidence': self.state.confidence,
            'reputation': self.state.reputation,
            'experience': self.state.experience_count,
            'avg_reward': np.mean(self.state.last_rewards) if self.state.last_rewards else 0.0
        }

class RationalBehaviorModel(BaseBehaviorModel):
    """理性行为模型 - 基于效用最大化"""
    
    def __init__(self, parameters: BehaviorParameters):
        super().__init__(BehaviorType.RATIONAL, parameters)
        self.utility_function = self._build_utility_function()
        self.decision_weights = {
            'immediate_reward': 0.4,
            'future_reward': 0.4,
            'risk_adjustment': 0.2
        }
    
    def _build_utility_function(self) -> Callable:
        """构建效用函数"""
        def utility(action: np.ndarray, context: Dict[str, Any]) -> float:
            # 基本收益
            base_utility = np.sum(action * context.get('reward_weights', np.ones_like(action)))
            
            # 风险调整
            risk_penalty = self.parameters.risk_tolerance * np.var(action)
            
            # 社会影响
            social_bonus = self._compute_social_utility(action, context)
            
            return base_utility - risk_penalty + social_bonus
        
        return utility
    
    def _compute_social_utility(self, action: np.ndarray, context: Dict[str, Any]) -> float:
        """计算社会效用"""
        if 'other_actions' not in context:
            return 0.0
            
        cooperation_bonus = 0.0
        for role, other_action in context['other_actions'].items():
            if role in self.state.trust_levels:
                # 基于信任度的合作奖励
                cooperation_score = np.dot(action, other_action) / (np.linalg.norm(action) * np.linalg.norm(other_action) + 1e-8)
                cooperation_bonus += self.state.trust_levels[role] * cooperation_score * 0.1
                
        return cooperation_bonus
    
    def compute_action_probabilities(self, observation: np.ndarray,
                                   available_actions: np.ndarray,
                                   context: Dict[str, Any]) -> np.ndarray:
        """基于理性效用计算行动概率"""
        utilities = []
        
        for action in available_actions:
            utility = self.utility_function(action, context)
            utilities.append(utility)
        
        utilities = np.array(utilities)
        
        # 理性程度决定选择的确定性
        if self.parameters.rationality_level > 0.9:
            # 高理性：选择最优行动
            probabilities = np.zeros_like(utilities)
            probabilities[np.argmax(utilities)] = 1.0
        else:
            # 有限理性：softmax选择
            temperature = 1.0 / (self.parameters.rationality_level + 0.1)
            probabilities = np.exp(utilities / temperature)
            probabilities /= np.sum(probabilities)
        
        return probabilities
    
    def update_behavior_state(self, observation: np.ndarray, action: np.ndarray,
                            reward: float, context: Dict[str, Any]):
        """更新理性行为状态"""
        self.add_experience(reward)
        
        # 更新信心水平
        if reward > 0:
            self.state.confidence = min(1.0, self.state.confidence + 0.02)
        else:
            self.state.confidence = max(0.0, self.state.confidence - 0.01)
        
        # 更新压力水平（基于决策复杂度）
        decision_complexity = np.linalg.norm(action)
        self.state.stress_level = min(1.0, 0.1 * decision_complexity + 0.9 * self.state.stress_level)

class BoundedRationalBehaviorModel(BaseBehaviorModel):
    """有限理性行为模型 - 简化启发式决策"""
    
    def __init__(self, parameters: BehaviorParameters):
        super().__init__(BehaviorType.BOUNDED_RATIONAL, parameters)
        self.heuristics = self._initialize_heuristics()
        self.cognitive_load_threshold = 0.7
        
    def _initialize_heuristics(self) -> List[Callable]:
        """初始化启发式规则"""
        return [
            self._satisficing_heuristic,
            self._imitation_heuristic,
            self._threshold_heuristic,
            self._default_heuristic
        ]
    
    def _satisficing_heuristic(self, observation: np.ndarray, 
                             available_actions: np.ndarray,
                             context: Dict[str, Any]) -> Optional[np.ndarray]:
        """满足化启发式：选择第一个满足条件的行动"""
        threshold = context.get('satisficing_threshold', 0.6)
        
        for action in available_actions:
            if self._evaluate_action_quality(action, observation) >= threshold:
                return action
        return None
    
    def _imitation_heuristic(self, observation: np.ndarray,
                           available_actions: np.ndarray,
                           context: Dict[str, Any]) -> Optional[np.ndarray]:
        """模仿启发式：模仿成功的其他智能体"""
        if 'successful_actions' not in context:
            return None
            
        # 选择最成功的行动进行模仿
        best_action = None
        best_performance = -float('inf')
        
        for role, (action, performance) in context['successful_actions'].items():
            if performance > best_performance and role in self.state.trust_levels:
                trust_weight = self.state.trust_levels[role]
                if trust_weight > 0.5:  # 只模仿信任的角色
                    best_action = action
                    best_performance = performance
        
        return best_action
    
    def _threshold_heuristic(self, observation: np.ndarray,
                           available_actions: np.ndarray,
                           context: Dict[str, Any]) -> Optional[np.ndarray]:
        """阈值启发式：基于简单阈值规则"""
        # 如果某个观测值超过阈值，采取对应行动
        for i, obs_value in enumerate(observation):
            if obs_value > 0.8:  # 高阈值
                action = np.zeros(len(available_actions[0]))
                if i < len(action):
                    action[i] = 0.8
                return action
            elif obs_value < 0.2:  # 低阈值
                action = np.zeros(len(available_actions[0]))
                if i < len(action):
                    action[i] = -0.5
                return action
        return None
    
    def _default_heuristic(self, observation: np.ndarray,
                         available_actions: np.ndarray,
                         context: Dict[str, Any]) -> np.ndarray:
        """默认启发式：保守的平均行动"""
        return np.mean(available_actions, axis=0) * 0.5
    
    def _evaluate_action_quality(self, action: np.ndarray, observation: np.ndarray) -> float:
        """评估行动质量（简化版本）"""
        # 简单的质量评估：行动与观测的匹配度
        quality = 1.0 - np.mean(np.abs(action - observation[:len(action)]))
        return np.clip(quality, 0, 1)
    
    def compute_action_probabilities(self, observation: np.ndarray,
                                   available_actions: np.ndarray,
                                   context: Dict[str, Any]) -> np.ndarray:
        """使用启发式计算行动概率"""
        # 计算认知负荷
        cognitive_load = self._compute_cognitive_load(observation, context)
        
        if cognitive_load > self.cognitive_load_threshold:
            # 高认知负荷：使用简单启发式
            selected_action = None
            for heuristic in self.heuristics:
                selected_action = heuristic(observation, available_actions, context)
                if selected_action is not None:
                    break
            
            if selected_action is not None:
                # 找到最接近的可用行动
                distances = [np.linalg.norm(action - selected_action) for action in available_actions]
                best_idx = np.argmin(distances)
                probabilities = np.zeros(len(available_actions))
                probabilities[best_idx] = 1.0
                return probabilities
        
        # 低认知负荷：使用更精细的计算
        qualities = [self._evaluate_action_quality(action, observation) for action in available_actions]
        probabilities = np.array(qualities)
        probabilities = probabilities / (np.sum(probabilities) + 1e-8)
        
        return probabilities
    
    def _compute_cognitive_load(self, observation: np.ndarray, context: Dict[str, Any]) -> float:
        """计算认知负荷"""
        # 基于观测复杂度、时间压力、压力水平
        complexity = np.var(observation)
        time_pressure = context.get('time_pressure', 0.0)
        
        cognitive_load = 0.4 * complexity + 0.3 * time_pressure + 0.3 * self.state.stress_level
        return np.clip(cognitive_load, 0, 1)
    
    def update_behavior_state(self, observation: np.ndarray, action: np.ndarray,
                            reward: float, context: Dict[str, Any]):
        """更新有限理性行为状态"""
        self.add_experience(reward)
        
        # 基于奖励调整启发式偏好
        if reward > 0:
            self.state.confidence = min(1.0, self.state.confidence + 0.01)
        else:
            self.state.confidence = max(0.0, self.state.confidence - 0.02)
        
        # 认知负荷影响压力
        cognitive_load = self._compute_cognitive_load(observation, context)
        self.state.stress_level = 0.8 * self.state.stress_level + 0.2 * cognitive_load

class EmotionalBehaviorModel(BaseBehaviorModel):
    """情感驱动行为模型"""
    
    def __init__(self, parameters: BehaviorParameters):
        super().__init__(BehaviorType.EMOTIONAL, parameters)
        self.emotion_dimensions = {
            'valence': 0.0,    # 效价 [-1,1] (负面到正面)
            'arousal': 0.0,    # 唤醒度 [-1,1] (平静到激动)
            'dominance': 0.0   # 控制感 [-1,1] (无力到有力)
        }
        self.emotion_decay_rate = 0.1
        
    def _update_emotions(self, observation: np.ndarray, reward: float, context: Dict[str, Any]):
        """更新情感状态"""
        # 基于奖励更新效价
        if reward > 0:
            self.emotion_dimensions['valence'] = min(1.0, 
                self.emotion_dimensions['valence'] + 0.2 * reward)
        else:
            self.emotion_dimensions['valence'] = max(-1.0, 
                self.emotion_dimensions['valence'] + 0.3 * reward)
        
        # 基于观测变化更新唤醒度
        if hasattr(self, 'last_observation'):
            observation_change = np.linalg.norm(observation - self.last_observation)
            self.emotion_dimensions['arousal'] = np.clip(
                self.emotion_dimensions['arousal'] + 0.1 * observation_change - 0.05, -1, 1)
        
        # 基于行动成功率更新控制感
        if len(self.state.last_rewards) > 0:
            success_rate = np.mean([r > 0 for r in self.state.last_rewards[-5:]])
            target_dominance = 2 * success_rate - 1  # 映射到[-1,1]
            self.emotion_dimensions['dominance'] = 0.9 * self.emotion_dimensions['dominance'] + 0.1 * target_dominance
        
        # 情感衰减
        for emotion in self.emotion_dimensions:
            self.emotion_dimensions[emotion] *= (1 - self.emotion_decay_rate)
        
        self.last_observation = observation.copy()
    
    def _compute_emotion_influence(self, action: np.ndarray) -> np.ndarray:
        """计算情感对行动的影响"""
        valence = self.emotion_dimensions['valence']
        arousal = self.emotion_dimensions['arousal']
        dominance = self.emotion_dimensions['dominance']
        
        # 情感影响因子
        emotion_factor = np.zeros_like(action)
        
        # 正面情感增强行动强度
        if valence > 0:
            emotion_factor += valence * 0.2
        else:
            emotion_factor += valence * 0.1  # 负面情感减弱行动
        
        # 高唤醒增加行动变异性
        if arousal > 0:
            noise = np.random.normal(0, arousal * 0.1, action.shape)
            emotion_factor += noise
        
        # 低控制感导致保守行动
        if dominance < 0:
            emotion_factor *= (1 + dominance * 0.3)
        
        return emotion_factor
    
    def compute_action_probabilities(self, observation: np.ndarray,
                                   available_actions: np.ndarray,
                                   context: Dict[str, Any]) -> np.ndarray:
        """基于情感状态计算行动概率"""
        probabilities = np.zeros(len(available_actions))
        
        for i, action in enumerate(available_actions):
            # 基础效用
            base_utility = self._compute_base_utility(action, observation)
            
            # 情感影响
            emotion_influence = np.sum(self._compute_emotion_influence(action))
            
            # 情感权重调整
            total_utility = ((1 - self.parameters.emotional_weight) * base_utility + 
                           self.parameters.emotional_weight * emotion_influence)
            
            probabilities[i] = total_utility
        
        # 情感状态影响决策确定性
        arousal = abs(self.emotion_dimensions['arousal'])
        temperature = 1.0 + arousal  # 高唤醒降低决策确定性
        
        probabilities = np.exp(probabilities / temperature)
        probabilities = probabilities / (np.sum(probabilities) + 1e-8)
        
        return probabilities
    
    def _compute_base_utility(self, action: np.ndarray, observation: np.ndarray) -> float:
        """计算基础效用"""
        # 简单的观测-行动匹配
        return np.dot(action, observation[:len(action)]) / (np.linalg.norm(action) + 1e-8)
    
    def update_behavior_state(self, observation: np.ndarray, action: np.ndarray,
                            reward: float, context: Dict[str, Any]):
        """更新情感行为状态"""
        self.add_experience(reward)
        self._update_emotions(observation, reward, context)
        
        # 情感影响整体状态
        valence = self.emotion_dimensions['valence']
        self.state.current_mood = valence
        self.state.stress_level = max(0, 0.5 - valence * 0.3 + abs(self.emotion_dimensions['arousal']) * 0.2)

class SocialBehaviorModel(BaseBehaviorModel):
    """社会性行为模型 - 强调合作和社会影响"""
    
    def __init__(self, parameters: BehaviorParameters):
        super().__init__(BehaviorType.SOCIAL, parameters)
        self.social_norms = self._initialize_social_norms()
        self.reputation_weight = 0.3
        self.conformity_tendency = 0.6
        
    def _initialize_social_norms(self) -> Dict[str, float]:
        """初始化社会规范"""
        return {
            'cooperation': 0.8,        # 合作规范强度
            'fairness': 0.7,          # 公平规范强度
            'reciprocity': 0.9,       # 互惠规范强度
            'transparency': 0.6,      # 透明度规范强度
            'professional_ethics': 0.9 # 职业伦理规范强度
        }
    
    def _compute_social_pressure(self, action: np.ndarray, context: Dict[str, Any]) -> float:
        """计算社会压力"""
        if 'group_consensus' not in context:
            return 0.0
        
        group_action = context['group_consensus']
        
        # 行动与群体共识的偏差
        deviation = np.linalg.norm(action - group_action)
        
        # 社会压力强度
        pressure = self.conformity_tendency * deviation
        
        return pressure
    
    def _compute_reputation_impact(self, action: np.ndarray, context: Dict[str, Any]) -> float:
        """计算声誉影响"""
        reputation_change = 0.0
        
        # 合作行为提升声誉
        cooperation_score = self._evaluate_cooperation(action, context)
        reputation_change += cooperation_score * 0.1
        
        # 公平行为提升声誉
        fairness_score = self._evaluate_fairness(action, context)
        reputation_change += fairness_score * 0.1
        
        # 伦理合规性影响声誉
        ethics_score = context.get('ethics_compliance', 0.8)
        reputation_change += (ethics_score - 0.5) * 0.05
        
        return reputation_change
    
    def _evaluate_cooperation(self, action: np.ndarray, context: Dict[str, Any]) -> float:
        """评估合作程度"""
        if 'other_actions' not in context:
            return 0.5
        
        cooperation_scores = []
        for role, other_action in context['other_actions'].items():
            if role in self.state.trust_levels:
                # 计算行动相似性作为合作指标
                similarity = np.dot(action, other_action) / (np.linalg.norm(action) * np.linalg.norm(other_action) + 1e-8)
                cooperation_scores.append(max(0, similarity))
        
        return np.mean(cooperation_scores) if cooperation_scores else 0.5
    
    def _evaluate_fairness(self, action: np.ndarray, context: Dict[str, Any]) -> float:
        """评估公平性"""
        if 'resource_distribution' not in context:
            return 0.5
        
        # 评估行动是否促进资源公平分配
        current_distribution = context['resource_distribution']
        action_impact = context.get('action_impact_on_distribution', np.zeros_like(current_distribution))
        
        # 计算分配的基尼系数变化
        before_gini = self._compute_gini(current_distribution)
        after_gini = self._compute_gini(current_distribution + action_impact)
        
        # 减少不平等的行动被认为更公平
        fairness_score = max(0, before_gini - after_gini + 0.5)
        
        return np.clip(fairness_score, 0, 1)
    
    def _compute_gini(self, distribution: np.ndarray) -> float:
        """计算基尼系数"""
        if len(distribution) == 0:
            return 0
        
        sorted_dist = np.sort(distribution)
        n = len(sorted_dist)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * sorted_dist)) / (n * np.sum(sorted_dist)) - (n + 1) / n
        return gini
    
    def compute_action_probabilities(self, observation: np.ndarray,
                                   available_actions: np.ndarray,
                                   context: Dict[str, Any]) -> np.ndarray:
        """基于社会因素计算行动概率"""
        utilities = []
        
        for action in available_actions:
            # 基础效用
            base_utility = np.sum(action * observation[:len(action)])
            
            # 社会压力惩罚
            social_pressure = self._compute_social_pressure(action, context)
            
            # 声誉影响
            reputation_impact = self._compute_reputation_impact(action, context)
            
            # 社会规范符合度
            norm_compliance = self._evaluate_norm_compliance(action, context)
            
            # 综合效用
            total_utility = (base_utility - 
                           self.parameters.social_influence * social_pressure +
                           self.reputation_weight * reputation_impact +
                           0.2 * norm_compliance)
            
            utilities.append(total_utility)
        
        utilities = np.array(utilities)
        
        # 社会性行为倾向于更保守的选择
        probabilities = np.exp(utilities / 0.5)  # 较低温度
        probabilities = probabilities / (np.sum(probabilities) + 1e-8)
        
        return probabilities
    
    def _evaluate_norm_compliance(self, action: np.ndarray, context: Dict[str, Any]) -> float:
        """评估社会规范符合度"""
        compliance_score = 0.0
        
        # 合作规范
        cooperation_score = self._evaluate_cooperation(action, context)
        compliance_score += self.social_norms['cooperation'] * cooperation_score
        
        # 公平规范
        fairness_score = self._evaluate_fairness(action, context)
        compliance_score += self.social_norms['fairness'] * fairness_score
        
        # 职业伦理规范
        ethics_score = context.get('ethics_compliance', 0.8)
        compliance_score += self.social_norms['professional_ethics'] * ethics_score
        
        return compliance_score / 3.0  # 归一化
    
    def update_behavior_state(self, observation: np.ndarray, action: np.ndarray,
                            reward: float, context: Dict[str, Any]):
        """更新社会行为状态"""
        self.add_experience(reward)
        
        # 更新声誉
        reputation_change = self._compute_reputation_impact(action, context)
        self.state.reputation = np.clip(self.state.reputation + reputation_change, 0, 1)
        
        # 更新信任水平
        if 'interaction_outcomes' in context:
            for role, outcome in context['interaction_outcomes'].items():
                if role in self.state.trust_levels:
                    if outcome > 0:  # 正面互动
                        self.state.trust_levels[role] = min(1.0, self.state.trust_levels[role] + 0.02)
                    else:  # 负面互动
                        self.state.trust_levels[role] = max(0.0, self.state.trust_levels[role] - 0.05)

class AdaptiveBehaviorModel(BaseBehaviorModel):
    """适应性行为模型 - 基于学习和环境变化调整行为"""
    
    def __init__(self, parameters: BehaviorParameters):
        super().__init__(BehaviorType.ADAPTIVE, parameters)
        self.learning_rate = parameters.adaptation_rate
        self.strategy_weights = np.ones(4) / 4  # 四种策略的权重
        self.strategy_performance = np.zeros(4)  # 策略性能历史
        self.adaptation_memory = []  # 适应历史
        
    def _strategy_exploration(self, observation: np.ndarray) -> np.ndarray:
        """探索策略：尝试新行动"""
        action = np.random.uniform(-1, 1, len(observation))
        return action
    
    def _strategy_exploitation(self, observation: np.ndarray) -> np.ndarray:
        """利用策略：使用已知最优行动"""
        if len(self.action_history) > 0:
            # 使用历史最佳行动
            best_idx = np.argmax(self.state.last_rewards) if self.state.last_rewards else 0
            best_idx = min(best_idx, len(self.action_history) - 1)
            return self.action_history[best_idx]
        else:
            return np.zeros(len(observation))
    
    def _strategy_imitation(self, observation: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """模仿策略：学习他人行为"""
        if 'other_actions' in context and context['other_actions']:
            # 选择最成功的角色进行模仿
            best_role = max(context['other_actions'].keys(), 
                          key=lambda r: self.state.trust_levels.get(r, 0))
            return context['other_actions'][best_role]
        else:
            return self._strategy_exploitation(observation)
    
    def _strategy_gradient_ascent(self, observation: np.ndarray) -> np.ndarray:
        """梯度上升策略：基于奖励梯度调整"""
        if len(self.action_history) >= 2 and len(self.state.last_rewards) >= 2:
            # 计算奖励梯度
            action_diff = self.action_history[-1] - self.action_history[-2]
            reward_diff = self.state.last_rewards[-1] - self.state.last_rewards[-2]
            
            # 在奖励提升方向上调整
            if np.linalg.norm(action_diff) > 1e-6:
                gradient = reward_diff * action_diff / np.linalg.norm(action_diff)
                base_action = self.action_history[-1]
                return base_action + self.learning_rate * gradient
        
        return self._strategy_exploitation(observation)
    
    def _update_strategy_weights(self, reward: float, used_strategy: int):
        """更新策略权重"""
        # 更新策略性能
        self.strategy_performance[used_strategy] = (0.9 * self.strategy_performance[used_strategy] + 
                                                   0.1 * reward)
        
        # 基于性能更新权重（softmax）
        exp_performance = np.exp(self.strategy_performance)
        self.strategy_weights = exp_performance / np.sum(exp_performance)
    
    def compute_action_probabilities(self, observation: np.ndarray,
                                   available_actions: np.ndarray,
                                   context: Dict[str, Any]) -> np.ndarray:
        """基于适应性策略计算行动概率"""
        # 选择策略
        strategy_probs = self.strategy_weights
        selected_strategy = np.random.choice(4, p=strategy_probs)
        
        # 执行选择的策略
        strategies = [
            self._strategy_exploration,
            self._strategy_exploitation,
            self._strategy_imitation,
            self._strategy_gradient_ascent
        ]
        
        if selected_strategy == 2:  # 模仿策略需要context
            target_action = strategies[selected_strategy](observation, context)
        else:
            target_action = strategies[selected_strategy](observation)
        
        # 找到最接近目标行动的可用行动
        distances = [np.linalg.norm(action - target_action) for action in available_actions]
        best_idx = np.argmin(distances)
        
        # 存储使用的策略以便后续更新
        self._last_used_strategy = selected_strategy
        
        # 创建概率分布（以最佳行动为中心的概率分布）
        probabilities = np.exp(-np.array(distances))
        probabilities = probabilities / (np.sum(probabilities) + 1e-8)
        
        return probabilities
    
    def update_behavior_state(self, observation: np.ndarray, action: np.ndarray,
                            reward: float, context: Dict[str, Any]):
        """更新适应性行为状态"""
        self.add_experience(reward)
        
        # 更新策略权重
        if hasattr(self, '_last_used_strategy'):
            self._update_strategy_weights(reward, self._last_used_strategy)
        
        # 记录适应历史
        adaptation_record = {
            'observation': observation.copy(),
            'action': action.copy(),
            'reward': reward,
            'strategy_weights': self.strategy_weights.copy()
        }
        self.adaptation_memory.append(adaptation_record)
        
        # 保持适应历史在合理大小
        if len(self.adaptation_memory) > 100:
            self.adaptation_memory.pop(0)

class BehaviorModelFactory:
    """行为模型工厂"""
    
    @staticmethod
    def create_behavior_model(behavior_type: BehaviorType, 
                            parameters: BehaviorParameters) -> BaseBehaviorModel:
        """创建行为模型"""
        if behavior_type == BehaviorType.RATIONAL:
            return RationalBehaviorModel(parameters)
        elif behavior_type == BehaviorType.BOUNDED_RATIONAL:
            return BoundedRationalBehaviorModel(parameters)
        elif behavior_type == BehaviorType.EMOTIONAL:
            return EmotionalBehaviorModel(parameters)
        elif behavior_type == BehaviorType.SOCIAL:
            return SocialBehaviorModel(parameters)
        elif behavior_type == BehaviorType.ADAPTIVE:
            return AdaptiveBehaviorModel(parameters)
        else:
            raise ValueError(f"Unknown behavior type: {behavior_type}")
    
    @staticmethod
    def create_role_specific_model(role: str) -> BaseBehaviorModel:
        """为特定角色创建合适的行为模型"""
        role_configs = {
            'doctors': {
                'behavior_type': BehaviorType.RATIONAL,
                'parameters': BehaviorParameters(
                    rationality_level=0.85,
                    emotional_weight=0.15,
                    social_influence=0.4,
                    risk_tolerance=0.3,  # 医生倾向于风险厌恶
                    cooperation_tendency=0.8,
                    fairness_concern=0.7
                )
            },
            'interns': {
                'behavior_type': BehaviorType.ADAPTIVE,
                'parameters': BehaviorParameters(
                    rationality_level=0.7,
                    emotional_weight=0.3,
                    social_influence=0.6,  # 更容易受社会影响
                    risk_tolerance=0.4,
                    adaptation_rate=0.2,  # 高适应率
                    cooperation_tendency=0.9
                )
            },
            'patients': {
                'behavior_type': BehaviorType.EMOTIONAL,
                'parameters': BehaviorParameters(
                    rationality_level=0.6,
                    emotional_weight=0.5,  # 高情感权重
                    social_influence=0.5,
                    risk_tolerance=0.2,  # 高风险厌恶
                    cooperation_tendency=0.7,
                    fairness_concern=0.8
                )
            },
            'accountants': {
                'behavior_type': BehaviorType.RATIONAL,
                'parameters': BehaviorParameters(
                    rationality_level=0.9,  # 高理性
                    emotional_weight=0.1,
                    social_influence=0.3,
                    risk_tolerance=0.5,
                    cooperation_tendency=0.6,
                    fairness_concern=0.6
                )
            },
            'government': {
                'behavior_type': BehaviorType.SOCIAL,
                'parameters': BehaviorParameters(
                    rationality_level=0.8,
                    emotional_weight=0.2,
                    social_influence=0.7,  # 高社会影响
                    risk_tolerance=0.4,
                    cooperation_tendency=0.7,
                    fairness_concern=0.9   # 高公平关注
                )
            }
        }
        
        if role not in role_configs:
            # 默认使用有限理性模型
            return BoundedRationalBehaviorModel(BehaviorParameters())
        
        config = role_configs[role]
        return BehaviorModelFactory.create_behavior_model(
            config['behavior_type'], 
            config['parameters']
        )

class BehaviorModelManager:
    """行为模型管理器"""
    
    def __init__(self):
        self.models: Dict[str, BaseBehaviorModel] = {}
        self.interaction_history: List[Dict] = []
        
    def register_model(self, role: str, model: BaseBehaviorModel):
        """注册行为模型"""
        self.models[role] = model
        print(f"Registered behavior model for role: {role}")
    
    def get_model(self, role: str) -> Optional[BaseBehaviorModel]:
        """获取行为模型"""
        return self.models.get(role)
    
    def create_all_role_models(self):
        """为所有角色创建行为模型"""
        roles = ['doctors', 'interns', 'patients', 'accountants', 'government']
        for role in roles:
            model = BehaviorModelFactory.create_role_specific_model(role)
            self.register_model(role, model)
    
    def update_all_models(self, observations: Dict[str, np.ndarray],
                         actions: Dict[str, np.ndarray],
                         rewards: Dict[str, float],
                         context: Dict[str, Any]):
        """更新所有模型的行为状态"""
        for role in self.models:
            if role in observations and role in actions and role in rewards:
                self.models[role].update_behavior_state(
                    observations[role], actions[role], rewards[role], context
                )
        
        # 记录交互历史
        interaction_record = {
            'observations': copy.deepcopy(observations),
            'actions': copy.deepcopy(actions),
            'rewards': copy.deepcopy(rewards),
            'context': copy.deepcopy(context),
            'timestamp': len(self.interaction_history)
        }
        self.interaction_history.append(interaction_record)
    
    def get_collective_behavior_metrics(self) -> Dict[str, Any]:
        """获取集体行为指标"""
        if not self.models:
            return {}
        
        metrics = {}
        for role, model in self.models.items():
            metrics[role] = model.get_behavior_metrics()
        
        # 计算整体指标
        all_moods = [metrics[role]['mood'] for role in metrics]
        all_stress = [metrics[role]['stress'] for role in metrics]
        all_confidence = [metrics[role]['confidence'] for role in metrics]
        
        metrics['collective'] = {
            'avg_mood': np.mean(all_moods),
            'avg_stress': np.mean(all_stress),
            'avg_confidence': np.mean(all_confidence),
            'mood_variance': np.var(all_moods),
            'interaction_count': len(self.interaction_history)
        }
        
        return metrics
    
    def analyze_behavioral_patterns(self) -> Dict[str, Any]:
        """分析行为模式"""
        if len(self.interaction_history) < 10:
            return {"message": "Insufficient data for pattern analysis"}
        
        patterns = {}
        
        # 分析每个角色的行为趋势
        for role in self.models:
            role_rewards = []
            role_actions = []
            
            for record in self.interaction_history:
                if role in record['rewards']:
                    role_rewards.append(record['rewards'][role])
                if role in record['actions']:
                    role_actions.append(record['actions'][role])
            
            if role_rewards:
                patterns[role] = {
                    'avg_reward_trend': np.mean(role_rewards[-10:]) - np.mean(role_rewards[:10]) if len(role_rewards) >= 20 else 0,
                    'reward_stability': 1.0 / (np.std(role_rewards) + 1e-6),
                    'action_consistency': self._compute_action_consistency(role_actions)
                }
        
        return patterns
    
    def _compute_action_consistency(self, actions: List[np.ndarray]) -> float:
        """计算行动一致性"""
        if len(actions) < 2:
            return 0.0
        
        consistencies = []
        for i in range(1, len(actions)):
            consistency = 1.0 - np.linalg.norm(actions[i] - actions[i-1]) / 2.0
            consistencies.append(max(0, consistency))
        
        return np.mean(consistencies)
