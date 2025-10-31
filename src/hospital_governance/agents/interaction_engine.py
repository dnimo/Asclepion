
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import deque
import random
from dataclasses import dataclass
from ..control.role_specific_reward_controllers import (
    DoctorRewardController, InternRewardController, PatientRewardController, AccountantRewardController, GovernmentRewardController
)
from ..control.reward_based_controller import RewardBasedController, RewardControlConfig
from ..core.kallipolis_mathematical_core import SystemState

@dataclass
class CrisisScenario:
    """危机场景"""
    name: str
    type: str  # "pandemic", "funding_cut", "financial_crisis"
    severity: float  # 严重程度 0-1
    duration: int    # 持续时间
    affected_metrics: List[str]  # 影响的指标

class KallipolisInteractionEngine:
    """Kallipolis医疗共和国交互引擎"""
    def __init__(self, role_manager, parliament, system_state: SystemState, holy_code_manager: Any = None):
        self.role_manager = role_manager
        self.parliament = parliament
        self.holy_code_manager = holy_code_manager
        self.system_state = system_state
        # 危机管理
        self.active_crises: List[CrisisScenario] = []
        self.crisis_history: List[Dict] = []
        # 性能指标
        self.performance_metrics = {
            'time_to_consensus': [],
            'disturbance_adaptation_time': [],
            'rule_update_success_rate': [],
            'resource_utility_recovery_time': []
        }
        # 经验回放
        self.experience_replay = ExperienceReplay(10000)
        self.time_step = 0
    
    def step(self, training: bool = False) -> Dict[str, Any]:
        """执行一个时间步"""
        self.time_step += 1
        
        # 1. 观察环境
        observations = self._get_observations()
        
        # 2. 生成提案和行动
        proposals = self._collect_proposals(observations)
        actions = self._get_actions(observations, training)
        
        # 3. 议会决策
        parliament_decisions = self._process_parliament_decisions(proposals)
        
        # 4. 执行行动并更新状态
        rewards = self._execute_actions(actions, parliament_decisions)
        
        # 5. 处理危机
        self._handle_crises()
        
        # 6. 更新系统状态
        next_observations = self._get_observations()
        
        # 7. 收集经验
        self._collect_experiences(observations, actions, rewards, next_observations)
        
        # 8. 计算性能指标
        metrics = self._calculate_performance_metrics()
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'parliament_decisions': parliament_decisions,
            'metrics': metrics,
            'system_state': self.system_state.copy()
        }
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取各角色的观察"""
        observations = {}
        # 构建系统状态向量（从 SystemState 对象获取）
        # 这里假定 SystemState 有相关属性，需与原 system_state dict 映射
        system_state_vector = np.array([
            getattr(self.system_state, 'resource_utilization', 0.7),
            getattr(self.system_state, 'financial_health', 0.8),
            getattr(self.system_state, 'patient_satisfaction', 0.75),
            getattr(self.system_state, 'medical_quality', 0.85),
            getattr(self.system_state, 'education_effectiveness', 0.7),
            getattr(self.system_state, 'operational_efficiency', 0.75),
            len(self.active_crises) / 5.0,  # 危机数量归一化
            max([c.severity for c in self.active_crises]) if self.active_crises else 0.0
        ])
        
        # 为每个角色提供定制化的观察
        for role, agent in self.role_manager.agents.items():
            if role == 'doctors':
                # 医生关注医疗质量和资源
                role_obs = system_state_vector[[3, 0, 5]]  # 质量、资源、效率
            elif role == 'interns':
                # 实习医生关注教育和资源
                role_obs = system_state_vector[[4, 0, 1]]  # 教育、资源、财务
            elif role == 'accountants':
                # 会计关注财务和效率
                role_obs = system_state_vector[[1, 5, 0]]  # 财务、效率、资源
            elif role == 'patients':
                # 患者代表关注满意度和质量
                role_obs = system_state_vector[[2, 3, 6]]  # 满意度、质量、危机
            elif role == 'government':
                # 政府关注系统和监管
                role_obs = system_state_vector[[0, 1, 7, 2]]  # 资源、财务、危机、满意度
            else:
                role_obs = system_state_vector
            
            # 添加噪声和角色特定信息
            noise = np.random.normal(0, 0.01, len(role_obs))
            observations[role] = role_obs + noise
        
        return observations
    
    def _collect_proposals(self, observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """收集各角色的提案"""
        proposals = {}
        
        for role, agent in self.role_manager.agents.items():
            if hasattr(agent, 'formulate_proposal'):
                context = self._determine_context()
                proposal = agent.formulate_proposal(observations[role], context)
                if proposal:
                    proposals[role] = proposal
        
        return proposals
    
    def _process_parliament_decisions(self, proposals: Dict[str, Any]) -> Dict[str, Any]:
        """处理议会决策"""
        decisions = {}
        
        if not self.parliament:
            # 如果没有议会，使用简化的投票机制
            for role, proposal in proposals.items():
                # 简单多数投票
                votes = 0
                total_voters = len(self.role_manager.agents)
                
                for voter_role, voter_agent in self.role_manager.agents.items():
                    if hasattr(voter_agent, 'vote_on_proposal'):
                        vote, rationale = voter_agent.vote_on_proposal(proposal, f"prop_{len(decisions)}")
                        if vote:
                            votes += 1
                    else:
                        # 默认投票逻辑
                        votes += 1 if np.random.random() > 0.4 else 0
                
                approved = votes > total_voters / 2
                approval_ratio = votes / total_voters
                
                decisions[f"prop_{len(decisions)}"] = {
                    'proposer': role,
                    'approved': approved,
                    'approval_ratio': approval_ratio,
                    'proposal': proposal
                }
            return decisions
        
        # 处理每个提案
        for role, proposal in proposals.items():
            if self.parliament:
                proposal_id = self.parliament.submit_proposal(proposal, role)
                
                # 收集投票
                for voter_role, voter_agent in self.role_manager.agents.items():
                    if hasattr(voter_agent, 'vote_on_proposal'):
                        vote, rationale = voter_agent.vote_on_proposal(proposal, proposal_id)
                        self.parliament.cast_vote(proposal_id, voter_role, vote, rationale)
                
                # 统计投票结果
                approved, approval_ratio = self.parliament.tally_votes(proposal_id)
                
                decisions[proposal_id] = {
                    'proposer': role,
                    'approved': approved,
                    'approval_ratio': approval_ratio,
                    'proposal': proposal
                }
                
                # 如果通过且是规则修改提案，更新神圣法典
                if approved and proposal.get('type') == 'rule_amendment':
                    if hasattr(self.parliament, 'update_holy_code'):
                        self.parliament.update_holy_code(proposal)
        
        return decisions
    
    def _get_actions(self, observations: Dict[str, np.ndarray], 
                    training: bool) -> Dict[str, np.ndarray]:
        """获取各角色的行动，集成HolyCode指导"""
        actions = {}
        for role, agent in self.role_manager.agents.items():
            holycode_guidance = None
            if self.holy_code_manager:
                decision_context = {
                    'agent_id': role,
                    'decision_type': 'routine_operation',
                    'current_state': self.system_state,
                    'proposed_action': {},
                    'state': self.system_state,
                }
                holycode_guidance = self.holy_code_manager.process_agent_decision_request(role, decision_context)
            action = agent.select_action(observations[role], holycode_guidance, training)
            actions[role] = action
        return actions
    
    def _execute_actions(self, actions: Dict[str, np.ndarray],
                        parliament_decisions: Dict[str, Any]) -> Dict[str, float]:
        """执行行动并计算奖励（调用 control 模块奖励控制器）"""
        rewards = {}
        # 初始化奖励控制器（可根据角色缓存或工厂模式优化）
        # 支持多种角色名映射（如 doctor -> doctors）
        role_map = {
            'doctor': 'doctors',
            'doctors': 'doctors',
            'intern': 'interns',
            'interns': 'interns',
            'patient': 'patients',
            'patients': 'patients',
            'accountant': 'accountants',
            'accountants': 'accountants',
            'government': 'government'
        }
        reward_controllers = {
            'doctors': DoctorRewardController,
            'interns': InternRewardController,
            'patients': PatientRewardController,
            'accountants': AccountantRewardController,
            'government': GovernmentRewardController
        }
        for role, action in actions.items():
            mapped_role = role_map.get(role, role)
            controller_cls = reward_controllers.get(mapped_role)
            if controller_cls:
                # 获取 agent 对象
                agent_obj = None
                # 支持多种角色名映射
                if hasattr(self.role_manager, 'get_agent'):
                    agent_obj = self.role_manager.get_agent(mapped_role)
                # 构造 config
                from hospital_governance.control.reward_based_controller import RewardControlConfig
                config_obj = RewardControlConfig(role=mapped_role)
                controller = controller_cls(config_obj, agent_obj)
                reward = controller.compute_reward(self.system_state, action, parliament_decisions)
            else:
                reward = 0.5  # 默认奖励
            rewards[role] = reward
            self._update_system_state(role, action, parliament_decisions)
        return rewards
    
    def _update_system_state(self, role: str, action: np.ndarray,
                           decisions: Dict[str, Any]):
        """更新系统状态（通过 SystemState 属性操作）"""
        action_impact = np.mean(action) if len(action) > 0 else 0.0
        # 这里假定 SystemState 有相关属性
        if role == 'doctors':
            if hasattr(self.system_state, 'medical_quality'):
                self.system_state.medical_quality += action_impact * 0.1
            if hasattr(self.system_state, 'operational_efficiency'):
                self.system_state.operational_efficiency += action_impact * 0.05
        elif role == 'interns':
            if hasattr(self.system_state, 'education_effectiveness'):
                self.system_state.education_effectiveness += action_impact * 0.1
            if hasattr(self.system_state, 'medical_quality'):
                self.system_state.medical_quality += action_impact * 0.05
        elif role == 'accountants':
            if hasattr(self.system_state, 'financial_health'):
                self.system_state.financial_health += action_impact * 0.1
            if hasattr(self.system_state, 'resource_utilization'):
                self.system_state.resource_utilization += action_impact * 0.08
        elif role == 'patients':
            if hasattr(self.system_state, 'patient_satisfaction'):
                self.system_state.patient_satisfaction += action_impact * 0.1
        elif role == 'government':
            if hasattr(self.system_state, 'operational_efficiency'):
                self.system_state.operational_efficiency += action_impact * 0.06
            if hasattr(self.system_state, 'financial_health'):
                self.system_state.financial_health += action_impact * 0.04
        # 确保状态值在合理范围内
        for attr in ['resource_utilization', 'financial_health', 'patient_satisfaction', 'medical_quality', 'education_effectiveness', 'operational_efficiency']:
            if hasattr(self.system_state, attr):
                val = getattr(self.system_state, attr)
                setattr(self.system_state, attr, float(np.clip(val, 0.1, 1.0)))
    
    def _handle_crises(self):
        """处理危机场景"""
        # 随机触发危机（简化）
        if random.random() < 0.05:  # 5%概率触发危机
            crisis_type = random.choice(['pandemic', 'funding_cut', 'financial_crisis'])
            crisis = CrisisScenario(
                name=f"{crisis_type}_{self.time_step}",
                type=crisis_type,
                severity=random.uniform(0.3, 0.8),
                duration=random.randint(5, 20),
                affected_metrics=self._get_affected_metrics(crisis_type)
            )
            self.active_crises.append(crisis)
            self.crisis_history.append({
                'crisis': crisis,
                'start_time': self.time_step,
                'handled': False
            })
        
        # 处理现有危机
        for crisis in self.active_crises[:]:
            # 应用危机影响
            for metric in crisis.affected_metrics:
                if metric in self.system_state:
                    impact = crisis.severity * 0.1
                    self.system_state[metric] -= impact
            
            crisis.duration -= 1
            if crisis.duration <= 0:
                self.active_crises.remove(crisis)
    
    def _get_affected_metrics(self, crisis_type: str) -> List[str]:
        """获取危机影响的指标"""
        if crisis_type == 'pandemic':
            return ['resource_utilization', 'medical_quality', 'operational_efficiency']
        elif crisis_type == 'funding_cut':
            return ['financial_health', 'education_effectiveness', 'resource_utilization']
        elif crisis_type == 'financial_crisis':
            return ['financial_health', 'patient_satisfaction', 'operational_efficiency']
        return []
    
    def _determine_context(self) -> str:
        """确定当前上下文"""
        if self.active_crises:
            return 'crisis'
        elif self.system_state['financial_health'] < 0.6:
            return 'financial'
        elif self.system_state['education_effectiveness'] < 0.65:
            return 'education'
        else:
            return 'normal'
    
    def _collect_experiences(self, observations: Dict[str, np.ndarray],
                           actions: Dict[str, np.ndarray],
                           rewards: Dict[str, float],
                           next_observations: Dict[str, np.ndarray]):
        """收集经验数据"""
        for role in observations.keys():
            experience = {
                'role': role,
                'state': observations[role],
                'action': actions[role],
                'reward': rewards[role],
                'next_state': next_observations[role],
                'done': False  # 简化，实际应该有终止条件
            }
            self.experience_replay.add(experience)
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        # 共识形成时间 (TTC)
        ttc = self._calculate_time_to_consensus()
        
        # 扰动适应时间 (DAT)
        dat = self._calculate_disturbance_adaptation_time()
        
        # 规则更新成功率 (RUSR)
        rusr = self._calculate_rule_update_success_rate()
        
        # 资源效用恢复时间 (RURT)
        rurt = self._calculate_resource_utility_recovery_time()
        
        metrics = {
            'TTC': ttc,
            'DAT': dat,
            'RUSR': rusr,
            'RURT': rurt
        }
        
        # 更新历史记录
        for key, value in metrics.items():
            self.performance_metrics[key.lower()].append(value)
        
        return metrics
    
    def _calculate_time_to_consensus(self) -> float:
        """计算共识形成时间"""
        if hasattr(self.parliament, 'get_consensus_metrics') and self.parliament:
            consensus_metrics = self.parliament.get_consensus_metrics()
            return 1.0 - consensus_metrics.get('consensus_convergence_rate', 0.5)
        else:
            # 简化的共识度量，基于最近的投票一致性
            return 0.7  # 默认值
    
    def _calculate_disturbance_adaptation_time(self) -> float:
        """计算扰动适应时间"""
        if not self.crisis_history:
            return 1.0
        
        recent_crises = [c for c in self.crisis_history if c.get('handled', False)]
        if not recent_crises:
            return 0.5
        
        # 简化实现
        return min(1.0, len(recent_crises) / 10.0)
    
    def _calculate_rule_update_success_rate(self) -> float:
        """计算规则更新成功率"""
        if hasattr(self.parliament, 'voting_history') and self.parliament:
            voting_history = self.parliament.voting_history
            if not voting_history:
                return 0.5
            
            rule_updates = [v for v in voting_history 
                           if v.get('content', {}).get('type') == 'rule_amendment']
            if not rule_updates:
                return 0.5
            
            successful_updates = len([u for u in rule_updates if u.get('approved', False)])
            return successful_updates / len(rule_updates)
        else:
            # 简化的成功率计算
            return 0.6  # 默认值
    
    def _calculate_resource_utility_recovery_time(self) -> float:
        """计算资源效用恢复时间"""
        # 基于资源利用率和财务健康度计算
        resource_util = self.system_state['resource_utilization']
        financial_health = self.system_state['financial_health']
        return (resource_util + financial_health) / 2
    
    def get_training_batch(self, batch_size: int = 32) -> List[Dict]:
        """获取训练批次"""
        return self.experience_replay.sample(batch_size)
    
    def add_crisis_scenario(self, crisis: CrisisScenario):
        """添加危机场景（用于测试）"""
        self.active_crises.append(crisis)
        self.crisis_history.append({
            'crisis': crisis,
            'start_time': self.time_step,
            'handled': False
        })

class ExperienceReplay:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Dict[str, Any]):
        """添加经验"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """采样批次"""
        if len(self.buffer) < batch_size:
            return random.sample(list(self.buffer), len(self.buffer))
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)