"""
多智能体交互引擎

这个模块提供了统一的多智能体交互接口，整合了行为模型、学习模型和LLM生成器。
解决了原有架构中的功能重叠和接口不一致问题。
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import copy

from .role_agents import RoleAgent, RoleManager
from .behavior_models import BehaviorModelManager
from .learning_models import MADDPGModel
from .llm_action_generator import LLMActionGenerator, LLMConfig


@dataclass
class InteractionConfig:
    """交互配置"""
    use_behavior_models: bool = True
    use_learning_models: bool = True
    use_llm_generation: bool = False
    conflict_resolution: str = "negotiation"  # "negotiation", "voting", "priority"
    cooperation_threshold: float = 0.6
    max_negotiation_rounds: int = 3


class MultiAgentInteractionEngine:
    """统一的多智能体交互引擎"""
    
    def __init__(self, role_manager: RoleManager, 
                 interaction_config: InteractionConfig = None,
                 llm_config: LLMConfig = None,
                 holy_code_manager: Optional[Any] = None):
        self.role_manager = role_manager
        self.config = interaction_config or InteractionConfig()
        self.holy_code_manager = holy_code_manager
        # 行为模型管理器
        if self.config.use_behavior_models:
            self.behavior_manager = BehaviorModelManager()
            self.behavior_manager.create_all_role_models()
            self._integrate_behavior_models()
        # 学习模型
        if self.config.use_learning_models:
            self.learning_model = None  # 将在需要时初始化
        # LLM生成器
        if self.config.use_llm_generation:
            self.llm_generator = LLMActionGenerator(llm_config)
        # 交互历史和指标
        self.interaction_history: List[Dict] = []
        self.coordination_metrics = {
            'conflict_count': 0,
            'cooperation_score': 0.0,
            'consensus_rate': 0.0,
            'negotiation_success_rate': 0.0
        }
        # 冲突解决状态
        self.active_conflicts: List[Dict] = []
        self.negotiation_history: List[Dict] = []
    
    def _integrate_behavior_models(self):
        """将行为模型集成到智能体中"""
        if not self.config.use_behavior_models:
            return
            
        for role, agent in self.role_manager.agents.items():
            behavior_model = self.behavior_manager.get_model(role)
            if behavior_model and hasattr(agent, 'set_behavior_model'):
                agent.set_behavior_model(behavior_model)
    
    def generate_actions(self, system_state: np.ndarray, 
                        context: Dict[str, Any],
                        training: bool = False) -> Dict[str, np.ndarray]:
        """生成所有智能体的协调行动"""
        
        # 1. 获取各智能体的观测
        observations = self._get_agent_observations(system_state, context)
        
        # 2. 生成初始行动
        initial_actions = self._generate_initial_actions(observations, context, training)
        
        # 3. 检测和解决冲突
        coordinated_actions = self._coordinate_actions(initial_actions, observations, context)
        
        # 4. 更新智能体状态
        self._update_agent_states(observations, coordinated_actions, context)
        
        # 5. 记录交互
        self._record_interaction(system_state, observations, coordinated_actions, context)
        
        return coordinated_actions
    
    def _get_agent_observations(self, system_state: np.ndarray, 
                               context: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """获取各智能体的观测"""
        observations = {}
        
        for role, agent in self.role_manager.agents.items():
            if hasattr(agent, 'observe'):
                # 使用智能体的observe方法
                obs = agent.observe(context.get('environment', {}))
            else:
                # 使用默认的观测提取
                obs = self._extract_role_observation(role, system_state)
            
            observations[role] = obs
        
        return observations
    
    def _extract_role_observation(self, role: str, system_state: np.ndarray) -> np.ndarray:
        """为特定角色提取观测"""
        if len(system_state) < 8:
            return system_state
        
        role_obs_map = {
            'doctors': system_state,  # 医生观测全部状态
            'interns': system_state[:8],  # 实习生观测前8个状态
            'patients': system_state[8:12] if len(system_state) > 12 else system_state[-4:],
            'accountants': system_state[4:8] if len(system_state) > 8 else system_state[:4],
            'government': system_state[[0, 1, 5, 12, 13]] if len(system_state) > 13 else system_state[:5]
        }
        
        return role_obs_map.get(role, system_state[:8])
    
    def _generate_initial_actions(self, observations: Dict[str, np.ndarray],
                                 context: Dict[str, Any],
                                 training: bool) -> Dict[str, np.ndarray]:
        """生成初始行动，集成HolyCode指导"""
        actions = {}
        for role, obs in observations.items():
            agent = self.role_manager.get_agent(role)
            if not agent:
                continue
            holycode_guidance = None
            if self.holy_code_manager:
                # 构造决策上下文
                decision_context = {
                    'agent_id': role,
                    'decision_type': context.get('decision_type', 'routine_operation'),
                    'current_state': context.get('environment', {}),
                    'proposed_action': {},
                    'state': context.get('environment', {}),
                }
                holycode_guidance = self.holy_code_manager.process_agent_decision_request(role, decision_context)
            # 选择行动生成方式
            if self.config.use_llm_generation and hasattr(self, 'llm_generator'):
                action = self.llm_generator.generate_action_sync(
                    role, obs, context.get('holy_code_state', {}), context
                )
            elif self.config.use_learning_models and hasattr(agent, 'select_action'):
                action = agent.select_action(obs, holycode_guidance, training)
            elif self.config.use_behavior_models and hasattr(agent, 'behavior_model') and agent.behavior_model:
                available_actions = self._get_available_actions(role)
                action_probs = agent.behavior_model.compute_action_probabilities(
                    obs, available_actions, context
                )
                selected_idx = np.argmax(action_probs)
                action = available_actions[selected_idx]
            else:
                action = self._generate_default_action(role, obs)
            actions[role] = action
        return actions
    
    def _get_available_actions(self, role: str) -> np.ndarray:
        """获取角色的可用行动集合"""
        action_dims = {
            'doctors': 4, 'interns': 3, 'patients': 3, 
            'accountants': 3, 'government': 3
        }
        
        dim = action_dims.get(role, 4)
        
        # 生成一组典型的可用行动
        actions = []
        for i in range(5):  # 5个可选行动
            action = np.random.uniform(-0.8, 0.8, dim)
            actions.append(action)
        
        return np.array(actions)
    
    def _generate_default_action(self, role: str, observation: np.ndarray) -> np.ndarray:
        """生成默认行动"""
        action_dims = {
            'doctors': 4, 'interns': 3, 'patients': 3, 
            'accountants': 3, 'government': 3
        }
        
        dim = action_dims.get(role, 4)
        
        # 基于观测的保守行动
        if len(observation) > 0:
            base = np.mean(observation) - 0.5
            action = np.full(dim, base * 0.3)
        else:
            action = np.zeros(dim)
        
        return np.clip(action, -1.0, 1.0)
    
    def _coordinate_actions(self, actions: Dict[str, np.ndarray],
                           observations: Dict[str, np.ndarray],
                           context: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """协调智能体行动"""
        
        # 1. 检测冲突
        conflicts = self._detect_conflicts(actions, observations, context)
        
        if not conflicts:
            return actions
        
        # 2. 根据配置选择冲突解决策略
        if self.config.conflict_resolution == "negotiation":
            return self._resolve_conflicts_by_negotiation(actions, conflicts, context)
        elif self.config.conflict_resolution == "voting":
            return self._resolve_conflicts_by_voting(actions, conflicts, context)
        elif self.config.conflict_resolution == "priority":
            return self._resolve_conflicts_by_priority(actions, conflicts, context)
        else:
            return actions
    
    def _detect_conflicts(self, actions: Dict[str, np.ndarray],
                         observations: Dict[str, np.ndarray],
                         context: Dict[str, Any]) -> List[Dict]:
        """检测行动冲突"""
        conflicts = []
        
        # 资源分配冲突
        resource_conflicts = self._detect_resource_conflicts(actions)
        conflicts.extend(resource_conflicts)
        
        # 目标冲突
        goal_conflicts = self._detect_goal_conflicts(actions, observations)
        conflicts.extend(goal_conflicts)
        
        # 优先级冲突
        priority_conflicts = self._detect_priority_conflicts(actions, context)
        conflicts.extend(priority_conflicts)
        
        return conflicts
    
    def _detect_resource_conflicts(self, actions: Dict[str, np.ndarray]) -> List[Dict]:
        """检测资源分配冲突"""
        conflicts = []
        
        if 'doctors' in actions and 'accountants' in actions:
            doctor_resource_demand = actions['doctors'][0] if len(actions['doctors']) > 0 else 0
            accountant_budget_control = actions['accountants'][0] if len(actions['accountants']) > 0 else 0
            
            if doctor_resource_demand > 0.5 and accountant_budget_control < -0.3:
                conflicts.append({
                    'type': 'resource_budget_conflict',
                    'roles': ['doctors', 'accountants'],
                    'severity': abs(doctor_resource_demand - accountant_budget_control),
                    'description': '医生要求增加资源与会计控制预算的冲突',
                    'actions': {
                        'doctors': actions['doctors'],
                        'accountants': actions['accountants']
                    }
                })
        
        if 'doctors' in actions and 'interns' in actions:
            doctor_workload = actions['doctors'][2] if len(actions['doctors']) > 2 else 0
            intern_learning_time = actions['interns'][0] if len(actions['interns']) > 0 else 0
            
            if doctor_workload > 0.4 and intern_learning_time > 0.6:
                conflicts.append({
                    'type': 'time_allocation_conflict',
                    'roles': ['doctors', 'interns'],
                    'severity': (doctor_workload + intern_learning_time) / 2,
                    'description': '医生工作负荷与实习生学习时间的冲突',
                    'actions': {
                        'doctors': actions['doctors'],
                        'interns': actions['interns']
                    }
                })
        
        return conflicts
    
    def _detect_goal_conflicts(self, actions: Dict[str, np.ndarray],
                              observations: Dict[str, np.ndarray]) -> List[Dict]:
        """检测目标冲突"""
        conflicts = []
        
        if 'doctors' in actions and 'accountants' in actions:
            doctor_quality_focus = actions['doctors'][-1] if len(actions['doctors']) > 0 else 0
            accountant_cost_focus = actions['accountants'][-1] if len(actions['accountants']) > 0 else 0
            
            if doctor_quality_focus > 0.5 and accountant_cost_focus < -0.4:
                conflicts.append({
                    'type': 'quality_cost_conflict',
                    'roles': ['doctors', 'accountants'],
                    'severity': abs(doctor_quality_focus - accountant_cost_focus),
                    'description': '医疗质量提升与成本控制的目标冲突',
                    'actions': {
                        'doctors': actions['doctors'],
                        'accountants': actions['accountants']
                    }
                })
        
        return conflicts
    
    def _detect_priority_conflicts(self, actions: Dict[str, np.ndarray],
                                  context: Dict[str, Any]) -> List[Dict]:
        """检测优先级冲突"""
        conflicts = []
        
        # 检查是否有紧急情况下的优先级冲突
        crisis_severity = context.get('crisis_severity', 0.0)
        if crisis_severity > 0.7:
            # 在危机情况下，长期投资行动可能与短期应急行动冲突
            long_term_actions = []
            short_term_actions = []
            
            for role, action in actions.items():
                action_urgency = np.mean(np.abs(action))  # 简化的紧急度评估
                if action_urgency < 0.3:
                    long_term_actions.append(role)
                elif action_urgency > 0.7:
                    short_term_actions.append(role)
            
            if long_term_actions and short_term_actions:
                conflicts.append({
                    'type': 'priority_conflict',
                    'roles': long_term_actions + short_term_actions,
                    'severity': crisis_severity,
                    'description': '危机情况下长期策略与短期应急的优先级冲突',
                    'long_term_roles': long_term_actions,
                    'short_term_roles': short_term_actions
                })
        
        return conflicts
    
    def _resolve_conflicts_by_negotiation(self, actions: Dict[str, np.ndarray],
                                        conflicts: List[Dict],
                                        context: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """通过协商解决冲突"""
        resolved_actions = copy.deepcopy(actions)
        
        for conflict in conflicts:
            negotiation_result = self._negotiate_conflict(conflict, resolved_actions, context)
            
            if negotiation_result['success']:
                # 应用协商结果
                for role, new_action in negotiation_result['actions'].items():
                    resolved_actions[role] = new_action
                
                self.coordination_metrics['negotiation_success_rate'] += 1
                
                # 记录协商历史
                self.negotiation_history.append({
                    'conflict': conflict,
                    'rounds': negotiation_result['rounds'],
                    'success': True,
                    'final_actions': negotiation_result['actions']
                })
            else:
                # 协商失败，使用降级策略
                resolved_actions = self._apply_fallback_resolution(conflict, resolved_actions)
        
        return resolved_actions
    
    def _negotiate_conflict(self, conflict: Dict, current_actions: Dict[str, np.ndarray],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """协商单个冲突"""
        negotiation_rounds = 0
        max_rounds = self.config.max_negotiation_rounds
        
        involved_roles = conflict['roles']
        current_positions = {role: current_actions[role] for role in involved_roles}
        
        while negotiation_rounds < max_rounds:
            negotiation_rounds += 1
            
            # 计算妥协方案
            compromise = self._calculate_compromise(conflict, current_positions, context)
            
            # 检查所有参与方是否接受妥协
            acceptance = self._evaluate_compromise_acceptance(compromise, involved_roles, context)
            
            if all(acceptance.values()):
                return {
                    'success': True,
                    'rounds': negotiation_rounds,
                    'actions': compromise,
                    'acceptance': acceptance
                }
            
            # 调整立场
            current_positions = self._adjust_positions(current_positions, compromise, acceptance)
        
        return {
            'success': False,
            'rounds': negotiation_rounds,
            'actions': current_positions,
            'acceptance': acceptance
        }
    
    def _calculate_compromise(self, conflict: Dict, positions: Dict[str, np.ndarray],
                            context: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """计算妥协方案"""
        compromise = {}
        
        if conflict['type'] == 'resource_budget_conflict':
            # 资源-预算冲突的妥协
            doctor_action = positions['doctors'].copy()
            accountant_action = positions['accountants'].copy()
            
            # 医生适度降低资源需求，会计适度放松预算控制
            doctor_action[0] *= 0.8
            accountant_action[0] *= 0.7
            
            compromise['doctors'] = doctor_action
            compromise['accountants'] = accountant_action
            
        elif conflict['type'] == 'time_allocation_conflict':
            # 时间分配冲突的妥协
            doctor_action = positions['doctors'].copy()
            intern_action = positions['interns'].copy()
            
            # 平衡工作负荷和学习时间
            doctor_action[2] *= 0.85
            intern_action[0] *= 0.9
            
            compromise['doctors'] = doctor_action
            compromise['interns'] = intern_action
            
        elif conflict['type'] == 'quality_cost_conflict':
            # 质量-成本冲突的妥协
            doctor_action = positions['doctors'].copy()
            accountant_action = positions['accountants'].copy()
            
            # 寻找质量和成本的平衡点
            doctor_action[-1] *= 0.9  # 略微降低质量要求
            accountant_action[-1] *= 0.8  # 略微放松成本控制
            
            compromise['doctors'] = doctor_action
            compromise['accountants'] = accountant_action
            
        else:
            # 默认妥协：所有相关行动都适度调整
            for role in conflict['roles']:
                compromise[role] = positions[role] * 0.85
        
        return compromise
    
    def _evaluate_compromise_acceptance(self, compromise: Dict[str, np.ndarray],
                                      involved_roles: List[str],
                                      context: Dict[str, Any]) -> Dict[str, bool]:
        """评估妥协方案的接受度"""
        acceptance = {}
        
        for role in involved_roles:
            agent = self.role_manager.get_agent(role)
            
            if hasattr(agent, 'behavior_model') and agent.behavior_model:
                # 使用行为模型评估接受度
                behavior_state = agent.behavior_model.state
                
                # 基于信任水平、风险容忍度等因素评估
                trust_factor = np.mean(list(behavior_state.trust_levels.values()))
                risk_tolerance = agent.behavior_model.parameters.risk_tolerance
                cooperation_tendency = agent.behavior_model.parameters.cooperation_tendency
                
                acceptance_score = (trust_factor * 0.4 + 
                                  risk_tolerance * 0.3 + 
                                  cooperation_tendency * 0.3)
                
                acceptance[role] = acceptance_score > 0.5
            else:
                # 简化的接受度评估
                acceptance[role] = np.random.random() > 0.3  # 70%概率接受
        
        return acceptance
    
    def _adjust_positions(self, positions: Dict[str, np.ndarray],
                         compromise: Dict[str, np.ndarray],
                         acceptance: Dict[str, bool]) -> Dict[str, np.ndarray]:
        """根据接受度调整立场"""
        adjusted_positions = {}
        
        for role, position in positions.items():
            if acceptance.get(role, False):
                # 接受的角色向妥协方案靠拢
                adjusted_positions[role] = 0.7 * position + 0.3 * compromise.get(role, position)
            else:
                # 不接受的角色保持立场或轻微调整
                adjusted_positions[role] = 0.95 * position + 0.05 * compromise.get(role, position)
        
        return adjusted_positions
    
    def _apply_fallback_resolution(self, conflict: Dict, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用降级冲突解决策略"""
        fallback_actions = copy.deepcopy(actions)
        
        # 简单的降级策略：所有冲突角色的行动都减半
        for role in conflict['roles']:
            if role in fallback_actions:
                fallback_actions[role] *= 0.5
        
        return fallback_actions
    
    def _resolve_conflicts_by_voting(self, actions: Dict[str, np.ndarray],
                                   conflicts: List[Dict],
                                   context: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """通过投票解决冲突"""
        # 简化实现：让所有智能体对冲突解决方案投票
        resolved_actions = copy.deepcopy(actions)
        
        for conflict in conflicts:
            # 生成候选解决方案
            solutions = self._generate_conflict_solutions(conflict, actions)
            
            # 收集投票
            votes = self._collect_votes_for_solutions(solutions, conflict['roles'])
            
            # 选择得票最多的方案
            winning_solution = max(solutions, key=lambda s: votes.get(s['id'], 0))
            
            # 应用获胜方案
            for role, action in winning_solution['actions'].items():
                resolved_actions[role] = action
        
        return resolved_actions
    
    def _generate_conflict_solutions(self, conflict: Dict, actions: Dict[str, np.ndarray]) -> List[Dict]:
        """生成冲突解决方案"""
        solutions = []
        
        # 方案1：保守调整
        conservative_actions = {}
        for role in conflict['roles']:
            conservative_actions[role] = actions[role] * 0.7
        
        solutions.append({
            'id': 'conservative',
            'description': '保守调整方案',
            'actions': conservative_actions
        })
        
        # 方案2：妥协方案
        compromise_actions = self._calculate_compromise(conflict, actions, {})
        solutions.append({
            'id': 'compromise',
            'description': '妥协方案',
            'actions': compromise_actions
        })
        
        # 方案3：优先级方案（优先考虑某一方）
        if len(conflict['roles']) >= 2:
            priority_actions = actions.copy()
            # 简化：优先考虑第一个角色
            priority_role = conflict['roles'][0]
            for other_role in conflict['roles'][1:]:
                priority_actions[other_role] *= 0.5
            
            solutions.append({
                'id': 'priority',
                'description': f'优先{priority_role}方案',
                'actions': priority_actions
            })
        
        return solutions
    
    def _collect_votes_for_solutions(self, solutions: List[Dict], involved_roles: List[str]) -> Dict[str, int]:
        """收集投票"""
        votes = {solution['id']: 0 for solution in solutions}
        
        for role in self.role_manager.agents.keys():
            # 简化的投票逻辑
            if role in involved_roles:
                # 冲突相关角色更倾向于选择妥协方案
                preferred_solution = 'compromise'
            else:
                # 非冲突角色随机投票
                preferred_solution = np.random.choice([s['id'] for s in solutions])
            
            votes[preferred_solution] += 1
        
        return votes
    
    def _resolve_conflicts_by_priority(self, actions: Dict[str, np.ndarray],
                                     conflicts: List[Dict],
                                     context: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """基于优先级解决冲突"""
        resolved_actions = copy.deepcopy(actions)
        
        # 角色优先级（可配置）
        role_priorities = {
            'government': 5,
            'doctors': 4,
            'patients': 3,
            'accountants': 2,
            'interns': 1
        }
        
        for conflict in conflicts:
            # 找到优先级最高的角色
            highest_priority = -1
            priority_role = None
            
            for role in conflict['roles']:
                priority = role_priorities.get(role, 0)
                if priority > highest_priority:
                    highest_priority = priority
                    priority_role = role
            
            # 其他角色的行动向优先角色妥协
            if priority_role:
                for role in conflict['roles']:
                    if role != priority_role:
                        resolved_actions[role] *= 0.6  # 降低行动强度
        
        return resolved_actions
    
    def _update_agent_states(self, observations: Dict[str, np.ndarray],
                            actions: Dict[str, np.ndarray],
                            context: Dict[str, Any]):
        """更新智能体状态"""
        
        # 计算奖励（简化）
        rewards = self._calculate_rewards(observations, actions, context)
        
        # 更新行为模型状态
        if self.config.use_behavior_models and hasattr(self, 'behavior_manager'):
            self.behavior_manager.update_all_models(observations, actions, rewards, context)
        
        # 为智能体添加经验
        for role, agent in self.role_manager.agents.items():
            if role in observations and role in actions and role in rewards:
                if hasattr(agent, 'add_experience'):
                    experience = agent.add_experience(
                        observations[role], actions[role], rewards[role],
                        observations[role], False  # next_state, done
                    )
    
    def _calculate_rewards(self, observations: Dict[str, np.ndarray],
                          actions: Dict[str, np.ndarray],
                          context: Dict[str, Any]) -> Dict[str, float]:
        """计算奖励"""
        rewards = {}
        
        # 基于合作度的基础奖励
        cooperation_score = self._calculate_cooperation_score(actions)
        base_reward = cooperation_score * 0.5
        
        for role in actions.keys():
            # 角色特定奖励
            role_reward = base_reward
            
            if role == 'doctors':
                # 医生的奖励基于医疗质量
                quality_indicator = np.mean(observations.get(role, [0.5]))
                role_reward += quality_indicator * 0.3
            elif role == 'accountants':
                # 会计的奖励基于财务效率
                efficiency_indicator = 1.0 - np.std(actions[role])  # 行动稳定性
                role_reward += efficiency_indicator * 0.3
            
            # 冲突惩罚
            if self.active_conflicts:
                conflict_penalty = len([c for c in self.active_conflicts if role in c['roles']]) * 0.1
                role_reward -= conflict_penalty
            
            rewards[role] = np.clip(role_reward, -1.0, 1.0)
        
        return rewards
    
    def _calculate_cooperation_score(self, actions: Dict[str, np.ndarray]) -> float:
        """计算合作得分"""
        if len(actions) < 2:
            return 1.0
        
        action_vectors = list(actions.values())
        
        # 计算行动向量的平均相似度
        similarities = []
        for i in range(len(action_vectors)):
            for j in range(i + 1, len(action_vectors)):
                # 使用余弦相似度
                v1, v2 = action_vectors[i], action_vectors[j]
                
                # 确保向量长度一致
                min_len = min(len(v1), len(v2))
                v1, v2 = v1[:min_len], v2[:min_len]
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    similarities.append(max(0, similarity))  # 只考虑正相关
        
        return np.mean(similarities) if similarities else 0.5
    
    def _record_interaction(self, system_state: np.ndarray,
                           observations: Dict[str, np.ndarray],
                           actions: Dict[str, np.ndarray],
                           context: Dict[str, Any]):
        """记录交互历史"""
        
        interaction = {
            'timestamp': len(self.interaction_history),
            'system_state': system_state.tolist(),
            'observations': {role: obs.tolist() for role, obs in observations.items()},
            'actions': {role: act.tolist() for role, act in actions.items()},
            'context': copy.deepcopy(context),
            'cooperation_score': self._calculate_cooperation_score(actions),
            'conflict_count': len(self.active_conflicts),
            'coordination_metrics': copy.deepcopy(self.coordination_metrics)
        }
        
        self.interaction_history.append(interaction)
        
        # 限制历史长度
        if len(self.interaction_history) > 1000:
            self.interaction_history.pop(0)
    
    def get_interaction_metrics(self) -> Dict[str, Any]:
        """获取交互指标"""
        if not self.interaction_history:
            return {}
        
        recent_interactions = self.interaction_history[-100:]  # 最近100次交互
        
        avg_cooperation = np.mean([i['cooperation_score'] for i in recent_interactions])
        avg_conflicts = np.mean([i['conflict_count'] for i in recent_interactions])
        
        negotiation_success_rate = (self.coordination_metrics['negotiation_success_rate'] / 
                                   max(1, len(self.negotiation_history)))
        
        return {
            'average_cooperation_score': avg_cooperation,
            'average_conflict_count': avg_conflicts,
            'total_interactions': len(self.interaction_history),
            'negotiation_success_rate': negotiation_success_rate,
            'active_conflicts': len(self.active_conflicts),
            'coordination_metrics': self.coordination_metrics
        }
    
    def reset_interaction_state(self):
        """重置交互状态"""
        self.active_conflicts.clear()
        self.coordination_metrics = {
            'conflict_count': 0,
            'cooperation_score': 0.0,
            'consensus_rate': 0.0,
            'negotiation_success_rate': 0.0
        }
    
    def export_interaction_history(self, filepath: str, limit: int = 100):
        """导出交互历史"""
        import json
        
        export_data = {
            'config': {
                'use_behavior_models': self.config.use_behavior_models,
                'use_learning_models': self.config.use_learning_models,
                'use_llm_generation': self.config.use_llm_generation,
                'conflict_resolution': self.config.conflict_resolution
            },
            'metrics': self.get_interaction_metrics(),
            'interaction_history': self.interaction_history[-limit:],
            'negotiation_history': self.negotiation_history[-50:]  # 最近50次协商
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)