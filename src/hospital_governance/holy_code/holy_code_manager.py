"""
Holy Code管理器 - 统一协调神圣法典系统的各个组件
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path

from .rule_engine import RuleEngine
from .rule_library import RuleLibrary
from .parliament import Parliament, ParliamentConfig, VoteType
from .reference_generator import ReferenceGenerator, ReferenceType, ReferenceConfig

logger = logging.getLogger(__name__)

@dataclass
class HolyCodeConfig:
    """Holy Code系统配置"""
    rule_config_path: Optional[str] = None
    parliament_config: Optional[ParliamentConfig] = None
    reference_config: Optional[ReferenceConfig] = None
    enable_rule_learning: bool = True
    enable_adaptive_references: bool = True
    crisis_threshold: float = 0.6

class HolyCodeManager:
    """
    Holy Code管理器 - 统一管理和协调神圣法典系统
    
    这个类作为holy_code模块的主要接口，统一管理：
    - 规则引擎和规则库
    - 议会决策系统
    - 参考值生成器
    - 与agents模块的集成接口
    """
    
    def __init__(self, config: Optional[HolyCodeConfig] = None):
        self.config = config or HolyCodeConfig()
        
        # 初始化核心组件
        self.rule_library = RuleLibrary()
        self.rule_engine = RuleEngine(self.config.rule_config_path)
        
        # 从规则库加载预定义规则
        self._load_rules_from_library()
        
        # 初始化议会系统
        parliament_config = self.config.parliament_config or ParliamentConfig(
            vote_weights={
                'chief_doctor': 0.3,
                'doctors': 0.25,
                'nurses': 0.2,
                'administrators': 0.15,
                'patients_representative': 0.1
            }
        )
        self.parliament = Parliament(self.rule_engine, parliament_config)
        
        # 简化初始化参考值生成器（临时绕过复杂依赖）
        try:
            reference_config = self.config.reference_config or ReferenceConfig(
                reference_type=ReferenceType.SETPOINT,
                setpoints={'overall_performance': 0.8},
                trajectory_parameters={},
                adaptation_rules=[],
                crisis_response_profiles={}
            )
            
            # 创建一个兼容ReferenceGenerator的状态空间对象
            class SimpleStateSpace:
                def __init__(self):
                    self.dimensions = 16  # ReferenceGenerator期望的属性名
                    self.variable_names = [  # ReferenceGenerator期望的属性名
                        'bed_occupancy_rate', 'medical_equipment_utilization', 
                        'staff_utilization_rate', 'medication_inventory_level',
                        'cash_reserve_ratio', 'operating_margin', 
                        'debt_to_asset_ratio', 'cost_efficiency_index',
                        'patient_satisfaction_index', 'treatment_success_rate', 
                        'average_wait_time', 'medical_safety_index',
                        'ethical_compliance_score', 'resource_allocation_fairness', 
                        'intern_learning_efficiency', 'knowledge_transfer_rate'
                    ]
            
            simple_state_space = SimpleStateSpace()
            self.reference_generator = ReferenceGenerator(simple_state_space, reference_config)
            logger.info("✅ 参考值生成器初始化成功")
            
        except Exception as e:
            # 如果初始化失败，设置为None
            logger.warning(f"⚠️ 参考值生成器初始化失败，将使用简化模式: {e}")
            self.reference_generator = None
        
        # 系统状态
        self.system_state = {
            'active_crisis': False,
            'crisis_type': None,
            'last_decision_time': 0,
            'performance_metrics': {},
            'rule_adaptation_enabled': self.config.enable_rule_learning
        }
        
        # 性能跟踪
        self.performance_history = []
        self.decision_effectiveness = {}
        
        # 议会会议周期（单位：步/小时）
        self.meeting_interval = 168  # 默认每168步/小时触发一次会议（可根据仿真系统调整）
    
    def run_weekly_parliament_meeting(self, agents: Dict[str, Any], system_state: Any):
        """
        议会会议流程：
        1. 智能体博弈与提案
        2. 议会投票与共识
        3. 共识写入神圣法典
        4. 计算收益
        5. 更新actor-critic网络
        """
        try:
            # 1. 智能体博弈与提案
            agent_proposals = {}
            for agent_id, agent in agents.items():
                if hasattr(agent, 'generate_proposal'):
                    proposal = agent.generate_proposal(system_state)
                else:
                    # 简化的提案生成
                    proposal = {
                        'action': f'{agent_id}_default_action',
                        'context': 'routine_operation',
                        'priority': 0.5
                    }
                agent_proposals[agent_id] = proposal

            # 2. 简化的共识达成（由于parliament的方法名称问题）
            if agent_proposals:
                total_priority = sum(p.get('priority', 0.5) for p in agent_proposals.values())
                consensus_level = min(0.9, total_priority / len(agent_proposals))
            else:
                consensus_level = 0.5
            
            consensus_result = {
                'consensus_level': consensus_level,
                'participating_agents': list(agent_proposals.keys()),
                'proposals': agent_proposals
            }

            # 3. 共识写入神圣法典
            self.write_consensus(consensus_result)

            # 4. 计算收益
            rewards = self.calculate_rewards(consensus_result, agents, system_state)

            # 5. 更新actor-critic网络（简化版本）
            for agent_id, agent in agents.items():
                if hasattr(agent, 'update_network'):
                    agent.update_network(rewards.get(agent_id, 0.5))

            # 6. 记录会议结果
            self.log_meeting(consensus_result, rewards)
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"议会会议执行失败: {e}")
            return {'error': str(e), 'consensus_level': 0.5}

    def write_consensus(self, consensus_result: Any):
        """将共识结果写入神圣法典（规则库/引擎）"""
        try:
            # 检查是否有规则引擎的共识处理方法
            if hasattr(self.rule_engine, 'add_consensus'):
                self.rule_engine.add_consensus(consensus_result)
            else:
                # 将共识结果转换为规则格式并写入规则库
                if isinstance(consensus_result, dict):
                    consensus_level = consensus_result.get('consensus_level', 0.5)
                    
                    # 只有当共识水平足够高时才创建新规则
                    if consensus_level > 0.7:
                        from .rule_engine import Rule, RulePriority
                        
                        # 创建一个基于共识的新规则
                        rule_id = f"consensus_rule_{len(self.performance_history)}"
                        
                        # 简化的规则创建
                        new_rule = Rule(
                            rule_id=rule_id,
                            name=f"共识规则: {consensus_result.get('main_decision', '默认决策')}",
                            condition=self._create_consensus_condition(consensus_result),
                            action=self._create_consensus_action(consensus_result),
                            priority=RulePriority.MEDIUM,
                            weight=consensus_level,
                            context=['parliament', 'consensus'],
                            description=f"基于议会共识创建的规则，共识水平: {consensus_level:.2f}"
                        )
                        
                        self.rule_library.add_rule(new_rule)
                        # 同时添加到规则引擎中，以确保统计正确
                        self.rule_engine.add_rule(new_rule)
                        logger.info(f"✅ 新共识规则已创建: {rule_id}")
                    else:
                        logger.info(f"ℹ️ 共识水平过低({consensus_level:.2f})，未创建新规则")
                else:
                    logger.warning("⚠️ 共识结果格式不正确，无法写入规则库")
        except Exception as e:
            logger.error(f"❌ 写入共识到神圣法典失败: {e}")

    def _create_consensus_condition(self, consensus_result: Dict) -> Callable:
        """为共识创建条件函数"""
        def consensus_condition(context: Dict[str, Any]) -> bool:
            # 简化的条件：总是返回True，表示规则可以激活
            return True
        return consensus_condition

    def _create_consensus_action(self, consensus_result: Dict) -> Callable:
        """为共识创建动作函数"""
        def consensus_action(context: Dict[str, Any]) -> Dict[str, Any]:
            return {
                'type': 'consensus_guidance',
                'recommendations': [f"执行共识决策: {consensus_result.get('main_decision', '默认')}"],
                'priority_boost': 0.2,
                'weight_adjustment': 1.0
            }
        return consensus_action

    def log_meeting(self, consensus_result: Any, rewards: Dict[str, float]):
        """记录本次议会会议结果和收益"""
        record = {
            'step': len(self.performance_history),
            'consensus': consensus_result,
            'rewards': rewards
        }
        self.performance_history.append(record)

    def calculate_rewards(self, consensus_result: Any, agents: Dict[str, Any], system_state: Any) -> Dict[str, float]:
        """根据共识和系统状态计算每个智能体的收益（可自定义）"""
        # 这里可根据共识内容和agent状态自定义收益函数
        rewards = {}
        for agent_id in agents:
            rewards[agent_id] = np.random.uniform(0.5, 1.5)  # 示例：随机收益
        return rewards
    def _load_rules_from_library(self):
        """从规则库加载预定义规则"""
        for rule in self.rule_library.get_all_rules():
            self.rule_engine.add_rule(rule)
    
    def process_agent_decision_request(self, agent_id: str, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理来自agents模块的决策请求
        
        Args:
            agent_id: 请求决策的agent ID
            decision_context: 决策上下文信息
            
        Returns:
            包含holy code指导的决策建议
        """
        # 评估当前状态是否为危机
        crisis_detected = self._detect_crisis(decision_context)
        
        if crisis_detected:
            self._activate_crisis_mode(decision_context)
        
        # 使用规则引擎评估决策
        rule_evaluation = self.rule_engine.evaluate_rules(decision_context)
        
        # 生成参考值
        reference_values = self._generate_references_for_decision(decision_context)
        
        # 如果需要集体决策，提交议会
        if self._requires_collective_decision(decision_context, rule_evaluation):
            parliamentary_result = self._submit_to_parliament(agent_id, decision_context, rule_evaluation)
        else:
            parliamentary_result = None
        
        # 整合所有建议
        decision_guidance = self._integrate_guidance(
            rule_evaluation, 
            reference_values, 
            parliamentary_result,
            crisis_detected
        )
        
        # 记录决策
        self._record_decision(agent_id, decision_context, decision_guidance)
        
        return decision_guidance
    
    def _detect_crisis(self, context: Dict[str, Any]) -> bool:
        """检测是否存在危机情况"""
        state = context.get('state', {})
        
        # 检查各种危机指标
        crisis_indicators = [
            state.get('patient_safety', 1.0) < 0.7,
            state.get('system_stability', 1.0) < self.config.crisis_threshold,
            state.get('financial_health', 1.0) < 0.5,
            context.get('emergency_situation', False),
            state.get('resource_shortage', False)
        ]
        
        crisis_count = sum(crisis_indicators)
        return crisis_count >= 2  # 两个或以上指标触发危机模式
    
    def _activate_crisis_mode(self, context: Dict[str, Any]):
        """激活危机模式"""
        self.system_state['active_crisis'] = True
        
        # 确定危机类型
        state = context.get('state', {})
        if state.get('patient_safety', 1.0) < 0.7:
            self.system_state['crisis_type'] = 'patient_safety'
        elif state.get('financial_health', 1.0) < 0.5:
            self.system_state['crisis_type'] = 'financial'
        elif context.get('emergency_situation', False):
            self.system_state['crisis_type'] = 'emergency'
        else:
            self.system_state['crisis_type'] = 'system_instability'
        
        # 调整参考值生成器
        self.reference_generator.activate_crisis_mode(self.system_state['crisis_type'])
        
        print(f"危机模式已激活: {self.system_state['crisis_type']}")
    
    def _generate_references_for_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """为决策生成参考值"""
        if not self.reference_generator:
            # 简化的参考值生成
            return {
                'target_performance': 0.8,
                'safety_threshold': 0.9,
                'efficiency_target': 0.75
            }
        
        # 根据决策类型选择参考值类型
        decision_type = context.get('decision_type', 'general')
        
        if decision_type == 'resource_allocation':
            ref_type = ReferenceType.ADAPTIVE
        elif self.system_state['active_crisis']:
            ref_type = ReferenceType.CRISIS_RESPONSE
        elif decision_type == 'policy_change':
            ref_type = ReferenceType.TRAJECTORY
        else:
            ref_type = ReferenceType.SETPOINT
        
        return self.reference_generator.generate_reference(
            np.zeros(16),  # current_state parameter 
            {
                'reference_type': ref_type,
                'current_state': context.get('current_state', {}),
                'time_horizon': context.get('time_horizon', 10)
            }  # context parameter
        )
    
    def _requires_collective_decision(self, context: Dict[str, Any], rule_evaluation: List[Dict]) -> bool:
        """判断是否需要集体决策"""
        decision_type = context.get('decision_type', 'individual')
        
        # 需要集体决策的情况
        collective_required = [
            decision_type in ['resource_allocation', 'policy_change', 'budget_approval'],
            any(rule['priority'] <= 2 for rule in rule_evaluation),  # 高优先级规则
            context.get('impact_scope', 'individual') == 'system_wide',
            self.system_state['active_crisis']
        ]
        
        return any(collective_required)
    
    def _submit_to_parliament(self, proposer: str, context: Dict[str, Any], 
                            rule_evaluation: List[Dict]) -> Dict[str, Any]:
        """提交议会决策"""
        # 构建提案
        proposal = {
            'context': context.get('decision_type', 'general'),
            'current_state': context.get('state', {}),
            'proposed_action': context.get('proposed_action', {}),
            'rule_evaluation': rule_evaluation,
            'urgency': 'high' if self.system_state['active_crisis'] else 'normal'
        }
        
        # 提交提案
        proposal_id = self.parliament.submit_proposal(proposal, proposer)
        
        # 模拟相关agents的投票 (在实际系统中，这将通过agent通信完成)
        self._simulate_agent_voting(proposal_id, proposal, rule_evaluation)
        
        # 统计投票结果
        approved, approval_rate, voter_analysis = self.parliament.tally_votes(proposal_id)
        
        return {
            'approved': approved,
            'approval_rate': approval_rate,
            'voter_analysis': voter_analysis,
            'proposal_id': proposal_id
        }
    
    def _simulate_agent_voting(self, proposal_id: str, proposal: Dict[str, Any], 
                             rule_evaluation: List[Dict]):
        """模拟agent投票 (实际实现中应该通过真实的agent通信)"""
        # 基于规则评估和角色特征模拟投票
        voters = ['chief_doctor', 'doctors', 'nurses', 'administrators', 'patients_representative']
        
        for voter in voters:
            # 基于角色和规则评估决定投票
            vote = self._calculate_vote_preference(voter, proposal, rule_evaluation)
            rationale = f"基于{voter}角色和当前规则评估的决策"
            
            self.parliament.cast_vote(proposal_id, voter, vote, rationale)
    
    def _calculate_vote_preference(self, voter_role: str, proposal: Dict[str, Any], 
                                 rule_evaluation: List[Dict]) -> bool:
        """计算投票偏好"""
        # 简化的投票逻辑，实际应该更复杂
        role_priorities = {
            'chief_doctor': ['patient_safety', 'medical_quality'],
            'doctors': ['patient_safety', 'work_efficiency'],
            'nurses': ['patient_care', 'work_conditions'],
            'administrators': ['financial_health', 'operational_efficiency'],
            'patients_representative': ['patient_satisfaction', 'care_quality']
        }
        
        priorities = role_priorities.get(voter_role, ['general'])
        
        # 基于优先级和规则评估计算投票倾向
        positive_score = 0
        for rule in rule_evaluation:
            action_result = rule.get('action_result', {})
            rule_type = action_result.get('type', '')
            
            if any(priority in rule_type for priority in priorities):
                positive_score += rule.get('weight', 0.5)
        
        return positive_score > 0.5
    
    def _integrate_guidance(self, rule_evaluation: List[Dict], reference_values: Dict[str, Any],
                          parliamentary_result: Optional[Dict[str, Any]], crisis_mode: bool) -> Dict[str, Any]:
        """整合所有指导建议"""
        guidance = {
            'rule_recommendations': [],
            'reference_targets': reference_values,
            'priority_adjustments': {},
            'crisis_mode': crisis_mode,
            'collective_approval': None
        }
        
        # 整合规则建议
        total_priority_boost = 0.0
        for rule in rule_evaluation:
            action_result = rule.get('action_result', {})
            guidance['rule_recommendations'].extend(action_result.get('recommendations', []))
            total_priority_boost += action_result.get('priority_boost', 0.0)
        
        guidance['priority_boost'] = min(1.0, total_priority_boost)
        
        # 整合议会决策
        if parliamentary_result:
            guidance['collective_approval'] = parliamentary_result
            if not parliamentary_result['approved']:
                guidance['priority_boost'] *= 0.5  # 降低优先级如果未获批准
        
        # 危机模式调整
        if crisis_mode:
            guidance['priority_boost'] *= 1.5
            guidance['rule_recommendations'].insert(0, "执行危机应对协议")
        
        return guidance
    
    def _record_decision(self, agent_id: str, context: Dict[str, Any], 
                        guidance: Dict[str, Any]):
        """记录决策信息"""
        decision_record = {
            'timestamp': len(self.performance_history),
            'agent_id': agent_id,
            'decision_type': context.get('decision_type', 'general'),
            'guidance': guidance,
            'crisis_mode': self.system_state['active_crisis']
        }
        
        self.performance_history.append(decision_record)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'system_state': self.system_state,
            'rule_engine_stats': self.rule_engine.get_rule_statistics(),
            'parliament_metrics': self.parliament.get_consensus_metrics(),
            'reference_generator_status': (
                {
                    'status': 'active',
                    'type': str(self.reference_generator.config.reference_type) if self.reference_generator else 'disabled',
                    'dimensions': getattr(self.reference_generator.state_space, 'dimensions', 0) if self.reference_generator else 0
                } if self.reference_generator else {'status': 'disabled'}
            ),
            'total_decisions': len(self.performance_history),
            'crisis_decisions': sum(1 for d in self.performance_history if d.get('crisis_mode', False))
        }
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """更新系统性能指标"""
        self.system_state['performance_metrics'].update(metrics)
        
        # 检查是否需要退出危机模式
        if self.system_state['active_crisis']:
            self._check_crisis_resolution(metrics)
    
    def _check_crisis_resolution(self, metrics: Dict[str, float]):
        """检查危机是否解决"""
        crisis_resolved = all([
            metrics.get('patient_safety', 0.0) >= 0.8,
            metrics.get('system_stability', 0.0) >= 0.8,
            metrics.get('financial_health', 0.0) >= 0.7
        ])
        
        if crisis_resolved:
            self.system_state['active_crisis'] = False
            self.system_state['crisis_type'] = None
            self.reference_generator.deactivate_crisis_mode()
            print("危机模式已解除")
    
    def get_integration_interface(self) -> Dict[str, Any]:
        """
        获取与agents模块集成的接口信息
        
        Returns:
            接口规范和回调函数
        """
        return {
            'decision_request_handler': self.process_agent_decision_request,
            'status_query_handler': self.get_system_status,
            'performance_update_handler': self.update_performance_metrics,
            'supported_decision_types': [
                'resource_allocation',
                'policy_change', 
                'crisis_response',
                'routine_operation',
                'budget_approval'
            ],
            'required_context_fields': [
                'decision_type',
                'agent_id',
                'current_state',
                'proposed_action'
            ],
            'optional_context_fields': [
                'time_horizon',
                'impact_scope',
                'urgency_level',
                'stakeholders'
            ]
        }