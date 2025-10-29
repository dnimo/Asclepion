"""
角色特定奖励控制器实现
Role-Specific Reward Controllers Implementation

基于每个角色的特异性需求和目标函数，实现定制化的奖励调节逻辑
"""

import numpy as np
from typing import Dict, Any
import logging

from .reward_based_controller import RewardBasedController, RewardControlConfig
from ..core.state_space import SystemState
from ..agents.role_agents import RoleAgent

logger = logging.getLogger(__name__)


class DoctorRewardController(RewardBasedController):
    def compute_reward(self, system_state, action, decisions):
        return 1.0
    """医生奖励控制器
    
    重点关注：医疗质量、患者满意度、安全事故预防、资源利用效率
    控制策略：通过奖励调节激励医生优化医疗决策
    """
    
    def _compute_role_specific_adjustment(self, 
                                        current_state: SystemState,
                                        target_state: SystemState,
                                        context: Dict[str, Any]) -> float:
        """医生特异性奖励调节"""
        
        # 医疗质量偏差
        quality_error = target_state.care_quality_index - current_state.care_quality_index
        quality_adjustment = quality_error * 0.8  # 高权重
        
        # 安全事故率偏差（负向激励）
        safety_error = current_state.safety_incident_rate - target_state.safety_incident_rate
        safety_adjustment = safety_error * 1.0  # 安全优先
        
        # 患者满意度偏差
        satisfaction_error = target_state.patient_satisfaction - current_state.patient_satisfaction
        satisfaction_adjustment = satisfaction_error * 0.6
        
        # 资源利用效率
        resource_error = target_state.medical_resource_utilization - current_state.medical_resource_utilization
        resource_adjustment = resource_error * 0.4
        
        # 工作负荷平衡（防止过度工作）
        if 'workload_balance' in context:
            workload_ratio = context['workload_balance']
            if workload_ratio > 0.9:  # 过度工作
                workload_penalty = -0.3
            elif workload_ratio < 0.3:  # 工作不足
                workload_penalty = -0.2
            else:
                workload_penalty = 0.1  # 工作平衡奖励
        else:
            workload_penalty = 0.0
        
        # 团队协作奖励
        collaboration_bonus = 0.0
        if 'collaboration_score' in context:
            collaboration_bonus = min(0.2, context['collaboration_score'] * 0.2)
        
        total_adjustment = (quality_adjustment + 
                          safety_adjustment + 
                          satisfaction_adjustment + 
                          resource_adjustment + 
                          workload_penalty + 
                          collaboration_bonus)
        
        logger.debug(f"Doctor reward components: quality={quality_adjustment:.3f}, "
                    f"safety={safety_adjustment:.3f}, satisfaction={satisfaction_adjustment:.3f}")
        
        return total_adjustment


class InternRewardController(RewardBasedController):
    def compute_reward(self, system_state, action, decisions):
        return 0.8
    """实习医生奖励控制器
    
    重点关注：学习成长、培训质量、导师指导、专业发展
    控制策略：通过学习导向的奖励激励实习医生提升能力
    """
    
    def _compute_role_specific_adjustment(self, 
                                        current_state: SystemState,
                                        target_state: SystemState,
                                        context: Dict[str, Any]) -> float:
        """实习医生特异性奖励调节"""
        
        # 教育培训质量偏差
        training_error = target_state.education_training_quality - current_state.education_training_quality
        training_adjustment = training_error * 1.0  # 学习优先
        
        # 实习生满意度
        intern_satisfaction_error = target_state.intern_satisfaction - current_state.intern_satisfaction
        satisfaction_adjustment = intern_satisfaction_error * 0.7
        
        # 专业发展指标
        development_error = target_state.professional_development - current_state.professional_development
        development_adjustment = development_error * 0.8
        
        # 导师指导效果
        mentorship_error = target_state.mentorship_effectiveness - current_state.mentorship_effectiveness
        mentorship_adjustment = mentorship_error * 0.6
        
        # 学习进度奖励
        learning_bonus = 0.0
        if 'learning_progress' in context:
            progress = context['learning_progress']
            if progress > 0.8:  # 快速学习
                learning_bonus = 0.3
            elif progress > 0.5:  # 正常学习
                learning_bonus = 0.1
            else:  # 学习缓慢
                learning_bonus = -0.1
        
        # 实践机会奖励
        practice_bonus = 0.0
        if 'practice_opportunities' in context:
            opportunities = context['practice_opportunities']
            practice_bonus = min(0.2, opportunities * 0.1)
        
        # 错误学习惩罚（温和）
        mistake_penalty = 0.0
        if 'learning_mistakes' in context:
            mistakes = context['learning_mistakes']
            mistake_penalty = -min(0.1, mistakes * 0.05)  # 温和惩罚，鼓励试错学习
        
        total_adjustment = (training_adjustment + 
                          satisfaction_adjustment + 
                          development_adjustment + 
                          mentorship_adjustment + 
                          learning_bonus + 
                          practice_bonus + 
                          mistake_penalty)
        
        logger.debug(f"Intern reward components: training={training_adjustment:.3f}, "
                    f"development={development_adjustment:.3f}, learning_bonus={learning_bonus:.3f}")
        
        return total_adjustment


class PatientRewardController(RewardBasedController):
    def compute_reward(self, system_state, action, decisions):
        return 0.6
    """患者奖励控制器
    
    重点关注：医疗服务质量、等待时间、满意度、可及性
    控制策略：通过患者体验优化激励系统改善服务质量
    """
    
    def _compute_role_specific_adjustment(self, 
                                        current_state: SystemState,
                                        target_state: SystemState,
                                        context: Dict[str, Any]) -> float:
        """患者特异性奖励调节"""
        
        # 患者满意度偏差（最高权重）
        satisfaction_error = target_state.patient_satisfaction - current_state.patient_satisfaction
        satisfaction_adjustment = satisfaction_error * 1.2
        
        # 服务可及性偏差
        accessibility_error = target_state.service_accessibility - current_state.service_accessibility
        accessibility_adjustment = accessibility_error * 0.9
        
        # 医疗质量偏差
        quality_error = target_state.care_quality_index - current_state.care_quality_index
        quality_adjustment = quality_error * 0.8
        
        # 等待时间惩罚
        waiting_time_penalty = 0.0
        if 'average_waiting_time' in context:
            waiting_time = context['average_waiting_time']
            if waiting_time > 60:  # 超过1小时
                waiting_time_penalty = -min(0.5, (waiting_time - 60) / 120)
            elif waiting_time < 15:  # 短等待时间奖励
                waiting_time_penalty = 0.2
        
        # 医疗费用合理性
        cost_reasonableness = 0.0
        if 'cost_per_service' in context and 'service_quality' in context:
            cost = context['cost_per_service']
            quality = context['service_quality']
            value_ratio = quality / (cost + 1e-8)
            if value_ratio > 1.5:  # 高性价比
                cost_reasonableness = 0.3
            elif value_ratio < 0.5:  # 低性价比
                cost_reasonableness = -0.2
        
        # 医疗安全奖励
        safety_bonus = 0.0
        if current_state.safety_incident_rate < target_state.safety_incident_rate:
            safety_improvement = target_state.safety_incident_rate - current_state.safety_incident_rate
            safety_bonus = safety_improvement * 0.8
        
        # 个性化医疗奖励
        personalization_bonus = 0.0
        if 'personalized_care_score' in context:
            personalization_bonus = context['personalized_care_score'] * 0.2
        
        total_adjustment = (satisfaction_adjustment + 
                          accessibility_adjustment + 
                          quality_adjustment + 
                          waiting_time_penalty + 
                          cost_reasonableness + 
                          safety_bonus + 
                          personalization_bonus)
        
        logger.debug(f"Patient reward components: satisfaction={satisfaction_adjustment:.3f}, "
                    f"waiting_penalty={waiting_time_penalty:.3f}, safety_bonus={safety_bonus:.3f}")
        
        return total_adjustment


class AccountantRewardController(RewardBasedController):
    def compute_reward(self, system_state, action, decisions):
        return 0.9
    """会计奖励控制器
    
    重点关注：财务效率、成本控制、合规性、透明度
    控制策略：通过财务激励优化资源配置和成本管理
    """
    
    def _compute_role_specific_adjustment(self, 
                                        current_state: SystemState,
                                        target_state: SystemState,
                                        context: Dict[str, Any]) -> float:
        """会计特异性奖励调节"""
        
        # 财务指标偏差
        financial_error = target_state.financial_indicator - current_state.financial_indicator
        financial_adjustment = financial_error * 1.0
        
        # 运营效率偏差
        efficiency_error = target_state.operational_efficiency - current_state.operational_efficiency
        efficiency_adjustment = efficiency_error * 0.9
        
        # 资源利用率偏差
        resource_error = target_state.medical_resource_utilization - current_state.medical_resource_utilization
        resource_adjustment = resource_error * 0.7
        
        # 合规性偏差
        compliance_error = target_state.regulatory_compliance_score - current_state.regulatory_compliance_score
        compliance_adjustment = compliance_error * 0.8
        
        # 成本控制奖励
        cost_control_bonus = 0.0
        if 'cost_variance' in context:
            variance = context['cost_variance']
            if variance < 0.05:  # 成本控制良好
                cost_control_bonus = 0.3
            elif variance > 0.2:  # 成本超支严重
                cost_control_bonus = -0.4
        
        # 预算准确性奖励
        budget_accuracy = 0.0
        if 'budget_accuracy' in context:
            accuracy = context['budget_accuracy']
            budget_accuracy = (accuracy - 0.5) * 0.4  # 以50%为基准
        
        # 财务透明度奖励
        transparency_bonus = 0.0
        if 'financial_transparency' in context:
            transparency_bonus = context['financial_transparency'] * 0.2
        
        # 投资回报率
        roi_bonus = 0.0
        if 'roi' in context:
            roi = context['roi']
            if roi > 0.1:  # 10%以上回报
                roi_bonus = min(0.3, roi * 2)
            elif roi < 0:  # 负回报惩罚
                roi_bonus = max(-0.2, roi * 2)
        
        total_adjustment = (financial_adjustment + 
                          efficiency_adjustment + 
                          resource_adjustment + 
                          compliance_adjustment + 
                          cost_control_bonus + 
                          budget_accuracy + 
                          transparency_bonus + 
                          roi_bonus)
        
        logger.debug(f"Accountant reward components: financial={financial_adjustment:.3f}, "
                    f"efficiency={efficiency_adjustment:.3f}, cost_control={cost_control_bonus:.3f}")
        
        return total_adjustment


class GovernmentRewardController(RewardBasedController):
    def compute_reward(self, system_state, action, decisions):
        return 0.7
    """政府奖励控制器
    
    重点关注：政策效果、公平性、合规监管、公共利益
    控制策略：通过政策激励优化整体医疗治理效果
    """
    
    def _compute_role_specific_adjustment(self, 
                                        current_state: SystemState,
                                        target_state: SystemState,
                                        context: Dict[str, Any]) -> float:
        """政府特异性奖励调节"""
        
        # 监管合规偏差
        compliance_error = target_state.regulatory_compliance_score - current_state.regulatory_compliance_score
        compliance_adjustment = compliance_error * 1.1  # 合规最优先
        
        # 伦理遵循偏差
        ethical_error = target_state.ethical_compliance - current_state.ethical_compliance
        ethical_adjustment = ethical_error * 1.0
        
        # 患者满意度偏差（公共利益）
        satisfaction_error = target_state.patient_satisfaction - current_state.patient_satisfaction
        public_satisfaction_adjustment = satisfaction_error * 0.8
        
        # 服务可及性偏差（公平性）
        accessibility_error = target_state.service_accessibility - current_state.service_accessibility
        equity_adjustment = accessibility_error * 0.9
        
        # 危机响应能力
        crisis_error = target_state.crisis_response_capability - current_state.crisis_response_capability
        crisis_adjustment = crisis_error * 0.7
        
        # 政策执行效果
        policy_effectiveness = 0.0
        if 'policy_implementation_rate' in context:
            implementation = context['policy_implementation_rate']
            if implementation > 0.9:  # 高执行率
                policy_effectiveness = 0.3
            elif implementation < 0.5:  # 低执行率
                policy_effectiveness = -0.3
        
        # 跨部门协调奖励
        coordination_bonus = 0.0
        if 'inter_department_coordination' in context:
            coordination = context['inter_department_coordination']
            coordination_bonus = coordination * 0.25
        
        # 公众信任度
        trust_bonus = 0.0
        if 'public_trust_score' in context:
            trust = context['public_trust_score']
            trust_bonus = (trust - 0.5) * 0.4  # 以50%为基准
        
        # 长期可持续性
        sustainability_bonus = 0.0
        if 'sustainability_index' in context:
            sustainability = context['sustainability_index']
            sustainability_bonus = sustainability * 0.3
        
        # 创新政策奖励
        innovation_bonus = 0.0
        if 'policy_innovation_score' in context:
            innovation = context['policy_innovation_score']
            innovation_bonus = innovation * 0.2
        
        total_adjustment = (compliance_adjustment + 
                          ethical_adjustment + 
                          public_satisfaction_adjustment + 
                          equity_adjustment + 
                          crisis_adjustment + 
                          policy_effectiveness + 
                          coordination_bonus + 
                          trust_bonus + 
                          sustainability_bonus + 
                          innovation_bonus)
        
        logger.debug(f"Government reward components: compliance={compliance_adjustment:.3f}, "
                    f"ethical={ethical_adjustment:.3f}, policy_effect={policy_effectiveness:.3f}")
        
        return total_adjustment