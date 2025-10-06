import numpy as np
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .rule_engine import RuleEngine

class VoteType(Enum):
    """投票类型"""
    RESOURCE_ALLOCATION = "resource_allocation"
    RULE_AMENDMENT = "rule_amendment" 
    POLICY_CHANGE = "policy_change"
    CRISIS_RESPONSE = "crisis_response"
    BUDGET_APPROVAL = "budget_approval"

@dataclass
class ParliamentConfig:
    """议会配置"""
    vote_weights: Dict[str, float]          # 投票权重
    decision_threshold: float = 0.7         # 决策阈值
    proposal_timeout: int = 100             # 提案超时
    consensus_requirement: float = 0.8      # 共识要求

class Parliament:
    """议会决策系统"""
    
    def __init__(self, rule_engine: 'RuleEngine', config: ParliamentConfig):
        self.rule_engine = rule_engine
        self.config = config
        
        # 决策状态
        self.active_proposals: Dict[str, Dict] = {}
        self.voting_history: List[Dict] = []
        self.decision_history: List[Dict] = []
        self.member_contributions: Dict[str, int] = {}
        
        # 性能指标
        self.consensus_metrics = {
            'total_decisions': 0,
            'approved_decisions': 0,
            'average_approval_rate': 0.0,
            'consensus_time_history': []
        }
    
    def submit_proposal(self, proposal: Dict[str, Any], proposer: str) -> str:
        """提交提案"""
        proposal_id = f"proposal_{len(self.active_proposals) + 1}"
        
        proposal_data = {
            'id': proposal_id,
            'content': proposal,
            'proposer': proposer,
            'votes': {},
            'status': 'pending',
            'submission_time': len(self.voting_history),
            'context': proposal.get('context', 'general')
        }
        
        self.active_proposals[proposal_id] = proposal_data
        
        # 更新成员贡献
        self.member_contributions[proposer] = self.member_contributions.get(proposer, 0) + 1
        
        print(f"新提案提交: {proposal_id} by {proposer}")
        return proposal_id
    
    def cast_vote(self, proposal_id: str, voter: str, vote: bool, 
                 rationale: str = "") -> bool:
        """投票"""
        if proposal_id not in self.active_proposals:
            return False
        
        proposal = self.active_proposals[proposal_id]
        
        # 检查投票权重
        voter_weight = self.config.vote_weights.get(voter, 0.1)
        
        # 评估投票合理性（使用规则引擎）
        vote_context = {
            'proposal': proposal['content'],
            'voter_role': voter,
            'rationale': rationale,
            'context_type': proposal['context']
        }
        
        rule_evaluation = self.rule_engine.evaluate_rules(vote_context)
        rule_compliance = len(rule_evaluation) > 0
        
        # 记录投票
        proposal['votes'][voter] = {
            'vote': vote,
            'rationale': rationale,
            'weight': voter_weight,
            'rule_compliance': rule_compliance,
            'rule_evaluations': rule_evaluation
        }
        
        # 记录投票历史
        vote_record = {
            'proposal_id': proposal_id,
            'voter': voter,
            'vote': vote,
            'rationale': rationale,
            'weight': voter_weight,
            'timestamp': len(self.voting_history)
        }
        self.voting_history.append(vote_record)
        
        return True
    
    def tally_votes(self, proposal_id: str) -> Tuple[bool, float, Dict[str, Any]]:
        """统计投票结果"""
        if proposal_id not in self.active_proposals:
            return False, 0.0, {}
        
        proposal = self.active_proposals[proposal_id]
        votes = proposal['votes']
        
        if not votes:
            return False, 0.0, {}
        
        # 计算加权投票结果
        total_weight = 0.0
        yes_weight = 0.0
        no_weight = 0.0
        
        voter_analysis = {}
        
        for voter, vote_data in votes.items():
            weight = vote_data['weight']
            vote = vote_data['vote']
            
            total_weight += weight
            
            if vote:
                yes_weight += weight
            else:
                no_weight += weight
            
            # 分析投票模式
            voter_analysis[voter] = {
                'vote': vote,
                'weight': weight,
                'rule_compliance': vote_data['rule_compliance'],
                'rationale': vote_data.get('rationale', '')
            }
        
        # 计算批准率
        approval_rate = yes_weight / total_weight if total_weight > 0 else 0.0
        approved = approval_rate >= self.config.decision_threshold
        
        # 更新提案状态
        proposal['status'] = 'approved' if approved else 'rejected'
        proposal['approval_rate'] = approval_rate
        proposal['vote_analysis'] = voter_analysis
        
        # 记录决策历史
        decision_record = {
            'proposal_id': proposal_id,
            'proposer': proposal['proposer'],
            'approved': approved,
            'approval_rate': approval_rate,
            'total_voters': len(votes),
            'context': proposal['context'],
            'timestamp': len(self.decision_history)
        }
        self.decision_history.append(decision_record)
        
        # 更新性能指标
        self.consensus_metrics['total_decisions'] += 1
        if approved:
            self.consensus_metrics['approved_decisions'] += 1
        
        # 计算平均批准率
        recent_decisions = self.decision_history[-50:]  # 最近50个决策
        if recent_decisions:
            avg_approval = np.mean([d['approval_rate'] for d in recent_decisions])
            self.consensus_metrics['average_approval_rate'] = avg_approval
        
        # 记录共识形成时间
        consensus_time = len(self.voting_history) - proposal['submission_time']
        self.consensus_metrics['consensus_time_history'].append(consensus_time)
        
        # 从活跃提案中移除
        del self.active_proposals[proposal_id]
        
        return approved, approval_rate, voter_analysis
    
    def evaluate_proposal_with_rules(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """使用规则引擎评估提案"""
        evaluation_context = {
            'proposal': proposal,
            'context_type': proposal.get('context', 'general'),
            'state': proposal.get('current_state', {}),
            'timestamp': len(self.voting_history)
        }
        
        activated_rules = self.rule_engine.evaluate_rules(evaluation_context)
        
        # 计算规则合规分数
        total_weight = 0.0
        compliance_score = 0.0
        
        for rule in activated_rules:
            weight = rule['weight']
            action = rule['action_result']
            
            total_weight += weight
            compliance_score += weight * action.get('weight_adjustment', 1.0)
        
        if total_weight > 0:
            compliance_score /= total_weight
        
        return {
            'activated_rules': activated_rules,
            'compliance_score': compliance_score,
            'recommendations': self._extract_recommendations(activated_rules),
            'priority_boost': self._calculate_priority_boost(activated_rules)
        }
    
    def _extract_recommendations(self, activated_rules: List[Dict]) -> List[str]:
        """从激活的规则中提取建议"""
        recommendations = []
        for rule in activated_rules:
            action_result = rule['action_result']
            rule_recommendations = action_result.get('recommendations', [])
            recommendations.extend(rule_recommendations)
        
        # 去重并限制数量
        return list(dict.fromkeys(recommendations))[:5]
    
    def _calculate_priority_boost(self, activated_rules: List[Dict]) -> float:
        """计算优先级提升"""
        if not activated_rules:
            return 0.0
        
        total_boost = 0.0
        for rule in activated_rules:
            action_result = rule['action_result']
            total_boost += action_result.get('priority_boost', 0.0)
        
        return min(1.0, total_boost)
    
    def get_consensus_metrics(self) -> Dict[str, Any]:
        """获取共识形成指标"""
        if not self.consensus_metrics['consensus_time_history']:
            return self.consensus_metrics
        
        consensus_times = self.consensus_metrics['consensus_time_history']
        
        metrics = self.consensus_metrics.copy()
        metrics.update({
            'average_consensus_time': np.mean(consensus_times),
            'min_consensus_time': np.min(consensus_times),
            'max_consensus_time': np.max(consensus_times),
            'consensus_time_std': np.std(consensus_times),
            'approval_rate': (self.consensus_metrics['approved_decisions'] / 
                            self.consensus_metrics['total_decisions'] 
                            if self.consensus_metrics['total_decisions'] > 0 else 0.0)
        })
        
        return metrics
    
    def get_member_performance(self) -> Dict[str, Dict[str, Any]]:
        """获取成员绩效"""
        performance = {}
        
        for member, contributions in self.member_contributions.items():
            # 分析成员的投票历史
            member_votes = [v for v in self.voting_history if v['voter'] == member]
            member_proposals = [d for d in self.decision_history if d['proposer'] == member]
            
            if member_votes:
                approval_votes = sum(1 for v in member_votes if v['vote'])
                total_votes = len(member_votes)
                vote_approval_rate = approval_votes / total_votes
                
                # 规则合规性
                compliant_votes = sum(1 for v in member_votes 
                                    if self.voting_history.index(v) < len(self.rule_engine.rule_history))
            else:
                vote_approval_rate = 0.0
                compliant_votes = 0
            
            # 提案成功率
            approved_proposals = sum(1 for p in member_proposals if p['approved'])
            total_proposals = len(member_proposals)
            proposal_success_rate = approved_proposals / total_proposals if total_proposals > 0 else 0.0
            
            performance[member] = {
                'contributions': contributions,
                'total_votes': len(member_votes),
                'vote_approval_rate': vote_approval_rate,
                'total_proposals': total_proposals,
                'approved_proposals': approved_proposals,
                'proposal_success_rate': proposal_success_rate,
                'vote_weight': self.config.vote_weights.get(member, 0.1)
            }
        
        return performance
    
    def cleanup_expired_proposals(self) -> int:
        """清理过期提案"""
        current_time = len(self.voting_history)
        expired_proposals = []
        
        for proposal_id, proposal in self.active_proposals.items():
            submission_time = proposal['submission_time']
            if current_time - submission_time > self.config.proposal_timeout:
                expired_proposals.append(proposal_id)
        
        for proposal_id in expired_proposals:
            del self.active_proposals[proposal_id]
        
        return len(expired_proposals)