"""
Report 3 Agent - 集成 Fixed LLM Actor + Semantic Critic

基于 Report 3 架构的完整实现：
- 继承 RoleAgent 基类
- 使用 FixedLLMCandidateGenerator 生成候选动作（参数冻结）
- 使用 SemanticCritic 评估动作价值
- 通过 Bellman 更新训练 Critic
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging

from .role_agents import RoleAgent, AgentConfig, AgentState, SystemState
from .learning_models import (
    FixedLLMCandidateGenerator,
    NaturalLanguageActionParser,
    LLM_PARAMETERS_FROZEN
)
from .semantic_critic import (
    SemanticEncoder,
    SemanticCritic,
    SemanticCriticTrainer,
    SemanticReplayBuffer,
    SemanticTransition,
    create_augmented_state
)

logger = logging.getLogger(__name__)


class Report3Agent(RoleAgent):
    """
    Report 3 架构的智能体实现
    
    核心特性：
    1. Fixed LLM Actor：生成 K 个候选动作（参数冻结）
    2. Semantic Critic：评估 Q(s̃_t, a_t)，其中 s̃_t = [φ(x_t), ξ(HC_t)]
    3. Bellman 训练：通过经验回放训练 Critic
    4. Holy Code 集成：语义嵌入作为状态增强
    """
    
    def __init__(
        self,
        config: AgentConfig,
        num_candidates: int = 5,
        hc_embedding_dim: int = 384,
        action_embedding_dim: int = 384,
        use_mock_llm: bool = True,
        replay_buffer_capacity: int = 10000,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        # 初始化父类
        super().__init__(config)
        
        self.num_candidates = num_candidates
        self.device = device
        
        # 1. Fixed LLM 候选生成器
        self.llm_generator = FixedLLMCandidateGenerator(
            num_candidates=num_candidates,
            use_mock=use_mock_llm
        )
        
        # 2. 自然语言动作解析器
        self.action_parser = NaturalLanguageActionParser(
            action_dim=config.action_dim
        )
        
        # 3. 语义编码器
        self.semantic_encoder = SemanticEncoder(
            llm_embedding_dim=hc_embedding_dim,
            use_mock=use_mock_llm
        )
        
        # 4. Semantic Critic 网络
        self.critic = SemanticCritic(
            state_dim=16,  # 固定使用16维全局状态
            hc_embedding_dim=hc_embedding_dim,
            action_embedding_dim=action_embedding_dim
        )
        
        # 5. Critic 训练器
        self.critic_trainer = SemanticCriticTrainer(
            critic=self.critic,
            encoder=self.semantic_encoder,
            lr=critic_lr,
            gamma=gamma,
            device=device
        )
        
        # 6. 经验回放缓冲区
        self.replay_buffer = SemanticReplayBuffer(capacity=replay_buffer_capacity)
        
        # 统计
        self.episode_count = 0
        self.training_steps = 0
        self.generation_count = 0
        
        logger.info(
            f"✓ Initialized Report3Agent ({config.role}): "
            f"LLM frozen={LLM_PARAMETERS_FROZEN}, K={num_candidates}, "
            f"critic_dim={16 + hc_embedding_dim + action_embedding_dim}"
        )
    
    def select_action(
        self,
        observation: np.ndarray,
        holy_code_guidance: Optional[Dict[str, Any]] = None,
        training: bool = False,
        exploration_epsilon: float = 0.0
    ) -> np.ndarray:
        """
        选择动作：LLM 生成候选 → Critic 评估 → 选择最优
        
        Args:
            observation: 局部观测（会被全局状态覆盖）
            holy_code_guidance: Holy Code 指导
            training: 是否训练模式
            exploration_epsilon: 探索率（> 0 则随机选择）
            
        Returns:
            action_vector: 选定的动作向量
        """
        # 使用全局16维状态（如果已注入）
        if self._global_state_vector is not None and len(self._global_state_vector) == 16:
            system_state = self._global_state_vector
        else:
            # 回退：扩展局部观测到16维
            system_state = np.zeros(16)
            system_state[:min(len(observation), 16)] = observation[:16]
        
        # Holy Code 状态
        holy_code_state = holy_code_guidance or {
            'active_rules': ['Patient safety first', 'Resource optimization', 'Ethical compliance'],
            'priority_level': 0.8
        }
        
        # Step 1: LLM 生成候选动作（参数冻结）
        llm_result = self.llm_generator.generate_candidates(
            role=self.role,
            state={'system_state': system_state.tolist()},
            holy_code=holy_code_state
        )
        
        self.generation_count += 1
        
        # Step 2: 编码 Holy Code
        hc_embedding = self.semantic_encoder.encode_holy_code(holy_code_state)
        
        # Step 3: 构建增强状态 s̃_t = [φ(x_t), ξ(HC_t)]
        augmented_state = create_augmented_state(system_state, hc_embedding)
        
        # Step 4: 探索 vs 利用
        if training and np.random.random() < exploration_epsilon:
            # 探索：随机选择候选
            selected_idx = np.random.randint(0, len(llm_result.candidates))
            selected_action = llm_result.candidates[selected_idx]
            q_value = 0.0
            logger.info(f"🎲 探索模式：随机选择候选 {selected_idx}")
        else:
            # 利用：Critic 选择最优候选
            selected_action, q_value = self.critic_trainer.select_best_action(
                augmented_state=augmented_state,
                action_candidates=llm_result.candidates
            )
            selected_idx = llm_result.candidates.index(selected_action)
            logger.info(f"🎯 利用模式：选择最优候选 (Q={q_value:.3f})")
        
        # Step 5: 解析为动作向量
        action_vector = self.action_parser.parse(selected_action, role=self.role)
        
        # Step 6: 缓存动作信息（用于后续经验存储）
        action_embedding = self.semantic_encoder.encode_action(selected_action)
        
        self._last_action_info = {
            'action_text': selected_action,
            'action_vector': action_vector,
            'action_embedding': action_embedding.detach().cpu().numpy(),
            'augmented_state': augmented_state,
            'q_value': q_value,
            'candidates': llm_result.candidates,
            'selected_idx': selected_idx,
            'generation_id': llm_result.metadata['generation_id']
        }
        
        return action_vector
    
    def store_transition(
        self,
        reward: float,
        next_observation: np.ndarray,
        next_holy_code_guidance: Optional[Dict[str, Any]] = None,
        done: bool = False
    ):
        """
        存储转换到经验回放缓冲区
        
        Args:
            reward: 奖励
            next_observation: 下一状态观测
            next_holy_code_guidance: 下一状态的 Holy Code
            done: 是否终止
        """
        if not hasattr(self, '_last_action_info'):
            logger.warning("⚠️ No action info to store (call select_action first)")
            return
        
        # 构建下一状态
        if self._global_state_vector is not None and len(self._global_state_vector) == 16:
            next_system_state = self._global_state_vector
        else:
            next_system_state = np.zeros(16)
            next_system_state[:min(len(next_observation), 16)] = next_observation[:16]
        
        # 下一 Holy Code 状态
        next_hc_state = next_holy_code_guidance or {
            'active_rules': ['Patient safety first'],
            'priority_level': 0.8
        }
        
        # 编码下一 Holy Code
        next_hc_embedding = self.semantic_encoder.encode_holy_code(next_hc_state)
        
        # 构建下一增强状态
        next_augmented_state = create_augmented_state(next_system_state, next_hc_embedding)
        
        # 创建转换
        transition = SemanticTransition(
            augmented_state=self._last_action_info['augmented_state'],
            action_text=self._last_action_info['action_text'],
            action_embedding=self._last_action_info['action_embedding'],
            reward=reward,
            next_augmented_state=next_augmented_state,
            done=done,
            role=self.role
        )
        
        # 存储
        self.replay_buffer.add(transition)
        
        logger.info(f"💾 存储转换：reward={reward:.3f}, buffer_size={len(self.replay_buffer)}")
    
    def train_critic(
        self,
        batch_size: int = 32,
        num_epochs: int = 1
    ) -> Dict[str, float]:
        """
        训练 Semantic Critic（Bellman 更新）
        
        Args:
            batch_size: 批次大小
            num_epochs: 训练轮数
            
        Returns:
            训练统计
        """
        if len(self.replay_buffer) < batch_size:
            logger.info(f"⏸️  经验不足：{len(self.replay_buffer)}/{batch_size}")
            return {}
        
        # 定义候选生成函数（用于计算 Bellman 目标）
        def next_candidates_fn(role: str, state: np.ndarray) -> List[str]:
            result = self.llm_generator.generate_candidates(
                role=role,
                state={'augmented_state': state[:16].tolist()},
                holy_code=None
            )
            return result.candidates
        
        # 训练循环
        total_stats = {}
        
        for epoch in range(num_epochs):
            batch = self.replay_buffer.sample(batch_size)
            stats = self.critic_trainer.train_step(batch, next_candidates_fn)
            
            # 累加统计
            for k, v in stats.items():
                if k not in total_stats:
                    total_stats[k] = 0.0
                total_stats[k] += v
            
            self.training_steps += 1
            
            logger.info(
                f"📊 训练步骤 {self.training_steps}: "
                f"loss={stats['loss']:.4f}, mean_Q={stats['mean_q']:.3f}"
            )
        
        # 平均统计
        for k in total_stats:
            total_stats[k] /= num_epochs
        
        return total_stats
    
    def observe(self, environment: Dict[str, Any]) -> np.ndarray:
        """实现父类的抽象方法：观察环境"""
        # 从环境中提取角色相关的观测
        obs = np.array([
            environment.get('medical_resource_utilization', 0.7),
            environment.get('patient_waiting_time', 0.3),
            environment.get('financial_indicator', 0.7),
            environment.get('ethical_compliance', 0.9),
            environment.get('education_training_quality', 0.8),
            environment.get('patient_satisfaction', 0.85),
            environment.get('operational_efficiency', 0.75),
            environment.get('crisis_response_capability', 0.8)
        ])
        return obs
    
    def compute_local_value(self, system_state: SystemState, action: int) -> float:
        """实现父类的抽象方法：计算局部价值"""
        # 使用 Critic 网络评估价值
        # 这里需要将 SystemState 转换为向量
        state_vec = self._system_state_to_vector(system_state)
        
        # 构建增强状态（使用默认 Holy Code）
        hc_embedding = self.semantic_encoder.encode_holy_code({
            'active_rules': ['Default rule'],
            'priority_level': 0.5
        })
        augmented_state = create_augmented_state(state_vec, hc_embedding)
        
        # 生成候选动作
        candidates = self.llm_generator.generate_candidates(
            role=self.role,
            state={'system_state': state_vec.tolist()},
            holy_code=None
        )
        
        # 评估第一个候选
        if candidates.candidates:
            _, q_value = self.critic_trainer.select_best_action(
                augmented_state=augmented_state,
                action_candidates=candidates.candidates[:1]
            )
            return q_value
        
        return 0.0
    
    def _apply_holy_code_recommendations(
        self,
        action: np.ndarray,
        recommendations: List[str]
    ) -> np.ndarray:
        """实现父类的抽象方法：应用 Holy Code 推荐"""
        # Report 3 架构中，Holy Code 已经通过语义嵌入集成
        # 这里保持动作不变
        return action
    
    def _system_state_to_vector(self, system_state: SystemState) -> np.ndarray:
        """将 SystemState 对象转换为16维向量"""
        return np.array([
            system_state.medical_resource_utilization,
            system_state.patient_waiting_time,
            system_state.financial_indicator,
            getattr(system_state, 'ethical_compliance', 0.9),
            getattr(system_state, 'education_training_quality', 0.8),
            getattr(system_state, 'intern_satisfaction', 0.7),
            getattr(system_state, 'patient_satisfaction', 0.85),
            getattr(system_state, 'service_accessibility', 0.8),
            getattr(system_state, 'care_quality_index', 0.9),
            getattr(system_state, 'safety_incident_rate', 0.05),
            getattr(system_state, 'operational_efficiency', 0.75),
            getattr(system_state, 'staff_workload_balance', 0.7),
            getattr(system_state, 'crisis_response_capability', 0.8),
            getattr(system_state, 'regulatory_compliance_score', 0.9),
            getattr(system_state, 'innovation_index', 0.7),
            getattr(system_state, 'sustainability_score', 0.8)
        ])
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'role': self.role,
            'episode_count': self.episode_count,
            'training_steps': self.training_steps,
            'generation_count': self.generation_count,
            'replay_buffer_size': len(self.replay_buffer),
            'parameters_frozen': LLM_PARAMETERS_FROZEN,
            'critic_stats': {
                'losses': self.critic_trainer.training_stats['losses'][-10:],
                'q_values': self.critic_trainer.training_stats['q_values'][-10:],
            }
        }


def create_report3_agent(role: str, **kwargs) -> Report3Agent:
    """
    创建 Report3Agent 的工厂函数
    
    Args:
        role: 角色名称 ('doctors', 'interns', 'patients', etc.)
        **kwargs: 额外配置参数
        
    Returns:
        Report3Agent 实例
    """
    # 角色特定配置
    role_configs = {
        'doctors': AgentConfig(role='doctors', action_dim=17, observation_dim=8),
        'interns': AgentConfig(role='interns', action_dim=17, observation_dim=8),
        'patients': AgentConfig(role='patients', action_dim=17, observation_dim=8),
        'accountants': AgentConfig(role='accountants', action_dim=17, observation_dim=8),
        'government': AgentConfig(role='government', action_dim=17, observation_dim=8)
    }
    
    config = role_configs.get(role, AgentConfig(role=role, action_dim=17, observation_dim=8))
    
    return Report3Agent(config, **kwargs)
