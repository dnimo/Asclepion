"""
Report 3 架构集成示例

演示如何将以下组件整合：
1. Fixed LLM Actor（固定参数生成器）
2. Semantic Critic（Q 值评估）
3. Holy Code 语义编码
4. Bellman 训练循环
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
import logging

from src.hospital_governance.agents.learning_models import (
    FixedLLMCandidateGenerator,
    NaturalLanguageActionParser,
    LLMGenerationResult,
    LLM_PARAMETERS_FROZEN
)
from src.hospital_governance.agents.semantic_critic import (
    SemanticEncoder,
    SemanticCritic,
    SemanticCriticTrainer,
    SemanticReplayBuffer,
    SemanticTransition,
    create_augmented_state
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Report3Agent:
    """
    基于 Report 3 架构的智能体
    
    核心流程：
    1. LLM 生成 K 个候选动作（参数冻结）
    2. Holy Code 编码为语义约束 ξ(HC_t)
    3. 构建增强状态 s̃_t = [φ(x_t), ξ(HC_t)]
    4. Critic 评估所有候选的 Q 值
    5. 选择 argmax Q 的动作执行
    6. 收集转换并训练 Critic（Bellman 更新）
    """
    
    def __init__(self,
                 role: str,
                 state_dim: int = 16,
                 hc_embedding_dim: int = 384,
                 action_embedding_dim: int = 384,
                 num_candidates: int = 5,
                 use_real_llm: bool = False):
        
        self.role = role
        self.state_dim = state_dim
        
        # 1. 固定 LLM 生成器
        self.llm_generator = FixedLLMCandidateGenerator(
            num_candidates=num_candidates,
            use_mock=not use_real_llm  # Convert to use_mock parameter
        )
        
        # 2. 语义编码器
        self.semantic_encoder = SemanticEncoder(
            llm_embedding_dim=hc_embedding_dim,
            use_mock=True
        )
        
        # 3. Critic 网络
        self.critic = SemanticCritic(
            state_dim=state_dim,
            hc_embedding_dim=hc_embedding_dim,
            action_embedding_dim=action_embedding_dim
        )
        
        # 4. Critic 训练器
        self.critic_trainer = SemanticCriticTrainer(
            critic=self.critic,
            encoder=self.semantic_encoder,
            lr=3e-4,
            gamma=0.99
        )
        
        # 5. 经验回放
        self.replay_buffer = SemanticReplayBuffer(capacity=10000)
        
        # 6. 动作解析器
        self.action_parser = NaturalLanguageActionParser()
        
        # 统计
        self.episode_count = 0
        self.training_steps = 0
    
    def select_action(self,
                     system_state: np.ndarray,
                     holy_code_state: Dict[str, Any],
                     exploration_epsilon: float = 0.0) -> Dict[str, Any]:
        """
        选择动作（Report 3 流程）
        
        Args:
            system_state: 16 维系统状态 φ(x_t)
            holy_code_state: Holy Code 状态
            exploration_epsilon: 探索率（0 = 纯利用）
            
        Returns:
            action_info: 包含动作文本、向量、Q 值等信息
        """
        # Step 1: LLM 生成候选（参数冻结）
        llm_result = self.llm_generator.generate_candidates(
            role=self.role,
            state={"system_state": system_state.tolist() if hasattr(system_state, 'tolist') else system_state},
            holy_code=holy_code_state
        )
        
        logger.info(f"✓ LLM 生成 {len(llm_result.candidates)} 个候选动作")
        
        # Step 2: 编码 Holy Code
        hc_embedding = self.semantic_encoder.encode_holy_code(holy_code_state)
        
        # Step 3: 构建增强状态
        augmented_state = create_augmented_state(system_state, hc_embedding)
        
        # Step 4: Exploration vs Exploitation
        if np.random.random() < exploration_epsilon:
            # 随机选择（探索）
            selected_idx = np.random.randint(len(llm_result.candidates))
            selected_action = llm_result.candidates[selected_idx]
            q_value = 0.0
            logger.info(f"🎲 探索模式：随机选择候选 {selected_idx}")
        else:
            # Critic 选择最优（利用）
            selected_action, q_value = self.critic_trainer.select_best_action(
                augmented_state=augmented_state,
                action_candidates=llm_result.candidates
            )
            selected_idx = llm_result.candidates.index(selected_action)
            logger.info(f"🎯 利用模式：选择最优候选 (Q={q_value:.3f})")
        
        # Step 5: 解析为动作向量
        action_vector = self.action_parser.parse(selected_action, self.role)
        
        # Step 6: 编码动作
        action_embedding = self.semantic_encoder.encode_action(selected_action)
        
        return {
            'action_text': selected_action,
            'action_vector': action_vector,
            'action_embedding': action_embedding.detach().cpu().numpy(),
            'q_value': q_value,
            'augmented_state': augmented_state,
            'candidates': llm_result.candidates,
            'selected_idx': selected_idx,
            'generation_id': llm_result.metadata.get('generation_id', 0),
            'holy_code_embedding': hc_embedding.detach().cpu().numpy()
        }
    
    def store_transition(self,
                        action_info: Dict[str, Any],
                        reward: float,
                        next_system_state: np.ndarray,
                        next_holy_code_state: Dict[str, Any],
                        done: bool):
        """
        存储转换到经验回放
        
        Args:
            action_info: select_action 返回的信息
            reward: 即时奖励 r_t
            next_system_state: 下一状态 x_{t+1}
            next_holy_code_state: 下一个 Holy Code 状态
            done: 是否终止
        """
        # 构建下一个增强状态
        next_hc_embedding = self.semantic_encoder.encode_holy_code(next_holy_code_state)
        next_augmented_state = create_augmented_state(next_system_state, next_hc_embedding)
        
        # 创建转换
        transition = SemanticTransition(
            augmented_state=action_info['augmented_state'],
            action_text=action_info['action_text'],
            action_embedding=action_info['action_embedding'],
            reward=reward,
            next_augmented_state=next_augmented_state,
            done=done,
            role=self.role
        )
        
        self.replay_buffer.add(transition)
        logger.info(f"💾 存储转换：reward={reward:.3f}, buffer_size={len(self.replay_buffer)}")
    
    def train_critic(self,
                    batch_size: int = 32,
                    next_candidates_fn: Optional[callable] = None) -> Dict[str, float]:
        """
        训练 Critic 网络（Bellman 更新）
        
        Args:
            batch_size: 批次大小
            next_candidates_fn: 生成下一状态候选的函数
            
        Returns:
            training_stats: 训练统计
        """
        if len(self.replay_buffer) < batch_size:
            logger.info(f"⏸️  经验不足：{len(self.replay_buffer)}/{batch_size}")
            return {}
        
        # 采样 batch
        batch = self.replay_buffer.sample(batch_size)
        
        # 默认候选生成函数
        if next_candidates_fn is None:
            def default_fn(role, state):
                # 使用 LLM 生成下一步候选
                result = self.llm_generator.generate_candidates(
                    role=role,
                    state={"augmented_state": state[:self.state_dim].tolist()},
                    holy_code=None
                )
                return result.candidates
            next_candidates_fn = default_fn
        
        # 训练步骤
        stats = self.critic_trainer.train_step(batch, next_candidates_fn)
        
        self.training_steps += 1
        
        logger.info(f"📊 训练步骤 {self.training_steps}: loss={stats.get('loss', 0):.4f}, "
                   f"mean_Q={stats.get('mean_q', 0):.3f}")
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'role': self.role,
            'episode_count': self.episode_count,
            'training_steps': self.training_steps,
            'replay_buffer_size': len(self.replay_buffer),
            'llm_generation_count': self.llm_generator.generation_count,
            'critic_stats': {
                'losses': self.critic_trainer.training_stats['losses'][-10:],
                'q_values': self.critic_trainer.training_stats['q_values'][-10:],
            }
        }


def demo_report3_architecture():
    """演示 Report 3 架构的完整流程"""
    
    print("=" * 80)
    print("Report 3 架构演示")
    print("=" * 80)
    
    # 创建 Agent
    agent = Report3Agent(
        role='doctors',
        num_candidates=5,
        use_real_llm=False  # 使用 Mock
    )
    
    print(f"\n✓ 创建 Agent: {agent.role}")
    print(f"  - LLM 参数冻结: {LLM_PARAMETERS_FROZEN}")
    print(f"  - 使用 Mock LLM: {agent.llm_generator.use_mock}")
    print(f"  - Critic 网络: {agent.critic.state_dim} + {agent.critic.hc_embedding_dim} → Q")
    
    # 模拟环境状态
    system_state = np.array([
        0.7,  # 患者满意度
        0.8,  # 医疗质量
        0.6,  # 资源利用率
        0.5,  # 成本效益
        0.7,  # 系统稳定性
        0.65, # 床位占用率
        0.4,  # 等待时间
        0.75, # 治疗成功率
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 填充到 16 维
    ])
    
    holy_code_state = {
        'active_rules': [
            {'description': '患者安全第一'},
            {'description': '公平分配医疗资源'},
            {'description': '知情同意原则'}
        ]
    }
    
    print(f"\n初始状态:")
    print(f"  - 系统状态: 患者满意度={system_state[0]:.2f}, 医疗质量={system_state[1]:.2f}")
    print(f"  - Holy Code: {len(holy_code_state['active_rules'])} 条活跃规则")
    
    # 运行 5 个 episodes
    for episode in range(5):
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}")
        print(f"{'='*80}")
        
        # Step 1: 选择动作
        action_info = agent.select_action(
            system_state=system_state,
            holy_code_state=holy_code_state,
            exploration_epsilon=0.2 if episode < 3 else 0.0  # 前3轮探索
        )
        
        print(f"\n📋 候选动作:")
        for i, cand in enumerate(action_info['candidates']):
            marker = "✓" if i == action_info['selected_idx'] else " "
            print(f"  [{marker}] {i+1}. {cand}")
        
        print(f"\n🎯 选定动作: {action_info['action_text']}")
        print(f"   Q 值: {action_info['q_value']:.3f}")
        print(f"   生成 ID: {action_info['generation_id']}")
        
        # Step 2: 模拟环境反馈
        reward = np.random.uniform(0.3, 0.9)  # 模拟奖励
        next_system_state = system_state + np.random.randn(16) * 0.05  # 状态演化
        next_system_state = np.clip(next_system_state, 0, 1)
        done = False
        
        print(f"\n📈 环境反馈:")
        print(f"   奖励: {reward:.3f}")
        print(f"   下一状态: 满意度={next_system_state[0]:.2f}, 质量={next_system_state[1]:.2f}")
        
        # Step 3: 存储转换
        agent.store_transition(
            action_info=action_info,
            reward=reward,
            next_system_state=next_system_state,
            next_holy_code_state=holy_code_state,
            done=done
        )
        
        # Step 4: 训练 Critic（从第 2 轮开始）
        if episode >= 1:
            print(f"\n🔧 训练 Critic:")
            train_stats = agent.train_critic(batch_size=min(4, len(agent.replay_buffer)))
            
            if train_stats:
                print(f"   损失: {train_stats.get('loss', 0):.4f}")
                print(f"   平均 Q: {train_stats.get('mean_q', 0):.3f}")
                print(f"   目标 Q: {train_stats.get('mean_target_q', 0):.3f}")
        
        # 更新状态
        system_state = next_system_state
        agent.episode_count += 1
    
    # 最终统计
    print(f"\n{'='*80}")
    print("最终统计")
    print(f"{'='*80}")
    
    stats = agent.get_statistics()
    print(f"\nAgent: {stats['role']}")
    print(f"  - Episodes: {stats['episode_count']}")
    print(f"  - Training steps: {stats['training_steps']}")
    print(f"  - Replay buffer: {stats['replay_buffer_size']}")
    print(f"  - LLM generation count: {stats['llm_generation_count']}")
    print(f"  - Parameters frozen: {LLM_PARAMETERS_FROZEN}")
    
    print(f"\nCritic 训练曲线（最近 10 步）:")
    if stats['critic_stats']['losses']:
        print(f"  - 损失: {[f'{x:.4f}' for x in stats['critic_stats']['losses']]}")
        print(f"  - Q 值: {[f'{x:.3f}' for x in stats['critic_stats']['q_values']]}")
    
    print(f"\n✓ Report 3 架构演示完成！")


if __name__ == "__main__":
    demo_report3_architecture()
