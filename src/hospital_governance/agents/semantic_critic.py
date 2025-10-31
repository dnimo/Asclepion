"""
Semantic Critic with LLM-based Action and Holy Code Embedding

基于 Report 3 的核心架构：
- LLM 作为固定 Actor 生成候选动作
- Critic 学习价值函数 Q_θ(s̃_t, a_t) = g_θ([s̃_t, ψ(a_t)])
- Holy Code 嵌入 ξ(HC_t) 作为语义约束
- 增强状态 s̃_t = [φ(x_t), ξ(HC_t)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticTransition:
    """语义转换：包含增强状态、动作嵌入和奖励"""
    augmented_state: np.ndarray  # s̃_t = [φ(x_t), ξ(HC_t)]
    action_text: str
    action_embedding: np.ndarray  # ψ(a_t)
    reward: float
    next_augmented_state: np.ndarray  # s̃_{t+1}
    done: bool
    role: str


class SemanticEncoder(nn.Module):
    """
    语义编码器：使用 LLM 的倒数第二层编码动作和 Holy Code
    
    ψ(a_t) = f_enc^LLM(a_t)
    ξ(HC_t) = f_enc^LLM(HC_t)
    """
    
    def __init__(self, 
                 llm_embedding_dim: int = 384,
                 use_mock: bool = True):
        super().__init__()
        self.embedding_dim = llm_embedding_dim
        self.use_mock = use_mock
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(llm_embedding_dim)
        
        # 缓存：避免重复编码
        self._cache: Dict[str, torch.Tensor] = {}
        
        if not use_mock:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("使用 sentence-transformers 进行语义编码")
            except ImportError:
                logger.warning("sentence-transformers 未安装，使用 mock 编码")
                self.use_mock = True
    
    def encode_action(self, action_text: str) -> torch.Tensor:
        """
        编码动作文本为语义向量 ψ(a_t)
        
        Args:
            action_text: 自然语言动作描述
            
        Returns:
            action_embedding: [embedding_dim] 归一化后的语义向量
        """
        cache_key = f"action:{action_text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self.use_mock:
            # Mock: 基于文本哈希的确定性向量
            hash_val = hash(action_text)
            np.random.seed(hash_val % (2**32))
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        else:
            embedding = self.encoder.encode(action_text, convert_to_numpy=True)
        
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        embedding_tensor = self.layer_norm(embedding_tensor)
        
        self._cache[cache_key] = embedding_tensor
        return embedding_tensor
    
    def encode_holy_code(self, holy_code_state: Dict[str, Any]) -> torch.Tensor:
        """
        编码 Holy Code 为语义约束向量 ξ(HC_t)
        
        Args:
            holy_code_state: 包含活跃规则和伦理约束的字典
            
        Returns:
            hc_embedding: [embedding_dim] 归一化后的语义向量
        """
        # 将 Holy Code 转换为文本表示
        hc_text = self._holy_code_to_text(holy_code_state)
        
        cache_key = f"hc:{hash(hc_text)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self.use_mock:
            hash_val = hash(hc_text)
            np.random.seed(hash_val % (2**32))
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        else:
            embedding = self.encoder.encode(hc_text, convert_to_numpy=True)
        
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        embedding_tensor = self.layer_norm(embedding_tensor)
        
        self._cache[cache_key] = embedding_tensor
        return embedding_tensor
    
    def _holy_code_to_text(self, hc_state: Dict[str, Any]) -> str:
        """将 Holy Code 状态转换为文本描述"""
        if not hc_state:
            return "无活跃伦理规则"
        
        rules = hc_state.get('active_rules', [])
        if not rules:
            return "无活跃伦理规则"
        
        text_parts = ["活跃伦理规则："]
        for rule in rules[:5]:  # 最多5条
            if isinstance(rule, dict):
                text_parts.append(f"- {rule.get('description', str(rule))}")
            else:
                text_parts.append(f"- {str(rule)}")
        
        return "\n".join(text_parts)
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


class SemanticCritic(nn.Module):
    """
    语义 Critic 网络
    
    Q_θ(s̃_t, a_t) = g_θ([s̃_t, ψ(a_t)])
    
    其中 s̃_t = [φ(x_t), ξ(HC_t)] 是增强状态
    """
    
    def __init__(self,
                 state_dim: int = 16,
                 hc_embedding_dim: int = 384,
                 action_embedding_dim: int = 384,
                 hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.state_dim = state_dim
        self.hc_embedding_dim = hc_embedding_dim
        self.action_embedding_dim = action_embedding_dim
        
        # 增强状态维度 = state_dim + hc_embedding_dim
        augmented_state_dim = state_dim + hc_embedding_dim
        
        # 输入维度 = 增强状态 + 动作嵌入
        input_dim = augmented_state_dim + action_embedding_dim
        
        # 构建深度网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # 输出层：Q 值
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, augmented_state: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Q 值
        
        Args:
            augmented_state: [batch, state_dim + hc_embedding_dim]
            action_embedding: [batch, action_embedding_dim]
            
        Returns:
            q_values: [batch, 1]
        """
        # 拼接增强状态和动作嵌入
        x = torch.cat([augmented_state, action_embedding], dim=-1)
        
        # 通过网络计算 Q 值
        q_values = self.network(x)
        
        return q_values


class SemanticReplayBuffer:
    """
    语义经验回放缓冲区
    存储 (s̃_t, a_t, r_t, s̃_{t+1}) 四元组
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, transition: SemanticTransition):
        """添加转换"""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[SemanticTransition]:
        """采样 batch"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class SemanticCriticTrainer:
    """
    Semantic Critic 训练器
    
    实现 Bellman 目标和优化循环：
    y_t = r_t + γ max_{a' ∈ A_{t+1}} Q_{θ⁻}(s̃_{t+1}, a')
    L(θ) = E[(Q_θ(s̃_t, a_t) - y_t)²] + λ_reg ||θ||²
    """
    
    def __init__(self,
                 critic: SemanticCritic,
                 encoder: SemanticEncoder,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 reg_lambda: float = 1e-5,
                 target_update_freq: int = 100,
                 device: str = 'cpu'):
        
        self.critic = critic.to(device)
        self.encoder = encoder.to(device)
        self.device = device
        
        # 目标网络 θ⁻
        self.target_critic = SemanticCritic(
            state_dim=critic.state_dim,
            hc_embedding_dim=critic.hc_embedding_dim,
            action_embedding_dim=critic.action_embedding_dim
        ).to(device)
        self.target_critic.load_state_dict(critic.state_dict())
        self.target_critic.eval()
        
        # 优化器
        self.optimizer = torch.optim.Adam(critic.parameters(), lr=lr, weight_decay=reg_lambda)
        
        # 超参数
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # 统计
        self.training_stats = {
            'losses': [],
            'q_values': [],
            'target_q_values': []
        }
    
    def compute_target(self,
                      rewards: torch.Tensor,
                      next_augmented_states: torch.Tensor,
                      next_action_candidates: List[List[str]],
                      dones: torch.Tensor) -> torch.Tensor:
        """
        计算 Bellman 目标
        
        y_t = r_t + γ max_{a' ∈ A_{t+1}} Q_{θ⁻}(s̃_{t+1}, a')
        
        Args:
            rewards: [batch]
            next_augmented_states: [batch, aug_state_dim]
            next_action_candidates: 每个样本的下一步候选动作列表
            dones: [batch]
            
        Returns:
            targets: [batch]
        """
        batch_size = rewards.shape[0]
        max_q_values = []
        
        with torch.no_grad():
            for i in range(batch_size):
                if dones[i]:
                    max_q_values.append(0.0)
                    continue
                
                # 获取所有候选动作的 Q 值
                candidates = next_action_candidates[i]
                if not candidates:
                    max_q_values.append(0.0)
                    continue
                
                q_vals = []
                for candidate in candidates:
                    action_emb = self.encoder.encode_action(candidate).to(self.device)
                    action_emb = action_emb.unsqueeze(0)  # [1, action_dim]
                    
                    next_state = next_augmented_states[i:i+1]  # [1, aug_state_dim]
                    q_val = self.target_critic(next_state, action_emb)
                    q_vals.append(q_val.item())
                
                max_q_values.append(max(q_vals))
        
        max_q_tensor = torch.tensor(max_q_values, device=self.device, dtype=torch.float32)
        targets = rewards + self.gamma * max_q_tensor * (1 - dones)
        
        return targets
    
    def train_step(self, batch: List[SemanticTransition], 
                   next_candidates_fn: callable) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch: 采样的转换批次
            next_candidates_fn: 函数，输入角色和状态，返回候选动作列表
            
        Returns:
            stats: 训练统计
        """
        if not batch:
            return {}
        
        # 准备数据
        augmented_states = torch.tensor(
            np.stack([t.augmented_state for t in batch]),
            dtype=torch.float32,
            device=self.device
        )
        
        action_embeddings = torch.stack([
            self.encoder.encode_action(t.action_text).to(self.device)
            for t in batch
        ])
        
        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self.device
        )
        
        next_augmented_states = torch.tensor(
            np.stack([t.next_augmented_state for t in batch]),
            dtype=torch.float32,
            device=self.device
        )
        
        dones = torch.tensor(
            [float(t.done) for t in batch],
            dtype=torch.float32,
            device=self.device
        )
        
        # 获取下一步候选动作
        next_candidates = []
        for t in batch:
            candidates = next_candidates_fn(t.role, t.next_augmented_state)
            next_candidates.append(candidates)
        
        # 计算当前 Q 值
        current_q = self.critic(augmented_states, action_embeddings).squeeze(-1)
        
        # 计算目标 Q 值
        target_q = self.compute_target(rewards, next_augmented_states, next_candidates, dones)
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 记录统计
        stats = {
            'loss': loss.item(),
            'mean_q': current_q.mean().item(),
            'mean_target_q': target_q.mean().item(),
            'max_q': current_q.max().item(),
            'min_q': current_q.min().item()
        }
        
        self.training_stats['losses'].append(loss.item())
        self.training_stats['q_values'].append(current_q.mean().item())
        self.training_stats['target_q_values'].append(target_q.mean().item())
        
        return stats
    
    def evaluate_action(self,
                       augmented_state: np.ndarray,
                       action_text: str) -> float:
        """
        评估单个动作的 Q 值
        
        Args:
            augmented_state: 增强状态 s̃_t
            action_text: 动作文本
            
        Returns:
            q_value: 标量 Q 值
        """
        self.critic.eval()
        
        with torch.no_grad():
            state_tensor = torch.tensor(
                augmented_state,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            
            action_emb = self.encoder.encode_action(action_text).to(self.device).unsqueeze(0)
            
            q_value = self.critic(state_tensor, action_emb).item()
        
        self.critic.train()
        return q_value
    
    def select_best_action(self,
                          augmented_state: np.ndarray,
                          action_candidates: List[str]) -> Tuple[str, float]:
        """
        从候选中选择最优动作
        
        Args:
            augmented_state: 增强状态
            action_candidates: 候选动作列表
            
        Returns:
            best_action: 最优动作文本
            best_q: 对应的 Q 值
        """
        if not action_candidates:
            raise ValueError("候选动作列表为空")
        
        q_values = []
        for action in action_candidates:
            q_val = self.evaluate_action(augmented_state, action)
            q_values.append(q_val)
        
        best_idx = np.argmax(q_values)
        return action_candidates[best_idx], q_values[best_idx]
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_counter': self.update_counter,
            'training_stats': self.training_stats
        }, path)
        logger.info(f"Semantic Critic 模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_counter = checkpoint['update_counter']
        self.training_stats = checkpoint['training_stats']
        logger.info(f"Semantic Critic 模型已从 {path} 加载")


def create_augmented_state(system_state: np.ndarray,
                           holy_code_embedding: torch.Tensor) -> np.ndarray:
    """
    创建增强状态 s̃_t = [φ(x_t), ξ(HC_t)]
    
    Args:
        system_state: 系统状态向量 φ(x_t)，长度 16
        holy_code_embedding: Holy Code 嵌入 ξ(HC_t)
        
    Returns:
        augmented_state: 增强状态，长度 16 + embedding_dim
    """
    if isinstance(holy_code_embedding, torch.Tensor):
        hc_np = holy_code_embedding.detach().cpu().numpy()
    else:
        hc_np = holy_code_embedding
    
    augmented = np.concatenate([system_state.astype(np.float32), hc_np.astype(np.float32)])
    return augmented
