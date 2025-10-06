import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Tuple
import os

class LearningModel:
    """基础学习模型类"""
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        """预测动作"""
        return np.random.randn(self.action_dim)
        
    def train(self, experiences: List[Tuple]) -> Dict[str, float]:
        """训练模型"""
        return {"loss": 0.0}

class DQNModel:
    """DQN模型类"""
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def predict(self, state: np.ndarray) -> int:
        """预测动作"""
        return np.random.randint(0, self.action_dim)
        
    def train(self, experiences: List[Tuple]) -> Dict[str, float]:
        """训练模型"""
        return {"loss": 0.0}

class BaseNetwork(nn.Module):
    """基础网络类"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class Actor(nn.Module):
    """Actor网络（策略网络）"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    """Critic网络（价值网络）"""
    def __init__(self, total_state_dim: int, total_action_dim: int, hidden_dim: int = 128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        return self.network(x)

class MADDPGModel:
    """多智能体深度确定性策略梯度模型"""
    
    def __init__(self, state_dim: int, action_dims: Dict[str, int], 
                 hidden_dim: int = 128, actor_lr: float = 0.001, critic_lr: float = 0.002,
                 tau: float = 0.01, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.gamma = gamma
        
        # 计算Critic网络的总输入维度
        self.total_state_dim = state_dim
        self.total_action_dim = sum(action_dims.values())
        
        # Actor和Critic网络
        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        self.target_actors = nn.ModuleDict()
        self.target_critics = nn.ModuleDict()
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        
        # 初始化网络
        self._build_networks(actor_lr, critic_lr)
    
    def _build_networks(self, actor_lr: float, critic_lr: float):
        """构建网络"""
        for role, action_dim in self.action_dims.items():
            # Actor网络
            self.actors[role] = Actor(self.state_dim, action_dim, self.hidden_dim)
            self.target_actors[role] = Actor(self.state_dim, action_dim, self.hidden_dim)
            
            # Critic网络
            self.critics[role] = Critic(self.total_state_dim, self.total_action_dim, self.hidden_dim)
            self.target_critics[role] = Critic(self.total_state_dim, self.total_action_dim, self.hidden_dim)
            
            # 优化器
            self.actor_optimizers[role] = optim.Adam(self.actors[role].parameters(), lr=actor_lr)
            self.critic_optimizers[role] = optim.Adam(self.critics[role].parameters(), lr=critic_lr)
        
        # 硬初始化目标网络
        self.hard_update_target_networks()
    
    def hard_update_target_networks(self):
        """硬更新目标网络（初始化为相同权重）"""
        for role in self.action_dims.keys():
            self.target_actors[role].load_state_dict(self.actors[role].state_dict())
            self.target_critics[role].load_state_dict(self.critics[role].state_dict())
    
    def get_actions(self, observations: Dict[str, np.ndarray], 
                   training: bool = False) -> Dict[str, np.ndarray]:
        """获取所有智能体的行动"""
        actions = {}
        
        for role, obs in observations.items():
            if training and np.random.random() < 0.1:  # 探索
                action = np.random.uniform(-1, 1, self.action_dims[role])
            else:  # 利用
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action = self.actors[role](state_tensor).squeeze(0).numpy()
            
            actions[role] = action
        
        return actions
    
    def train(self, batch: List[Dict]) -> Dict[str, float]:
        """训练模型"""
        losses = {}
        
        for role in self.action_dims.keys():
            role_batch = [exp for exp in batch if exp['role'] == role]
            if not role_batch:
                continue
                
            loss = self._train_agent(role, role_batch)
            losses[role] = loss
        
        return losses
    
    def _train_agent(self, role: str, batch: List[Dict]) -> float:
        """训练单个智能体"""
        # 准备训练数据
        states = []
        actions = []
        rewards = []
        next_states = []
        
        for experience in batch:
            states.append(experience['state'])
            actions.append(experience['action'])
            rewards.append(experience['reward'])
            next_states.append(experience['next_state'])
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        
        # Critic训练
        critic_loss = self._train_critic(role, states, actions, rewards, next_states)
        
        # Actor训练
        actor_loss = self._train_actor(role, states)
        
        # 更新目标网络
        self._update_target_networks(role)
        
        return (critic_loss + actor_loss) / 2
    
    def _train_critic(self, role: str, states: torch.Tensor, actions: torch.Tensor,
                     rewards: torch.Tensor, next_states: torch.Tensor) -> float:
        """训练Critic网络"""
        self.critic_optimizers[role].zero_grad()
        
        # 计算目标行动
        target_actions = {}
        for other_role in self.action_dims.keys():
            target_actions[other_role] = self.target_actors[other_role](next_states)
        
        # 构建Critic输入
        target_critic_input_state = states
        target_critic_input_actions = torch.cat([target_actions[r] for r in sorted(self.action_dims.keys())], dim=1)
        
        # 目标Q值
        with torch.no_grad():
            target_q = rewards + self.gamma * self.target_critics[role](
                target_critic_input_state, target_critic_input_actions
            )
        
        # 当前Q值
        current_critic_input_actions = torch.cat([actions if r == role else self.actors[r](states) 
                                                for r in sorted(self.action_dims.keys())], dim=1)
        current_q = self.critics[role](states, current_critic_input_actions)
        
        # Critic损失
        critic_loss = nn.MSELoss()(current_q, target_q)
        critic_loss.backward()
        self.critic_optimizers[role].step()
        
        return critic_loss.item()
    
    def _train_actor(self, role: str, states: torch.Tensor) -> float:
        """训练Actor网络"""
        self.actor_optimizers[role].zero_grad()
        
        # 获取当前策略的行动
        current_actions = self.actors[role](states)
        
        # 构建Critic输入
        all_actions = []
        for other_role in sorted(self.action_dims.keys()):
            if other_role == role:
                all_actions.append(current_actions)
            else:
                with torch.no_grad():
                    all_actions.append(self.actors[other_role](states))
        
        critic_input_actions = torch.cat(all_actions, dim=1)
        
        # 策略梯度损失（最大化Q值）
        q_values = self.critics[role](states, critic_input_actions)
        policy_loss = -q_values.mean()
        
        policy_loss.backward()
        self.actor_optimizers[role].step()
        
        return policy_loss.item()
    
    def _update_target_networks(self, role: str):
        """软更新目标网络"""
        with torch.no_grad():
            # 更新Actor目标网络
            for param, target_param in zip(self.actors[role].parameters(), self.target_actors[role].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # 更新Critic目标网络
            for param, target_param in zip(self.critics[role].parameters(), self.target_critics[role].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_models(self, filepath: str):
        """保存模型"""
        os.makedirs(filepath, exist_ok=True)
        for role in self.action_dims.keys():
            torch.save({
                'actor_state_dict': self.actors[role].state_dict(),
                'critic_state_dict': self.critics[role].state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizers[role].state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizers[role].state_dict(),
            }, f"{filepath}/maddpg_{role}.pth")
    
    def load_models(self, filepath: str):
        """加载模型"""
        for role in self.action_dims.keys():
            checkpoint = torch.load(f"{filepath}/maddpg_{role}.pth")
            self.actors[role].load_state_dict(checkpoint['actor_state_dict'])
            self.critics[role].load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizers[role].load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizers[role].load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # 更新目标网络
            self.target_actors[role].load_state_dict(self.actors[role].state_dict())
            self.target_critics[role].load_state_dict(self.critics[role].state_dict())