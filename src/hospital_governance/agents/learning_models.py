
"""
Multi-Agent Learning Models for Hospital Governance

包含以下模型：
1. CTDE PPO - 集中训练分散执行的 PPO
2. Fixed LLM Actor - Report 3 架构的固定参数 LLM 生成器
3. Semantic Critic - Report 3 架构的语义评价网络
"""

# CTDE PPO 多智能体模型实现
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Report 3 Architecture: Fixed LLM Actor
# ============================================================================

# Global flag: LLM parameters frozen
LLM_PARAMETERS_FROZEN = True


@dataclass
class LLMGenerationResult:
    """Result from LLM candidate generation"""
    candidates: List[str]
    raw_response: str
    metadata: Dict[str, Any]


class FixedLLMCandidateGenerator:
    """Fixed-parameter LLM candidate generator for Report 3 architecture."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", num_candidates: int = 5, use_mock: bool = True):
        assert LLM_PARAMETERS_FROZEN, "LLM parameters must be frozen"
        self.model_name = model_name
        self.num_candidates = num_candidates
        self.use_mock = use_mock
        self.generation_count = 0
        logger.info(f"Initialized FixedLLMCandidateGenerator (K={num_candidates})")
    
    def generate_candidates(self, role: str, state: Dict[str, Any], 
                           holy_code: Optional[Dict[str, Any]] = None,
                           temperature: float = 0.7) -> LLMGenerationResult:
        """Generate K candidate actions via prompt engineering."""
        assert LLM_PARAMETERS_FROZEN, "Cannot generate with unfrozen parameters"
        prompt = self._build_prompt(role, state, holy_code)
        if self.use_mock:
            raw_response = self._mock_generate(prompt, temperature)
        else:
            raw_response = self._api_generate(prompt, temperature)
        candidates = self._parse_candidates(raw_response)
        self.generation_count += 1
        return LLMGenerationResult(
            candidates=candidates,
            raw_response=raw_response,
            metadata={"role": role, "temperature": temperature, 
                     "generation_id": self.generation_count}
        )
    
    def _build_prompt(self, role: str, state: Dict[str, Any], 
                     holy_code: Optional[Dict[str, Any]]) -> str:
        """Build prompt with Holy Code constraints"""
        parts = [f"You are a {role} in a hospital governance system.", 
                f"Current state: {state}"]
        if holy_code:
            parts.append(f"Holy Code constraints: {holy_code.get('rules', [])}")
        parts.append(f"Generate {self.num_candidates} diverse action candidates.")
        return "\n".join(parts)
    
    def _mock_generate(self, prompt: str, temperature: float) -> str:
        """Mock LLM generation for testing"""
        base_actions = [
            "Allocate 5 units of resources to emergency department",
            "Schedule a training session for medical staff",
            "Review patient care protocols and update guidelines",
            "Coordinate with nursing staff on shift assignments",
            "Initiate quality improvement initiative for patient safety"
        ]
        np.random.seed(self.generation_count)
        selected = np.random.choice(base_actions, 
                                   size=min(self.num_candidates, len(base_actions)), 
                                   replace=False)
        return "\n".join(selected)
    
    def _api_generate(self, prompt: str, temperature: float) -> str:
        """Real LLM API generation"""
        try:
            import openai
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            return response.choices[0].message.content
        except ImportError:
            logger.warning("OpenAI not installed, falling back to mock")
            return self._mock_generate(prompt, temperature)
        except Exception as e:
            logger.error(f"LLM API error: {e}, using mock")
            return self._mock_generate(prompt, temperature)
    
    def _parse_candidates(self, raw_response: str) -> List[str]:
        """Extract candidate actions from response"""
        lines = raw_response.strip().split("\n")
        candidates = [line.strip() for line in lines 
                     if line.strip() and not line.strip().startswith("#")]
        while len(candidates) < self.num_candidates:
            candidates.append("Maintain current state")
        return candidates[:self.num_candidates]


class NaturalLanguageActionParser:
    """Converts natural language action descriptions to 17-dimensional vectors."""
    
    KEYWORD_MAPPING = {
        "emergency": (0, 1.0), "icu": (1, 1.0), "ward": (2, 1.0),
        "outpatient": (3, 1.0), "admin": (4, 1.0),
        "doctor": (5, 1.0), "physician": (5, 1.0),
        "nurse": (6, 1.0), "nursing": (6, 1.0),
        "support": (7, 0.8), "training": (8, 1.0), "meeting": (9, 0.7),
        "quality": (10, 1.0), "safety": (11, 1.0),
        "satisfaction": (12, 0.8), "outcome": (13, 0.9),
        "policy": (14, 1.0), "protocol": (14, 0.8),
        "budget": (15, 1.0), "cost": (15, 0.7), "report": (16, 0.8)
    }
    
    def __init__(self, action_dim: int = 17):
        self.action_dim = action_dim
    
    def parse(self, action_text: str, role: str = "default") -> np.ndarray:
        """Convert natural language action to vector."""
        action_vec = np.zeros(self.action_dim)
        text_lower = action_text.lower()
        for keyword, (dim_idx, weight) in self.KEYWORD_MAPPING.items():
            if keyword in text_lower:
                action_vec[dim_idx] += weight
        if action_vec.max() > 0:
            action_vec = action_vec / action_vec.max()
        action_vec = self._apply_role_bias(action_vec, role)
        return action_vec.astype(np.float32)
    
    def _apply_role_bias(self, action_vec: np.ndarray, role: str) -> np.ndarray:
        """Apply role-specific bias to action vector"""
        role_lower = role.lower()
        if "doctor" in role_lower or "physician" in role_lower:
            action_vec[10:14] *= 1.2
        elif "nurse" in role_lower:
            action_vec[10:12] *= 1.3
        elif "admin" in role_lower:
            action_vec[[0, 4, 14, 15]] *= 1.2
        return np.clip(action_vec, 0.0, 1.0)
    
    def batch_parse(self, action_texts: List[str], role: str = "default") -> np.ndarray:
        """Parse multiple actions to matrix."""
        return np.array([self.parse(text, role) for text in action_texts])


# ============================================================================
# CTDE PPO Models
# ============================================================================

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Actor(nn.Module):
    def __init__(self, obs_dim, hidden=64, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.apply(init_weights)

    def forward(self, x):
        logits = self.net(x)
        return logits

class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim, hidden=128, n_agents=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.value_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_agents)])
        self.apply(init_weights)

    def forward(self, global_state):
        h = self.net(global_state)
        vals = [head(h) for head in self.value_heads]
        vals = torch.cat(vals, dim=1)
        return vals

@dataclass
class AgentStep:
    obs: np.ndarray
    action: int
    logp: float
    reward: float
    global_state: np.ndarray
    done: bool

class RolloutBuffer:
    def __init__(self, n_agents, device='cpu'):
        self.n = n_agents
        self.buffers = [[] for _ in range(n_agents)]
        self.device = device

    def add(self, per_agent_steps: List[AgentStep]):
        assert len(per_agent_steps) == self.n
        for i, s in enumerate(per_agent_steps):
            self.buffers[i].append(s)

    def get(self):
        return self.buffers

    def clear(self):
        self.buffers = [[] for _ in range(self.n)]

def compute_gae_and_returns(critic, buffer: RolloutBuffer, gamma=0.99, lam=0.95, device='cpu'):
    n_agents = buffer.n
    T = len(buffer.buffers[0])
    global_states = np.stack([buffer.buffers[0][t].global_state for t in range(T)], axis=0)
    global_states_t = torch.tensor(global_states, dtype=torch.float32, device=device)
    with torch.no_grad():
        values_all = critic(global_states_t).cpu().numpy()

    results = []
    for i in range(n_agents):
        obs = np.stack([s.obs for s in buffer.buffers[i]], axis=0)
        actions = np.array([s.action for s in buffer.buffers[i]])
        logps = np.array([s.logp for s in buffer.buffers[i]])
        rewards = np.array([s.reward for s in buffer.buffers[i]])
        dones = np.array([s.done for s in buffer.buffers[i]])
        values = values_all[:, i]

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values

        results.append({
            "obs": torch.tensor(obs, dtype=torch.float32, device=device),
            "actions": torch.tensor(actions, dtype=torch.long, device=device),
            "logps": torch.tensor(logps, dtype=torch.float32, device=device),
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
        })
    return results

class CTDEPPOModel:
    """集中训练分散执行的PPO多智能体模型"""
    def __init__(self, obs_dim, n_agents, n_actions=2, hidden=64, critic_hidden=128, device='cpu'):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.actors = [Actor(obs_dim, hidden, n_actions).to(device) for _ in range(n_agents)]
        self.actor_opts = [optim.Adam(a.parameters(), lr=1e-3) for a in self.actors]
        self.critic = CentralizedCritic(obs_dim * n_agents, critic_hidden, n_agents).to(device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)

    def get_actions(self, obs_list: List[np.ndarray], deterministic=False) -> List[int]:
        actions = []
        for i in range(self.n_agents):
            obs = torch.tensor(obs_list[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.actors[i](obs)
            if deterministic:
                act = torch.argmax(logits, dim=-1).item()
            else:
                dist = Categorical(logits=logits)
                act = dist.sample().item()
            actions.append(int(act))
        return actions

    def train(self, buffer: RolloutBuffer, gamma=0.99, lam=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, n_epochs=4, mini_batch_size=64):
        batch_data = compute_gae_and_returns(self.critic, buffer, gamma=gamma, lam=lam, device=self.device)
        n_agents = self.n_agents
        T = batch_data[0]["obs"].shape[0]
        global_states = []
        returns_matrix = []
        for t in range(T):
            obs_t = []
            for i in range(n_agents):
                obs_t.append(batch_data[i]["obs"][t].cpu().numpy())
            global_states.append(np.concatenate(obs_t, axis=0))
            returns_matrix.append([batch_data[i]["returns"][t].item() for i in range(n_agents)])
        global_states = torch.tensor(np.stack(global_states, axis=0), dtype=torch.float32, device=self.device)
        returns_matrix = torch.tensor(np.stack(returns_matrix, axis=0), dtype=torch.float32, device=self.device)

        for _ in range(n_epochs):
            self.critic_opt.zero_grad()
            values_pred = self.critic(global_states)
            value_loss = ((values_pred - returns_matrix) ** 2).mean()
            (vf_coef * value_loss).backward()
            self.critic_opt.step()

        for i in range(n_agents):
            actor = self.actors[i]
            opt = self.actor_opts[i]
            data = batch_data[i]
            obs = data["obs"]
            actions = data["actions"]
            old_logps = data["logps"].detach()
            advantages = data["advantages"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            N = obs.shape[0]
            idxs = np.arange(N)
            for _ in range(n_epochs):
                np.random.shuffle(idxs)
                for start in range(0, N, mini_batch_size):
                    mb_idx = idxs[start:start + mini_batch_size]
                    mb_obs = obs[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_logps = old_logps[mb_idx]
                    mb_adv = advantages[mb_idx]
                    logits = actor(mb_obs)
                    dist = Categorical(logits=logits)
                    mb_logps = dist.log_prob(mb_actions)
                    mb_entropy = dist.entropy().mean()
                    ratio = torch.exp(mb_logps - mb_old_logps)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    loss = policy_loss - ent_coef * mb_entropy
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def save_models(self, filepath: str):
        import os
        os.makedirs(filepath, exist_ok=True)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{filepath}/ctdeppo_actor_{i}.pth")
        torch.save(self.critic.state_dict(), f"{filepath}/ctdeppo_critic.pth")

    def load_models(self, filepath: str):
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(f"{filepath}/ctdeppo_actor_{i}.pth"))
        self.critic.load_state_dict(torch.load(f"{filepath}/ctdeppo_critic.pth"))