"""
真实LLM API提供者配置
支持OpenAI GPT、Anthropic Claude、本地模型等
"""

import asyncio
import httpx
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from .llm_action_generator import BaseLLMProvider, LLMConfig

@dataclass
class APIConfig:
    """API配置"""
    api_key: str
    base_url: str
    model_name: str
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: float = 30.0

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API提供者"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = config.base_url or "https://api.openai.com/v1"
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass in config.")
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """异步生成文本"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_prompt(context)
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    def generate_text_sync(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """同步生成文本"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_prompt(context)
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _build_system_prompt(self, context: Dict[str, Any] = None) -> str:
        """构建系统提示"""
        role = context.get('role', 'unknown') if context else 'unknown'
        
        base_prompt = """你是医院治理系统中的智能决策助手。你需要基于当前观测和约束条件，为特定角色生成合理的行动决策。

重要规则：
1. 必须返回一个数值向量，格式为 [数值1, 数值2, ...]
2. 每个数值必须在-1.0到1.0之间
3. 考虑神圣法典（HolyCode）的伦理约束
4. 基于角色特点做出合理决策

"""
        
        role_specifics = {
            'doctors': """
你代表医生群体，关注：
- 医疗质量和患者安全
- 资源合理利用
- 专业医疗标准
- 危机应急响应
返回4维向量：[质量改进, 资源申请, 工作负荷调整, 安全措施]
""",
            'interns': """
你代表实习医生群体，关注：
- 教育培训机会
- 工作负荷平衡
- 职业发展
- 学习资源获取
返回3维向量：[培训需求, 工作调整, 发展计划]
""",
            'patients': """
你代表患者群体，关注：
- 医疗服务质量
- 等待时间和可及性
- 患者安全和权益
- 满意度改善
返回3维向量：[服务改善, 可及性优化, 安全关注]
""",
            'accountants': """
你代表会计群体，关注：
- 财务健康和成本控制
- 运营效率
- 预算管理
- 透明度和合规性
返回3维向量：[成本控制, 效率提升, 预算优化]
""",
            'government': """
你代表政府监管方，关注：
- 系统稳定性
- 监管合规
- 公共利益
- 政策协调
返回3维向量：[监管措施, 政策调整, 协调行动]
"""
        }
        
        return base_prompt + role_specifics.get(role, "你是通用决策者，返回4维向量。")

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API提供者"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv('ANTHROPIC_API_KEY')
        self.base_url = config.base_url or "https://api.anthropic.com/v1"
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """异步生成文本"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": f"{self._build_system_prompt(context)}\\n\\n{prompt}"
                }
            ]
        }
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["content"][0]["text"]
    
    def generate_text_sync(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """同步生成文本"""
        import requests
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": f"{self._build_system_prompt(context)}\\n\\n{prompt}"
                }
            ]
        }
        
        response = requests.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result["content"][0]["text"]
    
    def _build_system_prompt(self, context: Dict[str, Any] = None) -> str:
        """构建系统提示"""
        return """你是医院治理系统的智能决策助手。请基于给定的观测数据和约束条件，为指定角色生成数值化的行动决策。

请严格按照以下格式返回：
[数值1, 数值2, 数值3, ...]

其中每个数值在-1.0到1.0之间，代表该维度的行动强度。"""

class LocalModelProvider(BaseLLMProvider):
    """本地模型提供者（如Ollama）"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """异步生成文本"""
        payload = {
            "model": self.config.model_name,
            "prompt": f"{self._build_system_prompt(context)}\\n\\n{prompt}",
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["response"]
    
    def generate_text_sync(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """同步生成文本"""
        import requests
        
        payload = {
            "model": self.config.model_name,
            "prompt": f"{self._build_system_prompt(context)}\\n\\n{prompt}",
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
    
    def _build_system_prompt(self, context: Dict[str, Any] = None) -> str:
        """构建系统提示"""
        return "你是医院治理系统的智能助手。请返回数值向量格式的决策，例如：[0.5, -0.2, 0.8]"

def create_llm_provider(provider_type: str, config: LLMConfig) -> BaseLLMProvider:
    """创建LLM提供者工厂函数"""
    providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'local': LocalModelProvider,
        'mock': lambda cfg: __import__('hospital_governance.agents.llm_action_generator', fromlist=['MockLLMProvider']).MockLLMProvider(cfg)
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unsupported provider type: {provider_type}. Available: {list(providers.keys())}")
    
    return providers[provider_type](config)

# 预设配置
PRESET_CONFIGS = {
    'openai_gpt4': LLMConfig(
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=800,
        timeout=30.0
    ),
    'openai_gpt35': LLMConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=800,
        timeout=20.0
    ),
    'anthropic_claude': LLMConfig(
        model_name="claude-3-sonnet-20240229",
        temperature=0.7,
        max_tokens=800,
        timeout=30.0
    ),
    'local_llama': LLMConfig(
        model_name="llama2:7b",
        temperature=0.7,
        max_tokens=800,
        timeout=60.0
    ),
    'mock': LLMConfig(
        model_name="mock",
        temperature=0.0,
        max_tokens=500,
        timeout=1.0
    )
}

def get_preset_config(preset_name: str) -> LLMConfig:
    """获取预设配置"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    return PRESET_CONFIGS[preset_name]