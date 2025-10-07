"""
智能体注册中心 - 统一管理智能体创建、配置和LLM服务集成
Agent Registry - Unified management for agent creation, configuration and LLM service integration
"""

import os
import logging
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from enum import Enum

from .role_agents import RoleAgent, AgentConfig, DoctorAgent, InternAgent, PatientAgent, AccountantAgent, GovernmentAgent
from .llm_action_generator import LLMActionGenerator, LLMConfig
from .llm_providers import create_llm_provider, get_preset_config

logger = logging.getLogger(__name__)

class LLMProviderType(Enum):
    """LLM提供者类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    LOCAL = "local"
    MOCK = "mock"

@dataclass
class AgentRegistryConfig:
    """智能体注册配置"""
    # LLM配置
    llm_provider: LLMProviderType = LLMProviderType.MOCK
    llm_preset: str = "mock"  # 预设配置名称
    
    # API密钥环境变量名映射
    api_key_env_vars: Dict[str, str] = field(default_factory=lambda: {
        LLMProviderType.OPENAI.value: "OPENAI_API_KEY",
        LLMProviderType.ANTHROPIC.value: "ANTHROPIC_API_KEY"
    })
    
    # 角色配置
    default_action_dim: int = 5
    default_observation_dim: int = 8
    default_learning_rate: float = 0.001
    
    # 性能配置
    enable_llm_generation: bool = True
    fallback_to_mock: bool = True  # API失败时是否回退到mock
    api_validation_timeout: float = 5.0

class AgentRegistry:
    """智能体注册中心"""
    
    def __init__(self, config: AgentRegistryConfig = None):
        self.config = config or AgentRegistryConfig()
        self.agents: Dict[str, RoleAgent] = {}
        self.llm_generators: Dict[str, LLMActionGenerator] = {}
        self.api_status: Dict[str, bool] = {}  # API可用性状态
        
        # 角色类映射
        self.agent_classes: Dict[str, Type[RoleAgent]] = {
            'doctors': DoctorAgent,
            'interns': InternAgent, 
            'patients': PatientAgent,
            'accountants': AccountantAgent,
            'government': GovernmentAgent
        }
        
        # 初始化API状态检查
        self._validate_api_keys()
    
    def _validate_api_keys(self) -> None:
        """验证API密钥可用性"""
        logger.info("🔑 验证API密钥可用性...")
        
        for provider_type, env_var in self.config.api_key_env_vars.items():
            api_key = os.getenv(env_var)
            if api_key:
                self.api_status[provider_type] = True
                logger.info(f"✅ {provider_type} API密钥已配置 (环境变量: {env_var})")
            else:
                self.api_status[provider_type] = False
                logger.warning(f"⚠️  {provider_type} API密钥未配置 (环境变量: {env_var})")
        
        # 检查选定的提供者是否可用
        selected_provider = self.config.llm_provider.value
        if selected_provider not in ['mock', 'local'] and not self.api_status.get(selected_provider, False):
            if self.config.fallback_to_mock:
                logger.warning(f"🔄 {selected_provider} API不可用，回退到Mock模式")
                self.config.llm_provider = LLMProviderType.MOCK
                self.config.llm_preset = "mock"
            else:
                raise ValueError(f"LLM提供者 {selected_provider} 的API密钥未配置，且未启用回退模式")
    
    def _create_llm_generator(self, role: str) -> LLMActionGenerator:
        """为特定角色创建LLM生成器"""
        try:
            # 获取LLM配置
            llm_config = get_preset_config(self.config.llm_preset)
            
            # 从环境变量设置API密钥
            if self.config.llm_provider != LLMProviderType.MOCK:
                env_var = self.config.api_key_env_vars.get(self.config.llm_provider.value)
                if env_var:
                    llm_config.api_key = os.getenv(env_var)
            
            # 创建LLM提供者
            provider = create_llm_provider(self.config.llm_provider.value, llm_config)
            
            # 创建LLM行动生成器
            generator = LLMActionGenerator(llm_config, provider)
            
            logger.info(f"✅ 为角色 {role} 创建LLM生成器成功 (提供者: {self.config.llm_provider.value})")
            return generator
            
        except Exception as e:
            logger.error(f"❌ 为角色 {role} 创建LLM生成器失败: {e}")
            
            if self.config.fallback_to_mock:
                logger.info(f"🔄 为角色 {role} 回退到Mock LLM生成器")
                mock_config = get_preset_config("mock")
                mock_provider = create_llm_provider("mock", mock_config)
                return LLMActionGenerator(mock_config, mock_provider)
            else:
                raise
    
    def register_agent(self, role: str, custom_config: Optional[AgentConfig] = None) -> RoleAgent:
        """注册单个智能体"""
        if role in self.agents:
            logger.warning(f"⚠️  角色 {role} 已存在，将覆盖现有智能体")
        
        # 创建智能体配置
        if custom_config:
            agent_config = custom_config
        else:
            agent_config = AgentConfig(
                role=role,
                action_dim=self.config.default_action_dim,
                observation_dim=self.config.default_observation_dim,
                learning_rate=self.config.default_learning_rate
            )
        
        # 获取智能体类
        agent_class = self.agent_classes.get(role)
        if not agent_class:
            raise ValueError(f"不支持的角色类型: {role}. 支持的角色: {list(self.agent_classes.keys())}")
        
        # 创建智能体实例
        agent = agent_class(agent_config)
        
        # 创建并集成LLM生成器
        if self.config.enable_llm_generation:
            try:
                llm_generator = self._create_llm_generator(role)
                agent.llm_generator = llm_generator
                self.llm_generators[role] = llm_generator
                logger.info(f"🤖 为角色 {role} 集成LLM生成器")
            except Exception as e:
                logger.error(f"❌ 为角色 {role} 集成LLM失败: {e}")
                if not self.config.fallback_to_mock:
                    raise
        
        # 注册智能体
        self.agents[role] = agent
        logger.info(f"✅ 成功注册角色智能体: {role}")
        
        return agent
    
    def register_all_agents(self, custom_configs: Optional[Dict[str, AgentConfig]] = None) -> Dict[str, RoleAgent]:
        """注册所有标准角色的智能体"""
        logger.info("🚀 开始注册所有智能体...")
        
        registered_agents = {}
        for role in self.agent_classes.keys():
            try:
                config = custom_configs.get(role) if custom_configs else None
                agent = self.register_agent(role, config)
                registered_agents[role] = agent
            except Exception as e:
                logger.error(f"❌ 注册角色 {role} 失败: {e}")
                if not self.config.fallback_to_mock:
                    raise
        
        logger.info(f"✅ 智能体注册完成! 总计: {len(registered_agents)} 个角色")
        return registered_agents
    
    def get_agent(self, role: str) -> Optional[RoleAgent]:
        """获取指定角色的智能体"""
        return self.agents.get(role)
    
    def get_all_agents(self) -> Dict[str, RoleAgent]:
        """获取所有已注册的智能体"""
        return self.agents.copy()
    
    def get_llm_generator(self, role: str) -> Optional[LLMActionGenerator]:
        """获取指定角色的LLM生成器"""
        return self.llm_generators.get(role)
    
    def update_llm_config(self, new_provider: LLMProviderType, new_preset: str = None):
        """更新LLM配置并重新初始化生成器"""
        logger.info(f"🔄 更新LLM配置: {new_provider.value}")
        
        self.config.llm_provider = new_provider
        if new_preset:
            self.config.llm_preset = new_preset
        
        # 重新验证API
        self._validate_api_keys()
        
        # 重新创建LLM生成器
        if self.config.enable_llm_generation:
            for role in self.agents.keys():
                try:
                    llm_generator = self._create_llm_generator(role)
                    self.agents[role].llm_generator = llm_generator
                    self.llm_generators[role] = llm_generator
                    logger.info(f"✅ 角色 {role} LLM生成器已更新")
                except Exception as e:
                    logger.error(f"❌ 更新角色 {role} LLM生成器失败: {e}")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """获取注册中心状态"""
        return {
            'total_agents': len(self.agents),
            'registered_roles': list(self.agents.keys()),
            'llm_provider': self.config.llm_provider.value,
            'llm_preset': self.config.llm_preset,
            'api_status': self.api_status.copy(),
            'llm_enabled': self.config.enable_llm_generation,
            'generators_count': len(self.llm_generators)
        }
    
    def test_llm_generation(self, role: str = None) -> Dict[str, Any]:
        """测试LLM生成功能"""
        test_results = {}
        
        roles_to_test = [role] if role else list(self.agents.keys())
        
        for test_role in roles_to_test:
            if test_role not in self.agents:
                test_results[test_role] = {'status': 'failed', 'error': 'Agent not found'}
                continue
            
            agent = self.agents[test_role]
            if not hasattr(agent, 'llm_generator') or agent.llm_generator is None:
                test_results[test_role] = {'status': 'failed', 'error': 'LLM generator not available'}
                continue
            
            try:
                # 测试生成
                import numpy as np
                test_observation = np.random.uniform(0.3, 0.7, self.config.default_observation_dim)
                test_context = {'role': test_role, 'context_type': 'test'}
                
                action = agent.llm_generator.generate_action_sync(
                    test_role, test_observation, {}, test_context
                )
                
                test_results[test_role] = {
                    'status': 'success',
                    'action_shape': action.shape,
                    'action_range': [float(action.min()), float(action.max())],
                    'provider': self.config.llm_provider.value
                }
                
            except Exception as e:
                test_results[test_role] = {'status': 'failed', 'error': str(e)}
        
        return test_results
    
    def export_config(self, filepath: str):
        """导出配置到文件"""
        import json
        
        config_data = {
            'llm_provider': self.config.llm_provider.value,
            'llm_preset': self.config.llm_preset,
            'enable_llm_generation': self.config.enable_llm_generation,
            'api_key_env_vars': self.config.api_key_env_vars,
            'registry_status': self.get_registry_status(),
            'api_status': self.api_status
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 配置已导出到: {filepath}")

# 便捷函数
def create_agent_registry(llm_provider: str = "mock", 
                         enable_llm: bool = True,
                         fallback_to_mock: bool = True) -> AgentRegistry:
    """创建智能体注册中心的便捷函数"""
    provider_enum = LLMProviderType(llm_provider)
    
    config = AgentRegistryConfig(
        llm_provider=provider_enum,
        llm_preset=llm_provider if llm_provider in ['mock', 'openai_gpt4', 'anthropic_claude'] else 'mock',
        enable_llm_generation=enable_llm,
        fallback_to_mock=fallback_to_mock
    )
    
    return AgentRegistry(config)

# 全局注册中心实例（单例模式）
_global_registry: Optional[AgentRegistry] = None

def get_global_agent_registry() -> AgentRegistry:
    """获取全局智能体注册中心"""
    global _global_registry
    if _global_registry is None:
        _global_registry = create_agent_registry()
    return _global_registry

def reset_global_registry():
    """重置全局注册中心"""
    global _global_registry
    _global_registry = None