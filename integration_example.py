"""
控制系统集成示例和使用指南
Control System Integration Examples and Usage Guide

演示如何使用新的奖励驱动控制系统和迁移指南
"""

import asyncio
import numpy as np
from typing import Dict, Any
import logging

from .distributed_reward_control import DistributedRewardControlSystem, DistributedRewardControlConfig
from .reward_control_adapter import RewardControlAdapter, LegacyControlMapping
from ..core.kallipolis_mathematical_core import SystemState, Agent
from ..agents.role_agents import RoleAgent, AgentConfig

logger = logging.getLogger(__name__)


class ControlSystemIntegrationExample:
    """控制系统集成示例"""
    
    def __init__(self):
        # 创建奖励控制系统
        reward_config = DistributedRewardControlConfig(
            global_coordination_weight=0.3,
            local_autonomy_weight=0.7,
            consensus_threshold=0.8
        )
        self.reward_control_system = DistributedRewardControlSystem(reward_config)
        
        # 创建适配器（支持传统系统迁移）
        legacy_mapping = LegacyControlMapping(
            control_signal_to_reward_scale=2.0,
            compatibility_mode=True
        )
        self.adapter = RewardControlAdapter(self.reward_control_system, legacy_mapping)
        
        # 模拟智能体
        self.agents: Dict[str, RoleAgent] = {}
        
        logger.info("Initialized control system integration example")
    
    def setup_agents(self):
        """设置示例智能体"""
        
        roles = ['doctor', 'intern', 'patient', 'accountant', 'government']
        
        for role in roles:
            # 创建智能体配置
            config = AgentConfig(
                role=role,
                action_dim=8,
                observation_dim=16,
                learning_rate=0.001
            )
            
            # 创建智能体（这里使用简化版本）
            agent = self._create_mock_agent(config)
            self.agents[role] = agent
            
            # 注册到奖励控制系统
            self.reward_control_system.register_agent(role, agent)
        
        logger.info(f"Setup {len(roles)} agents for control system")
    
    def _create_mock_agent(self, config: AgentConfig) -> RoleAgent:
        """创建模拟智能体（简化版本）"""
        # 这里应该创建实际的RoleAgent实例
        # 为了示例，我们创建一个简化的mock对象
        
        class MockRoleAgent:
            def __init__(self, config):
                self.config = config
                self.role = config.role
                self.baseline_reward = 0.0
                
            def compute_local_value(self, system_state, action):
                # 简化的局部价值计算
                return np.random.uniform(0.3, 0.8)
                
            def observe(self, environment):
                # 简化的观察函数
                return np.random.uniform(-1, 1, self.config.observation_dim)
        
        return MockRoleAgent(config)
    
    async def run_reward_control_cycle(self, 
                                     current_state: SystemState,
                                     target_state: SystemState) -> Dict[str, Any]:
        """运行奖励控制周期"""
        
        # 1. 模拟基础奖励
        base_rewards = {
            'doctor': 0.6,
            'intern': 0.4,
            'patient': 0.5,
            'accountant': 0.7,
            'government': 0.3
        }
        
        # 2. 模拟全局效用
        global_utility = 0.65
        
        # 3. 创建控制上下文
        control_context = {
            'doctor': {
                'workload_balance': 0.75,
                'collaboration_score': 0.8
            },
            'intern': {
                'learning_progress': 0.6,
                'practice_opportunities': 0.7
            },
            'patient': {
                'average_waiting_time': 45,
                'cost_per_service': 100,
                'service_quality': 0.8
            },
            'accountant': {
                'cost_variance': 0.08,
                'budget_accuracy': 0.85,
                'roi': 0.12
            },
            'government': {
                'policy_implementation_rate': 0.9,
                'public_trust_score': 0.7,
                'sustainability_index': 0.75
            }
        }
        
        # 4. 执行分布式奖励控制
        final_rewards = await self.reward_control_system.compute_distributed_rewards(
            base_rewards, global_utility, control_context
        )
        
        # 5. 获取系统指标
        system_metrics = self.reward_control_system.get_system_metrics()
        
        return {
            'base_rewards': base_rewards,
            'final_rewards': final_rewards,
            'reward_adjustments': {
                role: final_rewards[role] - base_rewards[role] 
                for role in final_rewards
            },
            'system_metrics': system_metrics,
            'global_utility': global_utility
        }
    
    async def run_hybrid_control_cycle(self,
                                     current_state: SystemState,
                                     target_state: SystemState) -> Dict[str, Any]:
        """运行混合控制周期（支持传统系统迁移）"""
        
        # 1. 模拟传统控制信号
        legacy_control_signals = {
            'primary_controller': np.array([0.3, -0.1, 0.2, 0.4]),
            'observer_controller': np.array([0.1, 0.2, -0.1, 0.3]),
            'patient_controller': np.array([0.2, 0.3, -0.2]),
            'constraint_controller': np.array([0.4, -0.1, 0.2]),
            'government_controller': np.array([0.1, 0.2])
        }
        
        # 2. 模拟基础奖励
        base_rewards = {
            'doctor': 0.5,
            'intern': 0.4,
            'patient': 0.6,
            'accountant': 0.7,
            'government': 0.3
        }
        
        # 3. 模拟全局效用
        global_utility = 0.6
        
        # 4. 创建控制上下文
        control_context = {
            'doctor': {'workload_balance': 0.8},
            'intern': {'learning_progress': 0.7},
            'patient': {'average_waiting_time': 30},
            'accountant': {'cost_variance': 0.05},
            'government': {'policy_implementation_rate': 0.85}
        }
        
        # 5. 执行混合控制
        results = await self.adapter.hybrid_control_step(
            legacy_control_signals, base_rewards, global_utility,
            current_state, target_state, control_context
        )
        
        return results
    
    def demonstrate_migration_path(self) -> Dict[str, Any]:
        """演示迁移路径"""
        
        migration_steps = []
        
        # 步骤1：评估当前系统
        migration_steps.append({
            'step': 1,
            'title': '评估当前传统控制系统',
            'description': '分析现有控制器的性能和局限性',
            'actions': [
                '收集传统控制器性能数据',
                '识别控制瓶颈和问题',
                '评估智能体奖励分布情况'
            ]
        })
        
        # 步骤2：并行部署
        migration_steps.append({
            'step': 2,
            'title': '并行部署奖励控制系统',
            'description': '在不影响现有系统的情况下部署新系统',
            'actions': [
                '部署DistributedRewardControlSystem',
                '配置RewardControlAdapter',
                '启用混合控制模式'
            ]
        })
        
        # 步骤3：渐进式迁移
        migration_steps.append({
            'step': 3,
            'title': '渐进式切换控制权重',
            'description': '逐步增加奖励控制的权重',
            'actions': [
                '50%传统控制 + 50%奖励控制',
                '30%传统控制 + 70%奖励控制',
                '10%传统控制 + 90%奖励控制'
            ]
        })
        
        # 步骤4：完全迁移
        migration_steps.append({
            'step': 4,
            'title': '完全切换到奖励控制',
            'description': '停用传统控制，完全使用奖励驱动控制',
            'actions': [
                '停用传统控制器',
                '优化奖励控制参数',
                '监控系统稳定性'
            ]
        })
        
        # 获取当前迁移状态
        migration_status = self.adapter.get_migration_status()
        
        return {
            'migration_steps': migration_steps,
            'current_status': migration_status,
            'recommendations': self._get_migration_recommendations(migration_status)
        }
    
    def _get_migration_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """获取迁移建议"""
        
        recommendations = []
        
        progress = status.get('migration_progress', 0.0)
        
        if progress < 0.5:
            recommendations.extend([
                '建议首先完成所有智能体的注册',
                '确保奖励控制器配置正确',
                '进行基础功能测试'
            ])
        elif progress < 1.0:
            recommendations.extend([
                '可以开始混合控制模式测试',
                '监控控制一致性指标',
                '逐步调整控制权重分配'
            ])
        else:
            recommendations.extend([
                '系统已准备好完全迁移',
                '建议进行全面性能测试',
                '可以考虑停用传统控制器'
            ])
        
        if not status.get('adapter_ready', False):
            recommendations.append('建议检查适配器配置')
        
        return recommendations


async def main():
    """主函数 - 演示控制系统使用"""
    
    # 创建示例
    example = ControlSystemIntegrationExample()
    
    # 设置智能体
    example.setup_agents()
    
    # 创建示例系统状态
    current_state = SystemState(
        medical_resource_utilization=0.7,
        patient_satisfaction=0.6,
        care_quality_index=0.8,
        # ... 其他状态变量
    )
    
    target_state = SystemState(
        medical_resource_utilization=0.8,
        patient_satisfaction=0.8,
        care_quality_index=0.9,
        # ... 其他状态变量
    )
    
    print("=== 奖励控制系统演示 ===")
    
    # 运行奖励控制周期
    reward_results = await example.run_reward_control_cycle(current_state, target_state)
    print(f"基础奖励: {reward_results['base_rewards']}")
    print(f"调节后奖励: {reward_results['final_rewards']}")
    print(f"奖励调节量: {reward_results['reward_adjustments']}")
    
    print("\n=== 混合控制系统演示 ===")
    
    # 运行混合控制周期
    hybrid_results = await example.run_hybrid_control_cycle(current_state, target_state)
    print(f"奖励调节: {hybrid_results['reward_adjustments']}")
    print(f"混合指标: {hybrid_results['hybrid_metrics']}")
    
    print("\n=== 迁移路径演示 ===")
    
    # 演示迁移路径
    migration_info = example.demonstrate_migration_path()
    print(f"迁移进度: {migration_info['current_status']['migration_progress']:.2%}")
    print("迁移建议:")
    for rec in migration_info['recommendations']:
        print(f"  - {rec}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())