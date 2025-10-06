#!/usr/bin/env python3
"""
医院治理系统完整仿真
集成LLM决策、分布式控制、神圣法典规则引擎
"""

import numpy as np
import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import yaml

# 简化导入，避免复杂依赖
class SimulationConfig:
    """仿真配置"""
    def __init__(self):
        # 仿真参数
        self.duration = 100  # 仿真时长
        self.dt = 0.1  # 时间步长
        self.num_agents = 5  # 智能体数量
        
        # LLM配置
        self.llm_provider = 'mock'  # 可选：openai, anthropic, local, mock
        self.llm_model = 'gpt-4'
        self.api_key = None
        
        # 系统配置
        self.system_dim = 16  # 系统状态维度
        self.control_dim = 17  # 控制输入维度
        self.disturbance_level = 0.1  # 扰动强度
        
        # 输出配置
        self.save_results = True
        self.plot_results = True
        self.export_data = True
        self.output_dir = 'simulation_results'

class HospitalSystem:
    """医院系统动力学模型"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state_dim = config.system_dim
        self.control_dim = config.control_dim
        
        # 系统矩阵（简化版本）
        self.A = np.eye(self.state_dim) + 0.1 * np.random.randn(self.state_dim, self.state_dim) * 0.1
        self.B = np.random.randn(self.state_dim, self.control_dim) * 0.2
        self.C = np.eye(self.state_dim)  # 观测矩阵
        
        # 初始状态
        self.x = np.random.rand(self.state_dim) * 0.5
        self.x_ref = np.zeros(self.state_dim)  # 参考状态
        
        # 扰动模型
        self.disturbance_variance = config.disturbance_level
        
    def update(self, u: np.ndarray, dt: float) -> np.ndarray:
        """更新系统状态"""
        # 扰动
        w = np.random.normal(0, self.disturbance_variance, self.state_dim)
        
        # 状态更新：x_{k+1} = A*x_k + B*u_k + w_k
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u) + w * dt
        
        # 保持状态在合理范围内
        self.x = np.clip(self.x, -2.0, 2.0)
        
        return self.get_observation()
    
    def get_observation(self) -> np.ndarray:
        """获取观测"""
        # 添加观测噪声
        noise = np.random.normal(0, 0.01, self.state_dim)
        return np.dot(self.C, self.x) + noise
    
    def get_state(self) -> np.ndarray:
        """获取真实状态"""
        return self.x.copy()

class SimpleRuleEngine:
    """简化的规则引擎"""
    
    def __init__(self):
        self.rules = {
            'ETHICS_001': {
                'name': '患者生命权优先',
                'priority': 1,
                'condition': lambda state: np.mean(state[:4]) < 0.3,  # 健康指标低
                'constraints': {'min_health_level': 0.5, 'min_quality_control': 0.4}
            },
            'RESOURCE_001': {
                'name': '资源公平分配',
                'priority': 2,
                'condition': lambda state: np.std(state[4:8]) > 0.5,  # 资源分布不均
                'constraints': {'max_resource_waste': 0.2, 'min_efficiency': 0.6}
            },
            'CRISIS_001': {
                'name': '危机应急响应',
                'priority': 1,
                'condition': lambda state: np.max(np.abs(state)) > 1.5,  # 系统异常
                'constraints': {'max_response_time': 0.1, 'min_emergency_reserve': 0.8}
            }
        }
    
    def evaluate(self, state: np.ndarray) -> Dict[str, Any]:
        """评估规则并返回约束"""
        active_rules = []
        constraints = {}
        
        for rule_id, rule in self.rules.items():
            if rule['condition'](state):
                active_rules.append(rule_id)
                constraints.update(rule['constraints'])
        
        return {
            'active_rules': active_rules,
            'ethical_constraints': constraints,
            'crisis_level': 'high' if 'CRISIS_001' in active_rules else 'normal'
        }

class SimpleLLMProvider:
    """简化的LLM提供者"""
    
    def __init__(self, provider_type: str = 'mock'):
        self.provider_type = provider_type
        self.role_templates = {
            'doctors': self._doctor_decision,
            'interns': self._intern_decision,
            'patients': self._patient_decision,
            'accountants': self._accountant_decision,
            'government': self._government_decision
        }
    
    def generate_action(self, role: str, observation: np.ndarray, constraints: Dict) -> np.ndarray:
        """生成行动决策"""
        if self.provider_type == 'mock':
            return self.role_templates.get(role, self._default_decision)(observation, constraints)
        else:
            # 这里可以集成真实的LLM API
            return self._call_real_llm(role, observation, constraints)
    
    def _doctor_decision(self, obs: np.ndarray, constraints: Dict) -> np.ndarray:
        """医生决策逻辑"""
        # 基于观测和约束的智能决策
        quality_concern = obs[0] if len(obs) > 0 else 0.5
        resource_need = obs[1] if len(obs) > 1 else 0.3
        
        action = np.array([
            0.6 if quality_concern < 0.4 else 0.2,  # 质量改进
            0.5 if resource_need < 0.3 else 0.1,   # 资源申请
            -0.3 if np.mean(obs[:4]) > 0.7 else 0.1,  # 工作负荷调整
            0.7 if constraints.get('min_quality_control', 0) > 0.3 else 0.3  # 安全措施
        ])
        
        # 应用约束
        if 'min_quality_control' in constraints:
            action[3] = max(action[3], constraints['min_quality_control'])
        
        return np.clip(action, -1, 1)
    
    def _intern_decision(self, obs: np.ndarray, constraints: Dict) -> np.ndarray:
        """实习医生决策逻辑"""
        training_need = 0.6 if np.mean(obs[:3]) < 0.4 else 0.2
        workload_pressure = obs[2] if len(obs) > 2 else 0.5
        
        action = np.array([
            training_need,  # 培训需求
            -0.4 if workload_pressure > 0.7 else 0.1,  # 工作调整
            0.5  # 发展计划
        ])
        
        return np.clip(action, -1, 1)
    
    def _patient_decision(self, obs: np.ndarray, constraints: Dict) -> np.ndarray:
        """患者决策逻辑"""
        satisfaction = obs[4] if len(obs) > 4 else 0.5
        
        action = np.array([
            0.7 if satisfaction < 0.4 else 0.2,  # 服务改善
            0.6 if obs[5] < 0.3 else 0.1,  # 可及性优化
            0.5 if constraints.get('min_health_level', 0) > 0.4 else 0.2  # 安全关注
        ])
        
        return np.clip(action, -1, 1)
    
    def _accountant_decision(self, obs: np.ndarray, constraints: Dict) -> np.ndarray:
        """会计决策逻辑"""
        cost_efficiency = obs[8] if len(obs) > 8 else 0.5
        
        action = np.array([
            0.8 if cost_efficiency < 0.4 else 0.3,  # 成本控制
            0.6 if constraints.get('min_efficiency', 0) > 0.5 else 0.2,  # 效率提升
            0.4  # 预算优化
        ])
        
        return np.clip(action, -1, 1)
    
    def _government_decision(self, obs: np.ndarray, constraints: Dict) -> np.ndarray:
        """政府决策逻辑"""
        system_stability = np.std(obs)
        
        action = np.array([
            0.7 if system_stability > 0.5 else 0.2,  # 监管措施
            0.5,  # 政策调整
            0.6 if len(constraints) > 2 else 0.1  # 协调行动
        ])
        
        return np.clip(action, -1, 1)
    
    def _default_decision(self, obs: np.ndarray, constraints: Dict) -> np.ndarray:
        """默认决策"""
        return np.random.rand(4) * 0.4 - 0.2
    
    def _call_real_llm(self, role: str, observation: np.ndarray, constraints: Dict) -> np.ndarray:
        """调用真实LLM API（待实现）"""
        # TODO: 集成真实LLM API
        return self._default_decision(observation, constraints)

class MultiAgentController:
    """多智能体控制器"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.llm_provider = SimpleLLMProvider(config.llm_provider)
        
        # 观测掩码 - 定义每个角色能观测到的状态
        self.observation_masks = {
            'doctors': slice(0, 16),  # 全部状态
            'interns': slice(0, 12),  # 前12个状态
            'patients': slice(4, 12), # 财务和质量状态
            'accountants': slice(8, 12), # 财务状态
            'government': [0, 1, 2, 3, 12, 13, 14, 15]  # 资源和伦理状态
        }
        
        # 控制分配
        self.control_allocation = {
            'doctors': slice(0, 4),
            'interns': slice(4, 8),
            'patients': slice(8, 11),
            'accountants': slice(11, 14),
            'government': slice(14, 17)
        }
    
    def compute_control(self, full_state: np.ndarray, holy_code_state: Dict) -> np.ndarray:
        """计算多智能体控制信号"""
        u_global = np.zeros(self.config.control_dim)
        
        for role in ['doctors', 'interns', 'patients', 'accountants', 'government']:
            # 获取局部观测
            mask = self.observation_masks[role]
            if isinstance(mask, slice):
                local_obs = full_state[mask]
            else:
                local_obs = full_state[mask]
            
            # 生成局部控制
            local_control = self.llm_provider.generate_action(
                role, local_obs, holy_code_state.get('ethical_constraints', {})
            )
            
            # 分配到全局控制向量
            control_slice = self.control_allocation[role]
            u_global[control_slice] = local_control[:len(range(*control_slice.indices(self.config.control_dim)))]
        
        return u_global

class HospitalSimulation:
    """医院治理系统仿真主类"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        
        # 初始化组件
        self.system = HospitalSystem(self.config)
        self.rule_engine = SimpleRuleEngine()
        self.controller = MultiAgentController(self.config)
        
        # 数据记录
        self.time_history = []
        self.state_history = []
        self.control_history = []
        self.rule_history = []
        self.performance_history = []
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def run_simulation(self):
        """运行完整仿真"""
        print("🚀 开始医院治理系统仿真...")
        print(f"仿真时长: {self.config.duration} 步")
        print(f"LLM提供者: {self.config.llm_provider}")
        print(f"系统维度: {self.config.system_dim}D 状态, {self.config.control_dim}D 控制")
        
        start_time = time.time()
        
        for step in range(self.config.duration):
            # 获取当前状态和观测
            state = self.system.get_state()
            observation = self.system.get_observation()
            
            # 评估神圣法典规则
            holy_code_state = self.rule_engine.evaluate(state)
            
            # 计算控制输入
            control = self.controller.compute_control(observation, holy_code_state)
            
            # 更新系统
            next_observation = self.system.update(control, self.config.dt)
            
            # 记录数据
            self._record_step(step, state, control, holy_code_state)
            
            # 进度显示
            if step % 20 == 0:
                print(f"步骤 {step}/{self.config.duration}, 激活规则: {len(holy_code_state['active_rules'])}")
        
        simulation_time = time.time() - start_time
        print(f"✅ 仿真完成，耗时: {simulation_time:.2f}秒")
        
        # 分析结果
        self._analyze_results()
        
        # 保存和可视化
        if self.config.save_results:
            self._save_results()
        
        if self.config.plot_results:
            self._plot_results()
        
        return self._get_summary()
    
    def _record_step(self, step: int, state: np.ndarray, control: np.ndarray, holy_code_state: Dict):
        """记录单步数据"""
        self.time_history.append(step * self.config.dt)
        self.state_history.append(state.copy())
        self.control_history.append(control.copy())
        self.rule_history.append(holy_code_state.copy())
        
        # 计算性能指标
        performance = {
            'stability': np.linalg.norm(state),
            'control_effort': np.linalg.norm(control),
            'rule_compliance': len(holy_code_state['active_rules']),
            'system_health': 1.0 / (1.0 + np.linalg.norm(state - self.system.x_ref))
        }
        self.performance_history.append(performance)
    
    def _analyze_results(self):
        """分析仿真结果"""
        print("\\n📊 仿真结果分析:")
        
        # 系统稳定性
        final_stability = self.performance_history[-1]['stability']
        avg_stability = np.mean([p['stability'] for p in self.performance_history])
        print(f"  最终稳定性: {final_stability:.3f}")
        print(f"  平均稳定性: {avg_stability:.3f}")
        
        # 控制努力
        avg_control_effort = np.mean([p['control_effort'] for p in self.performance_history])
        print(f"  平均控制努力: {avg_control_effort:.3f}")
        
        # 规则激活统计
        total_rule_activations = sum(p['rule_compliance'] for p in self.performance_history)
        print(f"  总规则激活次数: {total_rule_activations}")
        
        # 系统健康度
        avg_health = np.mean([p['system_health'] for p in self.performance_history])
        print(f"  平均系统健康度: {avg_health:.3f}")
        
        # 规则激活频率
        rule_counts = {}
        for rule_state in self.rule_history:
            for rule in rule_state['active_rules']:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        print(f"  规则激活频率:")
        for rule, count in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {rule}: {count} 次")
    
    def _save_results(self):
        """保存结果到文件"""
        print("\\n💾 保存仿真结果...")
        
        # 保存配置
        with open(f"{self.config.output_dir}/config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # 保存数据
        results = {
            'time': self.time_history,
            'states': [state.tolist() for state in self.state_history],
            'controls': [control.tolist() for control in self.control_history],
            'rules': self.rule_history,
            'performance': self.performance_history
        }
        
        with open(f"{self.config.output_dir}/simulation_data.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存摘要
        summary = self._get_summary()
        with open(f"{self.config.output_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  结果已保存到: {self.config.output_dir}/")
    
    def _plot_results(self):
        """绘制结果图表"""
        print("\\n📈 生成可视化图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 系统状态轨迹
        axes[0, 0].plot(self.time_history, [s[0] for s in self.state_history], label='状态1')
        axes[0, 0].plot(self.time_history, [s[1] for s in self.state_history], label='状态2')
        axes[0, 0].plot(self.time_history, [s[2] for s in self.state_history], label='状态3')
        axes[0, 0].set_title('关键系统状态')
        axes[0, 0].set_xlabel('时间')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 控制信号
        axes[0, 1].plot(self.time_history, [c[0] for c in self.control_history], label='医生控制')
        axes[0, 1].plot(self.time_history, [c[4] for c in self.control_history], label='实习医生控制')
        axes[0, 1].plot(self.time_history, [c[8] for c in self.control_history], label='患者控制')
        axes[0, 1].set_title('多智能体控制信号')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 性能指标
        axes[1, 0].plot(self.time_history, [p['stability'] for p in self.performance_history], label='稳定性')
        axes[1, 0].plot(self.time_history, [p['system_health'] for p in self.performance_history], label='系统健康度')
        axes[1, 0].set_title('系统性能指标')
        axes[1, 0].set_xlabel('时间')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 规则激活
        rule_activation = [len(r['active_rules']) for r in self.rule_history]
        axes[1, 1].plot(self.time_history, rule_activation, 'r-', label='激活规则数')
        axes[1, 1].set_title('神圣法典规则激活')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('激活规则数量')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/simulation_results.png", dpi=300, bbox_inches='tight')
        print(f"  图表已保存到: {self.config.output_dir}/simulation_results.png")
        
        if self.config.plot_results:
            plt.show()
    
    def _get_summary(self) -> Dict[str, Any]:
        """生成仿真摘要"""
        return {
            'simulation_config': asdict(self.config),
            'final_performance': self.performance_history[-1] if self.performance_history else {},
            'average_performance': {
                'stability': np.mean([p['stability'] for p in self.performance_history]),
                'control_effort': np.mean([p['control_effort'] for p in self.performance_history]),
                'system_health': np.mean([p['system_health'] for p in self.performance_history])
            },
            'rule_statistics': self._get_rule_statistics(),
            'simulation_duration': len(self.time_history) * self.config.dt
        }
    
    def _get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计"""
        rule_counts = {}
        for rule_state in self.rule_history:
            for rule in rule_state['active_rules']:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        return {
            'total_activations': sum(rule_counts.values()),
            'unique_rules_activated': len(rule_counts),
            'activation_frequency': rule_counts,
            'most_active_rule': max(rule_counts.items(), key=lambda x: x[1])[0] if rule_counts else None
        }

def main():
    """主函数"""
    print("🏥 医院治理系统LLM驱动仿真")
    print("=" * 60)
    
    # 创建配置
    config = SimulationConfig()
    config.duration = 50  # 减少步数以便快速测试
    config.llm_provider = 'mock'  # 使用模拟LLM
    config.save_results = True
    config.plot_results = False  # 关闭自动显示图表
    
    # 运行仿真
    simulation = HospitalSimulation(config)
    summary = simulation.run_simulation()
    
    print("\\n🎉 仿真完成！")
    print("=" * 60)
    print("摘要:")
    print(f"  仿真时长: {summary['simulation_duration']:.1f} 时间单位")
    print(f"  最终系统健康度: {summary['final_performance'].get('system_health', 0):.3f}")
    print(f"  平均稳定性: {summary['average_performance']['stability']:.3f}")
    print(f"  总规则激活: {summary['rule_statistics']['total_activations']} 次")
    print(f"  最活跃规则: {summary['rule_statistics'].get('most_active_rule', 'None')}")

if __name__ == '__main__':
    main()