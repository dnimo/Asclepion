#!/usr/bin/env python3
"""
医院治理系统 - 数据导出集成示例
展示如何在仿真中集成数据导出功能
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.hospital_governance.interfaces.data_export import (
    DataExporter, DataImporter, SimulationMetadata, 
    TimeSeriesData, AgentDecisionData
)

class SimulationWithExport:
    """带数据导出功能的仿真系统"""
    
    def __init__(self, simulation_id: str = None, export_dir: str = "simulation_exports"):
        """
        初始化仿真系统
        
        Args:
            simulation_id: 仿真ID
            export_dir: 导出目录
        """
        self.simulation_id = simulation_id or f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exporter = DataExporter(export_dir)
        self.importer = DataImporter(export_dir)
        
        # 仿真状态
        self.start_time = None
        self.end_time = None
        self.system_state = np.array([1.0, 0.5, 0.8, 0.3, 0.6])  # 5维状态
        self.control_input = np.array([0.0, 0.0, 0.0])  # 3维控制
        
        # 数据收集
        self.time_series_data = {
            'timestamps': [],
            'states': [],
            'controls': [],
            'observations': [],
            'rule_activations': [],
            'performance_indices': [],
            'stability_metrics': []
        }
        
        self.agent_data = {
            'doctor': AgentDecisionData(
                agent_id="doctor",
                decision_history=[],
                llm_responses=[],
                reasoning_chains=[],
                performance_scores=[]
            ),
            'nurse': AgentDecisionData(
                agent_id="nurse", 
                decision_history=[],
                llm_responses=[],
                reasoning_chains=[],
                performance_scores=[]
            )
        }
        
    def run_simulation(self, duration: int = 20):
        """
        运行仿真并收集数据
        
        Args:
            duration: 仿真持续步数
        """
        print(f"🏥 开始仿真 {self.simulation_id} ({duration} 步)")
        self.start_time = datetime.now()
        
        for step in range(duration):
            self._simulate_step(step)
            
        self.end_time = datetime.now()
        print(f"✅ 仿真完成，用时 {(self.end_time - self.start_time).total_seconds():.2f} 秒")
        
    def _simulate_step(self, step: int):
        """执行单步仿真"""
        
        # 模拟系统动态
        noise = np.random.normal(0, 0.1, 5)
        self.system_state += noise
        self.system_state = np.clip(self.system_state, 0, 1)  # 保持在[0,1]范围
        
        # 模拟控制决策
        error = 0.5 - self.system_state[:3]  # 期望值为0.5
        self.control_input = 0.1 * error + np.random.normal(0, 0.05, 3)
        
        # 应用控制
        self.system_state[:3] += 0.1 * self.control_input
        self.system_state[:3] = np.clip(self.system_state[:3], 0, 1)
        
        # 计算性能指标
        performance_index = 1.0 / (1.0 + np.sum((self.system_state - 0.5)**2))
        stability_metric = np.exp(-np.linalg.norm(self.control_input))
        
        # 模拟规则激活
        rule_activations = {}
        if performance_index < 0.5:
            rule_activations["emergency_protocol"] = {
                "activated": True,
                "severity": 1.0 - performance_index,
                "description": "系统性能低于阈值"
            }
        
        if np.any(self.system_state > 0.9):
            rule_activations["resource_allocation"] = {
                "activated": True, 
                "severity": np.max(self.system_state) - 0.9,
                "description": "资源使用率过高"
            }
        
        # 模拟智能体决策
        self._simulate_agent_decisions(step, performance_index)
        
        # 记录数据
        self.time_series_data['timestamps'].append(step)
        self.time_series_data['states'].append(self.system_state.copy())
        self.time_series_data['controls'].append(self.control_input.copy())
        self.time_series_data['observations'].append(self.system_state.copy())  # 假设完全可观测
        self.time_series_data['rule_activations'].append(rule_activations)
        self.time_series_data['performance_indices'].append(performance_index)
        self.time_series_data['stability_metrics'].append(stability_metric)
        
        if step % 5 == 0:
            print(f"  步骤 {step:2d}: 性能={performance_index:.3f}, 稳定性={stability_metric:.3f}, 规则={len(rule_activations)}")
    
    def _simulate_agent_decisions(self, step: int, performance_index: float):
        """模拟智能体决策过程"""
        
        # 医生智能体决策
        doctor_decision = {
            "action": "diagnose" if step % 3 == 0 else "treat",
            "confidence": min(0.9, performance_index + 0.2),
            "priority": "high" if performance_index < 0.6 else "normal"
        }
        
        doctor_response = f"医生在步骤{step}决定{doctor_decision['action']}，置信度{doctor_decision['confidence']:.2f}"
        doctor_reasoning = [
            f"分析当前系统状态: {self.system_state[:3]}",
            f"评估性能指标: {performance_index:.3f}",
            f"确定行动方案: {doctor_decision['action']}"
        ]
        
        self.agent_data['doctor'].decision_history.append(doctor_decision)
        self.agent_data['doctor'].llm_responses.append(doctor_response)
        self.agent_data['doctor'].reasoning_chains.append(doctor_reasoning)
        self.agent_data['doctor'].performance_scores.append(performance_index)
        
        # 护士智能体决策
        nurse_decision = {
            "action": "monitor" if step % 2 == 0 else "assist",
            "patient_id": f"patient_{step % 5}",
            "urgency": "urgent" if performance_index < 0.4 else "routine"
        }
        
        nurse_response = f"护士执行{nurse_decision['action']}任务，患者{nurse_decision['patient_id']}"
        nurse_reasoning = [
            f"检查患者状态",
            f"确定紧急程度: {nurse_decision['urgency']}",
            f"执行相应行动"
        ]
        
        self.agent_data['nurse'].decision_history.append(nurse_decision)
        self.agent_data['nurse'].llm_responses.append(nurse_response)
        self.agent_data['nurse'].reasoning_chains.append(nurse_reasoning)
        self.agent_data['nurse'].performance_scores.append(performance_index * 0.9)  # 护士性能稍低
    
    def export_results(self, format_type: str = "all") -> dict:
        """
        导出仿真结果
        
        Args:
            format_type: 导出格式
            
        Returns:
            导出文件路径字典
        """
        print(f"📊 导出仿真结果...")
        
        # 准备元数据
        metadata = SimulationMetadata(
            simulation_id=self.simulation_id,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_steps=len(self.time_series_data['timestamps']),
            llm_provider="mock",
            system_parameters={
                "state_dimension": len(self.system_state),
                "control_dimension": len(self.control_input),
                "noise_level": 0.1,
                "control_gain": 0.1
            },
            performance_metrics={
                "average_performance": np.mean(self.time_series_data['performance_indices']),
                "average_stability": np.mean(self.time_series_data['stability_metrics']),
                "total_rule_activations": sum(len(ra) for ra in self.time_series_data['rule_activations'])
            }
        )
        
        # 准备时序数据
        time_series = TimeSeriesData(
            timestamps=self.time_series_data['timestamps'],
            states=self.time_series_data['states'],
            controls=self.time_series_data['controls'],
            observations=self.time_series_data['observations'],
            rule_activations=self.time_series_data['rule_activations'],
            performance_indices=self.time_series_data['performance_indices'],
            stability_metrics=self.time_series_data['stability_metrics']
        )
        
        # 准备智能体数据
        agent_data = list(self.agent_data.values())
        
        # 执行导出
        exported_files = self.exporter.export_simulation_results(
            metadata, time_series, agent_data, format_type
        )
        
        print("✅ 导出完成:")
        for format_name, file_path in exported_files.items():
            if isinstance(file_path, dict):
                print(f"  {format_name}: {len(file_path)} 个文件")
                for sub_name, sub_path in file_path.items():
                    print(f"    - {sub_name}: {Path(sub_path).name}")
            else:
                print(f"  {format_name}: {Path(file_path).name}")
        
        return exported_files
    
    def load_and_analyze(self, json_file: str):
        """
        加载并分析已导出的数据
        
        Args:
            json_file: JSON文件路径
        """
        print(f"📈 分析导出数据: {json_file}")
        
        # 从JSON加载数据
        data = self.importer.import_from_json(json_file)
        
        # 分析元数据
        metadata = data['metadata']
        print(f"\\n仿真信息:")
        print(f"  ID: {metadata['simulation_id']}")
        print(f"  持续时间: {metadata['duration_steps']} 步")
        print(f"  LLM提供者: {metadata['llm_provider']}")
        
        # 分析性能指标
        time_series = data['time_series']
        avg_performance = np.mean(time_series['performance_indices'])
        avg_stability = np.mean(time_series['stability_metrics'])
        
        print(f"\\n性能分析:")
        print(f"  平均性能: {avg_performance:.3f}")
        print(f"  平均稳定性: {avg_stability:.3f}")
        print(f"  最终状态: {np.array(time_series['states'][-1])}")
        
        # 分析智能体表现
        agents = data['agents']
        print(f"\\n智能体分析:")
        for agent in agents:
            agent_id = agent['agent_id']
            avg_score = np.mean(agent['performance_scores'])
            total_decisions = len(agent['decision_history'])
            print(f"  {agent_id}: {total_decisions} 次决策, 平均得分 {avg_score:.3f}")

def run_demo():
    """运行完整演示"""
    print("🎯 医院治理系统 - 数据导出集成演示")
    print("=" * 60)
    
    # 创建仿真实例
    simulation = SimulationWithExport("demo_export_sim")
    
    # 运行仿真
    simulation.run_simulation(duration=15)
    
    # 导出结果
    exported_files = simulation.export_results(format_type="all")
    
    # 分析结果
    if 'json' in exported_files:
        simulation.load_and_analyze(exported_files['json'])
    
    print(f"\\n📁 所有文件已导出到: {simulation.exporter.output_dir}")
    print("\\n🎉 演示完成！")

if __name__ == "__main__":
    run_demo()