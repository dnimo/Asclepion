#!/usr/bin/env python3
"""
医院治理系统 - 独立数据导出演示
直接使用数据导出模块，不依赖其他组件
"""

import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# 直接导入数据导出模块
sys.path.insert(0, str(Path(__file__).parent / "src/hospital_governance/interfaces"))

from data_export import (
    DataExporter, DataImporter, SimulationMetadata, 
    TimeSeriesData, AgentDecisionData
)

def create_realistic_simulation_data():
    """创建逼真的仿真数据"""
    print("📊 生成仿真数据...")
    
    duration = 25
    start_time = datetime.now()
    
    # 仿真元数据
    metadata = SimulationMetadata(
        simulation_id="hospital_governance_demo",
        start_time=start_time,
        end_time=start_time,  # 稍后更新
        duration_steps=duration,
        llm_provider="gpt-4",
        system_parameters={
            "hospital_capacity": 500,
            "num_doctors": 20,
            "num_nurses": 50,
            "emergency_threshold": 0.3,
            "efficiency_weight": 0.6,
            "safety_weight": 0.4
        },
        performance_metrics={
            "patient_satisfaction": 0.85,
            "resource_efficiency": 0.78,
            "safety_score": 0.92,
            "response_time": 15.2
        }
    )
    
    # 时序数据生成
    timestamps = []
    states = []
    controls = []
    observations = []
    rule_activations = []
    performance_indices = []
    stability_metrics = []
    
    # 初始状态：[床位占用率, 医生工作负荷, 护士工作负荷, 药品库存, 设备可用性, 患者满意度, 急诊队列]
    current_state = np.array([0.7, 0.6, 0.65, 0.8, 0.9, 0.85, 0.2])
    
    for step in range(duration):
        timestamps.append(step * 0.5)  # 每步代表30分钟
        
        # 模拟随机事件影响
        if step == 8:  # 第8步发生紧急情况
            current_state[6] += 0.4  # 急诊队列增加
            current_state[1] += 0.2  # 医生工作负荷增加
        elif step == 15:  # 第15步药品补充
            current_state[3] = 0.95  # 药品库存补充
        
        # 添加噪声
        noise = np.random.normal(0, 0.05, len(current_state))
        current_state += noise
        current_state = np.clip(current_state, 0, 1)
        
        # 控制决策：[人员调配, 资源分配, 紧急响应]
        control = np.array([
            0.1 if current_state[1] > 0.8 else 0.0,  # 人员调配
            0.15 if current_state[3] < 0.3 else 0.05,  # 资源分配
            0.8 if current_state[6] > 0.5 else 0.1   # 紧急响应
        ])
        
        # 应用控制影响
        current_state[1] -= control[0] * 0.3  # 减少医生负荷
        current_state[2] -= control[0] * 0.2  # 减少护士负荷
        current_state[3] += control[1] * 0.4  # 增加资源
        current_state[6] -= control[2] * 0.6  # 减少急诊队列
        current_state = np.clip(current_state, 0, 1)
        
        # 观测（加入观测噪声）
        observation = current_state + np.random.normal(0, 0.02, len(current_state))
        observation = np.clip(observation, 0, 1)
        
        # 性能指标计算
        efficiency = 1 - np.mean([current_state[1], current_state[2]])  # 工作负荷越低效率越高
        safety = current_state[5]  # 患者满意度代表安全
        performance_index = 0.6 * efficiency + 0.4 * safety
        
        # 稳定性指标
        control_effort = np.linalg.norm(control)
        stability_metric = np.exp(-control_effort)
        
        # 规则激活
        activations = {}
        
        if current_state[1] > 0.85:  # 医生过载
            activations["doctor_overload_protocol"] = {
                "activated": True,
                "severity": current_state[1] - 0.85,
                "description": "医生工作负荷过高，启动支援协议"
            }
        
        if current_state[6] > 0.6:  # 急诊拥挤
            activations["emergency_overflow_protocol"] = {
                "activated": True,
                "severity": current_state[6] - 0.6,
                "description": "急诊科过载，启动分流协议"
            }
        
        if current_state[3] < 0.2:  # 药品短缺
            activations["medication_shortage_alert"] = {
                "activated": True,
                "severity": 0.2 - current_state[3],
                "description": "药品库存不足，需要紧急补充"
            }
        
        if performance_index > 0.9:  # 高效运营
            activations["high_performance_mode"] = {
                "activated": True,
                "severity": performance_index - 0.9,
                "description": "系统高效运营中，保持当前状态"
            }
        
        # 记录数据
        states.append(current_state.copy())
        controls.append(control.copy())
        observations.append(observation.copy())
        rule_activations.append(activations)
        performance_indices.append(performance_index)
        stability_metrics.append(stability_metric)
    
    # 更新结束时间
    metadata.end_time = datetime.now()
    
    # 创建时序数据对象
    time_series = TimeSeriesData(
        timestamps=timestamps,
        states=states,
        controls=controls,
        observations=observations,
        rule_activations=rule_activations,
        performance_indices=performance_indices,
        stability_metrics=stability_metrics
    )
    
    return metadata, time_series

def create_agent_data():
    """创建智能体决策数据"""
    print("🤖 生成智能体数据...")
    
    agents = []
    duration = 25
    
    # 主治医生智能体
    doctor_decisions = []
    doctor_responses = []
    doctor_reasoning = []
    doctor_scores = []
    
    for step in range(duration):
        # 医生决策逻辑
        if step < 8:
            decision = {
                "action": "routine_checkup",
                "priority": "normal", 
                "patients_assigned": 3,
                "estimated_duration": 45
            }
            response = f"医生进行常规查房，安排3名患者，预计45分钟完成"
            reasoning = [
                "评估当前患者状况",
                "安排常规医疗流程",
                "确保医疗质量"
            ]
            score = 0.8 + np.random.normal(0, 0.1)
            
        elif step < 15:  # 紧急期间
            decision = {
                "action": "emergency_response",
                "priority": "urgent",
                "patients_assigned": 5,
                "estimated_duration": 90
            }
            response = f"医生响应紧急情况，处理5名急诊患者，预计90分钟"
            reasoning = [
                "识别紧急医疗情况",
                "启动急救流程",
                "优先处理危重患者",
                "协调医疗资源"
            ]
            score = 0.9 + np.random.normal(0, 0.05)
            
        else:  # 恢复期
            decision = {
                "action": "recovery_monitoring",
                "priority": "normal",
                "patients_assigned": 4,
                "estimated_duration": 60
            }
            response = f"医生监控患者康复情况，跟进4名患者"
            reasoning = [
                "评估患者康复进度",
                "调整治疗方案",
                "安排后续护理"
            ]
            score = 0.85 + np.random.normal(0, 0.08)
        
        doctor_decisions.append(decision)
        doctor_responses.append(response)
        doctor_reasoning.append(reasoning)
        doctor_scores.append(np.clip(score, 0, 1))
    
    agents.append(AgentDecisionData(
        agent_id="senior_doctor",
        decision_history=doctor_decisions,
        llm_responses=doctor_responses,
        reasoning_chains=doctor_reasoning,
        performance_scores=doctor_scores
    ))
    
    # 护士长智能体
    nurse_decisions = []
    nurse_responses = []
    nurse_reasoning = []
    nurse_scores = []
    
    for step in range(duration):
        if step < 8:
            decision = {
                "action": "patient_monitoring",
                "shift_assignment": "day_shift",
                "nurses_coordinated": 8,
                "focus_area": "general_ward"
            }
            response = f"护士长协调日班8名护士，重点监护普通病房"
            reasoning = [
                "安排护理人员班次",
                "分配病房监护任务",
                "确保护理质量"
            ]
            score = 0.75 + np.random.normal(0, 0.1)
            
        elif step < 15:
            decision = {
                "action": "emergency_coordination",
                "shift_assignment": "emergency_shift",
                "nurses_coordinated": 12,
                "focus_area": "emergency_department"
            }
            response = f"护士长紧急调配12名护士支援急诊科"
            reasoning = [
                "响应紧急医疗需求",
                "重新分配护理资源",
                "确保急诊科护理覆盖",
                "维持其他科室基本护理"
            ]
            score = 0.88 + np.random.normal(0, 0.06)
            
        else:
            decision = {
                "action": "quality_assessment",
                "shift_assignment": "evaluation_mode",
                "nurses_coordinated": 10,
                "focus_area": "comprehensive_care"
            }
            response = f"护士长评估护理质量，协调10名护士提供全面护理"
            reasoning = [
                "评估护理服务质量",
                "总结应急响应经验", 
                "优化护理流程"
            ]
            score = 0.82 + np.random.normal(0, 0.07)
        
        nurse_decisions.append(decision)
        nurse_responses.append(response)
        nurse_reasoning.append(reasoning)
        nurse_scores.append(np.clip(score, 0, 1))
    
    agents.append(AgentDecisionData(
        agent_id="head_nurse",
        decision_history=nurse_decisions,
        llm_responses=nurse_responses,
        reasoning_chains=nurse_reasoning,
        performance_scores=nurse_scores
    ))
    
    # 管理员智能体
    admin_decisions = []
    admin_responses = []
    admin_reasoning = []
    admin_scores = []
    
    for step in range(duration):
        if step == 5:
            decision = {
                "action": "resource_procurement",
                "budget_allocated": 50000,
                "items": ["medication", "medical_supplies"],
                "priority": "routine"
            }
            response = f"管理员分配5万元预算采购药品和医疗用品"
            
        elif step == 12:
            decision = {
                "action": "emergency_budget_approval", 
                "budget_allocated": 80000,
                "items": ["emergency_staff", "equipment_rental"],
                "priority": "urgent"
            }
            response = f"管理员紧急批准8万元预算用于人员和设备支援"
            
        elif step == 20:
            decision = {
                "action": "performance_review",
                "metrics_analyzed": ["efficiency", "satisfaction", "cost"],
                "improvement_areas": ["emergency_response", "resource_planning"],
                "priority": "strategic"
            }
            response = f"管理员进行绩效分析，识别改进领域"
            
        else:
            decision = {
                "action": "routine_management",
                "tasks": ["scheduling", "reporting", "coordination"],
                "priority": "normal"
            }
            response = f"管理员执行日常管理任务"
        
        admin_decisions.append(decision)
        admin_responses.append(response)
        admin_reasoning.append([
            "分析运营数据",
            "评估资源需求",
            "制定管理决策"
        ])
        admin_scores.append(0.7 + np.random.normal(0, 0.12))
    
    agents.append(AgentDecisionData(
        agent_id="hospital_administrator",
        decision_history=admin_decisions,
        llm_responses=admin_responses,
        reasoning_chains=admin_reasoning,
        performance_scores=[np.clip(s, 0, 1) for s in admin_scores]
    ))
    
    return agents

def run_comprehensive_demo():
    """运行完整的数据导出演示"""
    print("🏥 医院治理系统 - 完整数据导出演示")
    print("=" * 60)
    
    # 生成仿真数据
    metadata, time_series = create_realistic_simulation_data()
    agent_data = create_agent_data()
    
    # 创建导出器
    exporter = DataExporter("comprehensive_export")
    
    print(f"\\n📊 仿真概况:")
    print(f"  仿真ID: {metadata.simulation_id}")
    print(f"  持续时间: {metadata.duration_steps} 步 ({metadata.duration_steps * 0.5} 小时)")
    print(f"  状态维度: {len(time_series.states[0])}")
    print(f"  控制维度: {len(time_series.controls[0])}")
    print(f"  智能体数量: {len(agent_data)}")
    
    # 导出所有格式
    print(f"\\n💾 导出数据...")
    exported_files = exporter.export_simulation_results(
        metadata, time_series, agent_data, format_type="all"
    )
    
    print(f"\\n✅ 导出完成:")
    for format_name, file_info in exported_files.items():
        if isinstance(file_info, dict):
            print(f"  📂 {format_name.upper()}: {len(file_info)} 个文件")
            for name, path in file_info.items():
                file_size = Path(path).stat().st_size / 1024  # KB
                print(f"     - {name}: {Path(path).name} ({file_size:.1f} KB)")
        else:
            file_size = Path(file_info).stat().st_size / 1024  # KB
            print(f"  📄 {format_name.upper()}: {Path(file_info).name} ({file_size:.1f} KB)")
    
    # 数据分析
    print(f"\\n📈 数据分析:")
    avg_performance = np.mean(time_series.performance_indices)
    avg_stability = np.mean(time_series.stability_metrics)
    total_rules = sum(len(ra) for ra in time_series.rule_activations)
    
    print(f"  平均性能指标: {avg_performance:.3f}")
    print(f"  平均稳定性: {avg_stability:.3f}")
    print(f"  规则激活总数: {total_rules}")
    print(f"  最终系统状态: {time_series.states[-1]}")
    
    print(f"\\n🤖 智能体表现:")
    for agent in agent_data:
        avg_score = np.mean(agent.performance_scores)
        decision_types = set(d.get('action', 'unknown') for d in agent.decision_history)
        print(f"  {agent.agent_id}: 平均得分 {avg_score:.3f}, 决策类型 {len(decision_types)} 种")
    
    # 演示数据加载
    if 'json' in exported_files:
        print(f"\\n🔄 演示数据加载...")
        importer = DataImporter("comprehensive_export")
        loaded_data = importer.import_from_json(exported_files['json'])
        
        print(f"  ✅ 成功加载JSON数据")
        print(f"  📊 时序数据点: {len(loaded_data['time_series']['timestamps'])}")
        print(f"  🤖 智能体记录: {len(loaded_data['agents'])}")
    
    print(f"\\n📁 所有文件位置: {exporter.output_dir}")
    print(f"\\n🎉 演示完成！")
    
    return exported_files

if __name__ == "__main__":
    run_comprehensive_demo()