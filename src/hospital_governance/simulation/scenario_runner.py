import numpy as np
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import json

class ScenarioType(Enum):
    """场景类型"""
    BASELINE = "baseline"                    # 基线场景
    FUNDING_CUT = "funding_cut"              # 资金削减
    PANDEMIC = "pandemic"                    # 疫情爆发
    FINANCIAL_CRISIS = "financial_crisis"    # 财务危机
    POLICY_CHANGE = "policy_change"          # 政策变化
    STAFF_SHORTAGE = "staff_shortage"        # 人员短缺
    EQUIPMENT_FAILURE = "equipment_failure"  # 设备故障

@dataclass
class CrisisScenario:
    """危机场景配置"""
    name: str
    scenario_type: ScenarioType
    severity: float  # 严重程度 0-1
    duration: int    # 持续时间（步数）
    start_step: int  # 开始步数
    affected_metrics: List[str]  # 影响的指标
    trigger_condition: Optional[Callable] = None  # 触发条件
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.scenario_type.value,
            'severity': self.severity,
            'duration': self.duration,
            'start_step': self.start_step,
            'affected_metrics': self.affected_metrics
        }

class ScenarioRunner:
    """场景运行器"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        self.active_scenarios: List[CrisisScenario] = []
        self.scenario_history: List[Dict] = []
        self.custom_metrics: Dict[str, List[float]] = {}
        self.scenarios: List[CrisisScenario] = []

    def load_scenarios_from_yaml(self, filepath: str) -> None:
        """从yaml文件加载场景"""
        import yaml
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                configs = yaml.safe_load(f)
            self.scenarios = [self.load_scenario(cfg) for cfg in configs]
            print(f"从 {filepath} 加载了 {len(self.scenarios)} 个场景")
        except Exception as e:
            print(f"加载场景yaml错误: {e}")

    def check_and_insert_event(self, current_step: int):
        """根据当前步数检测并插入危机事件"""
        for scenario in self.scenarios:
            if scenario.start_step == current_step:
                self.simulator._apply_crisis_effects({
                    'type': scenario.scenario_type.value,
                    'severity': scenario.severity
                })
                self.active_scenarios.append(scenario)
    
    def load_scenario(self, scenario_config: Dict[str, Any]) -> CrisisScenario:
        """加载场景配置"""
        scenario = CrisisScenario(
            name=scenario_config['name'],
            scenario_type=ScenarioType(scenario_config['type']),
            severity=scenario_config['severity'],
            duration=scenario_config['duration'],
            start_step=scenario_config.get('start_step', 0),
            affected_metrics=scenario_config['affected_metrics']
        )
        return scenario
    
    def load_scenarios_from_file(self, filepath: str) -> List[CrisisScenario]:
        """从文件加载场景"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            
            scenarios = []
            for config in configs:
                scenario = self.load_scenario(config)
                scenarios.append(scenario)
            
            print(f"从 {filepath} 加载了 {len(scenarios)} 个场景")
            return scenarios
        except Exception as e:
            print(f"加载场景文件错误: {e}")
            return []
    
    def create_preset_scenarios(self) -> Dict[str, CrisisScenario]:
        """创建预设场景"""
        presets = {
            'mild_funding_cut': CrisisScenario(
                name="轻度资金削减",
                scenario_type=ScenarioType.FUNDING_CUT,
                severity=0.3,
                duration=50,
                start_step=100,
                affected_metrics=['financial_health', 'resource_utilization']
            ),
            'moderate_pandemic': CrisisScenario(
                name="中度疫情爆发",
                scenario_type=ScenarioType.PANDEMIC,
                severity=0.6,
                duration=80,
                start_step=200,
                affected_metrics=['resource_utilization', 'medical_quality', 'operational_efficiency']
            ),
            'severe_financial_crisis': CrisisScenario(
                name="严重财务危机",
                scenario_type=ScenarioType.FINANCIAL_CRISIS,
                severity=0.8,
                duration=100,
                start_step=300,
                affected_metrics=['financial_health', 'patient_satisfaction', 'education_effectiveness']
            ),
            'policy_reform': CrisisScenario(
                name="政策改革",
                scenario_type=ScenarioType.POLICY_CHANGE,
                severity=0.4,
                duration=120,
                start_step=400,
                affected_metrics=['operational_efficiency', 'medical_quality', 'financial_health']
            )
        }
        return presets
    
    def run_scenario(self, scenario: CrisisScenario, steps: int = 500, 
                    training: bool = False) -> Dict[str, Any]:
        """运行单个场景"""
        print(f"开始运行场景: {scenario.name}")
        
        # 重置模拟器
        self.simulator.reset()
        
        # 运行场景
        results = self.simulator.run(steps, training)
        
        # 应用场景效果
        self._apply_scenario_effects(scenario, results)
        
        # 计算场景特定指标
        scenario_metrics = self._calculate_scenario_metrics(scenario, results)
        
        # 记录场景历史
        scenario_record = {
            'scenario': scenario.to_dict(),
            'results': results,
            'metrics': scenario_metrics,
            'summary': self.simulator.get_simulation_summary()
        }
        self.scenario_history.append(scenario_record)
        
        print(f"场景完成: {scenario.name}")
        return scenario_record
    
    def run_scenario_sequence(self, scenarios: List[CrisisScenario], 
                            steps_per_scenario: int = 300,
                            training: bool = False) -> List[Dict[str, Any]]:
        """运行场景序列"""
        all_results = []
        
        for i, scenario in enumerate(scenarios):
            print(f"运行场景 {i+1}/{len(scenarios)}: {scenario.name}")
            
            # 调整开始步数
            scenario.start_step = i * steps_per_scenario
            
            # 运行场景
            result = self.run_scenario(scenario, steps_per_scenario, training)
            all_results.append(result)
            
            # 场景间的过渡
            if i < len(scenarios) - 1:
                self._apply_inter_scenario_transition()
        
        return all_results
    
    def run_comparative_study(self, base_scenario: CrisisScenario,
                            variant_scenarios: List[CrisisScenario],
                            steps: int = 400) -> Dict[str, Any]:
        """运行对比研究"""
        comparative_results = {}
        
        # 运行基线场景
        print("运行基线场景...")
        base_results = self.run_scenario(base_scenario, steps)
        comparative_results['baseline'] = base_results
        
        # 运行变体场景
        for i, variant in enumerate(variant_scenarios):
            print(f"运行变体场景 {i+1}/{len(variant_scenarios)}...")
            variant_results = self.run_scenario(variant, steps)
            comparative_results[f'variant_{i+1}'] = variant_results
        
        # 计算对比指标
        comparative_metrics = self._calculate_comparative_metrics(comparative_results)
        comparative_results['comparative_metrics'] = comparative_metrics
        
        return comparative_results
    
    def _apply_scenario_effects(self, scenario: CrisisScenario, results: List[Dict[str, Any]]):
        """应用场景效果"""
        for step_data in results:
            if scenario.start_step <= step_data['step'] <= scenario.start_step + scenario.duration:
                # 应用场景影响
                for metric in scenario.affected_metrics:
                    if metric in step_data['system_state']:
                        impact = scenario.severity * 0.1
                        step_data['system_state'][metric] -= impact
        
        print(f"应用了场景效果: {scenario.name}")
    
    def _apply_inter_scenario_transition(self):
        """应用场景间过渡"""
        # 简化的过渡逻辑 - 可以扩展为更复杂的过渡策略
        transition_steps = 20
        
        for step in range(transition_steps):
            self.simulator.step(training=False)
        
        print("场景过渡完成")
    
    def _calculate_scenario_metrics(self, scenario: CrisisScenario, 
                                  results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算场景特定指标"""
        if not results:
            return {}
        
        # 提取相关数据
        steps = [r['step'] for r in results]
        system_states = [r['system_state'] for r in results]
        metrics = [r['metrics'] for r in results]
        
        # 场景期间的数据
        scenario_start = scenario.start_step
        scenario_end = scenario.start_step + scenario.duration
        scenario_indices = [i for i, step in enumerate(steps) 
                          if scenario_start <= step <= scenario_end]
        
        if not scenario_indices:
            return {}
        
        # 计算受影响指标的平均下降
        impact_metrics = {}
        for metric in scenario.affected_metrics:
            baseline = system_states[scenario_start - 10][metric] if scenario_start > 10 else 0.7
            scenario_values = [system_states[i][metric] for i in scenario_indices]
            avg_value = np.mean(scenario_values) if scenario_values else baseline
            impact_metrics[f'{metric}_impact'] = baseline - avg_value
        
        # 计算恢复指标
        recovery_indices = [i for i, step in enumerate(steps) if step > scenario_end]
        if recovery_indices:
            recovery_values = [system_states[i][metric] for i in recovery_indices[:20] 
                             for metric in scenario.affected_metrics]
            recovery_metric = np.mean(recovery_values) if recovery_values else 0.5
        else:
            recovery_metric = 0.5
        
        # 系统稳定性
        stability_values = [m.get('stability', 0.5) for m in metrics]
        avg_stability = np.mean(stability_values) if stability_values else 0.5
        
        return {
            'scenario_severity': scenario.severity,
            'duration_impact': scenario.duration / 100.0,
            'average_stability': avg_stability,
            'recovery_index': recovery_metric,
            **impact_metrics
        }
    
    def _calculate_comparative_metrics(self, comparative_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算对比指标"""
        metrics = {}
        
        baseline_summary = comparative_results['baseline']['summary']
        baseline_performance = baseline_summary['average_performance']
        
        for key, result in comparative_results.items():
            if key == 'baseline':
                continue
            
            summary = result['summary']
            performance = summary['average_performance']
            
            # 计算相对于基线的性能变化
            performance_change = (performance - baseline_performance) / baseline_performance
            
            metrics[key] = {
                'performance_change': performance_change,
                'final_stability': summary['final_system_state']['stability_index'],
                'crisis_handling': len(result['results']) / len(comparative_results['baseline']['results'])
            }
        
        return metrics
    
    def save_scenario_results(self, filepath: str):
        """保存场景结果"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 简化数据以节省空间
                simplified_history = []
                for record in self.scenario_history:
                    simplified = {
                        'scenario': record['scenario'],
                        'metrics': record['metrics'],
                        'summary': record['summary']
                    }
                    simplified_history.append(simplified)
                
                json.dump(simplified_history, f, indent=2, ensure_ascii=False)
            
            print(f"场景结果已保存到: {filepath}")
        except Exception as e:
            print(f"保存场景结果错误: {e}")
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """获取场景统计"""
        if not self.scenario_history:
            return {}
        
        total_scenarios = len(self.scenario_history)
        scenario_types = [record['scenario']['type'] for record in self.scenario_history]
        
        type_counts = {}
        for scenario_type in scenario_types:
            type_counts[scenario_type] = type_counts.get(scenario_type, 0) + 1
        
        avg_performance = np.mean([record['summary']['average_performance'] 
                                 for record in self.scenario_history])
        
        return {
            'total_scenarios': total_scenarios,
            'scenario_type_distribution': type_counts,
            'average_performance': avg_performance,
            'most_common_scenario': max(type_counts, key=type_counts.get) if type_counts else None
        }