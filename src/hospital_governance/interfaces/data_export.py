"""
医院治理系统 - 数据导出接口
支持多种格式的仿真数据导出功能
"""

import json
import csv
import pickle
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

# 可选依赖
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

logger = logging.getLogger(__name__)

@dataclass
class SimulationMetadata:
    """仿真元数据"""
    simulation_id: str
    start_time: datetime
    end_time: datetime
    duration_steps: int
    llm_provider: str
    system_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    
@dataclass
class TimeSeriesData:
    """时序数据"""
    timestamps: List[float]
    states: List[np.ndarray]
    controls: List[np.ndarray]
    observations: List[np.ndarray]
    rule_activations: List[Dict[str, Any]]
    performance_indices: List[float]
    stability_metrics: List[float]

@dataclass
class AgentDecisionData:
    """智能体决策数据"""
    agent_id: str
    decision_history: List[Dict[str, Any]]
    llm_responses: List[str]
    reasoning_chains: List[List[str]]
    performance_scores: List[float]

class DataExporter:
    """数据导出器"""
    
    def __init__(self, output_dir: str = "simulation_data"):
        """
        初始化数据导出器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_simulation_results(
        self,
        metadata: SimulationMetadata,
        time_series: TimeSeriesData,
        agent_data: List[AgentDecisionData],
        format_type: str = "all"
    ) -> Dict[str, str]:
        """
        导出完整仿真结果
        
        Args:
            metadata: 仿真元数据
            time_series: 时序数据
            agent_data: 智能体数据
            format_type: 导出格式 ("json", "csv", "excel", "pickle", "sqlite", "all")
            
        Returns:
            导出文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"simulation_{metadata.simulation_id}_{timestamp}"
        
        exported_files = {}
        
        if format_type in ["json", "all"]:
            json_file = self._export_to_json(metadata, time_series, agent_data, base_filename)
            exported_files["json"] = str(json_file)
            
        if format_type in ["csv", "all"]:
            csv_files = self._export_to_csv(metadata, time_series, agent_data, base_filename)
            exported_files["csv"] = csv_files
            
        if format_type in ["excel", "all"]:
            excel_file = self._export_to_excel(metadata, time_series, agent_data, base_filename)
            if excel_file:  # 只有在成功导出时才添加
                exported_files["excel"] = str(excel_file)
            
        if format_type in ["pickle", "all"]:
            pickle_file = self._export_to_pickle(metadata, time_series, agent_data, base_filename)
            exported_files["pickle"] = str(pickle_file)
            
        if format_type in ["sqlite", "all"]:
            db_file = self._export_to_sqlite(metadata, time_series, agent_data, base_filename)
            exported_files["sqlite"] = str(db_file)
            
        logger.info(f"导出完成: {len(exported_files)} 个文件")
        return exported_files
    
    def _export_to_json(
        self,
        metadata: SimulationMetadata,
        time_series: TimeSeriesData,
        agent_data: List[AgentDecisionData],
        base_filename: str
    ) -> Path:
        """导出为JSON格式"""
        json_file = self.output_dir / f"{base_filename}.json"
        
        # 准备JSON数据
        json_data = {
            "metadata": asdict(metadata),
            "time_series": {
                "timestamps": time_series.timestamps,
                "states": [state.tolist() for state in time_series.states],
                "controls": [ctrl.tolist() for ctrl in time_series.controls],
                "observations": [obs.tolist() for obs in time_series.observations],
                "rule_activations": time_series.rule_activations,
                "performance_indices": time_series.performance_indices,
                "stability_metrics": time_series.stability_metrics
            },
            "agents": [asdict(agent) for agent in agent_data]
        }
        
        # 处理datetime序列化
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=json_serializer)
            
        return json_file
    
    def _export_to_csv(
        self,
        metadata: SimulationMetadata,
        time_series: TimeSeriesData,
        agent_data: List[AgentDecisionData],
        base_filename: str
    ) -> Dict[str, str]:
        """导出为CSV格式"""
        csv_files = {}
        
        # 元数据CSV
        metadata_file = self.output_dir / f"{base_filename}_metadata.csv"
        with open(metadata_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(metadata).keys())
            writer.writeheader()
            
            # 处理复杂对象
            metadata_dict = asdict(metadata)
            metadata_dict['start_time'] = metadata.start_time.isoformat()
            metadata_dict['end_time'] = metadata.end_time.isoformat()
            metadata_dict['system_parameters'] = json.dumps(metadata.system_parameters)
            metadata_dict['performance_metrics'] = json.dumps(metadata.performance_metrics)
            
            writer.writerow(metadata_dict)
        csv_files["metadata"] = str(metadata_file)
        
        # 时序数据CSV
        timeseries_file = self.output_dir / f"{base_filename}_timeseries.csv"
        with open(timeseries_file, 'w', newline='', encoding='utf-8') as f:
            # 准备字段名
            fieldnames = ['timestamp', 'performance_index', 'stability_metric']
            
            # 添加状态字段
            if time_series.states:
                state_dim = len(time_series.states[0])
                fieldnames.extend([f'state_{i}' for i in range(state_dim)])
                
            # 添加控制字段
            if time_series.controls:
                control_dim = len(time_series.controls[0])
                fieldnames.extend([f'control_{i}' for i in range(control_dim)])
                
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 写入数据行
            for i in range(len(time_series.timestamps)):
                row = {
                    'timestamp': time_series.timestamps[i],
                    'performance_index': time_series.performance_indices[i],
                    'stability_metric': time_series.stability_metrics[i]
                }
                
                # 添加状态数据
                if time_series.states:
                    for j, val in enumerate(time_series.states[i]):
                        row[f'state_{j}'] = float(val)
                        
                # 添加控制数据
                if time_series.controls:
                    for j, val in enumerate(time_series.controls[i]):
                        row[f'control_{j}'] = float(val)
                        
                writer.writerow(row)
                
        csv_files["timeseries"] = str(timeseries_file)
        
        # 智能体数据CSV
        for agent in agent_data:
            agent_file = self.output_dir / f"{base_filename}_agent_{agent.agent_id}.csv"
            
            with open(agent_file, 'w', newline='', encoding='utf-8') as f:
                # 准备基础字段名
                base_fieldnames = ['step', 'agent_id', 'performance_score', 'llm_response']
                
                # 收集所有决策字段（从所有决策中获取）
                decision_fields = set()
                for decision in agent.decision_history:
                    for key in decision.keys():
                        decision_fields.add(f'decision_{key}')
                
                # 组合所有字段名
                fieldnames = base_fieldnames + sorted(list(decision_fields))
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # 写入智能体数据
                for i, decision in enumerate(agent.decision_history):
                    row = {
                        'step': i,
                        'agent_id': agent.agent_id,
                        'performance_score': agent.performance_scores[i] if i < len(agent.performance_scores) else None,
                        'llm_response': agent.llm_responses[i] if i < len(agent.llm_responses) else "",
                    }
                    
                    # 添加决策详情（先初始化所有字段为空）
                    for field in decision_fields:
                        row[field] = ""
                    
                    # 填入实际决策数据
                    for key, value in decision.items():
                        field_name = f'decision_{key}'
                        if field_name in fieldnames:
                            row[field_name] = str(value)
                        
                    writer.writerow(row)
                    
            csv_files[f"agent_{agent.agent_id}"] = str(agent_file)
        
        return csv_files
    
    def _export_to_excel(
        self,
        metadata: SimulationMetadata,
        time_series: TimeSeriesData,
        agent_data: List[AgentDecisionData],
        base_filename: str
    ) -> Path:
        """导出为Excel格式（需要pandas和openpyxl）"""
        if not HAS_PANDAS or not HAS_OPENPYXL:
            logger.warning("Excel导出需要pandas和openpyxl库，跳过Excel导出")
            return None
            
        excel_file = self.output_dir / f"{base_filename}.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 元数据工作表
            metadata_dict = asdict(metadata)
            metadata_dict['start_time'] = metadata.start_time.isoformat()
            metadata_dict['end_time'] = metadata.end_time.isoformat()
            metadata_dict['system_parameters'] = json.dumps(metadata.system_parameters)
            metadata_dict['performance_metrics'] = json.dumps(metadata.performance_metrics)
            
            metadata_df = pd.DataFrame([metadata_dict])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            # 时序数据工作表
            timeseries_df = pd.DataFrame({
                'timestamp': time_series.timestamps,
                'performance_index': time_series.performance_indices,
                'stability_metric': time_series.stability_metrics
            })
            
            # 添加状态和控制数据
            for i in range(len(time_series.states[0])):
                timeseries_df[f'state_{i}'] = [state[i] for state in time_series.states]
            for i in range(len(time_series.controls[0])):
                timeseries_df[f'control_{i}'] = [ctrl[i] for ctrl in time_series.controls]
                
            timeseries_df.to_excel(writer, sheet_name='TimeSeries', index=False)
            
            # 规则激活工作表
            rule_activations_data = []
            for i, activations in enumerate(time_series.rule_activations):
                for rule_name, details in activations.items():
                    rule_activations_data.append({
                        'step': i,
                        'rule_name': rule_name,
                        'activated': details.get('activated', False),
                        'severity': details.get('severity', 0),
                        'description': details.get('description', '')
                    })
            
            if rule_activations_data:
                rules_df = pd.DataFrame(rule_activations_data)
                rules_df.to_excel(writer, sheet_name='RuleActivations', index=False)
            
            # 智能体数据工作表
            for agent in agent_data:
                agent_rows = []
                for i, decision in enumerate(agent.decision_history):
                    row = {
                        'step': i,
                        'performance_score': agent.performance_scores[i] if i < len(agent.performance_scores) else None,
                        'llm_response_length': len(agent.llm_responses[i]) if i < len(agent.llm_responses) else 0,
                    }
                    # 添加决策摘要
                    row.update({f'decision_{k}': str(v)[:100] for k, v in decision.items()})
                    agent_rows.append(row)
                
                if agent_rows:
                    agent_df = pd.DataFrame(agent_rows)
                    sheet_name = f'Agent_{agent.agent_id}'[:31]  # Excel工作表名称限制
                    agent_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return excel_file
    
    def _export_to_pickle(
        self,
        metadata: SimulationMetadata,
        time_series: TimeSeriesData,
        agent_data: List[AgentDecisionData],
        base_filename: str
    ) -> Path:
        """导出为Pickle格式（保持Python对象结构）"""
        pickle_file = self.output_dir / f"{base_filename}.pkl"
        
        data = {
            'metadata': metadata,
            'time_series': time_series,
            'agent_data': agent_data
        }
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
            
        return pickle_file
    
    def _export_to_sqlite(
        self,
        metadata: SimulationMetadata,
        time_series: TimeSeriesData,
        agent_data: List[AgentDecisionData],
        base_filename: str
    ) -> Path:
        """导出为SQLite数据库"""
        db_file = self.output_dir / f"{base_filename}.db"
        
        conn = sqlite3.connect(db_file)
        
        try:
            # 创建表结构
            self._create_sqlite_schema(conn)
            
            # 插入元数据
            self._insert_metadata(conn, metadata)
            
            # 插入时序数据
            self._insert_timeseries(conn, time_series, metadata.simulation_id)
            
            # 插入智能体数据
            self._insert_agent_data(conn, agent_data, metadata.simulation_id)
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"SQLite导出失败: {e}")
            raise
        finally:
            conn.close()
            
        return db_file
    
    def _create_sqlite_schema(self, conn: sqlite3.Connection):
        """创建SQLite数据库架构"""
        cursor = conn.cursor()
        
        # 仿真元数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulations (
                simulation_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                duration_steps INTEGER,
                llm_provider TEXT,
                system_parameters TEXT,
                performance_metrics TEXT
            )
        ''')
        
        # 时序数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timeseries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT,
                step INTEGER,
                timestamp REAL,
                state_vector TEXT,
                control_vector TEXT,
                observation_vector TEXT,
                performance_index REAL,
                stability_metric REAL,
                FOREIGN KEY (simulation_id) REFERENCES simulations (simulation_id)
            )
        ''')
        
        # 规则激活表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rule_activations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT,
                step INTEGER,
                rule_name TEXT,
                activated BOOLEAN,
                severity REAL,
                description TEXT,
                FOREIGN KEY (simulation_id) REFERENCES simulations (simulation_id)
            )
        ''')
        
        # 智能体决策表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT,
                agent_id TEXT,
                step INTEGER,
                decision_data TEXT,
                llm_response TEXT,
                reasoning_chain TEXT,
                performance_score REAL,
                FOREIGN KEY (simulation_id) REFERENCES simulations (simulation_id)
            )
        ''')
        
    def _insert_metadata(self, conn: sqlite3.Connection, metadata: SimulationMetadata):
        """插入元数据"""
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO simulations 
            (simulation_id, start_time, end_time, duration_steps, llm_provider, 
             system_parameters, performance_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata.simulation_id,
            metadata.start_time.isoformat(),
            metadata.end_time.isoformat(),
            metadata.duration_steps,
            metadata.llm_provider,
            json.dumps(metadata.system_parameters),
            json.dumps(metadata.performance_metrics)
        ))
        
    def _insert_timeseries(self, conn: sqlite3.Connection, time_series: TimeSeriesData, sim_id: str):
        """插入时序数据"""
        cursor = conn.cursor()
        
        for i, timestamp in enumerate(time_series.timestamps):
            # 插入时序记录
            cursor.execute('''
                INSERT INTO timeseries 
                (simulation_id, step, timestamp, state_vector, control_vector, 
                 observation_vector, performance_index, stability_metric)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sim_id, i, timestamp,
                json.dumps(time_series.states[i].tolist()),
                json.dumps(time_series.controls[i].tolist()),
                json.dumps(time_series.observations[i].tolist()),
                time_series.performance_indices[i],
                time_series.stability_metrics[i]
            ))
            
            # 插入规则激活记录
            if i < len(time_series.rule_activations):
                for rule_name, details in time_series.rule_activations[i].items():
                    cursor.execute('''
                        INSERT INTO rule_activations 
                        (simulation_id, step, rule_name, activated, severity, description)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        sim_id, i, rule_name,
                        details.get('activated', False),
                        details.get('severity', 0),
                        details.get('description', '')
                    ))
    
    def _insert_agent_data(self, conn: sqlite3.Connection, agent_data: List[AgentDecisionData], sim_id: str):
        """插入智能体数据"""
        cursor = conn.cursor()
        
        for agent in agent_data:
            for i, decision in enumerate(agent.decision_history):
                cursor.execute('''
                    INSERT INTO agent_decisions 
                    (simulation_id, agent_id, step, decision_data, llm_response, 
                     reasoning_chain, performance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sim_id, agent.agent_id, i,
                    json.dumps(decision),
                    agent.llm_responses[i] if i < len(agent.llm_responses) else "",
                    json.dumps(agent.reasoning_chains[i] if i < len(agent.reasoning_chains) else []),
                    agent.performance_scores[i] if i < len(agent.performance_scores) else None
                ))

class DataImporter:
    """数据导入器"""
    
    def __init__(self, data_dir: str = "simulation_data"):
        """
        初始化数据导入器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = Path(data_dir)
        
    def import_from_json(self, json_file: Union[str, Path]) -> Dict[str, Any]:
        """从JSON文件导入数据"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 重建numpy数组
        if 'time_series' in data:
            ts = data['time_series']
            ts['states'] = [np.array(state) for state in ts['states']]
            ts['controls'] = [np.array(ctrl) for ctrl in ts['controls']]
            ts['observations'] = [np.array(obs) for obs in ts['observations']]
            
        return data
    
    def import_from_pickle(self, pickle_file: Union[str, Path]) -> Dict[str, Any]:
        """从Pickle文件导入数据"""
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    
    def import_from_sqlite(self, db_file: Union[str, Path], simulation_id: str) -> Dict[str, Any]:
        """从SQLite数据库导入数据"""
        conn = sqlite3.connect(db_file)
        
        try:
            # 导入元数据
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM simulations WHERE simulation_id = ?", [simulation_id])
            metadata_row = cursor.fetchone()
            
            if not metadata_row:
                raise ValueError(f"未找到仿真ID: {simulation_id}")
            
            # 获取列名
            cursor.execute("PRAGMA table_info(simulations)")
            columns = [row[1] for row in cursor.fetchall()]
            metadata = dict(zip(columns, metadata_row))
                
            # 导入时序数据
            cursor.execute("SELECT * FROM timeseries WHERE simulation_id = ? ORDER BY step", [simulation_id])
            timeseries_rows = cursor.fetchall()
            
            cursor.execute("PRAGMA table_info(timeseries)")
            ts_columns = [row[1] for row in cursor.fetchall()]
            timeseries = [dict(zip(ts_columns, row)) for row in timeseries_rows]
            
            # 导入规则激活数据
            cursor.execute("SELECT * FROM rule_activations WHERE simulation_id = ? ORDER BY step, rule_name", [simulation_id])
            rules_rows = cursor.fetchall()
            
            cursor.execute("PRAGMA table_info(rule_activations)")
            rules_columns = [row[1] for row in cursor.fetchall()]
            rules = [dict(zip(rules_columns, row)) for row in rules_rows]
            
            # 导入智能体数据
            cursor.execute("SELECT * FROM agent_decisions WHERE simulation_id = ? ORDER BY agent_id, step", [simulation_id])
            agents_rows = cursor.fetchall()
            
            cursor.execute("PRAGMA table_info(agent_decisions)")
            agents_columns = [row[1] for row in cursor.fetchall()]
            agents = [dict(zip(agents_columns, row)) for row in agents_rows]
            
            return {
                'metadata': metadata,
                'timeseries': timeseries,
                'rules': rules,
                'agents': agents
            }
            
        finally:
            conn.close()

def create_sample_data() -> tuple:
    """创建示例数据用于测试"""
    # 示例元数据
    metadata = SimulationMetadata(
        simulation_id="test_sim_001",
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_steps=10,
        llm_provider="mock",
        system_parameters={"param1": 1.0, "param2": "test"},
        performance_metrics={"stability": 0.95, "efficiency": 0.88}
    )
    
    # 示例时序数据
    time_series = TimeSeriesData(
        timestamps=list(range(10)),
        states=[np.random.rand(5) for _ in range(10)],
        controls=[np.random.rand(3) for _ in range(10)],
        observations=[np.random.rand(4) for _ in range(10)],
        rule_activations=[
            {"rule1": {"activated": True, "severity": 0.5, "description": "测试规则"}}
            for _ in range(10)
        ],
        performance_indices=np.random.rand(10).tolist(),
        stability_metrics=np.random.rand(10).tolist()
    )
    
    # 示例智能体数据
    agent_data = [
        AgentDecisionData(
            agent_id="doctor_agent",
            decision_history=[{"action": f"action_{i}", "confidence": 0.8} for i in range(10)],
            llm_responses=[f"Response {i}" for i in range(10)],
            reasoning_chains=[[f"step{i}_1", f"step{i}_2"] for i in range(10)],
            performance_scores=np.random.rand(10).tolist()
        )
    ]
    
    return metadata, time_series, agent_data

if __name__ == "__main__":
    # 测试数据导出功能
    print("🧪 测试数据导出功能...")
    
    # 创建示例数据
    metadata, time_series, agent_data = create_sample_data()
    
    # 创建导出器
    exporter = DataExporter("test_export")
    
    # 导出数据
    try:
        exported_files = exporter.export_simulation_results(
            metadata, time_series, agent_data, format_type="all"
        )
        
        print("✅ 导出成功!")
        for format_name, file_path in exported_files.items():
            print(f"  {format_name}: {file_path}")
            
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        
    print("\n📚 数据导出模块使用示例:")
    print("  from src.hospital_governance.interfaces.data_export import DataExporter")
    print("  exporter = DataExporter('output_dir')")
    print("  files = exporter.export_simulation_results(metadata, time_series, agent_data)")