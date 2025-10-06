import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import json
import csv
import os
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class SimulationMetrics:
    """模拟指标"""
    step: int
    timestamp: float
    system_state: Dict[str, float]
    performance_metrics: Dict[str, float]
    role_actions: Dict[str, List[float]]
    role_rewards: Dict[str, float]
    decisions: List[Dict[str, Any]]
    crises: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'timestamp': self.timestamp,
            'system_state': self.system_state,
            'performance_metrics': self.performance_metrics,
            'role_actions': self.role_actions,
            'role_rewards': self.role_rewards,
            'decisions': self.decisions,
            'crises': self.crises
        }

class DataLogger:
    """数据记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.current_session: Optional[str] = None
        self.metrics_history: List[SimulationMetrics] = []
        self.session_config: Dict[str, Any] = {}
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
    
    def start_session(self, session_name: Optional[str] = None, 
                     config: Dict[str, Any] = None):
        """开始记录会话"""
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = session_name
        self.session_config = config or {}
        self.metrics_history.clear()
        
        # 创建会话目录
        session_dir = os.path.join(self.log_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # 保存会话配置
        config_path = os.path.join(session_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.session_config, f, indent=2, ensure_ascii=False)
        
        print(f"开始记录会话: {session_name}")
    
    def end_session(self):
        """结束记录会话"""
        if self.current_session and self.metrics_history:
            self._save_session_data()
            print(f"会话结束: {self.current_session}, 记录了 {len(self.metrics_history)} 个数据点")
        
        self.current_session = None
        self.session_config = {}
    
    def log_step(self, step_data: Dict[str, Any]):
        """记录步骤数据"""
        if not self.current_session:
            return
        
        try:
            metrics = SimulationMetrics(
                step=step_data['step'],
                timestamp=step_data.get('time', 0.0),
                system_state=step_data['system_state'],
                performance_metrics=step_data['metrics'],
                role_actions=step_data['actions'],
                role_rewards=step_data['rewards'],
                decisions=list(step_data['decisions'].values()),
                crises=step_data.get('crises', [])
            )
            
            self.metrics_history.append(metrics)
            
            # 定期保存检查点
            if len(self.metrics_history) % 100 == 0:
                self._save_checkpoint()
                
        except Exception as e:
            print(f"记录步骤数据错误: {e}")
    
    def _save_session_data(self):
        """保存会话数据"""
        if not self.current_session or not self.metrics_history:
            return
        
        session_dir = os.path.join(self.log_dir, self.current_session)
        
        try:
            # 保存指标数据为JSON
            metrics_data = [asdict(metrics) for metrics in self.metrics_history]
            metrics_path = os.path.join(session_dir, "metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            # 保存为CSV格式（便于分析）
            self._save_as_csv(session_dir)
            
            # 保存摘要统计
            self._save_summary(session_dir)
            
            print(f"会话数据已保存到: {session_dir}")
            
        except Exception as e:
            print(f"保存会话数据错误: {e}")
    
    def _save_as_csv(self, session_dir: str):
        """保存为CSV格式"""
        if not self.metrics_history:
            return
        
        # 系统状态CSV
        system_data = []
        for metrics in self.metrics_history:
            row = {
                'step': metrics.step,
                'timestamp': metrics.timestamp,
                **metrics.system_state,
                **metrics.performance_metrics
            }
            system_data.append(row)
        
        system_df = pd.DataFrame(system_data)
        system_csv_path = os.path.join(session_dir, "system_metrics.csv")
        system_df.to_csv(system_csv_path, index=False)
        
        # 角色行动CSV
        action_data = []
        for metrics in self.metrics_history:
            for role, actions in metrics.role_actions.items():
                row = {'step': metrics.step, 'role': role}
                for i, action in enumerate(actions):
                    row[f'action_{i}'] = action
                row['reward'] = metrics.role_rewards.get(role, 0.0)
                action_data.append(row)
        
        if action_data:
            action_df = pd.DataFrame(action_data)
            action_csv_path = os.path.join(session_dir, "role_actions.csv")
            action_df.to_csv(action_csv_path, index=False)
        
        # 决策CSV
        decision_data = []
        for metrics in self.metrics_history:
            for decision in metrics.decisions:
                row = {
                    'step': metrics.step,
                    'proposer': decision.get('proposer', ''),
                    'approved': decision.get('approved', False),
                    'approval_ratio': decision.get('approval_ratio', 0.0)
                }
                decision_data.append(row)
        
        if decision_data:
            decision_df = pd.DataFrame(decision_data)
            decision_csv_path = os.path.join(session_dir, "decisions.csv")
            decision_df.to_csv(decision_csv_path, index=False)
    
    def _save_summary(self, session_dir: str):
        """保存摘要统计"""
        if not self.metrics_history:
            return
        
        summary = {
            'session_name': self.current_session,
            'total_steps': len(self.metrics_history),
            'start_time': self.metrics_history[0].timestamp,
            'end_time': self.metrics_history[-1].timestamp,
            'duration': self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp,
            'config': self.session_config
        }
        
        # 计算统计信息
        system_states = [metrics.system_state for metrics in self.metrics_history]
        performance_metrics = [metrics.performance_metrics for metrics in self.metrics_history]
        
        # 系统状态统计
        system_stats = {}
        for key in system_states[0].keys():
            values = [state[key] for state in system_states]
            system_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'final': values[-1]
            }
        
        # 性能指标统计
        performance_stats = {}
        for key in performance_metrics[0].keys():
            values = [metric[key] for metric in performance_metrics]
            performance_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        summary['system_statistics'] = system_stats
        summary['performance_statistics'] = performance_stats
        
        # 角色统计
        role_stats = {}
        all_roles = set()
        for metrics in self.metrics_history:
            all_roles.update(metrics.role_actions.keys())
        
        for role in all_roles:
            rewards = [metrics.role_rewards.get(role, 0.0) for metrics in self.metrics_history]
            role_stats[role] = {
                'mean_reward': np.mean(rewards),
                'total_reward': np.sum(rewards),
                'action_count': sum(1 for metrics in self.metrics_history if role in metrics.role_actions)
            }
        
        summary['role_statistics'] = role_stats
        
        # 决策统计
        all_decisions = []
        for metrics in self.metrics_history:
            all_decisions.extend(metrics.decisions)
        
        if all_decisions:
            approval_rate = len([d for d in all_decisions if d.get('approved', False)]) / len(all_decisions)
            summary['decision_statistics'] = {
                'total_decisions': len(all_decisions),
                'approval_rate': approval_rate,
                'average_approval_ratio': np.mean([d.get('approval_ratio', 0.0) for d in all_decisions])
            }
        
        # 保存摘要
        summary_path = os.path.join(session_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _save_checkpoint(self):
        """保存检查点"""
        if not self.current_session or len(self.metrics_history) < 100:
            return
        
        checkpoint_dir = os.path.join(self.log_dir, self.current_session, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_num = len(self.metrics_history) // 100
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num:04d}.json")
        
        # 只保存最近100个数据点作为检查点
        recent_metrics = self.metrics_history[-100:]
        checkpoint_data = [asdict(metrics) for metrics in recent_metrics]
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def load_session_data(self, session_name: str) -> List[SimulationMetrics]:
        """加载会话数据"""
        session_dir = os.path.join(self.log_dir, session_name)
        metrics_path = os.path.join(session_dir, "metrics.json")
        
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            metrics_history = []
            for data in metrics_data:
                metrics = SimulationMetrics(
                    step=data['step'],
                    timestamp=data['timestamp'],
                    system_state=data['system_state'],
                    performance_metrics=data['performance_metrics'],
                    role_actions=data['role_actions'],
                    role_rewards=data['role_rewards'],
                    decisions=data['decisions'],
                    crises=data['crises']
                )
                metrics_history.append(metrics)
            
            print(f"从会话 {session_name} 加载了 {len(metrics_history)} 个数据点")
            return metrics_history
            
        except Exception as e:
            print(f"加载会话数据错误: {e}")
            return []
    
    def get_recent_metrics(self, window_size: int = 100) -> List[SimulationMetrics]:
        """获取最近指标"""
        if len(self.metrics_history) <= window_size:
            return self.metrics_history
        return self.metrics_history[-window_size:]
    
    def export_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """导出为DataFrame"""
        if not self.metrics_history:
            return {}
        
        # 系统指标DataFrame
        system_data = []
        for metrics in self.metrics_history:
            row = {
                'step': metrics.step,
                'timestamp': metrics.timestamp,
                **metrics.system_state,
                **metrics.performance_metrics
            }
            system_data.append(row)
        
        system_df = pd.DataFrame(system_data)
        
        # 角色行动DataFrame
        action_data = []
        for metrics in self.metrics_history:
            for role, actions in metrics.role_actions.items():
                row = {'step': metrics.step, 'role': role}
                for i, action in enumerate(actions):
                    row[f'action_{i}'] = action
                row['reward'] = metrics.role_rewards.get(role, 0.0)
                action_data.append(row)
        
        action_df = pd.DataFrame(action_data) if action_data else pd.DataFrame()
        
        return {
            'system_metrics': system_df,
            'role_actions': action_df
        }