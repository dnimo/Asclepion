import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DashboardConfig:
    """仪表板配置"""
    update_interval: float = 1.0  # 更新间隔（秒）
    history_window: int = 200     # 历史窗口大小
    enable_plots: bool = True
    enable_metrics: bool = True
    enable_alerts: bool = True

class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, simulator, data_logger, config: DashboardConfig = None):
        self.simulator = simulator
        self.data_logger = data_logger
        self.config = config or DashboardConfig()
        
        self.is_monitoring = False
        self.last_update_time = 0
        self.alert_history: List[Dict[str, Any]] = []
        
        # 性能阈值
        self.performance_thresholds = {
            'stability': 0.6,
            'resource_efficiency': 0.5,
            'overall_performance': 0.6,
            'TTC': 0.4  # 共识时间阈值（越低越好）
        }
    
    def start_monitoring(self):
        """开始监控"""
        self.is_monitoring = True
        print("开始实时监控...")
        
        # 创建仪表板
        self.dashboard = Dashboard(self.simulator, self.data_logger, self.config)
        self.dashboard.show()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if hasattr(self, 'dashboard'):
            self.dashboard.close()
        print("停止实时监控")
    
    def update(self):
        """更新监控数据"""
        if not self.is_monitoring:
            return
        
        current_time = time.time()
        if current_time - self.last_update_time < self.config.update_interval:
            return
        
        self.last_update_time = current_time
        
        # 检查性能警报
        self._check_performance_alerts()
        
        # 更新仪表板
        if hasattr(self, 'dashboard'):
            self.dashboard.update()
    
    def _check_performance_alerts(self):
        """检查性能警报"""
        if not self.data_logger.metrics_history:
            return
        
        recent_metrics = self.data_logger.get_recent_metrics(10)
        if not recent_metrics:
            return
        
        # 检查最近的性能指标
        for metrics in recent_metrics[-5:]:  # 检查最近5个数据点
            for metric_name, threshold in self.performance_thresholds.items():
                metric_value = metrics.performance_metrics.get(metric_name, 1.0)
                
                # 对于TTC，值越低越好
                if metric_name == 'TTC':
                    if metric_value > threshold:
                        self._trigger_alert(metric_name, metric_value, threshold)
                else:
                    # 对于其他指标，值越高越好
                    if metric_value < threshold:
                        self._trigger_alert(metric_name, metric_value, threshold)
    
    def _trigger_alert(self, metric_name: str, current_value: float, threshold: float):
        """触发警报"""
        alert = {
            'timestamp': datetime.now(),
            'metric': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'severity': 'high' if abs(current_value - threshold) > 0.2 else 'medium'
        }
        
        self.alert_history.append(alert)
        
        # 打印警报
        print(f"🚨 警报: {metric_name} = {current_value:.3f} (阈值: {threshold:.3f})")
        
        # 这里可以添加其他警报机制，如声音、邮件等
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.data_logger.metrics_history:
            return {}
        
        recent_metrics = self.data_logger.get_recent_metrics(50)
        
        # 计算趋势
        trends = {}
        for metric in ['stability', 'overall_performance', 'resource_efficiency']:
            values = [m.performance_metrics.get(metric, 0.0) for m in recent_metrics]
            if len(values) >= 2:
                trend = values[-1] - values[0]
                trends[metric] = '上升' if trend > 0.05 else '下降' if trend < -0.05 else '稳定'
            else:
                trends[metric] = '未知'
        
        return {
            'total_steps': len(self.data_logger.metrics_history),
            'active_alerts': len([a for a in self.alert_history[-10:] if a['severity'] == 'high']),
            'system_stability': recent_metrics[-1].performance_metrics.get('stability', 0.0),
            'performance_trends': trends,
            'last_update': datetime.now()
        }

class Dashboard:
    """实时仪表板"""
    
    def __init__(self, simulator, data_logger, config: DashboardConfig):
        self.simulator = simulator
        self.data_logger = data_logger
        self.config = config
        
        self.fig = None
        self.axes = {}
        self.lines = {}
        
        self._setup_dashboard()
    
    def _setup_dashboard(self):
        """设置仪表板"""
        if not self.config.enable_plots:
            return
        
        # 创建图形和子图
        self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Kallipolis医疗共和国 - 实时监控仪表板', fontsize=16)
        
        # 系统状态图
        self.axes['system_state'] = ax1
        ax1.set_title('系统状态指标')
        ax1.set_ylabel('数值')
        ax1.set_ylim(0, 1)
        ax1.grid(True)
        
        # 性能指标图
        self.axes['performance'] = ax2
        ax2.set_title('性能指标')
        ax2.set_ylabel('数值')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        
        # 角色奖励图
        self.axes['rewards'] = ax3
        ax3.set_title('角色奖励')
        ax3.set_ylabel('奖励')
        ax3.grid(True)
        
        # 决策统计图
        self.axes['decisions'] = ax4
        ax4.set_title('决策统计')
        ax4.set_ylabel('数量')
        ax4.grid(True)
        
        # 初始化线条
        self._initialize_plots()
        
        plt.tight_layout()
    
    def _initialize_plots(self):
        """初始化绘图线条"""
        # 系统状态线条
        system_metrics = ['resource_utilization', 'financial_health', 
                         'patient_satisfaction', 'medical_quality']
        colors = ['blue', 'green', 'red', 'orange']
        
        for metric, color in zip(system_metrics, colors):
            line, = self.axes['system_state'].plot([], [], label=metric, color=color)
            self.lines[metric] = line
        
        self.axes['system_state'].legend()
        
        # 性能指标线条
        performance_metrics = ['stability', 'overall_performance', 'resource_efficiency']
        colors = ['purple', 'brown', 'pink']
        
        for metric, color in zip(performance_metrics, colors):
            line, = self.axes['performance'].plot([], [], label=metric, color=color)
            self.lines[metric] = line
        
        self.axes['performance'].legend()
        
        # 角色奖励线条
        roles = ['senior_doctor', 'junior_doctor', 'accountant', 'patient_rep']
        colors = ['blue', 'green', 'red', 'orange']
        
        for role, color in zip(roles, colors):
            line, = self.axes['rewards'].plot([], [], label=role, color=color)
            self.lines[f'reward_{role}'] = line
        
        self.axes['rewards'].legend()
    
    def show(self):
        """显示仪表板"""
        if self.fig:
            plt.ion()  # 交互模式
            plt.show()
    
    def close(self):
        """关闭仪表板"""
        if self.fig:
            plt.close(self.fig)
            plt.ioff()  # 关闭交互模式
    
    def update(self):
        """更新仪表板"""
        if not self.fig or not self.data_logger.metrics_history:
            return
        
        recent_metrics = self.data_logger.get_recent_metrics(self.config.history_window)
        if len(recent_metrics) < 2:
            return
        
        steps = [m.step for m in recent_metrics]
        
        # 更新系统状态图
        system_metrics = ['resource_utilization', 'financial_health', 
                         'patient_satisfaction', 'medical_quality']
        
        for metric in system_metrics:
            values = [m.system_state[metric] for m in recent_metrics]
            self.lines[metric].set_data(steps, values)
        
        self.axes['system_state'].relim()
        self.axes['system_state'].autoscale_view()
        self.axes['system_state'].set_xlim(steps[0], steps[-1])
        
        # 更新性能指标图
        performance_metrics = ['stability', 'overall_performance', 'resource_efficiency']
        
        for metric in performance_metrics:
            values = [m.performance_metrics.get(metric, 0.0) for m in recent_metrics]
            self.lines[metric].set_data(steps, values)
        
        self.axes['performance'].relim()
        self.axes['performance'].autoscale_view()
        self.axes['performance'].set_xlim(steps[0], steps[-1])
        
        # 更新角色奖励图
        roles = ['senior_doctor', 'junior_doctor', 'accountant', 'patient_rep']
        
        for role in roles:
            values = []
            for metrics in recent_metrics:
                reward = metrics.role_rewards.get(role, 0.0)
                values.append(reward)
            
            self.lines[f'reward_{role}'].set_data(steps, values)
        
        self.axes['rewards'].relim()
        self.axes['rewards'].autoscale_view()
        self.axes['rewards'].set_xlim(steps[0], steps[-1])
        
        # 更新决策统计图
        self.axes['decisions'].clear()
        self.axes['decisions'].set_title('决策统计')
        self.axes['decisions'].set_ylabel('数量')
        
        # 计算决策统计
        approved_decisions = []
        rejected_decisions = []
        
        window_size = 20
        for i in range(0, len(recent_metrics), window_size):
            window_metrics = recent_metrics[i:i+window_size]
            approved = 0
            rejected = 0
            
            for metrics in window_metrics:
                for decision in metrics.decisions:
                    if decision.get('approved', False):
                        approved += 1
                    else:
                        rejected += 1
            
            approved_decisions.append(approved)
            rejected_decisions.append(rejected)
        
        x_pos = np.arange(len(approved_decisions))
        self.axes['decisions'].bar(x_pos - 0.2, approved_decisions, 0.4, label='通过', color='green')
        self.axes['decisions'].bar(x_pos + 0.2, rejected_decisions, 0.4, label='拒绝', color='red')
        self.axes['decisions'].legend()
        
        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save_snapshot(self, filename: str = None):
        """保存仪表板快照"""
        if not self.fig:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_snapshot_{timestamp}.png"
        
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"仪表板快照已保存: {filename}")

# 使用示例
def create_monitoring_system(simulator, data_logger):
    """创建监控系统"""
    config = DashboardConfig(
        update_interval=2.0,
        history_window=100,
        enable_plots=True,
        enable_metrics=True,
        enable_alerts=True
    )
    
    monitor = RealTimeMonitor(simulator, data_logger, config)
    return monitor