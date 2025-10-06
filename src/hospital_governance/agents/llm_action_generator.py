import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json
import asyncio
from abc import ABC, abstractmethod

@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    context_window: int = 8000
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    use_async: bool = True
    timeout: float = 30.0

class BaseLLMProvider(ABC):
    """LLM提供者基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    def generate_text_sync(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """同步生成文本"""
        pass

class MockLLMProvider(BaseLLMProvider):
    """模拟LLM提供者（用于测试和开发）"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.role_templates = {
            'doctors': "基于医疗质量和患者安全考虑...",
            'interns': "基于学习需求和职业发展考虑...",
            'patients': "基于患者权益和服务质量考虑...",
            'accountants': "基于财务健康和成本效益考虑...",
            'government': "基于系统稳定性和合规性考虑..."
        }
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None) -> str:
        # 模拟异步生成
        await asyncio.sleep(0.1)
        return self.generate_text_sync(prompt, context)
    
    def generate_text_sync(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """模拟文本生成"""
        role = context.get('role', 'unknown') if context else 'unknown'
        template = self.role_templates.get(role, "基于当前情况考虑...")
        
        # 简单的决策逻辑
        if "质量" in prompt:
            return f"{template}建议提高医疗质量标准。"
        elif "成本" in prompt:
            return f"{template}建议优化成本控制措施。"
        elif "资源" in prompt:
            return f"{template}建议合理分配资源。"
        else:
            return f"{template}建议保持现状并观察。"

class LLMActionGenerator:
    """LLM行动生成器"""
    
    def __init__(self, config: LLMConfig = None, provider: BaseLLMProvider = None):
        self.config = config or LLMConfig()
        self.provider = provider or MockLLMProvider(self.config)
        
        # 上下文管理
        self.context_history: List[Dict] = []
        self.role_contexts: Dict[str, List[str]] = {
            'doctors': [],
            'interns': [], 
            'patients': [],
            'accountants': [],
            'government': []
        }
        
        # 行动模板
        self.action_templates = self._initialize_action_templates()
        
        # 性能统计
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'average_response_time': 0.0,
            'error_count': 0
        }
    
    def _initialize_action_templates(self) -> Dict[str, Dict]:
        """初始化行动模板"""
        return {
            'doctors': {
                'medical_quality': "作为医生，考虑当前医疗质量指标{quality}，患者安全指标{safety}，建议采取的医疗质量改进行动：",
                'resource_allocation': "基于资源adequacy指标{resources}，建议的资源分配策略：",
                'crisis_response': "面对{crisis_type}危机，严重程度{severity}，建议的应急响应措施："
            },
            'interns': {
                'education_request': "作为实习医生，当前教育质量{education}，培训时间{training_hours}小时，建议的学习需求：",
                'workload_adjustment': "当前工作负荷{workload}，建议的工作负荷调整：",
                'career_development': "职业发展机会评分{career}，建议的发展计划："
            },
            'patients': {
                'satisfaction_improvement': "患者满意度{satisfaction}，护理质量{care_quality}，建议的改进措施：",
                'accessibility_request': "医疗可及性{accessibility}，等待时间{waiting_time}，建议的改善方案：",
                'safety_concern': "安全指数{safety}，建议的安全改进措施："
            },
            'accountants': {
                'financial_optimization': "财务健康度{financial_health}，利润率{profit_margin}，建议的财务优化策略：",
                'cost_control': "运营效率{efficiency}，成本控制{cost_control}，建议的成本管理措施：",
                'budget_allocation': "预算分配{budget}，建议的资源优化方案："
            },
            'government': {
                'regulatory_action': "系统稳定性{stability}，合规性{compliance}，建议的监管措施：",
                'policy_adjustment': "公众信任{trust}，透明度{transparency}，建议的政策调整：",
                'crisis_coordination': "系统性危机{crisis}，建议的协调措施："
            }
        }
    
    async def generate_action_async(self, role: str, observation: np.ndarray, 
                                  holy_code_state: Dict, context: Dict[str, Any]) -> np.ndarray:
        """异步生成行动"""
        try:
            prompt = self._build_action_prompt(role, observation, holy_code_state, context)
            
            generation_context = {
                'role': role,
                'observation_dim': len(observation),
                'context_type': context.get('context_type', 'normal')
            }
            
            response = await self.provider.generate_text(prompt, generation_context)
            action = self._parse_action_response(response, role)
            
            self._update_context_history(role, prompt, response, action)
            self.generation_stats['successful_generations'] += 1
            
            return action
            
        except Exception as e:
            self.generation_stats['error_count'] += 1
            # 返回默认行动
            return self._get_default_action(role, observation)
        finally:
            self.generation_stats['total_generations'] += 1
    
    def generate_action_sync(self, role: str, observation: np.ndarray,
                           holy_code_state: Dict, context: Dict[str, Any]) -> np.ndarray:
        """同步生成行动"""
        try:
            prompt = self._build_action_prompt(role, observation, holy_code_state, context)
            
            generation_context = {
                'role': role,
                'observation_dim': len(observation),
                'context_type': context.get('context_type', 'normal')
            }
            
            response = self.provider.generate_text_sync(prompt, generation_context)
            action = self._parse_action_response(response, role)
            
            self._update_context_history(role, prompt, response, action)
            self.generation_stats['successful_generations'] += 1
            
            return action
            
        except Exception as e:
            self.generation_stats['error_count'] += 1
            # 返回默认行动
            return self._get_default_action(role, observation)
        finally:
            self.generation_stats['total_generations'] += 1
    
    def _build_action_prompt(self, role: str, observation: np.ndarray,
                           holy_code_state: Dict, context: Dict[str, Any]) -> str:
        """构建行动生成提示"""
        # 基础观测信息
        obs_str = ", ".join([f"{val:.2f}" for val in observation])
        
        # 角色特定模板
        templates = self.action_templates.get(role, {})
        context_type = context.get('context_type', 'normal')
        
        # 选择合适的模板
        if context_type == 'crisis' and 'crisis_response' in templates:
            template = templates['crisis_response']
            crisis_info = context.get('crisis_info', {})
            prompt = template.format(
                crisis_type=crisis_info.get('type', 'unknown'),
                severity=crisis_info.get('severity', 0.5)
            )
        elif 'medical_quality' in templates and role == 'doctors':
            template = templates['medical_quality']
            prompt = template.format(
                quality=observation[0] if len(observation) > 0 else 0.5,
                safety=observation[1] if len(observation) > 1 else 0.5
            )
        elif 'education_request' in templates and role == 'interns':
            template = templates['education_request']
            prompt = template.format(
                education=observation[0] if len(observation) > 0 else 0.5,
                training_hours=context.get('training_hours', 40)
            )
        else:
            # 通用模板
            prompt = f"作为{role}角色，基于当前观测{obs_str}，考虑神圣法典规则，建议采取的行动："
        
        # 添加神圣法典上下文
        if holy_code_state:
            rules_summary = self._summarize_holy_code_rules(holy_code_state)
            prompt += f"\\n\\n当前有效规则：{rules_summary}"
        
        # 添加历史上下文
        recent_context = self._get_recent_context(role, limit=3)
        if recent_context:
            prompt += f"\\n\\n最近的决策上下文：{recent_context}"
        
        prompt += "\\n\\n请返回一个具体的数值行动向量（例如：[0.5, -0.2, 0.8, 0.1]），每个数值在-1到1之间。"
        
        return prompt
    
    def _parse_action_response(self, response: str, role: str) -> np.ndarray:
        """解析LLM响应为行动向量"""
        try:
            # 尝试提取数值向量
            import re
            
            # 查找方括号中的数值
            vector_match = re.search(r'\\[([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?(?:,\\s*[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)*)\\]', response)
            
            if vector_match:
                vector_str = vector_match.group(1)
                values = [float(x.strip()) for x in vector_str.split(',')]
                action = np.array(values)
                
                # 确保值在有效范围内
                action = np.clip(action, -1.0, 1.0)
                
                # 确保正确的维度
                expected_dim = self._get_action_dimension(role)
                if len(action) != expected_dim:
                    if len(action) > expected_dim:
                        action = action[:expected_dim]
                    else:
                        action = np.pad(action, (0, expected_dim - len(action)), 'constant', constant_values=0)
                
                return action
            
            else:
                # 如果无法解析，尝试从文本中推断行动
                return self._infer_action_from_text(response, role)
                
        except Exception as e:
            # 解析失败，返回默认行动
            return self._get_default_action(role, np.zeros(8))
    
    def _infer_action_from_text(self, text: str, role: str) -> np.ndarray:
        """从文本中推断行动"""
        action_dim = self._get_action_dimension(role)
        action = np.zeros(action_dim)
        
        # 基于关键词推断行动强度
        text_lower = text.lower()
        
        if role == 'doctors':
            if '提高' in text or '增加' in text or '改善' in text:
                action[0] = 0.6  # 质量改进
            if '资源' in text and ('申请' in text or '需要' in text):
                action[1] = 0.5  # 资源请求
            if '紧急' in text or '危机' in text:
                action[2] = 0.8  # 危机响应
                
        elif role == 'interns':
            if '学习' in text or '培训' in text:
                action[0] = 0.7  # 培训请求
            if '工作' in text and ('减少' in text or '调整' in text):
                action[1] = 0.4  # 工作负荷调整
                
        elif role == 'accountants':
            if '控制' in text or '节约' in text:
                action[0] = 0.6  # 成本控制
            if '优化' in text or '效率' in text:
                action[1] = 0.5  # 效率提升
                
        elif role == 'patients':
            if '改善' in text or '提高' in text:
                action[0] = 0.6  # 满意度改进
            if '等待' in text or '时间' in text:
                action[1] = 0.5  # 等待时间优化
                
        elif role == 'government':
            if '监管' in text or '规范' in text:
                action[0] = 0.7  # 监管加强
            if '透明' in text or '公开' in text:
                action[1] = 0.5  # 透明度提升
        
        return action
    
    def _get_action_dimension(self, role: str) -> int:
        """获取角色的行动维度"""
        role_dimensions = {
            'doctors': 4,
            'interns': 3,
            'patients': 3,
            'accountants': 3,
            'government': 3
        }
        return role_dimensions.get(role, 4)
    
    def _get_default_action(self, role: str, observation: np.ndarray) -> np.ndarray:
        """获取默认行动"""
        action_dim = self._get_action_dimension(role)
        
        # 基于观测的简单默认策略
        if len(observation) > 0:
            # 基于观测值的加权平均
            base_value = np.mean(observation) - 0.5  # 中心化到[-0.5, 0.5]
            action = np.full(action_dim, base_value * 0.5)  # 保守的行动
        else:
            action = np.zeros(action_dim)
        
        return np.clip(action, -1.0, 1.0)
    
    def _summarize_holy_code_rules(self, holy_code_state: Dict) -> str:
        """总结神圣法典规则"""
        if not holy_code_state:
            return "无特殊规则"
        
        rules = holy_code_state.get('active_rules', [])
        if not rules:
            return "无激活规则"
        
        # 简化的规则总结
        rule_summaries = []
        for rule in rules[:3]:  # 只取前3个规则
            rule_summaries.append(f"规则{rule.get('id', 'unknown')}: {rule.get('description', 'unknown')[:50]}...")
        
        return "; ".join(rule_summaries)
    
    def _get_recent_context(self, role: str, limit: int = 3) -> str:
        """获取最近的上下文"""
        role_history = self.role_contexts.get(role, [])
        if not role_history:
            return ""
        
        recent = role_history[-limit:]
        return "; ".join(recent)
    
    def _update_context_history(self, role: str, prompt: str, response: str, action: np.ndarray):
        """更新上下文历史"""
        context_entry = f"行动{len(self.role_contexts[role])}: {action.tolist()}"
        
        self.role_contexts[role].append(context_entry)
        
        # 限制历史长度
        if len(self.role_contexts[role]) > 10:
            self.role_contexts[role].pop(0)
        
        # 全局历史
        self.context_history.append({
            'role': role,
            'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
            'response': response[:200] + "..." if len(response) > 200 else response,
            'action': action.tolist(),
            'timestamp': len(self.context_history)
        })
        
        # 限制全局历史长度
        if len(self.context_history) > 100:
            self.context_history.pop(0)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        success_rate = (self.generation_stats['successful_generations'] / 
                       max(1, self.generation_stats['total_generations']))
        
        return {
            'total_generations': self.generation_stats['total_generations'],
            'success_rate': success_rate,
            'error_rate': self.generation_stats['error_count'] / max(1, self.generation_stats['total_generations']),
            'average_response_time': self.generation_stats['average_response_time']
        }
    
    def clear_context(self, role: str = None):
        """清除上下文历史"""
        if role:
            if role in self.role_contexts:
                self.role_contexts[role].clear()
        else:
            for role_history in self.role_contexts.values():
                role_history.clear()
            self.context_history.clear()
    
    def export_context_history(self, filepath: str):
        """导出上下文历史"""
        import json
        export_data = {
            'generation_stats': self.generation_stats,
            'role_contexts': self.role_contexts,
            'context_history': self.context_history[-50:]  # 只导出最近50条
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def generate_proposal_sync(self, role: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """同步生成议会提案"""
        try:
            # 构建提案生成提示
            prompt = self._build_proposal_prompt(role, context)
            
            # 生成提案
            if self.config.use_async:
                try:
                    response = asyncio.run(self.provider.generate_text(prompt, context))
                except:
                    response = self.mock_provider.generate_text_sync(prompt, context)
            else:
                response = self.mock_provider.generate_text_sync(prompt, context)
            
            # 解析提案响应
            proposal = self._parse_proposal_response(response, role)
            
            # 更新统计
            self.generation_stats['total_generations'] += 1
            self.generation_stats['successful_generations'] += 1
            
            return proposal
            
        except Exception as e:
            self.generation_stats['total_generations'] += 1
            self.generation_stats['error_count'] += 1
            
            # 返回默认提案
            return self._get_default_proposal(role)
    
    def _build_proposal_prompt(self, role: str, context: Dict[str, Any]) -> str:
        """构建提案生成提示"""
        system_state = context.get('system_state', {})
        current_step = context.get('current_step', 0)
        agent_performance = context.get('agent_performance', 0.5)
        parliament_history = context.get('parliament_history', [])
        
        # 基础提示
        prompt = f"""作为{role}角色，你需要为医院治理议会提出一项具体的提案。

当前系统状态：
- 医疗质量指标: {system_state.get('medical_quality', 'N/A')}
- 患者满意度: {system_state.get('patient_satisfaction', 'N/A')}
- 运营效率: {system_state.get('operational_efficiency', 'N/A')}
- 当前步骤: {current_step}
- 角色表现: {agent_performance}

请基于你的角色职责和当前状况，提出一个具体的改进提案。"""

        # 添加历史上下文
        if parliament_history:
            prompt += f"\n\n最近的议会决议历史:\n"
            for i, meeting in enumerate(parliament_history[-3:]):
                proposals = meeting.get('proposals', {})
                prompt += f"第{i+1}次会议提案数: {len(proposals)}\n"
        
        # 角色特定的提案指导
        role_guidance = {
            'doctors': '请专注于医疗质量、患者安全和医疗服务改进',
            'interns': '请专注于培训质量、学习资源和职业发展',
            'patients': '请专注于患者体验、等待时间和医疗费用',
            'accountants': '请专注于成本控制、财务透明度和预算优化',
            'government': '请专注于监管合规、公共利益和医疗公平'
        }
        
        guidance = role_guidance.get(role, '请提出有助于整体医院治理的建议')
        prompt += f"\n\n{guidance}"
        
        prompt += """

请按以下格式返回你的提案：
{
  "proposal": "具体的提案内容",
  "priority": 0.8,
  "benefit": 0.7,
  "cost": 0.3,
  "reasoning": "提案的理由说明"
}

其中priority、benefit、cost都是0-1之间的数值。"""
        
        return prompt
    
    def _parse_proposal_response(self, response: str, role: str) -> Dict[str, Any]:
        """解析提案响应"""
        try:
            import json
            import re
            
            # 尝试提取JSON
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                proposal_data = json.loads(json_str)
                
                return {
                    'agent_id': role,
                    'proposal_text': proposal_data.get('proposal', '维持现状'),
                    'priority': float(proposal_data.get('priority', 0.5)),
                    'expected_benefit': float(proposal_data.get('benefit', 0.5)),
                    'implementation_cost': float(proposal_data.get('cost', 0.3)),
                    'reasoning': proposal_data.get('reasoning', 'LLM生成提案'),
                    'generation_method': 'LLM'
                }
            else:
                # 如果无法解析JSON，尝试提取文本
                return {
                    'agent_id': role,
                    'proposal_text': response[:200] + "..." if len(response) > 200 else response,
                    'priority': 0.5,
                    'expected_benefit': 0.5,
                    'implementation_cost': 0.3,
                    'reasoning': '基于LLM文本生成',
                    'generation_method': 'LLM_fallback'
                }
                
        except Exception as e:
            return self._get_default_proposal(role)
    
    def _get_default_proposal(self, role: str) -> Dict[str, Any]:
        """获取默认提案"""
        default_proposals = {
            'doctors': '提高医疗服务质量标准',
            'interns': '加强实习生培训计划',
            'patients': '改善患者就诊体验',
            'accountants': '优化医院财务管理',
            'government': '加强医疗监管措施'
        }
        
        return {
            'agent_id': role,
            'proposal_text': default_proposals.get(role, '维持现状'),
            'priority': 0.5,
            'expected_benefit': 0.5,
            'implementation_cost': 0.3,
            'reasoning': '默认模板提案',
            'generation_method': 'template'
        }
