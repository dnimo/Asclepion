#!/usr/bin/env python3
"""
医院治理系统 - 真实LLM API集成版本
支持OpenAI GPT、Anthropic Claude等真实LLM API
"""

import numpy as np
import asyncio
import json
import time
import os
from typing import Dict, List, Any, Optional

class RealLLMProvider:
    """真实LLM API提供者"""
    
    def __init__(self, provider_type: str = 'openai', api_key: str = None, model: str = None):
        self.provider_type = provider_type
        self.api_key = api_key or os.getenv(f'{provider_type.upper()}_API_KEY')
        
        if provider_type == 'openai':
            self.model = model or 'gpt-4'
            self.base_url = "https://api.openai.com/v1"
        elif provider_type == 'anthropic':
            self.model = model or 'claude-3-sonnet-20240229'
            self.base_url = "https://api.anthropic.com/v1"
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")
        
        self.request_count = 0
        self.total_tokens = 0
    
    async def generate_decision(self, role: str, observation: np.ndarray, 
                              constraints: Dict, context: str = "normal") -> np.ndarray:
        """使用真实LLM生成决策"""
        try:
            # 构建提示
            prompt = self._build_decision_prompt(role, observation, constraints, context)
            
            # 调用LLM API
            response = await self._call_llm_api(prompt)
            
            # 解析响应为数值向量
            action = self._parse_response_to_action(response, role)
            
            self.request_count += 1
            print(f"[LLM] {role} 决策: {action.tolist()[:3]}... (来源: {self.provider_type})")
            
            return action
            
        except Exception as e:
            print(f"[LLM错误] {role}: {e}")
            # 回退到简单决策
            return self._fallback_decision(role, observation, constraints)
    
    def _build_decision_prompt(self, role: str, observation: np.ndarray, 
                             constraints: Dict, context: str) -> str:
        """构建决策提示"""
        # 观测数据概述
        obs_summary = f"观测数据: {observation[:5].tolist()}..." if len(observation) > 5 else f"观测数据: {observation.tolist()}"
        
        # 约束概述
        constraints_text = ", ".join([f"{k}={v}" for k, v in constraints.items()]) if constraints else "无特殊约束"
        
        # 角色特定提示
        role_prompts = {
            'doctors': f"""
你是医院的主治医生团队代表。基于当前医疗系统状态，你需要做出关键决策。

{obs_summary}
当前约束: {constraints_text}
情境: {context}

请基于以下考虑因素做出决策:
1. 患者生命安全和医疗质量（最高优先级）
2. 医疗资源合理分配
3. 工作负荷平衡
4. 应急响应能力

请返回一个4维决策向量 [质量改进力度, 资源申请强度, 工作负荷调整, 安全措施强度]
每个值在-1.0到1.0之间，例如: [0.6, 0.4, -0.2, 0.8]
""",
            'interns': f"""
你是实习医生群体的代表。基于当前学习和工作环境，你需要表达需求和建议。

{obs_summary}
当前约束: {constraints_text}
情境: {context}

请基于以下考虑因素做出决策:
1. 教育培训质量和机会
2. 工作负荷的合理性
3. 职业发展路径
4. 学习资源获取

请返回一个3维决策向量 [培训需求强度, 工作负荷调整, 发展计划优先级]
每个值在-1.0到1.0之间，例如: [0.7, -0.3, 0.5]
""",
            'patients': f"""
你是患者群体的代表。基于当前医疗服务体验，你需要表达关切和需求。

{obs_summary}
当前约束: {constraints_text}
情境: {context}

请基于以下考虑因素做出决策:
1. 医疗服务质量和满意度
2. 医疗可及性和等待时间
3. 患者安全和权益保护
4. 医疗费用合理性

请返回一个3维决策向量 [服务改善需求, 可及性优化, 安全关注度]
每个值在-1.0到1.0之间，例如: [0.5, 0.7, 0.4]
""",
            'accountants': f"""
你是医院财务团队的代表。基于当前财务状况，你需要提出财务管理建议。

{obs_summary}
当前约束: {constraints_text}
情境: {context}

请基于以下考虑因素做出决策:
1. 成本控制和财务健康
2. 运营效率优化
3. 预算分配合理性
4. 财务透明度和合规性

请返回一个3维决策向量 [成本控制力度, 效率提升优先级, 预算优化强度]
每个值在-1.0到1.0之间，例如: [0.6, 0.4, 0.3]
""",
            'government': f"""
你是政府监管部门的代表。基于当前医院系统状态，你需要制定监管和政策措施。

{obs_summary}
当前约束: {constraints_text}
情境: {context}

请基于以下考虑因素做出决策:
1. 系统稳定性和公众安全
2. 监管合规和政策执行
3. 公平性和透明度
4. 应急协调和危机管理

请返回一个3维决策向量 [监管介入强度, 政策调整力度, 协调措施优先级]
每个值在-1.0到1.0之间，例如: [0.4, 0.2, 0.6]
"""
        }
        
        base_prompt = role_prompts.get(role, "请做出合理决策并返回数值向量。")
        
        return base_prompt + "\\n\\n重要：请直接返回数值向量，格式如 [0.6, 0.4, -0.2, 0.8]"
    
    async def _call_llm_api(self, prompt: str) -> str:
        """调用LLM API"""
        if self.provider_type == 'openai':
            return await self._call_openai_api(prompt)
        elif self.provider_type == 'anthropic':
            return await self._call_anthropic_api(prompt)
    
    async def _call_openai_api(self, prompt: str) -> str:
        """调用OpenAI API"""
        import httpx
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是医院治理系统的智能决策助手。请基于给定信息做出数值化决策。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    async def _call_anthropic_api(self, prompt: str) -> str:
        """调用Anthropic API"""
        import httpx
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 200,
            "temperature": 0.7,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["content"][0]["text"]
    
    def _parse_response_to_action(self, response: str, role: str) -> np.ndarray:
        """解析LLM响应为行动向量"""
        try:
            import re
            
            # 查找数值向量模式
            patterns = [
                r'\\[([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?(?:,\\s*[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)*)\\]',
                r'([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?(?:,\\s*[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)*)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    vector_str = match.group(1)
                    values = [float(x.strip()) for x in vector_str.split(',')]
                    action = np.array(values)
                    
                    # 约束到[-1, 1]范围
                    action = np.clip(action, -1.0, 1.0)
                    
                    # 确保正确维度
                    expected_dims = {'doctors': 4, 'interns': 3, 'patients': 3, 'accountants': 3, 'government': 3}
                    expected_dim = expected_dims.get(role, 4)
                    
                    if len(action) != expected_dim:
                        if len(action) > expected_dim:
                            action = action[:expected_dim]
                        else:
                            padded_action = np.zeros(expected_dim)
                            padded_action[:len(action)] = action
                            action = padded_action
                    
                    return action
            
            # 如果无法解析，使用文本推理
            return self._infer_from_text(response, role)
            
        except Exception as e:
            print(f"[解析错误] {role}: {e}")
            return self._fallback_decision(role, np.zeros(8), {})
    
    def _infer_from_text(self, text: str, role: str) -> np.ndarray:
        """从文本中推断决策意图"""
        text_lower = text.lower()
        
        # 基于关键词推断决策强度
        action_dims = {'doctors': 4, 'interns': 3, 'patients': 3, 'accountants': 3, 'government': 3}
        action = np.zeros(action_dims.get(role, 4))
        
        # 通用关键词映射
        if '紧急' in text or '危机' in text or '严重' in text:
            action[0] = 0.8
        elif '提高' in text or '改善' in text or '增强' in text:
            action[0] = 0.6
        elif '维持' in text or '保持' in text:
            action[0] = 0.2
        elif '减少' in text or '降低' in text:
            action[0] = -0.3
        
        if len(action) > 1:
            if '资源' in text or '申请' in text:
                action[1] = 0.5
            if '调整' in text or '优化' in text:
                action[1] = 0.4
        
        if len(action) > 2:
            if '安全' in text or '质量' in text:
                action[2] = 0.6
        
        return action
    
    def _fallback_decision(self, role: str, observation: np.ndarray, constraints: Dict) -> np.ndarray:
        """回退决策（当LLM调用失败时）"""
        action_dims = {'doctors': 4, 'interns': 3, 'patients': 3, 'accountants': 3, 'government': 3}
        dim = action_dims.get(role, 4)
        
        # 基于观测的简单决策
        if len(observation) > 0:
            avg_obs = np.mean(observation)
            if avg_obs < 0.3:
                return np.full(dim, 0.6)  # 积极干预
            elif avg_obs > 0.7:
                return np.full(dim, -0.2)  # 适度调整
            else:
                return np.full(dim, 0.1)  # 轻微调整
        else:
            return np.zeros(dim)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取API使用统计"""
        return {
            'provider': self.provider_type,
            'model': self.model,
            'request_count': self.request_count,
            'total_tokens': self.total_tokens
        }

class AdvancedHospitalSimulation:
    """高级医院仿真系统 - 集成真实LLM"""
    
    def __init__(self, llm_provider: str = 'mock', api_key: str = None, duration: int = 20):
        self.duration = duration
        self.llm_provider_type = llm_provider
        
        # 初始化LLM提供者
        if llm_provider == 'mock':
            # 使用简化版本的模拟决策
            self.use_real_llm = False
            print("🤖 使用模拟LLM决策")
        else:
            self.llm_provider = RealLLMProvider(llm_provider, api_key)
            self.use_real_llm = True
            print(f"🧠 使用真实LLM: {llm_provider}")
        
        # 系统初始化
        self.state = np.random.rand(16) * 0.5
        self.time_history = []
        self.state_history = []
        self.control_history = []
        self.rule_history = []
        self.llm_decision_history = []
        
        # 系统动力学
        self.A = np.eye(16) + np.random.randn(16, 16) * 0.03
        self.B = np.random.randn(16, 17) * 0.1
        
        print(f"📊 系统初始化: {len(self.state)}D状态, 仿真{duration}步")
    
    async def run_simulation_async(self):
        """异步运行仿真（支持真实LLM API调用）"""
        print("\\n🚀 开始异步仿真...")
        start_time = time.time()
        
        for step in range(self.duration):
            current_time = step * 0.1
            self.time_history.append(current_time)
            
            # 评估规则
            holy_code_state = self._evaluate_rules(self.state)
            
            # 多智能体LLM决策
            if self.use_real_llm:
                control = await self._compute_llm_control_async(self.state, holy_code_state)
            else:
                control = self._compute_mock_control(self.state, holy_code_state)
            
            # 系统更新
            disturbance = np.random.normal(0, 0.05, 16)
            self.state = np.dot(self.A, self.state) + np.dot(self.B, control) + disturbance * 0.1
            self.state = np.clip(self.state, -2, 2)
            
            # 记录数据
            self.state_history.append(self.state.copy())
            self.control_history.append(control.copy())
            self.rule_history.append(holy_code_state.copy())
            
            # 进度显示
            if step % 5 == 0:
                rules_count = len(holy_code_state['active_rules'])
                health = 1.0 / (1.0 + np.linalg.norm(self.state))
                print(f"步骤 {step:2d}: 规则={rules_count}, 健康度={health:.3f}")
        
        simulation_time = time.time() - start_time
        print(f"\\n✅ 异步仿真完成，耗时 {simulation_time:.2f} 秒")
        
        if self.use_real_llm:
            stats = self.llm_provider.get_stats()
            print(f"📈 LLM统计: {stats['request_count']} 次调用")
        
        return self._analyze_results()
    
    async def _compute_llm_control_async(self, state: np.ndarray, holy_code_state: Dict) -> np.ndarray:
        """异步计算LLM控制信号"""
        control = np.zeros(17)
        
        # 观测分配
        observations = {
            'doctors': state,
            'interns': state[:12],
            'patients': state[4:12],
            'accountants': state[8:12],
            'government': state[[0,1,2,3,12,13,14,15]]
        }
        
        # 控制分配
        control_slices = {
            'doctors': slice(0, 4),
            'interns': slice(4, 8),
            'patients': slice(8, 11),
            'accountants': slice(11, 14),
            'government': slice(14, 17)
        }
        
        # 并行LLM调用
        tasks = []
        for role, obs in observations.items():
            context = holy_code_state.get('crisis_level', 'normal')
            task = self.llm_provider.generate_decision(
                role, obs, holy_code_state.get('ethical_constraints', {}), context
            )
            tasks.append((role, task))
        
        # 等待所有LLM响应
        for role, task in tasks:
            action = await task
            control_slice = control_slices[role]
            slice_size = control_slice.stop - control_slice.start
            
            if len(action) >= slice_size:
                control[control_slice] = action[:slice_size]
            else:
                control[control_slice][:len(action)] = action
        
        return control
    
    def _compute_mock_control(self, state: np.ndarray, holy_code_state: Dict) -> np.ndarray:
        """计算模拟控制信号"""
        control = np.zeros(17)
        
        # 简化决策逻辑
        if np.mean(state[:4]) < 0.3:  # 健康危机
            control[:4] = [0.7, 0.5, -0.2, 0.8]  # 医生积极响应
        
        if np.mean(state[4:8]) < 0.4:  # 资源不足
            control[4:8] = [0.6, -0.3, 0.4, 0.2]  # 实习医生调整
        
        control[8:11] = [0.3, 0.4, 0.5]  # 患者基础需求
        control[11:14] = [0.4, 0.5, 0.3]  # 会计控制
        control[14:17] = [0.2, 0.1, 0.3]  # 政府监管
        
        return control
    
    def _evaluate_rules(self, state: np.ndarray) -> Dict[str, Any]:
        """评估神圣法典规则"""
        rules = []
        constraints = {}
        
        if np.mean(state[:4]) < 0.3:
            rules.append("HEALTH_CRISIS")
            constraints.update({'min_health_level': 0.6, 'emergency_response': True})
        
        if np.std(state) > 0.7:
            rules.append("SYSTEM_INSTABILITY")
            constraints.update({'stability_priority': True})
        
        if np.mean(state[8:12]) < 0.2:
            rules.append("FINANCIAL_CRISIS")
            constraints.update({'cost_control': True})
        
        return {
            'active_rules': rules,
            'ethical_constraints': constraints,
            'crisis_level': 'high' if len(rules) >= 2 else 'normal'
        }
    
    def _analyze_results(self):
        """分析结果"""
        print("\\n📊 仿真结果分析:")
        
        final_stability = np.linalg.norm(self.state_history[-1])
        avg_stability = np.mean([np.linalg.norm(s) for s in self.state_history])
        
        print(f"  最终稳定性: {final_stability:.3f}")
        print(f"  平均稳定性: {avg_stability:.3f}")
        
        # 规则统计
        total_rules = sum(len(r['active_rules']) for r in self.rule_history)
        print(f"  总规则激活: {total_rules} 次")
        
        return {
            'final_stability': final_stability,
            'average_stability': avg_stability,
            'total_rule_activations': total_rules,
            'llm_provider': self.llm_provider_type
        }

def main():
    """主函数"""
    print("🏥 医院治理系统 - 高级LLM集成仿真")
    print("=" * 60)
    
    # 检查API密钥
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    print("API密钥状态:")
    print(f"  OpenAI: {'✅ 已配置' if openai_key else '❌ 未配置'}")
    print(f"  Anthropic: {'✅ 已配置' if anthropic_key else '❌ 未配置'}")
    
    # 选择LLM提供者
    if openai_key:
        provider = 'openai'
        api_key = openai_key
        print("\\n🔧 使用 OpenAI GPT")
    elif anthropic_key:
        provider = 'anthropic'
        api_key = anthropic_key
        print("\\n🔧 使用 Anthropic Claude")
    else:
        provider = 'mock'
        api_key = None
        print("\\n🔧 使用模拟LLM（无需API密钥）")
    
    # 创建并运行仿真
    simulation = AdvancedHospitalSimulation(provider, api_key, duration=10)
    
    # 运行仿真
    if provider != 'mock':
        # 异步运行真实LLM仿真
        summary = asyncio.run(simulation.run_simulation_async())
    else:
        # 同步运行模拟仿真
        summary = asyncio.run(simulation.run_simulation_async())
    
    print("\\n🎉 高级仿真完成！")
    print("=" * 60)
    print(f"LLM提供者: {summary['llm_provider']}")
    print(f"系统稳定性: {summary['final_stability']:.3f}")
    print(f"规则激活总数: {summary['total_rule_activations']}")

if __name__ == '__main__':
    main()