"""
测试 LLM-Actor 决策系统
验证候选生成、嵌入、选择流程
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.hospital_governance.agents.llm_actor_system import (
    LLMCandidateGenerator,
    SemanticEmbedder,
    CandidateSelector,
    NaturalLanguageActionParser,
    LLMActorDecisionSystem
)

def test_candidate_generator():
    """测试候选生成器"""
    print("\n" + "="*70)
    print("测试 1: LLM候选生成器")
    print("="*70)
    
    generator = LLMCandidateGenerator(llm_provider="mock", n_candidates=5)
    
    # 模拟系统状态
    state = np.random.uniform(0.3, 0.9, 16)
    
    # 生成候选
    candidates, tokens = generator.generate_candidates(
        role='doctors',
        system_state=state,
        history=[],
        prompt_version=0
    )
    
    print(f"\n✅ 生成了 {len(candidates)} 个候选")
    print(f"📊 消耗 tokens: {tokens}")
    print("\n候选列表:")
    for i, cand in enumerate(candidates):
        print(f"  {i+1}. {cand}")
    
    assert len(candidates) == 5
    assert tokens > 0
    print("\n✅ 候选生成器测试通过")

def test_semantic_embedder():
    """测试语义嵌入器"""
    print("\n" + "="*70)
    print("测试 2: 语义嵌入器")
    print("="*70)
    
    embedder = SemanticEmbedder()
    
    candidates = [
        "增加床位分配到重症监护",
        "优化医疗流程，提升效率",
        "加强团队协作，改善服务"
    ]
    
    embeddings = embedder.embed_candidates(candidates)
    
    print(f"\n✅ 嵌入形状: {embeddings.shape}")
    print(f"📊 嵌入维度: {embeddings.shape[1]}")
    print(f"🔢 第一个嵌入的范数: {np.linalg.norm(embeddings[0]):.3f}")
    
    assert embeddings.shape == (3, embedder.embedding_dim)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=0.01)  # 归一化检查
    print("\n✅ 语义嵌入器测试通过")

def test_candidate_selector():
    """测试候选选择器"""
    print("\n" + "="*70)
    print("测试 3: 候选选择器网络")
    print("="*70)
    
    selector = CandidateSelector(
        state_dim=16,
        n_candidates=5,
        embedding_dim=384
    )
    
    # 模拟输入
    state = torch.randn(1, 16)
    candidate_embeddings = torch.randn(1, 5, 384)
    
    # 前向传播
    logits = selector.forward(state, candidate_embeddings)
    print(f"\n✅ 输出 logits 形状: {logits.shape}")
    print(f"📊 Logits 值: {logits[0].detach().numpy()}")
    
    # 选择动作
    action_idx, log_prob = selector.select_action(state, candidate_embeddings, deterministic=False)
    print(f"🎯 选择的候选索引: {action_idx}")
    print(f"📈 对数概率: {log_prob:.4f}")
    
    assert logits.shape == (1, 6)  # 5个候选 + 1个拒绝选项
    assert 0 <= action_idx <= 5
    print("\n✅ 候选选择器测试通过")

def test_action_parser():
    """测试动作解析器"""
    print("\n" + "="*70)
    print("测试 4: 自然语言动作解析器")
    print("="*70)
    
    parser = NaturalLanguageActionParser()
    
    test_cases = [
        ("doctors", "大幅增加重症监护资源投入"),
        ("interns", "适度提升学习培训强度"),
        ("accountants", "谨慎控制医疗支出成本")
    ]
    
    for role, action in test_cases:
        vector = parser.parse(action, role)
        print(f"\n角色: {role}")
        print(f"动作: {action}")
        print(f"向量形状: {vector.shape}")
        print(f"非零元素: {np.count_nonzero(vector)}")
        print(f"向量范围: [{vector.min():.3f}, {vector.max():.3f}]")
        
        assert vector.shape == (17,)
        assert np.any(vector != 0)  # 至少有一些非零元素
    
    print("\n✅ 动作解析器测试通过")

def test_full_decision_system():
    """测试完整决策系统"""
    print("\n" + "="*70)
    print("测试 5: 完整LLM-Actor决策系统")
    print("="*70)
    
    system = LLMActorDecisionSystem(
        llm_provider="mock",
        n_candidates=5,
        state_dim=16,
        device='cpu'
    )
    
    # 模拟系统状态
    state = np.random.uniform(0.3, 0.9, 16)
    
    print("\n📊 系统状态:")
    print(f"  资源利用: {state[0]:.2%}")
    print(f"  患者满意度: {state[8]:.2%}")
    print(f"  安全指数: {state[11]:.2%}")
    
    # 测试多个角色
    roles = ['doctors', 'interns', 'patients']
    
    for role in roles:
        print(f"\n🤖 测试角色: {role}")
        result = system.get_action(
            role=role,
            state=state,
            deterministic=False
        )
        
        print(f"  ✅ 选择的动作: {result.selected_action[:60]}...")
        print(f"  📊 消耗 tokens: {result.tokens_used}")
        print(f"  🎯 选择索引: {result.selected_idx}")
        print(f"  ❌ 是否被拒绝: {result.was_rejected}")
        print(f"  📈 Log概率: {result.log_prob:.4f}")
        print(f"  🔢 控制向量非零元素: {np.count_nonzero(result.action_vector)}")
        
        assert result.action_vector.shape == (17,)
        assert result.tokens_used > 0
    
    # 查看统计信息
    stats = system.get_statistics()
    print(f"\n📈 系统统计:")
    print(f"  总tokens: {stats['total_tokens']}")
    print(f"  总拒绝次数: {stats['total_rejects']}")
    print(f"  各角色拒绝: {stats['reject_counts']}")
    
    print("\n✅ 完整决策系统测试通过")

def test_rejection_and_retry():
    """测试拒绝和重试机制"""
    print("\n" + "="*70)
    print("测试 6: 拒绝和重试机制")
    print("="*70)
    
    system = LLMActorDecisionSystem(
        llm_provider="mock",
        n_candidates=5,
        state_dim=16,
        device='cpu'
    )
    
    # 手动触发拒绝场景（通过修改选择器输出）
    state = np.random.uniform(0.3, 0.9, 16)
    
    # 强制选择器总是拒绝（用于测试重试逻辑）
    original_select = system.selector.select_action
    
    reject_count = [0]
    
    def mock_select(*args, **kwargs):
        if reject_count[0] < 2:  # 前两次拒绝
            reject_count[0] += 1
            return 5, torch.tensor(-1.0)  # 5 = 拒绝选项
        else:
            return 0, torch.tensor(-0.5)  # 接受第一个候选
    
    system.selector.select_action = mock_select
    
    result = system.get_action(
        role='doctors',
        state=state,
        deterministic=False,
        max_retries=3
    )
    
    print(f"\n✅ 拒绝次数: {reject_count[0]}")
    print(f"📊 最终选择: {result.selected_action[:60]}...")
    print(f"🔄 是否经过重试: {result.was_rejected}")
    
    # 恢复原始方法
    system.selector.select_action = original_select
    
    print("\n✅ 拒绝和重试机制测试通过")

if __name__ == "__main__":
    print("\n🏥 LLM-Actor 决策系统测试套件")
    print("="*70)
    
    try:
        test_candidate_generator()
        test_semantic_embedder()
        test_candidate_selector()
        test_action_parser()
        test_full_decision_system()
        test_rejection_and_retry()
        
        print("\n" + "="*70)
        print("🎉 所有测试通过！")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
