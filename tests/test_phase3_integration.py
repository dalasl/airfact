"""Phase 3 Step 3.3: CognitionService 端到端集成测试"""
import json
import sys
import numpy as np

def main():
    print("=" * 60)
    print("Phase 3 Step 3.3: CognitionService Integration Test")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. RAG: seed injection + retrieval + Top-p filtering
    # ----------------------------------------------------------
    print("\n[1/5] RAG + Top-p filtering...")

    from src.ch3_sensitive_grading.rag_retriever import RAGRetriever

    retriever = RAGRetriever(k=3, top_p=0.85, temperature=1.0)

    seed_cases = [
        {"doc_summary": "2024Q1 financial report with revenue and cost details",
         "level": "L3", "original_level": "L2", "reason": "core financial data"},
        {"doc_summary": "employee salary breakdown with bonuses",
         "level": "L4", "original_level": "L3", "reason": "personal privacy"},
        {"doc_summary": "company press release about product launch",
         "level": "L1", "original_level": "L1", "reason": "public info"},
        {"doc_summary": "internal technical architecture and API specs",
         "level": "L2", "original_level": "L2", "reason": "internal docs"},
        {"doc_summary": "client contract with pricing and NDA terms",
         "level": "L4", "original_level": "L3", "reason": "trade secret"},
    ]
    retriever.inject_seed_cases(seed_cases)
    print(f"  Seeds injected: {len(seed_cases)}")
    print(f"  Vector store size: {len(retriever._records)}")

    query_text = "quarterly budget and expense report"
    cases, query_vec = retriever.build_context(query_text)
    print(f"  Query: '{query_text}'")
    print(f"  Retrieved: k={retriever.k} -> Top-p filtered: {len(cases)}")
    for c in cases:
        print(f"    - [{c.corrected_level}] (w={c.attention_weight:.3f}) {c.doc_summary[:50]}")

    assert len(cases) > 0, "Top-p results must not be empty"
    weight_sum = sum(c.attention_weight for c in cases)
    assert abs(weight_sum - 1.0) < 0.01, f"Weights should sum to 1.0, got {weight_sum}"
    print("  [PASS] RAG + Top-p verified")

    # ----------------------------------------------------------
    # 2. Five-layer prompt construction
    # ----------------------------------------------------------
    print("\n[2/5] Five-layer prompt construction...")

    from src.ch3_sensitive_grading.prompt_builder import PromptBuilder, UserProfile

    profile = UserProfile(
        user_id="vm2_bob",
        role="finance",
        department="Finance",
        projects=["Phoenix", "Atlas"],
        permissions=["read", "write", "export"],
        business_keywords=["budget", "cost", "revenue"],
        cluster_id=1,
    )

    builder = PromptBuilder()
    prompt = builder.build_prompt(
        profile=profile,
        cases=cases,
        file_path="C:/Users/bob/Q1_budget.xlsx",
        content_snippet="Q1 budget execution: total 12M, spent 8.5M, rate 70.8%...",
    )

    assert "System:" in prompt, "Missing layer 1: system role"
    assert "Context:" in prompt, "Missing layer 2: profile"
    assert "Few-Shot:" in prompt, "Missing layer 3: RAG"
    assert "Input:" in prompt, "Missing layer 4: data"
    assert "Output:" in prompt, "Missing layer 5: output constraint"
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Layers: System / Context / Few-Shot({len(cases)}) / Input / Output")
    print("  [PASS] Five-layer prompt verified")

    # ----------------------------------------------------------
    # 3. Self-consistency verifier (MC-Dropout + majority vote)
    # ----------------------------------------------------------
    print("\n[3/5] Self-consistency verifier...")

    from src.ch3_sensitive_grading.self_consistency import SelfConsistencyVerifier

    verifier = SelfConsistencyVerifier(
        T=5, rho_min=0.6, c_min=0.85, gamma=1.5, beta=0.6
    )

    # Scenario A: high consistency (5/5 L3)
    def consistent_llm(prompt, seed=0):
        return json.dumps({
            "level": "L3", "category": "financial",
            "reason": f"seed={seed}", "confidence": 0.92,
        })

    dec_a = verifier.verify(consistent_llm, "test")
    print(f"  A [high consistency]: level={dec_a.level}, rho={dec_a.consistency_ratio:.2f}, "
          f"c_bar={dec_a.confidence:.2f}, U={dec_a.uncertainty:.3f}")
    print(f"    accepted={dec_a.is_accepted}, review={dec_a.needs_human_review}")
    assert dec_a.level == "L3"
    assert dec_a.consistency_ratio == 1.0
    assert dec_a.is_accepted is True
    assert dec_a.needs_human_review is False
    print("    [PASS]")

    # Scenario B: moderate consistency (3/5 L3, 2/5 L2)
    def moderate_llm(prompt, seed=0):
        level = "L3" if seed in (0, 1, 3) else "L2"
        conf = 0.88 if level == "L3" else 0.75
        return json.dumps({
            "level": level, "category": "financial",
            "reason": f"seed={seed}", "confidence": conf,
        })

    dec_b = verifier.verify(moderate_llm, "test")
    print(f"  B [moderate]: level={dec_b.level}, rho={dec_b.consistency_ratio:.2f}, "
          f"c_bar={dec_b.confidence:.2f}, U={dec_b.uncertainty:.3f}")
    print(f"    accepted={dec_b.is_accepted}, review={dec_b.needs_human_review}")
    assert dec_b.level == "L3"
    assert dec_b.consistency_ratio == 0.6
    print("    [PASS]")

    # Scenario C: scattered predictions -> needs review
    def scattered_llm(prompt, seed=0):
        levels = ["L1", "L2", "L3", "L4", "L2"]
        return json.dumps({
            "level": levels[seed % 5], "category": "mixed",
            "reason": f"seed={seed}", "confidence": 0.5,
        })

    dec_c = verifier.verify(scattered_llm, "test")
    print(f"  C [scattered]: level={dec_c.level}, rho={dec_c.consistency_ratio:.2f}, "
          f"c_bar={dec_c.confidence:.2f}, U={dec_c.uncertainty:.3f}")
    print(f"    accepted={dec_c.is_accepted}, review={dec_c.needs_human_review}")
    assert dec_c.needs_human_review is True
    print("    [PASS]")

    # PAC bound
    pac_70 = SelfConsistencyVerifier.pac_bound(0.7, 5)
    pac_80 = SelfConsistencyVerifier.pac_bound(0.8, 5)
    print(f"  PAC: p=0.7,T=5 -> {pac_70:.3f} | p=0.8,T=5 -> {pac_80:.3f}")
    assert pac_70 > 0.83
    assert pac_80 > 0.94
    print("    [PASS]")

    # ----------------------------------------------------------
    # 4. GradingPipeline end-to-end + ablation
    # ----------------------------------------------------------
    print("\n[4/5] GradingPipeline end-to-end...")

    from src.ch3_sensitive_grading.grading_pipeline import GradingPipeline

    pipeline = GradingPipeline(
        llm_fn=consistent_llm,
        config={
            "rag": {"k": 3, "top_p": 0.85, "temperature": 1.0},
            "self_consistency": {"T": 5, "rho_min": 0.6, "c_min": 0.85, "gamma": 1.5, "beta": 0.6},
        },
    )
    pipeline.retriever.inject_seed_cases(seed_cases)

    decision = pipeline.grade_document(
        file_path="C:/Users/bob/Q1_budget.xlsx",
        content_snippet="Q1 budget execution report, total 12M...",
        profile=profile,
    )
    print(f"  Result: level={decision.level}, rho={decision.consistency_ratio:.2f}, "
          f"c_bar={decision.confidence:.2f}")
    assert decision.level == "L3"
    assert decision.is_accepted is True

    # Ablation: no profile
    ab_np = pipeline.run_ablation_no_profile("C:/test.xlsx", "budget report")
    print(f"  Ablation[no_profile]: level={ab_np.level}, rho={ab_np.consistency_ratio:.2f}")

    # Ablation: no RAG
    ab_nr = pipeline.run_ablation_no_rag("C:/test.xlsx", "budget report", profile)
    print(f"  Ablation[no_rag]: level={ab_nr.level}, rho={ab_nr.consistency_ratio:.2f}")

    # Ablation: no consistency (T=1)
    ab_nc = pipeline.run_ablation_no_consistency("C:/test.xlsx", "budget report", profile)
    print(f"  Ablation[no_consistency]: level={ab_nc.level}, rho={ab_nc.consistency_ratio:.2f}")
    assert ab_nc.consistency_ratio == 1.0
    print("  [PASS] GradingPipeline all modes verified")

    # ----------------------------------------------------------
    # 5. CognitionService.process() integration
    # ----------------------------------------------------------
    print("\n[5/5] CognitionService.process()...")

    from src.server import CognitionService, ProfileMatchResult

    config = {
        "grading": {
            "rag": {"k": 3, "top_p": 0.85},
            "self_consistency": {"T": 5, "rho_min": 0.6, "c_min": 0.85, "gamma": 1.5},
        },
    }
    cognition = CognitionService(config, llm_fn=consistent_llm)
    cognition.inject_seeds(seed_cases)

    profile_result = ProfileMatchResult(
        user_id="vm2_bob",
        user_profile=profile,
        cluster_id=1,
        profile_version=3,
        feature_vector=np.random.randn(896).astype(np.float32),
        behavior_pattern=np.random.randn(96).astype(np.float32),
        content_text="Q1 financial budget report with department expenses...",
        drift_detected=False,
        correlation_id="test_corr_001",
    )

    grading_result = cognition.process(
        file_path="C:/Users/bob/Q1_budget.xlsx",
        profile_result=profile_result,
    )
    print(f"  GradingResult:")
    print(f"    level={grading_result.level}, effective={grading_result.effective_level}")
    print(f"    confidence={grading_result.confidence:.2f}")
    print(f"    accepted={grading_result.is_accepted}, review={grading_result.needs_human_review}")
    print(f"    correlation_id={grading_result.correlation_id}")

    assert grading_result.level == "L3"
    assert grading_result.effective_level == "L3"  # conf 0.92 >= 0.7, no upgrade
    assert grading_result.is_accepted is True
    assert grading_result.correlation_id == "test_corr_001"

    # Test confidence-based level upgrade
    def low_conf_llm(prompt, seed=0):
        return json.dumps({
            "level": "L2", "category": "internal",
            "reason": f"seed={seed}", "confidence": 0.55,
        })

    cognition_low = CognitionService(config, llm_fn=low_conf_llm)
    gr_low = cognition_low.process(file_path="C:/test.xlsx", profile_result=profile_result)
    print(f"  Level upgrade: level={gr_low.level} -> effective={gr_low.effective_level} "
          f"(conf={gr_low.confidence:.2f})")
    assert gr_low.level == "L2"
    assert gr_low.effective_level == "L3", f"L2 with conf<0.7 should upgrade to L3"
    print("  [PASS] CognitionService integration verified")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("Phase 3 Step 3.3: ALL TESTS PASSED")
    print("=" * 60)
    print("  RAG vector store + Top-p attention filtering")
    print("  Five-layer structured prompt construction")
    print("  MC-Dropout T=5 + self-consistency majority vote")
    print("  Uncertainty quantification (H + beta*tr(Var))")
    print("  Dual-threshold decision (rho_min=0.6, c_min=0.85)")
    print("  PAC bound (p=0.7->0.837, p=0.8->0.942)")
    print("  GradingPipeline + 3 ablation experiments")
    print("  CognitionService.process() end-to-end")
    print("  Confidence-based level upgrade mechanism")


if __name__ == "__main__":
    main()
