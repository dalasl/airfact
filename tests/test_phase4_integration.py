"""Phase 4 Step 4.4: Execution Layer 端到端集成测试"""
import json
import time
import numpy as np


def main():
    print("=" * 60)
    print("Phase 4: Execution Layer Integration Test")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Constrained Decoder: 3-tier fallback
    # ----------------------------------------------------------
    print("\n[1/6] Constrained Decoder: outlines fallback chain...")

    from src.ch4_rule_generation.constrained_decoder import (
        ConstrainedDecoder,
        DFA,
        QUADRUPLE_JSON_SCHEMA,
    )

    cd = ConstrainedDecoder()

    # 1a. DFA decode
    dfa_result, states = cd.decode(lambda: None, "test")
    print(f"  DFA: result='{dfa_result[:60]}...', states={len(states)}")
    assert len(states) > 1

    # 1b. Schema decode with mock LLM
    def mock_llm(prompt, seed=0):
        return json.dumps({
            "subject": {"role": "engineer", "department": "R&D"},
            "object": {"type": "source_code", "sensitivity": "L3"},
            "action": "copy",
            "context": {"network": "external", "work_hours": False},
        })

    result_schema = cd.decode_with_schema(mock_llm, "engineer copies source code externally")
    assert result_schema["action"] == "copy"
    assert result_schema["object"]["sensitivity"] == "L3"
    print(f"  Schema decode: action={result_schema['action']}, sens={result_schema['object']['sensitivity']}")

    # 1c. Rule-based fallback
    result_rule = cd._rule_based_parse("研发人员不得在外部网络下载机密文件")
    assert result_rule is not None
    assert result_rule["action"] == "download"
    print(f"  Rule-based: action={result_rule['action']}")

    # 1d. Invalid LLM → fallback
    def bad_llm(prompt, seed=0):
        return "I cannot parse this invalid text"

    result_fallback = cd.decode_with_schema(bad_llm, "财务人员读取机密报表")
    assert result_fallback is not None
    print(f"  Fallback: action={result_fallback['action']}")
    print("  [PASS] Constrained Decoder verified")

    # ----------------------------------------------------------
    # 2. Z3 Formal Verification
    # ----------------------------------------------------------
    print("\n[2/6] Z3 Formal Verification: Hoare triple...")

    from src.ch4_rule_generation.template_synthesizer import TemplateSynthesizer
    from src.ch4_rule_generation.quadruple_ir import QuadrupleIR

    ts = TemplateSynthesizer()
    print(f"  Templates loaded: {len(ts.templates)}")

    # 2a. Verify all templates pass Z3
    ir_test = QuadrupleIR.from_natural_language("禁止员工通过邮件发送机密文件")
    for t in ts.templates:
        ok = ts.z3_verify(t, ir_test)
        print(f"  Z3 [{t.template_id}] {t.name}: {'PASS' if ok else 'FAIL'}")
        assert ok, f"Z3 verification failed for {t.template_id}"

    # 2b. Resource budget violation
    ts_strict = TemplateSynthesizer(resource_budget={"mem_kb": 3, "instructions": 200})
    ir2 = QuadrupleIR.from_natural_language("test")
    budget_fail = ts_strict.z3_verify(ts.templates[2], ir2)  # file_003: 12KB > 3KB
    print(f"  Budget violation (12KB > 3KB limit): {'REJECT' if not budget_fail else 'MISS'}")
    assert not budget_fail, "Should reject template exceeding resource budget"

    # 2c. Conflict detection
    conflicts = ts.z3_detect_conflicts(ts.templates)
    print(f"  Inter-template conflicts: {len(conflicts)}")
    print("  [PASS] Z3 Formal Verification verified")

    # ----------------------------------------------------------
    # 3. Template Synthesis Pipeline
    # ----------------------------------------------------------
    print("\n[3/6] Template Synthesis Pipeline...")

    # 3a. Copy action → file templates
    ir_copy = QuadrupleIR.from_natural_language("禁止通过USB拷贝敏感文件")
    vql_copy = ts.synthesize(ir_copy)
    assert vql_copy is not None
    assert "SELECT" in vql_copy
    print(f"  Copy synthesis: {len(vql_copy)} chars")
    print(f"  First line: {vql_copy.split(chr(10))[0]}")

    # 3b. Upload action → network templates
    ir_upload = QuadrupleIR.from_natural_language("禁止上传L3级以上文件到云盘")
    vql_upload = ts.synthesize(ir_upload)
    assert vql_upload is not None
    print(f"  Upload synthesis: {len(vql_upload)} chars")

    # 3c. Send action
    ir_send = QuadrupleIR.from_natural_language("禁止通过邮件发送机密文件")
    vql_send = ts.synthesize(ir_send)
    assert vql_send is not None
    print(f"  Send synthesis: {len(vql_send)} chars")

    # 3d. Generic fallback (no matching template)
    ir_generic = QuadrupleIR.from_natural_language("禁止删除系统日志")
    vql_generic = ts.synthesize(ir_generic)
    assert vql_generic is not None
    assert "generic_monitor" in vql_generic or "SELECT" in vql_generic
    print(f"  Generic fallback: {len(vql_generic)} chars")
    print("  [PASS] Template Synthesis Pipeline verified")

    # ----------------------------------------------------------
    # 4. VQL Compiler + Jinja2 Templates
    # ----------------------------------------------------------
    print("\n[4/6] VQL Compiler + Jinja2 Templates...")

    from src.ch4_rule_generation.vql_compiler import VQLCompiler

    compiler = VQLCompiler()

    # 4a. Direct IR compilation
    ir_dl = QuadrupleIR.from_natural_language("研发人员不得在外部网络下载L3级文件")
    script_dl = compiler.compile(ir_dl)
    assert script_dl.syntax_valid
    assert script_dl.logic_valid
    print(f"  IR compile: syntax={script_dl.syntax_valid}, logic={script_dl.logic_valid}")
    assert "LET events" in script_dl.script
    assert "DLP.AutoGenerated" in script_dl.script
    print(f"  Format: LET binding + VQL comment verified")

    # 4b. Jinja2 template rendering
    tmpl_vql = ts.templates[0].vql_template  # file_001
    rendered = compiler.render_template(tmpl_vql, ir_dl)
    assert "SELECT" in rendered
    assert "FROM" in rendered
    print(f"  Jinja2 render: {len(rendered)} chars")

    # 4c. Template VQL compilation
    script_tmpl = compiler.compile(ir_dl, template_vql=ts.templates[0].vql_template)
    assert script_tmpl.syntax_valid
    print(f"  Template compile: syntax={script_tmpl.syntax_valid}")

    # 4d. Logic conflict detection
    clean_vql = "SELECT * FROM watch() WHERE x > 5 AND y < 10"
    assert not compiler.detect_logic_conflicts(clean_vql)
    conflict_vql = "SELECT * FROM watch() WHERE x > 5 AND x < 3"
    assert compiler.detect_logic_conflicts(conflict_vql)
    print("  Conflict detection: clean=OK, conflict=DETECTED")

    # 4e. Batch compile
    irs = [ir_copy, ir_upload, ir_send]
    scripts = compiler.batch_compile(irs)
    assert len(scripts) == 3
    assert all(s.syntax_valid for s in scripts)
    print(f"  Batch compile: {len(scripts)} scripts, all syntax-valid")
    print("  [PASS] VQL Compiler + Jinja2 verified")

    # ----------------------------------------------------------
    # 5. Adaptive Decision Engine
    # ----------------------------------------------------------
    print("\n[5/6] Adaptive Decision Engine...")

    from src.ch4_rule_generation.adaptive_decision import (
        AdaptiveDecisionEngine,
        OperationEvent,
        ProfileBaseline,
        ResponseAction,
    )

    engine = AdaptiveDecisionEngine(
        alpha=0.35, beta=0.40, gamma=0.25,
        theta_2=0.7, theta_1_ratio=0.6,
    )

    dim = 32
    np.random.seed(42)

    # Set up baseline
    baseline = ProfileBaseline(
        mean=np.zeros(dim),
        covariance=np.eye(dim) * 0.1,
    )
    engine.set_baseline("user_alice", baseline)

    # 5a. Normal operation: internal, work hours, similar profile
    event_normal = OperationEvent(
        user_id="user_alice", action="read", file_path="/data/report.xlsx",
        process_name="excel", network_env="internal", is_work_hours=True,
    )
    profile_alice = np.random.randn(dim) * 0.1
    profile_owner = np.random.randn(dim) * 0.1
    behavior_normal = np.random.randn(dim) * 0.05

    result_normal = engine.decide(event_normal, profile_alice, profile_owner, behavior_normal)
    print(f"  Normal: action={result_normal.action.value}, R={result_normal.risk_score:.3f}")
    print(f"    s_u={result_normal.subject_consistency:.3f}, "
          f"d_b={result_normal.behavior_deviation:.3f}, "
          f"c_e={result_normal.env_confidence:.3f}")

    # 5b. Suspicious: external, off-hours, different profile
    event_suspicious = OperationEvent(
        user_id="user_bob", action="copy", file_path="/data/secret.doc",
        process_name="explorer", network_env="external",
        is_work_hours=False, device_type="usb",
    )
    baseline_bob = ProfileBaseline(
        mean=np.ones(dim) * 2.0,
        covariance=np.eye(dim) * 0.05,
    )
    engine.set_baseline("user_bob", baseline_bob)

    profile_bob = np.random.randn(dim) * 3.0
    profile_owner2 = np.ones(dim) * 0.5
    behavior_anomaly = np.ones(dim) * 5.0

    result_suspicious = engine.decide(event_suspicious, profile_bob, profile_owner2, behavior_anomaly)
    print(f"  Suspicious: action={result_suspicious.action.value}, R={result_suspicious.risk_score:.3f}")
    print(f"    s_u={result_suspicious.subject_consistency:.3f}, "
          f"d_b={result_suspicious.behavior_deviation:.3f}, "
          f"c_e={result_suspicious.env_confidence:.3f}")
    assert result_suspicious.risk_score > result_normal.risk_score, \
        "Suspicious operation should have higher risk"

    # 5c. Three-tier response ordering
    assert result_normal.action in (ResponseAction.SILENT_MONITOR, ResponseAction.ALERT)
    print(f"  Three-tier: normal={result_normal.action.value}, suspicious={result_suspicious.action.value}")

    # 5d. Dynamic whitelist
    wl_key = "user_alice:/data/report.xlsx:read"
    engine.whitelist.add(wl_key)
    assert engine.whitelist.check(wl_key)
    print(f"  Whitelist: added and checked OK")

    # 5e. Baseline EMA update
    old_mean = engine._baselines["user_alice"].mean.copy()
    engine.update_baseline("user_alice", np.ones(dim) * 0.5)
    new_mean = engine._baselines["user_alice"].mean
    assert not np.allclose(old_mean, new_mean)
    print(f"  EMA update: mean shifted by {np.linalg.norm(new_mean - old_mean):.4f}")
    print("  [PASS] Adaptive Decision Engine verified")

    # ----------------------------------------------------------
    # 6. End-to-End: Policy → IR → Template → VQL → Decision
    # ----------------------------------------------------------
    print("\n[6/6] End-to-End Pipeline...")

    policy = "财务人员不得在非工作时间通过外部网络上传L3级以上财务报表"
    print(f"  Policy: {policy}")

    # Step 1: Parse to IR
    ir_e2e = QuadrupleIR.from_natural_language(policy)
    print(f"  IR: subject={len(ir_e2e.subject.constraints)} constraints, "
          f"action={len(ir_e2e.action.constraints)} constraints")

    # Step 2: Constrained decode (schema path)
    decoded = cd.decode_with_schema(mock_llm, policy)
    print(f"  Decoded: {json.dumps(decoded, ensure_ascii=False)[:80]}...")

    # Step 3: Template synthesis
    vql_e2e = ts.synthesize(ir_e2e)
    assert vql_e2e is not None
    print(f"  VQL: {len(vql_e2e)} chars, first line: {vql_e2e.split(chr(10))[0]}")

    # Step 4: VQL compile + validate
    script_e2e = compiler.compile(ir_e2e)
    assert script_e2e.syntax_valid
    assert script_e2e.logic_valid
    print(f"  Compile: syntax={script_e2e.syntax_valid}, logic={script_e2e.logic_valid}")

    # Step 5: Decision
    event_e2e = OperationEvent(
        user_id="user_finance", action="upload", file_path="/finance/Q1.xlsx",
        process_name="chrome", network_env="external", is_work_hours=False,
    )
    engine.set_baseline("user_finance", ProfileBaseline(
        mean=np.zeros(dim), covariance=np.eye(dim) * 0.1,
    ))
    profile_fin = np.random.randn(dim) * 0.3
    profile_own = np.random.randn(dim) * 0.3
    behavior_fin = np.ones(dim) * 2.0

    decision_e2e = engine.decide(event_e2e, profile_fin, profile_own, behavior_fin)
    print(f"  Decision: {decision_e2e.action.value}, R={decision_e2e.risk_score:.3f}")
    print(f"  Reason: {decision_e2e.reason}")
    print("  [PASS] End-to-End Pipeline verified")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("Phase 4: ALL TESTS PASSED")
    print("=" * 60)
    print("  Constrained Decoder: outlines FSM + JSON Schema + rule-based fallback")
    print("  Z3 Hoare Triple: precondition implication + resource budget + conflicts")
    print("  Template Synthesis: filter -> rank -> Z3 verify -> Jinja2 compile")
    print("  VQL Compiler: Jinja2 rendering + LET bindings + Velociraptor format")
    print("  Adaptive Decision: KL-div + Mahalanobis + sigmoid + 3-tier threshold")
    print("  End-to-End: Policy -> IR -> Template -> VQL -> Decision")


if __name__ == "__main__":
    main()
