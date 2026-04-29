"""Phase 5: 对比基线端到端集成测试"""


def main():
    print("=" * 60)
    print("Phase 5: Baseline Comparison Integration Test")
    print("=" * 60)

    # ----------------------------------------------------------
    # 测试文档（覆盖 L1-L4 各等级场景）
    # ----------------------------------------------------------
    test_documents = [
        {
            "name": "公开新闻稿",
            "content": "公开发布：本公司2024年产品手册发布会将于下周举行，欢迎媒体参加。",
            "file_path": "/public/press_release.docx",
            "expected_level": "L1",
        },
        {
            "name": "内部会议纪要",
            "content": "内部会议纪要：2024年Q2项目进度汇报，Phoenix项目完成80%，Atlas项目进入测试阶段。通讯录已更新。",
            "file_path": "/internal/meeting_minutes.docx",
            "expected_level": "L2",
        },
        {
            "name": "财务报表",
            "content": "机密 - 2024年Q1财务报告：总收入12M，成本8.5M，净利润3.5M。员工薪酬支出占比35%。confidential financial report.",
            "file_path": "/finance/Q1_report.xlsx",
            "expected_level": "L3",
        },
        {
            "name": "密钥与证件",
            "content": "绝密文件：生产环境数据库密钥 private key: sk-abc123。SSN: 123-45-6789。passport number: E12345678。",
            "file_path": "/secret/credentials.txt",
            "expected_level": "L4",
        },
        {
            "name": "含PII邮件",
            "content": "Dear John Smith, your credit card 4111-1111-1111-1111 has been charged. Contact: john@company.com, +1-555-123-4567. IP: 192.168.1.100",
            "file_path": "/email/notification.eml",
            "expected_level": "L3",
        },
    ]

    # ----------------------------------------------------------
    # 1. 静态规则基线
    # ----------------------------------------------------------
    print("\n[1/4] Static Rules Baseline...")

    from src.baselines.static_rules import StaticRuleGrader

    static_grader = StaticRuleGrader()

    static_results = []
    for doc in test_documents:
        result = static_grader.grade(
            content=doc["content"],
            file_path=doc["file_path"],
        )
        static_results.append(result)
        match = "OK" if result.level == doc["expected_level"] else "MISS"
        print(f"  [{match}] {doc['name']}: predicted={result.level}, "
              f"expected={doc['expected_level']}, rules={len(result.matched_rules)}")

    # 验证关键场景
    assert static_results[0].level == "L1", f"公开文档应为L1, got {static_results[0].level}"
    assert static_results[3].level == "L4", f"绝密文档应为L4, got {static_results[3].level}"

    # 批量分级
    batch_results = static_grader.grade_batch([
        {"content": d["content"], "file_path": d["file_path"]}
        for d in test_documents
    ])
    assert len(batch_results) == len(test_documents)
    print(f"  Batch: {len(batch_results)} documents graded")

    # 通道管控
    usb_result = static_grader.grade("普通文件内容", channel="usb")
    assert usb_result.level >= "L2", "USB通道应提升等级"
    print(f"  Channel (USB): {usb_result.level}")
    print("  [PASS] Static Rules Baseline verified")

    # ----------------------------------------------------------
    # 2. DeBERTa 基线（mock 模式，无需真实模型）
    # ----------------------------------------------------------
    print("\n[2/4] DeBERTa Classifier Baseline (mock)...")

    from src.baselines.deberta_classifier import DeBERTaClassifier

    deberta = DeBERTaClassifier()

    deberta_results = []
    for doc in test_documents:
        result = deberta._mock_grade(doc["content"])
        deberta_results.append(result)
        match = "OK" if result.level == doc["expected_level"] else "MISS"
        print(f"  [{match}] {doc['name']}: predicted={result.level}, "
              f"expected={doc['expected_level']}, conf={result.confidence:.2f}")

    assert deberta_results[3].level == "L4", f"绝密文档应为L4, got {deberta_results[3].level}"

    batch_deberta = deberta.grade_batch([d["content"] for d in test_documents])
    assert len(batch_deberta) == len(test_documents)
    print(f"  Batch: {len(batch_deberta)} documents graded")
    print("  [PASS] DeBERTa Baseline verified (mock mode)")

    # ----------------------------------------------------------
    # 3. Presidio NER 基线（正则回退模式）
    # ----------------------------------------------------------
    print("\n[3/4] Presidio NER Baseline (regex fallback)...")

    from src.baselines.presidio_grading import PresidioGrader

    presidio = PresidioGrader()

    presidio_results = []
    for doc in test_documents:
        result = presidio.grade(doc["content"])
        presidio_results.append(result)
        entity_count = len(result.entities) if result.entities else 0
        match = "OK" if result.level == doc["expected_level"] else "MISS"
        print(f"  [{match}] {doc['name']}: predicted={result.level}, "
              f"expected={doc['expected_level']}, entities={entity_count}")
        if result.entities:
            for e in result.entities[:3]:
                print(f"         {e.entity_type}: '{e.text}' ({e.severity})")

    # PII 文档应检测到实体
    assert len(presidio_results[4].entities) > 0, "PII邮件应检测到实体"
    pii_level = presidio_results[4].level
    assert pii_level in ("L3", "L4"), f"PII邮件至少L3, got {pii_level}"

    batch_presidio = presidio.grade_batch([d["content"] for d in test_documents])
    assert len(batch_presidio) == len(test_documents)
    print(f"  Batch: {len(batch_presidio)} documents graded")
    print("  [PASS] Presidio NER Baseline verified (regex fallback)")

    # ----------------------------------------------------------
    # 4. 基线对比汇总
    # ----------------------------------------------------------
    print("\n[4/4] Baseline Comparison Summary...")

    level_order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}

    print()
    header = f"{'Document':<16} {'Expected':<10} {'Static':<10} {'DeBERTa':<10} {'Presidio':<10}"
    print(header)
    print("-" * len(header))

    static_correct = 0
    deberta_correct = 0
    presidio_correct = 0

    for i, doc in enumerate(test_documents):
        expected = doc["expected_level"]
        s = static_results[i].level
        d = deberta_results[i].level
        p = presidio_results[i].level

        if s == expected:
            static_correct += 1
        if d == expected:
            deberta_correct += 1
        if p == expected:
            presidio_correct += 1

        print(f"{doc['name']:<16} {expected:<10} {s:<10} {d:<10} {p:<10}")

    total = len(test_documents)
    print("-" * len(header))
    print(f"{'Accuracy':<16} {'---':<10} "
          f"{static_correct}/{total}={static_correct/total:.0%}    "
          f"{deberta_correct}/{total}={deberta_correct/total:.0%}    "
          f"{presidio_correct}/{total}={presidio_correct/total:.0%}")

    # 验证接口一致性
    for result_list, name in [
        (static_results, "Static"),
        (deberta_results, "DeBERTa"),
        (presidio_results, "Presidio"),
    ]:
        for r in result_list:
            assert hasattr(r, "level"), f"{name}: missing 'level'"
            assert hasattr(r, "confidence"), f"{name}: missing 'confidence'"
            assert hasattr(r, "consistency_ratio"), f"{name}: missing 'consistency_ratio'"
            assert hasattr(r, "is_accepted"), f"{name}: missing 'is_accepted'"
            assert hasattr(r, "reason"), f"{name}: missing 'reason'"
            assert r.level in ("L1", "L2", "L3", "L4"), f"{name}: invalid level '{r.level}'"
            assert 0 <= r.confidence <= 1.0, f"{name}: confidence out of range"
    print("\n  Interface consistency: ALL PASS")
    print("  [PASS] Baseline Comparison verified")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("Phase 5: ALL TESTS PASSED")
    print("=" * 60)
    print("  Static Rules: 38 keywords + file attributes + channel control")
    print("  DeBERTa-ft: microsoft/deberta-v3-base L1-L4 classification (mock)")
    print("  Presidio-NER: PII entity detection + severity mapping (regex fallback)")
    print("  Interface: level/confidence/consistency_ratio/is_accepted/reason")
    print(f"  Test accuracy: Static={static_correct}/{total}, "
          f"DeBERTa={deberta_correct}/{total}, Presidio={presidio_correct}/{total}")


if __name__ == "__main__":
    main()
