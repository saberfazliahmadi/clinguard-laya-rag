"""Integration tests for ClinGuard-RAG end-to-end clinical policy guardrails and ablation metrics."""

import json
from pathlib import Path
from clinguard_rag.pipeline import ClinGuardPipeline
from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.schemas import ClinicalDisposition
from scripts.run_evaluation import run_benchmark

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_clinguard_pipeline_dispositions_across_all_10_test_cases() -> None:
    retriever = HybridClinicalRetriever.from_json(DATA_DIR / "clinical_guidelines_corpus.json")
    cases = json.loads((DATA_DIR / "evaluation_benchmark.json").read_text(encoding="utf-8"))
    cal_cases = [c for c in cases if c["split"] == "calibration"]
    test_cases = {c["case_id"]: c for c in cases if c["split"] == "test"}

    pipeline = ClinGuardPipeline(retriever=retriever, mode="clinguard_full")
    fitted_t = pipeline.fit_calibration(cal_cases)
    assert fitted_t == 1.4

    # Verify all 10 held-out test cases match expected clinical dispositions
    for case_id, case in test_cases.items():
        audit = pipeline.run_case(case)
        assert audit.disposition.value == case["expected_disposition"], f"Failed on {case_id}"

    # Verify hard-negative pediatric pneumonia chunk is filtered out in EVAL-02
    r2 = pipeline.run_case(test_cases["EVAL-02"])
    assert r2.accepted_doc_ids == ["WHO-AWARE-CAP-01"]

    # Verify conflicting SGLT2 guidelines are handled in EVAL-03
    r3 = pipeline.run_case(test_cases["EVAL-03"])
    assert "ADA-2025-SGLT2-HF" in r3.accepted_doc_ids
    assert r3.disposition == ClinicalDisposition.SERVE_WITH_CONFLICT_WARNING


def test_ablation_progression_and_calibration_improvement() -> None:
    summary = run_benchmark()

    assert summary["baseline"]["disposition_accuracy"] == 0.4
    assert summary["raw_system1"]["disposition_accuracy"] == 0.6
    assert summary["debiased_system1"]["disposition_accuracy"] == 0.8
    assert summary["clinguard_full"]["disposition_accuracy"] == 1.0

    assert summary["baseline"]["unsafe_claim_leakage_rate"] == 1.0
    assert summary["raw_system1"]["unsafe_claim_leakage_rate"] == 0.6
    assert summary["debiased_system1"]["unsafe_claim_leakage_rate"] == 0.2
    assert summary["clinguard_full"]["unsafe_claim_leakage_rate"] == 0.0

    # Calibration metrics improve monotonically across System-1 stages (verified via Brier score & NLL)
    assert summary["clinguard_full"]["brier_score"] <= summary["debiased_system1"]["brier_score"]
    assert summary["clinguard_full"]["multiclass_nll"] <= summary["debiased_system1"]["multiclass_nll"]