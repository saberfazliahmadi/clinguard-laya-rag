"""Reproducible offline golden-fixture ablation and calibration benchmark runner."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from clinguard_rag.calibration import compute_ece_and_brier, multiclass_nll
from clinguard_rag.pipeline import AblationMode, ClinGuardPipeline
from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.schemas import ClinicalDisposition

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def run_benchmark() -> Dict[str, Any]:
    retriever = HybridClinicalRetriever.from_json(DATA_DIR / "clinical_guidelines_corpus.json")
    cases = json.loads((DATA_DIR / "evaluation_benchmark.json").read_text(encoding="utf-8"))

    cal_cases = [c for c in cases if c["split"] == "calibration"]
    test_cases = [c for c in cases if c["split"] == "test"]

    modes: List[AblationMode] = [
        "baseline",
        "retrieval_baseline",
        "raw_system1",
        "debiased_system1",
        "clinguard_full",
    ]
    summary: Dict[str, Any] = {}

    for mode in modes:
        pipeline = ClinGuardPipeline(retriever=retriever, mode=mode)
        fitted_t = pipeline.fit_calibration(cal_cases)

        correct_dispositions = 0
        unsafe_total = 0
        unsafe_leaked = 0

        confidences: List[float] = []
        correctness: List[int] = []
        claim_dists: List[Dict[str, float]] = []
        claim_labels: List[str] = []

        for case in test_cases:
            audit = pipeline.run_case(case)
            expected_disp = case["expected_disposition"]
            if audit.disposition.value == expected_disp:
                correct_dispositions += 1

            if case.get("is_unsafe_or_unsupported", False):
                unsafe_total += 1
                if audit.disposition in (
                    ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER,
                    ClinicalDisposition.SERVE_WITH_CONFLICT_WARNING,
                ):
                    unsafe_leaked += 1

            if audit.claim_verification is not None:
                probs = audit.claim_verification.verdict.probabilities
                sel = audit.claim_verification.verdict.selected_option
                true_label = case["claim_label"]
                confidences.append(probs[sel])
                correctness.append(1 if sel == true_label else 0)
                claim_dists.append(probs)
                claim_labels.append(true_label)

        ece, brier = compute_ece_and_brier(confidences, correctness, num_bins=10)
        nll = round(multiclass_nll(claim_dists, claim_labels, temperature=1.0), 4) if claim_dists else 0.0

        summary[mode] = {
            "temperature": fitted_t,
            "disposition_accuracy": round(correct_dispositions / len(test_cases), 4),
            "unsafe_claim_leakage_rate": round(unsafe_leaked / max(1, unsafe_total), 4),
            "ece_10_bin": ece,
            "brier_score": brier,
            "top_label_brier_score": brier,
            "multiclass_nll": nll,
        }

    return summary


if __name__ == "__main__":
    results = run_benchmark()
    print(json.dumps(results, indent=2))
    assert results["clinguard_full"]["disposition_accuracy"] == 1.0, "Full mode must reach 100% fixture accuracy"
    assert results["clinguard_full"]["unsafe_claim_leakage_rate"] == 0.0, "Full mode must have 0% unsafe leakage"
