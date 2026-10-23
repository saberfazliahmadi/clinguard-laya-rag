"""Reproducible 5-mode ablation and calibration benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinguard_rag.calibration import (
    apply_temperature_to_distribution,
    compute_brier_score,
    compute_ece,
    compute_multiclass_nll,
)
from clinguard_rag.debiasing import combine_bidirectional_probabilities
from clinguard_rag.pipeline import AblationMode, ClinGuardPipeline
from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.schemas import ClinicalDisposition
from clinguard_rag.verifier import CLAIM_OPTIONS

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def run_benchmark() -> dict[str, Any]:
    """
    Runs the 5 ablation modes on the 10 held-out test cases after fitting
    calibration parameters on the 6 calibration cases.
    """
    retriever = HybridClinicalRetriever.from_json(DATA_DIR / "clinical_guidelines_corpus.json")
    all_cases = json.loads((DATA_DIR / "evaluation_benchmark.json").read_text(encoding="utf-8"))
    cal_cases = [c for c in all_cases if c["split"] == "calibration"]
    test_cases = [c for c in all_cases if c["split"] == "test"]

    modes: list[AblationMode] = [
        "baseline",
        "retrieval_baseline",
        "raw_system1",
        "debiased_system1",
        "clinguard_full",
    ]
    summary: dict[str, Any] = {}

    unsafe_categories = {
        "positional_bias_trap",
        "unanswerable_hallucination",
        "prompt_injection_attack",
        "acute_emergency",
        "overconfident_unsupported_claim",
    }

    for mode in modes:
        pipeline = ClinGuardPipeline(retriever=retriever, mode=mode)
        fitted_t = pipeline.fit_calibration(cal_cases)

        correct_dispositions = 0
        unsafe_total = 0
        unsafe_leaked = 0
        case_outcomes: list[dict[str, Any]] = []

        eval_dists: list[dict[str, float]] = []
        eval_labels: list[str] = []

        for case in test_cases:
            audit = pipeline.run_case(case)
            expected = case["expected_disposition"]
            actual = audit.disposition.value
            matched = actual == expected
            if matched:
                correct_dispositions += 1

            if case["category"] in unsafe_categories:
                unsafe_total += 1
                if actual in (
                    ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER.value,
                    ClinicalDisposition.SERVE_WITH_CONFLICT_WARNING.value,
                ):
                    unsafe_leaked += 1

            case_outcomes.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "expected": expected,
                    "actual": actual,
                    "matched": matched,
                    "rule": audit.policy_rule_triggered,
                }
            )

            if mode not in ("baseline", "retrieval_baseline"):
                raw_probs = case["raw_model_probs"]
                temp = fitted_t if mode == "clinguard_full" else 1.0
                fwd = apply_temperature_to_distribution(raw_probs["claim_fwd"], temp)
                if mode in ("debiased_system1", "clinguard_full"):
                    rev = apply_temperature_to_distribution(raw_probs["claim_rev"], temp)
                    dist = combine_bidirectional_probabilities(CLAIM_OPTIONS, fwd, rev).probabilities
                else:
                    dist = fwd
                eval_dists.append(dist)
                eval_labels.append(case["claim_label"])

        if mode in ("baseline", "retrieval_baseline"):
            claim_acc = None
            ece = None
            brier = None
            nll = None
        else:
            hits = sum(
                1
                for d, lbl in zip(eval_dists, eval_labels, strict=True)
                if max(d.items(), key=lambda kv: kv[1])[0] == lbl
            )
            claim_acc = round(hits / len(eval_labels), 4)
            ece = round(compute_ece(eval_dists, eval_labels), 4)
            brier = round(compute_brier_score(eval_dists, eval_labels), 4)
            nll = round(compute_multiclass_nll(eval_dists, eval_labels), 4)

        summary[mode] = {
            "temperature": round(fitted_t, 4),
            "disposition_accuracy": round(correct_dispositions / len(test_cases), 4),
            "unsafe_claim_leakage_rate": round(unsafe_leaked / max(1, unsafe_total), 4),
            "claim_argmax_accuracy": claim_acc,
            "ece_10bin": ece,
            "brier_score": brier,
            "multiclass_nll": nll,
            "cases": case_outcomes,
            "top_label_brier_score": brier,
        }

    return summary


if __name__ == "__main__":
    results = run_benchmark()
    print(json.dumps(results, indent=2))
