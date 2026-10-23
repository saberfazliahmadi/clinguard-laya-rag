"""Integration and unit tests for ClinGuardPipeline, LayaLocalBackend, SystemOneHTTPBackend, and Policy Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinguard_rag.decision_engine import (
    DecisionBackend,
    LayaLocalBackend,
    SystemOneHTTPBackend,
    normalize_questions_dict,
)
from clinguard_rag.pipeline import ClinGuardPipeline, synthesize_extractive_claim
from clinguard_rag.policy import apply_clinical_policy
from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.router import evaluate_stage1_triage
from clinguard_rag.schemas import ClinicalDisposition
from scripts.run_evaluation import run_benchmark

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class MockLiveLayaBackend(DecisionBackend):
    """
    Mock live backend validating exact laya==0.4.12 input question schemas ('criteria' for choice & score)
    and returning exact laya==0.4.12 output shapes ('noul': float for noul, string-keyed dict for score).
    """

    backend_name = "mock-live-laya"
    checkpoint_id = "convaiinnovations/laya-typed-decisions-mock"

    def predict_raw(
        self,
        state: str,
        questions: dict[str, dict[str, Any]] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        norm_q = normalize_questions_dict(questions)
        answers: dict[str, Any] = {}
        for qid, qspec in norm_q.items():
            qtype = qspec["type"]
            assert "instructions" in qspec, f"Missing instructions in {qid}"
            if qtype == "score":
                assert "criteria" in qspec and isinstance(qspec["criteria"], list)
                assert "levels" not in qspec
                answers[qid] = {
                    "type": "score",
                    "score": 0.27,
                    "probabilities": {"0": 0.80, "1": 0.15, "2": 0.03, "3": 0.02},
                }
            elif qtype == "noul":
                if qid == "prompt_attack_or_unsafe":
                    answers[qid] = {"type": "noul", "noul": 0.02}
                elif qid == "guidelines_conflict_noul":
                    answers[qid] = {"type": "noul", "noul": 0.05}
                else:
                    answers[qid] = {"type": "noul", "noul": 0.88}
            elif qtype == "choice":
                assert "criteria" in qspec, f"choice question {qid} must use 'criteria'"
                assert "options" not in qspec
                criteria = qspec["criteria"]
                opts = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
                if "antimicrobial_stewardship" in opts:
                    answers[qid] = {
                        "type": "choice",
                        "choice": "antimicrobial_stewardship",
                        "probabilities": {
                            o: (0.76 if o == "antimicrobial_stewardship" else 0.06) for o in opts
                        },
                    }
                else:
                    answers[qid] = {
                        "type": "choice",
                        "choice": "supported",
                        "probabilities": {
                            o: (0.86 if o == "supported" else 0.07) for o in opts
                        },
                    }
        return {"answers": answers}


def test_clinguard_pipeline_dispositions_across_all_10_test_cases() -> None:
    retriever = HybridClinicalRetriever.from_json(DATA_DIR / "clinical_guidelines_corpus.json")
    cases = json.loads((DATA_DIR / "evaluation_benchmark.json").read_text(encoding="utf-8"))
    cal_cases = [c for c in cases if c["split"] == "calibration"]
    test_cases = {c["case_id"]: c for c in cases if c["split"] == "test"}

    pipeline = ClinGuardPipeline(retriever=retriever, mode="clinguard_full")
    fitted_t = pipeline.fit_calibration(cal_cases)
    assert fitted_t == 1.4

    for case_id, case in test_cases.items():
        audit = pipeline.run_case(case)
        assert audit.disposition.value == case["expected_disposition"], f"Failed on {case_id}"


def test_ablation_progression_and_calibration_improvement() -> None:
    summary = run_benchmark()

    assert summary["baseline"]["disposition_accuracy"] == 0.4
    assert summary["retrieval_baseline"]["disposition_accuracy"] == 0.5
    assert summary["retrieval_baseline"]["unsafe_claim_leakage_rate"] < summary["baseline"]["unsafe_claim_leakage_rate"]
    assert summary["retrieval_baseline"]["brier_score"] is None
    retriever = HybridClinicalRetriever.from_json(DATA_DIR / "clinical_guidelines_corpus.json")
    cases = json.loads((DATA_DIR / "evaluation_benchmark.json").read_text(encoding="utf-8"))
    rb_pipe = ClinGuardPipeline(retriever=retriever, mode="retrieval_baseline")
    assert rb_pipe.retrieval_overlap_threshold == 0.0
    rb_pipe.fit_calibration([c for c in cases if c["split"] == "calibration"])
    assert rb_pipe.retrieval_overlap_threshold == 0.25
    assert summary["raw_system1"]["disposition_accuracy"] == 0.6
    assert summary["debiased_system1"]["disposition_accuracy"] == 0.8
    assert summary["clinguard_full"]["disposition_accuracy"] == 1.0
    assert summary["clinguard_full"]["unsafe_claim_leakage_rate"] == 0.0
    assert summary["clinguard_full"]["brier_score"] < summary["debiased_system1"]["brier_score"]
    assert summary["clinguard_full"]["multiclass_nll"] < summary["debiased_system1"]["multiclass_nll"]


def test_live_backend_predict_case_and_direct_policy_rules(monkeypatch: Any) -> None:
    retriever = HybridClinicalRetriever.from_json(DATA_DIR / "clinical_guidelines_corpus.json")
    live_engine = MockLiveLayaBackend()
    pipeline = ClinGuardPipeline(retriever=retriever, engine=live_engine, mode="clinguard_full")

    live_case = {
        "case_id": "LIVE-CAP-01",
        "query": "What is the recommended oral amoxicillin dose for a 42-year-old adult with mild outpatient community-acquired pneumonia?",
    }
    audit = pipeline.run_case(live_case)
    assert audit.disposition == ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER
    assert audit.policy_rule_triggered == "RULE_6_VERIFIED_CITED_CLINICAL_ANSWER"
    assert "[WHO-AWARE-CAP-01]" in audit.final_response
    assert synthesize_extractive_claim([]) == ""

    # Direct unit test of LayaLocalBackend with a stubbed _agent
    local_backend = LayaLocalBackend(checkpoint_id="convaiinnovations/laya-typed-decisions")

    class _StubAgent:
        def predict(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
            assert "[REDACTED_EMAIL]" in state
            assert "criteria" in questions["q_choice"]
            return {"answers": {"q_choice": {"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}}}

    local_backend._agent = _StubAgent()
    local_out = local_backend.predict_raw(
        state="Email test@hospital.org",
        questions=[{"id": "q_choice", "type": "choice", "question": "Pick", "options": ["a", "b"]}],
    )
    assert local_out["answers"]["q_choice"]["probabilities"]["a"] == 0.9

    # Fail-closed check for non-local SystemOneHTTPBackend without explicit override
    import pytest
    with pytest.raises(ValueError, match="fail-closed"):
        SystemOneHTTPBackend(base_url="https://api.typesafe.ai")

    # Direct unit test of SystemOneHTTPBackend with monkeypatched httpx.Client
    http_backend = SystemOneHTTPBackend(base_url="http://localhost:8000", api_key="test-key")
    captured_payload: dict[str, Any] = {}

    class _DummyResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"answers": {"q_noul": {"type": "noul", "noul": 0.93}}}

    class _DummyClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> _DummyClient:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> _DummyResp:
            captured_payload["url"] = url
            captured_payload["json"] = json
            captured_payload["headers"] = headers
            return _DummyResp()

    monkeypatch.setattr("clinguard_rag.decision_engine.httpx.Client", _DummyClient)
    http_out = http_backend.predict_raw(
        state="Call +49 40 123456",
        questions={"q_noul": {"type": "noul", "instructions": "Check"}},
    )
    assert http_out["answers"]["q_noul"]["noul"] == 0.93
    assert "[REDACTED_PHONE]" in captured_payload["json"]["state"]
    assert captured_payload["headers"]["Authorization"] == "Bearer test-key"

    # Direct unit test of apply_clinical_policy for emergency escalation (Rule 2)
    emergency_triage = evaluate_stage1_triage(
        {
            "prompt_attack_or_unsafe": 0.01,
            "clinical_urgency_probs": [0.01, 0.02, 0.10, 0.87],
            "domain_fwd": {
                "antimicrobial_stewardship": 0.10,
                "cardiovascular_hypertension": 0.10,
                "endocrine_diabetes": 0.05,
                "renal_dosing_contraindications": 0.05,
                "out_of_scope": 0.70,
            },
            "domain_rev": {
                "antimicrobial_stewardship": 0.10,
                "cardiovascular_hypertension": 0.10,
                "endocrine_diabetes": 0.05,
                "renal_dosing_contraindications": 0.05,
                "out_of_scope": 0.70,
            },
        }
    )
    disp, rule, _ = apply_clinical_policy(
        triage=emergency_triage, accepted_doc_ids=[], verification=None
    )
    assert disp == ClinicalDisposition.ESCALATE_ACUTE_EMERGENCY
    assert rule == "RULE_2_ACUTE_CLINICAL_EMERGENCY"
