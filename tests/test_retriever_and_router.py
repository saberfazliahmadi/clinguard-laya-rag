"""Unit tests for Hybrid BM25 + Dense RRF retrieval and Stage 1 two-step clinical routing."""

from pathlib import Path
from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.router import evaluate_stage1_triage

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_hybrid_retriever_finds_exact_clinical_guidelines() -> None:
    retriever = HybridClinicalRetriever.from_json(DATA_DIR / "clinical_guidelines_corpus.json")
    results = retriever.retrieve(
        "metformin dose adjustment eGFR 36 mL/min/1.73m2",
        allowed_domains=["renal_dosing_contraindications", "endocrine_diabetes"],
        top_k=2,
    )
    assert len(results) == 2
    assert results[0].chunk.doc_id == "KDIGO-2024-MET-01"


def test_stage1_router_two_step_fan_out_and_safety_skip() -> None:
    # Ambiguous endocrine vs renal query triggers 2-domain fan-out (< 0.55 top domain)
    ambiguous_probs = {
        "prompt_attack_or_unsafe": 0.01,
        "clinical_urgency_probs": [0.1, 0.4, 0.4, 0.1],
        "domain_fwd": {
            "antimicrobial_stewardship": 0.02,
            "cardiovascular_hypertension": 0.08,
            "endocrine_diabetes": 0.48,
            "renal_dosing_contraindications": 0.40,
            "out_of_scope": 0.02,
        },
        "domain_rev": {
            "antimicrobial_stewardship": 0.02,
            "cardiovascular_hypertension": 0.08,
            "endocrine_diabetes": 0.46,
            "renal_dosing_contraindications": 0.42,
            "out_of_scope": 0.02,
        },
    }
    triage = evaluate_stage1_triage(ambiguous_probs)
    assert triage.skip_retrieval is False
    assert triage.routed_domains == ["endocrine_diabetes", "renal_dosing_contraindications"]

    # High prompt-attack probability skips retrieval immediately
    attack_probs = dict(ambiguous_probs, prompt_attack_or_unsafe=0.92)
    triage_attack = evaluate_stage1_triage(attack_probs)
    assert triage_attack.skip_retrieval is True
    assert triage_attack.routed_domains == []