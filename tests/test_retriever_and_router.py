"""Unit tests for Stage 1 two-step router, Stage 2 hybrid RRF retriever, and fail-closed gates."""

from pathlib import Path

from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.router import run_stage1_triage
from clinguard_rag.verifier import filter_retrieved_chunks

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_two_step_router_single_vs_fanout() -> None:
    single_raw = {
        "prompt_attack_or_unsafe": 0.01,
        "urgency_score_probs": [0.8, 0.1, 0.05, 0.05],
        "domain_fwd": {
            "antimicrobial_stewardship": 0.78,
            "cardiovascular_hypertension": 0.08,
            "endocrine_diabetes": 0.05,
            "renal_dosing_contraindications": 0.05,
            "out_of_scope": 0.04,
        },
        "domain_rev": {
            "antimicrobial_stewardship": 0.76,
            "cardiovascular_hypertension": 0.10,
            "endocrine_diabetes": 0.05,
            "renal_dosing_contraindications": 0.05,
            "out_of_scope": 0.04,
        },
    }
    res_single = run_stage1_triage(single_raw, temperature=1.0, use_bidirectional=True)
    assert res_single.routed_domains == ["antimicrobial_stewardship"]
    assert res_single.skip_retrieval is False

    fanout_raw = {
        "prompt_attack_or_unsafe": 0.02,
        "urgency_score_probs": [0.4, 0.4, 0.15, 0.05],
        "domain_fwd": {
            "antimicrobial_stewardship": 0.05,
            "cardiovascular_hypertension": 0.12,
            "endocrine_diabetes": 0.43,
            "renal_dosing_contraindications": 0.35,
            "out_of_scope": 0.05,
        },
        "domain_rev": {
            "antimicrobial_stewardship": 0.05,
            "cardiovascular_hypertension": 0.10,
            "endocrine_diabetes": 0.41,
            "renal_dosing_contraindications": 0.39,
            "out_of_scope": 0.05,
        },
    }
    res_fanout = run_stage1_triage(fanout_raw, temperature=1.0, use_bidirectional=True)
    assert len(res_fanout.routed_domains) == 2
    assert set(res_fanout.routed_domains) == {"endocrine_diabetes", "renal_dosing_contraindications"}


def test_hybrid_retriever_surfaces_relevant_and_hard_negative_candidates() -> None:
    retriever = HybridClinicalRetriever.from_json(DATA_DIR / "clinical_guidelines_corpus.json")
    results = retriever.retrieve(
        query="oral amoxicillin dose for outpatient community-acquired pneumonia",
        allowed_domains=["antimicrobial_stewardship"],
        top_k=2,
    )
    doc_ids = [r.chunk.doc_id for r in results]
    assert "WHO-AWARE-CAP-01" in doc_ids
    assert "HARDNEG-PED-CAP-02" in doc_ids

    # Verify fail-closed behavior on empty allowed_domains (L2) and zero-overlap queries (M5)
    assert retriever.retrieve("amoxicillin pneumonia", allowed_domains=[], top_k=2) == []
    assert retriever.retrieve("zzz qqq nonsense words", allowed_domains=None, top_k=2) == []

    # Verify fail-closed chunk relevance gate when doc_id is missing from raw_chunk_relevance (M6)
    gated = filter_retrieved_chunks(results, raw_chunk_relevance={}, temperature=1.0, relevance_threshold=0.20)
    assert all(not r.passed_gate for r in gated)
