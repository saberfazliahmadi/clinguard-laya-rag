"""Stage 3 & Stage 4: Evidence chunk relevance filtering and 3-way claim verification."""

from __future__ import annotations
import re
from typing import Any, Dict, List
from clinguard_rag.calibration import apply_temperature_to_binary, apply_temperature_to_distribution
from clinguard_rag.debiasing import combine_bidirectional_probabilities
from clinguard_rag.schemas import ClaimVerification, DebiasedChoiceResult, RetrievedChunk

CLAIM_OPTIONS: List[str] = ["supported", "contradicted", "not_mentioned"]
CITATION_PATTERN = re.compile(r"\[([A-Z0-9\-]+)\]")


def extract_citations(claim_text: str) -> List[str]:
    return CITATION_PATTERN.findall(claim_text)


def filter_retrieved_chunks(
    retrieved: List[RetrievedChunk],
    raw_chunk_relevance: Dict[str, float],
    temperature: float = 1.0,
    relevance_threshold: float = 0.60,
) -> List[RetrievedChunk]:
    """Evaluates retrieved passages with calibrated System-1 noul relevance scores (fail-closed on missing keys)."""
    updated: List[RetrievedChunk] = []
    for item in retrieved:
        if item.chunk.doc_id not in raw_chunk_relevance:
            cal_p = 0.0
            passed = False
        else:
            raw_p = raw_chunk_relevance[item.chunk.doc_id]
            cal_p = apply_temperature_to_binary(raw_p, temperature)
            passed = cal_p >= relevance_threshold
        updated.append(
            item.model_copy(
                update={
                    "system1_relevance_prob": round(cal_p, 4),
                    "passed_gate": passed,
                }
            )
        )
    return updated


def verify_clinical_claim(
    claim_text: str,
    raw_probs: Dict[str, Any],
    temperature: float = 1.0,
    use_bidirectional: bool = True,
    three_way: bool = True,
    drift_threshold: float = 0.25,
) -> ClaimVerification:
    """Runs 3-way (or 2-way baseline) claim verification with bidirectional order debiasing."""
    fwd = apply_temperature_to_distribution(raw_probs["claim_fwd"], temperature)
    rev = apply_temperature_to_distribution(raw_probs["claim_rev"], temperature)

    if not three_way:
        fwd_2 = {
            "supported": fwd["supported"],
            "contradicted": fwd["contradicted"] + fwd["not_mentioned"],
            "not_mentioned": 0.0,
        }
        rev_2 = {
            "supported": rev["supported"],
            "contradicted": rev["contradicted"] + rev["not_mentioned"],
            "not_mentioned": 0.0,
        }
        fwd, rev = fwd_2, rev_2

    if use_bidirectional:
        verdict = combine_bidirectional_probabilities(
            CLAIM_OPTIONS, fwd, rev, drift_threshold=drift_threshold
        )
    else:
        selected = max(CLAIM_OPTIONS, key=lambda o: fwd.get(o, 0.0))
        verdict = DebiasedChoiceResult(
            selected_option=selected,
            probabilities={k: round(v, 4) for k, v in fwd.items()},
            order_tvd_drift=0.0,
            is_positionally_stable=True,
            tokens_per_option_budget=58,
        )

    raw_conflict = raw_probs.get("guidelines_conflict_noul")
    if raw_conflict is None:
        for k, v in raw_probs.items():
            if "conflict" in k.lower() and isinstance(v, (int, float)):
                raw_conflict = float(v)
                break
    if raw_conflict is None:
        raw_conflict = 0.05

    conflict_prob = apply_temperature_to_binary(float(raw_conflict), temperature)
    return ClaimVerification(
        claim_text=claim_text,
        cited_doc_ids=extract_citations(claim_text),
        verdict=verdict,
        guidelines_conflict_prob=round(conflict_prob, 4),
    )


# Alias for evaluation script compatibility
CLAIM_VERIFICATION_OPTIONS_3WAY = CLAIM_OPTIONS
