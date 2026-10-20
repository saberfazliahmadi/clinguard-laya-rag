"""Stage 3 Post-Retrieval Evidence Gate and Stage 4 3-Way Claim Verification Judge."""

from __future__ import annotations
import re
from typing import Any, Dict, List
from clinguard_rag.calibration import (
    apply_temperature_to_binary,
    apply_temperature_to_distribution,
)
from clinguard_rag.debiasing import combine_bidirectional_probabilities
from clinguard_rag.schemas import (
    ClaimVerificationResult,
    DebiasedChoiceResult,
    RetrievedChunk,
)

CLAIM_VERIFICATION_OPTIONS_3WAY: List[str] = [
    "supported",
    "contradicted",
    "not_mentioned",
]


def filter_retrieved_chunks(
    retrieved: List[RetrievedChunk],
    raw_chunk_relevance: Dict[str, float],
    temperature: float = 1.0,
    relevance_threshold: float = 0.60,
) -> List[RetrievedChunk]:
    """
    Evaluates each retrieved clinical passage via Laya's `noul` relevance gate
    to reject BM25 lexical hard negatives (e.g., pediatric vs. adult dosing).
    """
    evaluated: List[RetrievedChunk] = []
    for item in retrieved:
        raw_p = float(raw_chunk_relevance.get(item.chunk.doc_id, 0.25))
        cal_p = apply_temperature_to_binary(raw_p, temperature)
        updated = item.model_copy(
            update={
                "relevance_prob": round(cal_p, 4),
                "passed_gate": cal_p >= relevance_threshold,
            }
        )
        evaluated.append(updated)
    return evaluated


def verify_clinical_claim(
    claim_text: str,
    raw_probs: Dict[str, Any],
    temperature: float = 1.0,
    use_bidirectional_debiasing: bool = True,
    use_three_way_rubric: bool = True,
) -> ClaimVerificationResult:
    """
    Verifies a synthesized clinical claim against retrieved evidence using the
    3-way choice rubric (`supported`, `contradicted`, `not_mentioned`) and
    bidirectional option-order ensembling.
    """
    citations = re.findall(r"\[([A-Z0-9\-]+)\]", claim_text)
    conflict_prob = apply_temperature_to_binary(
        float(raw_probs.get("guidelines_conflict", 0.05)), temperature
    )

    fwd_raw = dict(raw_probs["claim_fwd"])
    rev_raw = dict(raw_probs["claim_rev"])

    if not use_three_way_rubric:
        # 2-way baseline collapses contradicted and not_mentioned into not_supported
        fwd_2 = {
            "supported": fwd_raw["supported"],
            "not_supported": fwd_raw["contradicted"] + fwd_raw["not_mentioned"],
        }
        rev_2 = {
            "supported": rev_raw["supported"],
            "not_supported": rev_raw["contradicted"] + rev_raw["not_mentioned"],
        }
        fwd_cal = apply_temperature_to_distribution(fwd_2, temperature)
        rev_cal = apply_temperature_to_distribution(rev_2, temperature)
        best_opt = "supported" if fwd_cal["supported"] >= 0.5 else "not_supported"
        verdict = DebiasedChoiceResult(
            selected_option=best_opt,
            probabilities=fwd_cal,
            forward_probabilities=fwd_cal,
            reverse_probabilities=rev_cal,
            order_tvd_drift=round(abs(fwd_cal["supported"] - rev_cal["supported"]), 4),
            is_positionally_stable=True,
            tokens_per_option_budget=88,
        )
    else:
        fwd_cal = apply_temperature_to_distribution(fwd_raw, temperature)
        rev_cal = apply_temperature_to_distribution(rev_raw, temperature)
        if use_bidirectional_debiasing:
            verdict = combine_bidirectional_probabilities(
                CLAIM_VERIFICATION_OPTIONS_3WAY, fwd_cal, rev_cal
            )
        else:
            best_opt = max(fwd_cal.items(), key=lambda kv: kv[1])[0]
            verdict = DebiasedChoiceResult(
                selected_option=best_opt,
                probabilities=fwd_cal,
                forward_probabilities=fwd_cal,
                reverse_probabilities=rev_cal,
                order_tvd_drift=0.0,
                is_positionally_stable=True,
                tokens_per_option_budget=58,
            )

    return ClaimVerificationResult(
        claim_text=claim_text,
        cited_doc_ids=citations,
        verdict=verdict,
        guidelines_conflict_prob=round(conflict_prob, 4),
    )