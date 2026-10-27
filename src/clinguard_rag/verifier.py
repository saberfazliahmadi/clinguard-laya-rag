"""Stage 3 & Stage 4: Evidence chunk relevance filtering and 3-way claim verification."""

from __future__ import annotations

import logging
import re
from typing import Any

from clinguard_rag.calibration import apply_temperature_to_binary, apply_temperature_to_distribution
from clinguard_rag.debiasing import combine_bidirectional_probabilities
from clinguard_rag.schemas import ClaimVerification, DebiasedChoiceResult, RetrievedChunk

logger = logging.getLogger(__name__)

CLAIM_OPTIONS: list[str] = ["supported", "contradicted", "not_mentioned"]
# Document IDs: alphanumeric start, then letters/digits/underscore/dot/hyphen (no spaces).
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.\-]*)\]")


def extract_citations(claim_text: str) -> list[str]:
    return CITATION_PATTERN.findall(claim_text)


def filter_retrieved_chunks(
    retrieved: list[RetrievedChunk],
    raw_chunk_relevance: dict[str, float],
    temperature: float = 1.0,
    relevance_threshold: float = 0.60,
) -> list[RetrievedChunk]:
    """Evaluates retrieved passages with calibrated System-1 noul relevance scores (fail-closed on missing keys)."""
    updated: list[RetrievedChunk] = []
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
    raw_probs: dict[str, Any],
    temperature: float = 1.0,
    use_bidirectional: bool = True,
    three_way: bool = True,
    drift_threshold: float = 0.25,
    conflict_temperature: float | None = None,
) -> ClaimVerification:
    """
    Runs 3-way (or 2-way baseline) claim verification with bidirectional order debiasing.
    `temperature` scales the claim heads; `conflict_temperature` (defaults to `temperature`) scales the
    conflict noul head, which is a different question and should only be cooled with its own fitted value.
    Raises ValueError when a required System-1 signal is missing so callers can fail closed.
    """
    missing = [k for k in ("claim_fwd", "claim_rev", "guidelines_conflict_noul") if raw_probs.get(k) is None]
    if missing:
        raise ValueError(f"Missing required System-1 claim verification signals: {missing}")

    fwd_raw = {o: float(raw_probs["claim_fwd"].get(o, 0.0)) for o in CLAIM_OPTIONS}
    rev_raw = {o: float(raw_probs["claim_rev"].get(o, 0.0)) for o in CLAIM_OPTIONS}
    fwd = apply_temperature_to_distribution(fwd_raw, temperature)
    rev = apply_temperature_to_distribution(rev_raw, temperature)

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

    conflict_t = temperature if conflict_temperature is None else conflict_temperature
    conflict_prob = apply_temperature_to_binary(float(raw_probs["guidelines_conflict_noul"]), conflict_t)
    logger.debug("Stage 4 claim verification verdict=%s (stable=%s, conflict=%.4f)", verdict.selected_option, verdict.is_positionally_stable, conflict_prob)
    return ClaimVerification(
        claim_text=claim_text,
        cited_doc_ids=extract_citations(claim_text),
        verdict=verdict,
        guidelines_conflict_prob=round(conflict_prob, 4),
    )


# Alias for evaluation script compatibility
CLAIM_VERIFICATION_OPTIONS_3WAY = CLAIM_OPTIONS
