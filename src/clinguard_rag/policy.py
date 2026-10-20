"""Stage 5 Deterministic Clinical Policy Guard separating System 1 judgment from authority."""

from __future__ import annotations
from typing import List, Optional, Tuple
from clinguard_rag.schemas import (
    ClaimVerificationResult,
    ClinicalDisposition,
    RetrievedChunk,
    TriageDecision,
)


def apply_clinical_policy(
    triage: TriageDecision,
    evaluated_chunks: List[RetrievedChunk],
    claim_result: Optional[ClaimVerificationResult],
    candidate_claim: str,
    support_threshold: float = 0.75,
    conflict_threshold: float = 0.55,
    urgency_emergency_cutoff: float = 2.4,
) -> Tuple[ClinicalDisposition, str, str]:
    """
    Deterministic policy engine that inspects calibrated probabilities, option-order stability,
    evidence sufficiency, and citation validity before authorizing any clinical output.
    """
    # Rule 1: Prompt injection or out-of-scope safety block
    if triage.skip_retrieval:
        return (
            ClinicalDisposition.ABSTAIN_UNSAFE_OR_OUT_OF_SCOPE,
            "RULE_1_SAFETY_OR_OUT_OF_SCOPE_GATE",
            "ABSTAIN: Request flagged as unsafe/prompt-injection or outside supported clinical guideline domains.",
        )

    # Rule 2: Acute medical emergency escalation
    if triage.urgency_expected_score >= urgency_emergency_cutoff:
        return (
            ClinicalDisposition.ESCALATE_ACUTE_EMERGENCY,
            "RULE_2_ACUTE_CLINICAL_EMERGENCY",
            "ESCALATE TO EMERGENCY CARE: High clinical acuity detected. Immediate bedside clinician intervention required.",
        )

    # Rule 3: Post-retrieval evidence sufficiency check
    accepted_chunks = [c for c in evaluated_chunks if c.passed_gate]
    if not accepted_chunks or claim_result is None:
        return (
            ClinicalDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE,
            "RULE_3_INSUFFICIENT_RETRIEVED_EVIDENCE",
            "ABSTAIN: Retrieved guideline passages do not provide sufficient verified evidence to answer this query safely.",
        )

    # Rule 4: Positional instability or contradicted/unsupported claim check
    probs = claim_result.verdict.probabilities
    p_supported = probs.get("supported", 0.0)
    p_contradicted = probs.get("contradicted", 0.0)
    accepted_ids = {c.chunk.doc_id for c in accepted_chunks}
    has_valid_citation = any(cid in accepted_ids for cid in claim_result.cited_doc_ids)

    if (
        not claim_result.verdict.is_positionally_stable
        or claim_result.verdict.selected_option != "supported"
        or p_contradicted >= 0.30
        or p_supported < support_threshold
        or not has_valid_citation
    ):
        return (
            ClinicalDisposition.ESCALATE_UNVERIFIED_OR_CONFLICT,
            "RULE_4_UNVERIFIED_OR_INSTABLE_CLAIM",
            "ESCALATE TO CLINICIAN: Candidate answer failed System-1 claim verification or exhibited positional instability.",
        )

    # Rule 5: Explicit preservation of inter-guideline conflicts
    if claim_result.guidelines_conflict_prob >= conflict_threshold:
        return (
            ClinicalDisposition.SERVE_WITH_CONFLICT_WARNING,
            "RULE_5_PRESERVE_GUIDELINE_CONFLICT",
            f"CLINICAL CONFLICT WARNING (Multiple guidelines disagree): {candidate_claim}",
        )

    # Rule 6: Fully verified, cited, and stable clinical response
    return (
        ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER,
        "RULE_6_VERIFIED_CITED_CLINICAL_ANSWER",
        candidate_claim,
    )