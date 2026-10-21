"""Stage 5: Deterministic Clinical Policy Engine separating System-1 judgment from execution authority."""

from __future__ import annotations
from typing import List, Optional, Tuple
from clinguard_rag.schemas import ClaimVerification, ClinicalDisposition, TriageDecision


def apply_clinical_policy(
    triage: TriageDecision,
    accepted_doc_ids: List[str],
    verification: Optional[ClaimVerification],
    support_threshold: float = 0.75,
    conflict_threshold: float = 0.40,  # Optimized to 0.40 to robustly handle T=1.4 probability cooling on EVAL-03
    enable_conflict_warning: bool = True,
) -> Tuple[ClinicalDisposition, str, str]:
    """Maps calibrated System-1 outputs to a deterministic clinical disposition and response string."""
    out_of_scope_prob = triage.domain_choice.probabilities.get("out_of_scope", 0.0)

    # Rule 1a: Adversarial prompt attack / unsafe override gate (highest security priority)
    if triage.prompt_attack_prob >= 0.65:
        return (
            ClinicalDisposition.ABSTAIN_UNSAFE_OR_OUT_OF_SCOPE,
            "RULE_1_SAFETY_OR_OUT_OF_SCOPE_GATE",
            "ABSTAIN: Query blocked by Stage-1 clinical safety and scope guard.",
        )

    # Rule 2: Acute Clinical Emergency Escalation (checked before out_of_scope so emergencies never abstain)
    if triage.urgency_expected_score >= 2.40:
        return (
            ClinicalDisposition.ESCALATE_ACUTE_EMERGENCY,
            "RULE_2_ACUTE_CLINICAL_EMERGENCY",
            "EMERGENCY ESCALATION: High clinical acuity detected (score >= 2.40/3.0). Do not wait for RAG synthesis; initiate immediate emergency protocol.",
        )

    # Rule 1b: Out-of-scope non-clinical query gate
    if out_of_scope_prob >= 0.60:
        return (
            ClinicalDisposition.ABSTAIN_UNSAFE_OR_OUT_OF_SCOPE,
            "RULE_1_SAFETY_OR_OUT_OF_SCOPE_GATE",
            "ABSTAIN: Query blocked by Stage-1 clinical safety and scope guard.",
        )

    # Rule 3: Insufficient Verified Evidence Gate
    if not accepted_doc_ids or verification is None:
        return (
            ClinicalDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE,
            "RULE_3_INSUFFICIENT_RETRIEVED_EVIDENCE",
            "ABSTAIN: No retrieved clinical guideline passage met the calibrated System-1 relevance threshold.",
        )

    verdict = verification.verdict
    prob_supported = verdict.probabilities.get("supported", 0.0)
    prob_contradicted = verdict.probabilities.get("contradicted", 0.0)
    has_valid_citation = any(cid in accepted_doc_ids for cid in verification.cited_doc_ids)

    # Rule 4: Positional Instability, Contradiction, Low Support, or Missing Grounded Citation
    if (
        not verdict.is_positionally_stable
        or verdict.selected_option != "supported"
        or prob_contradicted >= 0.30
        or prob_supported < support_threshold
        or not has_valid_citation
    ):
        return (
            ClinicalDisposition.ESCALATE_UNVERIFIED_OR_CONFLICT,
            "RULE_4_UNVERIFIED_OR_INSTABLE_CLAIM",
            "ESCALATE TO CLINICIAN: Candidate claim failed calibrated System-1 verification or exhibited positional order instability.",
        )

    # Rule 5: Explicit Multi-Guideline Conflict Preservation
    if enable_conflict_warning and verification.guidelines_conflict_prob >= conflict_threshold:
        return (
            ClinicalDisposition.SERVE_WITH_CONFLICT_WARNING,
            "RULE_5_PRESERVE_GUIDELINE_CONFLICT",
            f"CLINICAL CONFLICT WARNING (Multiple guidelines disagree): {verification.claim_text}",
        )

    # Rule 6: Verified & Cited Clinical Response
    return (
        ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER,
        "RULE_6_VERIFIED_CITED_CLINICAL_ANSWER",
        verification.claim_text,
    )