"""Stage 5: Deterministic Clinical Policy Engine separating System-1 judgment from execution authority."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from clinguard_rag.schemas import ClaimVerification, ClinicalDisposition, TriageDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyConfig:
    """Single source of truth for every threshold shared by the router, evidence gate, and policy engine."""

    attack_threshold: float = 0.65
    out_of_scope_threshold: float = 0.60
    single_kb_threshold: float = 0.55
    emergency_urgency_threshold: float = 2.40
    relevance_threshold: float = 0.60
    drift_threshold: float = 0.25
    support_threshold: float = 0.75
    contradiction_threshold: float = 0.30
    conflict_threshold: float = 0.40

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_POLICY_CONFIG = PolicyConfig()

_SAFETY_GATE_RESPONSE = (
    ClinicalDisposition.ABSTAIN_UNSAFE_OR_OUT_OF_SCOPE,
    "RULE_1_SAFETY_OR_OUT_OF_SCOPE_GATE",
    "ABSTAIN: Query blocked by Stage-1 clinical safety and scope guard.",
)


def apply_clinical_policy(
    triage: TriageDecision,
    accepted_doc_ids: list[str],
    verification: ClaimVerification | None,
    support_threshold: float | None = None,
    conflict_threshold: float | None = None,
    enable_conflict_warning: bool = True,
    config: PolicyConfig | None = None,
) -> tuple[ClinicalDisposition, str, str]:
    """Maps calibrated System-1 outputs to a deterministic clinical disposition and response string."""
    cfg = config or DEFAULT_POLICY_CONFIG
    support_t = cfg.support_threshold if support_threshold is None else support_threshold
    conflict_t = cfg.conflict_threshold if conflict_threshold is None else conflict_threshold
    out_of_scope_prob = triage.domain_choice.probabilities.get("out_of_scope", 0.0)

    # Rule 1a: Adversarial prompt attack / unsafe override gate (highest security priority)
    logger.debug("Evaluating clinical policy rules (accepted_docs=%d)", len(accepted_doc_ids))
    if triage.prompt_attack_prob >= cfg.attack_threshold:
        return _SAFETY_GATE_RESPONSE

    # Rule 2: Acute Clinical Emergency Escalation (checked before out_of_scope so emergencies never abstain)
    if triage.urgency_expected_score >= cfg.emergency_urgency_threshold:
        return (
            ClinicalDisposition.ESCALATE_ACUTE_EMERGENCY,
            "RULE_2_ACUTE_CLINICAL_EMERGENCY",
            f"EMERGENCY ESCALATION: High clinical acuity detected (score >= {cfg.emergency_urgency_threshold:.2f}/3.0). "
            "Do not wait for RAG synthesis; initiate immediate emergency protocol.",
        )

    # Rule 1b: Out-of-scope non-clinical query gate. Stage 1 also halts retrieval if the router was
    # configured differently, so a skipped query can never fall through to the evidence rules.
    if out_of_scope_prob >= cfg.out_of_scope_threshold or triage.skip_retrieval:
        return _SAFETY_GATE_RESPONSE

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
    # Every cited ID must be an accepted passage; a single hallucinated citation fails grounding.
    cited = set(verification.cited_doc_ids)
    has_valid_citation = bool(cited) and cited <= set(accepted_doc_ids)

    # Rule 4: Positional Instability, Contradiction, Low Support, or Missing/Ungrounded Citation
    if (
        not verdict.is_positionally_stable
        or verdict.selected_option != "supported"
        or prob_contradicted >= cfg.contradiction_threshold
        or prob_supported < support_t
        or not has_valid_citation
    ):
        return (
            ClinicalDisposition.ESCALATE_UNVERIFIED_OR_CONFLICT,
            "RULE_4_UNVERIFIED_OR_INSTABLE_CLAIM",
            "ESCALATE TO CLINICIAN: Candidate claim failed calibrated System-1 verification or exhibited positional order instability.",
        )

    # Rule 5: Explicit Multi-Guideline Conflict Preservation
    if enable_conflict_warning and verification.guidelines_conflict_prob >= conflict_t:
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
