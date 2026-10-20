"""Typed data schemas for System 1 decisions, clinical retrieval, and audit records."""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DecisionPrimitive(str, Enum):
    """Three core System 1 decision primitives shared by Jev and Laya."""
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


class ClinicalDisposition(str, Enum):
    """Deterministic clinical policy actions separated from model judgment."""
    SERVE_VERIFIED_CITED_ANSWER = "SERVE_VERIFIED_CITED_ANSWER"
    SERVE_WITH_CONFLICT_WARNING = "SERVE_WITH_CONFLICT_WARNING"
    ESCALATE_ACUTE_EMERGENCY = "ESCALATE_ACUTE_EMERGENCY"
    ESCALATE_UNVERIFIED_OR_CONFLICT = "ESCALATE_UNVERIFIED_OR_CONFLICT"
    ABSTAIN_INSUFFICIENT_EVIDENCE = "ABSTAIN_INSUFFICIENT_EVIDENCE"
    ABSTAIN_UNSAFE_OR_OUT_OF_SCOPE = "ABSTAIN_UNSAFE_OR_OUT_OF_SCOPE"


class GuidelineChunk(BaseModel):
    doc_id: str
    title: str
    source_org: str
    publication_year: int
    domain: str
    text: str
    clinical_keywords: List[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk: GuidelineChunk
    bm25_score: float
    dense_score: float
    rrf_score: float
    relevance_prob: float = 0.0
    passed_gate: bool = False


class DebiasedChoiceResult(BaseModel):
    selected_option: str
    probabilities: Dict[str, float]
    forward_probabilities: Dict[str, float]
    reverse_probabilities: Dict[str, float]
    order_tvd_drift: float
    is_positionally_stable: bool
    tokens_per_option_budget: int


class TriageDecision(BaseModel):
    prompt_attack_prob: float
    urgency_expected_score: float
    urgency_probabilities: List[float]
    domain_choice: DebiasedChoiceResult
    routed_domains: List[str]
    skip_retrieval: bool


class ClaimVerificationResult(BaseModel):
    claim_text: str
    cited_doc_ids: List[str]
    verdict: DebiasedChoiceResult
    guidelines_conflict_prob: float


class AuditRecord(BaseModel):
    """Complete decision provenance record for clinical governance and observability."""
    case_id: str
    query: str
    engine_backend: str
    checkpoint_id: str
    temperature_parameter: float
    triage: TriageDecision
    retrieved_doc_ids: List[str]
    accepted_doc_ids: List[str]
    claim_verification: Optional[ClaimVerificationResult] = None
    disposition: ClinicalDisposition
    policy_rule_triggered: str
    final_response: str