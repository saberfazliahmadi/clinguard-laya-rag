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
    source_org: str = ""
    publication_year: int = 2025
    domain: str
    text: str
    clinical_keywords: List[str] = Field(default_factory=list)


# Alias ClinicalChunk to GuidelineChunk for seamless compatibility across modules
ClinicalChunk = GuidelineChunk


class RetrievedChunk(BaseModel):
    chunk: GuidelineChunk
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    bm25_rank: int = 0
    dense_rank: int = 0
    relevance_prob: float = 0.0
    system1_relevance_prob: float = 0.0
    passed_gate: bool = False


class DebiasedChoiceResult(BaseModel):
    selected_option: str
    probabilities: Dict[str, float]
    forward_probabilities: Dict[str, float] = Field(default_factory=dict)
    reverse_probabilities: Dict[str, float] = Field(default_factory=dict)
    order_tvd_drift: float = 0.0
    is_positionally_stable: bool = True
    tokens_per_option_budget: int = 35


# Alias DomainChoiceDecision to DebiasedChoiceResult
DomainChoiceDecision = DebiasedChoiceResult


class TriageDecision(BaseModel):
    prompt_attack_prob: float = 0.0
    urgency_expected_score: float = 0.0
    urgency_probabilities: List[float] = Field(default_factory=lambda: [0.7, 0.2, 0.08, 0.02])
    domain_choice: DebiasedChoiceResult
    routed_domains: List[str] = Field(default_factory=list)
    skip_retrieval: bool = False


class ClaimVerificationResult(BaseModel):
    claim_text: str
    cited_doc_ids: List[str] = Field(default_factory=list)
    verdict: DebiasedChoiceResult
    guidelines_conflict_prob: float = 0.0


# Alias ClaimVerification to ClaimVerificationResult
ClaimVerification = ClaimVerificationResult


class AuditRecord(BaseModel):
    """Complete decision provenance record for clinical governance and observability."""
    case_id: str
    query: str
    engine_backend: str
    checkpoint_id: str
    temperature_parameter: float
    triage: TriageDecision
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    accepted_doc_ids: List[str] = Field(default_factory=list)
    claim_verification: Optional[ClaimVerificationResult] = None
    disposition: ClinicalDisposition
    policy_rule_triggered: str
    final_response: str
