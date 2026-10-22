"""End-to-end 5-Stage ClinGuard-RAG Orchestrator supporting ablation modes."""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Literal, Optional
from clinguard_rag.calibration import TemperatureCalibrator
from clinguard_rag.decision_engine import DecisionBackend, DeterministicEvalBackend
from clinguard_rag.policy import apply_clinical_policy
from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.router import CLINICAL_DOMAIN_OPTIONS, evaluate_stage1_triage
from clinguard_rag.schemas import AuditRecord, ClinicalDisposition, RetrievedChunk
from clinguard_rag.verifier import filter_retrieved_chunks, verify_clinical_claim

logger = logging.getLogger(__name__)

AblationMode = Literal[
    "baseline",
    "retrieval_baseline",
    "raw_system1",
    "debiased_system1",
    "clinguard_full",
]


def synthesize_extractive_claim(chunks: List[RetrievedChunk]) -> str:
    """Deterministic extractive claim synthesizer fallback when no LLM generator is attached."""
    if not chunks:
        return ""
    top = chunks[0].chunk
    return f"{top.text} [{top.doc_id}]"


class ClinGuardPipeline:
    """Orchestrates Stage 1 Triage, Stage 2 Retrieval, Stage 3 Filtering, Stage 4 Verification, and Stage 5 Policy."""

    def __init__(
        self,
        retriever: HybridClinicalRetriever,
        engine: Optional[DecisionBackend] = None,
        mode: AblationMode = "clinguard_full",
        top_k: int = 2,
    ) -> None:
        self.retriever = retriever
        self.engine = engine or DeterministicEvalBackend()
        self.mode: AblationMode = mode
        self.top_k = top_k
        self.calibrator = TemperatureCalibrator(initial_temperature=1.0)

    def fit_calibration(self, calibration_cases: List[Dict[str, Any]]) -> float:
        """Fits post-hoc temperature T on calibration split cases."""
        if self.mode != "clinguard_full":
            self.calibrator.temperature = 1.0
            return 1.0
        dists = [c["raw_model_probs"]["claim_fwd"] for c in calibration_cases]
        labels = [c["claim_label"] for c in calibration_cases]
        fitted_t = self.calibrator.fit(dists, labels)
        logger.info("Fitted post-hoc calibration temperature T=%.4f on %d cases", fitted_t, len(calibration_cases))
        return fitted_t

    def run_case(self, case_record: Dict[str, Any]) -> AuditRecord:
        query = case_record["query"]
        temp = self.calibrator.temperature if self.mode == "clinguard_full" else 1.0
        bidirectional = self.mode in ("debiased_system1", "clinguard_full")

        if "raw_model_probs" in case_record:
            raw_probs = self.engine.predict_case(case_record)
            triage = evaluate_stage1_triage(raw_probs, temperature=temp, bidirectional=bidirectional)
            if self.mode in ("baseline", "retrieval_baseline"):
                all_domains = [d for d in CLINICAL_DOMAIN_OPTIONS if d != "out_of_scope"]
                retrieved = self.retriever.retrieve(query, allowed_domains=all_domains, top_k=self.top_k)
            elif not triage.skip_retrieval:
                retrieved = self.retriever.retrieve(query, allowed_domains=triage.routed_domains, top_k=self.top_k)
            else:
                retrieved = []
        else:
            all_domains = [d for d in CLINICAL_DOMAIN_OPTIONS if d != "out_of_scope"]
            initial_chunks = self.retriever.retrieve(query, allowed_domains=all_domains, top_k=self.top_k)
            raw_probs = self.engine.predict_case(case_record, retrieved_chunks=initial_chunks)
            triage = evaluate_stage1_triage(raw_probs, temperature=temp, bidirectional=bidirectional)
            if self.mode in ("baseline", "retrieval_baseline"):
                retrieved = initial_chunks
            elif not triage.skip_retrieval:
                retrieved = self.retriever.retrieve(query, allowed_domains=triage.routed_domains, top_k=self.top_k)
            else:
                retrieved = []

        candidate_claim = case_record.get("candidate_claim") or synthesize_extractive_claim(retrieved)

        if self.mode == "baseline":
            retrieved_ids = [r.chunk.doc_id for r in retrieved]
            return AuditRecord(
                case_id=case_record["case_id"],
                query=query,
                engine_backend=self.engine.backend_name,
                checkpoint_id=self.engine.checkpoint_id,
                temperature_parameter=1.0,
                triage=triage,
                retrieved_doc_ids=retrieved_ids,
                accepted_doc_ids=retrieved_ids,
                claim_verification=None,
                disposition=ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER,
                policy_rule_triggered="BASELINE_NO_GUARD",
                final_response=candidate_claim,
            )

        if self.mode == "retrieval_baseline":
            retrieved_ids = [r.chunk.doc_id for r in retrieved]
            accepted_ids = [
                r.chunk.doc_id for r in retrieved if r.cosine_score >= 0.22 and r.bm25_score > 0.0
            ]
            if not accepted_ids:
                return AuditRecord(
                    case_id=case_record["case_id"],
                    query=query,
                    engine_backend=self.engine.backend_name,
                    checkpoint_id=self.engine.checkpoint_id,
                    temperature_parameter=1.0,
                    triage=triage,
                    retrieved_doc_ids=retrieved_ids,
                    accepted_doc_ids=[],
                    claim_verification=None,
                    disposition=ClinicalDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE,
                    policy_rule_triggered="RETRIEVAL_SCORE_BELOW_THRESHOLD",
                    final_response="ABSTAIN: Top retrieval similarity score below threshold.",
                )
            return AuditRecord(
                case_id=case_record["case_id"],
                query=query,
                engine_backend=self.engine.backend_name,
                checkpoint_id=self.engine.checkpoint_id,
                temperature_parameter=1.0,
                triage=triage,
                retrieved_doc_ids=retrieved_ids,
                accepted_doc_ids=accepted_ids,
                claim_verification=None,
                disposition=ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER,
                policy_rule_triggered="RETRIEVAL_ONLY_GUARD",
                final_response=candidate_claim,
            )

        if triage.skip_retrieval:
            disposition, rule_id, response_text = apply_clinical_policy(
                triage=triage,
                accepted_doc_ids=[],
                verification=None,
            )
            logger.debug("Case %s halted at Stage 1 (%s -> %s)", case_record["case_id"], rule_id, disposition.value)
            return AuditRecord(
                case_id=case_record["case_id"],
                query=query,
                engine_backend=self.engine.backend_name,
                checkpoint_id=self.engine.checkpoint_id,
                temperature_parameter=temp,
                triage=triage,
                retrieved_doc_ids=[],
                accepted_doc_ids=[],
                claim_verification=None,
                disposition=disposition,
                policy_rule_triggered=rule_id,
                final_response=response_text,
            )

        rel_threshold = 0.60 if self.mode in ("debiased_system1", "clinguard_full") else 0.20
        retrieved, accepted_ids = filter_retrieved_chunks(
            retrieved,
            raw_probs.get("chunk_relevance", {}),
            temperature=temp,
            relevance_threshold=rel_threshold,
        )
        retrieved_ids = [r.chunk.doc_id for r in retrieved]

        verification = None
        if accepted_ids:
            verification = verify_clinical_claim(
                claim_text=candidate_claim,
                raw_probs=raw_probs,
                temperature=temp,
                bidirectional=bidirectional,
                use_three_way=(self.mode != "raw_system1"),
            )

        conflict_thresh = 0.40 if self.mode == "clinguard_full" else 0.99
        disposition, rule_id, response_text = apply_clinical_policy(
            triage=triage,
            accepted_doc_ids=accepted_ids,
            verification=verification,
            support_threshold=0.75,
            conflict_threshold=conflict_thresh,
        )
        logger.debug("Case %s completed (%s -> %s)", case_record["case_id"], rule_id, disposition.value)

        return AuditRecord(
            case_id=case_record["case_id"],
            query=query,
            engine_backend=self.engine.backend_name,
            checkpoint_id=self.engine.checkpoint_id,
            temperature_parameter=temp,
            triage=triage,
            retrieved_doc_ids=retrieved_ids,
            accepted_doc_ids=accepted_ids,
            claim_verification=verification,
            disposition=disposition,
            policy_rule_triggered=rule_id,
            final_response=response_text,
        )
