"""End-to-end 5-Stage ClinGuard-RAG Orchestrator supporting 5 ablation modes."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from clinguard_rag.calibration import TemperatureCalibrator
from clinguard_rag.decision_engine import DecisionBackend, DeterministicEvalBackend
from clinguard_rag.policy import apply_clinical_policy
from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.router import run_stage1_triage
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

_STOPWORDS = {
    "what", "when", "where", "which", "should", "with", "from", "that", "this",
    "dose", "daily", "patient", "adult", "year", "recommended", "oral", "after",
}


def _query_lexical_overlap(query: str, chunk: RetrievedChunk) -> float:
    """Computes fraction of informative query terms present in the candidate chunk."""
    q_tokens = {
        t for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) >= 4 and t not in _STOPWORDS
    }
    if not q_tokens:
        return 1.0
    doc_tokens = set(
        re.findall(r"[a-z0-9]+", f"{chunk.chunk.title} {chunk.chunk.text}".lower())
    )
    return len(q_tokens & doc_tokens) / len(q_tokens)


def synthesize_extractive_claim(chunks: list[RetrievedChunk]) -> str:
    """Deterministic extractive claim synthesizer when candidate_claim is not pre-supplied."""
    if not chunks:
        return "No supporting clinical guideline passage available."
    top = chunks[0].chunk
    return f"According to {top.title}, {top.text} [{top.doc_id}]"


class ClinGuardPipeline:
    """
    Coordinates the 5-stage System-1 Guarded Clinical RAG architecture:
      Stage 1: Pre-Retrieval Safety, Urgency & Domain Triage
      Stage 2: Hybrid BM25 + TF-IDF Cosine Retrieval with RRF
      Stage 3: Calibrated System-1 Evidence Relevance Gate
      Stage 4: Bidirectional 3-Way Claim & Citation Verification
      Stage 5: Deterministic Clinical Policy Engine
    """

    def __init__(
        self,
        retriever: HybridClinicalRetriever,
        engine: DecisionBackend | None = None,
        mode: str = "clinguard_full",
    ) -> None:
        if mode not in (
            "baseline",
            "retrieval_baseline",
            "raw_system1",
            "debiased_system1",
            "clinguard_full",
        ):
            raise ValueError(f"Unsupported pipeline mode: {mode}")
        self.retriever = retriever
        self.engine = engine or DeterministicEvalBackend()
        self.mode = mode
        self.calibrator = TemperatureCalibrator()
        self.retrieval_overlap_threshold: float = 0.25

    def fit_calibration(self, calibration_cases: list[dict[str, Any]]) -> float:
        """
        Fits post-hoc temperature T on calibration split for clinguard_full,
        or tunes the lexical-overlap threshold on calibration split for retrieval_baseline.
        """
        if self.mode == "retrieval_baseline":
            best_thresh = 0.25
            best_acc = -1.0
            for candidate_t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
                correct = 0
                for c in calibration_cases:
                    chunks = self.retriever.retrieve(query=c["query"], allowed_domains=None, top_k=2)
                    accepted = [
                        r.chunk.doc_id
                        for r in chunks
                        if _query_lexical_overlap(c["query"], r) >= candidate_t
                    ]
                    pred_disp = (
                        ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER.value
                        if accepted
                        else ClinicalDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE.value
                    )
                    if pred_disp == c["expected_disposition"]:
                        correct += 1
                acc = correct / max(1, len(calibration_cases))
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = candidate_t
            self.retrieval_overlap_threshold = best_thresh
            logger.info(
                "Fitted retrieval_baseline overlap threshold=%.2f (calibration accuracy=%.2f)",
                best_thresh,
                best_acc,
            )
            self.calibrator.temperature = 1.0
            return 1.0

        if self.mode != "clinguard_full":
            self.calibrator.temperature = 1.0
            return 1.0

        forward_dists: list[dict[str, float]] = []
        labels: list[str] = []
        for c in calibration_cases:
            probs = self.engine.predict_case(c)
            forward_dists.append(probs["claim_fwd"])
            labels.append(c["claim_label"])
        fitted_t = self.calibrator.fit(forward_dists, labels)
        logger.info("Fitted post-hoc temperature T=%.2f on %d calibration cases", fitted_t, len(calibration_cases))
        return fitted_t

    def run_case(self, case_record: dict[str, Any]) -> AuditRecord:
        case_id = str(case_record.get("case_id", "LIVE-01"))
        query = str(case_record["query"])

        prelim_chunks = self.retriever.retrieve(query=query, allowed_domains=None, top_k=2)
        raw_probs = self.engine.predict_case(case_record, retrieved_chunks=prelim_chunks)

        if self.mode == "retrieval_baseline":
            triage = run_stage1_triage(raw_probs, temperature=1.0, use_bidirectional=False)
            retrieved_ids = [r.chunk.doc_id for r in prelim_chunks]
            accepted_ids = [
                r.chunk.doc_id
                for r in prelim_chunks
                if _query_lexical_overlap(query, r) >= self.retrieval_overlap_threshold
            ]
            candidate_claim = case_record.get("candidate_claim") or synthesize_extractive_claim(prelim_chunks)
            if not accepted_ids:
                logger.info("Case %s abstained under retrieval_baseline (overlap < %.2f)", case_id, self.retrieval_overlap_threshold)
                return AuditRecord(
                    case_id=case_id,
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
                    final_response="ABSTAIN: Top retrieval lexical overlap score below calibrated threshold.",
                )
            return AuditRecord(
                case_id=case_id,
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
                final_response=str(candidate_claim),
            )

        if self.mode == "baseline":
            triage = run_stage1_triage(raw_probs, temperature=1.0, use_bidirectional=False)
            retrieved_ids = [r.chunk.doc_id for r in prelim_chunks]
            candidate_claim = case_record.get("candidate_claim") or synthesize_extractive_claim(prelim_chunks)
            return AuditRecord(
                case_id=case_id,
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
                final_response=str(candidate_claim),
            )

        use_bidir = self.mode in ("debiased_system1", "clinguard_full")
        use_three_way = self.mode in ("debiased_system1", "clinguard_full")
        temp = self.calibrator.temperature if self.mode == "clinguard_full" else 1.0

        triage = run_stage1_triage(
            raw_probs=raw_probs,
            temperature=temp,
            use_bidirectional=use_bidir,
        )

        if triage.skip_retrieval:
            disp, rule, final_msg = apply_clinical_policy(
                triage=triage,
                accepted_doc_ids=[],
                verification=None,
                preserve_conflicts=(self.mode == "clinguard_full"),
            )
            logger.info("Case %s halted at Stage 1 (%s)", case_id, rule)
            return AuditRecord(
                case_id=case_id,
                query=query,
                engine_backend=self.engine.backend_name,
                checkpoint_id=self.engine.checkpoint_id,
                temperature_parameter=temp,
                triage=triage,
                retrieved_doc_ids=[],
                accepted_doc_ids=[],
                claim_verification=None,
                disposition=disp,
                policy_rule_triggered=rule,
                final_response=final_msg,
            )

        retrieved_chunks = self.retriever.retrieve(
            query=query,
            allowed_domains=triage.routed_domains,
            top_k=2,
        )
        retrieved_ids = [r.chunk.doc_id for r in retrieved_chunks]

        rel_threshold = 0.60 if self.mode in ("debiased_system1", "clinguard_full") else 0.20
        _, accepted_ids = filter_retrieved_chunks(
            retrieved=retrieved_chunks,
            chunk_relevance_probs=raw_probs.get("chunk_relevance", {}),
            temperature=temp,
            relevance_threshold=rel_threshold,
        )

        if not accepted_ids:
            disp, rule, final_msg = apply_clinical_policy(
                triage=triage,
                accepted_doc_ids=[],
                verification=None,
                preserve_conflicts=(self.mode == "clinguard_full"),
            )
            return AuditRecord(
                case_id=case_id,
                query=query,
                engine_backend=self.engine.backend_name,
                checkpoint_id=self.engine.checkpoint_id,
                temperature_parameter=temp,
                triage=triage,
                retrieved_doc_ids=retrieved_ids,
                accepted_doc_ids=[],
                claim_verification=None,
                disposition=disp,
                policy_rule_triggered=rule,
                final_response=final_msg,
            )

        accepted_chunks = [r for r in retrieved_chunks if r.chunk.doc_id in accepted_ids]
        candidate_claim = case_record.get("candidate_claim") or synthesize_extractive_claim(accepted_chunks)

        verification = verify_clinical_claim(
            claim_text=str(candidate_claim),
            raw_probs=raw_probs,
            temperature=temp,
            use_bidirectional=use_bidir,
            use_three_way=use_three_way,
        )

        disp, rule, final_msg = apply_clinical_policy(
            triage=triage,
            accepted_doc_ids=accepted_ids,
            verification=verification,
            support_threshold=0.75,
            conflict_threshold=0.40,
            preserve_conflicts=(self.mode == "clinguard_full"),
        )

        return AuditRecord(
            case_id=case_id,
            query=query,
            engine_backend=self.engine.backend_name,
            checkpoint_id=self.engine.checkpoint_id,
            temperature_parameter=temp,
            triage=triage,
            retrieved_doc_ids=retrieved_ids,
            accepted_doc_ids=accepted_ids,
            claim_verification=verification,
            disposition=disp,
            policy_rule_triggered=rule,
            final_response=final_msg,
        )
