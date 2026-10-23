"""End-to-end 5-Stage ClinGuard-RAG Orchestrator supporting ablation modes."""

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
    "for", "the", "and", "are", "can", "how",
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


PipelineMode = Literal["baseline", "retrieval_baseline", "raw_system1", "debiased_system1", "clinguard_full"]


def synthesize_extractive_claim(accepted_chunks: list[RetrievedChunk]) -> str:
    """Deterministic extractive fallback claim generator when no fixture candidate_claim is provided."""
    if not accepted_chunks:
        return ""
    top = accepted_chunks[0].chunk
    return f"{top.text} [{top.doc_id}]"


class ClinGuardPipeline:
    """5-stage clinical RAG pipeline with configurable ablation modes."""

    def __init__(
        self,
        retriever: HybridClinicalRetriever,
        engine: DecisionBackend | None = None,
        mode: PipelineMode = "clinguard_full",
        temperature: float = 1.0,
    ) -> None:
        self.retriever = retriever
        self.engine = engine or DeterministicEvalBackend()
        self.mode = mode
        self.calibrator = TemperatureCalibrator(temperature=temperature)

    def fit_calibration(self, calibration_cases: list[dict[str, Any]]) -> float:
        """
        Fits post-hoc temperature T on the calibration split for clinguard_full,
        or selects the lexical-overlap threshold on the calibration split for retrieval_baseline.
        """
        if self.mode == "retrieval_baseline":
            best_t = 0.25
            best_acc = -1.0
            for cand_t in (0.25, 0.30, 0.35, 0.40, 0.50):
                hits = 0
                for c in calibration_cases:
                    chunks = self.retriever.retrieve(query=c["query"], allowed_domains=None, top_k=2)
                    acc = [r for r in chunks if _query_lexical_overlap(c["query"], r) >= cand_t]
                    want_serve = c["expected_disposition"] in (
                        ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER.value,
                        ClinicalDisposition.SERVE_WITH_CONFLICT_WARNING.value,
                    )
                    if bool(acc) == want_serve:
                        hits += 1
                score = hits / max(1, len(calibration_cases))
                if score > best_acc:
                    best_acc = score
                    best_t = cand_t
            self.retrieval_overlap_threshold = best_t
            logger.info(
                "Fitted retrieval_baseline overlap threshold=%.2f on %d calibration cases (cal_acc=%.2f)",
                best_t,
                len(calibration_cases),
                best_acc,
            )
            self.calibrator.temperature = 1.0
            return 1.0

        if self.mode != "clinguard_full":
            self.calibrator.temperature = 1.0
            return 1.0

        forward_dists = []
        labels = []
        for c in calibration_cases:
            probs = self.engine.predict_case(c)
            forward_dists.append(probs["claim_fwd"])
            labels.append(c["claim_label"])
        fitted_t = self.calibrator.fit(forward_dists, labels)
        logger.info("Fitted post-hoc temperature T=%.2f on %d calibration cases", fitted_t, len(calibration_cases))
        return fitted_t

    def run_case(self, case_record: dict[str, Any]) -> AuditRecord:
        case_id = case_record.get("case_id", "LIVE-01")
        query = case_record["query"]
        logger.debug("Running case %s in mode %s", case_id, self.mode)

        # Pre-retrieve candidate chunks for live backends that do not include raw_model_probs
        prelim_chunks = self.retriever.retrieve(query=query, allowed_domains=None, top_k=2)
        raw_probs = self.engine.predict_case(case_record, retrieved_chunks=prelim_chunks)

        if self.mode == "retrieval_baseline":
            triage = run_stage1_triage(raw_probs, temperature=1.0, use_bidirectional=False)
            retrieved_ids = [r.chunk.doc_id for r in prelim_chunks]
            accepted_ids = [
                r.chunk.doc_id for r in prelim_chunks if _query_lexical_overlap(query, r) >= self.retrieval_overlap_threshold
            ]
            candidate_claim = case_record.get("candidate_claim") or synthesize_extractive_claim(prelim_chunks)
            if not accepted_ids:
                logger.info("Case %s abstained under retrieval_baseline (0 chunks accepted)", case_id)
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
                    final_response="ABSTAIN: Top retrieval similarity score below threshold.",
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
                final_response=candidate_claim,
            )

        if self.mode == "baseline":
            retrieved = self.retriever.retrieve(query=query, allowed_domains=None, top_k=2)
            candidate_claim = case_record.get("candidate_claim") or synthesize_extractive_claim(retrieved)
            triage = run_stage1_triage(raw_probs, temperature=1.0, use_bidirectional=False)
            return AuditRecord(
                case_id=case_id,
                query=query,
                engine_backend=self.engine.backend_name,
                checkpoint_id=self.engine.checkpoint_id,
                temperature_parameter=1.0,
                triage=triage,
                retrieved_doc_ids=[r.chunk.doc_id for r in retrieved],
                accepted_doc_ids=[r.chunk.doc_id for r in retrieved],
                claim_verification=None,
                disposition=ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER,
                policy_rule_triggered="BASELINE_NO_GUARD",
                final_response=candidate_claim,
            )

        use_bi = self.mode in ("debiased_system1", "clinguard_full")
        three_way = self.mode in ("debiased_system1", "clinguard_full")
        temp = self.calibrator.temperature if self.mode == "clinguard_full" else 1.0
        rel_thresh = 0.60 if self.mode in ("debiased_system1", "clinguard_full") else 0.20
        enable_conflict = self.mode == "clinguard_full"

        # Stage 1: Triage & Routing
        triage = run_stage1_triage(raw_probs, temperature=temp, use_bidirectional=use_bi)

        if triage.skip_retrieval:
            logger.info("Case %s halted at Stage 1 triage (skip_retrieval=True)", case_id)
            disp, rule, msg = apply_clinical_policy(
                triage=triage,
                accepted_doc_ids=[],
                verification=None,
                enable_conflict_warning=enable_conflict,
            )
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
                final_response=msg,
            )

        # Stage 2: Hybrid Retrieval across routed domains
        retrieved = self.retriever.retrieve(
            query=query, allowed_domains=triage.routed_domains, top_k=2
        )

        # Stage 3: Evidence Gate
        gated = filter_retrieved_chunks(
            retrieved=retrieved,
            raw_chunk_relevance=raw_probs.get("chunk_relevance", {}),
            temperature=temp,
            relevance_threshold=rel_thresh,
        )
        accepted_chunks = [r for r in gated if r.passed_gate]
        accepted_ids = [r.chunk.doc_id for r in accepted_chunks]

        # Stage 4: Claim Verification
        verification = None
        if accepted_ids:
            candidate_claim = case_record.get("candidate_claim") or synthesize_extractive_claim(accepted_chunks)
            verification = verify_clinical_claim(
                claim_text=candidate_claim,
                raw_probs=raw_probs,
                temperature=temp,
                use_bidirectional=use_bi,
                three_way=three_way,
            )

        # Stage 5: Deterministic Policy Guard
        disp, rule, msg = apply_clinical_policy(
            triage=triage,
            accepted_doc_ids=accepted_ids,
            verification=verification,
            support_threshold=0.75,
            conflict_threshold=0.40,
            enable_conflict_warning=enable_conflict,
        )

        return AuditRecord(
            case_id=case_id,
            query=query,
            engine_backend=self.engine.backend_name,
            checkpoint_id=self.engine.checkpoint_id,
            temperature_parameter=temp,
            triage=triage,
            retrieved_doc_ids=[r.chunk.doc_id for r in retrieved],
            accepted_doc_ids=accepted_ids,
            claim_verification=verification,
            disposition=disp,
            policy_rule_triggered=rule,
            final_response=msg,
        )
