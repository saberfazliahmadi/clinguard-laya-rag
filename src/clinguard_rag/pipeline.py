"""End-to-End Orchestrator for ClinGuard-RAG and Baseline Ablation Variants."""

from __future__ import annotations
from typing import Any, Dict, List
from clinguard_rag.calibration import TemperatureCalibrator
from clinguard_rag.decision_engine import DeterministicEvalBackend, SystemOneEngine
from clinguard_rag.policy import apply_clinical_policy
from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.router import evaluate_stage1_triage
from clinguard_rag.schemas import AuditRecord, ClinicalDisposition
from clinguard_rag.verifier import filter_retrieved_chunks, verify_clinical_claim


class ClinGuardPipeline:
    """
    Configurable Healthcare RAG pipeline supporting 4 controlled evaluation modes:
      - `baseline`: Standard RAG without a System-1 decision layer
      - `raw_system1`: Uncalibrated Laya (T=1.0), single option order, 2-way claim check
      - `debiased_system1`: Bidirectional option-order debiasing + 3-way claim rubric (T=1.0)
      - `clinguard_full`: Full calibrated (fitted T) + debiased + conflict-preserving ClinGuard-RAG
    """

    def __init__(
        self,
        retriever: HybridClinicalRetriever,
        engine: SystemOneEngine | None = None,
        temperature: float = 1.0,
        mode: str = "clinguard_full",
    ) -> None:
        self.retriever = retriever
        self.engine = engine or DeterministicEvalBackend()
        self.temperature = temperature
        self.mode = mode

    def fit_calibration(self, calibration_cases: List[Dict[str, Any]]) -> float:
        """Fits post-hoc temperature scaling on held-out calibration cases."""
        calibrator = TemperatureCalibrator()
        dists = [c["raw_model_probs"]["claim_fwd"] for c in calibration_cases]
        labels = [c["claim_label"] for c in calibration_cases]
        self.temperature = calibrator.fit(dists, labels)
        return self.temperature

    def run_case(self, case_record: Dict[str, Any]) -> AuditRecord:
        query = case_record["query"]
        candidate_claim = case_record["candidate_claim"]
        raw_probs = self.engine.predict_case(case_record)

        if self.mode == "baseline":
            retrieved = self.retriever.retrieve(query, allowed_domains=None, top_k=2)
            triage = evaluate_stage1_triage(
                raw_probs, temperature=1.0, use_bidirectional_debiasing=False
            )
            return AuditRecord(
                case_id=case_record["case_id"],
                query=query,
                engine_backend="none-baseline-rag",
                checkpoint_id="none",
                temperature_parameter=1.0,
                triage=triage,
                retrieved_doc_ids=[r.chunk.doc_id for r in retrieved],
                accepted_doc_ids=[r.chunk.doc_id for r in retrieved],
                claim_verification=None,
                disposition=ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER,
                policy_rule_triggered="BASELINE_NO_GUARD",
                final_response=candidate_claim,
            )

        use_debias = self.mode in ("debiased_system1", "clinguard_full")
        use_3way = self.mode in ("debiased_system1", "clinguard_full")
        temp = self.temperature if self.mode == "clinguard_full" else 1.0
        # Hold support_threshold = 0.75 constant across all System-1 variants for a controlled ablation
        support_thresh = 0.75
        relevance_thresh = 0.60 if self.mode in ("debiased_system1", "clinguard_full") else 0.20
        conflict_thresh = 0.55 if self.mode == "clinguard_full" else 0.99

        triage = evaluate_stage1_triage(
            raw_probs,
            temperature=temp,
            use_bidirectional_debiasing=use_debias,
        )

        if triage.skip_retrieval:
            retrieved = []
            evaluated_chunks = []
            claim_res = None
        else:
            retrieved = self.retriever.retrieve(
                query, allowed_domains=triage.routed_domains, top_k=2
            )
            evaluated_chunks = filter_retrieved_chunks(
                retrieved,
                raw_probs["chunk_relevance"],
                temperature=temp,
                relevance_threshold=relevance_thresh,
            )
            claim_res = verify_clinical_claim(
                candidate_claim,
                raw_probs,
                temperature=temp,
                use_bidirectional_debiasing=use_debias,
                use_three_way_rubric=use_3way,
            )

        disposition, rule_id, final_text = apply_clinical_policy(
            triage=triage,
            evaluated_chunks=evaluated_chunks,
            claim_result=claim_res,
            candidate_claim=candidate_claim,
            support_threshold=support_thresh,
            conflict_threshold=conflict_thresh,
        )

        return AuditRecord(
            case_id=case_record["case_id"],
            query=query,
            engine_backend=self.engine.backend_name,
            checkpoint_id=self.engine.checkpoint_id,
            temperature_parameter=temp,
            triage=triage,
            retrieved_doc_ids=[r.chunk.doc_id for r in retrieved],
            accepted_doc_ids=[r.chunk.doc_id for r in evaluated_chunks if r.passed_gate],
            claim_verification=claim_res,
            disposition=disposition,
            policy_rule_triggered=rule_id,
            final_response=final_text,
        )