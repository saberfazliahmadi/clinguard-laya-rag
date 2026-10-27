"""End-to-end 5-Stage ClinGuard-RAG Orchestrator supporting ablation modes."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, get_args

from clinguard_rag.calibration import TemperatureCalibrator
from clinguard_rag.decision_engine import DecisionBackend, DeterministicEvalBackend, redact_phi_hook
from clinguard_rag.policy import DEFAULT_POLICY_CONFIG, PolicyConfig, apply_clinical_policy
from clinguard_rag.retriever import HybridClinicalRetriever
from clinguard_rag.router import run_stage1_triage
from clinguard_rag.schemas import (
    AuditRecord,
    ClaimVerification,
    ClinicalDisposition,
    RetrievedChunk,
    TriageDecision,
)
from clinguard_rag.verifier import filter_retrieved_chunks, verify_clinical_claim

logger = logging.getLogger(__name__)

PipelineMode = Literal["baseline", "retrieval_baseline", "raw_system1", "debiased_system1", "clinguard_full"]
# Backwards-compatible alias
AblationMode = PipelineMode
VALID_MODES: tuple[str, ...] = get_args(PipelineMode)

FAIL_CLOSED_RULE = "SYSTEM1_BACKEND_FAILURE_FAIL_CLOSED"
FAIL_CLOSED_RESPONSE = (
    "ESCALATE TO CLINICIAN: The System-1 decision layer was unavailable or returned incomplete signals; "
    "no automated answer was produced."
)

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
        policy_config: PolicyConfig | None = None,
        top_k: int = 2,
    ) -> None:
        if mode not in VALID_MODES:
            # A typo must never silently downgrade to a weaker guard configuration.
            raise ValueError(f"Unknown pipeline mode '{mode}'. Valid modes: {', '.join(VALID_MODES)}.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        self.retriever = retriever
        self.engine = engine or DeterministicEvalBackend()
        self.mode = mode
        self.top_k = top_k
        self.policy_config = policy_config or DEFAULT_POLICY_CONFIG
        self.calibrator = TemperatureCalibrator(temperature=temperature)
        self.retrieval_overlap_threshold: float = 0.0

    def head_temperatures(self) -> dict[str, float]:
        """
        Per-head temperatures. Only the claim head has labelled calibration data, so only it is cooled;
        safety heads (attack/urgency/domain), relevance, and conflict stay at T=1.0 because applying the
        claim-head temperature to them would push e.g. P(attack)=0.70 below the 0.65 safety gate.
        """
        claim_t = self.calibrator.temperature if self.mode == "clinguard_full" else 1.0
        return {"claim": claim_t, "triage": 1.0, "relevance": 1.0, "conflict": 1.0}

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
                    chunks = self.retriever.retrieve(query=c["query"], allowed_domains=None, top_k=self.top_k)
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
        temps = self.head_temperatures()
        for c in calibration_cases:
            # Judge each calibration claim against the same evidence the runtime path would give it
            # (routed retrieval + relevance gate); triage-halted cases are judged with no evidence.
            triage = self._triage(self.engine.predict_triage(c), temps, use_bi=True)
            evidence: list[RetrievedChunk] = []
            if not self._halts_at_stage1(triage):
                _, evidence = self._retrieve_and_gate(c, str(c["query"]), triage, temps, self.policy_config.relevance_threshold)
            probs = self.engine.predict_claim(c, str(c["candidate_claim"]), evidence)
            forward_dists.append(probs["claim_fwd"])
            labels.append(c["claim_label"])
        fitted_t = self.calibrator.fit(forward_dists, labels)
        logger.info("Fitted post-hoc claim-head temperature T=%.2f on %d calibration cases", fitted_t, len(calibration_cases))
        return fitted_t

    def _audit(
        self,
        case_id: str,
        query: str,
        temperature: float,
        triage: TriageDecision | None,
        retrieved_ids: list[str],
        accepted_ids: list[str],
        verification: ClaimVerification | None,
        disposition: ClinicalDisposition,
        rule: str,
        response: str,
    ) -> AuditRecord:
        return AuditRecord(
            case_id=case_id,
            # Audit trails are long-lived: never persist the raw (possibly PHI-bearing) query.
            query=redact_phi_hook(query),
            engine_backend=self.engine.backend_name,
            checkpoint_id=self.engine.checkpoint_id,
            temperature_parameter=temperature,
            head_temperatures=self.head_temperatures(),
            policy_config=self.policy_config.as_dict(),
            triage=triage,
            retrieved_doc_ids=retrieved_ids,
            accepted_doc_ids=accepted_ids,
            claim_verification=verification,
            disposition=disposition,
            policy_rule_triggered=rule,
            final_response=response,
        )

    def _triage(self, triage_raw: dict[str, Any], temps: dict[str, float], use_bi: bool) -> TriageDecision:
        cfg = self.policy_config
        return run_stage1_triage(
            triage_raw,
            temperature=temps["triage"],
            use_bidirectional=use_bi,
            out_of_scope_threshold=cfg.out_of_scope_threshold,
            unsafe_threshold=cfg.attack_threshold,
            single_kb_threshold=cfg.single_kb_threshold,
        )

    def _halts_at_stage1(self, triage: TriageDecision) -> bool:
        # Emergencies are decided at Stage 1 as well: Rule 2 escalates regardless of evidence, and a later
        # backend failure must not downgrade an acute emergency to a generic fail-closed escalation.
        return triage.skip_retrieval or triage.urgency_expected_score >= self.policy_config.emergency_urgency_threshold

    def _retrieve_and_gate(
        self,
        case_record: dict[str, Any],
        query: str,
        triage: TriageDecision,
        temps: dict[str, float],
        relevance_threshold: float,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """Stages 2-3: returns (retrieved, accepted) passages."""
        # top_k per routed domain (primary domain first) so a two-domain fan-out cannot be starved by
        # one domain and drop the guideline the other domain contributes.
        retrieved = [
            r
            for domain in triage.routed_domains
            for r in self.retriever.retrieve(query=query, allowed_domains=[domain], top_k=self.top_k)
        ]
        # Relevance is scored on exactly the passages retrieved for the routed domains.
        relevance = self.engine.predict_chunk_relevance(case_record, retrieved) if retrieved else {}
        gated = filter_retrieved_chunks(
            retrieved=retrieved,
            raw_chunk_relevance=relevance,
            temperature=temps["relevance"],
            relevance_threshold=relevance_threshold,
        )
        return retrieved, [r for r in gated if r.passed_gate]

    def run_case(self, case_record: dict[str, Any]) -> AuditRecord:
        case_id = str(case_record.get("case_id", "LIVE-01"))
        query = str(case_record["query"])
        logger.debug("Running case %s in mode %s", case_id, self.mode)
        try:
            return self._run_case(case_id, query, case_record)
        except Exception:
            # Backend outages, malformed responses, and missing signals must never yield a served answer.
            logger.exception("Case %s failed inside the decision pipeline; failing closed.", case_id)
            return self._audit(
                case_id, query, 1.0, None, [], [], None,
                ClinicalDisposition.ESCALATE_UNVERIFIED_OR_CONFLICT, FAIL_CLOSED_RULE, FAIL_CLOSED_RESPONSE,
            )

    def _run_case(self, case_id: str, query: str, case_record: dict[str, Any]) -> AuditRecord:
        cfg = self.policy_config
        temps = self.head_temperatures()
        triage_raw = self.engine.predict_triage(case_record)

        if self.mode in ("baseline", "retrieval_baseline"):
            triage = run_stage1_triage(triage_raw, temperature=1.0, use_bidirectional=False)
            retrieved = self.retriever.retrieve(query=query, allowed_domains=None, top_k=self.top_k)
            retrieved_ids = [r.chunk.doc_id for r in retrieved]
            candidate_claim = case_record.get("candidate_claim") or synthesize_extractive_claim(retrieved)

            if self.mode == "baseline":
                return self._audit(
                    case_id, query, 1.0, triage, retrieved_ids, retrieved_ids, None,
                    ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER, "BASELINE_NO_GUARD", candidate_claim,
                )

            accepted_ids = [
                r.chunk.doc_id for r in retrieved
                if _query_lexical_overlap(query, r) >= self.retrieval_overlap_threshold
            ]
            if not accepted_ids:
                logger.info("Case %s abstained under retrieval_baseline (0 chunks accepted)", case_id)
                return self._audit(
                    case_id, query, 1.0, triage, retrieved_ids, [], None,
                    ClinicalDisposition.ABSTAIN_INSUFFICIENT_EVIDENCE, "RETRIEVAL_SCORE_BELOW_THRESHOLD",
                    "ABSTAIN: Top retrieval similarity score below threshold.",
                )
            return self._audit(
                case_id, query, 1.0, triage, retrieved_ids, accepted_ids, None,
                ClinicalDisposition.SERVE_VERIFIED_CITED_ANSWER, "RETRIEVAL_ONLY_GUARD", candidate_claim,
            )

        use_bi = self.mode in ("debiased_system1", "clinguard_full")
        three_way = use_bi
        rel_thresh = cfg.relevance_threshold if use_bi else 0.20
        enable_conflict = self.mode == "clinguard_full"

        # Stage 1: Triage & Routing (runs before any retrieval or model call on guideline text)
        triage = self._triage(triage_raw, temps, use_bi)
        if self._halts_at_stage1(triage):
            logger.info("Case %s halted at Stage 1 triage (skip_retrieval=%s)", case_id, triage.skip_retrieval)
            disp, rule, msg = apply_clinical_policy(
                triage=triage,
                accepted_doc_ids=[],
                verification=None,
                enable_conflict_warning=enable_conflict,
                config=cfg,
            )
            return self._audit(case_id, query, temps["claim"], triage, [], [], None, disp, rule, msg)

        # Stages 2-3: Routed Hybrid Retrieval and Evidence Gate
        retrieved, accepted_chunks = self._retrieve_and_gate(case_record, query, triage, temps, rel_thresh)
        accepted_ids = [r.chunk.doc_id for r in accepted_chunks]

        # Stage 4: Claim Verification of the exact claim that may be served, against accepted evidence only
        verification = None
        if accepted_chunks:
            candidate_claim = case_record.get("candidate_claim") or synthesize_extractive_claim(accepted_chunks)
            claim_raw = self.engine.predict_claim(case_record, candidate_claim, accepted_chunks)
            verification = verify_clinical_claim(
                claim_text=candidate_claim,
                raw_probs=claim_raw,
                temperature=temps["claim"],
                use_bidirectional=use_bi,
                three_way=three_way,
                drift_threshold=cfg.drift_threshold,
                conflict_temperature=temps["conflict"],
            )

        # Stage 5: Deterministic Policy Guard
        disp, rule, msg = apply_clinical_policy(
            triage=triage,
            accepted_doc_ids=accepted_ids,
            verification=verification,
            enable_conflict_warning=enable_conflict,
            config=cfg,
        )
        return self._audit(
            case_id, query, temps["claim"], triage, [r.chunk.doc_id for r in retrieved],
            accepted_ids, verification, disp, rule, msg,
        )
