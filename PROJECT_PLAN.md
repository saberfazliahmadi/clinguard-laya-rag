# Project Plan (`PROJECT_PLAN.md`)

## 1. Project Vision & Scope
Build a reproducible, well-engineered GitHub project demonstrating practical expertise in **System 1 Decision Models (Laya / Jev-compatible interfaces)** integrated with **Healthcare Retrieval-Augmented Generation (RAG)**. Rather than treating a decision model as a replacement for retrieval or clinical reasoning, the project will use Laya (and a Jev-compatible `/v1/systemone` abstraction) as a **calibrated control, evidence-gating, and claim-verification layer** around a clinical RAG pipeline using public biomedical/clinical guideline data.

## 2. Preliminary Architectural Decisions (Subject to Phase 2 Verification & Phase 3 Finalization)
1. **Interchangeable Decision Interface (`SystemOneClient`):** Support local Laya inference (`laya` Python SDK / ONNX or Hugging Face checkpoint) alongside a Jev-compatible HTTP client (`POST /v1/systemone`) so workflows can run 100% locally on-premises (preserving healthcare privacy) or against a hosted endpoint.
2. **Three-Stage Healthcare RAG Control Layer:**
   - **Stage 1 — Pre-Retrieval Triage (`choice`, `score`, `noul`):** Classify query clinical category (with an explicit `unclear_or_out_of_scope` option), score acuity/urgency, and screen for unsafe/adversarial requests.
   - **Stage 2 — Post-Retrieval Evidence Gate (`score`, `noul`):** Score retrieved clinical chunks (sized within Laya's 512/1,024-token window) for relevance and evidence sufficiency before allowing answer synthesis.
   - **Stage 3 — Post-Generation Claim Verification (`noul`):** Verify whether generated clinical statements are entailed by the cited source chunks and flag contradictions.
3. **Calibration & Deterministic Policy Engine:** Include post-hoc temperature scaling (`ECE` and `Brier score` evaluation) and a deterministic policy layer that separates model judgment from execution authority (`auto_answer`, `cautious_answer_with_warning`, `abstain_and_escalate_to_clinician`), recording a full JSON audit trail for every query.

## 3. Implementation Milestones & Acceptance Criteria

| Phase | Milestone | Deliverable(s) | Acceptance Criteria |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Document Analysis | `docs/document_analysis.md`, `PROJECT_STATUS.md`, `PROJECT_PLAN.md` | All attached documents analyzed accurately with zero fabricated facts; limitations and healthcare RAG mappings documented. |
| **Phase 2** | Focused Online Research | `docs/research_review.md` + updated status/plan | Official Laya/Jev repos, PyPI/HF artifacts, API signatures, and arXiv papers verified with reliable citations. |
| **Phase 3** | Architecture & Design | `docs/architecture.md` + updated status/plan | Concrete healthcare RAG use case, public dataset choice, module boundaries, and evaluation protocol defined. |
| **Phase 4** | Repository Implementation | Source code (`src/`), configs, sample public clinical corpus, `tests/` | Clean, modular Python codebase with working retrieval, decision layer, calibration, policy engine, audit logging, and unit tests. |
| **Phase 5** | Testing & Evaluation | Evaluation script, test report, baseline comparison | Reproducible evaluation comparing standard RAG baseline vs. decision-gated RAG using real metrics (Accuracy, ECE, Brier score, Hallucination/Abstention rate). |
| **Phase 6** | Recruiter Readiness | `README.md`, setup guide, architecture diagrams, limitations | Professional, honest documentation clearly distinguishing implemented features from future work without unsubstantiated clinical claims. |