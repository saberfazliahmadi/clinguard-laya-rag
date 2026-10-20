# Project Status (`PROJECT_STATUS.md`)

## Current Phase
- **Phase 1: Understand the documents** — **COMPLETED**

## Completed Tasks
- [x] Examined all 7 attached PDF documents covering Jev, Laya, Kev, Decider, Jul, `layaMOE`, `layaAsRagJudge`, ONNX SDK ports, and enterprise decision-layer architectures.
- [x] Extracted technical definitions, model architectures (`ModernBERT-large` 421M / `mmBERT-base` 322M + 2-layer `[MASK]` option-scoring decision head), training methodology (RLCD with strictly proper scoring rules), benchmarks, and failure modes.
- [x] Identified healthcare RAG architectural opportunities (local privacy boundary, pre-retrieval clinical triage, chunk-level evidence sufficiency scoring, post-generation claim verification, temperature calibration, and deterministic policy guardrails with audit logging).
- [x] Created `docs/document_analysis.md`, `PROJECT_STATUS.md`, and `PROJECT_PLAN.md`.

## Outstanding Tasks
- [ ] **Phase 2 (Focused online research):** Verify primary sources via online search (`github.com/NandhaKishorM/laya`, Hugging Face `convaiinnovations/laya` & `convaiinnovations/laya-typed-decisions`, TypeSafe AI Jev docs, Vishal Mysore's `layaMOE`/`layaAsRagJudge` repos, and relevant arXiv papers including `arXiv:2511.12723` and *Type-Safe Is Not Error-Free*) and save findings to `docs/research_review.md`.
- [ ] **Phase 3 (Design the project):** Define the concrete healthcare RAG use case, system architecture, technology stack, evaluation metrics, and implementation plan in `docs/architecture.md`.
- [ ] **Phase 4 (Build the repository):** Implement the modular source code, configuration, dependencies, public clinical dataset/corpus loader, and unit/integration tests.
- [ ] **Phase 5 (Test and improve):** Execute tests, run reproducible baseline vs. decision-gated RAG evaluations, and document actual measured metrics.
- [ ] **Phase 6 (Prepare for recruiters):** Finalize `README.md`, architecture diagrams, setup instructions, limitations, and repository polish.

## Known Problems & Technical Risks to Mitigate
1. **Base Laya Zero-Shot Weakness (`0.362` on `typed-decisions`):** Relying blindly on uncalibrated zero-shot base checkpoints risks poor decision accuracy; Phase 2 must inspect exact Hugging Face checkpoint availability (`convaiinnovations/laya`, `laya-typed-decisions`, `laya-multilingual`), SDK APIs, and lightweight calibration/adaptation options.
2. **Out-of-the-Box Overconfidence (`0.466` unfitted ECE):** Raw probabilities cannot be used directly for safety thresholds without post-hoc temperature scaling on a held-out calibration split.
3. **512/1,024-Token Context Limit & <=20 Option Budget:** Healthcare RAG passages must be chunked to fit Laya's context window, and `choice` schemas must be kept compact (<=10–15 options) with an explicit `unclear`/`other` fallback option.
4. **Chat Environment Constraint:** Because direct disk-write tools are not attached to this chat environment, project files must be maintained in clean, modular Markdown/code blocks and synchronized across phases.

## Exact Next Action
- Wait for user instruction to start **Phase 2: Focused online research**, then verify official repositories, package versions, API signatures, and research papers via web search and write `docs/research_review.md`.