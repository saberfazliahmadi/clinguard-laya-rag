# Project Status: ClinGuard-RAG (`clinguard-laya-rag`)

**Current Version:** `0.1.0`  
**Overall Status:** Complete & Verified (All 6 Phases + Post-Review Hardening Completed)  
**License:** Apache License 2.0  

---

## 1. Phase-by-Phase Execution Summary

| Phase | Milestone | Primary Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Document & Domain Analysis** | [`docs/document_analysis.md`](docs/document_analysis.md) — Analysis of System-1 decision models (`choice`, `score`, `noul`), Laya (`ModernBERT-large` 421M) vs. Jev (`/v1/systemone`), option token budgets, and clinical RAG failure modes. | **Completed** |
| **Phase 2** | **Literature & SDK Verification** | [`docs/research_review.md`](docs/research_review.md) — Verification of `laya` SDK primitives, Hugging Face checkpoints (`convaiinnovations/laya-typed-decisions`), positional bias findings (`arXiv:2610.02267`), and option-name binding behavior (`arXiv:2609.26758`). | **Completed** |
| **Phase 3** | **System Architecture Design** | [`docs/architecture.md`](docs/architecture.md) — 5-stage clinical RAG control architecture separating probabilistic System-1 judgment from deterministic Python policy enforcement. | **Completed** |
| **Phase 4** | **Core Implementation & Data** | [`src/clinguard_rag/`](src/clinguard_rag/), [`data/clinical_guidelines_corpus.json`](data/clinical_guidelines_corpus.json) (10 guideline excerpts + hard negatives), and [`data/evaluation_benchmark.json`](data/evaluation_benchmark.json) (6 calibration + 10 held-out test scenarios). | **Completed** |
| **Phase 5** | **Testing & Ablation Evaluation** | [`tests/`](tests/), [`scripts/run_evaluation.py`](scripts/run_evaluation.py), [`scripts/demo_cli.py`](scripts/demo_cli.py), and [`docs/evaluation_report.md`](docs/evaluation_report.md). | **Completed** |
| **Phase 6** | **Documentation & CI Pipeline** | [`README.md`](README.md), [`pyproject.toml`](pyproject.toml), and [`.github/workflows/ci.yml`](.github/workflows/ci.yml). | **Completed** |

---

## 2. Implemented Core Modules (`src/clinguard_rag/`)

| Module | Responsibility | Verification Status |
| :--- | :--- | :--- |
| `schemas.py` | Pydantic v2 schemas for `ClinicalChunk`, `RetrievedChunk`, `DebiasedChoiceResult`, `TriageDecision`, `ClaimVerification`, `ClinicalDisposition`, and `AuditRecord`. | Verified |
| `debiasing.py` | Enforces the $\le 20$ option cardinality limit ($\max(4, \lfloor(192-16)/K\rfloor)$ tokens/option) and computes bidirectional forward/reverse averages with Total Variation Distance ($\Delta_{\text{order}}$) stability gating. | Verified |
| `calibration.py` | Fits scalar temperature $T > 0$ via multi-class Negative Log-Likelihood (NLL) grid search (`0.50` to `3.00`, step `0.05`) and computes 10-bin Expected Calibration Error (ECE) and Brier Score. | Verified |
| `router.py` | Stage 1 pre-retrieval triage (`prompt_attack_or_unsafe`, `clinical_urgency` expected value $E[\text{urgency}]$, and two-step 1-domain vs. 2-domain knowledge-base fan-out). | Verified |
| `retriever.py` | Stage 2 hybrid lexical (`BM25Okapi`) + L2-normalized IDF cosine similarity retriever combined via Reciprocal Rank Fusion ($k_{\text{RRF}} = 60$), with fail-closed handling for empty domain lists and zero-overlap queries. | Verified |
| `verifier.py` | Stage 3 chunk relevance gate (fail-closed on missing document scores) and Stage 4 bidirectional 3-way claim verification (`supported`, `contradicted`, `not_mentioned`) with `[DOC-ID]` citation provenance checking. | Verified |
| `policy.py` | Stage 5 deterministic 6-rule clinical policy guard mapping calibrated outputs to `ClinicalDisposition` while prioritizing acute emergency escalation (`RULE_2`) ahead of out-of-scope abstention. | Verified |
| `decision_engine.py` | Provides `DeterministicEvalBackend` for offline reproducible CI benchmarks, alongside `LayaLocalBackend` and `SystemOneHTTPBackend` supporting `{question_id: definition}` dictionary normalization, live `predict_case` mapping, and pre-inference regex PHI scrubbing. | Verified |
| `pipeline.py` | End-to-end `ClinGuardPipeline` orchestrator supporting 4 ablation modes (`baseline`, `raw_system1`, `debiased_system1`, `clinguard_full`) and extractive claim fallback. | Verified |

---

## 3. Quality Gates & Benchmark Verification

### 3.1 Automated Test Suite (`pytest -v`)
- **`tests/test_debiasing_and_calibration.py`**: Validates option token budget calculations, bidirectional TVD instability detection, `TemperatureCalibrator.fit()` NLL reduction ($T = 1.4$), ECE/Brier computation, PHI regex redaction (`EMAIL`, `PHONE`, `MRN`, `DOB` without false positives on lab values), and question dictionary normalization.
- **`tests/test_retriever_and_router.py`**: Validates single-domain vs. two-domain router fan-out, hybrid RRF retrieval of adult and pediatric hard-negative chunks, fail-closed retrieval on `allowed_domains=[]` or zero-overlap queries, and fail-closed Stage 3 chunk gating on missing relevance keys.
- **`tests/test_pipeline_and_policy.py`**: Validates all 10 held-out test dispositions (`EVAL-01..EVAL-10`), hard-negative filtering (`EVAL-02`), guideline conflict preservation (`EVAL-03`), and monotonic Brier Score and Multi-Class NLL improvement across System-1 ablation modes.

### 3.2 Reproducible Ablation Summary (`python scripts/run_evaluation.py`)
Evaluated on the 10 held-out test scenarios (`EVAL-01..EVAL-10`) using `DeterministicEvalBackend` with temperature $T = 1.4$ fitted on the 6 calibration scenarios (`CAL-01..CAL-06`):

| Pipeline Mode | Temp ($T$) | Disposition Accuracy (`N=10`) | Unsafe Claim Leakage (`N=5` unsafe) | Claim Judge Accuracy | Brier Score ($\downarrow$) | Multi-Class NLL ($\downarrow$) | 10-Bin ECE ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | `1.0` | `40.0%` (`4/10`) | `100.0%` (`5/5`) | — | — | — | — |
| **`raw_system1`** | `1.0` | `60.0%` (`6/10`) | `60.0%` (`3/5`) | `70.0%` (`7/10`) | `0.2158` | `0.7688` | `0.2400` |
| **`debiased_system1`** | `1.0` | `80.0%` (`8/10`) | `20.0%` (`1/5`) | `80.0%` (`8/10`) | `0.1773` | `0.6458` | `0.2315` |
| **`clinguard_full`** | **`1.4`** | **`100.0%` (`10/10`)** | **`0.0%` (`0/5`)** | **`80.0%` (`8/10`)** | **`0.1668`** | **`0.6251`** | `0.2574` |

---

## 4. Known Scope Boundaries & Future Extensions

1. **Deterministic Fixture Replay in CI:** Default benchmark runs use pre-recorded probability fixtures (`DeterministicEvalBackend`) and pre-generated candidate claims in `data/evaluation_benchmark.json` so CI completes in `< 2s` without GPU hardware or external API keys.
2. **Live Model Evaluation:** Evaluating `LayaLocalBackend` with live Hugging Face weights (`convaiinnovations/laya-typed-decisions`) or `SystemOneHTTPBackend` against a live `laya-serve` / Jev endpoint across larger clinical datasets is supported via the updated `predict_case` mapping and can be run in GPU-enabled environments (`pip install -e ".[laya,dev]"`).
3. **Production De-Identification:** `redact_phi_hook` provides lightweight regex scrubbing for emails, formatted phone numbers, MRNs, and DOBs; production clinical deployments require dedicated clinical NER de-identification pipelines.