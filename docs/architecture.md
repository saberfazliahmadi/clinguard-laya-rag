# ClinGuard-RAG System Architecture

> **Evaluation Scope Note:** All benchmark metrics in this repository represent a deterministic **policy-logic test on offline golden fixtures** (`data/evaluation_benchmark.json`) for the System-1 decision layer.

## 1. Architectural Overview

`ClinGuard-RAG` decouples free-text generation (System-2) from structured clinical safety governance (System-1). Rather than relying on an autoregressive LLM to self-police retrieval relevance, dosing safety, and citation grounding, the pipeline surrounds hybrid search with fast, single-pass System-1 decision checkpoints (`choice`, `score`, `noul`) and a deterministic Python policy engine.

## 2. Five-Stage Execution Pipeline

1. **Stage 1: Pre-Retrieval Triage & Two-Step Router (`src/clinguard_rag/router.py`)**
   - Screens incoming queries for prompt injection or unsafe dosing overrides (`prompt_attack_or_unsafe >= 0.65`) and acute clinical emergencies (`urgency_expected_score >= 2.40` on a `0..3` rubric).
   - Evaluates 5 domain options (`antimicrobial_stewardship`, `cardiovascular_hypertension`, `endocrine_diabetes`, `renal_dosing_contraindications`, `out_of_scope`) using bidirectional option-order averaging.
   - If in scope, renormalizes probabilities across the 4 clinical specialty domains: routes to a single domain if the top probability is `>= 0.55`, or fans out to the top 2 domains when cross-specialty ambiguity exists.

2. **Stage 2: Hybrid Lexical + TF-IDF Cosine Retriever (`src/clinguard_rag/retriever.py`)**
   - Restricts search to the routed domains from Stage 1 (returning an empty list if `allowed_domains == []`).
   - Ranks candidate guideline passages using `BM25Okapi` ($k_1 = 1.5, b = 0.75$) and L2-normalized smoothed TF-IDF cosine similarity, merging both ranked lists via Reciprocal Rank Fusion ($k_{\text{RRF}} = 60$) and filtering out zero-overlap noise.

3. **Stage 3: Fail-Closed Evidence Relevance Gate (`src/clinguard_rag/verifier.py`)**
   - Scores each retrieved passage with a calibrated `noul` relevance judgment (`system1_relevance_prob`).
   - Enforces fail-closed gating: passages missing a relevance score or scoring below `relevance_threshold = 0.60` (such as pediatric or hepatic hard negatives) are rejected before claim verification.

4. **Stage 4: Bidirectional 3-Way Claim & Conflict Verifier (`src/clinguard_rag/verifier.py`)**
   - Evaluates candidate claims across `["supported", "contradicted", "not_mentioned"]` in both forward and reverse option order.
   - Computes Total Variation Distance ($\Delta_{\text{order}}$) between forward and reverse passes, flagging positional instability whenever $\Delta_{\text{order}} > 0.25$.
   - Extracts bracketed `[DOC-ID]` citations and evaluates inter-guideline disagreement (`guidelines_conflict_prob`).

5. **Stage 5: Deterministic Clinical Policy Engine (`src/clinguard_rag/policy.py`)**
   - Applies 6 ordered deterministic rules (`RULE_1` through `RULE_6`) to emit a validated Pydantic `AuditRecord` with one of five clinical dispositions: `ABSTAIN_UNSAFE_OR_OUT_OF_SCOPE`, `ESCALATE_ACUTE_EMERGENCY`, `ABSTAIN_INSUFFICIENT_EVIDENCE`, `ESCALATE_UNVERIFIED_OR_CONFLICT`, `SERVE_WITH_CONFLICT_WARNING`, or `SERVE_VERIFIED_CITED_ANSWER`.