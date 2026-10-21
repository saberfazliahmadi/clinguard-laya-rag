# Phase 1: Document Analysis â€” Laya, Jev, and System 1 Decision Models for Healthcare RAG

## 1. Corpus Overview
Seven PDF documents were analyzed, representing six unique primary sources published between September 23, 2026, and October 12, 2026 (Documents 3 and 6 are identical copies of Celia Lozano Grijalba's LinkedIn article):
1. **Eva Wong (reviewed by Kristian Sundet), ZimaSpace (Sep 24, 2026):** *Jev vs Laya: Hosted Decision API vs Open-Source Local Model (2026)*.
2. **Vishal Mysore, Medium (Sep 26, 2026):** *What Is Laya? Laya vs Jev, with Live Demo!!*.
3. **Celia Lozano Grijalba, DataPills / LinkedIn (Sep 27, 2026):** *DataPills â€” Jev opened the door. Laya made it open* (provided twice).
4. **Nilay Baranwal, LinkedIn (Sep 23, 2026):** *Jev vs Laya: Decision Models* (including practitioner commentary and benchmark critiques).
5. **Vinayak Kale, LinkedIn (Sep 26, 2026):** *Jev, Kev and Laya explained* (including ecosystem tool links in comments).
6. **Kevin Gatimu, LinkedIn (Oct 12, 2026):** *My OpenSource contribution to Laya (System 1 decision model)*.
7. **Aritra Ghosh, LinkedIn (Sep 28, 2026):** *JEV and Laya: Why Insurance AI Needs a Decision Layer*.

---

## 2. Core Definitions & Taxonomy

### 2.1 System 1 Decision Models
* **Conceptual Definition:** Named by analogy to Daniel Kahneman's fast, intuitive "System 1" cognition (as an architectural metaphor rather than human cognition), System 1 decision models replace token-by-token text generation with a single forward pass that takes an input state and structured questions and returns calibrated probability distributions over predefined outputs.
* **Core Abstraction:** `State + Typed Question -> Decision Model -> Probability / Selected Option -> Application Policy -> Action`.
* **Three Standard Decision Primitives:**
  * **`choice`:** Selects among a predefined list of categorical options supplied at request time and returns a probability distribution over those options (e.g., routing a ticket to `billing`, `technical_support`, or `sales`).
  * **`score`:** Evaluates the input state against an ordered rubric/scale (e.g., urgency from `0` to `4`), returning per-level probabilities and an expected scalar score.
  * **`noul`:** Evaluates a binary (yes/no or true/false) proposition and returns the probability that the statement holds.

### 2.2 Jev (TypeSafe AI)
* **Overview:** A proprietary, closed-weight, cloud-hosted System 1 decision engine launched on September 15, 2026, by TypeSafe AI (founded by Diogo Almeida, Sasha Sheng, and Erik Gafni) after two years of development.
* **Deployment & Pricing:** Accessed exclusively via a managed cloud API (`POST /v1/systemone`) at `$0.042` per million input tokens with `$0` charge for output tokens.
* **Capabilities:** Supports long contexts (up to `64k` tokens) and large option sets (up to `255` options, scoring `0.870` zero-shot on `Banking77`).

### 2.3 Laya (Convai Innovations)
* **Overview:** An open-source, non-autoregressive System 1 decision model released on September 18, 2026 (three days after Jev) by Nandakishor Mukkunnoth (Convai Innovations) under the Apache 2.0 license.
* **Distribution:** Available via GitHub (`github.com/NandhaKishorM/laya`), Hugging Face (`convaiinnovations/laya`), and PyPI (`pip install laya`, `v0.4.4` shown in October 2026), with official/community ONNX runtimes for WebAssembly, `.NET` (`Laya.Onnx`), and `Rust` (`laya-onnx`).
* **Drop-in API Compatibility:** Includes `laya-serve`, which exposes the same `POST /v1/systemone` request/response schema as Jev so client applications can switch by changing their base URL.
* **Terminology Disambiguation:** A November 2025 paper titled *LAYA* (`arXiv:2511.12723`) proposed a specialized output head moving beyond last-token representations, but Celia Lozano Grijalba explicitly notes that `arXiv:2511.12723` is **not** the same implementation as Convai Innovations' September 2026 Laya decision model.

### 2.4 Related Ecosystem Technologies
* **Kev (`github.com/jaredpalmer/kev`):** Created by Jared Palmer; an Apache 2.0 family of local-first, Qwen-based decision models in `0.8B`, `4B`, and `9B` parameter sizes capable of running locally on a MacBook Pro (with separate dataset licensing considerations noted in its repository).
* **Decider (`github.com/Mapika/decider`):** Created by Mark Marosi (Mapika); an independent Qwen-based Apache 2.0 project providing typed decisions.
* **Jul (`github.com/usejul/jul`):** "Juste un LLM" â€” an open wrapper matching the Jev SDK interface that runs zero-shot on local hardware using an open LLM of choice without fine-tuning.
* **layaMOE & Browser Port Suite (Vishal Mysore):** Unofficial ONNX Runtime Web / WebAssembly implementations including `layaForWeb` (int8/int4 browser port), `layaMOE` (frozen shared Laya encoder + 26.5M-parameter domain expert heads + Laya router question), `layaForWorkflows` (decision-graph routing), `layaAsRagJudge` (in-browser RAG claim verification), and `layaAgent` (System 1 + System 2 browser guard).
* **laya-cli (`github.com/Mlt9/laya-cli`):** Ergonomic CLI for Laya with streaming batch mode.
* **cortex-1-large:** Fine-tuned Laya evaluation repository by Mukti Mishra publishing per-decision CSV logs (`matched_jev_head_to_head.csv`).

---

## 3. Technical Principles & Architecture of Laya

### 3.1 Model Checkpoints & Local Router
Laya is a family of three checkpoints paired with a local `Router`:
1. **`laya` (Base English):** `ModernBERT-large` encoder, `421M` total parameters (`395M` encoder + `~26.5M` decision head), `512`-token context window.
2. **`laya-multilingual` (100+ Languages):** `mmBERT-base` encoder, `322M` parameters, `1,024`-token default context window (configurable up to `max_len=8192`).
3. **`laya-typed-decisions` (Specialized English):** `ModernBERT-large` encoder, `421M` parameters, `1,024`-token context window, fine-tuned on the `typed-decisions` benchmark workflows.
4. **Built-in `Router`:** Inspects incoming requests (including script/language detection) and routes them locally to the appropriate checkpoint.

### 3.2 Input Sequence Formatting & Single-Pass Scoring
* **Token Sequence Layout:** Laya packs the question type, question, candidate options, and state into a single bidirectional encoder input:
  `[CLS] <type> question [SEP] [MASK] option1 [MASK] option2 [SEP] state [SEP]`.
* **Token Budget Allocation:** In the 512-token English configuration, options share a fixed `192`-token budget.
* **Decision Head Mechanics:** Above the bidirectional encoder sits a 2-layer transformer decision head with a learned type embedding per question type and a scorer that directly evaluates the contextual representation at each option's `[MASK]` marker, applying a softmax across the options for that question:
  $$P(y_i \mid x, q, O) = \frac{\exp(z_i)}{\sum_{j=1}^{\vert{}O\vert{}} \exp(z_j)}$$
  where $x$ is the state, $q$ is the question, $O$ is the set of options, and $z_i$ is the logit scored at option $i$'s `[MASK]` token.
* **Multi-Question Parallelism:** Multiple questions concerning the same input state are evaluated in a single non-autoregressive forward pass.

### 3.3 Training Objective (RLCD) & Calibration
* **Reinforcement Learning for Calibrated Decisions (RLCD):** Both TypeSafe (Jev) and Convai Innovations (Laya) reference RLCD. Laya trains via reinforcement learning against **strictly proper scoring rules**â€”specifically log and spherical scoring rules for categorical choices/propositions, plus the **ranked probability score (RPS)** for ordinal `score` questionsâ€”where expected reward is mathematically maximized only when the model reports honest probabilities matching empirical frequencies.
* **Post-Hoc Temperature Scaling:** Out of the box, Laya checkpoints are overconfident (base model mean ECE is `0.466` before scaling). Laya fits a temperature parameter per `(question type, option count)` on held-out calibration data, reducing ECE to `0.081` on its evaluation suite.

---

## 4. Empirical Evidence & Benchmark Analysis

### 4.1 Published Benchmark Comparison (`typed-decisions` & General Suites)
| Metric / Dimension | Laya (`laya-typed-decisions` / Base) | Jev (`1.13.0` Published Reference) | Critical Context & Caveats |
| :--- | :--- | :--- | :--- |
| **`typed-decisions` Accuracy** | `0.766` (fine-tuned) / `0.362` (base zero-shot) | `0.727` (zero-shot) | Evaluated across 400 test cases (2,000 decisions); random baseline is `0.318` and majority baseline is `0.461`. Jev figures are third-party published, not rerun identically. |
| **Soft Accuracy (`typed-decisions`)** | `0.471` | `0.580` | Jev leads on soft probability quality on this benchmark. |
| **Brier Score (`typed-decisions`)** | `0.062` | `0.148` | Lower is better; favors specialized Laya on `typed-decisions`. |
| **ECE (`typed-decisions`)** | `0.213` | `0.144` | Lower is better; Jev is better calibrated out-of-the-box on `typed-decisions`. |
| **Score MAE (`typed-decisions`)** | `0.242` | `0.391` | Mean Absolute Error on ordinal scoring questions. |
| **ECE (After Temperature Fitting)** | `0.081` (vs `0.466` unfitted) | `0.246` | Requires fitting temperature per `(question type, option count)` on target validation data. |
| **`Banking77` (77 / 72 labels)** | `0.425` | `0.870` | Laya degrades sharply when option count exceeds ~20 due to its 192-token option budget. |
| **Anthus AI Paired Benchmark** | `0.722` (base) -> `0.896` (fine-tuned on 140 labels) | `0.788` (base) -> `0.870` (with feedback layer) | 140-label fine-tune took ~100s on M1 Max, but shifted 42% of answers on 8 untrained questions. |
| **Single-Decision Latency** | `32.8 ms` (multilingual) â€“ `39.5 ms` (English) on local Tesla T4 | `236â€“276 ms` p50 via hosted API | Batched (10 questions on T4): `72.3 ms` (multilingual) / `158.6 ms` (English), or `~7.2 ms` per question. Local GPU vs network API round-trip. |
| **RAG Claim Verification (`layaAsRagJudge`)** | `75%` accuracy, `0.80` AUROC | Not tested in experiment | Tested by Vishal Mysore on 72 hand-written claims with in-browser retrieval. |

---

## 5. Known Limitations, Failure Modes & Engineering Risks

1. **"Type-Safe Is Not Error-Free":** Eliminating malformed JSON or out-of-schema text generation does not prevent semantic errors; a decision model can return a syntactically valid option with high confidence and still be wrong.
2. **Weak Zero-Shot Generalization on Base Laya:** Base English `laya` scores `0.362` on `typed-decisions` (below the `0.461` majority baseline), meaning Laya must be treated as a base model to specialize (via fine-tuning or domain expert heads) or carefully validated per task rather than assumed to be a universal zero-shot oracle.
3. **High-Cardinality Option Degradation:** Because candidate options share a 192-token budget, Laya struggles when given more than ~20 choices (`0.425` on `Banking77`). High-cardinality tasks require coarse-to-fine hierarchical routing or candidate shortlisting.
4. **Out-of-the-Box Overconfidence & Misleading Confidence Statistics:**
   * Laya ships overconfident (`0.466` base ECE before temperature scaling; `0.213` ECE on `laya-typed-decisions`) and must be recalibrated on held-out domain data before confidence thresholds are trusted.
   * Jev's API `confidence` field is a distribution-derived statistic, **not** the calibrated empirical probability that the chosen answer is correct.
5. **Catastrophic Forgetting / Weight Drift During Fine-Tuning:** In Anthus AI's experiment, fine-tuning Laya on 140 labels for one question altered `42%` of its predictions on eight unrelated questions. Freezing the shared encoder and training isolated expert heads (`layaMOE` pattern) or maintaining a locked multi-question regression suite is necessary to prevent silent drift.
6. **Context Length Ceiling:** Base English `laya` accepts only `512` tokens (`1,024` for `laya-typed-decisions`; up to `8,192` for `laya-multilingual`), requiring chunking or passage-level scoring for long documents.
7. **Script Blindness & Multilingual Slice Disparities:** Laya's English checkpoint scored `8%` accuracy at `94.5%` confidence on Bengali script because it cannot read the script and lacks out-of-distribution awareness. Even `laya-multilingual` shows uneven accuracy across languages (`0.387` Hindi, `0.290` Bengali, `0.250` Tamil). Language/script routing and per-slice calibration thresholds are mandatory.
8. **Shared Blind Spots (No Multi-Step Reasoning, Arithmetic, or Prompt-Injection Immunity):** Neither Jev nor Laya performs multi-step reasoning (e.g., complex conflicting legal/clinical clauses), reliable counting/arithmetic, or resists adversarial text injected into the input state.
9. **Cross-Runtime Drift:** Porting Laya across runtimes (Python PyTorch vs ONNX CPU/CUDA/WASM) exposes tokenization, truncation, calibration, and floating-point kernel drift, requiring golden-output parity tests.

---

## 6. Applications to Healthcare Retrieval-Augmented Generation (RAG)

The reviewed documents establish both a direct healthcare RAG precedentâ€”Kevin Gatimu's on-premises health-record RAG system using Laya so sensitive clinical data never leaves the facility and hallucinated control decisions are eliminatedâ€”and concrete architectural patterns for integrating Laya into a healthcare RAG pipeline:

1. **Privacy-Preserving Local Triage & Query Routing (Pre-Retrieval):**
   * Before any patient query or clinical note leaves local hospital infrastructure, a local Laya model can classify clinical intent (`choice`, kept under 20 options), estimate clinical urgency/acuity (`score`), and flag out-of-scope or unsafe requests (`noul`) in ~33â€“40 ms on GPU (or local CPU/ONNX).
   * Ambiguous or out-of-distribution queries can be routed to an explicit `unclear`/`escalate` branch rather than forced into a wrong clinical queue.
2. **Retrieved Evidence Relevance & Sufficiency Gate (Post-Retrieval / Pre-Generation):**
   * Since English Laya has a `512`â€“`1,024` token context limit, it naturally fits evaluating individual retrieved clinical guideline chunks or short evidence passages against a clinical question.
   * Using `score` (evidence relevance rubric) and `noul` ("Does this retrieved passage contain sufficient evidence to answer the query?"), Laya can filter irrelevant chunks and halt LLM generation if retrieval evidence is insufficient, preventing ungrounded synthesis.
3. **Claim Verification & Hallucination Guard (`layaAsRagJudge` Pattern, Post-Generation):**
   * Following Vishal Mysore's `layaAsRagJudge` experiment (`75%` accuracy, `0.80` AUROC on claim verification), a generative LLM ("System 2") can draft a cited clinical response while Laya ("System 1") evaluates each generated claim against its cited source passage via `noul` entailment/contradiction checks.
   * If retrieved clinical records conflict, the system preserves the conflict and escalates rather than allowing a fluent LLM summary to erase unresolved clinical facts.
4. **Separation of Judgment from Authority (Deterministic Clinical Policy Engine):**
   * Following Eva Wong's and Aritra Ghosh's guardrail architectures, Laya never executes actions directly.
   * Temperature-calibrated probabilities feed a deterministic application policy (`>= high_threshold`: return verified cited answer; `medium_threshold`: trigger deeper verification/warning; `< low_threshold` or safety flag: abstain and escalate to clinician review) with a complete, traceable audit log (evidence IDs, model checkpoint/version, question definitions, raw and calibrated probabilities, policy rule triggered, and final disposition).
