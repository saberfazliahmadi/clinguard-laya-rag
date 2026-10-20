# ClinGuard-RAG (`clinguard-laya-rag`)

**Calibrated System-1 Decision Layer (Laya / Jev) for On-Premises Healthcare Retrieval-Augmented Generation**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

`ClinGuard-RAG` is a reference engineering repository demonstrating how non-autoregressive **System 1 decision models**—specifically **Laya** (`convaiinnovations/laya-typed-decisions`, Apache 2.0) and **Jev-compatible** `/v1/systemone` APIs—can serve as a fast, calibrated control and verification layer around a clinical guideline Retrieval-Augmented Generation (RAG) pipeline[cite: 6, 11, 27, 30, 91].

> **Important Engineering & Clinical Disclaimer:** This repository is a research and software architecture demonstration built on short excerpts from public clinical guidelines (WHO AWaRe, KDIGO, AHA/ACC, ESC, ADA). It is **not** a validated medical device, has **not** undergone clinical validation, and must **never** be used for real patient care or clinical decision-making.

---

## 1. Why a System-1 Decision Layer for Healthcare RAG?

Most production AI pipelines use autoregressive Large Language Models ("System 2") for bounded operational decisions—asking a generative model to write text or JSON token-by-token and parsing the result back into a routing label, relevance score, or safety flag[cite: 28, 72, 82]. In clinical RAG workflows, that pattern introduces four failure modes:

1. **Privacy Boundary Violations:** Sending raw clinical queries or retrieved patient-adjacent state to external cloud LLMs before local filtering exposes sensitive data outside hospital infrastructure[cite: 7, 8, 91].
2. **BM25 Lexical Hard-Negative Traps:** Keyword retrieval often surfaces passages that share drug names and lab units (e.g., *pediatric* `40 mg/kg` amoxicillin vs. *adult* `1 g` amoxicillin, or *hepatic* vs. *renal* lactic acidosis risks), leading generative models to synthesize fluent but clinically wrong answers.
3. **Erasing Inter-Guideline Conflicts:** When clinical guidelines evolve or conflict (e.g., legacy glycemic eGFR cutoffs vs. updated heart-failure eGFR cutoffs for SGLT2 inhibitors), generative summaries frequently smooth over contradictions instead of preserving them for the clinician.
4. **Type-Safety Without Calibration:** Both **Jev** (TypeSafe AI's commercial hosted API) and **Laya** (Convai Innovations' 421M-parameter `ModernBERT-large` open-weight model) evaluate `choice`, `score`, and `noul` (yes/no) questions in a single forward pass[cite: 4, 27, 28, 29]. However, *syntactic type-safety is not semantic correctness*[cite: 5, 57, 132]: Laya ships overconfident out of the box (requiring post-hoc temperature scaling on held-out target data)[cite: 32, 43, 126], degrades when option sets exceed ~20 choices due to its 192-token option budget, and is sensitive to option ordering.

---

## 2. System Architecture

`ClinGuard-RAG` separates **System 1 bounded judgment**, **hybrid evidence retrieval**, **System 2 grounded claim synthesis**, and **deterministic application policy**[cite: 18, 22, 66, 68, 121, 132]:

```mermaid
flowchart TD
    Q[Clinical Query] --> S1[Stage 1: Pre-Retrieval System-1 Triage & Router]
    S1 -->|noul: prompt_attack_or_unsafe >= 0.65\nor choice: out_of_scope >= 0.60| B1[RULE 1: ABSTAIN_UNSAFE_OR_OUT_OF_SCOPE]
    S1 -->|score: E_urgency >= 2.40| B2[RULE 2: ESCALATE_ACUTE_EMERGENCY]
    S1 -->|Two-Step KB Fan-Out:\nTop Domain >= 0.55 -> 1 KB\nElse -> Top-2 KBs| S2[Stage 2: Hybrid BM25 + Dense RRF Retriever]
    S2 --> S3[Stage 3: Post-Retrieval System-1 Evidence Gate]
    S3 -->|No chunk passes calibrated\nP_relevance >= 0.60| B3[RULE 3: ABSTAIN_INSUFFICIENT_EVIDENCE]
    S3 -->|Accepted Clinical Chunks| S4[Stage 4: Cited Claim Synthesis + 3-Way System-1 Claim Judge]
    S4 --> S5[Stage 5: Post-Hoc Temperature Calibration + Deterministic Policy Engine]
    S5 -->|Unstable Order TVD > 0.25\nor P_supported < 0.75\nor Missing Citation| B4[RULE 4: ESCALATE_UNVERIFIED_OR_CONFLICT]
    S5 -->|Supported Claim +\nP_guidelines_conflict >= 0.55| B5[RULE 5: SERVE_WITH_CONFLICT_WARNING]
    S5 -->|Supported Claim >= 0.75 +\nStable + Valid Citation| B6[RULE 6: SERVE_VERIFIED_CITED_ANSWER]