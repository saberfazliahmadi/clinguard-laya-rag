# Phase 3: System Architecture & Engineering Design (`docs/architecture.md`)

## 1. Project Identity & Healthcare RAG Use Case

* **Repository Name:** `clinguard-laya-rag` (**ClinGuard-RAG**: *Calibrated System-1 Decision Layer for Clinical Guideline Retrieval-Augmented Generation*)
* **Target Clinical Use Case:** **On-Premises Clinical Guideline & Drug-Safety Evidence Support**
  * Clinicians and clinical pharmacists frequently query institutional clinical practice guidelines for renal dosing adjustments, antimicrobial stewardship, hypertension management, and diabetes contraindications (e.g., *"Can metformin be continued in a patient with eGFR 24 mL/min/1.73mÂ²?"*).
  * Standard generative RAG pipelines suffer from four well-documented operational failures in healthcare:
    1. **Privacy Leakage:** Sending clinical queries or patient context to external generative APIs before local filtering crosses the hospital trust boundary.
    2. **Lexical Hard-Negative Trap:** BM25 retrieves passages mentioning the right drug (`metformin`) and lab (`eGFR`) but the wrong threshold (`eGFR 45â€“60`), and the LLM synthesizes a fluent but dangerous answer.
    3. **Silent Erasing of Clinical Conflicts:** When two guidelines disagree or special population exclusions apply, generative summaries smooth over contradictions instead of surfacing them.
    4. **Unparsable or Uncalibrated Control Logic:** Using a generative LLM for routing, sufficiency gating, and claim verification introduces token-generation latency, JSON parsing failures, and uncalibrated confidence scores.

### Why Laya & Jev Fit This Architecture
ClinGuard-RAG separates **System 1 bounded judgment** (Laya locally or Jev via `/v1/systemone`), **hybrid retrieval** (BM25 + dense vector search), **System 2 grounded synthesis**, and **deterministic clinical policy execution**:
