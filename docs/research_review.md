# Phase 2: Focused Online Research & Verification (`docs/research_review.md`)

## 1. Executive Summary of Verification
Online verification against official GitHub repositories, Hugging Face model cards, package registries, and October 2026 arXiv preprints confirms the primary claims in the Phase 1 document corpus while uncovering **critical implementation details and newly published failure modes** that directly shape how Laya and Jev-compatible models should be engineered in a healthcare RAG system:
1. **Live Package & Checkpoint Verification:** Laya (`github.com/NandhaKishorM/laya`, PyPI `laya==0.4.8` as of October 15, 2026) and its three Hugging Face checkpoints (`convaiinnovations/laya`, `laya-multilingual`, `laya-typed-decisions`) are active Apache 2.0 projects.
2. **Confirmed Calibration Bug (`#186`):** The official `convaiinnovations/laya-typed-decisions` model card discloses that its shipped per-type temperatures (`[1.0148, 1.0374, 1.0575]`) were accidentally fitted on a slice of training data (`#186`) and are overridden by `temperature_by_options` inherited from the base checkpoint, leaving raw confidence uncalibrated (ECE `0.213` vs. Jev's `0.144`).
3. **Newly Verified arXiv Research on System 1 Decision Models (Sepâ€“Oct 2026):**
   - **Option-Name Polarity Bias (`arXiv:2609.26758v3`):** *Type-Safe Is Not Error-Free: Typed Decision Models Follow the Option Name, Not the Definition Bound to It* demonstrates that swapping option names (`0/1` vs. `no/yes`) while keeping definitions identical flips `70.4%â€“76.9%` of answers and drops AUC from `0.94` to `0.23` with `0%` type errors.
   - **Positional Bias, Option Truncation & RAG Error Profiles (`arXiv:2610.02267v1`):** *Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses* proves that reversing option order changes `30.3%` of Laya's decisions (vs. `1.8%` for Jev), explains the exact option truncation formula (`max(4, (192 - 16) / K)` tokens per option), shows that Platt scaling reduces Laya's RAG ECE from `0.22` to `0.027`, and reveals that in RAG relevance ($n=1,920$), **Laya excels at rejecting BM25 hard negatives (`61.8%` vs. Jev's `49.3%`)** while Jev has higher positive recall (`58.3%` vs. `45.4%`).
4. **RAG Claim Verification Design (`layaAsRagJudge`):** Vishal Mysore's `github.com/vishalmysore/layaAsRagJudge` confirms that switching Laya's RAG claim verification prompt from 2 options (`supported`, `not supported`) to **3 options (`supported`, `contradicted`, `not mentioned`)** reduces false/hallucinated claims passing the gate from `28%` down to `11%`.

---

## 2. Verification of Official Repositories, SDKs, and Checkpoints

### 2.1 Laya Core Repository & PyPI Package (`NandhaKishorM/laya`)
* **Primary Sources:**
  * GitHub Repository: `https://github.com/NandhaKishorM/laya`
  * Official Documentation Site: `https://nandhakishorm.github.io/laya/`
  * Benchmark Evidence Branch: `https://github.com/NandhaKishorM/laya/blob/main/research/README.md`
  * PyPI Package: `https://pypi.org/project/laya/` (`v0.4.8` released October 15, 2026; `v0.4.4` shown in early October screenshot; `v0.3.6` released September 22, 2026).
* **Runtime Requirements & Extras:**
  * Requires Python `>=3.10`; core dependencies are `torch`, `transformers`, `safetensors`, `huggingface-hub`, and `numpy`.
  * Optional installation extras verified from documentation:
    * `pip install "laya[serve]"` â€” runs `laya-serve`, exposing `POST /v1/systemone` compatible with TypeSafe's Jev API. Supports bearer auth when `LAYA_API_KEY` is set (returning `401` on malformed headers and `422` on malformed questions).
    * `pip install "laya[onnx]"` â€” provides `ONNXAgent` for CPU/CUDA/DirectML inference without PyTorch runtime overhead.
    * `pip install "laya[langchain]"` â€” provides native LangChain and LangGraph decision/routing components.
    * `pip install "laya[mcp]"` â€” exposes a built-in Model Context Protocol (MCP) stdio server.
    * `pip install "laya[fast]"` â€” enables `compile=True` and the TileLang GPU fast path.
* **Verified Python SDK Signatures:**
  ```python
  import os
  os.environ["USE_TF"] = "0"  # Prevents transformers/abseil deadlock if TensorFlow is installed
  import laya
  from laya import Router

  # 1. Direct Agent Loading
  agent = laya.load("convaiinnovations/laya-typed-decisions")
  result = agent.predict(state, questions)          # Primary prediction call
  result_s1 = agent.system_one(state, questions)    # Jev-compatible alias supporting hooks
  batch_res = agent.predict_batch(states, questions) # Shared forward pass over a queue of states

  # 2. Production Router Usage
  router = Router()
  router.preload(["typed-decisions", "multilingual"])
  # Explicit routing to typed-decisions (Router does not pick typed-decisions by default unless auto_task_detection=True)
  routed_res = router.predict(state, questions, model="typed-decisions")
  # Long-document routing (up to 8,192 tokens on multilingual checkpoint)
  long_res = router.predict(long_state, questions, model="multilingual", max_len=8192)
