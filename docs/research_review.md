# Literature & Technical Research Review

## 1. Primary System-1 Decision Model Sources

1. **Laya Open-Weight Decision Engine (Convai Innovations):**
   - Built on `ModernBERT-large` (421M parameters), exposing three structured decision primitives (`choice`, `score`, `noul`) via single forward-pass heads rather than autoregressive token generation.
   - Enforces a 192-token total option budget across `choice` options (16 tokens reserved for special formatting tokens).

2. **TypeSafe AI System-One & Jev API (`/v1/systemone`):**
   - Commercial System-1 decision service exposing calibrated probabilities over structured `choice`, `score`, and `noul` schemas.

3. **Paired Evaluation of System-1 Harnesses (`arXiv:2610.02267v1 [cs.AI]`):**
   - *Abstract Summary:* Evaluates System-1 decision models inside LLM agent harnesses, reporting that open-weight Laya flips its categorical selection on roughly 30% of queries when option ordering is reversed, whereas commercial System-1 models remain substantially more order-stable, and both models tie on basic relevance gating.
   - *Implications for ClinGuard-RAG:* Motivates bidirectional option-order debiasing (`combine_bidirectional_probabilities`), explicit Total Variation Distance ($\Delta_{\text{order}} \le 0.25$) stability checks, and post-hoc temperature calibration ($T = 1.4$).

4. **Option-Name vs. Bound-Definition Bias (`arXiv:2609.26758v3 [cs.CL]`):**
   - *Abstract Summary:* Demonstrates that typed decision models follow canonical option names rather than arbitrary contradictory definitions bound to those names (with accuracy dropping up to 70.4 percentage points and AUC dropping from 93.8% to 23.2% under adversarial definition swaps).
   - *Implications for ClinGuard-RAG:* All `choice` schemas in Stage 1 and Stage 4 use self-explanatory, canonical clinical option names (`supported`, `contradicted`, `not_mentioned`) rather than opaque labels (`Option A`, `Option B`).

5. **Acronym Disambiguation (`arXiv:2511.12723v2 [cs.LG]`):**
   - Gennaro Vessio's paper *"LAYA: Layer-wise Attention Aggregation for Interpretable Depth-Aware Neural Networks"* shares the LAYA acronym but addresses vision/depth-aggregation architectures and is unrelated to Convai Innovations' System-1 decision engine.

## 2. Implementation Safeguards Derived from Literature

```python
# Canonical option naming + bidirectional order debiasing + token budget check
from clinguard_rag.debiasing import combine_bidirectional_probabilities, validate_choice_options

options = ["supported", "contradicted", "not_mentioned"]
budget = validate_choice_options(options)  # 58 tokens per option (<= 192 total budget)
result = combine_bidirectional_probabilities(options, forward_probs, reverse_probs, drift_threshold=0.25)
```

## 3. Summary of Architectural Mitigations

| Failure Mode Identified in Literature | ClinGuard-RAG Mitigation | Module |
| :--- | :--- | :--- |
| Option-order sensitivity (~30% flip rate) | Forward/reverse option averaging + TVD drift gate ($\Delta_{\text{order}} \le 0.25$) | `src/clinguard_rag/debiasing.py` |
| Option-name vs. bound-definition bias | Self-explanatory canonical clinical option names | `src/clinguard_rag/router.py`, `verifier.py` |
| 192-token option budget truncation | Automatic token budget validation ($K \le 20$) | `src/clinguard_rag/debiasing.py` |
| Uncalibrated confidence scores | Post-hoc Temperature Scaling ($T = 1.4$) fitted via NLL grid search | `src/clinguard_rag/calibration.py` |
