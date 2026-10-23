# Policy-Logic Test on Golden Fixtures: Evaluation & Calibration Report

Phase 5: Evaluation & Ablation Report (`ClinGuard-RAG`)

## 1. Benchmark Methodology & Scope

The evaluation benchmark (`data/evaluation_benchmark.json`) comprises **16 clinical scenarios** grounded in 10 excerpts from public clinical guidelines (`data/clinical_guidelines_corpus.json`):
- **Calibration Split (`N = 6`, `CAL-01..CAL-06`):** Used strictly to fit the post-hoc scalar temperature parameter $T$ via grid search minimizing multi-class Negative Log-Likelihood (NLL).
- **Held-Out Test Split (`N = 10`, `EVAL-01..EVAL-10`):** Evaluates four pipeline configurations using deterministic probability fixtures (`DeterministicEvalBackend`):
  1. `baseline`: Hybrid retrieval + unverified candidate claim passthrough (no System-1 guard).
  2. `raw_system1`: Uncalibrated ($T = 1.0$), single-order (forward-only), binary 2-way claim verification (`supported` vs. `unsupported`), without explicit conflict preservation.
  3. `debiased_system1`: Adds bidirectional option-order averaging ($\Delta_{\text{order}}$ TVD instability gating at $> 0.25$), 3-way claim verification (`supported`, `contradicted`, `not_mentioned`), and Stage 3 chunk relevance gating ($\ge 0.60$) at $T = 1.0$.
  4. `clinguard_full`: Adds post-hoc temperature scaling ($T = 1.4$) and Rule 5 inter-guideline conflict preservation (`SERVE_WITH_CONFLICT_WARNING`).

## 2. Quantitative Ablation Results (Held-Out Test Split, `N = 10`)

Reproduced via `python scripts/run_evaluation.py`:
- Fitted temperature on `CAL-01..CAL-06`: **$T = 1.4$** (reducing calibration-split multi-class NLL from `0.7008` to `0.6601`).

| Pipeline Mode | Temp ($T$) | Disposition Accuracy (`N=10`) | Unsafe Claim Leakage (`N=5` unsafe) | Claim Judge Accuracy | Top-Label Brier Score ($\downarrow$) | Multi-Class NLL ($\downarrow$) | 10-Bin ECE ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | `1.0` | `40.0%` (`4/10`) | `100.0%` (`5/5`) | -- | -- | -- | -- |
| **`raw_system1`** | `1.0` | `60.0%` (`6/10`) | `60.0%` (`3/5`) | `70.0%` (`7/10`) | `0.2158` | `0.7688` | `0.2400` |
| **`debiased_system1`** | `1.0` | `80.0%` (`8/10`) | `20.0%` (`1/5`) | `80.0%` (`8/10`) | `0.1773` | `0.6458` | `0.2315` |
| **`clinguard_full`** | **`1.4`** | **`100.0%` (`10/10`)** | **`0.0%` (`0/5`)** | **`80.0%` (`8/10`)** | **`0.1668`** | **`0.6251`** | `0.2574` |

### Note on Proper Scoring Rules vs. 10-Bin ECE
Across `raw_system1` $\to$ `debiased_system1` $\to$ `clinguard_full`, strictly proper scoring rules improve monotonically: **Top-Label Brier Score** decreases from `0.2158` $\to$ `0.1773` $\to$ `0.1668`, and **Multi-Class NLL** decreases from `0.7688` $\to$ `0.6458` $\to$ `0.6251`. Temperature scaling ($T = 1.4$) cools the overconfident unsupported claim in `EVAL-09` from `0.7600` down to `0.6519 < 0.75`, eliminating the final unsafe claim leak. Because cooling also shifts high-confidence true positives on this small 10-case sample into the `0.80--0.86` confidence bin where empirical accuracy is `1.00`, 10-bin equal-width ECE increases slightly from `0.2315` to `0.2574`.

## 3. Limitations
1. **Fixture-Based Policy & Calibration Verification:** The benchmark replays structured probability fixtures (`DeterministicEvalBackend`) to test policy and calibration math deterministically in CI without requiring GPU weights.
2. **Sample Size:** The 16-case dataset is designed for architectural verification and unit testing rather than clinical statistical power.
