"""Unit tests for option token budget validation, bidirectional debiasing, calibration, and PHI scrubbing."""

import json
from pathlib import Path
import pytest
from clinguard_rag.calibration import (
    TemperatureCalibrator,
    apply_temperature_to_distribution,
    compute_ece_and_brier,
    multiclass_nll,
)
from clinguard_rag.debiasing import (
    combine_bidirectional_probabilities,
    compute_option_token_budget,
    validate_choice_options,
)
from clinguard_rag.decision_engine import normalize_questions_dict, redact_phi_hook

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_option_token_budget_formula() -> None:
    assert compute_option_token_budget(3) == 58
    assert compute_option_token_budget(5) == 35
    assert compute_option_token_budget(77) == 4
    with pytest.raises(ValueError):
        validate_choice_options([f"opt_{i}" for i in range(25)], max_options=20)


def test_bidirectional_debiasing_flags_positional_instability() -> None:
    options = ["supported", "contradicted", "not_mentioned"]
    fwd = {"supported": 0.74, "contradicted": 0.20, "not_mentioned": 0.06}
    rev = {"supported": 0.16, "contradicted": 0.78, "not_mentioned": 0.06}
    res = combine_bidirectional_probabilities(options, fwd, rev, drift_threshold=0.25)

    assert res.selected_option == "contradicted"
    assert res.order_tvd_drift > 0.50
    assert res.is_positionally_stable is False


def test_temperature_calibrator_fits_and_reduces_nll_and_overconfidence() -> None:
    raw = {"supported": 0.92, "contradicted": 0.05, "not_mentioned": 0.03}
    cooled = apply_temperature_to_distribution(raw, temperature=1.8)
    assert cooled["supported"] < raw["supported"]
    assert abs(sum(cooled.values()) - 1.0) < 1e-6

    cases = json.loads((DATA_DIR / "evaluation_benchmark.json").read_text(encoding="utf-8"))
    cal_cases = [c for c in cases if c["split"] == "calibration"]
    dists = [c["raw_model_probs"]["claim_fwd"] for c in cal_cases]
    labels = [c["claim_label"] for c in cal_cases]

    nll_before = multiclass_nll(dists, labels, temperature=1.0)
    calibrator = TemperatureCalibrator()
    fitted_t = calibrator.fit(dists, labels)
    nll_after = multiclass_nll(dists, labels, temperature=fitted_t)

    assert fitted_t == 1.4
    assert nll_after < nll_before

    ece_before, brier_before = compute_ece_and_brier([0.95, 0.92, 0.88, 0.90], [1, 0, 1, 0])
    ece_after, brier_after = compute_ece_and_brier([0.80, 0.52, 0.78, 0.48], [1, 0, 1, 0])
    assert ece_after < ece_before
    assert brier_after < brier_before


def test_phi_redaction_and_question_dict_normalization() -> None:
    sample = "Pt email a@b.com DOB 1961-04-03 MRN 00481923 call +49 40 123456, eGFR trend 120 130 1500"
    scrubbed = redact_phi_hook(sample)
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_DOB]" in scrubbed
    assert "[REDACTED_MRN]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "eGFR trend 120 130 1500" in scrubbed

    q_list = [{"id": "q1", "type": "noul", "instructions": "test"}]
    q_dict = normalize_questions_dict(q_list)
    assert "q1" in q_dict and q_dict["q1"]["type"] == "noul"