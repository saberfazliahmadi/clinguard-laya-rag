"""Unit tests for token budget math, bidirectional debiasing, temperature calibration, and PHI scrubbing."""

from __future__ import annotations

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
    assert validate_choice_options(["a", "b", "c"]) == 58
    with pytest.raises(ValueError):
        validate_choice_options([f"opt_{i}" for i in range(25)])


def test_bidirectional_debiasing_flags_positional_instability() -> None:
    options = ["supported", "contradicted", "not_mentioned"]
    fwd = {"supported": 0.78, "contradicted": 0.14, "not_mentioned": 0.08}
    rev = {"supported": 0.12, "contradicted": 0.80, "not_mentioned": 0.08}

    res = combine_bidirectional_probabilities(options, fwd, rev, drift_threshold=0.25)
    assert res.selected_option == "contradicted"
    assert res.order_tvd_drift > 0.25
    assert res.is_positionally_stable is False


def test_temperature_calibrator_fits_and_reduces_nll_and_overconfidence() -> None:
    cases = json.loads((DATA_DIR / "evaluation_benchmark.json").read_text(encoding="utf-8"))
    cal_cases = [c for c in cases if c["split"] == "calibration"]
    dists = [c["raw_model_probs"]["claim_fwd"] for c in cal_cases]
    labels = [c["claim_label"] for c in cal_cases]

    calibrator = TemperatureCalibrator()
    fitted_t = calibrator.fit(dists, labels)
    assert fitted_t == 1.4

    nll_before = multiclass_nll(dists, labels, temperature=1.0)
    nll_after = multiclass_nll(dists, labels, temperature=fitted_t)
    assert nll_after < nll_before

    conf_before, conf_after, corr = [], [], []
    for d, true_lbl in zip(dists, labels):
        pred_lbl = max(d.items(), key=lambda kv: kv[1])[0]
        scaled_d = apply_temperature_to_distribution(d, fitted_t)
        conf_before.append(float(d[pred_lbl]))
        conf_after.append(float(scaled_d[pred_lbl]))
        corr.append(1 if pred_lbl == true_lbl else 0)

    ece_before, brier_before = compute_ece_and_brier(conf_before, corr, num_bins=10)
    ece_after, brier_after = compute_ece_and_brier(conf_after, corr, num_bins=10)
    assert ece_after < ece_before
    assert brier_after < brier_before


def test_phi_redaction_and_question_dict_normalization() -> None:
    sample = (
        "Pt John Smith DOB 1961-04-03 MRN 00481923 lives 12 Elm St, "
        "call +49 40 123456; eGFR trend 120 130 1500; a@b.com; "
        "tel 040 1234 5678; (555) 123-4567; 555-123-4567; "
        "Patient: Maria Lopez; Dr. Jane Miller reviewed; "
        "she lives at Hauptstrasse 5, 20095 Hamburg; Tel 040 1234; "
        "Maria Lopez presented with chest pain; her husband Peter Braun called; contact 0176 12345678; "
        "Anna Schmidt was seen yesterday; John Doe is stable; "
        "Klaus Weber, 54, has diabetes; Anna Schmidt takes metformin daily; "
        "Contact Anna Schmidt at home; Anna Schmidt, DOB 12.03.1970; "
        "Anna Schmidt: metformin 500 mg; Her name is Anna Schmidt. "
        "Peter Braun is married; Anna Schmidt was my neighbour; Frank Mild takes it; "
        "Heart Failure has severe symptoms; Acute Kidney had injury; Hamburg Hospital requires dose"
    )
    scrubbed = redact_phi_hook(sample)
    assert "Maria Lopez" not in scrubbed
    assert "Jane Miller" not in scrubbed
    assert "Hauptstrasse 5" not in scrubbed
    assert "040 1234" not in scrubbed
    assert "Peter Braun" not in scrubbed
    assert "0176 12345678" not in scrubbed
    assert "Anna Schmidt" not in scrubbed
    assert "John Doe" not in scrubbed
    assert "Klaus Weber" not in scrubbed
    assert "12.03.1970" not in scrubbed
    assert "Frank Mild" not in scrubbed
    assert "Heart Failure has severe" in scrubbed
    assert "Acute Kidney had injury" in scrubbed
    assert "Hamburg Hospital requires dose" in scrubbed
    assert "[REDACTED_NAME]" in scrubbed and "John Smith" not in scrubbed
    assert "[REDACTED_ADDRESS]" in scrubbed and "12 Elm St" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed and "a@b.com" not in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "+49 40 123456" not in scrubbed
    assert "040 1234 5678" not in scrubbed
    assert "(555) 123-4567" not in scrubbed
    assert "555-123-4567" not in scrubbed
    assert "[REDACTED_MRN]" in scrubbed
    assert "[REDACTED_DOB]" in scrubbed
    assert "120 130 1500" in scrubbed

    q_list = [
        {"id": "q1", "type": "choice", "question": "Pick one", "options": ["a", "b"]},
        {"id": "q2", "type": "score", "question": "Rate urgency", "levels": ["0", "1", "2", "3"]},
    ]
    norm = normalize_questions_dict(q_list)
    assert "q1" in norm
    assert norm["q1"]["instructions"] == "Pick one"
    assert "criteria" in norm["q1"] and "options" not in norm["q1"]
    assert "criteria" in norm["q2"] and "levels" not in norm["q2"]
