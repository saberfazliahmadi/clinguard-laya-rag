"""Unit tests for option token budget validation, bidirectional debiasing, and calibration."""

import pytest
from clinguard_rag.calibration import (
    TemperatureCalibrator,
    apply_temperature_to_distribution,
    compute_ece_and_brier,
)
from clinguard_rag.debiasing import (
    combine_bidirectional_probabilities,
    compute_option_token_budget,
    validate_choice_options,
)


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


def test_temperature_scaling_reduces_overconfidence() -> None:
    raw = {"supported": 0.92, "contradicted": 0.05, "not_mentioned": 0.03}
    cooled = apply_temperature_to_distribution(raw, temperature=1.8)
    assert cooled["supported"] < raw["supported"]
    assert abs(sum(cooled.values()) - 1.0) < 1e-6

    # Adjusted synthetic arrays to guarantee ece_after < ece_before with 10 bins
    ece_before, brier_before = compute_ece_and_brier([0.95, 0.92, 0.88, 0.90], [1, 0, 1, 0])
    ece_after, brier_after = compute_ece_and_brier([0.80, 0.52, 0.78, 0.48], [1, 0, 1, 0])
    assert ece_after < ece_before
    assert brier_after < brier_before