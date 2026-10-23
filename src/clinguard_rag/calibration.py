"""Post-hoc Temperature Scaling and calibration metrics (ECE, Top-Label Brier Score, NLL)."""

from __future__ import annotations

import math

import numpy as np


def multiclass_nll(dists: list[dict[str, float]], labels: list[str], temperature: float = 1.0) -> float:
    total_loss = 0.0
    eps = 1e-15
    for dist, label in zip(dists, labels):
        cooled = apply_temperature_to_distribution(dist, temperature)
        prob = max(eps, cooled.get(label, eps))
        total_loss -= math.log(prob)
    return total_loss / max(1, len(dists))


def apply_temperature_to_distribution(
    probs: dict[str, float], temperature: float
) -> dict[str, float]:
    """Applies temperature scaling T > 0 to a categorical probability dictionary."""
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive.")
    eps = 1e-8
    keys = list(probs.keys())
    p_arr = np.clip(np.array([probs[k] for k in keys], dtype=np.float64), eps, 1.0)
    logits = np.log(p_arr) / temperature
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    scaled = exp_logits / np.sum(exp_logits)
    return {k: float(scaled[i]) for i, k in enumerate(keys)}


def apply_temperature_to_binary(prob: float, temperature: float) -> float:
    """Applies temperature scaling T > 0 to a binary noul probability."""
    dist = apply_temperature_to_distribution({"true": prob, "false": 1.0 - prob}, temperature)
    return dist["true"]


def compute_ece_and_brier(
    confidences: list[float], correctness: list[int], num_bins: int = 10
) -> tuple[float, float]:
    """
    Computes Expected Calibration Error (ECE) across equal-width bins and
    Top-Label Confidence Brier Score: mean((top_confidence - is_correct)^2).
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    if len(conf) == 0:
        return 0.0, 0.0

    brier = float(np.mean((conf - corr) ** 2))
    bin_Boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(conf)

    for i in range(num_bins):
        low, high = bin_Boundaries[i], bin_Boundaries[i + 1]
        if i == 0:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf > low) & (conf <= high)
        if np.any(mask):
            bin_acc = float(np.mean(corr[mask]))
            bin_conf = float(np.mean(conf[mask]))
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)

    return round(float(ece), 4), round(brier, 4)


class TemperatureCalibrator:
    """Fits a scalar temperature T on held-out calibration predictions via NLL grid search."""

    def __init__(
        self,
        initial_temperature: float = 1.0,
        temperature: float | None = None,
    ) -> None:
        target_temp = temperature if temperature is not None else initial_temperature
        self.temperature: float = target_temp

    def fit(self, distributions: list[dict[str, float]], true_labels: list[str]) -> float:
        best_t = 1.0
        best_nll = float("inf")
        eps = 1e-8

        for t in np.linspace(0.5, 4.0, 141):
            nll = 0.0
            for dist, label in zip(distributions, true_labels):
                scaled = apply_temperature_to_distribution(dist, float(t))
                p_true = max(eps, scaled.get(label, eps))
                nll -= np.log(p_true)
            nll /= max(1, len(distributions))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        self.temperature = round(best_t, 4)
        return self.temperature
