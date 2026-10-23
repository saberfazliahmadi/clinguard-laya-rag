"""Bidirectional option-order debiasing and option-token budget validation for Laya."""

from __future__ import annotations

from clinguard_rag.schemas import DebiasedChoiceResult


def compute_option_token_budget(num_options: int, head_max_len: int = 192) -> int:
    """
    Computes Laya's per-option token allocation: max(4, floor((head_max_len - 16) / K)).
    Keeping K <= 6 guarantees >= 29 tokens per option, preventing truncation.
    """
    if num_options <= 0:
        raise ValueError("num_options must be >= 1")
    return max(4, (head_max_len - 16) // num_options)


def validate_choice_options(options: list[str], max_options: int = 20) -> int:
    """Enforces Laya's recommended option cardinality bound (<= 20 options)."""
    if len(options) < 2:
        raise ValueError("Choice questions require at least 2 options.")
    if len(options) > max_options:
        raise ValueError(
            f"Option count ({len(options)}) exceeds safe Laya limit ({max_options}). "
            "Use coarse-to-fine hierarchical routing instead."
        )
    return compute_option_token_budget(len(options))


def combine_bidirectional_probabilities(
    options: list[str],
    forward_probs: dict[str, float],
    reverse_probs: dict[str, float],
    drift_threshold: float = 0.25,
) -> DebiasedChoiceResult:
    """
    Averages forward and reversed option-order distributions from a single forward pass
    and computes Total Variation Distance (TVD) as an epistemic instability signal.
    """
    budget = validate_choice_options(options)
    combined: dict[str, float] = {}
    tvd_sum = 0.0

    for opt in options:
        p_f = float(forward_probs.get(opt, 0.0))
        p_r = float(reverse_probs.get(opt, 0.0))
        combined[opt] = 0.5 * (p_f + p_r)
        tvd_sum += abs(p_f - p_r)

    total = sum(combined.values()) or 1.0
    normalized = {k: v / total for k, v in combined.items()}
    order_tvd = 0.5 * tvd_sum
    best_opt = max(normalized.items(), key=lambda kv: kv[1])[0]

    return DebiasedChoiceResult(
        selected_option=best_opt,
        probabilities=normalized,
        forward_probabilities={opt: float(forward_probs.get(opt, 0.0)) for opt in options},
        reverse_probabilities={opt: float(reverse_probs.get(opt, 0.0)) for opt in options},
        order_tvd_drift=round(order_tvd, 4),
        is_positionally_stable=(order_tvd <= drift_threshold),
        tokens_per_option_budget=budget,
    )
