"""Stage 1 Pre-Retrieval Clinical Triage and Two-Step KB Fan-Out Router."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from clinguard_rag.calibration import (
    apply_temperature_to_binary,
    apply_temperature_to_distribution,
)
from clinguard_rag.debiasing import combine_bidirectional_probabilities
from clinguard_rag.schemas import DebiasedChoiceResult, TriageDecision

CLINICAL_DOMAIN_OPTIONS: List[str] = [
    "antimicrobial_stewardship",
    "cardiovascular_hypertension",
    "endocrine_diabetes",
    "renal_dosing_contraindications",
    "out_of_scope",
]


def evaluate_stage1_triage(
    raw_probs: Dict[str, Any],
    temperature: float = 1.0,
    use_bidirectional_debiasing: bool = True,
    use_bidirectional: Optional[bool] = None,
    out_of_scope_threshold: float = 0.60,
    unsafe_threshold: float = 0.65,
    single_kb_threshold: float = 0.55,
) -> TriageDecision:
    """
    Executes Stage 1 Triage (`noul` safety check, `score` urgency, `choice` domain routing)
    and applies the two-step KB fan-out rule:
      1. Skip retrieval only if P(out_of_scope) >= 0.60 or P(prompt_attack) >= 0.65.
      2. Renormalize over the 4 clinical domains; route to 1 domain if top >= 0.55, else top-2 fan-out.
    """
    if use_bidirectional is not None:
        use_bidirectional_debiasing = use_bidirectional

    attack_prob = apply_temperature_to_binary(
        float(raw_probs.get("prompt_attack_or_unsafe", 0.0)), temperature
    )

    u_raw_source = (
        raw_probs.get("clinical_urgency_probs")
        or raw_probs.get("urgency_score_probs")
        or [0.7, 0.2, 0.08, 0.02]
    )
    u_raw = [float(x) for x in u_raw_source]
    u_sum = sum(u_raw) or 1.0
    u_probs = [x / u_sum for x in u_raw]
    expected_urgency = sum(idx * p for idx, p in enumerate(u_probs))

    fwd_dom = apply_temperature_to_distribution(raw_probs.get("domain_fwd", {}), temperature)
    rev_dom = apply_temperature_to_distribution(raw_probs.get("domain_rev", {}), temperature)

    if use_bidirectional_debiasing and fwd_dom and rev_dom:
        domain_choice = combine_bidirectional_probabilities(
            CLINICAL_DOMAIN_OPTIONS, fwd_dom, rev_dom
        )
    else:
        dist = fwd_dom if fwd_dom else {d: 0.2 for d in CLINICAL_DOMAIN_OPTIONS}
        best_opt = max(dist.items(), key=lambda kv: kv[1])[0]
        domain_choice = DebiasedChoiceResult(
            selected_option=best_opt,
            probabilities=dist,
            forward_probabilities=dist,
            reverse_probabilities=rev_dom if rev_dom else dist,
            order_tvd_drift=0.0,
            is_positionally_stable=True,
            tokens_per_option_budget=35,
        )

    p_oos = domain_choice.probabilities.get("out_of_scope", 0.0)
    skip = (p_oos >= out_of_scope_threshold) or (attack_prob >= unsafe_threshold)

    clinical_only = {
        d: domain_choice.probabilities[d]
        for d in CLINICAL_DOMAIN_OPTIONS
        if d != "out_of_scope"
    }
    clin_total = sum(clinical_only.values()) or 1.0
    norm_clin = {d: p / clin_total for d, p in clinical_only.items()}
    ranked_domains = sorted(norm_clin.items(), key=lambda kv: kv[1], reverse=True)

    if skip:
        routed_domains: List[str] = []
    elif ranked_domains and ranked_domains[0][1] >= single_kb_threshold:
        routed_domains = [ranked_domains[0][0]]
    elif len(ranked_domains) >= 2:
        routed_domains = [ranked_domains[0][0], ranked_domains[1][0]]
    elif ranked_domains:
        routed_domains = [ranked_domains[0][0]]
    else:
        routed_domains = []

    return TriageDecision(
        prompt_attack_prob=round(attack_prob, 4),
        urgency_expected_score=round(expected_urgency, 4),
        urgency_probabilities=[round(p, 4) for p in u_probs],
        domain_choice=domain_choice,
        routed_domains=routed_domains,
        skip_retrieval=skip,
    )


# Alias to support both function names across pipeline and tests
run_stage1_triage = evaluate_stage1_triage