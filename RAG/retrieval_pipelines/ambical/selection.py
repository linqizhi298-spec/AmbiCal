"""Risk-aware evidence competition and AmbiCal adaptive stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .models import ChunkRecord, EvidenceItem
from .numerics import (
    as_finite_vector,
    robust_sigmoid_map,
    robust_z_scores,
    round_half_up,
    stable_softmax,
)


@dataclass(frozen=True)
class ContextGateResult:
    observed_indices: Tuple[int, ...]
    competition: float
    weighted_distance: float
    context_split: float
    route_total_variation: float
    path_conflict: float
    evidence_strength: float
    risk: float
    triggered: bool
    # v2-only diagnostics; None keeps v1 records byte-comparable.
    bimodality: Optional[float] = None
    branch_disagreement: Optional[float] = None
    linear_predictor: Optional[float] = None

    def diagnostics(self) -> Dict[str, Any]:
        payload = {
            "observed_indices": list(self.observed_indices),
            "competition": self.competition,
            "weighted_distance": self.weighted_distance,
            "context_split": self.context_split,
            "route_total_variation": self.route_total_variation,
            "path_conflict": self.path_conflict,
            "evidence_strength": self.evidence_strength,
            "risk": self.risk,
            "triggered": self.triggered,
        }
        if self.bimodality is not None:
            payload["bimodality"] = self.bimodality
        if self.branch_disagreement is not None:
            payload["branch_disagreement"] = self.branch_disagreement
        if self.linear_predictor is not None:
            payload["linear_predictor"] = self.linear_predictor
        return payload


@dataclass(frozen=True)
class GreedySelection:
    indices: Tuple[int, ...]
    marginal_utilities: Tuple[float, ...]
    redundancies: Tuple[float, ...]

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "indices": list(self.indices),
            "marginal_utilities": list(self.marginal_utilities),
            "redundancies": list(self.redundancies),
        }


@dataclass(frozen=True)
class AdaptiveKResult:
    selected_k: int
    adaptive_k: int
    entropy_k: int
    gap_k: int
    gap_strength: float
    raw_entropy: float
    corrected_entropy: float
    mapped_utilities: Tuple[float, ...]
    probabilities: Tuple[float, ...]
    evidence_strength: float
    null_probability: Optional[float]
    null_region: str

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "selected_k": self.selected_k,
            "adaptive_k": self.adaptive_k,
            "entropy_k": self.entropy_k,
            "gap_k": self.gap_k,
            "gap_strength": self.gap_strength,
            "raw_entropy": self.raw_entropy,
            "corrected_entropy": self.corrected_entropy,
            "mapped_utilities": list(self.mapped_utilities),
            "probabilities": list(self.probabilities),
            "evidence_strength": self.evidence_strength,
            "null_probability": self.null_probability,
            "null_region": self.null_region,
        }


def _weighted_pair_distance(
    weights: torch.Tensor, distances: torch.Tensor, epsilon: float
) -> float:
    count = int(weights.numel())
    numerator = torch.tensor(0.0, dtype=torch.float64)
    denominator = torch.tensor(0.0, dtype=torch.float64)
    for left in range(count):
        for right in range(left + 1, count):
            pair_weight = weights[left] * weights[right]
            numerator += pair_weight * distances[left, right]
            denominator += pair_weight
    return float((numerator / (denominator + epsilon)).item())


def evidence_strength(
    utilities: Sequence[float] | torch.Tensor,
    *,
    epsilon: float,
    z_max: float,
    iqr_normal_consistency: float,
) -> Tuple[float, Dict[str, Any]]:
    values = as_finite_vector(utilities, "evidence_utilities")
    top = float(torch.max(values).item())
    z_values, stats = robust_z_scores(
        values,
        epsilon=epsilon,
        z_max=z_max,
        iqr_normal_consistency=iqr_normal_consistency,
    )
    top_index = int(torch.argmax(values).item())
    strength = min(1.0, max(0.0, top)) * float(torch.sigmoid(z_values[top_index]).item())
    return strength, {"top_utility": top, "robust_stats": stats.diagnostics()}


def context_gate(
    initial_scores: Sequence[float] | torch.Tensor,
    source_branch_support: Sequence[float] | torch.Tensor,
    target_branch_support: Optional[Sequence[float] | torch.Tensor],
    pairwise_cosine: torch.Tensor,
    *,
    chi: float,
    top_m: int,
    temperature: float,
    threshold: float,
    epsilon: float,
    z_max: float,
    iqr_normal_consistency: float,
) -> ContextGateResult:
    f0 = as_finite_vector(initial_scores, "initial_scores")
    source = as_finite_vector(source_branch_support, "source_branch_support")
    if source.numel() != f0.numel():
        raise ValueError("Source support and initial score lengths differ.")
    target = None
    if target_branch_support is not None:
        target = as_finite_vector(target_branch_support, "target_branch_support")
        if target.numel() != f0.numel():
            raise ValueError("Target support and initial score lengths differ.")
    if pairwise_cosine.shape != (f0.numel(), f0.numel()):
        raise ValueError("pairwise_cosine has the wrong shape.")

    observed_count = min(max(1, int(top_m)), int(f0.numel()))
    # Stable sort keeps the original union rank for equal scores.
    order = torch.argsort(f0, descending=True, stable=True)[:observed_count]
    observed = tuple(int(value) for value in order.tolist())
    gate_weights = stable_softmax(f0[order], temperature)

    if observed_count == 1:
        competition = 0.0
    else:
        concentration = float(torch.sum(gate_weights.square()).item())
        competition = (1.0 - concentration) / (1.0 - 1.0 / observed_count)
        competition = min(1.0, max(0.0, competition))

    local_cosine = pairwise_cosine[order][:, order]
    local_distances = torch.clamp((1.0 - local_cosine) / 2.0, 0.0, 1.0)
    weighted_distance = _weighted_pair_distance(
        gate_weights, local_distances, epsilon
    )
    context_split = competition * weighted_distance

    if target is None:
        route_tv = 0.0
    else:
        source_distribution = stable_softmax(source[order], temperature)
        target_distribution = stable_softmax(target[order], temperature)
        route_tv = 0.5 * float(
            torch.sum(torch.abs(source_distribution - target_distribution)).item()
        )
    path_conflict = (route_tv + 1.0 - float(chi)) / 2.0
    path_conflict = min(1.0, max(0.0, path_conflict))
    strength, _ = evidence_strength(
        f0,
        epsilon=epsilon,
        z_max=z_max,
        iqr_normal_consistency=iqr_normal_consistency,
    )
    risk = strength * context_split * (1.0 + path_conflict) / 2.0
    return ContextGateResult(
        observed_indices=observed,
        competition=competition,
        weighted_distance=weighted_distance,
        context_split=context_split,
        route_total_variation=route_tv,
        path_conflict=path_conflict,
        evidence_strength=strength,
        risk=risk,
        triggered=risk >= threshold,
    )


def _top_m_bimodality(observed_scores: torch.Tensor, epsilon: float) -> float:
    """Separation of the top-M scores into two clusters.

    A single dominant sense yields one head plus a smooth tail (low value);
    two competing senses yield two score levels separated by the largest
    internal gap (high value).  ``balance`` discounts splits that isolate a
    single outlier from the rest.
    """

    count = int(observed_scores.numel())
    if count < 2:
        return 0.0
    ordered = torch.sort(observed_scores, descending=True).values
    gaps = ordered[:-1] - ordered[1:]
    split = int(torch.argmax(gaps).item())
    score_range = float((ordered[0] - ordered[-1]).item())
    separation = float(gaps[split].item()) / (score_range + epsilon)
    balance = min(split + 1, count - (split + 1)) / max(1, count // 2)
    return min(1.0, max(0.0, separation)) * min(1.0, balance)


def _branch_rank_disagreement(
    observed: Tuple[int, ...],
    source_support: torch.Tensor,
    target_support: Optional[torch.Tensor],
    epsilon: float,
) -> float:
    """Mean normalized rank distance between the two branch orderings.

    Two queries agreeing on the evidence order produce near-zero values;
    each branch ranking the same chunks differently (typical for a term with
    competing senses) produces large values.
    """

    if target_support is None:
        return 0.0
    total = int(source_support.numel())
    if total < 2:
        return 0.0
    source_rank = torch.argsort(torch.argsort(source_support, descending=True))
    target_rank = torch.argsort(torch.argsort(target_support, descending=True))
    differences = [
        abs(int(source_rank[i].item()) - int(target_rank[i].item())) / (total - 1)
        for i in observed
    ]
    return float(sum(differences) / (len(differences) + epsilon))


def context_gate_v2(
    initial_scores: Sequence[float] | torch.Tensor,
    source_branch_support: Sequence[float] | torch.Tensor,
    target_branch_support: Optional[Sequence[float] | torch.Tensor],
    pairwise_cosine: torch.Tensor,
    *,
    chi: float,
    top_m: int,
    temperature: float,
    threshold: float,
    epsilon: float,
    z_max: float,
    iqr_normal_consistency: float,
    bias: float,
    weight_strength: float,
    weight_competition: float,
    weight_distance: float,
    weight_path_conflict: float,
    weight_bimodality: float,
    weight_branch_disagreement: float,
) -> ContextGateResult:
    """Calibratable context gate for ``ambical-online-v2``.

    The v1 multiplicative form ``strength * context_split * (1+conflict)/2``
    compresses its output into roughly 0.05-0.08 on observed data, making the
    preregistered 0.20 threshold unreachable; the v2 form keeps every feature
    in [0, 1] and maps their weighted sum through a sigmoid so the threshold
    stays meaningful and the weights can be fitted on gate development labels.
    """

    f0 = as_finite_vector(initial_scores, "initial_scores")
    source = as_finite_vector(source_branch_support, "source_branch_support")
    target = None
    if target_branch_support is not None:
        target = as_finite_vector(target_branch_support, "target_branch_support")
    if source.numel() != f0.numel():
        raise ValueError("Source support and initial score lengths differ.")
    if target is not None and target.numel() != f0.numel():
        raise ValueError("Target support and initial score lengths differ.")
    if pairwise_cosine.shape != (f0.numel(), f0.numel()):
        raise ValueError("pairwise_cosine has the wrong shape.")

    observed_count = min(max(1, int(top_m)), int(f0.numel()))
    order = torch.argsort(f0, descending=True, stable=True)[:observed_count]
    observed = tuple(int(value) for value in order.tolist())
    gate_weights = stable_softmax(f0[order], temperature)

    if observed_count == 1:
        competition = 0.0
    else:
        concentration = float(torch.sum(gate_weights.square()).item())
        competition = (1.0 - concentration) / (1.0 - 1.0 / observed_count)
        competition = min(1.0, max(0.0, competition))

    local_cosine = pairwise_cosine[order][:, order]
    local_distances = torch.clamp((1.0 - local_cosine) / 2.0, 0.0, 1.0)
    weighted_distance = _weighted_pair_distance(gate_weights, local_distances, epsilon)

    if target is None:
        route_tv = 0.0
    else:
        source_distribution = stable_softmax(source[order], temperature)
        target_distribution = stable_softmax(target[order], temperature)
        route_tv = 0.5 * float(
            torch.sum(torch.abs(source_distribution - target_distribution)).item()
        )
    path_conflict = (route_tv + 1.0 - float(chi)) / 2.0
    path_conflict = min(1.0, max(0.0, path_conflict))
    strength, _ = evidence_strength(
        f0,
        epsilon=epsilon,
        z_max=z_max,
        iqr_normal_consistency=iqr_normal_consistency,
    )
    bimodality = _top_m_bimodality(f0[order], epsilon)
    branch_disagreement = _branch_rank_disagreement(observed, source, target, epsilon)

    linear = (
        float(bias)
        + float(weight_strength) * strength
        + float(weight_competition) * competition
        + float(weight_distance) * weighted_distance
        + float(weight_path_conflict) * path_conflict
        + float(weight_bimodality) * bimodality
        + float(weight_branch_disagreement) * branch_disagreement
    )
    risk = float(torch.sigmoid(torch.tensor(linear, dtype=torch.float64)).item())
    return ContextGateResult(
        observed_indices=observed,
        competition=competition,
        weighted_distance=weighted_distance,
        context_split=competition * weighted_distance,
        route_total_variation=route_tv,
        path_conflict=path_conflict,
        evidence_strength=strength,
        risk=risk,
        triggered=risk >= threshold,
        bimodality=bimodality,
        branch_disagreement=branch_disagreement,
        linear_predictor=linear,
    )


def symmetric_correction_factors(
    signed_dependency: Sequence[float] | torch.Tensor,
    *,
    lambda_context: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Symmetric (promote/demote) correction factors for method v2.

    v1 only ever down-weights a chunk by ``1 - lambda * risk``, which cannot
    promote the suppressed correct sense and left 28/30 forced corrections
    neutral.  v2 consumes the signed robust z of ``raw - counterfactual``
    (aggregated over the selected phrases by the caller): chunks whose score
    collapses without the phrase are demoted, chunks that the phrase was
    suppressing are promoted, so the ranking can move in both directions.
    Factors stay within ``[1 - lambda, 1 + lambda]``.
    """

    signed = as_finite_vector(signed_dependency, "signed_dependency")
    signed = torch.clamp(signed, -1.0, 1.0)
    factors = (
        1.0
        - float(lambda_context) * torch.relu(signed)
        + float(lambda_context) * torch.relu(-signed)
    )
    factors = torch.clamp(
        factors, 1.0 - float(lambda_context), 1.0 + float(lambda_context)
    )
    return factors, {
        "signed_dependency": [float(value) for value in signed.tolist()],
        "factors": [float(value) for value in factors.tolist()],
    }


def positive_context_promotion(
    original_fused_scores: Sequence[float] | torch.Tensor,
    counterfactual_fused_views: Sequence[Sequence[float] | torch.Tensor],
    *,
    lambda_positive: float,
    epsilon: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Promote one context-consistent candidate without demoting any candidate.

    Method v3 requires two independent counterfactual views.  Each view first
    combines the original-query relevance with the remaining-context support;
    promotion is allowed only when both views select the same candidate.  The
    agreed candidate receives a bounded additive bonus proportional to its
    continuous advantage over the runner-up.  There is deliberately no hard
    candidate-margin threshold.
    """

    original = torch.clamp(
        as_finite_vector(original_fused_scores, "original_fused_scores"),
        0.0,
        1.0,
    )
    if not 0.0 <= float(lambda_positive) <= 1.0:
        raise ValueError("lambda_positive must be in [0, 1].")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be a finite positive number.")

    views = [
        torch.clamp(as_finite_vector(view, f"counterfactual_view_{index}"), 0.0, 1.0)
        for index, view in enumerate(counterfactual_fused_views)
    ]
    if any(view.numel() != original.numel() for view in views):
        raise ValueError(
            "Counterfactual context views must match the original score length."
        )

    diagnostics: Dict[str, Any] = {
        "promotion_executed": False,
        "promotion_reason": "insufficient_valid_phrases",
        "valid_phrase_count": len(views),
        "view_winners": [],
        "phrase_agreement": False,
        "promoted_candidate_index": None,
        "promotion_amount": 0.0,
        "candidate_margin": 0.0,
        "top1_before": None,
        "top1_after": None,
        "counterfactual_context_scores": [
            [float(value) for value in view.tolist()] for view in views
        ],
        "semantic_alignment_views": [],
        "semantic_alignment_scores": None,
    }
    if original.numel() == 0:
        diagnostics["promotion_reason"] = "no_candidate"
        return original, diagnostics

    original_top = int(torch.argmax(original).item())
    diagnostics["top1_before"] = original_top
    diagnostics["top1_after"] = original_top
    if len(views) != 2:
        return original.clone(), diagnostics

    alignment_views = [torch.sqrt(torch.clamp(original * view, 0.0, 1.0)) for view in views]
    winners = [int(torch.argmax(alignment).item()) for alignment in alignment_views]
    diagnostics["view_winners"] = winners
    diagnostics["semantic_alignment_views"] = [
        [float(value) for value in alignment.tolist()]
        for alignment in alignment_views
    ]
    if winners[0] != winners[1]:
        diagnostics["promotion_reason"] = "phrase_disagreement"
        return original.clone(), diagnostics

    diagnostics["phrase_agreement"] = True
    winner = winners[0]
    mean_context = torch.mean(torch.stack(views), dim=0)
    alignment = torch.sqrt(torch.clamp(original * mean_context, 0.0, 1.0))
    diagnostics["semantic_alignment_scores"] = [
        float(value) for value in alignment.tolist()
    ]

    if original.numel() == 1:
        margin = 0.0
    else:
        competitor_mask = torch.ones(original.numel(), dtype=torch.bool)
        competitor_mask[winner] = False
        margin = max(
            0.0,
            float(alignment[winner].item())
            - float(torch.max(alignment[competitor_mask]).item()),
        )
    diagnostics["candidate_margin"] = margin
    if winner == original_top:
        diagnostics["promotion_reason"] = "aligned_noop"
        return original.clone(), diagnostics

    bonus = float(lambda_positive) * (1.0 - float(original[winner].item())) * margin
    if bonus <= float(epsilon):
        diagnostics["promotion_reason"] = "zero_bonus"
        return original.clone(), diagnostics

    promoted = original.clone()
    promoted[winner] = min(1.0, float(promoted[winner].item()) + bonus)
    diagnostics.update(
        {
            "promotion_executed": True,
            "promotion_reason": "promoted",
            "promoted_candidate_index": winner,
            "promotion_amount": float(promoted[winner].item())
            - float(original[winner].item()),
            "top1_after": int(torch.argmax(promoted).item()),
        }
    )
    return promoted, diagnostics


def reranker_promotion(
    original_fused_scores: Sequence[float] | torch.Tensor,
    relevance_scores: Sequence[float] | torch.Tensor,
    observed_indices: Sequence[int],
    *,
    lambda_positive: float,
    threshold: float,
    top_m: int,
    epsilon: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Bounded cross-encoder promotion for method v4.

    Candidates inside the first ``top_m`` entries of ``observed_indices``
    (the pre-bonus fused order) whose cross-encoder relevance score reaches
    the preregistered ``threshold`` receive one bounded additive bonus
    ``lambda_positive * (1 - score) * margin`` with
    ``margin = (rel - threshold) / (1 - threshold)``.  No candidate is ever
    demoted; below-threshold and out-of-scope candidates keep their scores.
    """

    original = torch.clamp(
        as_finite_vector(original_fused_scores, "original_fused_scores"),
        0.0,
        1.0,
    )
    relevance = as_finite_vector(relevance_scores, "relevance_scores")
    if relevance.numel() != original.numel():
        raise ValueError("Relevance scores must match the fused score length.")
    for name, value in (
        ("lambda_positive", lambda_positive),
        ("threshold", threshold),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if int(top_m) < 1:
        raise ValueError("top_m must be a positive integer.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be a finite positive number.")

    window = [int(index) for index in list(observed_indices)[: int(top_m)]]
    if len(set(window)) != len(window):
        raise ValueError("observed_indices contains duplicates in the reranker window.")
    if any(index < 0 or index >= original.numel() for index in window):
        raise ValueError("observed_indices contains an out-of-range candidate index.")
    diagnostics: Dict[str, Any] = {
        "promotion_executed": False,
        "promotion_reason": "no_candidate_above_threshold",
        "reranker_threshold": float(threshold),
        "reranker_top_m": int(top_m),
        "considered": [],
        "promoted_candidate_indices": [],
        "top1_before": None,
        "top1_after": None,
        "top1_changed": False,
    }
    if original.numel() == 0:
        diagnostics["promotion_reason"] = "no_candidate"
        return original, diagnostics

    top1_before = int(torch.argmax(original).item())
    diagnostics["top1_before"] = top1_before
    promoted = original.clone()
    promoted_indices: List[int] = []
    denominator = max(1.0 - float(threshold), float(epsilon))
    for index in window:
        rel = float(relevance[index].item())
        margin = max(0.0, rel - float(threshold)) / denominator
        bonus = (
            float(lambda_positive)
            * (1.0 - float(promoted[index].item()))
            * margin
        )
        diagnostics["considered"].append(
            {
                "index": index,
                "relevance": rel,
                "margin": margin,
                "bonus": bonus,
                "above_threshold": rel >= float(threshold),
            }
        )
        if rel >= float(threshold) and bonus > float(epsilon):
            promoted[index] = promoted[index] + bonus
            promoted_indices.append(index)

    top1_after = int(torch.argmax(promoted).item())
    diagnostics["promoted_candidate_indices"] = promoted_indices
    diagnostics["top1_after"] = top1_after
    diagnostics["top1_changed"] = top1_after != top1_before
    if promoted_indices:
        diagnostics["promotion_executed"] = True
        diagnostics["promotion_reason"] = "bounded_bonus_applied"
    return promoted, diagnostics


def topk_counterfactual_promotion(
    original_fused_scores: Sequence[float] | torch.Tensor,
    counterfactual_fused_views: Sequence[Sequence[float] | torch.Tensor],
    *,
    lambda_positive: float,
    promotion_top_k: int,
    epsilon: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Add rank-decayed bonuses to context-stable Top-k candidates per phrase.

    The two deleted-phrase views are processed independently.  Stability is
    the geometric mean of original dual-route support and remaining-context
    support.  Only each view's first ``promotion_top_k`` candidates can gain a
    bonus; candidates are never demoted, and bonuses from both views add.
    """

    original = torch.clamp(
        as_finite_vector(original_fused_scores, "original_fused_scores"),
        0.0,
        1.0,
    )
    if not 0.0 <= float(lambda_positive) <= 1.0:
        raise ValueError("lambda_positive must be in [0, 1].")
    if isinstance(promotion_top_k, bool) or int(promotion_top_k) < 1:
        raise ValueError("promotion_top_k must be a positive integer.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be a finite positive number.")

    views = [
        torch.clamp(
            as_finite_vector(view, f"counterfactual_view_{index}"), 0.0, 1.0
        )
        for index, view in enumerate(counterfactual_fused_views)
    ]
    if any(view.numel() != original.numel() for view in views):
        raise ValueError(
            "Counterfactual context views must match the original score length."
        )

    diagnostics: Dict[str, Any] = {
        "promotion_executed": False,
        "promotion_reason": "insufficient_valid_phrases",
        "valid_phrase_count": len(views),
        "promotion_top_k": int(promotion_top_k),
        "per_view_topk_indices": [],
        "per_view_alignment_scores": [],
        "per_view_bonuses": [],
        "promoted_candidate_indices": [],
        "promotion_amounts": {},
        "top1_before": None,
        "top1_after": None,
        "top1_changed": False,
    }
    if original.numel() == 0:
        diagnostics["promotion_reason"] = "no_candidate"
        return original, diagnostics

    top1_before = int(torch.argmax(original).item())
    diagnostics["top1_before"] = top1_before
    diagnostics["top1_after"] = top1_before
    if len(views) != 2:
        return original.clone(), diagnostics

    total_bonuses = torch.zeros_like(original)
    limit = min(int(promotion_top_k), original.numel())
    for view in views:
        alignment = torch.sqrt(torch.clamp(original * view, 0.0, 1.0))
        order = torch.argsort(alignment, descending=True, stable=True)[:limit]
        indices = [int(index) for index in order.tolist()]
        view_bonuses: List[Dict[str, Any]] = []
        for rank, index in enumerate(indices, start=1):
            rank_weight = float(limit - rank + 1) / float(limit)
            bonus = (
                float(lambda_positive)
                * rank_weight
                * float(alignment[index].item())
            )
            if bonus > float(epsilon):
                total_bonuses[index] += bonus
            view_bonuses.append(
                {
                    "index": index,
                    "rank": rank,
                    "rank_weight": rank_weight,
                    "alignment": float(alignment[index].item()),
                    "bonus": bonus,
                }
            )
        diagnostics["per_view_topk_indices"].append(indices)
        diagnostics["per_view_alignment_scores"].append(
            [float(value) for value in alignment.tolist()]
        )
        diagnostics["per_view_bonuses"].append(view_bonuses)

    promoted = torch.clamp(original + total_bonuses, 0.0, 1.0)
    promoted_indices = [
        int(index)
        for index in torch.nonzero(total_bonuses > float(epsilon), as_tuple=False)
        .flatten()
        .tolist()
    ]
    top1_after = int(torch.argmax(promoted).item())
    diagnostics.update(
        {
            "promotion_executed": bool(promoted_indices),
            "promotion_reason": (
                "topk_positive_bonus_applied" if promoted_indices else "zero_bonus"
            ),
            "promoted_candidate_indices": promoted_indices,
            "promotion_amounts": {
                str(index): float(promoted[index].item() - original[index].item())
                for index in promoted_indices
            },
            "top1_after": top1_after,
            "top1_changed": top1_after != top1_before,
        }
    )
    return promoted, diagnostics


def cluster_champions(
    pre_order: Sequence[int],
    cluster_ids: Sequence[int],
    top_n: int,
) -> List[int]:
    """First representative of each context cluster within the top-n order.

    Used by the conflict-expansion correction mode: champion chunks are
    force-included in the final evidence (appended after the selected k,
    never removing anything) so the generator can arbitrate between
    competing document contexts.  Pure function over candidate indices.
    """

    champions: List[int] = []
    seen: set = set()
    for index in list(pre_order)[: max(0, int(top_n))]:
        cluster = int(cluster_ids[index])
        if cluster not in seen:
            seen.add(cluster)
            champions.append(int(index))
    return champions


def counterfactual_phrase_risk(
    original_normalized_scores: Sequence[float] | torch.Tensor,
    original_raw_scores: Sequence[float] | torch.Tensor,
    counterfactual_raw_scores: Sequence[float] | torch.Tensor,
    pairwise_cosine: torch.Tensor,
    *,
    query_impact: float,
    epsilon: float,
    z_max: float,
    iqr_normal_consistency: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    x_values = as_finite_vector(original_normalized_scores, "normalized_scores")
    raw_values = as_finite_vector(original_raw_scores, "raw_scores")
    counterfactual = as_finite_vector(
        counterfactual_raw_scores, "counterfactual_scores"
    )
    if not (x_values.numel() == raw_values.numel() == counterfactual.numel()):
        raise ValueError("Counterfactual score vectors must have equal lengths.")
    count = int(x_values.numel())
    if pairwise_cosine.shape != (count, count):
        raise ValueError("pairwise_cosine has the wrong shape.")

    raw_difference = raw_values - counterfactual
    difference_z, difference_stats = robust_z_scores(
        raw_difference,
        epsilon=epsilon,
        z_max=z_max,
        iqr_normal_consistency=iqr_normal_consistency,
    )
    # The documented delta uses positive centred raw differences divided by the
    # robust scale.  robust_z_scores already implements exactly that centring and
    # scale, so only its positive part is retained and clipped.
    dependency = torch.clamp(torch.relu(difference_z), 0.0, 1.0)
    remaining_support, remaining_stats = robust_sigmoid_map(
        counterfactual,
        epsilon=epsilon,
        z_max=z_max,
        iqr_normal_consistency=iqr_normal_consistency,
    )
    attraction = x_values * dependency
    attraction_sum = float(torch.sum(attraction).item())

    if attraction_sum <= epsilon or count == 1:
        breadth = 0.0
        dispersion = 0.0
        beta = 0.0
        weights = torch.zeros_like(attraction)
    else:
        weights = attraction / attraction_sum
        breadth = (1.0 - float(torch.sum(weights.square()).item())) / (
            1.0 - 1.0 / count
        )
        breadth = min(1.0, max(0.0, breadth))
        distances = torch.clamp((1.0 - pairwise_cosine) / 2.0, 0.0, 1.0)
        dispersion = _weighted_pair_distance(weights, distances, epsilon)
        beta = min(1.0, max(0.0, float(query_impact) * breadth * dispersion))

    risk = beta * dependency * (1.0 - remaining_support)
    risk = torch.clamp(risk, 0.0, 1.0)
    return risk, {
        "query_impact": float(query_impact),
        "raw_difference": [float(value) for value in raw_difference.tolist()],
        "difference_stats": difference_stats.diagnostics(),
        "dependency": [float(value) for value in dependency.tolist()],
        "signed_dependency": [
            float(value)
            for value in torch.clamp(difference_z, -1.0, 1.0).tolist()
        ],
        "remaining_support": [float(value) for value in remaining_support.tolist()],
        "remaining_support_stats": remaining_stats.diagnostics(),
        "attraction_weights": [float(value) for value in weights.tolist()],
        "breadth": breadth,
        "dispersion": dispersion,
        "beta": beta,
        "risk": [float(value) for value in risk.tolist()],
    }


def greedy_evidence_competition(
    base_values: Sequence[float] | torch.Tensor,
    pairwise_cosine: torch.Tensor,
    candidate_ids: Sequence[str],
    union_ranks: Sequence[int],
    *,
    max_k: int,
    redundancy_enabled: bool,
    lambda_redundancy: float,
) -> GreedySelection:
    values = as_finite_vector(base_values, "base_values")
    count = int(values.numel())
    if len(candidate_ids) != count or len(union_ranks) != count:
        raise ValueError("Candidate identities and score lengths differ.")
    if pairwise_cosine.shape != (count, count):
        raise ValueError("pairwise_cosine has the wrong shape.")
    limit = min(max(1, int(max_k)), count)
    selected: List[int] = []
    marginal_values: List[float] = []
    redundancies: List[float] = []
    remaining = set(range(count))

    for _ in range(limit):
        choices: List[Tuple[float, int, str, int, float]] = []
        for index in remaining:
            if not selected or not redundancy_enabled:
                redundancy = 0.0
            else:
                similarities = pairwise_cosine[index, selected]
                redundancy = float(
                    torch.max(torch.clamp(similarities, 0.0, 1.0)).item()
                )
            marginal = float(values[index].item()) - (
                float(lambda_redundancy) * redundancy if redundancy_enabled else 0.0
            )
            choices.append(
                (
                    -marginal,
                    int(union_ranks[index]),
                    str(candidate_ids[index]),
                    index,
                    redundancy,
                )
            )
        _, _, _, best, redundancy = min(choices)
        selected.append(best)
        remaining.remove(best)
        redundancies.append(redundancy)
        marginal_values.append(
            float(values[best].item())
            - (float(lambda_redundancy) * redundancy if redundancy_enabled else 0.0)
        )
    return GreedySelection(
        indices=tuple(selected),
        marginal_utilities=tuple(marginal_values),
        redundancies=tuple(redundancies),
    )


def empirical_null_probability(
    evidence_strength_value: float, null_strengths: Sequence[float]
) -> float:
    if not null_strengths:
        raise ValueError("The Null distribution is empty.")
    count = sum(
        1 for value in null_strengths if float(value) >= float(evidence_strength_value)
    )
    return (1.0 + count) / (len(null_strengths) + 1.0)


def select_adaptive_k(
    marginal_utilities: Sequence[float] | torch.Tensor,
    *,
    chi: float,
    temperature: float,
    eta: float,
    epsilon: float,
    z_max: float,
    iqr_normal_consistency: float,
    adaptive_enabled: bool,
    entropy_enabled: bool,
    maximum_gap_enabled: bool,
    consistency_entropy_enabled: bool,
    fixed_k: Optional[int],
    null_gate_enabled: bool,
    null_strengths: Sequence[float],
    null_low_threshold: Optional[float],
    null_high_threshold: Optional[float],
) -> AdaptiveKResult:
    utilities = as_finite_vector(marginal_utilities, "marginal_utilities")
    count = int(utilities.numel())
    if count > 1 and bool(torch.any(utilities[:-1] + epsilon < utilities[1:])):
        raise ValueError(
            "AmbiCal adaptive-k requires the greedy marginal-utility order."
        )
    mapped, _ = robust_sigmoid_map(
        utilities,
        epsilon=epsilon,
        z_max=z_max,
        iqr_normal_consistency=iqr_normal_consistency,
    )
    probabilities = stable_softmax(mapped, temperature)
    raw_entropy = float(
        -torch.sum(probabilities * torch.log(probabilities.clamp_min(epsilon))).item()
    )
    corrected_entropy = raw_entropy
    if consistency_entropy_enabled:
        corrected_entropy *= float(eta) + (1.0 - float(eta)) * float(chi)
    entropy_k = max(1, min(count, round_half_up(math.exp(corrected_entropy))))

    if count == 1:
        gap_k = 1
        gap_strength = 0.0
    else:
        gaps = mapped[:-1] - mapped[1:]
        gap_index = int(torch.argmax(gaps).item())
        gap_k = gap_index + 1
        denominator = float((mapped[0] - mapped[-1]).item()) + epsilon
        gap_strength = min(
            1.0, max(0.0, float(gaps[gap_index].item()) / denominator)
        )

    if not adaptive_enabled:
        if fixed_k is None:
            raise ValueError("fixed_k is required when adaptive-k is disabled.")
        adaptive_k = max(1, min(count, int(fixed_k)))
    elif entropy_enabled and maximum_gap_enabled:
        adaptive_k = round_half_up(
            gap_strength * gap_k + (1.0 - gap_strength) * entropy_k
        )
        adaptive_k = max(1, min(count, adaptive_k))
    elif entropy_enabled:
        adaptive_k = entropy_k
    elif maximum_gap_enabled:
        adaptive_k = gap_k
    else:
        raise ValueError("No adaptive-k candidate mechanism is active.")

    strength, _ = evidence_strength(
        utilities,
        epsilon=epsilon,
        z_max=z_max,
        iqr_normal_consistency=iqr_normal_consistency,
    )
    null_probability: Optional[float] = None
    selected_k = adaptive_k
    null_region = "disabled"
    if null_gate_enabled:
        if null_low_threshold is None or null_high_threshold is None:
            raise ValueError("Null thresholds are required when Null gating is enabled.")
        null_probability = empirical_null_probability(strength, null_strengths)
        if null_probability <= null_low_threshold:
            null_region = "strong_evidence"
        elif null_probability < null_high_threshold:
            null_region = "boundary"
            selected_k = min(3, adaptive_k)
        else:
            null_region = "null_like"
            selected_k = 1

    return AdaptiveKResult(
        selected_k=selected_k,
        adaptive_k=adaptive_k,
        entropy_k=entropy_k,
        gap_k=gap_k,
        gap_strength=gap_strength,
        raw_entropy=raw_entropy,
        corrected_entropy=corrected_entropy,
        mapped_utilities=tuple(float(value) for value in mapped.tolist()),
        probabilities=tuple(float(value) for value in probabilities.tolist()),
        evidence_strength=strength,
        null_probability=null_probability,
        null_region=null_region,
    )


def selected_items(
    chunks: Sequence[ChunkRecord],
    base_values: Sequence[float],
    greedy: GreedySelection,
    selected_k: int,
) -> List[EvidenceItem]:
    values = list(base_values)
    items: List[EvidenceItem] = []
    for rank, candidate_index in enumerate(greedy.indices[:selected_k], start=1):
        chunk = chunks[candidate_index]
        items.append(
            EvidenceItem(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                base_value=float(values[candidate_index]),
                marginal_utility=float(greedy.marginal_utilities[rank - 1]),
                selected_rank=rank,
                metadata=chunk.metadata,
            )
        )
    return items


def merge_structured_evidence(
    items: Sequence[EvidenceItem],
) -> Tuple[List[EvidenceItem], Dict[str, Any]]:
    """Merge only adjacent/overlapping intervals with trustworthy coordinates.

    Selection order is preserved.  Missing metadata leaves the sequence unchanged
    and is explicitly reported; no page or offset is inferred from text.
    """

    if not items:
        return [], {"applied": False, "reason": "empty", "merged_count": 0}
    if any(not item.metadata.has_merge_coordinates for item in items):
        return list(items), {
            "applied": False,
            "reason": "missing_structured_offsets",
            "merged_count": len(items),
        }

    merged: List[EvidenceItem] = []
    for item in items:
        if not merged:
            merged.append(item)
            continue
        previous = merged[-1]
        left = previous.metadata
        right = item.metadata
        same_document = left.document_id == right.document_id
        # ``left.chunk_index`` stays at the first interval after a merge.  Use
        # the number of represented chunks so a run of 0,1,2,... can continue
        # merging instead of stopping after the first pair.
        previous_span_count = len(
            previous.merged_chunk_ids or (previous.chunk_id,)
        )
        adjacent_index = (  # type: ignore[operator]
            right.chunk_index == left.chunk_index + previous_span_count
        )
        overlaps = right.char_start <= left.char_end  # type: ignore[operator]
        if not (same_document and adjacent_index and overlaps):
            merged.append(item)
            continue

        overlap_count = max(0, int(left.char_end) - int(right.char_start))
        suffix = item.text[overlap_count:] if overlap_count <= len(item.text) else ""
        combined_text = previous.text + suffix
        combined_ids = (
            previous.merged_chunk_ids or (previous.chunk_id,)
        ) + (item.chunk_id,)
        combined_metadata = type(left)(
            document_id=left.document_id,
            chunk_index=left.chunk_index,
            section=left.section if left.section == right.section else left.section,
            page_start=left.page_start,
            page_end=right.page_end if right.page_end is not None else left.page_end,
            char_start=left.char_start,
            char_end=max(int(left.char_end), int(right.char_end)),
        )
        merged[-1] = EvidenceItem(
            chunk_id=previous.chunk_id,
            text=combined_text,
            base_value=max(previous.base_value, item.base_value),
            marginal_utility=previous.marginal_utility,
            selected_rank=previous.selected_rank,
            metadata=combined_metadata,
            merged_chunk_ids=combined_ids,
        )
    return merged, {
        "applied": True,
        "reason": "structured_offsets_available",
        "merged_count": len(merged),
    }
