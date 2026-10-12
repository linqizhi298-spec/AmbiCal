"""Numerically stable primitives shared by every AmbiCal stage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as torch_functional


@dataclass(frozen=True)
class RobustStats:
    median: float
    mad: float
    scale: float
    scale_source: str
    z_values: Tuple[float, ...]

    def diagnostics(self) -> Dict[str, object]:
        return {
            "median": self.median,
            "mad": self.mad,
            "scale": self.scale,
            "scale_source": self.scale_source,
            "z_values": list(self.z_values),
        }


@dataclass(frozen=True)
class BranchShape:
    probabilities: Tuple[float, ...]
    normalized_entropy: float
    normalized_top_gap: float
    top_score: float
    confidence: float

    def diagnostics(self) -> Dict[str, object]:
        return {
            "probabilities": list(self.probabilities),
            "normalized_entropy": self.normalized_entropy,
            "normalized_top_gap": self.normalized_top_gap,
            "top_score": self.top_score,
            "confidence": self.confidence,
        }


def as_finite_vector(values: Sequence[float] | torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    else:
        tensor = torch.tensor(list(values), dtype=torch.float64).flatten()
    if tensor.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return tensor


def l2_normalize(values: Sequence[float] | torch.Tensor, name: str) -> torch.Tensor:
    tensor = as_finite_vector(values, name)
    norm = torch.linalg.vector_norm(tensor)
    if not torch.isfinite(norm) or float(norm.item()) <= 0:
        raise ValueError(f"{name} is zero or non-finite and cannot be normalized.")
    return tensor / norm


def cosine_scores(
    query_embedding: Sequence[float] | torch.Tensor,
    document_embeddings: Sequence[Sequence[float]] | torch.Tensor,
) -> torch.Tensor:
    query = l2_normalize(query_embedding, "query_embedding")
    documents = torch.as_tensor(document_embeddings, dtype=torch.float64)
    if documents.ndim != 2 or documents.shape[0] == 0:
        raise ValueError("document_embeddings must be a non-empty 2-D matrix.")
    if documents.shape[1] != query.numel():
        raise ValueError("Query and document embedding dimensions do not match.")
    if not torch.isfinite(documents).all():
        raise ValueError("document_embeddings contains NaN or infinite values.")
    norms = torch.linalg.vector_norm(documents, dim=1)
    if bool(torch.any(norms <= 0)):
        raise ValueError("A document embedding is a zero vector.")
    documents = torch_functional.normalize(documents, p=2, dim=1)
    return documents @ query


def pairwise_cosine(
    document_embeddings: Sequence[Sequence[float]] | torch.Tensor,
) -> torch.Tensor:
    documents = torch.as_tensor(document_embeddings, dtype=torch.float64)
    if documents.ndim != 2 or documents.shape[0] == 0:
        raise ValueError("document_embeddings must be a non-empty 2-D matrix.")
    if not torch.isfinite(documents).all():
        raise ValueError("document_embeddings contains NaN or infinite values.")
    norms = torch.linalg.vector_norm(documents, dim=1)
    if bool(torch.any(norms <= 0)):
        raise ValueError("A document embedding is a zero vector.")
    documents = torch_functional.normalize(documents, p=2, dim=1)
    return torch.clamp(documents @ documents.T, -1.0, 1.0)


def stable_softmax(
    values: Sequence[float] | torch.Tensor, temperature: float
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Softmax temperature must be positive.")
    tensor = as_finite_vector(values, "softmax_values") / float(temperature)
    shifted = tensor - torch.max(tensor)
    probabilities = torch.exp(shifted)
    probabilities = probabilities / torch.sum(probabilities)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Softmax produced non-finite probabilities.")
    return probabilities


def robust_z_scores(
    values: Sequence[float] | torch.Tensor,
    *,
    epsilon: float,
    z_max: float,
    iqr_normal_consistency: float,
) -> Tuple[torch.Tensor, RobustStats]:
    """Return the one shared MAD/IQR rule required by the method.

    * all exactly equal -> every z is zero;
    * positive MAD -> 1.4826 * MAD;
    * zero MAD but non-identical -> IQR/1.349;
    * a degenerate zero IQR uses only ``epsilon`` and is clipped by ``z_max``.
    """

    if epsilon <= 0 or z_max <= 0 or iqr_normal_consistency <= 0:
        raise ValueError("Robust-score numerical constants must be positive.")
    tensor = as_finite_vector(values, "robust_values")
    median = torch.quantile(tensor, 0.5)
    deviations = torch.abs(tensor - median)
    mad = torch.quantile(deviations, 0.5)
    span = torch.max(tensor) - torch.min(tensor)

    if float(span.item()) == 0.0:
        z_values = torch.zeros_like(tensor)
        scale = 0.0
        source = "all_equal"
    elif float(mad.item()) > 0.0:
        scale = 1.4826 * float(mad.item())
        source = "mad"
        z_values = (tensor - median) / (scale + epsilon)
    else:
        q25 = torch.quantile(tensor, 0.25)
        q75 = torch.quantile(tensor, 0.75)
        iqr = max(0.0, float((q75 - q25).item()))
        scale = iqr / iqr_normal_consistency
        source = "iqr" if scale > 0 else "iqr_degenerate"
        z_values = (tensor - median) / (scale + epsilon)

    z_values = torch.clamp(z_values, -float(z_max), float(z_max))
    if not torch.isfinite(z_values).all():
        raise ValueError("Robust standardization produced non-finite values.")
    stats = RobustStats(
        median=float(median.item()),
        mad=float(mad.item()),
        scale=float(scale),
        scale_source=source,
        z_values=tuple(float(value) for value in z_values.tolist()),
    )
    return z_values, stats


def robust_sigmoid_map(
    values: Sequence[float] | torch.Tensor,
    *,
    epsilon: float,
    z_max: float,
    iqr_normal_consistency: float,
) -> Tuple[torch.Tensor, RobustStats]:
    z_values, stats = robust_z_scores(
        values,
        epsilon=epsilon,
        z_max=z_max,
        iqr_normal_consistency=iqr_normal_consistency,
    )
    mapped = torch.sigmoid(z_values)
    return mapped, stats


def cosine_unit_interval(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    tensor = as_finite_vector(values, "cosine_scores")
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def branch_shape(
    scores: Sequence[float] | torch.Tensor,
    *,
    temperature: float,
    epsilon: float,
    use_confidence: bool,
) -> BranchShape:
    values = as_finite_vector(scores, "branch_scores")
    probabilities = stable_softmax(values, temperature)
    count = int(values.numel())
    if count == 1:
        entropy = 0.0
        gap = 0.0
        top_score = float(values[0].item())
    else:
        raw_entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(epsilon))
        )
        entropy = float((raw_entropy / math.log(count)).item())
        entropy = min(1.0, max(0.0, entropy))
        ordered = torch.sort(values, descending=True, stable=True).values
        numerator = float((ordered[0] - ordered[1]).item())
        denominator = float((ordered[0] - ordered[-1]).item()) + epsilon
        gap = min(1.0, max(0.0, numerator / denominator))
        top_score = float(ordered[0].item())
    confidence = (top_score + (1.0 - entropy) + gap) / 3.0
    if not use_confidence:
        confidence = 1.0
    return BranchShape(
        probabilities=tuple(float(value) for value in probabilities.tolist()),
        normalized_entropy=entropy,
        normalized_top_gap=gap,
        top_score=top_score,
        confidence=float(confidence),
    )


def translation_consistency(
    source_embedding: Sequence[float] | torch.Tensor,
    target_embedding: Sequence[float] | torch.Tensor,
) -> float:
    source = l2_normalize(source_embedding, "source_query_embedding")
    target = l2_normalize(target_embedding, "target_query_embedding")
    if source.numel() != target.numel():
        raise ValueError("Source and target query embedding dimensions do not match.")
    cosine = float(torch.dot(source, target).clamp(-1.0, 1.0).item())
    return min(1.0, max(0.0, (1.0 + cosine) / 2.0))


def dynamic_weights(
    source_confidence: float,
    target_confidence: Optional[float],
    rho: Optional[float],
    *,
    use_dynamic_weights: bool,
    use_translation_consistency: bool,
    epsilon: float,
) -> Tuple[float, float]:
    if target_confidence is None:
        return 1.0, 0.0
    if not use_dynamic_weights:
        return 0.5, 0.5
    source_reliability = max(0.0, float(source_confidence))
    target_reliability = max(0.0, float(target_confidence))
    if use_translation_consistency:
        if rho is None:
            raise ValueError("Translation consistency is active but rho is unavailable.")
        target_reliability *= float(rho)
    denominator = source_reliability + target_reliability + epsilon
    alpha_source = source_reliability / denominator
    alpha_target = target_reliability / denominator
    return float(alpha_source), float(alpha_target)


def ranking_overlap(source_ids: Sequence[str], target_ids: Sequence[str]) -> float:
    denominator = max(1, min(len(source_ids), len(target_ids)))
    return len(set(source_ids).intersection(target_ids)) / denominator


def active_consistency(signals: Iterable[Optional[float]]) -> Tuple[float, int]:
    active = [float(value) for value in signals if value is not None]
    if not active:
        # Neutral degeneration for a single active retrieval branch.  No disabled
        # signal is inserted into an otherwise active average.
        return 1.0, 0
    return sum(active) / len(active), len(active)


def round_half_up(value: float) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("round_half_up expects a finite non-negative value.")
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
