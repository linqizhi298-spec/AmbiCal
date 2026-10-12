"""Configuration and ablation switches for AmbiCal.

The values described as development-set hyperparameters in the method document
are deliberately not guessed here.  They are required whenever the
corresponding component is enabled.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AmbiCalSwitches:
    """Independent method switches with the exact documented degradations."""

    translation_branch: bool = True
    robust_normalization: bool = True
    branch_confidence: bool = True
    translation_consistency: bool = True
    ranking_overlap: bool = True
    dynamic_branch_weights: bool = True
    false_positive_prior: bool = True
    context_correction: bool = True
    context_gate: bool = True
    soft_redundancy: bool = True
    consistency_entropy: bool = True
    maximum_gap: bool = True
    entropy_candidate: bool = True
    null_gate: bool = True
    adaptive_k: bool = True
    merge_structured_evidence: bool = True

    def validate(self) -> None:
        if self.adaptive_k and not (self.maximum_gap or self.entropy_candidate):
            raise ValueError(
                "AmbiCal adaptive-k requires the entropy candidate, the "
                "maximum-gap candidate, or both."
            )


@dataclass(frozen=True)
class AmbiCalConfig:
    """Numerical settings for the method.

    ``lambda_*`` and ``theta_*`` values have no theoretically fixed value in
    the source method.  A caller must supply them when the associated component
    is active; ``None`` therefore means "not configured", not zero.
    """

    lambda_false_positive: Optional[float] = None
    lambda_context: Optional[float] = None
    lambda_positive: Optional[float] = None
    lambda_redundancy: Optional[float] = None
    context_gate_threshold: Optional[float] = None
    null_low_threshold: Optional[float] = None
    null_high_threshold: Optional[float] = None
    fixed_k: Optional[int] = None

    candidate_top_n: int = 40
    max_k: int = 40
    gate_top_m: int = 40
    branch_temperature: float = 0.1
    gate_temperature: float = 0.1
    adaptive_temperature: float = 0.15
    consistency_floor_eta: float = 0.5
    max_counterfactual_phrases: int = 3
    z_max: float = 8.0
    epsilon: float = 1e-8
    # The method requires one fixed IQR fallback rule but does not prescribe its
    # constant.  IQR/1.349 is the standard normal-consistent scale estimate.
    iqr_normal_consistency: float = 1.349
    minimum_counterfactual_content_tokens: int = 1
    # AmbiCal stores unit-normalized dense vectors.  Enforcing the contract
    # prevents Chroma's configured distance metric from silently changing the
    # exact cosine re-scoring required by AmbiCal.
    require_normalized_collection_embeddings: bool = True
    collection_norm_tolerance: float = 0.02
    method_version: str = "ambical-online-v1"
    # --- v2/v3 calibratable context gate ---
    # risk = sigmoid(bias + w_s*strength + w_c*competition + w_d*distance
    #                 + w_p*path_conflict + w_b*bimodality + w_a*branch_disagreement)
    # The weights have no theoretically fixed value; v2/v3 require them to be
    # supplied explicitly (fitted on gate development labels, never on formal
    # test results) so that a run cannot silently reuse unset defaults.
    gate_v2_bias: Optional[float] = None
    gate_v2_weight_strength: Optional[float] = None
    gate_v2_weight_competition: Optional[float] = None
    gate_v2_weight_distance: Optional[float] = None
    gate_v2_weight_path_conflict: Optional[float] = None
    gate_v2_weight_bimodality: Optional[float] = None
    gate_v2_weight_branch_disagreement: Optional[float] = None
    # --- v4 cross-encoder bounded promotion (ambical-online-v4) ---
    # The reranker relevance threshold and top-M scope must be preregistered
    # from gate development data (never formal results) and supplied
    # explicitly; candidates whose cross-encoder score exceeds the threshold
    # receive one bounded bonus, nothing is ever demoted.
    reranker_score_threshold: Optional[float] = None
    reranker_top_m: Optional[int] = None
    # --- v5 counterfactual Top-k positive promotion ---
    counterfactual_promotion_top_k: Optional[int] = None
    switches: AmbiCalSwitches = AmbiCalSwitches()

    _SUPPORTED_METHOD_VERSIONS = (
        "ambical-online-v1",
        "ambical-online-v2",
        "ambical-online-v3",
        "ambical-online-v3.1",
        "ambical-online-v4",
        "ambical-online-v5",
    )

    def validate(self) -> None:
        self.switches.validate()
        self._require_positive_integer("candidate_top_n", self.candidate_top_n)
        self._require_positive_integer("max_k", self.max_k)
        self._require_positive_integer("gate_top_m", self.gate_top_m)
        self._require_positive_integer(
            "max_counterfactual_phrases", self.max_counterfactual_phrases
        )
        self._require_positive_integer(
            "minimum_counterfactual_content_tokens",
            self.minimum_counterfactual_content_tokens,
        )
        self._require_finite("branch_temperature", self.branch_temperature)
        self._require_finite("gate_temperature", self.gate_temperature)
        self._require_finite("adaptive_temperature", self.adaptive_temperature)
        self._require_finite("consistency_floor_eta", self.consistency_floor_eta)
        self._require_finite("z_max", self.z_max)
        self._require_finite("epsilon", self.epsilon)
        self._require_finite("iqr_normal_consistency", self.iqr_normal_consistency)
        self._require_finite(
            "collection_norm_tolerance", self.collection_norm_tolerance
        )
        optional_numbers = {
            "lambda_false_positive": self.lambda_false_positive,
            "lambda_context": self.lambda_context,
            "lambda_positive": self.lambda_positive,
            "lambda_redundancy": self.lambda_redundancy,
            "context_gate_threshold": self.context_gate_threshold,
            "null_low_threshold": self.null_low_threshold,
            "null_high_threshold": self.null_high_threshold,
        }
        for name, value in optional_numbers.items():
            if value is not None:
                self._require_finite(name, value)
        if self.fixed_k is not None:
            self._require_positive_integer("fixed_k", self.fixed_k)
        if self.branch_temperature <= 0:
            raise ValueError("branch_temperature must be positive.")
        if self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive.")
        if self.adaptive_temperature <= 0:
            raise ValueError("adaptive_temperature must be positive.")
        if not 0 <= self.consistency_floor_eta <= 1:
            raise ValueError("consistency_floor_eta must be in [0, 1].")
        if self.z_max <= 0:
            raise ValueError("z_max must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if self.iqr_normal_consistency <= 0:
            raise ValueError("iqr_normal_consistency must be positive.")
        if self.collection_norm_tolerance < 0:
            raise ValueError("collection_norm_tolerance must be non-negative.")

        if self.method_version not in self._SUPPORTED_METHOD_VERSIONS:
            raise ValueError(
                "method_version must be one of "
                + ", ".join(self._SUPPORTED_METHOD_VERSIONS)
                + f"; got {self.method_version!r}."
            )
        gate_v2_weights = {
            "gate_v2_bias": self.gate_v2_bias,
            "gate_v2_weight_strength": self.gate_v2_weight_strength,
            "gate_v2_weight_competition": self.gate_v2_weight_competition,
            "gate_v2_weight_distance": self.gate_v2_weight_distance,
            "gate_v2_weight_path_conflict": self.gate_v2_weight_path_conflict,
            "gate_v2_weight_bimodality": self.gate_v2_weight_bimodality,
            "gate_v2_weight_branch_disagreement": (
                self.gate_v2_weight_branch_disagreement
            ),
        }
        if self.method_version in {
            "ambical-online-v2",
            "ambical-online-v3",
            "ambical-online-v3.1",
            "ambical-online-v4",
            "ambical-online-v5",
        }:
            if self.switches.context_gate:
                for name, value in gate_v2_weights.items():
                    self._require_finite(name, value)
        else:
            stray = sorted(
                name for name, value in gate_v2_weights.items() if value is not None
            )
            if stray:
                raise ValueError(
                    "gate_v2_* fields require method_version "
                    "'ambical-online-v2', 'ambical-online-v3', "
                    "'ambical-online-v3.1', 'ambical-online-v4', "
                    "or 'ambical-online-v5'; "
                    "unset them for v1. Found: "
                    + ", ".join(stray)
                )

        promotion_versions = {
            "ambical-online-v3",
            "ambical-online-v3.1",
            "ambical-online-v4",
            "ambical-online-v5",
        }
        if self.method_version in promotion_versions:
            if self.lambda_context is not None:
                raise ValueError(
                    "lambda_context is not used by "
                    + self.method_version
                    + "; unset it and configure lambda_positive instead."
                )
        elif self.lambda_positive is not None:
            raise ValueError(
                "lambda_positive requires method_version "
                "'ambical-online-v3', 'ambical-online-v3.1', "
                "'ambical-online-v4', or 'ambical-online-v5'."
            )
        if self.method_version == "ambical-online-v4":
            if self.reranker_score_threshold is None or self.reranker_top_m is None:
                raise ValueError(
                    "ambical-online-v4 requires reranker_score_threshold "
                    "and reranker_top_m (preregistered on gate development data)."
                )
            self._require_unit_interval(
                "reranker_score_threshold", self.reranker_score_threshold
            )
            self._require_positive_integer("reranker_top_m", self.reranker_top_m)
        elif (
            self.reranker_score_threshold is not None
            or self.reranker_top_m is not None
        ):
            raise ValueError(
                "reranker_* fields require method_version "
                "'ambical-online-v4'."
            )
        if self.method_version == "ambical-online-v5":
            if self.counterfactual_promotion_top_k is None:
                raise ValueError(
                    "ambical-online-v5 requires "
                    "counterfactual_promotion_top_k."
                )
            self._require_positive_integer(
                "counterfactual_promotion_top_k",
                self.counterfactual_promotion_top_k,
            )
        elif self.counterfactual_promotion_top_k is not None:
            raise ValueError(
                "counterfactual_promotion_top_k requires method_version "
                "'ambical-online-v5'."
            )

        if self.switches.false_positive_prior:
            self._require_nonnegative(
                "lambda_false_positive", self.lambda_false_positive
            )
        if self.switches.context_correction:
            if self.method_version in promotion_versions:
                self._require_unit_interval("lambda_positive", self.lambda_positive)
            else:
                self._require_unit_interval("lambda_context", self.lambda_context)
            if self.method_version in {
                "ambical-online-v3",
                "ambical-online-v3.1",
                "ambical-online-v5",
            } and self.max_counterfactual_phrases != 2:
                raise ValueError(
                    "ambical-online-v3/v3.1/v5 requires "
                    "max_counterfactual_phrases=2."
                )
            if self.switches.context_gate:
                self._require_unit_interval(
                    "context_gate_threshold", self.context_gate_threshold
                )
        if self.switches.soft_redundancy:
            self._require_nonnegative("lambda_redundancy", self.lambda_redundancy)
        if self.switches.null_gate:
            self._require_unit_interval(
                "null_low_threshold", self.null_low_threshold
            )
            self._require_unit_interval(
                "null_high_threshold", self.null_high_threshold
            )
            if not self.null_low_threshold < self.null_high_threshold:  # type: ignore[operator]
                raise ValueError(
                    "null_low_threshold must be smaller than null_high_threshold."
                )
        if not self.switches.adaptive_k:
            if self.fixed_k is None:
                raise ValueError(
                    "fixed_k must be at least 1 when adaptive-k is disabled."
                )

    @staticmethod
    def _require_finite(name: str, value: float) -> None:
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number.") from exc
        if not finite:
            raise ValueError(f"{name} must be a finite number.")

    @staticmethod
    def _require_positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer of at least 1.")

    @staticmethod
    def _require_nonnegative(name: str, value: Optional[float]) -> None:
        if value is None:
            raise ValueError(f"{name} must be configured when its component is enabled.")
        AmbiCalConfig._require_finite(name, value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    @staticmethod
    def _require_unit_interval(name: str, value: Optional[float]) -> None:
        if value is None:
            raise ValueError(f"{name} must be configured when its component is enabled.")
        AmbiCalConfig._require_finite(name, value)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1].")

    def diagnostics(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Version-specific fields are omitted from older records so extending
        # the method never changes already-published diagnostics/signatures.
        if self.method_version not in {
            "ambical-online-v3",
            "ambical-online-v3.1",
            "ambical-online-v4",
            "ambical-online-v5",
        }:
            payload.pop("lambda_positive", None)
        if self.method_version != "ambical-online-v4":
            payload.pop("reranker_score_threshold", None)
            payload.pop("reranker_top_m", None)
        if self.method_version != "ambical-online-v5":
            payload.pop("counterfactual_promotion_top_k", None)
        return payload

    def utility_signature(self) -> str:
        """Fingerprint the online utility/stopping configuration.

        A future calibration builder can persist this exact value.  The online
        retriever rejects an explicitly fingerprinted profile from another
        configuration instead of silently applying a mismatched Null scale.
        """

        canonical = json.dumps(
            self.diagnostics(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
