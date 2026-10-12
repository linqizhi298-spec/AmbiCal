"""Strict experiment-side loading helpers for AmbiCal.

The online method package deliberately does not create calibration artifacts or
choose experiment parameters.  These helpers only parse an explicitly supplied
method configuration and load an existing read-only profile.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional

from retrieval_pipelines.ambical import (
    AmbiCalConfig,
    AmbiCalProfile,
    AmbiCalSwitches,
)


@dataclass(frozen=True)
class LoadedAmbiCalProfile:
    profile: AmbiCalProfile
    absolute_path: str
    sha256: str
    scope: Mapping[str, Any]


EXPERIMENT_PARAMETER_FIELDS = {
    "source_language",
    "translator_model_name",
    "translation_cache_mode",
    "translation_prompt_path",
    "translation_cache_path",
    "profile_path",
    "reranker_model_path",
    "method_config",
}

REQUIRED_METHOD_CONFIG_FIELDS = {
    "candidate_top_n",
    "max_k",
    "gate_top_m",
    "branch_temperature",
    "gate_temperature",
    "adaptive_temperature",
    "consistency_floor_eta",
    "max_counterfactual_phrases",
    "z_max",
    "epsilon",
    "iqr_normal_consistency",
    "minimum_counterfactual_content_tokens",
    "require_normalized_collection_embeddings",
    "collection_norm_tolerance",
    "method_version",
    "switches",
}

INTEGER_METHOD_FIELDS = {
    "counterfactual_promotion_top_k",
    "reranker_top_m",
    "candidate_top_n",
    "max_k",
    "gate_top_m",
    "max_counterfactual_phrases",
    "minimum_counterfactual_content_tokens",
    "fixed_k",
}

NUMBER_METHOD_FIELDS = {
    "reranker_score_threshold",
    "lambda_false_positive",
    "lambda_context",
    "lambda_positive",
    "lambda_redundancy",
    "context_gate_threshold",
    "null_low_threshold",
    "null_high_threshold",
    "branch_temperature",
    "gate_temperature",
    "adaptive_temperature",
    "consistency_floor_eta",
    "z_max",
    "epsilon",
    "iqr_normal_consistency",
    "collection_norm_tolerance",
    "gate_v2_bias",
    "gate_v2_weight_strength",
    "gate_v2_weight_competition",
    "gate_v2_weight_distance",
    "gate_v2_weight_path_conflict",
    "gate_v2_weight_bimodality",
    "gate_v2_weight_branch_disagreement",
}

BOOLEAN_METHOD_FIELDS = {"require_normalized_collection_embeddings"}
STRING_METHOD_FIELDS = {"method_version"}
CALIBRATION_COMPLETE_MARKER = "CALIBRATION_COMPLETE"
CALIBRATION_INCOMPLETE_MARKER = "CALIBRATION_INCOMPLETE"
CALIBRATION_MANIFEST_FILENAME = "ambical_calibration_manifest.json"


def build_ambical_config(
    parameters: Mapping[str, Any],
) -> AmbiCalConfig:
    """Build and validate the typed method config without guessing parameters."""

    if not isinstance(parameters, Mapping):
        raise ValueError("'ambical_parameters' must be a JSON object.")
    unknown_parameters = sorted(set(parameters) - EXPERIMENT_PARAMETER_FIELDS)
    if unknown_parameters:
        raise ValueError(
            "Unknown ambical_parameters fields: "
            + ", ".join(unknown_parameters)
        )
    source_language = parameters.get("source_language")
    if not isinstance(source_language, str) or not source_language.strip():
        raise ValueError(
            "ambical_parameters.source_language must be a non-empty string."
        )
    for name in (
        "translator_model_name",
        "translation_cache_mode",
        "translation_prompt_path",
        "translation_cache_path",
        "profile_path",
        "reranker_model_path",
    ):
        if name not in parameters or parameters[name] in (None, ""):
            continue
        if not isinstance(parameters[name], str) or not parameters[name].strip():
            raise ValueError(
                f"ambical_parameters.{name} must be a non-empty string."
            )
    raw_method_config = parameters.get("method_config")
    if not isinstance(raw_method_config, Mapping):
        raise ValueError(
            "ambical_parameters.method_config must be a JSON object."
        )

    missing_method_config = sorted(
        REQUIRED_METHOD_CONFIG_FIELDS - set(raw_method_config)
    )
    if missing_method_config:
        raise ValueError(
            "AmbiCal formal method_config must explicitly provide: "
            + ", ".join(missing_method_config)
        )

    method_values = dict(raw_method_config)
    raw_switches = method_values.pop("switches", {})
    if not isinstance(raw_switches, Mapping):
        raise ValueError(
            "ambical_parameters.method_config.switches must be a JSON object."
        )

    allowed_config = {item.name for item in fields(AmbiCalConfig)} - {
        "switches"
    }
    unknown_config = sorted(set(method_values) - allowed_config)
    if unknown_config:
        raise ValueError(
            "Unknown AmbiCal method_config fields: "
            + ", ".join(unknown_config)
        )

    _validate_method_value_types(method_values)

    allowed_switches = {item.name for item in fields(AmbiCalSwitches)}
    unknown_switches = sorted(set(raw_switches) - allowed_switches)
    if unknown_switches:
        raise ValueError(
            "Unknown AmbiCal switches: " + ", ".join(unknown_switches)
        )
    missing_switches = sorted(allowed_switches - set(raw_switches))
    if missing_switches:
        raise ValueError(
            "AmbiCal formal switches must explicitly provide: "
            + ", ".join(missing_switches)
        )
    non_boolean_switches = sorted(
        name for name, value in raw_switches.items() if not isinstance(value, bool)
    )
    if non_boolean_switches:
        raise ValueError(
            "AmbiCal switches must be JSON booleans: "
            + ", ".join(non_boolean_switches)
        )

    switches = AmbiCalSwitches(**dict(raw_switches))
    config = AmbiCalConfig(switches=switches, **method_values)
    config.validate()
    reranker_path = parameters.get("reranker_model_path")
    if (
        reranker_path not in (None, "")
        and (
            config.method_version != "ambical-online-v4"
            or not config.switches.context_correction
        )
    ):
        raise ValueError(
            "ambical_parameters.reranker_model_path is only valid for "
            "ambical-online-v4 with context_correction enabled."
        )
    return config


def _validate_method_value_types(method_values: Mapping[str, Any]) -> None:
    """Keep JSON booleans, integers, numbers, and strings unambiguous."""

    for name in INTEGER_METHOD_FIELDS & set(method_values):
        value = method_values[name]
        if value is None and name == "fixed_k":
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"AmbiCal method_config.{name} must be an integer.")
    for name in NUMBER_METHOD_FIELDS & set(method_values):
        value = method_values[name]
        if value is None and name in {
            "lambda_false_positive",
            "lambda_context",
            "lambda_positive",
            "lambda_redundancy",
            "context_gate_threshold",
            "null_low_threshold",
            "null_high_threshold",
        }:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"AmbiCal method_config.{name} must be a number.")
    for name in BOOLEAN_METHOD_FIELDS & set(method_values):
        if not isinstance(method_values[name], bool):
            raise ValueError(
                f"AmbiCal method_config.{name} must be a JSON boolean."
            )
    for name in STRING_METHOD_FIELDS & set(method_values):
        value = method_values[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"AmbiCal method_config.{name} must be a non-empty string."
            )


def load_ambical_profile(path: str | Path) -> LoadedAmbiCalProfile:
    """Load one atomically published profile and verify its completion chain."""

    profile_path = Path(path).expanduser().resolve(strict=True)
    artifact_dir = profile_path.parent
    incomplete_marker = artifact_dir / CALIBRATION_INCOMPLETE_MARKER
    if incomplete_marker.exists():
        raise RuntimeError(
            f"AmbiCal calibration is incomplete: {incomplete_marker}"
        )
    complete_marker = artifact_dir / CALIBRATION_COMPLETE_MARKER
    if not complete_marker.is_file():
        raise RuntimeError(
            "AmbiCal profile has no CALIBRATION_COMPLETE marker."
        )
    completion = _strict_json_object(complete_marker, "completion marker")
    if completion.get("status") != "complete":
        raise RuntimeError("AmbiCal completion marker does not say complete.")
    if completion.get("profile_filename") != profile_path.name:
        raise RuntimeError("AmbiCal completion marker names another profile.")

    raw_bytes = profile_path.read_bytes()
    profile_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if completion.get("profile_sha256") != profile_sha256:
        raise RuntimeError("AmbiCal profile SHA-256 does not match completion marker.")
    _validate_hash_sidecar(
        profile_path.with_suffix(profile_path.suffix + ".sha256"),
        profile_path.name,
        profile_sha256,
    )

    manifest_name = completion.get("manifest_filename")
    if manifest_name != CALIBRATION_MANIFEST_FILENAME:
        raise RuntimeError("AmbiCal completion marker names an invalid manifest.")
    manifest_path = (artifact_dir / str(manifest_name)).resolve(strict=True)
    if manifest_path.parent != artifact_dir:
        raise RuntimeError("AmbiCal calibration manifest escaped its directory.")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if completion.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("AmbiCal manifest SHA-256 does not match completion marker.")
    manifest_sidecar_name = completion.get("manifest_sha256_sidecar")
    expected_manifest_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if manifest_sidecar_name != expected_manifest_sidecar.name:
        raise RuntimeError("AmbiCal completion marker names an invalid manifest sidecar.")
    _validate_hash_sidecar(
        expected_manifest_sidecar,
        manifest_path.name,
        manifest_sha256,
    )
    manifest = _strict_json_object(manifest_path, "calibration manifest")
    if manifest.get("artifact_type") != "ambical_offline_calibration":
        raise RuntimeError("AmbiCal calibration manifest has the wrong artifact type.")
    outputs = manifest.get("outputs")
    manifest_profile = outputs.get("profile") if isinstance(outputs, Mapping) else None
    if not isinstance(manifest_profile, Mapping):
        raise RuntimeError("AmbiCal manifest has no profile output identity.")
    if (
        manifest_profile.get("absolute_path") != str(profile_path)
        or manifest_profile.get("sha256") != profile_sha256
    ):
        raise RuntimeError("AmbiCal manifest profile identity does not match.")

    try:
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"AmbiCal profile is not valid UTF-8 JSON: {profile_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("AmbiCal profile JSON must contain an object.")
    if "profile" in payload:
        raw_profile = payload.get("profile")
        raw_scope = payload.get("scope", {})
        if not isinstance(raw_profile, Mapping):
            raise ValueError("AmbiCal profile.profile must be a JSON object.")
        if not isinstance(raw_scope, Mapping):
            raise ValueError("AmbiCal profile.scope must be a JSON object.")
    else:
        raw_profile = payload
        raw_scope = {}
    profile = AmbiCalProfile.from_dict(raw_profile)
    return LoadedAmbiCalProfile(
        profile=profile,
        absolute_path=str(profile_path),
        sha256=profile_sha256,
        scope=dict(raw_scope),
    )


def _strict_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"AmbiCal {label} is not strict UTF-8 JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"AmbiCal {label} must contain a JSON object.")
    return value


def _validate_hash_sidecar(path: Path, artifact_name: str, expected_digest: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"AmbiCal SHA-256 sidecar is missing: {path}")
    parts = path.read_text(encoding="ascii").strip().split(maxsplit=1)
    if len(parts) != 2:
        raise RuntimeError(f"AmbiCal SHA-256 sidecar is malformed: {path}")
    digest, recorded_name = parts
    if digest != expected_digest or recorded_name != artifact_name:
        raise RuntimeError(f"AmbiCal SHA-256 sidecar does not match: {path}")


def ambical_result_identifier(
    config: AmbiCalConfig,
    experiment_identity: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return a filesystem-safe identifier unique to the resolved method config."""

    version = re.sub(r"[^A-Za-z0-9_-]+", "_", config.method_version).strip("_")
    version = version or "ambical"
    identity = {
        "utility_signature": config.utility_signature(),
        "experiment": dict(experiment_identity or {}),
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{version}_{digest[:16]}"


def profile_required(config: AmbiCalConfig) -> bool:
    """Whether active components consume a precomputed calibration profile."""

    return bool(
        config.switches.false_positive_prior or config.switches.null_gate
    )


def validate_profile_for_config(
    config: AmbiCalConfig,
    profile: Optional[AmbiCalProfile],
) -> None:
    """Fail before an experiment starts if its profile cannot satisfy the method."""

    if profile_required(config) and profile is None:
        raise ValueError(
            "ambical_parameters.profile_path is required when the "
            "false-positive prior or Null gate is enabled."
        )
    if profile is None:
        return
    if profile_required(config) and profile.utility_signature is None:
        raise ValueError(
            "A formal AmbiCal profile must contain utility_signature."
        )
    if (
        config.switches.false_positive_prior
        and profile.false_positive_priors is None
    ):
        raise ValueError(
            "AmbiCal false-positive correction is enabled, but the profile "
            "does not contain false_positive_priors."
        )
    if config.switches.null_gate and not profile.null_strengths:
        raise ValueError(
            "AmbiCal Null gating is enabled, but the profile has no "
            "null_strengths."
        )
    if (
        profile.utility_signature is not None
        and profile.utility_signature != config.utility_signature()
    ):
        raise ValueError(
            "The AmbiCal profile utility signature does not match the "
            "resolved method configuration."
        )


def validate_profile_scope(
    config: AmbiCalConfig,
    scope: Mapping[str, Any],
    expected_scope: Mapping[str, Any],
) -> None:
    """Bind calibrated artifacts to the exact online collection configuration."""

    if not profile_required(config):
        return
    missing = sorted(set(expected_scope) - set(scope))
    if missing:
        raise ValueError(
            "AmbiCal profile.scope is missing collection identity fields: "
            + ", ".join(missing)
        )
    mismatches = [
        key
        for key, expected in expected_scope.items()
        if scope.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "AmbiCal profile.scope does not match the experiment: "
            + ", ".join(sorted(mismatches))
        )

    calibration_hashes = []
    if config.switches.false_positive_prior:
        calibration_hashes.append(
            _require_sha256_scope_value(scope, "false_positive_calibration_sha256")
        )
        _require_zero_overlap(scope, "false_positive_test_overlap_count")
    if config.switches.null_gate:
        calibration_hashes.append(
            _require_sha256_scope_value(scope, "null_calibration_sha256")
        )
        _require_zero_overlap(scope, "null_test_overlap_count")
    if len(calibration_hashes) == 2 and calibration_hashes[0] == calibration_hashes[1]:
        raise ValueError(
            "False-positive and Null calibration artifacts must be independent."
        )
    if len(calibration_hashes) == 2:
        _require_zero_overlap(scope, "false_positive_null_overlap_count")


def _require_sha256_scope_value(scope: Mapping[str, Any], key: str) -> str:
    value = str(scope.get(key, "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(
            f"AmbiCal profile.scope.{key} must be a SHA-256 digest."
        )
    return value


def _require_zero_overlap(scope: Mapping[str, Any], key: str) -> None:
    value = scope.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise ValueError(
            f"AmbiCal profile.scope.{key} must be the integer 0."
        )
