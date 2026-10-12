"""Strict input validation for independent AmbiCal calibration runs.

This module validates files and experiment identity before a persistent Chroma
collection, embedding model, or translation model is opened.  Formal test
questions are read only to prove set independence; they are never returned as
queries to the calibration scorer.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from retrieval_pipelines.ambical import AmbiCalConfig
from retrieval_pipelines.ambical.calibration import (
    CALIBRATION_ALGORITHM_VERSION,
)
from utils.ambical_artifacts import sha256_file
from utils.ambical_experiment import build_ambical_config


CALIBRATION_CONFIG_SCHEMA_VERSION = 1
CALIBRATION_ANSWER = "Not found in context"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "calibration_algorithm_version",
    "source_language",
    "target_language",
    "document_language",
    "chunk_size",
    "overlap_size",
    "collection_name",
    "chroma_persist_dir",
    "manual_path",
    "manual_sha256",
    "embedding_model_name",
    "embedding_max_length",
    "llm_models",
    "translator_model_type",
    "translator_model_name",
    "translation_prompt_path",
    "translation_prompt_sha256",
    "translation_cache_mode",
    "translation_cache_path",
    "false_positive_calibration",
    "null_calibration",
    "formal_test_sets",
    "method_config",
    "output_dir",
    "experiment_tier",
}

_OPTIONAL_TOP_LEVEL_FIELDS = {"experiment_tier"}

_FILE_SPEC_FIELDS = {"path", "sha256", "expected_count"}


@dataclass(frozen=True)
class QuestionArtifact:
    path: Path
    sha256: str
    expected_count: int
    questions: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CalibrationRunInputs:
    config_path: Path
    raw_config: Mapping[str, Any]
    source_language: str
    target_language: str
    document_language: str
    chunk_size: int
    overlap_size: int
    collection_name: str
    chroma_persist_dir: Path
    manual_path: Path
    manual_sha256: str
    embedding_model_name: str
    embedding_max_length: int
    llm_models: Mapping[str, Any]
    translator_model_type: str
    translator_model_name: str
    translation_prompt_path: Path
    translation_prompt_sha256: str
    translation_cache_mode: str
    translation_cache_path: Path
    false_positive: QuestionArtifact
    null: QuestionArtifact
    formal_test_sets: Tuple[QuestionArtifact, ...]
    method_config: AmbiCalConfig
    output_dir: Path
    overlap_counts: Mapping[str, int]


def _strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid strict UTF-8 JSON: {path}") from exc


def _required_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Calibration config {name} must be a non-empty string.")
    return value.strip()


def _positive_integer(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Calibration config {name} must be an integer of at least 1.")
    return value


def _sha256_value(payload: Mapping[str, Any], name: str) -> str:
    value = _required_string(payload, name).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Calibration config {name} must be a SHA-256 digest.")
    return value


def _absolute_path(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Calibration config {name} must be an absolute path.")
    return path.resolve(strict=False)


def normalize_question(value: str) -> str:
    """Normalize only for leakage/duplicate checks, never for model input."""

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized)
        .split()
    )


def _load_question_artifact(
    raw_spec: Any,
    *,
    label: str,
    expected_purpose: str | None,
) -> QuestionArtifact:
    if not isinstance(raw_spec, Mapping):
        raise ValueError(f"Calibration config {label} must be a JSON object.")
    unknown = sorted(set(raw_spec) - _FILE_SPEC_FIELDS)
    missing = sorted(_FILE_SPEC_FIELDS - set(raw_spec))
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        if missing:
            details.append("missing=" + ",".join(missing))
        raise ValueError(f"Invalid {label} file specification: {'; '.join(details)}")
    path = _absolute_path(_required_string(raw_spec, "path"), f"{label}.path")
    if not path.is_file():
        raise FileNotFoundError(f"Calibration question file does not exist: {path}")
    expected_sha256 = _sha256_value(raw_spec, "sha256")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Calibration question hash mismatch for {label}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    expected_count = _positive_integer(raw_spec, "expected_count")
    payload = _strict_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Question artifact must contain a JSON array: {path}")
    if len(payload) != expected_count:
        raise ValueError(
            f"Question count mismatch for {label}: {len(payload)} != {expected_count}"
        )
    questions = []
    seen = set()
    seen_ids = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} item {index} must be a JSON object.")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{label} item {index} has no non-empty question.")
        normalized = normalize_question(question)
        if not normalized or normalized in seen:
            raise ValueError(f"{label} contains a duplicate question at item {index}.")
        seen.add(normalized)
        if expected_purpose is not None:
            if item.get("answer") != CALIBRATION_ANSWER:
                raise ValueError(f"{label} item {index} has the wrong answer label.")
            if item.get("page") is not None:
                raise ValueError(f"{label} item {index} must use page=null.")
            if item.get("calibration_purpose") != expected_purpose:
                raise ValueError(f"{label} item {index} has the wrong purpose.")
            calibration_id = item.get("calibration_id")
            if not isinstance(calibration_id, str) or not calibration_id.strip():
                raise ValueError(f"{label} item {index} has no calibration_id.")
            if calibration_id in seen_ids:
                raise ValueError(f"{label} contains duplicate calibration_id {calibration_id}.")
            seen_ids.add(calibration_id)
        questions.append(dict(item))
    return QuestionArtifact(
        path=path,
        sha256=actual_sha256,
        expected_count=expected_count,
        questions=tuple(questions),
    )


def _overlap_count(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> int:
    right_questions = {normalize_question(item["question"]) for item in right}
    return sum(
        1 for item in left if normalize_question(item["question"]) in right_questions
    )


def load_calibration_run_inputs(config_path: str | Path) -> CalibrationRunInputs:
    """Validate every immutable calibration input before model/database loading."""

    path = Path(config_path).expanduser().resolve(strict=True)
    payload = _strict_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("AmbiCal calibration config must be a JSON object.")
    unknown = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    missing = sorted((_TOP_LEVEL_FIELDS - _OPTIONAL_TOP_LEVEL_FIELDS) - set(payload))
    if unknown or missing:
        details = []
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        if missing:
            details.append("missing=" + ",".join(missing))
        raise ValueError("Invalid calibration config fields: " + "; ".join(details))
    if payload.get("schema_version") != CALIBRATION_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CALIBRATION_CONFIG_SCHEMA_VERSION}."
        )
    if payload.get("calibration_algorithm_version") != CALIBRATION_ALGORITHM_VERSION:
        raise ValueError(
            "calibration_algorithm_version does not match the implemented estimator."
        )
    if "experiment_tier" in payload:
        _required_string(payload, "experiment_tier")

    source_language = _required_string(payload, "source_language")
    target_language = _required_string(payload, "target_language")
    document_language = _required_string(payload, "document_language")
    if source_language.casefold() == target_language.casefold():
        raise ValueError("Cross-language calibration requires distinct source and target languages.")
    if target_language.casefold() != document_language.casefold():
        raise ValueError("target_language must match document_language.")

    chunk_size = _positive_integer(payload, "chunk_size")
    overlap_size = payload.get("overlap_size")
    if isinstance(overlap_size, bool) or not isinstance(overlap_size, int) or overlap_size < 0:
        raise ValueError("overlap_size must be a non-negative integer.")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size.")
    embedding_max_length = _positive_integer(payload, "embedding_max_length")

    collection_name = _required_string(payload, "collection_name")
    chroma_persist_dir = _absolute_path(
        _required_string(payload, "chroma_persist_dir"), "chroma_persist_dir"
    )
    if not chroma_persist_dir.is_dir():
        raise FileNotFoundError(f"Chroma persist directory does not exist: {chroma_persist_dir}")
    manual_path = _absolute_path(_required_string(payload, "manual_path"), "manual_path")
    if not manual_path.is_file():
        raise FileNotFoundError(f"Manual does not exist: {manual_path}")
    manual_sha256 = _sha256_value(payload, "manual_sha256")
    if sha256_file(manual_path) != manual_sha256:
        raise ValueError("Manual SHA-256 does not match calibration config.")

    embedding_model_name = _required_string(payload, "embedding_model_name")
    embedding_model_path = _absolute_path(embedding_model_name, "embedding_model_name")
    if not embedding_model_path.is_dir():
        raise FileNotFoundError(f"Embedding model directory does not exist: {embedding_model_path}")

    llm_models = payload.get("llm_models")
    if not isinstance(llm_models, Mapping):
        raise ValueError("llm_models must be a JSON object.")
    translator_model_type = _required_string(payload, "translator_model_type").casefold()
    if translator_model_type != "ollama":
        raise ValueError(
            "Formal calibration currently requires translator_model_type=ollama "
            "so the exact model binary can be bound by SHA-256."
        )
    translator_model_name = _required_string(payload, "translator_model_name")
    typed_models = llm_models.get(translator_model_type)
    if not isinstance(typed_models, Mapping) or translator_model_name not in typed_models:
        raise ValueError("Configured translator model alias is missing from llm_models.")

    translation_prompt_path = _absolute_path(
        _required_string(payload, "translation_prompt_path"),
        "translation_prompt_path",
    )
    if not translation_prompt_path.is_file():
        raise FileNotFoundError(f"Translation prompt does not exist: {translation_prompt_path}")
    translation_prompt_sha256 = _sha256_value(payload, "translation_prompt_sha256")
    if sha256_file(translation_prompt_path) != translation_prompt_sha256:
        raise ValueError("Translation prompt SHA-256 does not match calibration config.")

    cache_mode = _required_string(payload, "translation_cache_mode")
    if cache_mode not in {"read_write", "read_only"}:
        raise ValueError("translation_cache_mode must be read_write or read_only; refresh is forbidden.")
    cache_path = _absolute_path(
        _required_string(payload, "translation_cache_path"),
        "translation_cache_path",
    )
    if cache_mode == "read_write" and cache_path.exists():
        raise FileExistsError(f"A new translation cache path is required: {cache_path}")
    if cache_mode == "read_only" and not cache_path.is_file():
        raise FileNotFoundError(f"Read-only translation cache does not exist: {cache_path}")

    output_dir = _absolute_path(_required_string(payload, "output_dir"), "output_dir")
    if output_dir.exists():
        raise FileExistsError(f"Calibration output_dir already exists: {output_dir}")

    false_positive = _load_question_artifact(
        payload.get("false_positive_calibration"),
        label="false_positive_calibration",
        expected_purpose="false_positive_prior",
    )
    null = _load_question_artifact(
        payload.get("null_calibration"),
        label="null_calibration",
        expected_purpose="null_distribution",
    )
    if false_positive.sha256 == null.sha256:
        raise ValueError("False-positive and Null calibration artifacts must differ.")
    raw_formal = payload.get("formal_test_sets")
    if not isinstance(raw_formal, list) or not raw_formal:
        raise ValueError("formal_test_sets must be a non-empty JSON array.")
    formal = tuple(
        _load_question_artifact(
            item,
            label=f"formal_test_sets[{index}]",
            expected_purpose=None,
        )
        for index, item in enumerate(raw_formal)
    )
    formal_questions = tuple(
        question for artifact in formal for question in artifact.questions
    )
    overlap_counts: Dict[str, int] = {
        "false_positive_test_overlap_count": _overlap_count(
            false_positive.questions, formal_questions
        ),
        "null_test_overlap_count": _overlap_count(null.questions, formal_questions),
        "false_positive_null_overlap_count": _overlap_count(
            false_positive.questions, null.questions
        ),
    }
    if any(overlap_counts.values()):
        raise ValueError(
            "Calibration/test question leakage detected: "
            + ", ".join(f"{key}={value}" for key, value in overlap_counts.items())
        )
    false_ids = {item["calibration_id"] for item in false_positive.questions}
    null_ids = {item["calibration_id"] for item in null.questions}
    if false_ids.intersection(null_ids):
        raise ValueError("False-positive and Null calibration IDs must be disjoint.")

    method_values = payload.get("method_config")
    if not isinstance(method_values, Mapping):
        raise ValueError("method_config must be a JSON object.")
    method_config = build_ambical_config(
        {"source_language": source_language, "method_config": dict(method_values)}
    )
    if not method_config.switches.translation_branch:
        raise ValueError("Vietnamese calibration requires translation_branch=true.")
    # false_positive_prior=false is a permitted calibration mode for the
    # w/o FP prior ablation: the builder still stores FP priors and Null
    # strengths, and the online retriever applies FP priors only when the
    # switch is enabled.
    # null_gate=false is a permitted calibration mode for the w/o Null Gate
    # ablation: the builder still measures and stores the same pre-gate Null
    # strengths (the probe path never applies the gate), and the online
    # retriever applies Null gating only when the switch is enabled.

    return CalibrationRunInputs(
        config_path=path,
        raw_config=dict(payload),
        source_language=source_language,
        target_language=target_language,
        document_language=document_language,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        collection_name=collection_name,
        chroma_persist_dir=chroma_persist_dir,
        manual_path=manual_path,
        manual_sha256=manual_sha256,
        embedding_model_name=str(embedding_model_path),
        embedding_max_length=embedding_max_length,
        llm_models=dict(llm_models),
        translator_model_type=translator_model_type,
        translator_model_name=translator_model_name,
        translation_prompt_path=translation_prompt_path,
        translation_prompt_sha256=translation_prompt_sha256,
        translation_cache_mode=cache_mode,
        translation_cache_path=cache_path,
        false_positive=false_positive,
        null=null,
        formal_test_sets=formal,
        method_config=method_config,
        output_dir=output_dir,
        overlap_counts=overlap_counts,
    )


def calibration_resolution() -> Mapping[str, float | int]:
    """Document the exact empirical p-value resolution for 30 Null samples."""

    return {
        "null_sample_count": 30,
        "minimum_empirical_right_tail_probability": 1.0 / 31.0,
        "empirical_probability_step": 1.0 / 31.0,
    }


def validate_finite_profile_values(values: Sequence[float], label: str) -> None:
    if not values:
        raise ValueError(f"{label} cannot be empty.")
    if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in values):
        raise ValueError(f"{label} must contain only finite values in [0, 1].")
