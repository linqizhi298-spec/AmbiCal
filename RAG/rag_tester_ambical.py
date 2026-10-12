"""Dedicated experiment entry point for AmbiCal.

The shared QA, generation, evaluation, and result-writing pipeline remains in
``rag_tester.py``.  This entry point enforces method isolation so AmbiCal
runs cannot collide with other retrieval configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import ollama

from rag_tester import RagTester as BaseRagTester
from utils.ambical_artifacts import normalize_sha256_digest
from utils.ambical_experiment import (
    ambical_result_identifier,
    validate_profile_scope,
)


FORMAL_PYTHON_EXECUTABLE = Path(
    r"D:\Anaconda\anaconda3\envs\ambical\python.exe"
)
FORMAL_GPU_LABEL = "RTX 5070 Ti"
EVALUATION_PARTITIONS_KEY = "ambical_evaluation_partitions"


def _resolve_evaluation_partitions(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve primary and mechanism-stress datasets without changing old runs."""

    dataset_paths = payload.get("question_dataset_paths")
    if not isinstance(dataset_paths, Mapping) or not dataset_paths:
        if payload.get(EVALUATION_PARTITIONS_KEY) is not None:
            raise ValueError(
                f"{EVALUATION_PARTITIONS_KEY} requires non-empty "
                "question_dataset_paths."
            )
        return (), ()
    configured = tuple(str(name) for name in dataset_paths)
    raw = payload.get(EVALUATION_PARTITIONS_KEY)
    if raw is None:
        return configured, ()
    if not isinstance(raw, Mapping):
        raise ValueError(f"{EVALUATION_PARTITIONS_KEY} must be a JSON object.")
    unknown_fields = sorted(
        set(raw) - {"primary_metric_datasets", "mechanism_stress_datasets"}
    )
    if unknown_fields:
        raise ValueError(
            f"Unknown {EVALUATION_PARTITIONS_KEY} fields: "
            + ", ".join(unknown_fields)
        )

    def _names(key: str) -> tuple[str, ...]:
        value = raw.get(key, [])
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"{EVALUATION_PARTITIONS_KEY}.{key} must be a list of names.")
        normalized = tuple(item.strip() for item in value)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{EVALUATION_PARTITIONS_KEY}.{key} contains duplicates.")
        return normalized

    primary = _names("primary_metric_datasets")
    stress = _names("mechanism_stress_datasets")
    if not primary and not stress:
        raise ValueError(
            f"{EVALUATION_PARTITIONS_KEY} must assign at least one dataset."
        )
    overlap = sorted(set(primary) & set(stress))
    if overlap:
        raise ValueError(
            "Primary and mechanism-stress datasets must be disjoint: "
            + ", ".join(overlap)
        )
    assigned = set(primary) | set(stress)
    if assigned != set(configured):
        missing = sorted(set(configured) - assigned)
        unknown = sorted(assigned - set(configured))
        details = []
        if missing:
            details.append("unassigned=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise ValueError(
            f"{EVALUATION_PARTITIONS_KEY} must partition every configured dataset: "
            + "; ".join(details)
        )
    return primary, stress


def _validate_formal_runtime() -> None:
    """Fail before initialization when the formal Windows/GPU contract is unmet."""

    if platform.system().casefold() != "windows":
        raise RuntimeError("Formal AmbiCal experiments must run on Windows.")
    actual_python = Path(sys.executable).resolve(strict=True)
    expected_python = FORMAL_PYTHON_EXECUTABLE.resolve(strict=True)
    if os.path.normcase(str(actual_python)) != os.path.normcase(str(expected_python)):
        raise RuntimeError(
            "Formal AmbiCal experiments require Python at "
            f"{expected_python}; current executable is {actual_python}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Formal AmbiCal experiments require CUDA.")
    gpu_names = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
    if not any(FORMAL_GPU_LABEL.casefold() in name.casefold() for name in gpu_names):
        raise RuntimeError(
            f"Formal AmbiCal experiments require {FORMAL_GPU_LABEL}; "
            f"detected GPUs: {gpu_names}."
        )


def _load_and_preflight_config(config_path: str) -> Mapping[str, Any]:
    """Reject fallback configs, reevaluation, and any reusable output directory."""

    path = Path(config_path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"AmbiCal config is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("AmbiCal config must contain a JSON object.")
    if payload.get("enable_reevaluation") is True:
        raise ValueError(
            "AmbiCal entry point does not permit reevaluation because formal "
            "reruns must use a new output directory."
        )
    configured_output = payload.get("output_dir")
    if not isinstance(configured_output, str) or not configured_output.strip():
        raise ValueError(
            "AmbiCal config must provide an explicit absolute output_dir."
        )
    output_path = Path(configured_output).expanduser()
    if not output_path.is_absolute():
        raise ValueError("AmbiCal output_dir must be an absolute path.")
    if output_path.exists():
        raise FileExistsError(
            "AmbiCal output_dir already exists; use a new directory for "
            f"every formal run: {output_path}"
        )
    _resolve_evaluation_partitions(payload)
    return payload


def _installed_ollama_digest(actual_model: str) -> str:
    """Resolve one installed model to its normalized full SHA-256 digest."""

    requested = str(actual_model).strip()
    if not requested:
        raise RuntimeError("AmbiCal translator has no actual Ollama model name.")
    response = ollama.list()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, Mapping):
        models = response.get("models", [])
    for model in models or []:
        if isinstance(model, Mapping):
            name = model.get("model") or model.get("name")
            digest = model.get("digest")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            digest = getattr(model, "digest", None)
        if str(name) == requested:
            return normalize_sha256_digest(
                digest,
                f"installed Ollama model {requested} digest",
            )
    raise RuntimeError(f"Could not resolve installed Ollama model {requested!r}.")


class RagTester(BaseRagTester):
    """RAG tester restricted to one fully specified AmbiCal run."""

    _dedicated_ambical_entrypoint = True

    def _ambical_translator_binary_sha256(self, actual_model: str) -> str:
        return _installed_ollama_digest(actual_model)

    def __init__(self, config_path: str) -> None:
        preflight_payload = _load_and_preflight_config(config_path)
        super().__init__(config_path=config_path)
        primary, stress = _resolve_evaluation_partitions(preflight_payload)
        if not primary and not stress:
            primary = tuple(getattr(self, "target_dataset_names", ()))
        self.primary_metric_dataset_names = primary
        self.mechanism_stress_dataset_names = stress
        if self.retrieval_algorithms_to_test != ["ambical"]:
            raise ValueError(
                "rag_tester_ambical requires exactly one 'ambical' "
                "retrieval algorithm."
            )
        if self.adaptive_k_enabled:
            raise ValueError(
                "rag_tester_ambical requires the shared official "
                "adaptive_k_enabled flag to be false; AmbiCal uses its own "
                "independently recorded stopping switch."
            )
        if len(self.language_configs) != 1:
            raise ValueError(
                "An AmbiCal config must contain exactly one document "
                "language because its translation and calibration profile are "
                "collection-specific."
            )
        if len(self.chunk_sizes_to_test) != 1 or len(self.overlap_sizes_to_test) != 1:
            raise ValueError(
                "An AmbiCal config must contain exactly one chunk size and "
                "one overlap so a profile cannot be reused across chunkings."
            )
        if self.total_questions_to_answer < 1:
            raise ValueError(
                "AmbiCal requires at least one valid configured question."
            )
        language = str(self.language_configs[0]["language"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", language):
            raise ValueError(
                "AmbiCal language must be a filesystem-safe identifier "
                "containing only letters, digits, '_' or '-'."
            )
        self.result_manager.refuse_overwrite = True
        self.ambical_reranker = self._initialize_ambical_reranker()
        self._expected_result_targets = self._build_expected_result_targets()

    def _initialize_ambical_reranker(self):
        """Build the v4 cross-encoder scorer; None for every other version."""

        config = getattr(self, "ambical_config", None)
        if (
            config is None
            or config.method_version != "ambical-online-v4"
            or not config.switches.context_correction
        ):
            return None
        raw_path = (self.ambical_params or {}).get("reranker_model_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                "ambical-online-v4 requires "
                "ambical_parameters.reranker_model_path pointing at the "
                "local bge-reranker-v2-m3 model directory."
            )
        path = Path(self._resolve_config_relative_path(raw_path))
        if not path.is_dir():
            raise ValueError(f"reranker_model_path is not a directory: {path}")
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(str(path), max_length=1024)

        def _rerank(pairs):
            scores = model.predict(
                list(pairs),
                batch_size=8,
                show_progress_bar=False,
                activation_fn=torch.nn.Sigmoid(),
            )
            tensor = torch.as_tensor(scores, dtype=torch.float64).flatten()
            return [float(value) for value in tensor.tolist()]

        self.ambical_reranker_path = str(path)
        self.ambical_experiment_identity = {
            **self.ambical_experiment_identity,
            "reranker_artifact": _directory_identity(path),
        }
        return _rerank

    def _calculate_and_print_metrics(self, final_results_list_for_metrics):
        """Keep deliberately constructed stress questions out of primary metrics."""

        primary_names = set(getattr(self, "primary_metric_dataset_names", ()))
        primary_rows = [
            row
            for row in final_results_list_for_metrics
            if str(row.get("dataset")) in primary_names
        ]
        metrics = super()._calculate_and_print_metrics(primary_rows)
        if isinstance(metrics, dict):
            metrics["evaluation_scope"] = {
                "primary_metric_datasets": list(
                    getattr(self, "primary_metric_dataset_names", ())
                ),
                "mechanism_stress_datasets_excluded": list(
                    getattr(self, "mechanism_stress_dataset_names", ())
                ),
            }
        return metrics

    def _calculate_dataset_success_rates(self, evaluated_results_by_dataset):
        rates = super()._calculate_dataset_success_rates(
            evaluated_results_by_dataset
        )
        primary_names = set(getattr(self, "primary_metric_dataset_names", ()))
        return {name: value for name, value in rates.items() if name in primary_names}

    def _build_expected_result_targets(self) -> Mapping[str, Path]:
        """Resolve unique, in-directory result targets before a run can start."""

        output_root = Path(self.output_dir).resolve()
        language = str(self.language_configs[0]["language"])
        targets = {}
        seen_paths = {}
        for model_name in self.question_models_to_test:
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError(
                    "AmbiCal question model aliases must be non-empty strings."
                )
            filename = self.result_manager._generate_filename(
                "ambical",
                language,
                model_name,
                self.chunk_sizes_to_test[0],
                self.overlap_sizes_to_test[0],
                self.result_k_identifier,
            )
            path = (output_root / filename).resolve()
            try:
                path.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(
                    f"AmbiCal result path escapes output_dir: {path}"
                ) from exc
            if path.parent != output_root:
                raise ValueError(
                    f"AmbiCal result must be a direct child of output_dir: {path}"
                )
            collision_key = os.path.normcase(str(path))
            if collision_key in seen_paths:
                raise ValueError(
                    "AmbiCal question model aliases map to the same result "
                    f"file: {seen_paths[collision_key]!r} and {model_name!r}."
                )
            if path.exists():
                raise FileExistsError(
                    f"AmbiCal result target already exists: {path}"
                )
            seen_paths[collision_key] = model_name
            targets[model_name] = path
        if not targets:
            raise ValueError("AmbiCal requires at least one question model.")
        return targets

    def bind_ollama_model_digests(
        self, digests: Mapping[str, Mapping[str, str]]
    ) -> None:
        """Bind resolved model binaries into the identity before any result write."""

        if getattr(self, "ambical_profile_scope", {}):
            translator_alias = str(
                getattr(self, "ambical_translator_model_name", "")
            ).strip()
            translator_details = digests.get(translator_alias)
            if not translator_alias or not isinstance(translator_details, Mapping):
                raise RuntimeError(
                    "A calibrated cross-language profile requires the translator "
                    "Ollama digest before the formal run can start."
                )
            current_digest = normalize_sha256_digest(
                translator_details.get("digest"),
                "formal translator Ollama SHA-256",
            )
            initialized_digest = getattr(
                getattr(self, "ambical_translator", None),
                "model_binary_sha256",
                None,
            )
            if initialized_digest != current_digest:
                raise RuntimeError(
                    "The translator Ollama binary changed after AmbiCal "
                    "translator initialization."
                )
            validate_profile_scope(
                self.ambical_config,
                self.ambical_profile_scope,
                {
                    "translator_ollama_sha256": current_digest
                },
            )

        self.ambical_experiment_identity = {
            **self.ambical_experiment_identity,
            "ollama_model_digests": {
                str(alias): {
                    "model": str(details["model"]),
                    "digest": str(details["digest"]),
                }
                for alias, details in sorted(digests.items())
            },
        }
        self.result_k_identifier = ambical_result_identifier(
            self.ambical_config,
            self.ambical_experiment_identity,
        )
        self._expected_result_targets = self._build_expected_result_targets()

    def validate_completed_outputs(self) -> list[str]:
        """Require one complete, strict-JSON result for every configured model."""

        language = str(self.language_configs[0]["language"])
        output_paths = []
        for model_name in self.question_models_to_test:
            path = self._expected_result_targets[model_name]
            if path.is_symlink():
                raise RuntimeError(
                    f"AmbiCal result must not be a symbolic link: {path}"
                )
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(
                    "AmbiCal run did not produce its expected result: "
                    f"{path}"
                )
            try:
                payload = json.loads(
                    path.read_text(encoding="utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"Non-finite JSON constant: {value}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"AmbiCal result is not strict JSON: {path}"
                ) from exc
            parameters = payload.get("test_run_parameters", {})
            if parameters.get("retrieval_algorithm") != "ambical":
                raise RuntimeError(f"Result is mislabeled: {path}")
            if parameters.get("result_k_identifier") != self.result_k_identifier:
                raise RuntimeError(f"Result identifier mismatch: {path}")
            expected_parameters = {
                "question_model": model_name,
                "language_tested": language,
                "chunk_size": self.chunk_sizes_to_test[0],
                "overlap_size": self.overlap_sizes_to_test[0],
            }
            mismatches = [
                key
                for key, expected in expected_parameters.items()
                if parameters.get(key) != expected
            ]
            if mismatches:
                raise RuntimeError(
                    "AmbiCal result parameter mismatch for "
                    f"{path}: {', '.join(mismatches)}"
                )

            records = [
                item
                for dataset in payload.get("per_dataset_details", {}).values()
                for item in dataset.get("results", [])
                if isinstance(item, dict)
            ]
            if len(records) != self.total_questions_to_answer:
                raise RuntimeError(
                    "AmbiCal result question count mismatch for "
                    f"{path}: {len(records)} != {self.total_questions_to_answer}"
                )
            expected_questions = Counter(
                (str(dataset_name), str(question["question"]))
                for dataset_name, questions in self.loaded_datasets.items()
                for question in questions
            )
            actual_questions = Counter(
                (str(item.get("dataset")), str(item.get("question")))
                for item in records
            )
            if actual_questions != expected_questions:
                raise RuntimeError(
                    f"AmbiCal result question identity mismatch: {path}"
                )
            primary_question_count = sum(
                len(self.loaded_datasets.get(name, []))
                for name in self.primary_metric_dataset_names
            )
            reported_primary_count = payload.get("overall_metrics", {}).get(
                "total_questions"
            )
            if reported_primary_count != primary_question_count:
                raise RuntimeError(
                    "AmbiCal primary metric question count mismatch for "
                    f"{path}: {reported_primary_count} != {primary_question_count}"
                )
            scope = payload.get("overall_metrics", {}).get("evaluation_scope", {})
            if scope != {
                "primary_metric_datasets": list(
                    self.primary_metric_dataset_names
                ),
                "mechanism_stress_datasets_excluded": list(
                    self.mechanism_stress_dataset_names
                ),
            }:
                raise RuntimeError(
                    f"AmbiCal evaluation scope mismatch: {path}"
                )
            failed = [
                item.get("question", "<unknown>")
                for item in records
                if item.get("qa_error")
                or item.get("eval_error")
                or item.get("ambical_result") is None
                or not isinstance(item.get("selected_k"), int)
            ]
            if failed:
                raise RuntimeError(
                    "AmbiCal result contains failed or incomplete questions: "
                    + "; ".join(str(item) for item in failed[:5])
                )
            output_paths.append(str(path))
        if not output_paths:
            raise RuntimeError("AmbiCal run produced zero result files.")
        return output_paths

    def token_usage_summary(self) -> Mapping[str, Any]:
        per_connector = {}
        totals = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "unreported_calls": 0,
        }
        for label, connector in self.runtime_llm_connectors.items():
            usage_method = getattr(connector, "token_usage", None)
            if not callable(usage_method):
                raise RuntimeError(
                    f"Connector {label!r} does not expose token usage."
                )
            usage = dict(usage_method())
            per_connector[label] = usage
            for key in totals:
                totals[key] += int(usage.get(key, 0))
        if totals["unreported_calls"]:
            raise RuntimeError(
                "At least one LLM call did not report exact token usage; refusing "
                "to mark the AmbiCal run complete."
            )
        return {"totals": totals, "per_connector": per_connector}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_identity(path: str | Path) -> Mapping[str, Any]:
    """Content-bind a local model directory while ignoring download cache files."""

    root = Path(path).resolve(strict=True)
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(root).parts
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(root).as_posix()
        size = item.stat().st_size
        file_digest = _sha256_file(item)
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return {
        "absolute_path": str(root),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def _snapshot_config(tester: RagTester, config_path: str) -> Mapping[str, str]:
    source = Path(config_path).expanduser().resolve(strict=True)
    destination = Path(tester.output_dir).resolve() / "active_config.snapshot.json"
    if destination.exists():
        raise FileExistsError(f"Config snapshot already exists: {destination}")
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file)
    snapshot_sha256 = _sha256_file(destination)
    expected_sha256 = tester.ambical_experiment_identity.get(
        "active_config_sha256"
    )
    if snapshot_sha256 != expected_sha256:
        raise RuntimeError(
            "AmbiCal config changed between initialization and snapshot."
        )
    return {
        "source_path": str(source),
        "snapshot_path": str(destination),
        "sha256": snapshot_sha256,
    }


def _ollama_model_digests(tester: RagTester) -> Mapping[str, Mapping[str, str]]:
    aliases = list(tester.question_models_to_test)
    aliases.append(tester.config_loader.get_evaluator_model_name())
    if tester.ambical_translator_model_name:
        aliases.append(tester.ambical_translator_model_name)
    requested = {}
    for alias in aliases:
        if not alias:
            continue
        llm_type, model_config = tester.config_loader.get_llm_type_and_config(alias)
        if str(llm_type).casefold() == "ollama":
            requested[str(alias)] = str((model_config or {}).get("name", "")).strip()
    if not requested:
        return {}
    response = ollama.list()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, Mapping):
        models = response.get("models", [])
    installed = {}
    for model in models or []:
        if isinstance(model, Mapping):
            name = model.get("model") or model.get("name")
            digest = model.get("digest")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
            digest = getattr(model, "digest", None)
        if name and digest:
            installed[str(name)] = str(digest)
    missing = sorted(
        actual_name
        for actual_name in requested.values()
        if not actual_name or actual_name not in installed
    )
    if missing:
        raise RuntimeError(
            "Could not resolve Ollama model digest(s): " + ", ".join(missing)
        )
    invalid_digests = sorted(
        actual_name
        for actual_name in requested.values()
        if not re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", installed[actual_name])
    )
    if invalid_digests:
        raise RuntimeError(
            "Ollama did not report a full SHA-256 digest for: "
            + ", ".join(invalid_digests)
        )
    return {
        alias: {"model": actual_name, "digest": installed[actual_name]}
        for alias, actual_name in requested.items()
    }


def _runtime_environment() -> Mapping[str, Any]:
    packages = {}
    for package in (
        "chromadb",
        "langchain-ollama",
        "ollama",
        "sentence-transformers",
        "torch",
        "transformers",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    cuda_available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpus": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if cuda_available
            else []
        ),
    }


def _numeric_summary(values: Sequence[float]) -> Mapping[str, Any]:
    finite = [float(value) for value in values]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def _selected_k_summary(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    values = [
        int(item["selected_k"])
        for item in records
        if isinstance(item.get("selected_k"), int)
    ]
    summary = dict(_numeric_summary(values))
    summary["distribution"] = {
        str(key): value for key, value in sorted(Counter(values).items())
    }
    return summary


def _mechanism_stress_summary(
    dataset_name: str,
    records: Sequence[Mapping[str, Any]],
    source_questions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Summarize the method-document indicators for a diagnostic stress stratum."""

    metadata_by_question = {
        str(item.get("question")): item for item in source_questions
    }
    detail_rows = []
    gate_risks = []
    context_splits = []
    route_divergences = []
    consistencies = []
    path_conflicts = []
    evidence_strengths = []
    triggered = 0
    corrected = 0
    top1_changes = 0
    translation_active = 0
    nonzero_false_positive_prior = 0
    merge_applied = 0
    evaluator_yes = 0
    valid_evaluations = 0
    null_regions: Counter[str] = Counter()

    for record in records:
        method_result = record.get("ambical_result") or {}
        diagnostics = method_result.get("diagnostics") or {}
        gate = diagnostics.get("context_gate") or {}
        correction = diagnostics.get("context_correction") or {}
        adaptive = diagnostics.get("adaptive_k") or {}
        candidates = diagnostics.get("candidates") or []
        question = str(record.get("question", ""))
        source = metadata_by_question.get(question, {})

        risk = float(gate.get("risk", 0.0))
        split = float(gate.get("context_split", 0.0))
        route = float(gate.get("route_total_variation", 0.0))
        consistency = float(diagnostics.get("combined_consistency", 0.0))
        conflict = float(gate.get("path_conflict", 0.0))
        strength = float(gate.get("evidence_strength", 0.0))
        gate_risks.append(risk)
        context_splits.append(split)
        route_divergences.append(route)
        consistencies.append(consistency)
        path_conflicts.append(conflict)
        evidence_strengths.append(strength)

        gate_triggered = bool(gate.get("triggered"))
        correction_executed = bool(correction.get("executed"))
        triggered += int(gate_triggered)
        corrected += int(correction_executed)
        translation_active += int(bool(method_result.get("translation_active")))
        merge_applied += int(bool((diagnostics.get("merge") or {}).get("applied")))
        if any(
            float(candidate.get("source_false_positive_prior", 0.0)) > 0.0
            or float(candidate.get("target_false_positive_prior") or 0.0) > 0.0
            for candidate in candidates
            if isinstance(candidate, Mapping)
        ):
            nonzero_false_positive_prior += 1

        def _top_id(field: str) -> Optional[str]:
            valid = [
                item
                for item in candidates
                if isinstance(item, Mapping)
                and isinstance(item.get(field), (int, float))
            ]
            if not valid:
                return None
            return str(max(valid, key=lambda item: float(item[field])).get("chunk_id"))

        initial_top1 = _top_id("initial_fused_score")
        corrected_top1 = _top_id("base_value")
        changed = (
            correction_executed
            and initial_top1 is not None
            and corrected_top1 is not None
            and initial_top1 != corrected_top1
        )
        top1_changes += int(changed)
        null_region = str(adaptive.get("null_region", "missing"))
        null_regions[null_region] += 1
        judgment = record.get("self_evaluation")
        if judgment in {"yes", "no"} and not record.get("eval_error"):
            valid_evaluations += 1
            evaluator_yes += int(judgment == "yes")

        phrases = correction.get("phrases") or []
        betas = []
        phrase_risks = []
        for phrase in phrases:
            for branch in (phrase.get("source"), phrase.get("target")):
                if not isinstance(branch, Mapping):
                    continue
                if isinstance(branch.get("beta"), (int, float)):
                    betas.append(float(branch["beta"]))
                phrase_risks.extend(
                    float(value)
                    for value in branch.get("risk", [])
                    if isinstance(value, (int, float))
                )

        detail_rows.append(
            {
                "stress_id": source.get("stress_id"),
                "question": question,
                "intended_indicator_pattern": source.get("indicator_targets"),
                "observed_indicators": {
                    "B0_competition": float(gate.get("competition", 0.0)),
                    "Dbar_context_distance": float(gate.get("weighted_distance", 0.0)),
                    "D0_context_split": split,
                    "J_route": route,
                    "chi_consistency": consistency,
                    "C_path": conflict,
                    "E0_evidence_strength": strength,
                    "A_ctx": risk,
                    "g_ctx_triggered": gate_triggered,
                    "counterfactual_executed": correction_executed,
                    "counterfactual_generated_count": int(
                        correction.get("generated_count", 0)
                    ),
                    "counterfactual_selected_count": int(
                        correction.get("selected_count", 0)
                    ),
                    "max_beta": max(betas, default=0.0),
                    "max_phrase_risk": max(phrase_risks, default=0.0),
                    "top1_changed_after_correction": changed,
                    "selected_k": record.get("selected_k"),
                    "null_region": null_region,
                },
                "evaluator_judgment": judgment,
            }
        )

    return {
        "dataset": dataset_name,
        "role": "mechanism_stress_only_excluded_from_primary_metrics",
        "question_count": len(records),
        "evaluator_success": (
            evaluator_yes / valid_evaluations if valid_evaluations else None
        ),
        "indicator_mapping": {
            "B0(q)": "context_gate.competition",
            "D_bar(q)": "context_gate.weighted_distance",
            "D0(q)": "context_gate.context_split",
            "J_route(q)": "context_gate.route_total_variation",
            "chi(q)": "diagnostics.combined_consistency",
            "C_path(q)": "context_gate.path_conflict",
            "E0(q)": "context_gate.evidence_strength",
            "A_ctx(q)": "context_gate.risk",
            "g_ctx(q)": "context_gate.triggered",
            "beta_l(a)": "context_correction.phrases.*.(source|target).beta",
            "M_l(c_i)": "context_correction.phrases.*.(source|target).risk",
        },
        "coverage": {
            "translation_active": translation_active,
            "nonzero_false_positive_prior": nonzero_false_positive_prior,
            "context_gate_triggered": triggered,
            "context_correction_executed": corrected,
            "top1_changed_after_correction": top1_changes,
            "structured_merge_applied": merge_applied,
        },
        "indicator_summaries": {
            "D0_context_split": _numeric_summary(context_splits),
            "J_route": _numeric_summary(route_divergences),
            "chi_consistency": _numeric_summary(consistencies),
            "C_path": _numeric_summary(path_conflicts),
            "E0_evidence_strength": _numeric_summary(evidence_strengths),
            "A_ctx": _numeric_summary(gate_risks),
        },
        "selected_k": _selected_k_summary(records),
        "null_region_distribution": dict(sorted(null_regions.items())),
        "questions": detail_rows,
    }


def _write_run_manifest(
    tester: RagTester,
    *,
    config_snapshot: Mapping[str, str],
    ollama_digests: Mapping[str, Mapping[str, str]],
    result_paths: Sequence[str],
    token_usage: Mapping[str, Any],
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    execution_contract: Optional[Mapping[str, Any]] = None,
) -> str:
    output_dir = Path(tester.output_dir).resolve()
    manifest_path = output_dir / "ambical_run_manifest.json"
    manifest_hash_path = output_dir / "ambical_run_manifest.sha256"
    if manifest_path.exists() or manifest_hash_path.exists():
        raise FileExistsError("AmbiCal provenance manifest already exists.")
    outputs = [
        {
            "absolute_path": str(Path(path).resolve()),
            "sha256": _sha256_file(path),
        }
        for path in result_paths
    ]
    result_summaries = []
    for path in result_paths:
        result_payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records_by_dataset = {
            str(name): [
                item
                for item in content.get("results", [])
                if isinstance(item, dict)
            ]
            for name, content in result_payload.get(
                "per_dataset_details", {}
            ).items()
        }
        records = [
            item for values in records_by_dataset.values() for item in values
        ]
        stress_summaries = [
            _mechanism_stress_summary(
                name,
                records_by_dataset.get(name, []),
                tester.loaded_datasets.get(name, []),
            )
            for name in tester.mechanism_stress_dataset_names
        ]
        result_summaries.append(
            {
                "absolute_path": str(Path(path).resolve()),
                "selected_k": _selected_k_summary(records),
                "primary_metric_datasets": list(
                    tester.primary_metric_dataset_names
                ),
                "mechanism_stress_datasets": list(
                    tester.mechanism_stress_dataset_names
                ),
                "end_to_end_metrics": result_payload.get("overall_metrics", {}),
                "mechanism_stress_summaries": stress_summaries,
                "timing": result_payload.get("timing", {}),
                "retrieval_metrics": {
                    "status": "not_computed_by_end_to_end_qa_entrypoint",
                    "required_separate_metrics": [
                        "Recall@k",
                        "MRR",
                        "nDCG",
                        "Evidence Hit Rate",
                    ],
                },
            }
        )
    payload = {
        "method": "AmbiCal",
        "method_version": tester.ambical_config.method_version,
        "execution_contract": dict(
            execution_contract
            or {
                "tier": "formal_windows",
                "platform": "Windows",
                "required_gpu_label": FORMAL_GPU_LABEL,
                "required_python_executable": str(FORMAL_PYTHON_EXECUTABLE),
            }
        ),
        "utility_signature": tester.ambical_config.utility_signature(),
        "result_identifier": tester.result_k_identifier,
        "resolved_method_config": tester.ambical_config.diagnostics(),
        "experiment_identity": tester.ambical_experiment_identity,
        "config_snapshot": dict(config_snapshot),
        "command": {
            "argv": [sys.executable, *sys.argv],
            "python_argv": list(sys.argv),
            "windows_command_line": subprocess.list2cmdline(
                [sys.executable, *sys.argv]
            ),
            "working_directory": os.getcwd(),
        },
        "runtime_environment": _runtime_environment(),
        "ollama_model_digests": dict(ollama_digests),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "duration_seconds": duration_seconds,
        "token_usage": dict(token_usage),
        "outputs": outputs,
        "result_summaries": result_summaries,
    }
    with manifest_path.open("x", encoding="utf-8") as manifest_file:
        json.dump(
            payload,
            manifest_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    manifest_sha256 = _sha256_file(manifest_path)
    with manifest_hash_path.open("x", encoding="ascii") as hash_file:
        hash_file.write(f"{manifest_sha256}  {manifest_path.name}\n")
    return str(manifest_path)


def start_rag_tests(
    config_path: str,
    *,
    runtime_validator: Optional[Callable[[], None]] = None,
    execution_contract: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Run the dedicated AmbiCal experiment entry point."""

    started_wall = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    logging.info("--- Starting AmbiCal tests (config: %s) ---", config_path)
    validator = runtime_validator or _validate_formal_runtime
    validator()
    tester = RagTester(config_path=config_path)
    ollama_digests = _ollama_model_digests(tester)
    tester.bind_ollama_model_digests(ollama_digests)
    config_snapshot = _snapshot_config(tester, config_path)
    tester.run_tests()
    result_paths = tester.validate_completed_outputs()
    token_usage = tester.token_usage_summary()
    ended_at = datetime.now(timezone.utc).isoformat()
    duration_seconds = time.perf_counter() - started_wall
    manifest_path = _write_run_manifest(
        tester,
        config_snapshot=config_snapshot,
        ollama_digests=ollama_digests,
        result_paths=result_paths,
        token_usage=token_usage,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        execution_contract=execution_contract,
    )
    logging.info(
        "--- AmbiCal tests completed successfully; manifest: %s ---",
        manifest_path,
    )
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one fully specified AmbiCal experiment."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a dedicated AmbiCal JSON configuration.",
    )
    arguments = parser.parse_args(argv)
    start_rag_tests(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
