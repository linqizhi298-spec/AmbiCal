"""Build one immutable AmbiCal calibration profile without running QA.

The command uses only independent false-positive and Null questions.  Formal
test files are opened solely for leakage checks performed before any model or
collection is loaded.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import chromadb
import ollama
import torch

from llm_connectors.llm_connector_manager import LLMConnectorManager
from query_translation import QueryTranslator
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
from retrieval_pipelines.ambical import (
    AmbiCalProfile,
    build_false_positive_priors,
    build_null_strengths,
    read_full_collection,
)
from utils.ambical_artifacts import (
    collection_fingerprint,
    directory_manifest_fingerprint,
    sha256_file,
    translation_scope_identity,
)
from utils.ambical_calibration import (
    CalibrationRunInputs,
    calibration_resolution,
    load_calibration_run_inputs,
    validate_finite_profile_values,
)
from utils.ambical_experiment import (
    validate_profile_for_config,
    validate_profile_scope,
)


FORMAL_PYTHON_EXECUTABLE = Path(r"D:\Anaconda\anaconda3\envs\ambical\python.exe")
FORMAL_GPU_LABEL = "RTX 5070 Ti"
PROFILE_FILENAME = "ambical_profile.json"
TRACE_FILENAME = "ambical_calibration_trace.json"
MANIFEST_FILENAME = "ambical_calibration_manifest.json"
INCOMPLETE_MARKER = "CALIBRATION_INCOMPLETE"
COMPLETE_MARKER = "CALIBRATION_COMPLETE"


def _validate_formal_runtime() -> None:
    if platform.system().casefold() != "windows":
        raise RuntimeError("Formal AmbiCal calibration must run on Windows.")
    actual_python = Path(sys.executable).resolve(strict=True)
    expected_python = FORMAL_PYTHON_EXECUTABLE.resolve(strict=True)
    if os.path.normcase(str(actual_python)) != os.path.normcase(str(expected_python)):
        raise RuntimeError(
            "Formal AmbiCal calibration requires Python at "
            f"{expected_python}; current executable is {actual_python}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Formal AmbiCal calibration requires CUDA.")
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    if not any(FORMAL_GPU_LABEL.casefold() in name.casefold() for name in gpu_names):
        raise RuntimeError(
            f"Formal AmbiCal calibration requires {FORMAL_GPU_LABEL}; "
            f"detected GPUs: {gpu_names}."
        )


def _strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    data = _strict_json_bytes(payload)
    with path.open("xb") as output_file:
        output_file.write(data)


def _write_hash_sidecar(
    path: Path,
    *,
    sidecar_path: Optional[Path] = None,
    artifact_name: Optional[str] = None,
) -> Path:
    sidecar = sidecar_path or path.with_suffix(path.suffix + ".sha256")
    digest = sha256_file(path)
    with sidecar.open("x", encoding="ascii", newline="\n") as output_file:
        output_file.write(f"{digest}  {artifact_name or path.name}\n")
    return sidecar


def _snapshot_config(inputs: CalibrationRunInputs) -> Path:
    destination = inputs.output_dir / "active_calibration_config.snapshot.json"
    with inputs.config_path.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target)
    if sha256_file(destination) != sha256_file(inputs.config_path):
        raise RuntimeError("Calibration config changed while it was being snapshotted.")
    return destination


def _ollama_digest(actual_model: str) -> str:
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
        if str(name) == actual_model and digest:
            normalized = str(digest).removeprefix("sha256:").lower()
            if len(normalized) == 64 and all(char in "0123456789abcdef" for char in normalized):
                return normalized
            raise RuntimeError(f"Ollama reported an invalid digest for {actual_model}.")
    raise RuntimeError(f"Could not resolve the installed Ollama model {actual_model!r}.")


def _checkpoint_writable_cache(path: Path) -> None:
    with sqlite3.connect(str(path), timeout=30.0) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _remove_chroma_working_copy(path: Path, output_dir: Path) -> None:
    resolved_path = path.resolve(strict=True)
    resolved_output = output_dir.resolve(strict=True)
    try:
        relative = resolved_path.relative_to(resolved_output)
    except ValueError as exc:
        raise RuntimeError("Chroma working copy escaped the calibration output.") from exc
    if (
        len(relative.parts) != 1
        or not resolved_path.name.startswith(".ambical_chroma_snapshot_")
    ):
        raise RuntimeError(f"Refusing to remove unexpected path: {resolved_path}")
    shutil.rmtree(resolved_path)
    if resolved_path.exists():
        raise RuntimeError("Could not remove the temporary Chroma working copy.")


def _token_usage(connector: Any) -> Mapping[str, int]:
    method = getattr(connector, "token_usage", None)
    if not callable(method):
        raise RuntimeError("Translator connector does not expose exact token usage.")
    usage = {str(key): int(value) for key, value in dict(method()).items()}
    if usage.get("unreported_calls", 0):
        raise RuntimeError("A translation call did not report exact token usage.")
    return usage


def _question_identity(inputs: CalibrationRunInputs) -> Mapping[str, Any]:
    return {
        "false_positive": {
            "absolute_path": str(inputs.false_positive.path),
            "sha256": inputs.false_positive.sha256,
            "count": inputs.false_positive.expected_count,
        },
        "null": {
            "absolute_path": str(inputs.null.path),
            "sha256": inputs.null.sha256,
            "count": inputs.null.expected_count,
        },
        "formal_test_sets_used_only_for_overlap_check": [
            {
                "absolute_path": str(artifact.path),
                "sha256": artifact.sha256,
                "count": artifact.expected_count,
            }
            for artifact in inputs.formal_test_sets
        ],
        "overlap_counts": dict(inputs.overlap_counts),
    }


def preflight_calibration(config_path: str | Path) -> Mapping[str, Any]:
    """Validate parameters, files, hashes, and independence without loading models."""

    inputs = load_calibration_run_inputs(config_path)
    return {
        "status": "ready_for_offline_calibration",
        "config_path": str(inputs.config_path),
        "collection_name": inputs.collection_name,
        "method_utility_signature": inputs.method_config.utility_signature(),
        "question_artifacts": _question_identity(inputs),
        "output_dir": str(inputs.output_dir),
        "translation_cache_mode": inputs.translation_cache_mode,
        "translation_cache_path": str(inputs.translation_cache_path),
        "qa_or_evaluator_will_run": False,
    }


def build_profile(
    config_path: str | Path,
    *,
    runtime_validator: Optional[Callable[[], None]] = None,
    execution_contract: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Execute only offline calibration and publish an immutable profile."""

    started_at = datetime.now(timezone.utc).isoformat()
    inputs = load_calibration_run_inputs(config_path)
    validator = runtime_validator or _validate_formal_runtime
    validator()
    inputs.output_dir.mkdir(parents=True, exist_ok=False)
    incomplete_marker = inputs.output_dir / INCOMPLETE_MARKER
    _write_exclusive(
        incomplete_marker,
        {
            "status": "incomplete",
            "started_at_utc": started_at,
            "config_path": str(inputs.config_path),
            "recovery": (
                "Do not reuse this output directory or a partially created "
                "read_write translation cache; choose new paths for a retry."
            ),
        },
    )
    config_snapshot = _snapshot_config(inputs)

    cache_before = (
        sha256_file(inputs.translation_cache_path)
        if inputs.translation_cache_mode == "read_only"
        else None
    )
    prompt_template = inputs.translation_prompt_path.read_text(encoding="utf-8")
    connector_manager = LLMConnectorManager(dict(inputs.llm_models))
    connector = connector_manager.get_connector(
        inputs.translator_model_type, inputs.translator_model_name
    )
    connector_config = dict(getattr(connector, "config", {}) or {})
    actual_model = str(
        connector_config.get("name")
        or getattr(connector, "ollama_model_name", "")
    ).strip()
    translator_digest = _ollama_digest(actual_model)
    translator = QueryTranslator(
        connector=connector,
        prompt_template=prompt_template,
        cache_path=inputs.translation_cache_path,
        cache_mode=inputs.translation_cache_mode,
        translator_alias=inputs.translator_model_name,
        require_new_cache=inputs.translation_cache_mode == "read_write",
        model_binary_sha256=translator_digest,
    )
    if str(translator.actual_model) != actual_model:
        raise RuntimeError("Translator actual model changed during initialization.")
    translation_identity = translation_scope_identity(
        translator,
        source_language=inputs.source_language,
        target_language=inputs.target_language,
        prompt_file_sha256=inputs.translation_prompt_sha256,
        ollama_sha256=translator_digest,
    )
    embedding_identity_before = directory_manifest_fingerprint(
        inputs.embedding_model_name
    )
    chroma_storage_before = directory_manifest_fingerprint(
        inputs.chroma_persist_dir
    )
    working_root = Path(
        tempfile.mkdtemp(
            prefix=".ambical_chroma_snapshot_",
            dir=str(inputs.output_dir),
        )
    )
    working_persist_dir = working_root / "persist"
    shutil.copytree(inputs.chroma_persist_dir, working_persist_dir)
    if directory_manifest_fingerprint(inputs.chroma_persist_dir) != chroma_storage_before:
        raise RuntimeError("Original Chroma persist directory changed while being copied.")

    # Chroma is opened only on the disposable copy.  The production persist
    # directory is never handed to PersistentClient.
    client = chromadb.PersistentClient(path=str(working_persist_dir))
    collection = client.get_collection(inputs.collection_name)
    collection_before = collection_fingerprint(collection)
    embedder = EmbeddingRetriever(
        model_name=inputs.embedding_model_name,
        max_length=inputs.embedding_max_length,
    )
    collection_snapshot = read_full_collection(collection, inputs.method_config)
    if collection_snapshot.count != collection_before["count"]:
        raise RuntimeError("Collection snapshot and provenance count differ.")

    false_positive_priors, false_positive_trace = build_false_positive_priors(
        collection_snapshot,
        embedder,
        inputs.false_positive.questions,
        inputs.method_config,
        translator,
        inputs.source_language,
        inputs.target_language,
    )
    null_strengths, null_trace = build_null_strengths(
        collection,
        embedder,
        inputs.null.questions,
        inputs.method_config,
        false_positive_priors,
        translator,
        inputs.source_language,
        inputs.target_language,
    )
    validate_finite_profile_values(null_strengths, "null_strengths")
    if len(null_strengths) != inputs.null.expected_count:
        raise RuntimeError("Null strength count does not match the Null question count.")
    profile = AmbiCalProfile(
        false_positive_priors=false_positive_priors,
        null_strengths=tuple(null_strengths),
        utility_signature=inputs.method_config.utility_signature(),
    )
    validate_profile_for_config(inputs.method_config, profile)

    collection_after = collection_fingerprint(collection)
    if collection_after != collection_before:
        raise RuntimeError("Chroma collection changed during read-only calibration.")
    chroma_storage_after = directory_manifest_fingerprint(
        inputs.chroma_persist_dir
    )
    if chroma_storage_after != chroma_storage_before:
        raise RuntimeError(
            "Original Chroma persist directory changed during calibration."
        )
    embedding_identity_after = directory_manifest_fingerprint(
        inputs.embedding_model_name
    )
    if embedding_identity_after != embedding_identity_before:
        raise RuntimeError("Embedding model directory changed during calibration.")
    if inputs.translation_cache_mode == "read_write":
        _checkpoint_writable_cache(inputs.translation_cache_path)
    cache_after = sha256_file(inputs.translation_cache_path)
    if cache_before is not None and cache_after != cache_before:
        raise RuntimeError("Read-only translation cache changed during calibration.")
    _token_usage(connector)

    client.clear_system_cache()
    del collection
    del client
    gc.collect()
    _remove_chroma_working_copy(working_root, inputs.output_dir)

    scope = {
        "collection_name": inputs.collection_name,
        "collection_sha256": collection_before["sha256"],
        "document_language": inputs.document_language,
        "chunk_size": inputs.chunk_size,
        "overlap_size": inputs.overlap_size,
        "embedding_model_name": inputs.embedding_model_name,
        "embedding_max_length": inputs.embedding_max_length,
        "embedding_model_manifest_sha256": embedding_identity_before["sha256"],
        "embedding_model_manifest_scheme": embedding_identity_before["scheme"],
        "embedding_model_file_count": embedding_identity_before["file_count"],
        "embedding_model_total_bytes": embedding_identity_before["total_bytes"],
        "calibration_algorithm_version": inputs.raw_config[
            "calibration_algorithm_version"
        ],
        **translation_identity,
        "false_positive_calibration_sha256": inputs.false_positive.sha256,
        "null_calibration_sha256": inputs.null.sha256,
        **dict(inputs.overlap_counts),
    }
    expected_scope = dict(scope)
    validate_profile_scope(inputs.method_config, scope, expected_scope)

    trace_path = inputs.output_dir / TRACE_FILENAME
    profile_path = inputs.output_dir / PROFILE_FILENAME
    pending_profile_path = inputs.output_dir / f".{PROFILE_FILENAME}.pending"
    trace_payload = {
        "calibration_algorithm_version": inputs.raw_config[
            "calibration_algorithm_version"
        ],
        "false_positive": false_positive_trace,
        "null": null_trace,
    }
    _write_exclusive(trace_path, trace_payload)
    profile_payload = {
        "profile": profile.to_dict(),
        "scope": scope,
        "calibration": {
            "algorithm_version": inputs.raw_config[
                "calibration_algorithm_version"
            ],
            "manual_path": str(inputs.manual_path),
            "manual_sha256": inputs.manual_sha256,
            "source_language": inputs.source_language,
            "target_language": inputs.target_language,
            "collection_count": collection_before["count"],
            "collection_digest_scheme": collection_before["scheme"],
            "translation_prompt_path": str(inputs.translation_prompt_path),
            **translation_identity,
            "translation_cache_mode": inputs.translation_cache_mode,
            "translation_cache_path": str(inputs.translation_cache_path),
            "translation_cache_sha256": cache_after,
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": sha256_file(trace_path),
        },
    }
    _write_exclusive(pending_profile_path, profile_payload)
    profile_sha256 = sha256_file(pending_profile_path)
    profile_sidecar = _write_hash_sidecar(
        pending_profile_path,
        sidecar_path=profile_path.with_suffix(profile_path.suffix + ".sha256"),
        artifact_name=profile_path.name,
    )

    manifest_path = inputs.output_dir / MANIFEST_FILENAME
    manifest_payload = {
        "artifact_type": "ambical_offline_calibration",
        "method": "AmbiCal",
        "method_version": inputs.method_config.method_version,
        "execution_contract": dict(
            execution_contract
            or {
                "tier": "formal_windows",
                "platform": "Windows",
                "required_gpu_label": FORMAL_GPU_LABEL,
                "required_python_executable": str(FORMAL_PYTHON_EXECUTABLE),
            }
        ),
        "utility_signature": inputs.method_config.utility_signature(),
        "resolved_method_config": inputs.method_config.diagnostics(),
        "question_artifacts": _question_identity(inputs),
        "manual": {
            "absolute_path": str(inputs.manual_path),
            "sha256": inputs.manual_sha256,
        },
        "collection": {
            "persist_dir": str(inputs.chroma_persist_dir),
            **collection_before,
            "name": inputs.collection_name,
            "unchanged_after_calibration": True,
            "persist_directory_manifest": dict(chroma_storage_before),
            "persist_directory_unchanged_after_calibration": True,
            "opened_disposable_working_copy_only": True,
            "working_copy_removed_before_publication": True,
        },
        "embedding_model": dict(embedding_identity_before),
        "translation": {
            "prompt_path": str(inputs.translation_prompt_path),
            **translation_identity,
            "cache_mode": inputs.translation_cache_mode,
            "cache_path": str(inputs.translation_cache_path),
            "cache_initial_sha256": cache_before,
            "cache_final_sha256": cache_after,
        },
        "empirical_null_resolution": calibration_resolution(),
        "config_snapshot": {
            "source_path": str(inputs.config_path),
            "snapshot_path": str(config_snapshot.resolve()),
            "sha256": sha256_file(config_snapshot),
        },
        "qa_generation_run": False,
        "evaluator_run": False,
        "formal_test_questions_retrieved": False,
        "outputs": {
            "profile": {
                "absolute_path": str(profile_path.resolve()),
                "sha256": profile_sha256,
            },
            "profile_sha256_sidecar": str(profile_sidecar.resolve()),
            "trace": {
                "absolute_path": str(trace_path.resolve()),
                "sha256": sha256_file(trace_path),
            },
        },
    }
    _write_exclusive(manifest_path, manifest_payload)
    manifest_sidecar = _write_hash_sidecar(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    complete_marker = inputs.output_dir / COMPLETE_MARKER
    _write_exclusive(
        complete_marker,
        {
            "status": "complete",
            "profile_filename": profile_path.name,
            "profile_sha256": profile_sha256,
            "manifest_filename": manifest_path.name,
            "manifest_sha256": manifest_sha256,
            "manifest_sha256_sidecar": manifest_sidecar.name,
        },
    )
    incomplete_marker.unlink()
    os.replace(pending_profile_path, profile_path)
    logging.info("AmbiCal calibration profile created: %s", profile_path)
    return profile_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an independent AmbiCal calibration profile without QA."
    )
    parser.add_argument("--config", required=True, help="Calibration JSON config path.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate config, hashes, and set independence without loading models.",
    )
    arguments = parser.parse_args(argv)
    if arguments.preflight:
        print(
            json.dumps(
                preflight_calibration(arguments.config),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    build_profile(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
