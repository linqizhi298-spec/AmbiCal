"""Read-only provenance helpers shared by AmbiCal runs and calibration."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sha256_digest(value: Any, name: str = "SHA-256") -> str:
    normalized = str(value or "").strip().removeprefix("sha256:").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{name} must be a full SHA-256 digest.")
    return normalized


def directory_manifest_fingerprint(directory: str | Path) -> Dict[str, Any]:
    """Hash every regular file in a model directory using stable relative paths."""

    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"AmbiCal artifact is not a directory: {root}")
    paths = []
    for path in root.rglob("*"):
        if path.is_symlink() and path.is_dir():
            raise ValueError(
                "AmbiCal model directories cannot contain directory "
                f"symlinks: {path}"
            )
        if path.is_file():
            paths.append(path)
    if not paths:
        raise ValueError(f"AmbiCal artifact directory is empty: {root}")

    combined = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        size = path.stat().st_size
        total_bytes += size
        record = {
            "relative_path": path.relative_to(root).as_posix(),
            "size": size,
            "sha256": sha256_file(path),
        }
        combined.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        combined.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "sha256": combined.hexdigest(),
        "scheme": "sorted-file-manifest-sha256-v1:relative_path+size+sha256",
    }


def translation_scope_identity(
    translator: Any,
    *,
    source_language: str,
    target_language: str,
    prompt_file_sha256: str,
    ollama_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the translation identity fields that a profile must bind."""

    source = str(source_language).strip()
    target = str(target_language).strip()
    if not source or not target:
        raise ValueError("Translation source and target languages must be non-empty.")
    if translator is None:
        raise ValueError("A translator is required for cross-language profile identity.")
    alias = str(getattr(translator, "translator_alias", "")).strip()
    actual_model = str(getattr(translator, "actual_model", "")).strip()
    parser_version = str(getattr(translator, "PARSER_VERSION", "")).strip()
    if not alias or not actual_model or not parser_version:
        raise ValueError("Translator alias, actual model, and parser version are required.")
    try:
        model_parameters = json.loads(
            str(getattr(translator, "model_parameters_json", "") or "{}")
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Translator model parameters are not valid JSON.") from exc
    if not isinstance(model_parameters, Mapping):
        raise ValueError("Translator model parameters must be a JSON object.")
    prompt_content_sha256 = normalize_sha256_digest(
        getattr(translator, "prompt_sha256", ""),
        "translator prompt content SHA-256",
    )
    identity: Dict[str, Any] = {
        "source_language": source,
        "target_language": target,
        "translator_alias": alias,
        "translator_actual_model": actual_model,
        "translator_model_parameters": dict(model_parameters),
        "translation_prompt_file_sha256": normalize_sha256_digest(
            prompt_file_sha256,
            "translation prompt file SHA-256",
        ),
        "translation_prompt_sha256": prompt_content_sha256,
        "translation_parser_version": parser_version,
    }
    if ollama_sha256 is not None:
        identity["translator_ollama_sha256"] = normalize_sha256_digest(
            ollama_sha256,
            "translator Ollama SHA-256",
        )
    return identity


def json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return json_safe_value(to_list())
    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        return json_safe_value(scalar_item())
    raise TypeError(
        "AmbiCal collection contains a non-JSON provenance value: "
        f"{type(value).__name__}"
    )


def collection_batch_value(batch: Mapping[str, Any], key: str, index: int) -> Any:
    values = batch.get(key)
    if values is None:
        return None
    if index >= len(values):
        raise RuntimeError(f"AmbiCal collection batch has incomplete {key}.")
    return values[index]


def collection_fingerprint(collection: Any, batch_size: int = 256) -> Dict[str, Any]:
    """Hash all stored records without changing collection state.

    Sorting per-record digests makes the result independent of Chroma's returned
    order while retaining IDs, documents, embeddings, and metadata in the hash.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer of at least 1.")
    count = int(collection.count())
    if count < 1:
        raise ValueError("AmbiCal collection must contain at least one chunk.")
    record_digests = []
    seen_ids = set()
    for offset in range(0, count, batch_size):
        batch = collection.get(
            limit=min(batch_size, count - offset),
            offset=offset,
            include=["documents", "embeddings", "metadatas"],
        )
        raw_ids = batch.get("ids")
        identifiers = list(raw_ids) if raw_ids is not None else []
        if not identifiers:
            raise RuntimeError(
                "AmbiCal could not read every collection record for the "
                "provenance fingerprint."
            )
        for index, identifier in enumerate(identifiers):
            chunk_id = str(identifier)
            if chunk_id in seen_ids:
                raise RuntimeError(f"AmbiCal collection returned duplicate ID {chunk_id!r}.")
            seen_ids.add(chunk_id)
            record = {
                "id": identifier,
                "document": collection_batch_value(batch, "documents", index),
                "embedding": collection_batch_value(batch, "embeddings", index),
                "metadata": collection_batch_value(batch, "metadatas", index),
            }
            canonical = json.dumps(
                json_safe_value(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            record_digests.append(
                hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            )
    if len(record_digests) != count:
        raise RuntimeError(
            "AmbiCal collection changed or was incompletely read while "
            f"fingerprinting: {len(record_digests)} != {count}."
        )
    combined = hashlib.sha256()
    for record_digest in sorted(record_digests):
        combined.update(record_digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "count": count,
        "sha256": combined.hexdigest(),
        "scheme": "sorted-record-sha256-v1:id+document+embedding+metadata",
    }
