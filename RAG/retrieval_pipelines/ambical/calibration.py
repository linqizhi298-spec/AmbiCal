"""Pure, independent calibration primitives for AmbiCal.

This module deliberately stops at constructing in-memory calibration values.  It
does not read question files, write profiles, select experiment parameters, run
QA, or call an evaluator.  Callers are responsible for immutable artifact and
provenance handling around these functions.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .config import AmbiCalConfig
from .models import AmbiCalProfile, FalsePositivePriors
from .numerics import (
    cosine_scores,
    l2_normalize,
)
from .retriever import AmbiCalRetriever


CALIBRATION_ALGORITHM_VERSION = "ambical-calibration-v2"


def _sequence(value: Any, name: str) -> List[Any]:
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence.")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence.") from exc


def _embedding_tuple(value: Any, name: str) -> Tuple[float, ...]:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64).flatten()
    except Exception as exc:
        raise ValueError(f"{name} is not a valid embedding.") from exc
    if tensor.numel() == 0:
        raise ValueError(f"{name} is empty.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    norm = float(torch.linalg.vector_norm(tensor).item())
    if norm <= 0:
        raise ValueError(f"{name} is a zero vector.")
    return tuple(float(item) for item in tensor.tolist())


@dataclass(frozen=True)
class FullCollectionSnapshot:
    """A complete, ID-sorted, in-memory view of a Chroma collection."""

    ids: Tuple[str, ...]
    documents: Tuple[str, ...]
    embeddings: Tuple[Tuple[float, ...], ...]
    metadatas: Tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        raw_ids = _sequence(self.ids, "ids")
        raw_documents = _sequence(self.documents, "documents")
        raw_embeddings = _sequence(self.embeddings, "embeddings")
        raw_metadatas = _sequence(self.metadatas, "metadatas")
        lengths = {
            len(raw_ids),
            len(raw_documents),
            len(raw_embeddings),
            len(raw_metadatas),
        }
        if lengths == {0}:
            raise ValueError("A full collection snapshot cannot be empty.")
        if len(lengths) != 1:
            raise ValueError("Full collection snapshot fields have different lengths.")

        rows: List[
            Tuple[str, str, Tuple[float, ...], Mapping[str, Any]]
        ] = []
        seen = set()
        dimension: Optional[int] = None
        for index, (raw_id, document, embedding, metadata) in enumerate(
            zip(raw_ids, raw_documents, raw_embeddings, raw_metadatas)
        ):
            chunk_id = str(raw_id)
            if not chunk_id:
                raise ValueError(f"Collection row {index} has an empty ID.")
            if chunk_id in seen:
                raise ValueError(f"Collection contains duplicate ID {chunk_id!r}.")
            seen.add(chunk_id)
            if document is None:
                raise ValueError(f"Collection row {chunk_id!r} has no document text.")
            vector = _embedding_tuple(embedding, f"embedding for {chunk_id!r}")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("Collection embeddings have inconsistent dimensions.")
            if metadata is None:
                metadata_value: Mapping[str, Any] = {}
            elif isinstance(metadata, Mapping):
                metadata_value = dict(metadata)
            else:
                raise ValueError(
                    f"Metadata for collection row {chunk_id!r} must be an object."
                )
            rows.append((chunk_id, str(document), vector, metadata_value))

        # Python's sort is stable.  IDs are unique, so this also gives callers a
        # canonical score-vector order independent of Chroma's batch order.
        rows.sort(key=lambda row: row[0])
        object.__setattr__(self, "ids", tuple(row[0] for row in rows))
        object.__setattr__(self, "documents", tuple(row[1] for row in rows))
        object.__setattr__(self, "embeddings", tuple(row[2] for row in rows))
        object.__setattr__(self, "metadatas", tuple(row[3] for row in rows))

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def embedding_dimension(self) -> int:
        return len(self.embeddings[0])

    # Explicit aliases make the relationship to the online ChunkRecord naming
    # clear without duplicating any data.
    @property
    def chunk_ids(self) -> Tuple[str, ...]:
        return self.ids

    @property
    def texts(self) -> Tuple[str, ...]:
        return self.documents

    @property
    def metadata(self) -> Tuple[Mapping[str, Any], ...]:
        return self.metadatas


def _normalization_contract(config: AmbiCalConfig) -> Tuple[bool, float]:
    if not isinstance(config, AmbiCalConfig):
        raise TypeError("config must be an AmbiCalConfig.")
    required = config.require_normalized_collection_embeddings
    if not isinstance(required, bool):
        raise ValueError(
            "require_normalized_collection_embeddings must be a boolean."
        )
    tolerance = config.collection_norm_tolerance
    if isinstance(tolerance, bool):
        raise ValueError("collection_norm_tolerance must be a finite number.")
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "collection_norm_tolerance must be a finite number."
        ) from exc
    if not math.isfinite(tolerance_value) or tolerance_value < 0:
        raise ValueError("collection_norm_tolerance must be finite and non-negative.")
    return required, tolerance_value


def _check_snapshot_embedding_contract(
    snapshot: FullCollectionSnapshot, config: AmbiCalConfig
) -> None:
    required, tolerance = _normalization_contract(config)
    if not required:
        return
    for chunk_id, embedding in zip(snapshot.ids, snapshot.embeddings):
        norm = float(torch.linalg.vector_norm(torch.tensor(embedding)).item())
        if abs(norm - 1.0) > tolerance:
            raise ValueError(
                f"Collection chunk {chunk_id!r} has embedding norm {norm:.6f}; "
                f"expected unit normalization within {tolerance:.6f}."
            )


def read_full_collection(
    collection: Any,
    config: AmbiCalConfig,
    *,
    batch_size: int = 256,
) -> FullCollectionSnapshot:
    """Read every collection row without mutation and enforce collection contracts."""

    if collection is None:
        raise ValueError("collection cannot be None.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    _normalization_contract(config)
    try:
        expected_count = int(collection.count())
    except Exception as exc:
        raise ValueError("Could not read the collection count.") from exc
    if expected_count < 1:
        raise ValueError("AmbiCal calibration requires a non-empty collection.")

    ids: List[Any] = []
    documents: List[Any] = []
    embeddings: List[Any] = []
    metadatas: List[Any] = []
    for offset in range(0, expected_count, batch_size):
        requested = min(batch_size, expected_count - offset)
        batch = collection.get(
            limit=requested,
            offset=offset,
            include=["documents", "embeddings", "metadatas"],
        )
        if not isinstance(batch, Mapping):
            raise ValueError("Collection get() must return an object.")
        batch_ids = _sequence(batch.get("ids"), "collection batch ids")
        if len(batch_ids) != requested:
            raise RuntimeError(
                "Collection changed or returned an incomplete calibration batch: "
                f"offset {offset}, expected {requested}, received {len(batch_ids)}."
            )
        batch_documents = _sequence(
            batch.get("documents"), "collection batch documents"
        )
        batch_embeddings = _sequence(
            batch.get("embeddings"), "collection batch embeddings"
        )
        raw_metadatas = batch.get("metadatas")
        batch_metadatas = (
            [{} for _ in batch_ids]
            if raw_metadatas is None
            else _sequence(raw_metadatas, "collection batch metadatas")
        )
        if not (
            len(batch_documents)
            == len(batch_embeddings)
            == len(batch_metadatas)
            == len(batch_ids)
        ):
            raise ValueError("Collection batch fields have different lengths.")
        ids.extend(batch_ids)
        documents.extend(batch_documents)
        embeddings.extend(batch_embeddings)
        metadatas.extend(batch_metadatas)

    try:
        final_count = int(collection.count())
    except Exception as exc:
        raise ValueError("Could not re-read the collection count.") from exc
    if final_count != expected_count or len(ids) != expected_count:
        raise RuntimeError(
            "Collection changed or was incompletely read during calibration: "
            f"initial={expected_count}, final={final_count}, rows={len(ids)}."
        )
    snapshot = FullCollectionSnapshot(
        ids=tuple(ids),
        documents=tuple(documents),
        embeddings=tuple(embeddings),
        metadatas=tuple(metadatas),
    )
    if snapshot.count != expected_count:
        raise RuntimeError("Full collection snapshot count does not match collection count.")
    _check_snapshot_embedding_contract(snapshot, config)
    return snapshot


@dataclass(frozen=True)
class _Question:
    text: str
    identifier: Optional[str]


def _normalize_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).strip().split())


def _questions(values: Sequence[Any]) -> Tuple[_Question, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("questions must be a sequence, not a string.")
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError("questions must be a sequence.") from exc
    if not raw_values:
        raise ValueError("At least one calibration question is required.")
    result: List[_Question] = []
    for index, value in enumerate(raw_values):
        identifier: Optional[str] = None
        if isinstance(value, Mapping):
            if "question" not in value:
                raise ValueError(f"Calibration question {index} has no 'question' field.")
            raw_text = value.get("question")
            raw_identifier = value.get("calibration_id")
            if raw_identifier is not None:
                identifier = str(raw_identifier)
        else:
            raw_text = value
        text = _normalize_text(raw_text)
        if not text:
            raise ValueError(f"Calibration question {index} is empty.")
        result.append(_Question(text=text, identifier=identifier))
    return tuple(result)


def _embed_query(embedding_retriever: Any, text: str, label: str) -> torch.Tensor:
    if embedding_retriever is None:
        raise ValueError("embedding_retriever cannot be None.")
    try:
        encoded = embedding_retriever.vectorize_text(text)
    except Exception as exc:
        raise RuntimeError(f"Embedding failed for {label}.") from exc
    try:
        tensor = torch.as_tensor(encoded, dtype=torch.float64, device="cpu")
    except Exception as exc:
        raise ValueError(f"Embedding retriever returned an invalid value for {label}.") from exc
    if tensor.ndim == 1:
        vector = tensor
    elif tensor.ndim == 2 and tensor.shape[0] == 1:
        vector = tensor[0]
    else:
        raise ValueError(
            f"Embedding for {label} must be shaped [d] or [1, d]."
        )
    return l2_normalize(vector, f"embedding for {label}")


def _json_trace_value(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_trace_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_trace_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, torch.Tensor):
        return _json_trace_value(value.detach().cpu().tolist(), path)
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _json_trace_value(to_list(), path)
    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        return _json_trace_value(scalar_item(), path)
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}.")


def _translation_text(result: Any) -> str:
    if isinstance(result, str):
        raw = result
    elif isinstance(result, Mapping):
        status = result.get("status")
        if status is not None and str(status).casefold() in {"failed", "error"}:
            raise RuntimeError(f"Translator reported status {status!r}.")
        raw = (
            result.get("translated_query")
            or result.get("translation")
            or result.get("text")
            or ""
        )
    else:
        raw = getattr(result, "translated_query", "")
    translated = _normalize_text(raw)
    if not translated:
        raise ValueError("Translator returned an empty query.")
    return translated


def _translation_diagnostics(result: Any) -> Dict[str, Any]:
    diagnostics_method = getattr(result, "diagnostics", None)
    if callable(diagnostics_method):
        raw = diagnostics_method()
    elif isinstance(result, Mapping) and isinstance(result.get("diagnostics"), Mapping):
        raw = result["diagnostics"]
    else:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("Translator diagnostics must be an object.")
    return dict(_json_trace_value(dict(raw), "translation_diagnostics"))


def _strict_translate(
    translator: Any,
    query: str,
    source_language: str,
    target_language: str,
) -> Tuple[str, Dict[str, Any]]:
    if source_language.casefold() == target_language.casefold():
        return query, {
            "status": "identity",
            "source_language": source_language,
            "target_language": target_language,
        }
    if translator is None:
        raise RuntimeError("A translator is required for cross-language calibration.")
    result = translator.translate(query, source_language, target_language)
    translated = _translation_text(result)
    return translated, {"status": "ok", **_translation_diagnostics(result)}


def _top_r_trace(
    snapshot: FullCollectionSnapshot,
    raw_scores: torch.Tensor,
    requested_top_r: int,
) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
    """Select Top-r deterministically and retain its audit evidence."""

    if raw_scores.ndim != 1 or raw_scores.numel() != snapshot.count:
        raise ValueError("Calibration score vector does not match the collection.")
    effective_top_r = min(requested_top_r, snapshot.count)
    ranked_indices = tuple(
        sorted(
            range(snapshot.count),
            key=lambda index: (
                -float(raw_scores[index].item()),
                snapshot.ids[index],
            ),
        )
    )
    selected = ranked_indices[:effective_top_r]
    trace = {
        "requested_top_r": requested_top_r,
        "effective_top_r": effective_top_r,
        "ranking_tie_break": "score_desc_then_chunk_id_asc",
        "top_chunk_ids": [snapshot.ids[index] for index in selected],
        "top_raw_scores": {
            snapshot.ids[index]: float(raw_scores[index].item())
            for index in selected
        },
        "raw_scores": {
            chunk_id: float(value)
            for chunk_id, value in zip(snapshot.ids, raw_scores.tolist())
        },
    }
    return selected, trace


def build_false_positive_priors(
    collection_snapshot: FullCollectionSnapshot,
    embedding_retriever: Any,
    questions: Sequence[Any],
    config: AmbiCalConfig,
    translator: Any,
    source_language: str,
    target_language: str,
) -> Tuple[FalsePositivePriors, Dict[str, Any]]:
    """Estimate per-branch Top-r hit priors from independent hard negatives.

    Each question contributes one Bernoulli observation per chunk and branch:
    the chunk either appears in that branch's Top-r or it does not.  A
    Beta(1, 1) prior gives ``(1 + hit_count) / (L_fp + 2)`` and assigns a
    previously unseen chunk ``1 / (L_fp + 2)``.  The existing
    ``candidate_top_n`` is the retrieval depth and therefore the method's
    ``r``; calibration does not introduce or guess another core parameter.
    """

    if not isinstance(collection_snapshot, FullCollectionSnapshot):
        raise TypeError("collection_snapshot must be a FullCollectionSnapshot.")
    if not isinstance(config, AmbiCalConfig):
        raise TypeError("config must be an AmbiCalConfig.")
    source_language = str(source_language).strip()
    target_language = str(target_language).strip()
    if not source_language or not target_language:
        raise ValueError("Source and target languages must be non-empty.")
    _check_snapshot_embedding_contract(collection_snapshot, config)
    calibrated_questions = _questions(questions)

    requested_top_r = config.candidate_top_n
    effective_top_r = min(requested_top_r, collection_snapshot.count)
    source_hit_counts = [0] * collection_snapshot.count
    target_hit_counts = [0] * collection_snapshot.count
    question_traces: List[Dict[str, Any]] = []
    for index, question in enumerate(calibrated_questions):
        try:
            translated_query, translation_trace = _strict_translate(
                translator,
                question.text,
                source_language,
                target_language,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Translation failed for false-positive calibration question {index + 1}."
            ) from exc

        source_embedding = _embed_query(
            embedding_retriever, question.text, f"source calibration question {index + 1}"
        )
        target_embedding = _embed_query(
            embedding_retriever,
            translated_query,
            f"target calibration question {index + 1}",
        )
        source_raw = cosine_scores(source_embedding, collection_snapshot.embeddings)
        target_raw = cosine_scores(target_embedding, collection_snapshot.embeddings)
        source_top, source_trace = _top_r_trace(
            collection_snapshot, source_raw, requested_top_r
        )
        target_top, target_trace = _top_r_trace(
            collection_snapshot, target_raw, requested_top_r
        )
        for chunk_index in source_top:
            source_hit_counts[chunk_index] += 1
        for chunk_index in target_top:
            target_hit_counts[chunk_index] += 1
        question_traces.append(
            {
                "index": index,
                "calibration_id": question.identifier,
                "question": question.text,
                "translated_query": translated_query,
                "translation": translation_trace,
                "source": source_trace,
                "target": target_trace,
            }
        )

    denominator = len(calibrated_questions) + 2
    source_priors = {
        chunk_id: (1.0 + hit_count) / denominator
        for chunk_id, hit_count in zip(
            collection_snapshot.ids, source_hit_counts
        )
    }
    target_priors = {
        chunk_id: (1.0 + hit_count) / denominator
        for chunk_id, hit_count in zip(
            collection_snapshot.ids, target_hit_counts
        )
    }
    unseen_source = 1.0 / denominator
    unseen_target = 1.0 / denominator
    priors = FalsePositivePriors(
        source=source_priors,
        target=target_priors,
        unseen_source=unseen_source,
        unseen_target=unseen_target,
    )
    trace = {
        "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "kind": "false_positive_priors",
        "question_count": len(calibrated_questions),
        "chunk_count": collection_snapshot.count,
        "chunk_ids": list(collection_snapshot.ids),
        "source_language": source_language,
        "target_language": target_language,
        "estimator": "beta_binomial_top_r_hit_rate",
        "beta_prior": {"alpha": 1.0, "beta": 1.0},
        "requested_top_r": requested_top_r,
        "effective_top_r": effective_top_r,
        "denominator": denominator,
        "ranking_tie_break": "score_desc_then_chunk_id_asc",
        "source_hit_counts": {
            chunk_id: hit_count
            for chunk_id, hit_count in zip(
                collection_snapshot.ids, source_hit_counts
            )
        },
        "target_hit_counts": {
            chunk_id: hit_count
            for chunk_id, hit_count in zip(
                collection_snapshot.ids, target_hit_counts
            )
        },
        "questions": question_traces,
        "false_positive_priors": priors.to_dict(),
    }
    return priors, trace


class _StrictTranslationRecorder:
    """Record translation calls so online source-only fallback becomes fatal here."""

    def __init__(self, translator: Any) -> None:
        self.translator = translator
        self.calls: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []

    def translate(
        self, query: str, source_language: str, target_language: str
    ) -> Any:
        normalized_query = _normalize_text(query)
        try:
            if self.translator is None:
                raise RuntimeError(
                    "A translator is required for cross-language calibration."
                )
            result = self.translator.translate(
                normalized_query, source_language, target_language
            )
            translated = _translation_text(result)
            diagnostics = _translation_diagnostics(result)
        except Exception as exc:
            failure = {
                "query": normalized_query,
                "source_language": source_language,
                "target_language": target_language,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self.calls.append(failure)
            self.failures.append(failure)
            raise
        self.calls.append(
            {
                "query": normalized_query,
                "translated_query": translated,
                "source_language": source_language,
                "target_language": target_language,
                "status": "ok",
                "diagnostics": diagnostics,
            }
        )
        return result


def build_null_strengths(
    collection: Any,
    embedding_retriever: Any,
    questions: Sequence[Any],
    final_config: AmbiCalConfig,
    false_positive_priors: FalsePositivePriors,
    translator: Any,
    source_language: str,
    target_language: str,
) -> Tuple[Tuple[float, ...], Dict[str, Any]]:
    """Compute one pre-gate evidence strength for every independent Null query."""

    if not isinstance(final_config, AmbiCalConfig):
        raise TypeError("final_config must be an AmbiCalConfig.")
    if not isinstance(false_positive_priors, FalsePositivePriors):
        raise TypeError("false_positive_priors must be a FalsePositivePriors.")
    final_config.validate()
    # null_gate=false is a permitted final configuration for the w/o Null Gate
    # ablation.  The probe below already measures the shared pre-gate path, so
    # the stored Null strengths keep exactly the same semantics; the online
    # retriever applies Null gating only when the switch is enabled.
    source_language = str(source_language).strip()
    target_language = str(target_language).strip()
    if not source_language or not target_language:
        raise ValueError("Source and target languages must be non-empty.")
    calibrated_questions = _questions(questions)

    # Null strengths must be measured before the gate being calibrated.  Every
    # other switch and numerical value remains byte-for-byte represented by the
    # final config; the probe profile intentionally has no utility signature.
    extra_probe_off = {"null_gate": False}
    if final_config.method_version == "ambical-online-v4":
        # The v4 cross-encoder promotion never runs during Null calibration:
        # the Null distribution characterises pre-correction evidence strength
        # (identical to v1's effective behaviour where the gate never fired),
        # and this keeps the calibration builder free of reranker wiring.
        extra_probe_off["context_correction"] = False
    probe_switches = replace(final_config.switches, **extra_probe_off)
    probe_config = replace(final_config, switches=probe_switches)
    probe_config.validate()
    cross_language = source_language.casefold() != target_language.casefold()
    recorder: Optional[_StrictTranslationRecorder] = None
    probe_translator = translator
    if cross_language and probe_config.switches.translation_branch:
        recorder = _StrictTranslationRecorder(translator)
        probe_translator = recorder
    probe_profile = AmbiCalProfile(
        false_positive_priors=false_positive_priors,
        null_strengths=(),
        utility_signature=None,
    )
    retriever = AmbiCalRetriever(
        collection=collection,
        embedding_retriever=embedding_retriever,
        config=probe_config,
        translator=probe_translator,
        profile=probe_profile,
    )

    strengths: List[float] = []
    question_traces: List[Dict[str, Any]] = []
    for index, question in enumerate(calibrated_questions):
        call_start = len(recorder.calls) if recorder is not None else 0
        failure_start = len(recorder.failures) if recorder is not None else 0
        result = retriever.retrieve(
            question.text,
            source_language=source_language,
            target_language=target_language,
        )
        translation_calls = (
            [] if recorder is None else recorder.calls[call_start:]
        )
        translation_failures = (
            [] if recorder is None else recorder.failures[failure_start:]
        )
        if translation_failures:
            first = translation_failures[0]
            raise RuntimeError(
                "Translation failed during Null calibration question "
                f"{index + 1}: {first['error_type']}: {first['error']}"
            )
        if (
            cross_language
            and probe_config.switches.translation_branch
            and not result.translation_active
        ):
            raise RuntimeError(
                f"Null calibration question {index + 1} fell back to source-only."
            )
        adaptive = result.diagnostics.get("adaptive_k")
        if not isinstance(adaptive, Mapping) or "evidence_strength" not in adaptive:
            raise RuntimeError(
                f"Null calibration question {index + 1} has no evidence strength."
            )
        try:
            strength = float(adaptive["evidence_strength"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Null calibration question {index + 1} has invalid evidence strength."
            ) from exc
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError(
                f"Null calibration question {index + 1} evidence strength "
                "must be finite and in [0, 1]."
            )
        strengths.append(strength)
        question_traces.append(
            {
                "index": index,
                "calibration_id": question.identifier,
                "question": question.text,
                "translated_query": result.translated_query,
                "translation_calls": translation_calls,
                "evidence_strength": strength,
                "retrieval_result": result.to_dict(),
            }
        )

    values = tuple(strengths)
    if len(values) != len(calibrated_questions):
        raise RuntimeError("Null calibration did not produce one strength per question.")
    trace = {
        "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "kind": "null_strengths",
        "question_count": len(calibrated_questions),
        "source_language": source_language,
        "target_language": target_language,
        "final_utility_signature": final_config.utility_signature(),
        "probe_utility_signature": probe_config.utility_signature(),
        "probe_override": {"null_gate": False},
        "probe_profile_utility_signature": None,
        "null_strengths": list(values),
        "questions": question_traces,
    }
    return values, trace


__all__ = [
    "CALIBRATION_ALGORITHM_VERSION",
    "FullCollectionSnapshot",
    "build_false_positive_priors",
    "build_null_strengths",
    "read_full_collection",
]
