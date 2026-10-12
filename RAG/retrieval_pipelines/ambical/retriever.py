"""Complete online AmbiCal retrieval method.

This module intentionally does not build calibration sets, create experiment
configurations, call an answer generator, or write result files.  It consumes a
precomputed optional profile and returns ranked evidence plus complete numerical
diagnostics.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from llm_connectors.base_llm_connector import TokenUsageUnavailableError

from .config import AmbiCalConfig
from .models import (
    ChunkMetadata,
    ChunkRecord,
    AmbiCalProfile,
    AmbiCalResult,
    ensure_profile_ready,
)
from .numerics import (
    BranchShape,
    active_consistency,
    branch_shape,
    cosine_scores,
    cosine_unit_interval,
    dynamic_weights,
    l2_normalize,
    pairwise_cosine,
    ranking_overlap,
    robust_sigmoid_map,
    translation_consistency,
)
from .phrases import (
    CounterfactualPhrase,
    CounterfactualPhraseGenerator,
    select_compact_diverse_phrases,
    select_non_overlapping_phrases,
)
from .selection import (
    ContextGateResult,
    counterfactual_phrase_risk,
    context_gate,
    context_gate_v2,
    greedy_evidence_competition,
    merge_structured_evidence,
    positive_context_promotion,
    reranker_promotion,
    select_adaptive_k,
    selected_items,
    symmetric_correction_factors,
    topk_counterfactual_promotion,
)


class AmbiCalRetriever:
    """AmbiCal method layer over an existing Chroma collection."""

    def __init__(
        self,
        *,
        collection: Any,
        embedding_retriever: Any,
        config: AmbiCalConfig,
        translator: Any = None,
        profile: Optional[AmbiCalProfile] = None,
        phrase_generator: Optional[CounterfactualPhraseGenerator] = None,
        reranker: Any = None,
    ) -> None:
        config.validate()
        if collection is None:
            raise ValueError("AmbiCal requires a Chroma-compatible collection.")
        if embedding_retriever is None:
            raise ValueError("AmbiCal requires an embedding retriever.")
        if (
            config.method_version == "ambical-online-v4"
            and config.switches.context_correction
            and reranker is None
        ):
            raise ValueError(
                "ambical-online-v4 with context_correction enabled "
                "requires a cross-encoder reranker; refusing to run a silent "
                "no-op correction."
            )
        self.collection = collection
        self.embedding_retriever = embedding_retriever
        self.config = config
        self.translator = translator
        self.reranker = reranker
        self.profile = ensure_profile_ready(
            profile,
            require_false_positive=config.switches.false_positive_prior,
            require_null=config.switches.null_gate,
        )
        if (
            self.profile.utility_signature is not None
            and self.profile.utility_signature != config.utility_signature()
        ):
            raise ValueError(
                "The AmbiCal profile utility signature does not match the "
                "current method configuration."
            )
        self.phrase_generator = phrase_generator or CounterfactualPhraseGenerator(
            minimum_remaining_content_tokens=(
                config.minimum_counterfactual_content_tokens
            )
        )
        self._embedding_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(unicodedata.normalize("NFC", str(query)).strip().split())

    @classmethod
    def _json_safe(cls, value: Any, path: str = "value") -> Any:
        """Convert adapter diagnostics to strict, finite JSON-native values."""

        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{path} contains a non-finite number.")
            return value
        if isinstance(value, torch.Tensor):
            return cls._json_safe(value.detach().cpu().tolist(), path)
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_safe(item, f"{path}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                cls._json_safe(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        if isinstance(value, (set, frozenset)):
            return [
                cls._json_safe(item, f"{path}[{index}]")
                for index, item in enumerate(sorted(value, key=repr))
            ]
        to_list = getattr(value, "tolist", None)
        if callable(to_list):
            return cls._json_safe(to_list(), path)
        raise TypeError(f"{path} contains unsupported type {type(value).__name__}.")

    def _embed(self, text: str) -> torch.Tensor:
        normalized = self._normalize_query(text)
        if not normalized:
            raise ValueError("Cannot embed an empty query.")
        cached = self._embedding_cache.get(normalized)
        if cached is not None:
            return cached.clone()
        encoded = self.embedding_retriever.vectorize_text(normalized)
        try:
            if isinstance(encoded, torch.Tensor):
                tensor = encoded.detach().to(device="cpu", dtype=torch.float64)
            else:
                tensor = torch.as_tensor(
                    encoded, dtype=torch.float64, device="cpu"
                )
        except Exception as exc:  # pragma: no cover - defensive adapter check
            raise TypeError("Embedding retriever returned an invalid value.") from exc
        if tensor.ndim == 1:
            raw_embedding = tensor
        elif tensor.ndim == 2 and tensor.shape[0] == 1:
            raw_embedding = tensor[0]
        else:
            raise ValueError(
                "Embedding retriever must return one vector shaped [d] or [1, d]."
            )
        embedding = l2_normalize(raw_embedding, f"embedding({normalized!r})")
        self._embedding_cache[normalized] = embedding.clone()
        return embedding

    def _translate(
        self, query: str, source_language: str, target_language: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        if source_language.casefold() == target_language.casefold():
            # An identity translation would duplicate the same retrieval branch
            # and falsely inflate agreement/confidence.  Degrade to source-only.
            return None, {
                "status": "identity_source_only",
                "source_language": source_language,
                "target_language": target_language,
            }
        if self.translator is None:
            return None, {
                "status": "failed",
                "error_type": "translator_unavailable",
            }
        try:
            result = self.translator.translate(query, source_language, target_language)
            if isinstance(result, str):
                raw_translation = result
            elif isinstance(result, Mapping):
                raw_translation = (
                    result.get("translated_query")
                    or result.get("translation")
                    or result.get("text")
                    or ""
                )
            else:
                raw_translation = getattr(result, "translated_query", "")
            translated = self._normalize_query(raw_translation)
            if not translated:
                raise ValueError("Translator returned an empty query.")
            diagnostics_method = getattr(result, "diagnostics", None)
            if callable(diagnostics_method):
                details = diagnostics_method()
            elif isinstance(result, Mapping):
                raw_details = result.get("diagnostics")
                details = raw_details if isinstance(raw_details, Mapping) else {}
            else:
                details = {}
            try:
                safe_details = self._json_safe(dict(details), "translator_diagnostics")
            except (TypeError, ValueError) as diagnostics_error:
                safe_details = {
                    "diagnostics_status": "invalid_omitted",
                    "diagnostics_error_type": type(diagnostics_error).__name__,
                    "diagnostics_error": str(diagnostics_error),
                }
            return translated, {"status": "ok", **safe_details}
        except TokenUsageUnavailableError:
            # AmbiCal runs require exact provider accounting.  A silent
            # source-only fallback would preserve answers while invalidating the
            # token and mechanism audit, so this error must terminate the run.
            raise
        except Exception as exc:
            return None, {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def _query_ids(self, embedding: torch.Tensor, top_n: int) -> List[str]:
        result = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=int(top_n),
            include=[],
        )
        groups = result.get("ids") if isinstance(result, Mapping) else None
        if not groups or not groups[0]:
            return []
        seen = set()
        ordered: List[str] = []
        for raw_id in groups[0]:
            chunk_id = str(raw_id)
            if chunk_id not in seen:
                seen.add(chunk_id)
                ordered.append(chunk_id)
        return ordered

    def _coerce_embedding(self, value: Any, chunk_id: str) -> Tuple[float, ...]:
        tensor = torch.as_tensor(value, dtype=torch.float64).flatten()
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError(f"Candidate {chunk_id!r} has a missing or non-finite embedding.")
        norm = float(torch.linalg.vector_norm(tensor).item())
        if norm <= 0:
            raise ValueError(f"Candidate {chunk_id!r} has a zero embedding.")
        if (
            self.config.require_normalized_collection_embeddings
            and abs(norm - 1.0) > self.config.collection_norm_tolerance
        ):
            raise ValueError(
                f"Candidate {chunk_id!r} has embedding norm {norm:.6f}; "
                "AmbiCal expects the existing Chroma collection to store "
                "unit-normalized embeddings. Rebuild the collection or explicitly "
                "disable this contract only after verifying the metric semantics."
            )
        return tuple(float(item) for item in tensor.tolist())

    def _load_candidates(self, candidate_ids: Sequence[str]) -> List[ChunkRecord]:
        result = self.collection.get(
            ids=list(candidate_ids),
            include=["documents", "embeddings", "metadatas"],
        )
        returned_ids = [str(value) for value in (result.get("ids") or [])]
        documents = result.get("documents")
        embeddings = result.get("embeddings")
        metadatas = result.get("metadatas")
        if documents is None or embeddings is None:
            raise ValueError("ChromaDB omitted candidate documents or embeddings.")
        if len(returned_ids) != len(documents) or len(returned_ids) != len(embeddings):
            raise ValueError("ChromaDB candidate fields have inconsistent lengths.")
        if metadatas is None:
            metadatas = [{} for _ in returned_ids]
        if len(metadatas) != len(returned_ids):
            raise ValueError("ChromaDB metadata length does not match candidate IDs.")

        rows: Dict[str, Tuple[str, Tuple[float, ...], ChunkMetadata]] = {}
        for chunk_id, document, embedding, metadata in zip(
            returned_ids, documents, embeddings, metadatas
        ):
            if document is None:
                raise ValueError(f"Candidate {chunk_id!r} has no text.")
            rows[chunk_id] = (
                str(document),
                self._coerce_embedding(embedding, chunk_id),
                ChunkMetadata.from_mapping(metadata),
            )
        missing = [chunk_id for chunk_id in candidate_ids if chunk_id not in rows]
        if missing:
            raise ValueError(
                "ChromaDB omitted requested candidate IDs: " + ", ".join(missing)
            )
        return [
            ChunkRecord(
                chunk_id=chunk_id,
                text=rows[chunk_id][0],
                embedding=rows[chunk_id][1],
                metadata=rows[chunk_id][2],
                union_rank=index,
            )
            for index, chunk_id in enumerate(candidate_ids)
        ]

    def _normalize_scores(self, scores: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.config.switches.robust_normalization:
            mapped, stats = robust_sigmoid_map(
                scores,
                epsilon=self.config.epsilon,
                z_max=self.config.z_max,
                iqr_normal_consistency=self.config.iqr_normal_consistency,
            )
            return mapped, {"mode": "robust_mad_sigmoid", **stats.diagnostics()}
        mapped = cosine_unit_interval(scores)
        return mapped, {
            "mode": "fixed_cosine_map",
            "median": None,
            "mad": None,
            "scale": None,
            "scale_source": None,
            "z_values": None,
        }

    def _shape(self, scores: torch.Tensor) -> BranchShape:
        return branch_shape(
            scores,
            temperature=self.config.branch_temperature,
            epsilon=self.config.epsilon,
            use_confidence=self.config.switches.branch_confidence,
        )

    def _branch_weights(
        self,
        source_shape: BranchShape,
        target_shape: Optional[BranchShape],
        rho: Optional[float],
    ) -> Tuple[float, float]:
        return dynamic_weights(
            source_shape.confidence,
            None if target_shape is None else target_shape.confidence,
            rho,
            use_dynamic_weights=self.config.switches.dynamic_branch_weights,
            use_translation_consistency=(
                self.config.switches.translation_consistency
            ),
            epsilon=self.config.epsilon,
        )

    def _false_positive_vectors(
        self, candidate_ids: Sequence[str], target_active: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        count = len(candidate_ids)
        if not self.config.switches.false_positive_prior:
            source = torch.zeros(count, dtype=torch.float64)
            target = torch.zeros(count, dtype=torch.float64) if target_active else None
            return source, target
        priors = self.profile.false_positive_priors
        if priors is None:  # guarded in __init__, retained for type safety
            raise ValueError("False-positive priors are unavailable.")
        source = torch.tensor(
            [priors.source_for(chunk_id) for chunk_id in candidate_ids],
            dtype=torch.float64,
        )
        target = None
        if target_active:
            target = torch.tensor(
                [priors.target_for(chunk_id) for chunk_id in candidate_ids],
                dtype=torch.float64,
            )
        return source, target

    @staticmethod
    def _query_impact(original: torch.Tensor, counterfactual: torch.Tensor) -> float:
        cosine = float(torch.dot(original, counterfactual).clamp(-1.0, 1.0).item())
        return min(1.0, max(0.0, 1.0 - cosine))

    def _counterfactual_risks(
        self,
        *,
        source_query: str,
        source_language: str,
        target_language: str,
        source_query_embedding: torch.Tensor,
        target_query_embedding: Optional[torch.Tensor],
        chunks: Sequence[ChunkRecord],
        pairwise_kernel: torch.Tensor,
        source_raw_scores: torch.Tensor,
        target_raw_scores: Optional[torch.Tensor],
        source_normalized_scores: torch.Tensor,
        target_normalized_scores: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, Any],
    ]:
        count = len(chunks)
        zero_source = torch.zeros(count, dtype=torch.float64)
        zero_target = (
            torch.zeros(count, dtype=torch.float64)
            if target_query_embedding is not None
            else None
        )
        generated = self.phrase_generator.generate(source_query)
        if not generated:
            return zero_source, zero_target, zero_source.clone(), (
                zero_target.clone() if zero_target is not None else None
            ), {
                "generated_count": 0,
                "selected_count": 0,
                "phrases": [],
                "reason": "no_legal_phrase",
            }

        all_phrase_data: List[Dict[str, Any]] = []
        scored_phrases: List[Tuple[CounterfactualPhrase, float]] = []
        for phrase in generated:
            source_cf_embedding = self._embed(phrase.counterfactual_query)
            source_impact = self._query_impact(
                source_query_embedding, source_cf_embedding
            )
            # Phrase selection is deliberately source-anchored.  Translating
            # every generated deletion before applying the configured cap both
            # violates that contract and turns max_counterfactual_phrases into
            # a reporting-only limit rather than a computational limit.
            scored_phrases.append((phrase, source_impact))
            all_phrase_data.append(
                {
                    "phrase": phrase,
                    "source_cf_embedding": source_cf_embedding,
                    "source_impact": source_impact,
                    "target_cf_query": None,
                    "target_cf_embedding": None,
                    "target_impact": None,
                    "target_translation": {"status": "not_selected"},
                    "ranking_impact": source_impact,
                }
            )

        selected = select_non_overlapping_phrases(
            scored_phrases, self.config.max_counterfactual_phrases
        )
        selected_keys = {
            (item.start, item.end, item.counterfactual_query) for item, _ in selected
        }
        row_by_key = {
            (
                row["phrase"].start,
                row["phrase"].end,
                row["phrase"].counterfactual_query,
            ): row
            for row in all_phrase_data
        }
        document_embeddings = [chunk.embedding for chunk in chunks]
        maximum_source_risk = zero_source.clone()
        maximum_target_risk = zero_target.clone() if zero_target is not None else None
        maximum_source_signed = zero_source.clone()
        maximum_target_signed = (
            zero_target.clone() if zero_target is not None else None
        )
        phrase_diagnostics: List[Dict[str, Any]] = []

        for phrase, ranking_impact in selected:
            key = (phrase.start, phrase.end, phrase.counterfactual_query)
            row = row_by_key[key]
            if target_query_embedding is not None:
                (
                    row["target_cf_query"],
                    row["target_translation"],
                ) = self._translate(
                    phrase.counterfactual_query,
                    source_language,
                    target_language,
                )
                if row["target_cf_query"]:
                    row["target_cf_embedding"] = self._embed(
                        row["target_cf_query"]
                    )
                    row["target_impact"] = self._query_impact(
                        target_query_embedding,
                        row["target_cf_embedding"],
                    )
            source_h = cosine_scores(row["source_cf_embedding"], document_embeddings)
            source_risk, source_diagnostics = counterfactual_phrase_risk(
                source_normalized_scores,
                source_raw_scores,
                source_h,
                pairwise_kernel,
                query_impact=row["source_impact"],
                epsilon=self.config.epsilon,
                z_max=self.config.z_max,
                iqr_normal_consistency=self.config.iqr_normal_consistency,
            )
            maximum_source_risk = torch.maximum(maximum_source_risk, source_risk)
            source_signed = torch.tensor(
                source_diagnostics["signed_dependency"], dtype=torch.float64
            )
            maximum_source_signed = torch.where(
                source_signed.abs() >= maximum_source_signed.abs(),
                source_signed,
                maximum_source_signed,
            )

            target_diagnostics: Optional[Dict[str, Any]] = None
            if (
                maximum_target_risk is not None
                and row["target_cf_embedding"] is not None
                and target_normalized_scores is not None
                and target_raw_scores is not None
            ):
                target_h = cosine_scores(
                    row["target_cf_embedding"], document_embeddings
                )
                target_risk, target_diagnostics = counterfactual_phrase_risk(
                    target_normalized_scores,
                    target_raw_scores,
                    target_h,
                    pairwise_kernel,
                    query_impact=float(row["target_impact"]),
                    epsilon=self.config.epsilon,
                    z_max=self.config.z_max,
                    iqr_normal_consistency=self.config.iqr_normal_consistency,
                )
                maximum_target_risk = torch.maximum(
                    maximum_target_risk, target_risk
                )
                if maximum_target_signed is not None:
                    target_signed = torch.tensor(
                        target_diagnostics["signed_dependency"], dtype=torch.float64
                    )
                    maximum_target_signed = torch.where(
                        target_signed.abs() >= maximum_target_signed.abs(),
                        target_signed,
                        maximum_target_signed,
                    )

            phrase_diagnostics.append(
                {
                    "text": phrase.text,
                    "span": [phrase.start, phrase.end],
                    "token_span": [phrase.token_start, phrase.token_end],
                    "phrase_source": phrase.source,
                    "counterfactual_source_query": phrase.counterfactual_query,
                    "counterfactual_target_query": row["target_cf_query"],
                    "ranking_impact": ranking_impact,
                    "source": {
                        "query_impact": row["source_impact"],
                        **source_diagnostics,
                    },
                    "target": target_diagnostics,
                    "target_translation": row["target_translation"],
                }
            )

        return (
            maximum_source_risk,
            maximum_target_risk,
            maximum_source_signed,
            maximum_target_signed,
            {
                "generated_count": len(generated),
                "selected_count": len(selected),
                "selection_rule": "source_impact_then_nonoverlap_then_retranslate",
                "generated_phrases": [
                    {
                        "text": row["phrase"].text,
                        "span": [row["phrase"].start, row["phrase"].end],
                        "counterfactual_query": row["phrase"].counterfactual_query,
                        "ranking_impact": row["ranking_impact"],
                        "selected": (
                            row["phrase"].start,
                            row["phrase"].end,
                            row["phrase"].counterfactual_query,
                        )
                        in selected_keys,
                    }
                    for row in all_phrase_data
                ],
                "phrases": phrase_diagnostics,
                "source_max_risk": [
                    float(value) for value in maximum_source_risk.tolist()
                ],
                "target_max_risk": (
                    None
                    if maximum_target_risk is None
                    else [float(value) for value in maximum_target_risk.tolist()]
                ),
                "source_max_signed_dependency": [
                    float(value) for value in maximum_source_signed.tolist()
                ],
                "target_max_signed_dependency": (
                    None
                    if maximum_target_signed is None
                    else [
                        float(value)
                        for value in maximum_target_signed.tolist()
                    ]
                ),
            },
        )

    def _reranker_relevance(
        self,
        source_query: str,
        chunks: Sequence[ChunkRecord],
        candidate_indices: Sequence[int],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Cross-encoder relevance for only the preregistered candidate window."""

        indices = [int(index) for index in candidate_indices]
        if len(set(indices)) != len(indices):
            raise ValueError("The reranker candidate window contains duplicates.")
        if any(index < 0 or index >= len(chunks) for index in indices):
            raise ValueError("The reranker candidate window contains an invalid index.")
        pairs = [(source_query, chunks[index].text) for index in indices]
        scores = self.reranker(pairs)
        if len(scores) != len(pairs):
            raise ValueError(
                "The reranker must return exactly one score per (query, chunk) pair."
            )
        relevance = torch.full((len(chunks),), -1.0, dtype=torch.float64)
        for index, score in zip(indices, scores):
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("The reranker returned a non-finite score.")
            if not 0.0 <= value <= 1.0:
                raise ValueError("The reranker must return relevance in [0, 1].")
            relevance[index] = value
        diagnostics = {
            "scored_count": len(pairs),
            "candidate_indices": indices,
            "min": None if not scores else min(float(value) for value in scores),
            "max": None if not scores else max(float(value) for value in scores),
        }
        return relevance, diagnostics

    def _positive_context_views(
        self,
        *,
        source_query: str,
        source_language: str,
        target_language: str,
        source_query_embedding: torch.Tensor,
        target_query_embedding: Optional[torch.Tensor],
        chunks: Sequence[ChunkRecord],
        alpha_source: float,
        alpha_target: float,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Build two independent, dual-route remaining-context score views."""

        generated = self.phrase_generator.generate(source_query)
        if not generated:
            return [], {
                "generated_count": 0,
                "selected_count": 0,
                "valid_phrase_count": 0,
                "selection_rule": "source_impact_then_nonoverlap_then_retranslate",
                "generated_phrases": [],
                "phrases": [],
                "reason": "no_legal_phrase",
            }

        rows: List[Dict[str, Any]] = []
        scored: List[Tuple[CounterfactualPhrase, float]] = []
        for phrase in generated:
            source_cf_embedding = self._embed(phrase.counterfactual_query)
            source_impact = self._query_impact(
                source_query_embedding, source_cf_embedding
            )
            scored.append((phrase, source_impact))
            rows.append(
                {
                    "phrase": phrase,
                    "source_cf_embedding": source_cf_embedding,
                    "source_impact": source_impact,
                    "ranking_impact": source_impact,
                }
            )

        if self.config.method_version in {
            "ambical-online-v3.1",
            "ambical-online-v5",
        }:
            selected = select_compact_diverse_phrases(
                scored, self.config.max_counterfactual_phrases
            )
            selection_rule = (
                "compact_max_2_content_tokens_then_distinct_regions_then_impact"
            )
        else:
            selected = select_non_overlapping_phrases(
                scored, self.config.max_counterfactual_phrases
            )
            selection_rule = "source_impact_then_nonoverlap_then_retranslate"
        selected_keys = {
            (item.start, item.end, item.counterfactual_query) for item, _ in selected
        }
        row_by_key = {
            (
                row["phrase"].start,
                row["phrase"].end,
                row["phrase"].counterfactual_query,
            ): row
            for row in rows
        }
        document_embeddings = [chunk.embedding for chunk in chunks]
        valid_views: List[torch.Tensor] = []
        phrase_diagnostics: List[Dict[str, Any]] = []

        for phrase, ranking_impact in selected:
            key = (phrase.start, phrase.end, phrase.counterfactual_query)
            row = row_by_key[key]
            source_raw = cosine_scores(
                row["source_cf_embedding"], document_embeddings
            )
            source_scores, source_normalization = self._normalize_scores(source_raw)

            target_cf_query: Optional[str] = None
            target_translation: Dict[str, Any] = {"status": "not_required"}
            target_scores: Optional[torch.Tensor] = None
            target_normalization: Optional[Dict[str, Any]] = None
            if target_query_embedding is not None:
                target_cf_query, target_translation = self._translate(
                    phrase.counterfactual_query,
                    source_language,
                    target_language,
                )
                if target_cf_query:
                    target_cf_embedding = self._embed(target_cf_query)
                    target_raw = cosine_scores(
                        target_cf_embedding, document_embeddings
                    )
                    target_scores, target_normalization = self._normalize_scores(
                        target_raw
                    )

            # A cross-language v3 view is valid only when both of the original
            # retrieval routes are available.  Falling back to one route here
            # would make phrase agreement incomparable across questions.
            valid = target_query_embedding is None or target_scores is not None
            fused_scores: Optional[torch.Tensor] = None
            if valid:
                fused_scores = float(alpha_source) * source_scores
                if target_scores is not None:
                    fused_scores = fused_scores + float(alpha_target) * target_scores
                fused_scores = torch.clamp(fused_scores, 0.0, 1.0)
                valid_views.append(fused_scores)

            phrase_diagnostics.append(
                {
                    "text": phrase.text,
                    "span": [phrase.start, phrase.end],
                    "token_span": [phrase.token_start, phrase.token_end],
                    "phrase_source": phrase.source,
                    "counterfactual_source_query": phrase.counterfactual_query,
                    "counterfactual_target_query": target_cf_query,
                    "ranking_impact": float(ranking_impact),
                    "valid": valid,
                    "source_context_scores": self._tensor_list(source_scores),
                    "target_context_scores": self._tensor_list(target_scores),
                    "fused_context_scores": self._tensor_list(fused_scores),
                    "source_normalization": source_normalization,
                    "target_normalization": target_normalization,
                    "target_translation": target_translation,
                }
            )

        return valid_views, {
            "generated_count": len(generated),
            "selected_count": len(selected),
            "valid_phrase_count": len(valid_views),
            "selection_rule": selection_rule,
            "view_processing": "independent_from_original_query_then_agree",
            "generated_phrases": [
                {
                    "text": row["phrase"].text,
                    "span": [row["phrase"].start, row["phrase"].end],
                    "counterfactual_query": row["phrase"].counterfactual_query,
                    "ranking_impact": float(row["ranking_impact"]),
                    "selected": (
                        row["phrase"].start,
                        row["phrase"].end,
                        row["phrase"].counterfactual_query,
                    )
                    in selected_keys,
                }
                for row in rows
            ],
            "phrases": phrase_diagnostics,
        }

    @staticmethod
    def _tensor_list(values: Optional[torch.Tensor]) -> Optional[List[float]]:
        if values is None:
            return None
        return [float(value) for value in values.tolist()]

    def retrieve(
        self,
        query: str,
        *,
        source_language: str,
        target_language: str,
    ) -> AmbiCalResult:
        source_query = self._normalize_query(query)
        if not source_query:
            raise ValueError("Source query cannot be empty.")
        source_language = str(source_language).strip()
        target_language = str(target_language).strip()
        if not source_language or not target_language:
            raise ValueError("Source and target languages must be non-empty.")

        source_embedding = self._embed(source_query)
        translated_query: Optional[str] = None
        translation_diagnostics: Dict[str, Any] = {"status": "disabled"}
        target_embedding: Optional[torch.Tensor] = None
        if self.config.switches.translation_branch:
            translated_query, translation_diagnostics = self._translate(
                source_query, source_language, target_language
            )
            if translated_query:
                target_embedding = self._embed(translated_query)

        collection_count = int(self.collection.count())
        if collection_count < 1:
            raise ValueError("The ChromaDB collection is empty.")
        branch_top_n = min(self.config.candidate_top_n, collection_count)
        source_ids = self._query_ids(source_embedding, branch_top_n)
        target_ids: List[str] = []
        if target_embedding is not None:
            target_ids = self._query_ids(target_embedding, branch_top_n)
        candidate_ids = list(dict.fromkeys(source_ids + target_ids))
        if not candidate_ids:
            raise ValueError("AmbiCal retrieval returned no candidates.")

        chunks = self._load_candidates(candidate_ids)
        document_embeddings = [chunk.embedding for chunk in chunks]
        source_raw = cosine_scores(source_embedding, document_embeddings)
        target_raw = (
            None
            if target_embedding is None
            else cosine_scores(target_embedding, document_embeddings)
        )
        source_normalized, source_normalization = self._normalize_scores(source_raw)
        target_normalized: Optional[torch.Tensor] = None
        target_normalization: Optional[Dict[str, Any]] = None
        if target_raw is not None:
            target_normalized, target_normalization = self._normalize_scores(target_raw)

        source_shape_initial = self._shape(source_normalized)
        target_shape_initial = (
            None if target_normalized is None else self._shape(target_normalized)
        )
        rho = (
            None
            if target_embedding is None
            else translation_consistency(source_embedding, target_embedding)
        )
        omega = (
            None
            if target_embedding is None
            else ranking_overlap(source_ids, target_ids)
        )
        consistency_signals: List[Optional[float]] = []
        if target_embedding is not None and self.config.switches.translation_consistency:
            consistency_signals.append(rho)
        if target_embedding is not None and self.config.switches.ranking_overlap:
            consistency_signals.append(omega)
        chi, chi_signal_count = active_consistency(consistency_signals)
        alpha_source_initial, alpha_target_initial = self._branch_weights(
            source_shape_initial, target_shape_initial, rho
        )

        source_priors, target_priors = self._false_positive_vectors(
            candidate_ids, target_embedding is not None
        )
        lambda_fp = (
            float(self.config.lambda_false_positive or 0.0)
            if self.config.switches.false_positive_prior
            else 0.0
        )
        source_support = source_normalized - lambda_fp * source_priors
        target_support = (
            None
            if target_normalized is None or target_priors is None
            else target_normalized - lambda_fp * target_priors
        )
        initial_fused = alpha_source_initial * source_support
        if target_support is not None:
            initial_fused = initial_fused + alpha_target_initial * target_support

        kernel = pairwise_cosine(document_embeddings)
        if self.config.switches.context_correction:
            if self.config.method_version in {
                "ambical-online-v2",
                "ambical-online-v3",
                "ambical-online-v3.1",
                "ambical-online-v4",
                "ambical-online-v5",
            }:
                gate = context_gate_v2(
                    initial_fused,
                    source_support,
                    target_support,
                    kernel,
                    chi=chi,
                    top_m=self.config.gate_top_m,
                    temperature=self.config.gate_temperature,
                    threshold=float(self.config.context_gate_threshold or 0.0),
                    epsilon=self.config.epsilon,
                    z_max=self.config.z_max,
                    iqr_normal_consistency=self.config.iqr_normal_consistency,
                    bias=float(self.config.gate_v2_bias or 0.0),
                    weight_strength=float(
                        self.config.gate_v2_weight_strength or 0.0
                    ),
                    weight_competition=float(
                        self.config.gate_v2_weight_competition or 0.0
                    ),
                    weight_distance=float(
                        self.config.gate_v2_weight_distance or 0.0
                    ),
                    weight_path_conflict=float(
                        self.config.gate_v2_weight_path_conflict or 0.0
                    ),
                    weight_bimodality=float(
                        self.config.gate_v2_weight_bimodality or 0.0
                    ),
                    weight_branch_disagreement=float(
                        self.config.gate_v2_weight_branch_disagreement or 0.0
                    ),
                )
            else:
                gate = context_gate(
                    initial_fused,
                    source_support,
                    target_support,
                    kernel,
                    chi=chi,
                    top_m=self.config.gate_top_m,
                    temperature=self.config.gate_temperature,
                    threshold=float(self.config.context_gate_threshold or 0.0),
                    epsilon=self.config.epsilon,
                    z_max=self.config.z_max,
                    iqr_normal_consistency=self.config.iqr_normal_consistency,
                )
            execute_context = (
                gate.triggered if self.config.switches.context_gate else True
            )
        else:
            gate = ContextGateResult(
                observed_indices=(),
                competition=0.0,
                weighted_distance=0.0,
                context_split=0.0,
                route_total_variation=0.0,
                path_conflict=0.0,
                evidence_strength=0.0,
                risk=0.0,
                triggered=False,
            )
            execute_context = False

        source_context_risk = torch.zeros(len(chunks), dtype=torch.float64)
        target_context_risk = (
            torch.zeros(len(chunks), dtype=torch.float64)
            if target_embedding is not None
            else None
        )
        source_signed_dependency = torch.zeros(len(chunks), dtype=torch.float64)
        target_signed_dependency = (
            torch.zeros(len(chunks), dtype=torch.float64)
            if target_embedding is not None
            else None
        )
        context_diagnostics: Dict[str, Any] = {
            "executed": False,
            "reason": "component_disabled"
            if not self.config.switches.context_correction
            else "gate_not_triggered",
            "generated_count": 0,
            "selected_count": 0,
            "phrases": [],
        }
        positive_context_views: List[torch.Tensor] = []
        reranker_relevance: Optional[torch.Tensor] = None
        if execute_context:
            if self.config.method_version == "ambical-online-v4":
                reranker_window = list(gate.observed_indices)
                if not reranker_window:
                    reranker_window = [
                        int(index)
                        for index in torch.argsort(
                            initial_fused, descending=True, stable=True
                        ).tolist()
                    ]
                reranker_window = reranker_window[
                    : int(self.config.reranker_top_m or 1)
                ]
                reranker_relevance, relevance_diagnostics = (
                    self._reranker_relevance(
                        source_query, chunks, reranker_window
                    )
                )
                context_diagnostics = {
                    "generated_count": 0,
                    "selected_count": 0,
                    "phrases": [],
                    "reranker": relevance_diagnostics,
                }
            elif self.config.method_version in {
                "ambical-online-v3",
                "ambical-online-v3.1",
                "ambical-online-v5",
            }:
                positive_context_views, context_diagnostics = (
                    self._positive_context_views(
                        source_query=source_query,
                        source_language=source_language,
                        target_language=target_language,
                        source_query_embedding=source_embedding,
                        target_query_embedding=target_embedding,
                        chunks=chunks,
                        alpha_source=alpha_source_initial,
                        alpha_target=alpha_target_initial,
                    )
                )
            else:
                (
                    source_context_risk,
                    target_context_risk,
                    source_signed_dependency,
                    target_signed_dependency,
                    context_diagnostics,
                ) = self._counterfactual_risks(
                    source_query=source_query,
                    source_language=source_language,
                    target_language=target_language,
                    source_query_embedding=source_embedding,
                    target_query_embedding=target_embedding,
                    chunks=chunks,
                    pairwise_kernel=kernel,
                    source_raw_scores=source_raw,
                    target_raw_scores=target_raw,
                    source_normalized_scores=source_normalized,
                    target_normalized_scores=target_normalized,
                )
            context_diagnostics = {"executed": True, **context_diagnostics}

        lambda_context = (
            float(self.config.lambda_context or 0.0)
            if self.config.switches.context_correction
            else 0.0
        )
        if (
            self.config.method_version
            in {
                "ambical-online-v3",
                "ambical-online-v3.1",
                "ambical-online-v4",
                "ambical-online-v5",
            }
            and self.config.switches.context_correction
        ):
            # v3/v4/v5 leave both branches untouched and apply only positive
            # bonuses after their original dynamic fusion.
            source_corrected = source_normalized
            target_corrected = target_normalized
            if self.config.method_version == "ambical-online-v4":
                correction_mode = "reranker_promote_v4"
            elif self.config.method_version == "ambical-online-v5":
                correction_mode = "topk_counterfactual_positive_v5"
            else:
                correction_mode = "positive_context_v3"
            context_diagnostics["correction_mode"] = correction_mode
        elif (
            self.config.method_version == "ambical-online-v2"
            and self.config.switches.context_correction
        ):
            source_factors, source_factor_diagnostics = (
                symmetric_correction_factors(
                    source_signed_dependency, lambda_context=lambda_context
                )
            )
            source_corrected = source_normalized * source_factors
            target_corrected = None
            target_factor_diagnostics = None
            if target_normalized is not None and target_signed_dependency is not None:
                target_factors, target_factor_diagnostics = (
                    symmetric_correction_factors(
                        target_signed_dependency, lambda_context=lambda_context
                    )
                )
                target_corrected = target_normalized * target_factors
            context_diagnostics["correction_mode"] = "symmetric_v2"
            context_diagnostics["source_correction_factors"] = (
                source_factor_diagnostics
            )
            context_diagnostics["target_correction_factors"] = (
                target_factor_diagnostics
            )
        else:
            source_corrected = source_normalized * (
                1.0 - lambda_context * source_context_risk
            )
            target_corrected = None
            if target_normalized is not None and target_context_risk is not None:
                target_corrected = target_normalized * (
                    1.0 - lambda_context * target_context_risk
                )
            if self.config.switches.context_correction:
                context_diagnostics["correction_mode"] = "penalize_v1"
        source_shape_final = self._shape(source_corrected)
        target_shape_final = (
            None if target_corrected is None else self._shape(target_corrected)
        )
        alpha_source_final, alpha_target_final = self._branch_weights(
            source_shape_final, target_shape_final, rho
        )

        if (
            self.config.method_version
            in {
                "ambical-online-v3",
                "ambical-online-v3.1",
                "ambical-online-v4",
                "ambical-online-v5",
            }
            and self.config.switches.context_correction
        ):
            fused_relevance = alpha_source_final * source_corrected
            fused_prior = alpha_source_final * source_priors
            if target_corrected is not None and target_priors is not None:
                fused_relevance = fused_relevance + (
                    alpha_target_final * target_corrected
                )
                fused_prior = fused_prior + alpha_target_final * target_priors

            def _v4_promotion(relevance: Optional[torch.Tensor]):
                safe_relevance = (
                    relevance
                    if relevance is not None
                    else torch.full_like(fused_relevance, -1.0)
                )
                observed = gate.observed_indices
                if not observed:
                    # Gate switch disabled (forced diagnostic mode): derive the
                    # pre-bonus fused order directly so v4 keeps a valid window.
                    observed = tuple(
                        int(index)
                        for index in torch.argsort(
                            initial_fused, descending=True, stable=True
                        ).tolist()
                    )
                return reranker_promotion(
                    fused_relevance,
                    safe_relevance,
                    observed,
                    lambda_positive=float(self.config.lambda_positive or 0.0),
                    threshold=float(self.config.reranker_score_threshold or 0.0),
                    top_m=int(self.config.reranker_top_m or 1),
                    epsilon=self.config.epsilon,
                )

            if execute_context:
                if self.config.method_version == "ambical-online-v4":
                    promoted, promotion_diagnostics = _v4_promotion(
                        reranker_relevance
                    )
                elif self.config.method_version == "ambical-online-v5":
                    promoted, promotion_diagnostics = topk_counterfactual_promotion(
                        fused_relevance,
                        positive_context_views,
                        lambda_positive=float(self.config.lambda_positive or 0.0),
                        promotion_top_k=int(
                            self.config.counterfactual_promotion_top_k or 1
                        ),
                        epsilon=self.config.epsilon,
                    )
                else:
                    promoted, promotion_diagnostics = positive_context_promotion(
                        fused_relevance,
                        positive_context_views,
                        lambda_positive=float(self.config.lambda_positive or 0.0),
                        epsilon=self.config.epsilon,
                    )
            else:
                if self.config.method_version == "ambical-online-v4":
                    promoted, promotion_diagnostics = _v4_promotion(None)
                    promotion_diagnostics["promotion_reason"] = (
                        context_diagnostics["reason"]
                    )
                elif self.config.method_version == "ambical-online-v5":
                    promoted = fused_relevance.clone()
                    original_top = int(torch.argmax(fused_relevance).item())
                    promotion_diagnostics = {
                        "promotion_executed": False,
                        "promotion_reason": context_diagnostics["reason"],
                        "valid_phrase_count": 0,
                        "promotion_top_k": int(
                            self.config.counterfactual_promotion_top_k or 1
                        ),
                        "per_view_topk_indices": [],
                        "per_view_alignment_scores": [],
                        "per_view_bonuses": [],
                        "promoted_candidate_indices": [],
                        "promotion_amounts": {},
                        "top1_before": original_top,
                        "top1_after": original_top,
                        "top1_changed": False,
                    }
                else:
                    promoted = fused_relevance.clone()
                    original_top = int(torch.argmax(fused_relevance).item())
                    promotion_diagnostics = {
                        "promotion_executed": False,
                        "promotion_reason": context_diagnostics["reason"],
                        "valid_phrase_count": 0,
                        "view_winners": [],
                        "phrase_agreement": False,
                        "promoted_candidate_index": None,
                        "promotion_amount": 0.0,
                        "candidate_margin": 0.0,
                        "top1_before": original_top,
                        "top1_after": original_top,
                        "counterfactual_context_scores": [],
                        "semantic_alignment_views": [],
                        "semantic_alignment_scores": None,
                    }
            for field in ("view_winners",):
                indices = promotion_diagnostics.get(field) or []
                promotion_diagnostics[field.replace("winners", "winner_ids")] = [
                    candidate_ids[index] for index in indices
                ]
            promoted_index = promotion_diagnostics.get("promoted_candidate_index")
            promotion_diagnostics["promoted_candidate_id"] = (
                None
                if promoted_index is None
                else candidate_ids[int(promoted_index)]
            )
            promoted_indices_v4 = promotion_diagnostics.get(
                "promoted_candidate_indices"
            )
            if promoted_indices_v4:
                promotion_diagnostics["promoted_candidate_ids"] = [
                    candidate_ids[int(index)] for index in promoted_indices_v4
                ]
            per_view_topk = promotion_diagnostics.get("per_view_topk_indices") or []
            if per_view_topk:
                promotion_diagnostics["per_view_topk_ids"] = [
                    [candidate_ids[int(index)] for index in indices]
                    for indices in per_view_topk
                ]
            for entry in promotion_diagnostics.get("considered") or []:
                entry["chunk_id"] = candidate_ids[int(entry["index"])]
            for field in ("top1_before", "top1_after"):
                index = promotion_diagnostics.get(field)
                promotion_diagnostics[field + "_id"] = (
                    None if index is None else candidate_ids[int(index)]
                )
            context_diagnostics.update(promotion_diagnostics)
            base_values = promoted - lambda_fp * fused_prior
        else:
            base_values = alpha_source_final * (
                source_corrected - lambda_fp * source_priors
            )
            if target_corrected is not None and target_priors is not None:
                base_values = base_values + alpha_target_final * (
                    target_corrected - lambda_fp * target_priors
                )

        greedy = greedy_evidence_competition(
            base_values,
            kernel,
            candidate_ids,
            [chunk.union_rank for chunk in chunks],
            max_k=self.config.max_k,
            redundancy_enabled=self.config.switches.soft_redundancy,
            lambda_redundancy=(
                float(self.config.lambda_redundancy or 0.0)
                if self.config.switches.soft_redundancy
                else 0.0
            ),
        )
        adaptive = select_adaptive_k(
            greedy.marginal_utilities,
            chi=chi,
            temperature=self.config.adaptive_temperature,
            eta=self.config.consistency_floor_eta,
            epsilon=self.config.epsilon,
            z_max=self.config.z_max,
            iqr_normal_consistency=self.config.iqr_normal_consistency,
            adaptive_enabled=self.config.switches.adaptive_k,
            entropy_enabled=self.config.switches.entropy_candidate,
            maximum_gap_enabled=self.config.switches.maximum_gap,
            consistency_entropy_enabled=self.config.switches.consistency_entropy,
            fixed_k=self.config.fixed_k,
            null_gate_enabled=self.config.switches.null_gate,
            null_strengths=self.profile.null_strengths,
            null_low_threshold=self.config.null_low_threshold,
            null_high_threshold=self.config.null_high_threshold,
        )
        evidence = selected_items(
            chunks,
            [float(value) for value in base_values.tolist()],
            greedy,
            adaptive.selected_k,
        )
        if self.config.switches.merge_structured_evidence:
            evidence, merge_diagnostics = merge_structured_evidence(evidence)
        else:
            merge_diagnostics = {
                "applied": False,
                "reason": "switch_disabled",
                "merged_count": len(evidence),
            }

        selected_order = {
            candidate_index: rank
            for rank, candidate_index in enumerate(greedy.indices, start=1)
        }
        marginal_by_index = {
            candidate_index: greedy.marginal_utilities[rank - 1]
            for rank, candidate_index in enumerate(greedy.indices, start=1)
        }
        candidate_diagnostics: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            candidate_diagnostics.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "union_rank": chunk.union_rank,
                    "metadata": asdict(chunk.metadata),
                    "source_raw_score": float(source_raw[index].item()),
                    "target_raw_score": (
                        None
                        if target_raw is None
                        else float(target_raw[index].item())
                    ),
                    "source_normalized_score": float(
                        source_normalized[index].item()
                    ),
                    "target_normalized_score": (
                        None
                        if target_normalized is None
                        else float(target_normalized[index].item())
                    ),
                    "source_false_positive_prior": float(
                        source_priors[index].item()
                    ),
                    "target_false_positive_prior": (
                        None
                        if target_priors is None
                        else float(target_priors[index].item())
                    ),
                    "source_branch_support": float(source_support[index].item()),
                    "target_branch_support": (
                        None
                        if target_support is None
                        else float(target_support[index].item())
                    ),
                    "initial_fused_score": float(initial_fused[index].item()),
                    "source_context_risk": float(
                        source_context_risk[index].item()
                    ),
                    "target_context_risk": (
                        None
                        if target_context_risk is None
                        else float(target_context_risk[index].item())
                    ),
                    "source_corrected_score": float(
                        source_corrected[index].item()
                    ),
                    "target_corrected_score": (
                        None
                        if target_corrected is None
                        else float(target_corrected[index].item())
                    ),
                    "base_value": float(base_values[index].item()),
                    "greedy_rank": selected_order.get(index),
                    "marginal_utility": marginal_by_index.get(index),
                }
            )

        diagnostics: Dict[str, Any] = {
            "method": self.config.method_version,
            "config": self.config.diagnostics(),
            "translation": translation_diagnostics,
            "source_retrieved_ids": source_ids,
            "target_retrieved_ids": target_ids,
            "candidate_ids": candidate_ids,
            "candidate_union_size": len(candidate_ids),
            "candidate_top_n": branch_top_n,
            "collection_embedding_contract": {
                "unit_normalized_required": (
                    self.config.require_normalized_collection_embeddings
                ),
                "norm_tolerance": self.config.collection_norm_tolerance,
                "exact_union_rescoring": "cosine",
            },
            "source_normalization": source_normalization,
            "target_normalization": target_normalization,
            "source_shape_initial": source_shape_initial.diagnostics(),
            "target_shape_initial": (
                None
                if target_shape_initial is None
                else target_shape_initial.diagnostics()
            ),
            "translation_consistency": rho,
            "ranking_overlap": omega,
            "combined_consistency": chi,
            "combined_consistency_signal_count": chi_signal_count,
            "combined_consistency_fallback": (
                "neutral_single_or_no_active_signal"
                if chi_signal_count == 0
                else None
            ),
            "alpha_source_initial": alpha_source_initial,
            "alpha_target_initial": alpha_target_initial,
            "context_gate": gate.diagnostics(),
            "context_correction": context_diagnostics,
            "source_shape_final": source_shape_final.diagnostics(),
            "target_shape_final": (
                None
                if target_shape_final is None
                else target_shape_final.diagnostics()
            ),
            "alpha_source_final": alpha_source_final,
            "alpha_target_final": alpha_target_final,
            "greedy_selection": greedy.diagnostics(),
            "adaptive_k": adaptive.diagnostics(),
            "merge": merge_diagnostics,
            "selected_k_before_merge": adaptive.selected_k,
            "evidence_count_after_merge": len(evidence),
            "candidates": candidate_diagnostics,
        }
        return AmbiCalResult(
            source_query=source_query,
            translated_query=translated_query,
            translation_active=target_embedding is not None,
            selected_k=adaptive.selected_k,
            evidence=tuple(evidence),
            diagnostics=diagnostics,
        )

    def vectorize_text(self, text_chunk: str) -> List[List[float]]:
        return [self._embed(text_chunk).tolist()]
