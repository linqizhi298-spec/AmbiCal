"""Typed, JSON-serialisable records used by AmbiCal."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ChunkMetadata:
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "ChunkMetadata":
        data = dict(value or {})

        def integer(*names: str) -> Optional[int]:
            for name in names:
                raw = data.get(name)
                if raw is not None and raw != "":
                    if isinstance(raw, bool):
                        return None
                    if isinstance(raw, int):
                        return raw
                    if isinstance(raw, float):
                        if not math.isfinite(raw) or not raw.is_integer():
                            return None
                        return int(raw)
                    if isinstance(raw, str):
                        stripped = raw.strip()
                        unsigned = stripped[1:] if stripped[:1] in {"+", "-"} else stripped
                        if unsigned.isdigit():
                            return int(stripped)
                    # Chroma metadata may contain display labels such as Roman
                    # pages or ranges.  They are valid metadata but are not
                    # trustworthy merge coordinates, so degrade this field to
                    # unavailable rather than aborting retrieval.
                    return None
            return None

        def text(*names: str) -> Optional[str]:
            for name in names:
                raw = data.get(name)
                if raw is not None and str(raw).strip():
                    return str(raw).strip()
            return None

        return cls(
            document_id=text("document_id", "source_document_id"),
            chunk_index=integer("chunk_index", "chunk_number"),
            section=text("section", "section_title", "heading"),
            page_start=integer("page_start", "page", "page_number"),
            page_end=integer("page_end", "page", "page_number"),
            char_start=integer("char_start", "start_offset"),
            char_end=integer("char_end", "end_offset"),
        )

    @property
    def has_merge_coordinates(self) -> bool:
        return (
            self.document_id is not None
            and self.chunk_index is not None
            and self.chunk_index >= 0
            and self.char_start is not None
            and self.char_end is not None
            and self.char_start >= 0
            and self.char_start < self.char_end
        )


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    embedding: Tuple[float, ...]
    metadata: ChunkMetadata
    union_rank: int


@dataclass(frozen=True)
class FalsePositivePriors:
    """Precomputed priors; construction/calibration intentionally lives elsewhere."""

    source: Mapping[str, float]
    target: Mapping[str, float]
    unseen_source: float
    unseen_target: float

    def __post_init__(self) -> None:
        values = [self.unseen_source, self.unseen_target]
        values.extend(float(value) for value in self.source.values())
        values.extend(float(value) for value in self.target.values())
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("False-positive priors must all be in [0, 1].")

    def source_for(self, chunk_id: str) -> float:
        return float(self.source.get(chunk_id, self.unseen_source))

    def target_for(self, chunk_id: str) -> float:
        return float(self.target.get(chunk_id, self.unseen_target))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": {str(key): float(value) for key, value in self.source.items()},
            "target": {str(key): float(value) for key, value in self.target.items()},
            "unseen_source": float(self.unseen_source),
            "unseen_target": float(self.unseen_target),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FalsePositivePriors":
        source = value.get("source") or {}
        target = value.get("target") or {}
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise ValueError("False-positive prior maps must be JSON objects.")
        return cls(
            source={str(key): float(score) for key, score in source.items()},
            target={str(key): float(score) for key, score in target.items()},
            unseen_source=float(value["unseen_source"]),
            unseen_target=float(value["unseen_target"]),
        )


@dataclass(frozen=True)
class AmbiCalProfile:
    """Read-only online profile produced by a future independent calibration step."""

    false_positive_priors: Optional[FalsePositivePriors] = None
    null_strengths: Tuple[float, ...] = ()
    utility_signature: Optional[str] = None

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            for value in self.null_strengths
        ):
            raise ValueError("Null evidence strengths must be in [0, 1].")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "false_positive_priors": (
                self.false_positive_priors.to_dict()
                if self.false_positive_priors is not None
                else None
            ),
            "null_strengths": [float(value) for value in self.null_strengths],
            "utility_signature": self.utility_signature,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AmbiCalProfile":
        raw_priors = value.get("false_positive_priors")
        if raw_priors is not None and not isinstance(raw_priors, Mapping):
            raise ValueError("false_positive_priors must be a JSON object or null.")
        raw_null = value.get("null_strengths") or []
        if isinstance(raw_null, (str, bytes)) or not isinstance(raw_null, Sequence):
            raise ValueError("null_strengths must be a JSON array.")
        signature = value.get("utility_signature")
        return cls(
            false_positive_priors=(
                FalsePositivePriors.from_dict(raw_priors)
                if raw_priors is not None
                else None
            ),
            null_strengths=tuple(float(item) for item in raw_null),
            utility_signature=(str(signature) if signature is not None else None),
        )


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    text: str
    base_value: float
    marginal_utility: float
    selected_rank: int
    metadata: ChunkMetadata
    merged_chunk_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["merged_chunk_ids"] = list(self.merged_chunk_ids)
        return value


@dataclass(frozen=True)
class AmbiCalResult:
    source_query: str
    translated_query: Optional[str]
    translation_active: bool
    selected_k: int
    evidence: Tuple[EvidenceItem, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def documents(self) -> List[str]:
        return [item.text for item in self.evidence]

    @property
    def scores(self) -> List[float]:
        return [item.marginal_utility for item in self.evidence]

    @property
    def evidence_count_after_merge(self) -> int:
        return len(self.evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_query": self.source_query,
            "translated_query": self.translated_query,
            "translation_active": self.translation_active,
            # k is decided before optional interval merging.  Keep the legacy
            # name and expose both counts so downstream code cannot confuse a
            # stopping decision with the number of returned merged intervals.
            "selected_k": self.selected_k,
            "selected_chunk_count_before_merge": self.selected_k,
            "evidence_count_after_merge": self.evidence_count_after_merge,
            "evidence": [item.to_dict() for item in self.evidence],
            "diagnostics": dict(self.diagnostics),
        }


def ensure_profile_ready(
    profile: Optional[AmbiCalProfile],
    require_false_positive: bool,
    require_null: bool,
) -> AmbiCalProfile:
    if profile is None:
        if require_false_positive or require_null:
            raise ValueError(
                "This AmbiCal configuration requires a precomputed profile. "
                "Calibration construction is intentionally not performed online."
            )
        return AmbiCalProfile()
    if require_false_positive and profile.false_positive_priors is None:
        raise ValueError("False-positive correction is enabled but no priors were supplied.")
    if require_null and not profile.null_strengths:
        raise ValueError("Null gating is enabled but the Null distribution is empty.")
    return profile
