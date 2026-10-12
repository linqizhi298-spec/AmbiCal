"""Deterministic source-query phrase generation for counterfactual correction."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple


_TOKEN_PATTERN = re.compile(
    r"(?:[A-Z][A-Z0-9./%-]{1,})"
    r"|(?:\d+(?:[.,]\d+)?(?:\s*(?:%|°[CF]?|[A-Za-zµμ]+(?:/[A-Za-zµμ]+)?))?)"
    r"|(?:[^\W\d_]+(?:[-'][^\W\d_]+)*)",
    flags=re.UNICODE,
)


DEFAULT_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "to",
        "was",
        "were",
        "with",
        "can",
        "could",
        "does",
        "do",
        "did",
        "should",
        "would",
        "may",
        "might",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "compare",
        "explain",
        "describe",
        "summarize",
        "identify",
        "list",
        "calculate",
        "berapa",
        "apa",
        "siapa",
        "kapan",
        "di mana",
        "mengapa",
        "bagaimana",
    }
)


@dataclass(frozen=True)
class QueryToken:
    text: str
    start: int
    end: int
    token_index: int
    is_content: bool


@dataclass(frozen=True)
class CounterfactualPhrase:
    text: str
    start: int
    end: int
    token_start: int
    token_end: int
    counterfactual_query: str
    content_token_count: int
    source: str

    def overlaps(self, other: "CounterfactualPhrase") -> bool:
        return self.start < other.end and other.start < self.end


def normalize_query_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).strip().split())


class CounterfactualPhraseGenerator:
    """Generate content words, 1--3 word spans, NPs and numeric expressions.

    A caller may inject a frozen noun-phrase extractor.  Without one, a
    dependency-free conservative fallback emits longer multi-content-token
    nominal candidates; these remain candidates and are subsequently filtered
    by measured counterfactual impact and the non-overlap rule.
    """

    def __init__(
        self,
        *,
        stop_words: Optional[Iterable[str]] = None,
        noun_phrase_extractor: Optional[Callable[[str], Sequence[str]]] = None,
        minimum_remaining_content_tokens: int = 1,
    ) -> None:
        if minimum_remaining_content_tokens < 1:
            raise ValueError("minimum_remaining_content_tokens must be at least 1.")
        merged = set(DEFAULT_STOP_WORDS)
        merged.update(str(value).casefold() for value in (stop_words or ()))
        self.stop_words: Set[str] = merged
        self.noun_phrase_extractor = (
            noun_phrase_extractor or self._default_noun_phrase_candidates
        )
        self.minimum_remaining_content_tokens = minimum_remaining_content_tokens

    def tokenize(self, query: str) -> List[QueryToken]:
        normalized = unicodedata.normalize("NFC", str(query))
        tokens: List[QueryToken] = []
        for index, match in enumerate(_TOKEN_PATTERN.finditer(normalized)):
            text = match.group(0)
            folded = text.casefold()
            has_number = bool(re.search(r"\d", text))
            # Uppercase question words such as HOW/WHAT remain stop words; an
            # acronym is indivisible, but capitalization alone does not make a
            # stop word semantically informative.
            is_content = has_number or folded not in self.stop_words
            tokens.append(
                QueryToken(
                    text=text,
                    start=match.start(),
                    end=match.end(),
                    token_index=index,
                    is_content=is_content,
                )
            )
        return tokens

    @staticmethod
    def remove_span(query: str, start: int, end: int) -> str:
        remainder = query[:start] + " " + query[end:]
        remainder = re.sub(r"\s+([,.;:!?])", r"\1", remainder)
        remainder = re.sub(r"([([{])\s+", r"\1", remainder)
        remainder = re.sub(r"\s+", " ", remainder).strip(" ,;:-")
        return normalize_query_text(remainder)

    def _remaining_content_count(self, query: str) -> int:
        return sum(token.is_content for token in self.tokenize(query))

    def _make_phrase(
        self,
        query: str,
        tokens: Sequence[QueryToken],
        first: int,
        last: int,
        source: str,
    ) -> Optional[CounterfactualPhrase]:
        selected = tokens[first : last + 1]
        if not selected or not any(token.is_content for token in selected):
            return None
        start, end = selected[0].start, selected[-1].end
        counterfactual = self.remove_span(query, start, end)
        if not counterfactual or counterfactual == normalize_query_text(query):
            return None
        remaining = self._remaining_content_count(counterfactual)
        if remaining < self.minimum_remaining_content_tokens:
            return None
        return CounterfactualPhrase(
            text=query[start:end].strip(),
            start=start,
            end=end,
            token_start=selected[0].token_index,
            token_end=selected[-1].token_index,
            counterfactual_query=counterfactual,
            content_token_count=sum(token.is_content for token in selected),
            source=source,
        )

    def _noun_phrases(
        self, query: str, tokens: Sequence[QueryToken]
    ) -> List[CounterfactualPhrase]:
        values: List[CounterfactualPhrase] = []
        for raw_phrase in self.noun_phrase_extractor(query):
            phrase = normalize_query_text(raw_phrase)
            if not phrase:
                continue
            pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
            for match in pattern.finditer(query):
                covered = [
                    index
                    for index, token in enumerate(tokens)
                    if token.start >= match.start() and token.end <= match.end()
                ]
                if not covered:
                    continue
                candidate = self._make_phrase(
                    query, tokens, covered[0], covered[-1], "noun_phrase"
                )
                if candidate is not None:
                    values.append(candidate)
        return values

    def _default_noun_phrase_candidates(self, query: str) -> Sequence[str]:
        """Return conservative long nominal candidates without an NLP model.

        The method document does not prescribe a parser.  This fallback only
        fills the part not already covered by 1--3 content-word spans: windows
        of 4--6 content tokens, allowing connector stop words but never crossing
        hard punctuation.  A production caller can replace it with one fixed
        parser through ``noun_phrase_extractor``.
        """

        tokens = self.tokenize(query)
        content_indices = [
            index for index, token in enumerate(tokens) if token.is_content
        ]
        candidates: List[str] = []
        for start_position in range(len(content_indices)):
            for content_length in (4, 5, 6):
                end_position = start_position + content_length - 1
                if end_position >= len(content_indices):
                    continue
                first = content_indices[start_position]
                last = content_indices[end_position]
                start = tokens[first].start
                end = tokens[last].end
                between = query[start:end]
                if re.search(r"[,;:!?]", between):
                    continue
                if (
                    sum(token.is_content for token in tokens[first : last + 1])
                    != content_length
                ):
                    continue
                candidates.append(normalize_query_text(between))
        return candidates

    def generate(self, query: str) -> List[CounterfactualPhrase]:
        query = unicodedata.normalize("NFC", str(query)).strip()
        if not query:
            return []
        tokens = self.tokenize(query)
        values: List[CounterfactualPhrase] = []

        for index, token in enumerate(tokens):
            if token.is_content:
                candidate = self._make_phrase(query, tokens, index, index, "content_word")
                if candidate is not None:
                    values.append(candidate)

        # Contiguous 1--3 content-word spans.  A span may contain connecting
        # stop words, but it may not jump over another content word.
        content_indices = [i for i, token in enumerate(tokens) if token.is_content]
        for start_position in range(len(content_indices)):
            for length in (2, 3):
                end_position = start_position + length - 1
                if end_position >= len(content_indices):
                    continue
                first = content_indices[start_position]
                last = content_indices[end_position]
                content_inside = sum(token.is_content for token in tokens[first : last + 1])
                if content_inside != length:
                    continue
                candidate = self._make_phrase(
                    query, tokens, first, last, f"content_span_{length}"
                )
                if candidate is not None:
                    values.append(candidate)

        # Numeric/unit and acronym expressions are already emitted as indivisible
        # tokens; relabel them for diagnostics.
        for index, token in enumerate(tokens):
            if re.search(r"\d", token.text) or re.fullmatch(
                r"[A-Z][A-Z0-9./%-]{1,}", token.text
            ):
                candidate = self._make_phrase(
                    query, tokens, index, index, "numeric_unit_or_acronym"
                )
                if candidate is not None:
                    values.append(candidate)

        values.extend(self._noun_phrases(query, tokens))

        deduplicated: List[CounterfactualPhrase] = []
        seen_counterfactuals: Set[str] = set()
        seen_spans: Set[Tuple[int, int]] = set()
        for candidate in values:
            key = candidate.counterfactual_query.casefold()
            span = (candidate.start, candidate.end)
            if key in seen_counterfactuals or span in seen_spans:
                continue
            seen_counterfactuals.add(key)
            seen_spans.add(span)
            deduplicated.append(candidate)
        return deduplicated


def select_non_overlapping_phrases(
    scored: Sequence[Tuple[CounterfactualPhrase, float]], limit: int
) -> List[Tuple[CounterfactualPhrase, float]]:
    """Keep high-impact phrases with deterministic non-overlap filtering."""

    if limit < 1:
        return []
    ordered = sorted(
        scored,
        key=lambda item: (
            -float(item[1]),
            -item[0].content_token_count,
            item[0].start,
            item[0].end,
            item[0].text.casefold(),
        ),
    )
    chosen: List[Tuple[CounterfactualPhrase, float]] = []
    for candidate, impact in ordered:
        if any(candidate.overlaps(existing) for existing, _ in chosen):
            continue
        chosen.append((candidate, float(impact)))
        if len(chosen) == limit:
            break
    return chosen


def select_compact_diverse_phrases(
    scored: Sequence[Tuple[CounterfactualPhrase, float]], limit: int
) -> List[Tuple[CounterfactualPhrase, float]]:
    """Select compact deletion phrases from distinct query regions.

    AmbiCal v3.1 uses this stricter selector because raw embedding impact
    otherwise favours deleting most of the question. Such deletions leave a
    malformed or generic counterfactual query and do not isolate one semantic
    cue. An eligible phrase therefore contains at most two content tokens and
    spans at most three surface tokens. For the required two-view setting, the
    selected spans must also have at least one token between them whenever such
    a pair exists.

    The rule is deterministic and has no fitted threshold: compactness is a
    structural validity constraint, while ranking impact still determines the
    best valid pair. The legacy selector remains untouched for v1--v3.
    """

    if limit < 1:
        return []

    compact = [
        (phrase, float(impact))
        for phrase, impact in scored
        if phrase.content_token_count <= 2
        and (phrase.token_end - phrase.token_start + 1) <= 3
    ]
    if not compact:
        return []

    def item_key(
        item: Tuple[CounterfactualPhrase, float]
    ) -> Tuple[float, int, int, int, str]:
        phrase, impact = item
        return (
            -float(impact),
            -phrase.content_token_count,
            phrase.start,
            phrase.end,
            phrase.text.casefold(),
        )

    ordered = sorted(compact, key=item_key)
    if limit == 1:
        return ordered[:1]

    pairs: List[
        Tuple[
            float,
            float,
            Tuple[float, int, int, int, str],
            Tuple[float, int, int, int, str],
            Tuple[CounterfactualPhrase, float],
            Tuple[CounterfactualPhrase, float],
        ]
    ] = []
    for index, first in enumerate(ordered):
        first_phrase, first_impact = first
        for second in ordered[index + 1 :]:
            second_phrase, second_impact = second
            if first_phrase.overlaps(second_phrase):
                continue
            token_gap = max(
                second_phrase.token_start - first_phrase.token_end - 1,
                first_phrase.token_start - second_phrase.token_end - 1,
            )
            if token_gap < 1:
                continue
            center_distance = abs(
                (first_phrase.token_start + first_phrase.token_end)
                - (second_phrase.token_start + second_phrase.token_end)
            )
            pairs.append(
                (
                    -(float(first_impact) + float(second_impact)),
                    -float(center_distance),
                    item_key(first),
                    item_key(second),
                    first,
                    second,
                )
            )

    if pairs:
        best = min(pairs)
        chosen = sorted([best[-2], best[-1]], key=item_key)
    else:
        # Short queries may not contain two separated semantic regions. Keep
        # the compactness guarantee and fall back only on non-overlap.
        chosen = []
        for item in ordered:
            phrase, _ = item
            if any(phrase.overlaps(existing) for existing, _ in chosen):
                continue
            chosen.append(item)
            if len(chosen) == limit:
                break

    if len(chosen) >= limit:
        return chosen[:limit]

    # For limits above two, append further compact non-overlapping phrases.
    for item in ordered:
        phrase, _ = item
        if item in chosen or any(phrase.overlaps(existing) for existing, _ in chosen):
            continue
        chosen.append(item)
        if len(chosen) == limit:
            break
    return chosen
