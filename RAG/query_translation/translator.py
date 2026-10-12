"""Reproducible LLM query translation for AmbiCal."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from llm_connectors.base_llm_connector import BaseLLMConnector
from .cache import TranslationCache, TranslationCacheEntry


@dataclass(frozen=True)
class TranslationResult:
    source_query: str
    translated_query: str
    raw_response: str
    cache_key: str
    cache_hit: bool
    source_language: str
    target_language: str
    translator_alias: str
    actual_model: str
    prompt_sha256: str

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source_query": self.source_query,
            "translated_query": self.translated_query,
            "translation_cache_key": self.cache_key,
            "translation_cache_hit": self.cache_hit,
            "source_query_language": self.source_language,
            "translation_target_language": self.target_language,
            "translator_model_alias": self.translator_alias,
            "translator_actual_model": self.actual_model,
            "translation_prompt_sha256": self.prompt_sha256,
        }


class QueryTranslator:
    """Translate a search query once and reuse the exact output from SQLite."""

    PARSER_VERSION = "ambical-translation-v1"
    CACHE_MODES = {"read_write", "read_only", "refresh"}

    def __init__(
        self,
        connector: Optional[BaseLLMConnector],
        prompt_template: str,
        cache_path: str | Path,
        cache_mode: str = "read_write",
        translator_alias: Optional[str] = None,
        require_new_cache: bool = False,
        model_binary_sha256: Optional[str] = None,
    ) -> None:
        if cache_mode not in self.CACHE_MODES:
            raise ValueError(
                f"Unsupported translation cache mode '{cache_mode}'. "
                f"Choose one of {sorted(self.CACHE_MODES)}."
            )
        self.connector = connector
        self.prompt_template = prompt_template
        self.prompt_sha256 = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
        self.cache = TranslationCache(
            cache_path,
            read_only=cache_mode == "read_only",
            require_new=require_new_cache,
        )
        self.cache_mode = cache_mode
        self.translator_alias = translator_alias or getattr(connector, "model_name", "identity")
        connector_config = dict(getattr(connector, "config", {}) or {})
        self.actual_model = str(
            connector_config.get("name")
            or getattr(connector, "ollama_model_name", "identity")
        )
        self.model_binary_sha256 = None
        if model_binary_sha256 is not None:
            normalized_digest = (
                str(model_binary_sha256).strip().removeprefix("sha256:").lower()
            )
            if not re.fullmatch(r"[0-9a-f]{64}", normalized_digest):
                raise ValueError("model_binary_sha256 must be a full SHA-256 digest.")
            self.model_binary_sha256 = normalized_digest
        tracked_parameters = {
            key: connector_config.get(key)
            for key in ("name", "temperature", "num_predict", "context_window")
            if key in connector_config
        }
        if self.model_binary_sha256 is not None:
            tracked_parameters["model_binary_sha256"] = self.model_binary_sha256
        self.model_parameters_json = json.dumps(
            tracked_parameters, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        return unicodedata.normalize("NFC", str(query)).strip()

    def _cache_key(self, query: str, source_language: str, target_language: str) -> str:
        payload = {
            "query": query,
            "source_language": source_language.casefold(),
            "target_language": target_language.casefold(),
            "translator_alias": self.translator_alias,
            "actual_model": self.actual_model,
            "model_parameters": json.loads(self.model_parameters_json or "{}"),
            "prompt_sha256": self.prompt_sha256,
            "parser_version": self.PARSER_VERSION,
        }
        canonical = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_translation(response: str) -> str:
        text = str(response or "").strip()
        tagged = re.search(
            r"<translation>\s*(.*?)\s*</translation>", text, flags=re.IGNORECASE | re.DOTALL
        )
        if tagged:
            text = tagged.group(1).strip()
        else:
            text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
            text = re.sub(
                r"^(?:translation|translated query)\s*:\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
        text = " ".join(text.split())
        if not text:
            raise ValueError("Translator returned an empty query.")
        return text

    def translate(
        self, query: str, source_language: str, target_language: str
    ) -> TranslationResult:
        source_query = self._normalize_query(query)
        if not source_query:
            raise ValueError("Source query cannot be empty.")
        source_language = str(source_language).strip()
        target_language = str(target_language).strip()
        if not source_language or not target_language:
            raise ValueError("Source and target languages must be non-empty.")

        cache_key = self._cache_key(source_query, source_language, target_language)
        identity = source_language.casefold() == target_language.casefold()
        if identity:
            return TranslationResult(
                source_query=source_query,
                translated_query=source_query,
                raw_response=source_query,
                cache_key=cache_key,
                cache_hit=True,
                source_language=source_language,
                target_language=target_language,
                translator_alias="identity",
                actual_model="identity",
                prompt_sha256=self.prompt_sha256,
            )

        if self.cache_mode != "refresh":
            cached = self.cache.get(cache_key)
            if cached is not None:
                return TranslationResult(
                    source_query=cached.source_query,
                    translated_query=cached.translated_query,
                    raw_response=cached.raw_response,
                    cache_key=cached.cache_key,
                    cache_hit=True,
                    source_language=cached.source_language,
                    target_language=cached.target_language,
                    translator_alias=cached.translator_alias,
                    actual_model=cached.actual_model,
                    prompt_sha256=cached.prompt_sha256,
                )
        if self.cache_mode == "read_only":
            raise KeyError(
                "Translation cache miss in read_only mode for key " + cache_key
            )
        if self.connector is None:
            raise RuntimeError("A translator connector is required for non-identity translation.")

        prompt = self.prompt_template.format(
            source_language=source_language,
            target_language=target_language,
            query=source_query,
        )
        raw_response = str(self.connector.invoke(prompt))
        translated_query = self._parse_translation(raw_response)
        entry = TranslationCacheEntry(
            cache_key=cache_key,
            source_query=source_query,
            translated_query=translated_query,
            raw_response=raw_response,
            source_language=source_language,
            target_language=target_language,
            translator_alias=self.translator_alias,
            actual_model=self.actual_model,
            model_parameters_json=self.model_parameters_json,
            prompt_sha256=self.prompt_sha256,
            parser_version=self.PARSER_VERSION,
            created_at_utc=TranslationCache.now_utc(),
        )
        self.cache.put(entry, overwrite=self.cache_mode == "refresh")
        return TranslationResult(
            source_query=source_query,
            translated_query=translated_query,
            raw_response=raw_response,
            cache_key=cache_key,
            cache_hit=False,
            source_language=source_language,
            target_language=target_language,
            translator_alias=self.translator_alias,
            actual_model=self.actual_model,
            prompt_sha256=self.prompt_sha256,
        )
