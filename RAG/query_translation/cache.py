"""SQLite-backed translation cache used by AmbiCal experiments."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TranslationCacheEntry:
    cache_key: str
    source_query: str
    translated_query: str
    raw_response: str
    source_language: str
    target_language: str
    translator_alias: str
    actual_model: str
    model_parameters_json: str
    prompt_sha256: str
    parser_version: str
    created_at_utc: str


class TranslationCache:
    """Small durable cache with one row per translation configuration/query."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        read_only: bool = False,
        require_new: bool = False,
    ) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        self.require_new = bool(require_new)
        if self.read_only and self.require_new:
            raise ValueError("A read-only translation cache cannot require a new file.")
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"Read-only translation cache does not exist: {self.path}"
                )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.require_new:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.close(descriptor)
            self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"file:{self.path.resolve().as_posix()}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, timeout=30.0, uri=True)
        else:
            connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    cache_key TEXT PRIMARY KEY,
                    source_query TEXT NOT NULL,
                    translated_query TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    translator_alias TEXT NOT NULL,
                    actual_model TEXT NOT NULL,
                    model_parameters_json TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                )
                """
            )

    def get(self, cache_key: str) -> Optional[TranslationCacheEntry]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM translations WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        return TranslationCacheEntry(
            cache_key=row["cache_key"],
            source_query=row["source_query"],
            translated_query=row["translated_query"],
            raw_response=row["raw_response"],
            source_language=row["source_language"],
            target_language=row["target_language"],
            translator_alias=row["translator_alias"],
            actual_model=row["actual_model"],
            model_parameters_json=row["model_parameters_json"],
            prompt_sha256=row["prompt_sha256"],
            parser_version=row["parser_version"],
            created_at_utc=row["created_at_utc"],
        )

    def put(self, entry: TranslationCacheEntry, overwrite: bool = False) -> None:
        if self.read_only:
            raise PermissionError("Cannot write to a read-only translation cache.")
        statement = "INSERT OR REPLACE" if overwrite else "INSERT OR IGNORE"
        with self._connect() as connection:
            connection.execute(
                f"""
                {statement} INTO translations (
                    cache_key, source_query, translated_query, raw_response,
                    source_language, target_language, translator_alias,
                    actual_model, model_parameters_json, prompt_sha256,
                    parser_version, created_at_utc, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.cache_key,
                    entry.source_query,
                    entry.translated_query,
                    entry.raw_response,
                    entry.source_language,
                    entry.target_language,
                    entry.translator_alias,
                    entry.actual_model,
                    entry.model_parameters_json,
                    entry.prompt_sha256,
                    entry.parser_version,
                    entry.created_at_utc,
                    self.SCHEMA_VERSION,
                ),
            )

    @staticmethod
    def now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()
