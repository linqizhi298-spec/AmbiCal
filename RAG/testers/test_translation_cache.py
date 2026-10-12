import tempfile
import unittest
from pathlib import Path

from llm_connectors.base_llm_connector import BaseLLMConnector
from query_translation import QueryTranslator


class _FakeConnector(BaseLLMConnector):
    def __init__(self):
        super().__init__(
            "fake-translator",
            {
                "name": "fake-local-model",
                "temperature": 0.0,
                "num_predict": 64,
                "context_window": 2048,
            },
        )
        self.ollama_model_name = "fake-local-model"
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return "<translation>Berapa panjang kabel?</translation>"


class TranslationCacheTests(unittest.TestCase):
    def test_second_translation_is_replayed_from_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            connector = _FakeConnector()
            translator = QueryTranslator(
                connector=connector,
                prompt_template=(
                    "Translate {source_language} to {target_language}: {query}"
                ),
                cache_path=Path(temporary_directory) / "translations.sqlite3",
                translator_alias="fake-translator",
            )
            first = translator.translate("How long is the cable?", "English", "Indonesian")
            second = translator.translate("How long is the cable?", "English", "Indonesian")
            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(connector.calls, 1)
            self.assertEqual(first.translated_query, second.translated_query)

    def test_read_only_cache_miss_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "translations.sqlite3"
            writer = QueryTranslator(
                connector=_FakeConnector(),
                prompt_template="{query}",
                cache_path=cache_path,
                translator_alias="offline",
            )
            writer.translate("cached query", "English", "Vietnamese")
            before = cache_path.read_bytes()
            translator = QueryTranslator(
                connector=None,
                prompt_template="{query}",
                cache_path=cache_path,
                cache_mode="read_only",
                translator_alias="offline",
            )
            with self.assertRaises(KeyError):
                translator.translate("query", "English", "Vietnamese")
            self.assertEqual(cache_path.read_bytes(), before)

    def test_model_binary_digest_participates_in_cache_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "translations.sqlite3"
            writer = QueryTranslator(
                connector=_FakeConnector(),
                prompt_template="{query}",
                cache_path=cache_path,
                translator_alias="offline",
                model_binary_sha256="a" * 64,
            )
            writer.translate("cached query", "English", "Vietnamese")
            before = cache_path.read_bytes()
            reader = QueryTranslator(
                connector=_FakeConnector(),
                prompt_template="{query}",
                cache_path=cache_path,
                cache_mode="read_only",
                translator_alias="offline",
                model_binary_sha256="b" * 64,
            )
            with self.assertRaises(KeyError):
                reader.translate("cached query", "English", "Vietnamese")
            self.assertEqual(cache_path.read_bytes(), before)

    def test_require_new_cache_uses_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "translations.sqlite3"
            cache_path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                QueryTranslator(
                    connector=_FakeConnector(),
                    prompt_template="{query}",
                    cache_path=cache_path,
                    cache_mode="read_write",
                    translator_alias="offline",
                    require_new_cache=True,
                )
            self.assertEqual(cache_path.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
