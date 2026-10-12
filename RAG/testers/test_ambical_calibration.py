import ast
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from retrieval_pipelines.ambical import (
    AmbiCalConfig,
    AmbiCalSwitches,
    build_false_positive_priors,
    build_null_strengths,
    read_full_collection,
)
from utils.ambical_artifacts import (
    collection_fingerprint,
    directory_manifest_fingerprint,
    translation_scope_identity,
)


RAG_ROOT = Path(__file__).resolve().parents[1]


class FakeCollection:
    name = "fake_collection"

    def __init__(self, rows=None):
        self.rows = rows or {
            "b": ("beta", [0.0, 1.0], {"page": 2}),
            "c": ("middle", [2 ** -0.5, 2 ** -0.5], {"page": 3}),
            "a": ("alpha", [1.0, 0.0], {"page": 1}),
        }

    def count(self):
        return len(self.rows)

    def get(self, *, ids=None, limit=None, offset=None, include=None):
        if ids is not None:
            keys = [str(item) for item in ids]
        else:
            keys = list(self.rows)[int(offset or 0) : int(offset or 0) + int(limit)]
        return {
            "ids": keys,
            "documents": [self.rows[key][0] for key in keys],
            "embeddings": [self.rows[key][1] for key in keys],
            "metadatas": [self.rows[key][2] for key in keys],
        }

    def query(self, *, query_embeddings, n_results, include):
        query = query_embeddings[0]
        ranked = sorted(
            self.rows,
            key=lambda key: (
                -sum(x * y for x, y in zip(query, self.rows[key][1])),
                key,
            ),
        )
        return {"ids": [ranked[:n_results]]}


class FakeEmbedder:
    def __init__(self, mapping):
        self.mapping = dict(mapping)
        self.calls = []

    def vectorize_text(self, text):
        self.calls.append(text)
        return [self.mapping[text]]


class FakeTranslationResult:
    def __init__(self, translated_query):
        self.translated_query = translated_query
        self.cache_key = "cache-" + translated_query
        self.cache_hit = True

    def diagnostics(self):
        return {"translation_cache_hit": True}


class FakeTranslator:
    def __init__(self, mapping):
        self.mapping = dict(mapping)
        self.calls = []

    def translate(self, query, source_language, target_language):
        self.calls.append((query, source_language, target_language))
        value = self.mapping[query]
        if isinstance(value, Exception):
            raise value
        return FakeTranslationResult(value)


def calibration_config(*, robust_normalization=False, candidate_top_n=2):
    return AmbiCalConfig(
        lambda_false_positive=0.2,
        null_low_threshold=0.1,
        null_high_threshold=0.8,
        fixed_k=2,
        candidate_top_n=candidate_top_n,
        max_k=3,
        gate_top_m=2,
        switches=AmbiCalSwitches(
            translation_branch=True,
            robust_normalization=robust_normalization,
            branch_confidence=True,
            translation_consistency=True,
            ranking_overlap=True,
            dynamic_branch_weights=True,
            false_positive_prior=True,
            context_correction=False,
            context_gate=False,
            soft_redundancy=False,
            consistency_entropy=True,
            maximum_gap=True,
            entropy_candidate=True,
            null_gate=True,
            adaptive_k=False,
            merge_structured_evidence=False,
        ),
    )


class AmbiCalCalibrationTests(unittest.TestCase):
    def test_model_directory_and_translation_identity_are_content_bound(self):
        with tempfile.TemporaryDirectory(dir=RAG_ROOT / "testers") as directory:
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("one", encoding="utf-8")
            (model_dir / "weights.bin").write_bytes(b"weights")
            first = directory_manifest_fingerprint(model_dir)
            second = directory_manifest_fingerprint(model_dir)
            (model_dir / "weights.bin").write_bytes(b"changed")
            changed = directory_manifest_fingerprint(model_dir)

        self.assertEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(first["sha256"], changed["sha256"])
        translator = SimpleNamespace(
            translator_alias="translator",
            actual_model="qwen2.5:7b",
            model_parameters_json='{"temperature":0.0}',
            prompt_sha256="a" * 64,
            PARSER_VERSION="translation-v1",
        )
        identity = translation_scope_identity(
            translator,
            source_language="English",
            target_language="Vietnamese",
            prompt_file_sha256="b" * 64,
            ollama_sha256="sha256:" + "c" * 64,
        )
        self.assertEqual(identity["translator_ollama_sha256"], "c" * 64)
        self.assertEqual(identity["translation_parser_version"], "translation-v1")

    def test_collection_fingerprint_is_order_independent_and_content_sensitive(self):
        first_rows = {
            "a": ("alpha", [1.0, 0.0], {"page": 1}),
            "b": ("beta", [0.0, 1.0], {"page": 2}),
        }
        second_rows = dict(reversed(list(first_rows.items())))
        first = collection_fingerprint(FakeCollection(first_rows), batch_size=1)
        second = collection_fingerprint(FakeCollection(second_rows), batch_size=1)
        changed = collection_fingerprint(
            FakeCollection(
                {
                    "a": ("changed", [1.0, 0.0], {"page": 1}),
                    "b": first_rows["b"],
                }
            ),
            batch_size=1,
        )
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(first["sha256"], changed["sha256"])

    def test_full_collection_snapshot_is_sorted_and_complete(self):
        snapshot = read_full_collection(
            FakeCollection(), calibration_config(), batch_size=1
        )
        self.assertEqual(snapshot.ids, ("a", "b", "c"))
        self.assertEqual(snapshot.count, 3)
        self.assertEqual(snapshot.embedding_dimension, 2)

    def test_false_positive_priors_use_top_r_hits_and_beta_smoothing(self):
        config = calibration_config(robust_normalization=False)
        snapshot = read_full_collection(FakeCollection(), config)
        embedder = FakeEmbedder(
            {
                "source question": [1.0, 0.0],
                "cau hoi dich": [0.0, 1.0],
            }
        )
        translator = FakeTranslator({"source question": "cau hoi dich"})
        priors, trace = build_false_positive_priors(
            snapshot,
            embedder,
            [
                {
                    "calibration_id": "fp-001",
                    "question": "source question",
                }
            ],
            config,
            translator,
            "English",
            "Vietnamese",
        )
        self.assertEqual(set(priors.source), {"a", "b", "c"})
        self.assertEqual(set(priors.target), {"a", "b", "c"})
        self.assertAlmostEqual(priors.source["a"], 2.0 / 3.0)
        self.assertAlmostEqual(priors.source["b"], 1.0 / 3.0)
        self.assertAlmostEqual(priors.source["c"], 2.0 / 3.0)
        self.assertAlmostEqual(priors.target["a"], 1.0 / 3.0)
        self.assertAlmostEqual(priors.target["b"], 2.0 / 3.0)
        self.assertAlmostEqual(priors.target["c"], 2.0 / 3.0)
        self.assertAlmostEqual(priors.unseen_source, 1.0 / 3.0)
        self.assertAlmostEqual(priors.unseen_target, 1.0 / 3.0)
        self.assertEqual(trace["chunk_count"], 3)
        self.assertEqual(trace["requested_top_r"], 2)
        self.assertEqual(trace["effective_top_r"], 2)
        self.assertEqual(trace["source_hit_counts"], {"a": 1, "b": 0, "c": 1})
        self.assertEqual(trace["target_hit_counts"], {"a": 0, "b": 1, "c": 1})
        self.assertEqual(len(trace["questions"][0]["source"]["raw_scores"]), 3)

    def test_false_positive_top_r_ties_are_broken_by_chunk_id(self):
        rows = {
            "b": ("beta", [1.0, 0.0], {"page": 2}),
            "a": ("alpha", [1.0, 0.0], {"page": 1}),
            "c": ("gamma", [0.0, 1.0], {"page": 3}),
        }
        config = calibration_config(candidate_top_n=1)
        snapshot = read_full_collection(FakeCollection(rows), config)
        embedder = FakeEmbedder({"source": [1.0, 0.0], "target": [0.0, 1.0]})
        priors, trace = build_false_positive_priors(
            snapshot,
            embedder,
            [{"calibration_id": "fp-tie", "question": "source"}],
            config,
            FakeTranslator({"source": "target"}),
            "English",
            "Vietnamese",
        )

        self.assertEqual(trace["questions"][0]["source"]["top_chunk_ids"], ["a"])
        self.assertAlmostEqual(priors.source["a"], 2.0 / 3.0)
        self.assertAlmostEqual(priors.source["b"], 1.0 / 3.0)
        self.assertEqual(trace["questions"][0]["target"]["top_chunk_ids"], ["c"])

    def test_false_positive_translation_failure_is_fatal(self):
        config = calibration_config()
        snapshot = read_full_collection(FakeCollection(), config)
        with self.assertRaisesRegex(RuntimeError, "Translation failed"):
            build_false_positive_priors(
                snapshot,
                FakeEmbedder({"source question": [1.0, 0.0]}),
                [{"question": "source question"}],
                config,
                FakeTranslator({"source question": RuntimeError("offline")}),
                "English",
                "Vietnamese",
            )

    def test_null_strengths_use_shared_pre_gate_path_and_final_signature(self):
        config = calibration_config(robust_normalization=False)
        collection = FakeCollection()
        snapshot = read_full_collection(collection, config)
        embedder = FakeEmbedder(
            {
                "fp question": [1.0, 0.0],
                "fp dich": [0.0, 1.0],
                "null question": [1.0, 0.0],
                "null dich": [0.0, 1.0],
            }
        )
        translator = FakeTranslator(
            {"fp question": "fp dich", "null question": "null dich"}
        )
        priors, _ = build_false_positive_priors(
            snapshot,
            embedder,
            [{"calibration_id": "fp-001", "question": "fp question"}],
            config,
            translator,
            "English",
            "Vietnamese",
        )
        strengths, trace = build_null_strengths(
            collection,
            embedder,
            [{"calibration_id": "null-001", "question": "null question"}],
            config,
            priors,
            translator,
            "English",
            "Vietnamese",
        )
        self.assertEqual(len(strengths), 1)
        self.assertTrue(math.isfinite(strengths[0]))
        self.assertGreaterEqual(strengths[0], 0.0)
        self.assertLessEqual(strengths[0], 1.0)
        self.assertEqual(trace["final_utility_signature"], config.utility_signature())
        self.assertNotEqual(
            trace["probe_utility_signature"], config.utility_signature()
        )
        self.assertEqual(trace["probe_override"], {"null_gate": False})

    def test_null_translation_failure_does_not_publish_short_distribution(self):
        config = calibration_config(robust_normalization=False)
        collection = FakeCollection()
        snapshot = read_full_collection(collection, config)
        embedder = FakeEmbedder(
            {"fp": [1.0, 0.0], "fp-vn": [0.0, 1.0], "null": [1.0, 0.0]}
        )
        working_translator = FakeTranslator({"fp": "fp-vn"})
        priors, _ = build_false_positive_priors(
            snapshot,
            embedder,
            [{"question": "fp"}],
            config,
            working_translator,
            "English",
            "Vietnamese",
        )
        with self.assertRaisesRegex(RuntimeError, "Translation failed"):
            build_null_strengths(
                collection,
                embedder,
                [{"question": "null"}],
                config,
                priors,
                FakeTranslator({"null": RuntimeError("offline")}),
                "English",
                "Vietnamese",
            )

    def test_null_gate_false_final_config_still_stores_pre_gate_null_strengths(self):
        from dataclasses import replace

        full_config = calibration_config(robust_normalization=False)
        ablated = replace(
            full_config, switches=replace(full_config.switches, null_gate=False)
        )
        ablated.validate()
        self.assertNotEqual(
            ablated.utility_signature(), full_config.utility_signature()
        )
        collection = FakeCollection()
        snapshot = read_full_collection(collection, ablated)
        embedder = FakeEmbedder(
            {
                "fp question": [1.0, 0.0],
                "fp dich": [0.0, 1.0],
                "null question": [1.0, 0.0],
                "null dich": [0.0, 1.0],
            }
        )
        translator = FakeTranslator(
            {"fp question": "fp dich", "null question": "null dich"}
        )
        priors, _ = build_false_positive_priors(
            snapshot,
            embedder,
            [{"calibration_id": "fp-001", "question": "fp question"}],
            ablated,
            translator,
            "English",
            "Vietnamese",
        )
        strengths, trace = build_null_strengths(
            collection,
            embedder,
            [{"calibration_id": "null-001", "question": "null question"}],
            ablated,
            priors,
            translator,
            "English",
            "Vietnamese",
        )
        self.assertEqual(len(strengths), 1)
        self.assertTrue(math.isfinite(strengths[0]))
        self.assertGreaterEqual(strengths[0], 0.0)
        self.assertLessEqual(strengths[0], 1.0)
        self.assertEqual(trace["final_utility_signature"], ablated.utility_signature())
        # The gate-off override is a no-op for an already gate-off config, so
        # the probe path is the final configuration itself.
        self.assertEqual(trace["probe_utility_signature"], ablated.utility_signature())
        self.assertEqual(trace["probe_override"], {"null_gate": False})

    def test_entrypoint_has_no_rag_tester_or_evaluator_dependency(self):
        path = RAG_ROOT / "build_ambical_profile.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any("rag_tester" in name for name in imported))
        self.assertFalse(any("evaluation" in name for name in imported))
        self.assertIn(
            "chromadb.PersistentClient(path=str(working_persist_dir))",
            source,
        )
        self.assertNotIn(
            "chromadb.PersistentClient(path=str(inputs.chroma_persist_dir))",
            source,
        )
        self.assertIn("os.replace(pending_profile_path, profile_path)", source)


if __name__ == "__main__":
    unittest.main()
