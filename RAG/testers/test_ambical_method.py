import json
import math
import unittest
from dataclasses import dataclass

import torch

from llm_connectors.base_llm_connector import TokenUsageUnavailableError

from retrieval_pipelines.ambical import (
    ChunkMetadata,
    EvidenceItem,
    AmbiCalConfig,
    AmbiCalProfile,
    AmbiCalRetriever,
    AmbiCalSwitches,
    FalsePositivePriors,
)
from retrieval_pipelines.ambical.numerics import (
    branch_shape,
    dynamic_weights,
    robust_sigmoid_map,
    robust_z_scores,
    stable_softmax,
)
from retrieval_pipelines.ambical.phrases import CounterfactualPhrase
from retrieval_pipelines.ambical.phrases import CounterfactualPhraseGenerator
from retrieval_pipelines.ambical.phrases import select_compact_diverse_phrases
from retrieval_pipelines.ambical.selection import (
    context_gate,
    context_gate_v2,
    counterfactual_phrase_risk,
    empirical_null_probability,
    greedy_evidence_competition,
    merge_structured_evidence,
    positive_context_promotion,
    reranker_promotion,
    select_adaptive_k,
    symmetric_correction_factors,
    topk_counterfactual_promotion,
)


class FakeEmbeddingRetriever:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def vectorize_text(self, text):
        self.calls.append(text)
        return [list(self.vectors[text])]


class FakeOneDimensionalEmbeddingRetriever(FakeEmbeddingRetriever):
    def vectorize_text(self, text):
        self.calls.append(text)
        return torch.tensor(self.vectors[text], dtype=torch.float32)


@dataclass
class FakeTranslationResult:
    source_query: str
    translated_query: str

    def diagnostics(self):
        return {"source_query": self.source_query, "translated_query": self.translated_query}


class FakeTranslator:
    def __init__(self, translations, fail_on=()):
        self.translations = translations
        self.fail_on = set(fail_on)
        self.calls = []

    def translate(self, query, source_language, target_language):
        self.calls.append((query, source_language, target_language))
        if query in self.fail_on:
            raise RuntimeError("scripted translation failure")
        return FakeTranslationResult(query, self.translations[query])


class UnsafeDiagnosticsTranslator:
    def translate(self, query, source_language, target_language):
        class Result:
            translated_query = "target question"

            @staticmethod
            def diagnostics():
                return {
                    "tensor": torch.tensor([1.0, 2.0]),
                    "labels": {"b", "a"},
                    "non_finite": float("nan"),
                }

        return Result()


class UnaccountedTranslator:
    def translate(self, query, source_language, target_language):
        raise TokenUsageUnavailableError("missing provider metrics")


class FakePhraseGenerator:
    def __init__(self, phrases):
        self.phrases = list(phrases)
        self.calls = []

    def generate(self, query):
        self.calls.append(query)
        return list(self.phrases)


class FakeCollection:
    def __init__(self):
        self.rows = {
            "a": {
                "document": "source evidence",
                "embedding": [1.0, 0.2],
                "metadata": {},
            },
            "b": {
                "document": "target evidence",
                "embedding": [0.2, 1.0],
                "metadata": {},
            },
            "c": {
                "document": "shared evidence",
                "embedding": [0.7, 0.7],
                "metadata": {},
            },
        }
        self.query_count = 0

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, include):
        self.query_count += 1
        query = query_embeddings[0]
        ids = ["a", "c"] if query[0] >= query[1] else ["b", "c"]
        return {"ids": [ids[:n_results]], "documents": [[]]}

    def get(self, ids, include):
        returned = list(reversed(ids))
        return {
            "ids": returned,
            "documents": [self.rows[item]["document"] for item in returned],
            "embeddings": [self.rows[item]["embedding"] for item in returned],
            "metadatas": [self.rows[item]["metadata"] for item in returned],
        }


def method_config(**overrides):
    switches = overrides.pop(
        "switches",
        AmbiCalSwitches(
            false_positive_prior=False,
            context_correction=False,
            soft_redundancy=False,
            null_gate=False,
            adaptive_k=False,
            merge_structured_evidence=False,
        ),
    )
    values = {
        "fixed_k": 2,
        "switches": switches,
    }
    values.update(overrides)
    return AmbiCalConfig(**values)


class AmbiCalNumericsTests(unittest.TestCase):
    def test_configuration_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            method_config(branch_temperature=float("nan")).validate()
        with self.assertRaisesRegex(ValueError, "finite"):
            method_config(collection_norm_tolerance=float("inf")).validate()
        with self.assertRaisesRegex(ValueError, "finite"):
            method_config(lambda_context=float("nan")).validate()

    def test_equal_scores_and_zero_mad_iqr_fallback_are_finite(self):
        mapped, stats = robust_sigmoid_map(
            [2.0, 2.0, 2.0, 2.0],
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        self.assertTrue(torch.allclose(mapped, torch.full((4,), 0.5, dtype=torch.float64)))
        self.assertEqual(stats.scale_source, "all_equal")

        z_values, sparse_stats = robust_z_scores(
            [0.0, 0.0, 0.0, 1.0],
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        self.assertTrue(torch.isfinite(z_values).all())
        self.assertTrue(bool(torch.all(torch.abs(z_values) <= 8.0)))
        self.assertIn(sparse_stats.scale_source, {"iqr", "iqr_degenerate"})

    def test_stable_softmax_handles_large_logits(self):
        probabilities = stable_softmax([10000.0, 9999.0], 1.0)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum().item()), 1.0, places=12)

    def test_branch_confidence_uses_all_three_documented_terms(self):
        shape = branch_shape(
            [0.9, 0.5, 0.1],
            temperature=1.0,
            epsilon=1e-8,
            use_confidence=True,
        )
        expected = (
            shape.top_score
            + (1.0 - shape.normalized_entropy)
            + shape.normalized_top_gap
        ) / 3.0
        self.assertAlmostEqual(shape.confidence, expected, places=12)
        self.assertNotAlmostEqual(shape.confidence, 1.0 - shape.normalized_entropy)

    def test_dynamic_weights_keep_the_documented_epsilon_denominator(self):
        source, target = dynamic_weights(
            0.4,
            0.8,
            0.9,
            use_dynamic_weights=True,
            use_translation_consistency=True,
            epsilon=1e-8,
        )
        self.assertAlmostEqual(source, 0.4 / (0.4 + 0.72 + 1e-8), places=12)
        self.assertAlmostEqual(target, 0.72 / (0.4 + 0.72 + 1e-8), places=12)
        self.assertLessEqual(source + target, 1.0)

    def test_phrase_generator_filters_uppercase_question_words(self):
        generator = CounterfactualPhraseGenerator()
        phrases = generator.generate("HOW can irrigation pressure loss be reduced?")
        emitted = {phrase.text.casefold() for phrase in phrases}
        self.assertNotIn("how", emitted)
        self.assertNotIn("can", emitted)
        self.assertIn("irrigation", emitted)
        self.assertTrue(all(phrase.counterfactual_query for phrase in phrases))

    def test_default_phrase_generator_covers_long_nominal_candidates(self):
        generator = CounterfactualPhraseGenerator()
        phrases = generator.generate(
            "Compare integrated rice irrigation water management with operating costs"
        )
        noun_phrases = [item.text.casefold() for item in phrases if item.source == "noun_phrase"]
        self.assertIn("integrated rice irrigation water management", noun_phrases)


class AmbiCalSelectionTests(unittest.TestCase):
    def test_context_gate_matches_hand_calculation_and_threshold_is_inclusive(self):
        gate = context_gate(
            [0.8, 0.8],
            [0.8, 0.8],
            [0.8, 0.8],
            torch.eye(2, dtype=torch.float64),
            chi=1.0,
            top_m=2,
            temperature=1.0,
            threshold=0.0,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        self.assertAlmostEqual(gate.competition, 1.0, places=12)
        self.assertAlmostEqual(gate.weighted_distance, 0.5, places=7)
        self.assertAlmostEqual(gate.context_split, 0.5, places=7)
        self.assertAlmostEqual(gate.route_total_variation, 0.0, places=12)
        self.assertAlmostEqual(gate.path_conflict, 0.0, places=12)
        self.assertAlmostEqual(gate.evidence_strength, 0.4, places=12)
        self.assertAlmostEqual(gate.risk, 0.1, places=7)
        inclusive = context_gate(
            [0.8, 0.8],
            [0.8, 0.8],
            [0.8, 0.8],
            torch.eye(2, dtype=torch.float64),
            chi=1.0,
            top_m=2,
            temperature=1.0,
            threshold=gate.risk,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        self.assertTrue(inclusive.triggered)

    def test_context_gate_increases_under_route_conflict_and_low_consistency(self):
        common = dict(
            initial_scores=[0.8, 0.8],
            source_branch_support=[1.0, 0.0],
            pairwise_cosine=torch.eye(2, dtype=torch.float64),
            top_m=2,
            temperature=0.1,
            threshold=1.0,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        conflict = context_gate(
            target_branch_support=[0.0, 1.0], chi=0.0, **common
        )
        aligned = context_gate(
            target_branch_support=[1.0, 0.0], chi=1.0, **common
        )
        self.assertGreater(conflict.route_total_variation, 0.99)
        self.assertGreater(conflict.path_conflict, 0.99)
        self.assertGreater(conflict.risk, aligned.risk)

    def test_counterfactual_equal_raw_differences_have_zero_risk(self):
        risk, diagnostics = counterfactual_phrase_risk(
            [0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7],
            [0.7, 0.6, 0.5],
            torch.eye(3, dtype=torch.float64),
            query_impact=0.8,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        self.assertTrue(torch.allclose(risk, torch.zeros(3, dtype=torch.float64)))
        self.assertEqual(diagnostics["beta"], 0.0)

    def test_counterfactual_multi_cluster_dependency_has_positive_risk(self):
        risk, diagnostics = counterfactual_phrase_risk(
            [0.9, 0.8, 0.7, 0.6],
            [1.0, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            torch.eye(4, dtype=torch.float64),
            query_impact=1.0,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        self.assertGreater(diagnostics["breadth"], 0.0)
        self.assertGreater(diagnostics["dispersion"], 0.0)
        self.assertGreater(diagnostics["beta"], 0.0)
        self.assertGreater(float(risk.max().item()), 0.0)

    def test_greedy_recomputes_soft_redundancy_and_clips_negative_cosine(self):
        kernel = torch.tensor(
            [[1.0, 0.99, -0.5], [0.99, 1.0, 0.0], [-0.5, 0.0, 1.0]],
            dtype=torch.float64,
        )
        greedy = greedy_evidence_competition(
            [0.9, 0.85, 0.8],
            kernel,
            ["a", "b", "c"],
            [0, 1, 2],
            max_k=3,
            redundancy_enabled=True,
            lambda_redundancy=0.2,
        )
        self.assertEqual(greedy.indices, (0, 2, 1))
        self.assertEqual(greedy.redundancies[1], 0.0)
        self.assertGreater(greedy.redundancies[2], 0.9)

    def test_raw_entropy_is_not_normalized_and_null_boundaries_are_exact(self):
        result = select_adaptive_k(
            [1.0, 1.0, 1.0, 1.0],
            chi=1.0,
            temperature=0.2,
            eta=0.5,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
            adaptive_enabled=True,
            entropy_enabled=True,
            maximum_gap_enabled=True,
            consistency_entropy_enabled=True,
            fixed_k=None,
            null_gate_enabled=False,
            null_strengths=(),
            null_low_threshold=None,
            null_high_threshold=None,
        )
        self.assertAlmostEqual(result.raw_entropy, math.log(4), places=12)
        self.assertEqual(result.entropy_k, 4)
        self.assertEqual(result.selected_k, 4)
        self.assertEqual(result.gap_strength, 0.0)

        self.assertAlmostEqual(
            empirical_null_probability(0.5, [0.2, 0.5, 0.5, 0.9]), 0.8
        )
        low_boundary = select_adaptive_k(
            [0.5, 0.4, 0.3, 0.2],
            chi=1.0,
            temperature=0.2,
            eta=0.5,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
            adaptive_enabled=False,
            entropy_enabled=True,
            maximum_gap_enabled=True,
            consistency_entropy_enabled=True,
            fixed_k=4,
            null_gate_enabled=True,
            null_strengths=[0.0, 0.0, 0.0],
            null_low_threshold=0.25,
            null_high_threshold=0.75,
        )
        self.assertEqual(low_boundary.null_probability, 0.25)
        self.assertEqual(low_boundary.selected_k, 4)

    def test_consistency_factor_scales_raw_entropy_exactly(self):
        result = select_adaptive_k(
            [1.0, 1.0, 1.0, 1.0],
            chi=0.2,
            temperature=0.2,
            eta=0.5,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
            adaptive_enabled=True,
            entropy_enabled=True,
            maximum_gap_enabled=False,
            consistency_entropy_enabled=True,
            fixed_k=None,
            null_gate_enabled=False,
            null_strengths=(),
            null_low_threshold=None,
            null_high_threshold=None,
        )
        self.assertAlmostEqual(
            result.corrected_entropy,
            result.raw_entropy * (0.5 + 0.5 * 0.2),
            places=12,
        )

    def test_missing_offsets_do_not_claim_evidence_merge(self):
        items = [
            EvidenceItem("a", "one", 0.9, 0.9, 1, ChunkMetadata()),
            EvidenceItem("b", "two", 0.8, 0.8, 2, ChunkMetadata()),
        ]
        merged, diagnostics = merge_structured_evidence(items)
        self.assertEqual(len(merged), 2)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["reason"], "missing_structured_offsets")

    def test_untrustworthy_fractional_or_empty_metadata_cannot_enable_merge(self):
        fractional = ChunkMetadata.from_mapping(
            {
                "document_id": "doc",
                "chunk_index": 0.9,
                "char_start": 0,
                "char_end": 5,
            }
        )
        self.assertIsNone(fractional.chunk_index)
        self.assertFalse(fractional.has_merge_coordinates)
        empty_span_items = [
            EvidenceItem(
                "a", "first", 0.9, 0.9, 1, ChunkMetadata("doc", 0, None, None, None, 0, 0)
            ),
            EvidenceItem(
                "b", "second", 0.8, 0.8, 2, ChunkMetadata("doc", 1, None, None, None, 0, 0)
            ),
        ]
        merged, diagnostics = merge_structured_evidence(empty_span_items)
        self.assertEqual(len(merged), 2)
        self.assertFalse(diagnostics["applied"])

    def test_adjacent_overlapping_structured_chunks_merge_without_reordering(self):
        items = [
            EvidenceItem(
                "a",
                "abcde",
                0.9,
                0.9,
                1,
                ChunkMetadata(
                    document_id="doc",
                    chunk_index=0,
                    section="s",
                    page_start=1,
                    page_end=1,
                    char_start=0,
                    char_end=5,
                ),
            ),
            EvidenceItem(
                "b",
                "defgh",
                0.8,
                0.7,
                2,
                ChunkMetadata(
                    document_id="doc",
                    chunk_index=1,
                    section="s",
                    page_start=1,
                    page_end=2,
                    char_start=3,
                    char_end=8,
                ),
            ),
        ]
        merged, diagnostics = merge_structured_evidence(items)
        self.assertTrue(diagnostics["applied"])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "abcdefgh")
        self.assertEqual(merged[0].merged_chunk_ids, ("a", "b"))

    def test_three_consecutive_structured_chunks_merge_as_one_interval(self):
        items = [
            EvidenceItem(
                "a",
                "abcde",
                0.9,
                0.9,
                1,
                ChunkMetadata("doc", 0, "s", 1, 1, 0, 5),
            ),
            EvidenceItem(
                "b",
                "defgh",
                0.8,
                0.8,
                2,
                ChunkMetadata("doc", 1, "s", 1, 1, 3, 8),
            ),
            EvidenceItem(
                "c",
                "ghijk",
                0.7,
                0.7,
                3,
                ChunkMetadata("doc", 2, "s", 1, 2, 6, 11),
            ),
        ]
        merged, diagnostics = merge_structured_evidence(items)
        self.assertTrue(diagnostics["applied"])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "abcdefghijk")
        self.assertEqual(merged[0].merged_chunk_ids, ("a", "b", "c"))
        self.assertEqual(merged[0].metadata.char_end, 11)

    def test_null_high_threshold_is_inclusive(self):
        baseline = select_adaptive_k(
            [0.7, 0.5, 0.3, 0.1],
            chi=1.0,
            temperature=0.2,
            eta=0.5,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
            adaptive_enabled=False,
            entropy_enabled=True,
            maximum_gap_enabled=True,
            consistency_entropy_enabled=True,
            fixed_k=4,
            null_gate_enabled=False,
            null_strengths=(),
            null_low_threshold=None,
            null_high_threshold=None,
        )
        strength = baseline.evidence_strength
        gated = select_adaptive_k(
            [0.7, 0.5, 0.3, 0.1],
            chi=1.0,
            temperature=0.2,
            eta=0.5,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
            adaptive_enabled=False,
            entropy_enabled=True,
            maximum_gap_enabled=True,
            consistency_entropy_enabled=True,
            fixed_k=4,
            null_gate_enabled=True,
            null_strengths=[strength, strength, 0.0],
            null_low_threshold=0.25,
            null_high_threshold=0.75,
        )
        self.assertEqual(gated.null_probability, 0.75)
        self.assertEqual(gated.null_region, "null_like")
        self.assertEqual(gated.selected_k, 1)


class AmbiCalEndToEndTests(unittest.TestCase):
    def test_translation_token_accounting_failure_is_never_soft_fallback(self):
        retriever = AmbiCalRetriever.__new__(AmbiCalRetriever)
        retriever.translator = UnaccountedTranslator()
        with self.assertRaisesRegex(
            TokenUsageUnavailableError, "missing provider metrics"
        ):
            retriever._translate("question", "English", "Vietnamese")

    def test_invalid_adapter_diagnostics_are_omitted_without_breaking_translation(self):
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=FakeEmbeddingRetriever(
                {
                    "source question": [1.0, 0.0],
                    "target question": [0.0, 1.0],
                }
            ),
            translator=UnsafeDiagnosticsTranslator(),
            config=method_config(),
        )
        result = retriever.retrieve(
            "source question", source_language="English", target_language="Indonesian"
        )
        self.assertTrue(result.translation_active)
        self.assertEqual(
            result.diagnostics["translation"]["diagnostics_status"],
            "invalid_omitted",
        )
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_complete_online_path_with_precomputed_profile_is_json_safe(self):
        phrase = CounterfactualPhrase(
            text="pressure",
            start=7,
            end=15,
            token_start=1,
            token_end=1,
            counterfactual_query="reduce loss",
            content_token_count=1,
            source="content_word",
        )
        switches = AmbiCalSwitches(merge_structured_evidence=False)
        config = method_config(
            switches=switches,
            fixed_k=None,
            lambda_false_positive=0.2,
            lambda_context=0.5,
            lambda_redundancy=0.1,
            context_gate_threshold=0.0,
            null_low_threshold=0.25,
            null_high_threshold=0.75,
            candidate_top_n=3,
            max_k=3,
            gate_top_m=3,
        )
        profile = AmbiCalProfile(
            false_positive_priors=FalsePositivePriors(
                source={"a": 0.2},
                target={"b": 0.2},
                unseen_source=0.1,
                unseen_target=0.1,
            ),
            null_strengths=(0.1, 0.3, 0.7, 0.9),
        )
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=FakeEmbeddingRetriever(
                {
                    "reduce pressure loss": [1.0, 0.0],
                    "kurangi kehilangan tekanan": [0.0, 1.0],
                    "reduce loss": [0.7, 0.7],
                    "kurangi kehilangan": [0.6, 0.8],
                }
            ),
            translator=FakeTranslator(
                {
                    "reduce pressure loss": "kurangi kehilangan tekanan",
                    "reduce loss": "kurangi kehilangan",
                }
            ),
            config=config,
            profile=profile,
            phrase_generator=FakePhraseGenerator([phrase]),
        )
        result = retriever.retrieve(
            "reduce pressure loss",
            source_language="English",
            target_language="Indonesian",
        )
        payload = result.to_dict()
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.assertTrue(result.translation_active)
        self.assertTrue(result.diagnostics["context_correction"]["executed"])
        self.assertIsNotNone(result.diagnostics["adaptive_k"]["null_probability"])
        self.assertEqual(
            payload["selected_chunk_count_before_merge"], result.selected_k
        )
        self.assertEqual(
            payload["evidence_count_after_merge"], len(result.evidence)
        )

    def test_profile_round_trip_is_json_serialisable(self):
        profile = AmbiCalProfile(
            false_positive_priors=FalsePositivePriors(
                source={"a": 0.2},
                target={"b": 0.3},
                unseen_source=0.1,
                unseen_target=0.15,
            ),
            null_strengths=(0.1, 0.4, 0.8),
            utility_signature="signature",
        )
        encoded = json.loads(json.dumps(profile.to_dict(), allow_nan=False))
        restored = AmbiCalProfile.from_dict(encoded)
        self.assertEqual(restored, profile)

    def test_one_dimensional_embedder_output_is_supported(self):
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=FakeOneDimensionalEmbeddingRetriever(
                {"source question": [1.0, 0.0]}
            ),
            config=method_config(
                switches=AmbiCalSwitches(
                    translation_branch=False,
                    false_positive_prior=False,
                    context_correction=False,
                    soft_redundancy=False,
                    null_gate=False,
                    adaptive_k=False,
                    merge_structured_evidence=False,
                )
            ),
        )
        result = retriever.retrieve(
            "source question", source_language="English", target_language="Indonesian"
        )
        self.assertEqual(result.selected_k, 2)

    def test_same_language_query_uses_one_source_branch(self):
        collection = FakeCollection()
        retriever = AmbiCalRetriever(
            collection=collection,
            embedding_retriever=FakeEmbeddingRetriever(
                {"source question": [1.0, 0.0]}
            ),
            translator=FakeTranslator({}),
            config=method_config(),
        )
        result = retriever.retrieve(
            "source question", source_language="English", target_language="english"
        )
        self.assertFalse(result.translation_active)
        self.assertIsNone(result.translated_query)
        self.assertEqual(result.diagnostics["translation"]["status"], "identity_source_only")
        self.assertEqual(collection.query_count, 1)

    def test_non_normalized_collection_vector_is_rejected(self):
        collection = FakeCollection()
        collection.rows["a"]["embedding"] = [10.0, 0.0]
        retriever = AmbiCalRetriever(
            collection=collection,
            embedding_retriever=FakeEmbeddingRetriever(
                {"source question": [1.0, 0.0]}
            ),
            config=method_config(
                switches=AmbiCalSwitches(
                    translation_branch=False,
                    false_positive_prior=False,
                    context_correction=False,
                    soft_redundancy=False,
                    null_gate=False,
                    adaptive_k=False,
                    merge_structured_evidence=False,
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "unit-normalized"):
            retriever.retrieve(
                "source question",
                source_language="English",
                target_language="Indonesian",
            )

    def test_all_optional_scoring_components_can_degrade_cleanly(self):
        switches = AmbiCalSwitches(
            translation_branch=False,
            robust_normalization=False,
            branch_confidence=False,
            translation_consistency=False,
            ranking_overlap=False,
            dynamic_branch_weights=False,
            false_positive_prior=False,
            context_correction=False,
            context_gate=False,
            soft_redundancy=False,
            consistency_entropy=False,
            maximum_gap=False,
            entropy_candidate=True,
            null_gate=False,
            adaptive_k=False,
            merge_structured_evidence=False,
        )
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=FakeEmbeddingRetriever(
                {"source question": [1.0, 0.0]}
            ),
            translator=FakeTranslator({}),
            config=method_config(switches=switches, fixed_k=2),
        )
        result = retriever.retrieve(
            "source question", source_language="English", target_language="Indonesian"
        )
        self.assertFalse(result.translation_active)
        self.assertEqual(result.diagnostics["source_normalization"]["mode"], "fixed_cosine_map")
        self.assertEqual(result.diagnostics["source_shape_initial"]["confidence"], 1.0)
        self.assertEqual(result.diagnostics["combined_consistency_signal_count"], 0)
        self.assertEqual(result.selected_k, 2)

    def test_explicit_profile_signature_mismatch_is_rejected(self):
        config = method_config()
        with self.assertRaisesRegex(ValueError, "utility signature"):
            AmbiCalRetriever(
                collection=FakeCollection(),
                embedding_retriever=FakeEmbeddingRetriever(
                    {"source question": [1.0, 0.0]}
                ),
                config=config,
                profile=AmbiCalProfile(utility_signature="wrong"),
            )

    def test_profile_is_required_when_online_prior_is_enabled(self):
        switches = AmbiCalSwitches(
            false_positive_prior=True,
            context_correction=False,
            soft_redundancy=False,
            null_gate=False,
            adaptive_k=False,
            merge_structured_evidence=False,
        )
        with self.assertRaisesRegex(ValueError, "precomputed profile"):
            AmbiCalRetriever(
                collection=FakeCollection(),
                embedding_retriever=FakeEmbeddingRetriever(
                    {"source question": [1.0, 0.0]}
                ),
                config=method_config(
                    switches=switches,
                    lambda_false_positive=0.2,
                ),
            )

    def test_false_positive_prior_is_a_soft_branch_penalty(self):
        switches = AmbiCalSwitches(
            translation_branch=False,
            false_positive_prior=True,
            context_correction=False,
            soft_redundancy=False,
            null_gate=False,
            adaptive_k=False,
            merge_structured_evidence=False,
        )
        profile = AmbiCalProfile(
            false_positive_priors=FalsePositivePriors(
                source={"a": 0.9},
                target={},
                unseen_source=0.1,
                unseen_target=0.1,
            )
        )
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=FakeEmbeddingRetriever(
                {"source question": [1.0, 0.0]}
            ),
            config=method_config(
                switches=switches,
                lambda_false_positive=0.2,
            ),
            profile=profile,
        )
        result = retriever.retrieve(
            "source question", source_language="English", target_language="Indonesian"
        )
        rows = {row["chunk_id"]: row for row in result.diagnostics["candidates"]}
        self.assertAlmostEqual(
            rows["a"]["source_branch_support"],
            rows["a"]["source_normalized_score"] - 0.2 * 0.9,
        )
        self.assertIn("a", rows)

    def test_dual_union_rescores_both_branches_and_realigns_chroma_ids(self):
        collection = FakeCollection()
        embedder = FakeEmbeddingRetriever(
            {"source question": [1.0, 0.0], "target question": [0.0, 1.0]}
        )
        translator = FakeTranslator({"source question": "target question"})
        retriever = AmbiCalRetriever(
            collection=collection,
            embedding_retriever=embedder,
            translator=translator,
            config=method_config(),
        )
        result = retriever.retrieve(
            "source question", source_language="English", target_language="Indonesian"
        )
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics["candidate_ids"], ["a", "c", "b"])
        self.assertEqual(diagnostics["ranking_overlap"], 0.5)
        rows = {row["chunk_id"]: row for row in diagnostics["candidates"]}
        self.assertGreater(rows["a"]["target_raw_score"], 0.0)
        self.assertGreater(rows["b"]["source_raw_score"], 0.0)
        self.assertEqual(collection.query_count, 2)
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_translation_failure_degrades_to_source_only(self):
        collection = FakeCollection()
        embedder = FakeEmbeddingRetriever({"source question": [1.0, 0.0]})
        translator = FakeTranslator({}, fail_on={"source question"})
        retriever = AmbiCalRetriever(
            collection=collection,
            embedding_retriever=embedder,
            translator=translator,
            config=method_config(),
        )
        result = retriever.retrieve(
            "source question", source_language="English", target_language="Indonesian"
        )
        self.assertFalse(result.translation_active)
        self.assertEqual(result.diagnostics["alpha_source_initial"], 1.0)
        self.assertEqual(result.diagnostics["alpha_target_initial"], 0.0)
        self.assertEqual(collection.query_count, 1)

    def test_counterfactual_target_is_retranslated_and_never_requeries_collection(self):
        phrase = CounterfactualPhrase(
            text="pressure",
            start=7,
            end=15,
            token_start=1,
            token_end=1,
            counterfactual_query="reduce loss",
            content_token_count=1,
            source="content_word",
        )
        phrase_generator = FakePhraseGenerator([phrase])
        translator = FakeTranslator(
            {
                "reduce pressure loss": "kurangi kehilangan tekanan",
                "reduce loss": "kurangi kehilangan",
            }
        )
        embedder = FakeEmbeddingRetriever(
            {
                "reduce pressure loss": [1.0, 0.0],
                "kurangi kehilangan tekanan": [0.0, 1.0],
                "reduce loss": [0.7, 0.7],
                "kurangi kehilangan": [0.6, 0.8],
            }
        )
        switches = AmbiCalSwitches(
            false_positive_prior=False,
            context_correction=True,
            context_gate=False,
            soft_redundancy=False,
            null_gate=False,
            adaptive_k=False,
            merge_structured_evidence=False,
        )
        config = method_config(
            switches=switches,
            lambda_context=0.5,
        )
        collection = FakeCollection()
        retriever = AmbiCalRetriever(
            collection=collection,
            embedding_retriever=embedder,
            translator=translator,
            config=config,
            phrase_generator=phrase_generator,
        )
        result = retriever.retrieve(
            "reduce pressure loss",
            source_language="English",
            target_language="Indonesian",
        )
        translated_inputs = [call[0] for call in translator.calls]
        self.assertIn("reduce loss", translated_inputs)
        self.assertEqual(len(translated_inputs), 2)
        self.assertEqual(collection.query_count, 2)
        self.assertTrue(result.diagnostics["context_correction"]["executed"])
        self.assertEqual(
            result.diagnostics["context_correction"]["selection_rule"],
            "source_impact_then_nonoverlap_then_retranslate",
        )
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)

    def test_only_source_selected_counterfactuals_are_retranslated(self):
        strong = CounterfactualPhrase(
            text="pressure",
            start=7,
            end=15,
            token_start=1,
            token_end=1,
            counterfactual_query="strong deletion",
            content_token_count=1,
            source="content_word",
        )
        weak = CounterfactualPhrase(
            text="loss",
            start=16,
            end=20,
            token_start=2,
            token_end=2,
            counterfactual_query="weak deletion",
            content_token_count=1,
            source="content_word",
        )
        translator = FakeTranslator(
            {
                "source question": "target question",
                "strong deletion": "strong target",
                "weak deletion": "weak target",
            }
        )
        embedder = FakeEmbeddingRetriever(
            {
                "source question": [1.0, 0.0],
                "target question": [0.0, 1.0],
                "strong deletion": [0.0, 1.0],
                "weak deletion": [0.99, 0.1],
                "strong target": [0.2, 0.8],
                "weak target": [0.8, 0.2],
            }
        )
        switches = AmbiCalSwitches(
            false_positive_prior=False,
            context_correction=True,
            context_gate=False,
            soft_redundancy=False,
            null_gate=False,
            adaptive_k=False,
            merge_structured_evidence=False,
        )
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=embedder,
            translator=translator,
            config=method_config(
                switches=switches,
                lambda_context=0.5,
                max_counterfactual_phrases=1,
            ),
            phrase_generator=FakePhraseGenerator([strong, weak]),
        )
        result = retriever.retrieve(
            "source question",
            source_language="English",
            target_language="Indonesian",
        )
        translated_inputs = [call[0] for call in translator.calls]
        self.assertEqual(translated_inputs, ["source question", "strong deletion"])
        correction = result.diagnostics["context_correction"]
        self.assertEqual(correction["generated_count"], 2)
        self.assertEqual(correction["selected_count"], 1)

    def test_v3_builds_two_independent_views_and_only_promotes_agreed_candidate(self):
        first = CounterfactualPhrase(
            text="pressure",
            start=7,
            end=15,
            token_start=1,
            token_end=1,
            counterfactual_query="remaining context one",
            content_token_count=1,
            source="content_word",
        )
        second = CounterfactualPhrase(
            text="cost",
            start=20,
            end=24,
            token_start=3,
            token_end=3,
            counterfactual_query="remaining context two",
            content_token_count=1,
            source="content_word",
        )
        translator = FakeTranslator(
            {
                "source question": "target question",
                "remaining context one": "target context one",
                "remaining context two": "target context two",
            }
        )
        embedder = FakeEmbeddingRetriever(
            {
                "source question": [1.0, 0.0],
                "target question": [0.0, 1.0],
                "remaining context one": [0.0, 1.0],
                "target context one": [0.0, 1.0],
                "remaining context two": [0.0, 1.0],
                "target context two": [0.0, 1.0],
            }
        )
        switches = AmbiCalSwitches(
            dynamic_branch_weights=False,
            false_positive_prior=False,
            context_correction=True,
            context_gate=False,
            soft_redundancy=False,
            null_gate=False,
            adaptive_k=False,
            merge_structured_evidence=False,
        )
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=embedder,
            translator=translator,
            config=method_config(
                switches=switches,
                method_version="ambical-online-v3",
                lambda_positive=1.0,
                lambda_context=None,
                max_counterfactual_phrases=2,
            ),
            phrase_generator=FakePhraseGenerator([first, second]),
        )
        result = retriever.retrieve(
            "source question",
            source_language="English",
            target_language="Indonesian",
        )
        self.assertEqual(
            [call[0] for call in translator.calls],
            [
                "source question",
                "remaining context one",
                "remaining context two",
            ],
        )
        correction = result.diagnostics["context_correction"]
        self.assertTrue(correction["executed"])
        self.assertEqual(correction["correction_mode"], "positive_context_v3")
        self.assertEqual(correction["valid_phrase_count"], 2)
        self.assertTrue(correction["phrase_agreement"])
        self.assertTrue(correction["promotion_executed"])
        self.assertEqual(correction["promoted_candidate_id"], "b")
        self.assertEqual(correction["view_winner_ids"], ["b", "b"])

    def test_v3_gate_skip_preserves_gate_reason_and_generates_no_phrases(self):
        phrase_generator = FakePhraseGenerator([])
        switches = AmbiCalSwitches(
            translation_branch=False,
            false_positive_prior=False,
            context_correction=True,
            context_gate=True,
            soft_redundancy=False,
            null_gate=False,
            adaptive_k=False,
            merge_structured_evidence=False,
        )
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=FakeEmbeddingRetriever(
                {"source question": [1.0, 0.0]}
            ),
            config=method_config(
                switches=switches,
                method_version="ambical-online-v3",
                lambda_positive=0.2,
                lambda_context=None,
                max_counterfactual_phrases=2,
                context_gate_threshold=1.0,
                gate_v2_bias=-10.0,
                gate_v2_weight_strength=0.0,
                gate_v2_weight_competition=0.0,
                gate_v2_weight_distance=0.0,
                gate_v2_weight_path_conflict=0.0,
                gate_v2_weight_bimodality=0.0,
                gate_v2_weight_branch_disagreement=0.0,
            ),
            phrase_generator=phrase_generator,
        )
        result = retriever.retrieve(
            "source question",
            source_language="English",
            target_language="Indonesian",
        )
        correction = result.diagnostics["context_correction"]
        self.assertFalse(correction["executed"])
        self.assertFalse(correction["promotion_executed"])
        self.assertEqual(correction["promotion_reason"], "gate_not_triggered")
        self.assertEqual(phrase_generator.calls, [])


def _v2_gate_weights():
    return dict(
        bias=-2.5,
        weight_strength=1.0,
        weight_competition=1.5,
        weight_distance=2.0,
        weight_path_conflict=1.0,
        weight_bimodality=2.0,
        weight_branch_disagreement=1.5,
    )


def _v2_config(**overrides):
    base = dict(
        lambda_false_positive=0.2,
        lambda_context=0.5,
        lambda_redundancy=0.15,
        context_gate_threshold=0.2,
        null_low_threshold=0.1,
        null_high_threshold=0.5,
        method_version="ambical-online-v2",
        gate_v2_bias=-2.5,
        gate_v2_weight_strength=1.0,
        gate_v2_weight_competition=1.5,
        gate_v2_weight_distance=2.0,
        gate_v2_weight_path_conflict=1.0,
        gate_v2_weight_bimodality=2.0,
        gate_v2_weight_branch_disagreement=1.5,
    )
    base.update(overrides)
    return AmbiCalConfig(**base)


def _v3_config(**overrides):
    base = dict(
        lambda_false_positive=0.2,
        lambda_context=None,
        lambda_positive=0.2,
        lambda_redundancy=0.15,
        context_gate_threshold=0.2,
        null_low_threshold=0.1,
        null_high_threshold=0.5,
        max_counterfactual_phrases=2,
        method_version="ambical-online-v3",
        gate_v2_bias=-2.5,
        gate_v2_weight_strength=1.0,
        gate_v2_weight_competition=1.5,
        gate_v2_weight_distance=2.0,
        gate_v2_weight_path_conflict=1.0,
        gate_v2_weight_bimodality=2.0,
        gate_v2_weight_branch_disagreement=1.5,
    )
    base.update(overrides)
    return AmbiCalConfig(**base)


class AmbiCalV2FormulaTests(unittest.TestCase):
    def test_v2_requires_gate_weights_when_gate_enabled(self):
        with self.assertRaisesRegex(ValueError, "gate_v2_bias"):
            _v2_config(
                gate_v2_bias=None,
                gate_v2_weight_strength=None,
                gate_v2_weight_competition=None,
                gate_v2_weight_distance=None,
                gate_v2_weight_path_conflict=None,
                gate_v2_weight_bimodality=None,
                gate_v2_weight_branch_disagreement=None,
            ).validate()
        _v2_config().validate()

    def test_v1_rejects_stray_gate_v2_weights(self):
        with self.assertRaisesRegex(ValueError, "require method_version"):
            _v2_config(method_version="ambical-online-v1").validate()

    def test_unknown_method_version_rejected(self):
        with self.assertRaisesRegex(ValueError, "method_version"):
            _v2_config(method_version="ambical-online-v999").validate()

    def test_v2_and_v1_have_different_utility_signatures(self):
        v1 = _v2_config(method_version="ambical-online-v1")
        v1 = AmbiCalConfig(
            **{
                field: value
                for field, value in v1.__dict__.items()
                if not field.startswith("gate_v2_")
            }
        )
        self.assertNotEqual(v1.utility_signature(), _v2_config().utility_signature())

    def test_gate_v2_reaches_threshold_where_v1_formula_cannot(self):
        two_senses = [0.9, 0.88, 0.5, 0.48]
        common = dict(
            source_branch_support=two_senses,
            target_branch_support=None,
            pairwise_cosine=torch.eye(4, dtype=torch.float64),
            chi=1.0,
            top_m=4,
            temperature=1.0,
            threshold=0.2,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        legacy = context_gate(two_senses, **common)
        modern = context_gate_v2(two_senses, **common, **_v2_gate_weights())
        self.assertLess(legacy.risk, 0.2)
        self.assertFalse(legacy.triggered)
        self.assertGreaterEqual(modern.risk, 0.2)
        self.assertTrue(modern.triggered)
        self.assertGreater(modern.bimodality, 0.8)

    def test_gate_v2_bimodality_separates_two_sense_clusters(self):
        two_senses_scores = [0.9, 0.88, 0.5, 0.48]
        smooth_scores = [0.9, 0.8, 0.7, 0.6]
        two_senses = context_gate_v2(
            two_senses_scores,
            source_branch_support=two_senses_scores,
            target_branch_support=None,
            pairwise_cosine=torch.eye(4, dtype=torch.float64),
            chi=1.0,
            top_m=4,
            temperature=1.0,
            threshold=1.0,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
            **_v2_gate_weights(),
        )
        smooth = context_gate_v2(
            smooth_scores,
            source_branch_support=smooth_scores,
            target_branch_support=None,
            pairwise_cosine=torch.eye(4, dtype=torch.float64),
            chi=1.0,
            top_m=4,
            temperature=1.0,
            threshold=1.0,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
            **_v2_gate_weights(),
        )
        self.assertGreater(two_senses.bimodality, smooth.bimodality + 0.5)
        self.assertGreater(two_senses.risk, smooth.risk)
        self.assertFalse(smooth.triggered)

    def test_gate_v2_branch_disagreement_raises_risk(self):
        scores = [0.9, 0.88, 0.5, 0.48]
        common = dict(
            pairwise_cosine=torch.eye(4, dtype=torch.float64),
            chi=0.5,
            top_m=4,
            temperature=1.0,
            threshold=1.0,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        disagree = context_gate_v2(
            scores,
            source_branch_support=[1.0, 0.9, 0.1, 0.0],
            target_branch_support=[0.0, 0.1, 0.9, 1.0],
            **common,
            **_v2_gate_weights(),
        )
        agree = context_gate_v2(
            scores,
            source_branch_support=[1.0, 0.9, 0.1, 0.0],
            target_branch_support=[1.0, 0.9, 0.1, 0.0],
            **common,
            **_v2_gate_weights(),
        )
        self.assertAlmostEqual(agree.branch_disagreement, 0.0, places=7)
        self.assertGreater(disagree.branch_disagreement, 0.5)
        self.assertGreater(disagree.risk, agree.risk)

    def test_symmetric_factors_demote_promote_and_stay_bounded(self):
        factors, diagnostics = symmetric_correction_factors(
            [1.0, -1.0, 0.0, 0.5, 10.0], lambda_context=0.5
        )
        self.assertAlmostEqual(float(factors[0]), 0.5, places=12)
        self.assertAlmostEqual(float(factors[1]), 1.5, places=12)
        self.assertAlmostEqual(float(factors[2]), 1.0, places=12)
        self.assertAlmostEqual(float(factors[3]), 0.75, places=12)
        self.assertAlmostEqual(float(factors[4]), 0.5, places=12)
        self.assertTrue(
            (factors >= 1.0 - 0.5).all() and (factors <= 1.0 + 0.5).all()
        )
        self.assertIn("signed_dependency", diagnostics)
        self.assertIn("factors", diagnostics)

    def test_counterfactual_diagnostics_expose_signed_dependency(self):
        risk, diagnostics = counterfactual_phrase_risk(
            [0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7],
            [0.7, 0.6, 0.5],
            torch.eye(3, dtype=torch.float64),
            query_impact=0.8,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        self.assertEqual(len(diagnostics["signed_dependency"]), 3)
        for value in diagnostics["signed_dependency"]:
            self.assertAlmostEqual(value, 0.0, places=7)

    def test_v1_context_gate_diagnostics_remain_byte_comparable(self):
        gate = context_gate(
            [0.8, 0.8],
            [0.8, 0.8],
            [0.8, 0.8],
            torch.eye(2, dtype=torch.float64),
            chi=1.0,
            top_m=2,
            temperature=1.0,
            threshold=0.0,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
        )
        payload = gate.diagnostics()
        self.assertEqual(
            sorted(payload),
            [
                "competition",
                "context_split",
                "evidence_strength",
                "observed_indices",
                "path_conflict",
                "risk",
                "route_total_variation",
                "triggered",
                "weighted_distance",
            ],
        )
        modern = context_gate_v2(
            [0.9, 0.88, 0.5, 0.48],
            [0.9, 0.88, 0.5, 0.48],
            None,
            torch.eye(4, dtype=torch.float64),
            chi=1.0,
            top_m=4,
            temperature=1.0,
            threshold=1.0,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
            **_v2_gate_weights(),
        )
        self.assertIn("bimodality", modern.diagnostics())
        self.assertIn("linear_predictor", modern.diagnostics())


class AmbiCalV3PositivePromotionTests(unittest.TestCase):
    def test_v3_requires_one_new_weight_and_exactly_two_phrases(self):
        _v3_config().validate()
        _v3_config(method_version="ambical-online-v3.1").validate()
        with self.assertRaisesRegex(ValueError, "lambda_positive"):
            _v3_config(lambda_positive=None).validate()
        with self.assertRaisesRegex(ValueError, "lambda_context"):
            _v3_config(lambda_context=0.2).validate()
        with self.assertRaisesRegex(ValueError, "max_counterfactual_phrases=2"):
            _v3_config(max_counterfactual_phrases=3).validate()

    def test_v31_selector_rejects_long_deletions_and_separates_regions(self):
        query = (
            "What tractor power is associated with the tractor-mounted straw "
            "baler, not with a rice seeding machine?"
        )
        generated = CounterfactualPhraseGenerator().generate(query)
        by_text = {phrase.text.casefold(): phrase for phrase in generated}
        long_phrase = by_text[
            "tractor power is associated with the tractor-mounted straw baler"
        ]
        tractor_power = by_text["tractor power"]
        straw_baler = by_text["straw baler"]
        seeding_machine = by_text["seeding machine"]
        selected = select_compact_diverse_phrases(
            [
                (long_phrase, 0.99),
                (tractor_power, 0.80),
                (straw_baler, 0.75),
                (seeding_machine, 0.70),
            ],
            2,
        )
        selected_phrases = [phrase for phrase, _ in selected]
        self.assertNotIn(long_phrase, selected_phrases)
        self.assertEqual(selected_phrases, [tractor_power, straw_baler])
        self.assertGreater(
            straw_baler.token_start - tractor_power.token_end - 1,
            0,
        )

    def test_v3_field_does_not_change_legacy_diagnostics(self):
        self.assertNotIn("lambda_positive", _v2_config().diagnostics())
        self.assertIn("lambda_positive", _v3_config().diagnostics())

    def test_agreed_non_top_candidate_gets_the_only_positive_change(self):
        original = torch.tensor([0.70, 0.68, 0.20], dtype=torch.float64)
        promoted, diagnostics = positive_context_promotion(
            original,
            ([0.10, 0.95, 0.10], [0.15, 0.90, 0.10]),
            lambda_positive=0.5,
            epsilon=1e-8,
        )
        self.assertTrue(diagnostics["promotion_executed"])
        self.assertEqual(diagnostics["view_winners"], [1, 1])
        self.assertEqual(diagnostics["promoted_candidate_index"], 1)
        self.assertGreater(float(promoted[1]), float(original[1]))
        self.assertEqual(float(promoted[0]), float(original[0]))
        self.assertEqual(float(promoted[2]), float(original[2]))
        self.assertTrue(bool(torch.all(promoted >= original)))

    def test_phrase_disagreement_never_changes_any_score(self):
        original = torch.tensor([0.70, 0.68, 0.20], dtype=torch.float64)
        promoted, diagnostics = positive_context_promotion(
            original,
            ([0.10, 0.95, 0.10], [1.00, 0.10, 0.10]),
            lambda_positive=1.0,
            epsilon=1e-8,
        )
        self.assertEqual(diagnostics["promotion_reason"], "phrase_disagreement")
        self.assertFalse(diagnostics["promotion_executed"])
        self.assertTrue(torch.equal(promoted, original))

    def test_one_valid_phrase_is_insufficient(self):
        original = torch.tensor([0.70, 0.68, 0.20], dtype=torch.float64)
        promoted, diagnostics = positive_context_promotion(
            original,
            ([0.10, 0.95, 0.10],),
            lambda_positive=1.0,
            epsilon=1e-8,
        )
        self.assertEqual(
            diagnostics["promotion_reason"], "insufficient_valid_phrases"
        )
        self.assertFalse(diagnostics["promotion_executed"])
        self.assertTrue(torch.equal(promoted, original))

    def test_agreement_with_existing_top_is_a_noop(self):
        original = torch.tensor([0.70, 0.68, 0.20], dtype=torch.float64)
        promoted, diagnostics = positive_context_promotion(
            original,
            ([1.00, 0.10, 0.10], [0.95, 0.15, 0.10]),
            lambda_positive=1.0,
            epsilon=1e-8,
        )
        self.assertEqual(diagnostics["promotion_reason"], "aligned_noop")
        self.assertFalse(diagnostics["promotion_executed"])
        self.assertTrue(torch.equal(promoted, original))




class AmbiCalV4RerankerPromotionTests(unittest.TestCase):
    @staticmethod
    def _v4_config(**overrides):
        values = dict(
            method_version="ambical-online-v4",
            lambda_positive=0.5,
            reranker_score_threshold=0.5,
            reranker_top_m=3,
            fixed_k=3,
            gate_v2_bias=0.0,
            gate_v2_weight_strength=0.0,
            gate_v2_weight_competition=0.0,
            gate_v2_weight_distance=0.0,
            gate_v2_weight_path_conflict=0.0,
            gate_v2_weight_bimodality=0.0,
            gate_v2_weight_branch_disagreement=0.0,
            switches=AmbiCalSwitches(
                false_positive_prior=False,
                context_correction=True,
                context_gate=False,
                soft_redundancy=False,
                null_gate=False,
                adaptive_k=False,
                merge_structured_evidence=False,
            ),
        )
        values.update(overrides)
        return AmbiCalConfig(**values)

    def test_v4_config_requires_reranker_fields_and_rejects_lambda_context(self):
        self._v4_config().validate()
        with self.assertRaisesRegex(ValueError, "reranker"):
            self._v4_config(reranker_score_threshold=None, reranker_top_m=None).validate()
        with self.assertRaisesRegex(ValueError, "lambda_context"):
            self._v4_config(lambda_context=0.5).validate()
        with self.assertRaisesRegex(ValueError, "reranker_"):
            self._v4_config(
                method_version="ambical-online-v3",
                reranker_score_threshold=0.5,
            ).validate()

    def test_v4_fields_change_only_v4_diagnostics_and_signature(self):
        v3 = method_config(
            method_version="ambical-online-v3",
            lambda_positive=0.2,
            max_counterfactual_phrases=2,
        )
        self.assertNotIn("reranker_score_threshold", v3.diagnostics())
        self.assertNotIn("reranker_top_m", v3.diagnostics())
        v4_a = self._v4_config(reranker_score_threshold=0.4)
        v4_b = self._v4_config(reranker_score_threshold=0.6)
        self.assertIn("lambda_positive", v4_a.diagnostics())
        self.assertNotEqual(v4_a.utility_signature(), v4_b.utility_signature())

    def test_v4_requires_reranker_instance_when_correction_enabled(self):
        with self.assertRaisesRegex(ValueError, "cross-encoder reranker"):
            AmbiCalRetriever(
                collection=FakeCollection(),
                embedding_retriever=FakeEmbeddingRetriever({"q": [1.0, 0.0]}),
                config=self._v4_config(),
            )

    def test_reranker_promotion_noop_when_nothing_reaches_threshold(self):
        original = [0.9, 0.6, 0.3]
        relevance = [0.2, 0.4, 0.1]
        promoted, diagnostics = reranker_promotion(
            original,
            relevance,
            [0, 1, 2],
            lambda_positive=0.5,
            threshold=0.5,
            top_m=3,
            epsilon=1e-8,
        )
        self.assertEqual(promoted.tolist(), [0.9, 0.6, 0.3])
        self.assertFalse(diagnostics["promotion_executed"])
        self.assertFalse(diagnostics["top1_changed"])
        self.assertEqual(len(diagnostics["considered"]), 3)

    def test_reranker_promotion_applies_bounded_bonus_within_window(self):
        original = [0.9, 0.6, 0.3]
        relevance = [0.2, 0.9, 0.0]
        promoted, diagnostics = reranker_promotion(
            original,
            relevance,
            [0, 1, 2],
            lambda_positive=0.5,
            threshold=0.5,
            top_m=2,
            epsilon=1e-8,
        )
        # index 1: margin=(0.9-0.5)/0.5=0.8, bonus=0.5*(1-0.6)*0.8=0.16
        self.assertAlmostEqual(float(promoted[1]), 0.6 + 0.16, places=9)
        self.assertEqual(float(promoted[0]), 0.9)
        self.assertEqual(float(promoted[2]), 0.3)
        self.assertTrue(diagnostics["promotion_executed"])
        self.assertEqual(diagnostics["promoted_candidate_indices"], [1])
        self.assertFalse(diagnostics["top1_changed"])
        # top_m=2 keeps index 2 out of the window even if relevant
        promoted2, _ = reranker_promotion(
            original,
            [0.2, 0.0, 0.99],
            [0, 1, 2],
            lambda_positive=0.5,
            threshold=0.5,
            top_m=2,
            epsilon=1e-8,
        )
        self.assertEqual(float(promoted2[2]), 0.3)

    def test_reranker_promotion_bonus_never_exceeds_bound(self):
        original = [0.4, 0.2]
        relevance = [1.0, 1.0]
        promoted, _ = reranker_promotion(
            original,
            relevance,
            [0, 1],
            lambda_positive=0.5,
            threshold=0.0,
            top_m=2,
            epsilon=1e-8,
        )
        self.assertLessEqual(float(promoted[0]), 0.4 + 0.5 * 0.6 + 1e-9)
        self.assertLessEqual(float(promoted[1]), 0.2 + 0.5 * 0.8 + 1e-9)


    def test_v4_integration_promotes_cross_encoder_relevant_candidate(self):
        class FakeReranker:
            def __init__(self):
                self.calls = []

            def __call__(self, pairs):
                self.calls.extend(pairs)
                scores = []
                for query, text in pairs:
                    scores.append(0.95 if "target evidence" in text else 0.1)
                return scores

        embedder = FakeEmbeddingRetriever(
            {
                "source question": [1.0, 0.0],
                "target question": [0.1, 0.9],
            }
        )
        collection = FakeCollection()
        reranker = FakeReranker()
        retriever = AmbiCalRetriever(
            collection=collection,
            embedding_retriever=embedder,
            translator=FakeTranslator(
                {"source question": "target question"}
            ),
            config=self._v4_config(),
            reranker=reranker,
        )
        result = retriever.retrieve(
            "source question",
            source_language="English",
            target_language="Indonesian",
        )
        correction = result.diagnostics["context_correction"]
        self.assertEqual(correction["correction_mode"], "reranker_promote_v4")
        self.assertEqual(correction["executed"], True)
        self.assertTrue(reranker.calls)
        promotion = correction
        self.assertIn("promoted_candidate_indices", promotion)
        texts = [chunk[1] for chunk in reranker.calls]
        self.assertTrue(any("target evidence" in text for text in texts))
        self.assertEqual(len(reranker.calls), 3)

    def test_v4_uses_v2_context_gate(self):
        switches = AmbiCalSwitches(
            false_positive_prior=False,
            context_correction=True,
            context_gate=True,
            soft_redundancy=False,
            null_gate=False,
            adaptive_k=False,
            merge_structured_evidence=False,
        )
        config = self._v4_config(
            switches=switches,
            context_gate_threshold=0.5,
            gate_v2_bias=20.0,
        )
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=FakeEmbeddingRetriever(
                {"source question": [1.0, 0.0], "target question": [0.9, 0.1]}
            ),
            translator=FakeTranslator({"source question": "target question"}),
            config=config,
            reranker=lambda pairs: [0.1] * len(pairs),
        )
        result = retriever.retrieve(
            "source question",
            source_language="English",
            target_language="Indonesian",
        )
        gate = result.diagnostics["context_gate"]
        self.assertTrue(gate["triggered"])
        self.assertIn("linear_predictor", gate)

class AmbiCalV5TopKCounterfactualTests(unittest.TestCase):
    @staticmethod
    def _v5_config(**overrides):
        values = dict(
            method_version="ambical-online-v5",
            lambda_positive=0.2,
            counterfactual_promotion_top_k=3,
            max_counterfactual_phrases=2,
            fixed_k=3,
            gate_v2_bias=0.0,
            gate_v2_weight_strength=0.0,
            gate_v2_weight_competition=0.0,
            gate_v2_weight_distance=0.0,
            gate_v2_weight_path_conflict=0.0,
            gate_v2_weight_bimodality=0.0,
            gate_v2_weight_branch_disagreement=0.0,
            switches=AmbiCalSwitches(
                false_positive_prior=False,
                context_correction=True,
                context_gate=False,
                soft_redundancy=False,
                null_gate=False,
                adaptive_k=False,
                merge_structured_evidence=False,
            ),
        )
        values.update(overrides)
        return AmbiCalConfig(**values)

    def test_v5_requires_exactly_two_phrases_and_top_k(self):
        self._v5_config().validate()
        with self.assertRaisesRegex(ValueError, "counterfactual_promotion_top_k"):
            self._v5_config(counterfactual_promotion_top_k=None).validate()
        with self.assertRaisesRegex(ValueError, "max_counterfactual_phrases=2"):
            self._v5_config(max_counterfactual_phrases=3).validate()

    def test_each_view_promotes_only_its_top_three(self):
        original = torch.tensor([0.70, 0.60, 0.50, 0.40], dtype=torch.float64)
        views = (
            [0.70, 0.80, 0.60, 0.10],
            [0.65, 0.75, 0.55, 0.20],
        )
        promoted, diagnostics = topk_counterfactual_promotion(
            original,
            views,
            lambda_positive=0.2,
            promotion_top_k=3,
            epsilon=1e-8,
        )
        self.assertEqual(len(diagnostics["per_view_topk_indices"]), 2)
        self.assertTrue(
            all(len(indices) == 3 for indices in diagnostics["per_view_topk_indices"])
        )
        self.assertEqual(float(promoted[3]), float(original[3]))
        self.assertGreater(float(promoted[1]), float(original[1]))
        self.assertTrue(diagnostics["promotion_executed"])

    def test_two_view_support_accumulates_without_any_demotion(self):
        original = torch.tensor([0.60, 0.59, 0.10], dtype=torch.float64)
        promoted, diagnostics = topk_counterfactual_promotion(
            original,
            ([0.20, 1.00, 0.10], [0.20, 1.00, 0.10]),
            lambda_positive=0.2,
            promotion_top_k=1,
            epsilon=1e-8,
        )
        self.assertTrue(torch.all(promoted >= original))
        self.assertEqual(diagnostics["per_view_topk_indices"], [[1], [1]])
        self.assertEqual(diagnostics["top1_after"], 1)
        self.assertTrue(diagnostics["top1_changed"])

    def test_v5_retriever_runs_two_retranslated_views_before_fixed_k(self):
        first = CounterfactualPhrase(
            text="pressure",
            start=0,
            end=8,
            token_start=0,
            token_end=0,
            counterfactual_query="remaining context one",
            content_token_count=1,
            source="content_word",
        )
        second = CounterfactualPhrase(
            text="loss",
            start=20,
            end=24,
            token_start=3,
            token_end=3,
            counterfactual_query="remaining context two",
            content_token_count=1,
            source="content_word",
        )
        translator = FakeTranslator(
            {
                "source question": "target question",
                "remaining context one": "target context one",
                "remaining context two": "target context two",
            }
        )
        embedder = FakeEmbeddingRetriever(
            {
                "source question": [1.0, 0.0],
                "target question": [0.0, 1.0],
                "remaining context one": [0.0, 1.0],
                "target context one": [0.0, 1.0],
                "remaining context two": [0.0, 1.0],
                "target context two": [0.0, 1.0],
            }
        )
        retriever = AmbiCalRetriever(
            collection=FakeCollection(),
            embedding_retriever=embedder,
            translator=translator,
            config=self._v5_config(),
            phrase_generator=FakePhraseGenerator([first, second]),
        )
        result = retriever.retrieve(
            "source question",
            source_language="English",
            target_language="Indonesian",
        )
        correction = result.diagnostics["context_correction"]
        self.assertEqual(
            [call[0] for call in translator.calls],
            ["source question", "remaining context one", "remaining context two"],
        )
        self.assertEqual(correction["correction_mode"], "topk_counterfactual_positive_v5")
        self.assertEqual(correction["valid_phrase_count"], 2)
        self.assertEqual(len(correction["per_view_topk_ids"]), 2)
        self.assertEqual(result.selected_k, 3)




class AmbiCalNullGateThresholdTests(unittest.TestCase):
    """Lock the Null-gate threshold semantics before any retuning experiment."""

    @staticmethod
    def _run(nulls, low, high, utilities=None):
        utilities = utilities or [0.9, 0.89, 0.88, 0.87, 0.86, 0.5, 0.49, 0.48]
        return select_adaptive_k(
            utilities,
            chi=0.8,
            temperature=0.15,
            eta=0.5,
            epsilon=1e-8,
            z_max=8.0,
            iqr_normal_consistency=1.349,
            adaptive_enabled=True,
            entropy_enabled=True,
            maximum_gap_enabled=True,
            consistency_entropy_enabled=True,
            fixed_k=None,
            null_gate_enabled=True,
            null_strengths=nulls,
            null_low_threshold=low,
            null_high_threshold=high,
        )

    def _case(self, count_ge):
        """Build utilities + a null distribution placing p_null exactly at (1+count_ge)/31."""
        base = self._run([0.5], 0.5, 1.0, [0.9, 0.89, 0.88, 0.87, 0.86, 0.5, 0.49, 0.48])
        # first probe pass: gate effectively neutral, read strength/adaptive_k
        probe = select_adaptive_k(
            [0.9, 0.89, 0.88, 0.87, 0.86, 0.5, 0.49, 0.48],
            chi=0.8, temperature=0.15, eta=0.5, epsilon=1e-8, z_max=8.0,
            iqr_normal_consistency=1.349,
            adaptive_enabled=True, entropy_enabled=True, maximum_gap_enabled=True,
            consistency_entropy_enabled=True, fixed_k=None,
            null_gate_enabled=False, null_strengths=[0.5],
            null_low_threshold=None, null_high_threshold=None,
        )
        strength = probe.evidence_strength
        nulls = [strength + 1.0] * count_ge + [strength - 1.0] * (30 - count_ge)
        return nulls, probe.adaptive_k

    def test_raising_null_high_releases_null_like_to_boundary(self):
        nulls, ak = self._case(19)  # p = 20/31
        base = self._run(nulls, 3 / 31, 16 / 31)
        self.assertEqual(base.null_region, "null_like")
        self.assertEqual(base.selected_k, 1)
        raised = self._run(nulls, 3 / 31, 25 / 31)
        self.assertEqual(raised.null_region, "boundary")
        self.assertEqual(raised.selected_k, min(3, ak))

    def test_raising_null_low_releases_boundary_to_strong(self):
        nulls, ak = self._case(4)  # p = 5/31
        self.assertGreater(ak, 3)
        base = self._run(nulls, 3 / 31, 16 / 31)
        self.assertEqual(base.null_region, "boundary")
        self.assertEqual(base.selected_k, min(3, ak))
        raised = self._run(nulls, 9 / 31, 16 / 31)
        self.assertEqual(raised.null_region, "strong_evidence")
        self.assertEqual(raised.selected_k, ak)

    def test_lowering_thresholds_tightens_the_gate(self):
        nulls, _ = self._case(14)  # p = 15/31
        base = self._run(nulls, 3 / 31, 16 / 31)
        self.assertEqual(base.null_region, "boundary")
        tighter = self._run(nulls, 3 / 31, 15 / 31)
        self.assertEqual(tighter.null_region, "null_like")
        self.assertEqual(tighter.selected_k, 1)

    def test_null_thresholds_change_utility_signature(self):
        kwargs = dict(
            lambda_false_positive=0.2, lambda_positive=0.2, lambda_redundancy=0.15,
            context_gate_threshold=0.2, null_low_threshold=3 / 31,
            null_high_threshold=16 / 31, method_version="ambical-online-v5",
            gate_v2_bias=0.0, gate_v2_weight_strength=0.0,
            gate_v2_weight_competition=0.0, gate_v2_weight_distance=0.0,
            gate_v2_weight_path_conflict=0.0, gate_v2_weight_bimodality=0.0,
            gate_v2_weight_branch_disagreement=0.0,
        )
        s1 = AmbiCalConfig(**kwargs).utility_signature()
        s2 = AmbiCalConfig(
            **{**kwargs, "null_low_threshold": 5 / 31}
        ).utility_signature()
        s3 = AmbiCalConfig(
            **{**kwargs, "null_high_threshold": 19 / 31}
        ).utility_signature()
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    def test_null_threshold_ordering_still_enforced(self):
        with self.assertRaisesRegex(ValueError, "smaller"):
            AmbiCalConfig(
                lambda_false_positive=0.2, lambda_positive=0.2,
                lambda_redundancy=0.15, context_gate_threshold=0.2,
                null_low_threshold=0.7, null_high_threshold=0.5,
                counterfactual_promotion_top_k=3, max_counterfactual_phrases=2,
                method_version="ambical-online-v5",
                gate_v2_bias=0.0, gate_v2_weight_strength=0.0,
                gate_v2_weight_competition=0.0, gate_v2_weight_distance=0.0,
                gate_v2_weight_path_conflict=0.0, gate_v2_weight_bimodality=0.0,
                gate_v2_weight_branch_disagreement=0.0,
            ).validate()


if __name__ == "__main__":
    unittest.main()
