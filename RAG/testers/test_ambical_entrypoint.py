import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm_connectors.base_llm_connector import (
    BaseLLMConnector,
    TokenUsageUnavailableError,
)
from llm_connectors.ollama_connector import OllamaConnector
from rag_pipeline import initialize_retriever
from rag_tester import RagTester as BaseRagTester
from rag_tester_ambical import (
    FORMAL_PYTHON_EXECUTABLE,
    RagTester as AmbiCalRagTester,
    _installed_ollama_digest,
    _load_and_preflight_config,
    _ollama_model_digests,
    _validate_formal_runtime,
    main as ambical_main,
    start_rag_tests,
)
from retrieval_pipelines.ambical import (
    AmbiCalProfile,
    AmbiCalRetriever,
    FalsePositivePriors,
)
from utils.config_loader import ConfigLoader
from utils.result_manager import ResultManager
from utils.ambical_experiment import (
    build_ambical_config,
    ambical_result_identifier,
    load_ambical_profile,
    validate_profile_for_config,
    validate_profile_scope,
)


def method_parameters(**overrides):
    method_config = {
        "candidate_top_n": 2,
        "max_k": 3,
        "gate_top_m": 2,
        "branch_temperature": 0.1,
        "gate_temperature": 0.1,
        "adaptive_temperature": 0.15,
        "consistency_floor_eta": 0.5,
        "max_counterfactual_phrases": 3,
        "z_max": 8.0,
        "epsilon": 1e-8,
        "iqr_normal_consistency": 1.349,
        "minimum_counterfactual_content_tokens": 1,
        "require_normalized_collection_embeddings": True,
        "collection_norm_tolerance": 0.02,
        "method_version": "ambical-online-v1",
        "switches": {
            "translation_branch": True,
            "robust_normalization": True,
            "branch_confidence": True,
            "translation_consistency": True,
            "ranking_overlap": True,
            "dynamic_branch_weights": True,
            "false_positive_prior": False,
            "context_correction": False,
            "context_gate": False,
            "soft_redundancy": False,
            "consistency_entropy": True,
            "maximum_gap": True,
            "entropy_candidate": True,
            "null_gate": False,
            "adaptive_k": True,
            "merge_structured_evidence": True,
        },
    }
    switch_overrides = overrides.pop("switches", None)
    if switch_overrides is not None:
        method_config["switches"].update(switch_overrides)
    method_config.update(overrides)
    return {"source_language": "English", "method_config": method_config}


def write_complete_profile_artifact(directory, payload):
    root = Path(directory)
    profile_path = root / "ambical_profile.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    profile_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    profile_path.with_suffix(".json.sha256").write_text(
        f"{profile_sha256}  {profile_path.name}\n",
        encoding="ascii",
    )
    manifest_path = root / "ambical_calibration_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "ambical_offline_calibration",
                "outputs": {
                    "profile": {
                        "absolute_path": str(profile_path.resolve()),
                        "sha256": profile_sha256,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_sidecar = manifest_path.with_suffix(".json.sha256")
    manifest_sidecar.write_text(
        f"{manifest_sha256}  {manifest_path.name}\n",
        encoding="ascii",
    )
    (root / "CALIBRATION_COMPLETE").write_text(
        json.dumps(
            {
                "status": "complete",
                "profile_filename": profile_path.name,
                "profile_sha256": profile_sha256,
                "manifest_filename": manifest_path.name,
                "manifest_sha256": manifest_sha256,
                "manifest_sha256_sidecar": manifest_sidecar.name,
            }
        ),
        encoding="utf-8",
    )
    return profile_path


class _EmbeddingRetriever:
    def vectorize_text(self, text):
        if "terjemahan" in text:
            return [[0.0, 1.0]]
        return [[1.0, 0.0]]


class _Collection:
    name = "mock_collection"

    def __init__(self):
        self.rows = {
            "a": ("source evidence", [1.0, 0.0]),
            "b": ("translated evidence", [0.0, 1.0]),
            "c": ("shared evidence", [2 ** -0.5, 2 ** -0.5]),
        }

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, include):
        query = query_embeddings[0]
        ids = ["a", "c"] if query[0] >= query[1] else ["b", "c"]
        return {"ids": [ids[:n_results]]}

    def get(self, ids, include):
        return {
            "ids": list(ids),
            "documents": [self.rows[item][0] for item in ids],
            "embeddings": [self.rows[item][1] for item in ids],
            "metadatas": [{} for _ in ids],
        }


class _Client:
    def __init__(self, collection):
        self.collection = collection
        self.requested_names = []

    def get_collection(self, name):
        self.requested_names.append(name)
        return self.collection


class _TranslationResult:
    translated_query = "terjemahan pertanyaan"

    def diagnostics(self):
        return {"translation_cache_hit": True}


class _Translator:
    prompt_sha256 = "prompt-sha"

    def translate(self, query, source_language, target_language):
        return _TranslationResult()


class _QuestionConnector(BaseLLMConnector):
    def __init__(self):
        super().__init__("question", {"name": "question", "temperature": 0.0})

    def invoke(self, prompt):
        return "mock answer"


class AmbiCalExperimentSupportTests(unittest.TestCase):
    def test_config_builder_rejects_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "Unknown AmbiCal"):
            build_ambical_config(
                method_parameters(undeclared_parameter=1)
            )

    def test_config_builder_rejects_unknown_experiment_fields(self):
        parameters = method_parameters()
        parameters["profile_pat"] = "typo.json"
        with self.assertRaisesRegex(ValueError, "Unknown ambical_parameters"):
            build_ambical_config(parameters)

    def test_config_builder_rejects_non_boolean_switches(self):
        parameters = method_parameters(switches={"adaptive_k": "false"})
        with self.assertRaisesRegex(ValueError, "JSON booleans"):
            build_ambical_config(parameters)

    def test_config_builder_rejects_non_boolean_method_flag(self):
        parameters = method_parameters(
            require_normalized_collection_embeddings="false"
        )
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            build_ambical_config(parameters)

    def test_config_builder_rejects_boolean_numeric_field(self):
        parameters = method_parameters(branch_temperature=True)
        with self.assertRaisesRegex(ValueError, "must be a number"):
            build_ambical_config(parameters)

    def test_result_identifier_changes_with_resolved_config(self):
        first = build_ambical_config(method_parameters(max_k=3))
        second = build_ambical_config(method_parameters(max_k=4))
        self.assertNotEqual(
            ambical_result_identifier(first),
            ambical_result_identifier(second),
        )

    def test_profile_loader_records_absolute_path_and_sha256(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_complete_profile_artifact(
                temporary_directory,
                {
                    "profile": {
                        "false_positive_priors": None,
                        "null_strengths": [],
                        "utility_signature": None,
                    },
                    "scope": {"collection_name": "mock_collection"},
                },
            )
            loaded = load_ambical_profile(path)
            self.assertEqual(loaded.absolute_path, str(path.resolve()))
            self.assertEqual(len(loaded.sha256), 64)
            self.assertEqual(loaded.profile.null_strengths, ())
            self.assertEqual(loaded.scope["collection_name"], "mock_collection")

    def test_profile_loader_rejects_incomplete_publication(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_complete_profile_artifact(
                temporary_directory,
                {
                    "profile": {
                        "false_positive_priors": None,
                        "null_strengths": [],
                        "utility_signature": None,
                    },
                    "scope": {},
                },
            )
            (Path(temporary_directory) / "CALIBRATION_INCOMPLETE").write_text(
                "incomplete",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                load_ambical_profile(path)

    def test_profile_loader_rejects_tampered_published_profile(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_complete_profile_artifact(
                temporary_directory,
                {
                    "profile": {
                        "false_positive_priors": None,
                        "null_strengths": [],
                        "utility_signature": None,
                    },
                    "scope": {},
                },
            )
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completion marker"):
                load_ambical_profile(path)

    def test_profile_is_only_required_by_active_profile_components(self):
        no_profile_config = build_ambical_config(method_parameters())
        validate_profile_for_config(no_profile_config, None)

        requiring_profile = build_ambical_config(
            method_parameters(
                lambda_false_positive=0.2,
                switches={
                    "false_positive_prior": True,
                    "context_correction": False,
                    "soft_redundancy": False,
                    "null_gate": False,
                },
            )
        )
        with self.assertRaisesRegex(ValueError, "profile_path"):
            validate_profile_for_config(requiring_profile, None)

    def test_wo_null_gate_profile_signature_mismatch_is_rejected(self):
        config = build_ambical_config(
            method_parameters(
                lambda_false_positive=0.2,
                switches={
                    "false_positive_prior": True,
                    "context_correction": False,
                    "soft_redundancy": False,
                    "null_gate": False,
                },
            )
        )
        mismatched = AmbiCalProfile(
            false_positive_priors=FalsePositivePriors(
                source={"chunk-a": 0.1},
                target={"chunk-a": 0.1},
                unseen_source=0.05,
                unseen_target=0.05,
            ),
            null_strengths=(0.5,),
            utility_signature="f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "signature does not match"):
            validate_profile_for_config(config, mismatched)

        missing_priors = AmbiCalProfile(
            false_positive_priors=None,
            null_strengths=(0.5,),
            utility_signature=config.utility_signature(),
        )
        with self.assertRaisesRegex(ValueError, "false_positive_priors"):
            validate_profile_for_config(config, missing_priors)

        matching = AmbiCalProfile(
            false_positive_priors=FalsePositivePriors(
                source={"chunk-a": 0.1},
                target={"chunk-a": 0.1},
                unseen_source=0.05,
                unseen_target=0.05,
            ),
            null_strengths=(0.5,),
            utility_signature=config.utility_signature(),
        )
        validate_profile_for_config(config, matching)

    def test_profile_scope_is_bound_to_collection_identity(self):
        config = build_ambical_config(
            method_parameters(
                lambda_false_positive=0.2,
                switches={"false_positive_prior": True},
            )
        )
        expected = {
            "collection_name": "manual_cs200_os100",
            "document_language": "Indonesian",
        }
        scope = {
            **expected,
            "false_positive_calibration_sha256": "a" * 64,
            "false_positive_test_overlap_count": 0,
        }
        validate_profile_scope(config, scope, expected)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_profile_scope(
                config,
                {**scope, "collection_name": "wrong_collection"},
                expected,
            )

    def test_config_loader_requires_ambical_object(self):
        loader = ConfigLoader.__new__(ConfigLoader)
        loader.config = {"ambical_parameters": []}
        with self.assertRaisesRegex(ValueError, "JSON object"):
            loader.get_ambical_parameters()

    def test_collection_fingerprint_is_order_independent_and_content_sensitive(self):
        class FingerprintCollection:
            def __init__(self, rows):
                self.rows = list(rows)

            def count(self):
                return len(self.rows)

            def get(self, *, limit, offset, include):
                selected = self.rows[offset : offset + limit]
                return {
                    "ids": [row[0] for row in selected],
                    "documents": [row[1] for row in selected],
                    "embeddings": [row[2] for row in selected],
                    "metadatas": [row[3] for row in selected],
                }

        rows = [
            ("a", "alpha", [1.0, 0.0], {"page": 1}),
            ("b", "beta", [0.0, 1.0], {"page": 2}),
        ]
        first = BaseRagTester._ambical_collection_fingerprint(
            FingerprintCollection(rows), batch_size=1
        )
        reordered = BaseRagTester._ambical_collection_fingerprint(
            FingerprintCollection(list(reversed(rows))), batch_size=1
        )
        changed = BaseRagTester._ambical_collection_fingerprint(
            FingerprintCollection(
                [("a", "changed", [1.0, 0.0], {"page": 1}), rows[1]]
            ),
            batch_size=1,
        )
        self.assertEqual(first["sha256"], reordered["sha256"])
        self.assertNotEqual(first["sha256"], changed["sha256"])
        self.assertEqual(first["count"], 2)


class AmbiCalInitializerTests(unittest.TestCase):
    def test_initializer_binds_requested_collection_and_typed_dependencies(self):
        collection = _Collection()
        client = _Client(collection)
        config = build_ambical_config(method_parameters())
        translator = _Translator()
        with patch("rag_pipeline.EmbeddingRetriever", return_value=_EmbeddingRetriever()):
            retriever = initialize_retriever(
                "ambical",
                chroma_client=client,
                collection_name="expected_collection",
                ambical_config=config,
                ambical_translator=translator,
            )
        self.assertIsInstance(retriever, AmbiCalRetriever)
        self.assertIs(retriever.collection, collection)
        self.assertIs(retriever.translator, translator)
        self.assertEqual(client.requested_names, ["expected_collection"])

    def test_initializer_requires_resolved_method_config(self):
        with self.assertRaisesRegex(ValueError, "resolved ambical_config"):
            initialize_retriever(
                "ambical",
                chroma_client=_Client(_Collection()),
                collection_name="collection",
            )


class AmbiCalSharedTesterTests(unittest.TestCase):
    def test_qa_adapter_saves_full_method_result_without_official_selector(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tester = BaseRagTester.__new__(BaseRagTester)
            tester.loaded_datasets = {
                "general_questions": [
                    {
                        "question": "English question",
                        "answer": "expected",
                        "page": 1,
                        "dataset": "general_questions",
                    }
                ]
            }
            tester.total_questions_to_answer = 1
            tester.num_retrieved_docs = 99
            tester.adaptive_k_enabled = False
            tester.adaptive_k_strategy = "largest_gap"
            tester.adaptive_k_selector = None
            tester.ambical_source_language = "English"
            tester.ambical_target_languages = {"indonesian": "Indonesian"}
            tester.ambical_config = build_ambical_config(
                method_parameters()
            )
            tester.question_prompt_template = "Context: {context}\nQuestion: {question}"
            tester.output_dir = temporary_directory

            collection = _Collection()
            retriever = AmbiCalRetriever(
                collection=collection,
                embedding_retriever=_EmbeddingRetriever(),
                config=tester.ambical_config,
                translator=_Translator(),
            )
            results, _, count = tester._run_qa_phase(
                retriever=retriever,
                collection=collection,
                question_llm_connector=_QuestionConnector(),
                current_retrieval_algorithm="ambical",
                current_question_model_name="question",
                language="indonesian",
                chunk_size=200,
                overlap_size=100,
            )
            record = results["general_questions"]["results"][0]
            self.assertEqual(count, 1)
            self.assertEqual(record["model_answer"], "mock answer")
            self.assertIsNone(record["adaptive_threshold"])
            self.assertEqual(
                record["selected_k"],
                record["ambical_result"]["selected_chunk_count_before_merge"],
            )
            self.assertEqual(
                record["ambical_result"]["diagnostics"]["method"],
                tester.ambical_config.method_version,
            )
            self.assertTrue(record["ambical_adaptive_k_enabled"])
            json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))


class AmbiCalDedicatedEntrypointTests(unittest.TestCase):
    @staticmethod
    def _fake_base_init(instance, config_path):
        configured_output = json.loads(
            Path(config_path).read_text(encoding="utf-8")
        )["output_dir"]
        Path(configured_output).mkdir(parents=True)
        instance.retrieval_algorithms_to_test = ["ambical"]
        instance.adaptive_k_enabled = False
        instance.language_configs = [{"language": "indonesian"}]
        instance.chunk_sizes_to_test = [200]
        instance.overlap_sizes_to_test = [100]
        instance.question_models_to_test = ["question"]
        instance.total_questions_to_answer = 1
        instance.result_k_identifier = "ambical_test"
        instance.output_dir = configured_output
        instance.result_manager = ResultManager(configured_output)

    @staticmethod
    def _write_config(temporary_directory, *, output_exists=False):
        output_path = Path(temporary_directory) / "new_results"
        if output_exists:
            output_path.mkdir()
        config_path = Path(temporary_directory) / "config.json"
        config_path.write_text(
            json.dumps({"output_dir": str(output_path.resolve())}),
            encoding="utf-8",
        )
        return config_path, output_path

    def test_dedicated_entrypoint_accepts_isolated_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, _ = self._write_config(temporary_directory)
            with patch.object(
                BaseRagTester,
                "__init__",
                self._fake_base_init,
            ):
                tester = AmbiCalRagTester(str(config_path))
        self.assertEqual(tester.retrieval_algorithms_to_test, ["ambical"])

    def test_shared_entrypoint_cannot_bypass_dedicated_digest_binding(self):
        shared = BaseRagTester.__new__(BaseRagTester)
        shared.retrieval_algorithms_to_test = ["ambical"]
        with self.assertRaisesRegex(RuntimeError, "disabled in the shared"):
            shared._validate_ambical_entrypoint()

        dedicated = AmbiCalRagTester.__new__(AmbiCalRagTester)
        dedicated.retrieval_algorithms_to_test = ["ambical"]
        dedicated._validate_ambical_entrypoint()

    def test_dedicated_entrypoint_rejects_shared_official_adaptive_k(self):
        def invalid_init(instance, config_path):
            self._fake_base_init(instance, config_path)
            instance.adaptive_k_enabled = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, _ = self._write_config(temporary_directory)
            with patch.object(BaseRagTester, "__init__", invalid_init):
                with self.assertRaisesRegex(ValueError, "shared official"):
                    AmbiCalRagTester(str(config_path))

    def test_dedicated_entrypoint_rejects_nonisolated_dimensions(self):
        cases = (
            ("algorithm", "exactly one", {"retrieval_algorithms_to_test": ["embedding"]}),
            (
                "languages",
                "exactly one document language",
                {"language_configs": [{"language": "a"}, {"language": "b"}]},
            ),
            ("chunk sizes", "exactly one chunk size", {"chunk_sizes_to_test": [100, 200]}),
            ("overlaps", "exactly one chunk size", {"overlap_sizes_to_test": [0, 50]}),
        )
        for label, message, updates in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary_directory:
                config_path, _ = self._write_config(temporary_directory)

                def invalid_init(instance, config_path):
                    self._fake_base_init(instance, config_path)
                    for key, value in updates.items():
                        setattr(instance, key, value)

                with patch.object(BaseRagTester, "__init__", invalid_init):
                    with self.assertRaisesRegex(ValueError, message):
                        AmbiCalRagTester(str(config_path))

    def test_dedicated_entrypoint_rejects_unsafe_language(self):
        def invalid_init(instance, config_path):
            self._fake_base_init(instance, config_path)
            instance.language_configs = [{"language": "../escape"}]

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, _ = self._write_config(temporary_directory)
            with patch.object(BaseRagTester, "__init__", invalid_init):
                with self.assertRaisesRegex(ValueError, "filesystem-safe"):
                    AmbiCalRagTester(str(config_path))

    def test_dedicated_entrypoint_rejects_result_filename_collisions(self):
        def invalid_init(instance, config_path):
            self._fake_base_init(instance, config_path)
            instance.question_models_to_test = ["a/b", "a:b"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, _ = self._write_config(temporary_directory)
            with patch.object(BaseRagTester, "__init__", invalid_init):
                with self.assertRaisesRegex(ValueError, "same result file"):
                    AmbiCalRagTester(str(config_path))

    def test_dedicated_entrypoint_requires_existing_config(self):
        with self.assertRaises(FileNotFoundError):
            AmbiCalRagTester("missing-ambical-config.json")

    def test_preflight_rejects_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path, _ = self._write_config(
                temporary_directory, output_exists=True
            )
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                _load_and_preflight_config(str(config_path))

    def test_preflight_rejects_reevaluation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "new_results"
            config_path = Path(temporary_directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "output_dir": str(output_path.resolve()),
                        "enable_reevaluation": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not permit reevaluation"):
                _load_and_preflight_config(str(config_path))

    def test_completion_check_rejects_zero_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            tester = AmbiCalRagTester.__new__(AmbiCalRagTester)
            tester.language_configs = [{"language": "indonesian"}]
            tester.question_models_to_test = ["question"]
            tester.chunk_sizes_to_test = [200]
            tester.overlap_sizes_to_test = [100]
            tester.result_k_identifier = "ambical_test"
            tester.output_dir = temporary_directory
            tester.total_questions_to_answer = 1
            tester.result_manager = Mock()
            tester.result_manager._generate_filename.return_value = "missing.json"
            tester._expected_result_targets = {
                "question": Path(temporary_directory) / "missing.json"
            }
            with self.assertRaisesRegex(RuntimeError, "did not produce"):
                tester.validate_completed_outputs()

    def test_start_entrypoint_constructs_tester_and_runs_once(self):
        fake_tester = Mock()
        fake_tester.validate_completed_outputs.return_value = ["result.json"]
        fake_tester.token_usage_summary.return_value = {
            "totals": {"total_tokens": 1},
            "per_connector": {},
        }
        with (
            patch(
                "rag_tester_ambical.RagTester", return_value=fake_tester
            ) as tester_class,
            patch(
                "rag_tester_ambical._snapshot_config",
                return_value={"sha256": "config-sha"},
            ) as snapshot,
            patch(
                "rag_tester_ambical._ollama_model_digests",
                return_value={},
            ) as digests,
            patch(
                "rag_tester_ambical._write_run_manifest",
                return_value="manifest.json",
            ) as manifest,
            patch("rag_tester_ambical._validate_formal_runtime") as runtime,
        ):
            self.assertTrue(start_rag_tests("explicit.json"))
        runtime.assert_called_once_with()
        tester_class.assert_called_once_with(config_path="explicit.json")
        fake_tester.bind_ollama_model_digests.assert_called_once_with({})
        fake_tester.run_tests.assert_called_once_with()
        fake_tester.validate_completed_outputs.assert_called_once_with()
        fake_tester.token_usage_summary.assert_called_once_with()
        snapshot.assert_called_once_with(fake_tester, "explicit.json")
        digests.assert_called_once_with(fake_tester)
        manifest.assert_called_once()

    def test_cli_forwards_required_config(self):
        with patch(
            "rag_tester_ambical.start_rag_tests", return_value=True
        ) as start:
            self.assertEqual(
                ambical_main(["--config", "explicit.json"]), 0
            )
        start.assert_called_once_with("explicit.json")

    def test_formal_runtime_accepts_configured_windows_gpu(self):
        with (
            patch("rag_tester_ambical.platform.system", return_value="Windows"),
            patch(
                "rag_tester_ambical.FORMAL_PYTHON_EXECUTABLE",
                Path(FORMAL_PYTHON_EXECUTABLE),
            ),
            patch("rag_tester_ambical.torch.cuda.is_available", return_value=True),
            patch("rag_tester_ambical.torch.cuda.device_count", return_value=1),
            patch(
                "rag_tester_ambical.torch.cuda.get_device_name",
                return_value="NVIDIA GeForce RTX 5070 Ti",
            ),
        ):
            _validate_formal_runtime()

    def test_ollama_digest_uses_full_api_digest(self):
        tester = Mock()
        tester.question_models_to_test = ["question"]
        tester.ambical_translator_model_name = ""
        tester.config_loader.get_evaluator_model_name.return_value = "evaluator"
        tester.config_loader.get_llm_type_and_config.side_effect = lambda alias: (
            "ollama",
            {"name": f"{alias}:latest"},
        )
        full_digest = "sha256:" + "a" * 64
        response = SimpleNamespace(
            models=[
                SimpleNamespace(model="question:latest", digest=full_digest),
                SimpleNamespace(model="evaluator:latest", digest=full_digest),
            ]
        )
        with patch("rag_tester_ambical.ollama.list", return_value=response):
            resolved = _ollama_model_digests(tester)
            translator_digest = _installed_ollama_digest("question:latest")
        self.assertEqual(resolved["question"]["digest"], full_digest)
        self.assertEqual(translator_digest, "a" * 64)

    def test_formal_run_rejects_profile_from_another_translator_binary(self):
        tester = AmbiCalRagTester.__new__(AmbiCalRagTester)
        tester.ambical_config = build_ambical_config(
            method_parameters(
                lambda_false_positive=0.2,
                switches={"false_positive_prior": True},
            )
        )
        tester.ambical_profile_scope = {
            "translator_ollama_sha256": "a" * 64,
            "false_positive_calibration_sha256": "b" * 64,
            "false_positive_test_overlap_count": 0,
        }
        tester.ambical_translator_model_name = "translator"
        tester.ambical_translator = SimpleNamespace(
            model_binary_sha256="c" * 64
        )
        tester.ambical_experiment_identity = {}
        with self.assertRaisesRegex(ValueError, "does not match"):
            tester.bind_ollama_model_digests(
                {
                    "translator": {
                        "model": "qwen2.5:7b",
                        "digest": "sha256:" + "c" * 64,
                    }
                }
            )

    def test_strict_result_manager_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manager = ResultManager(temporary_directory, refuse_overwrite=True)
            filename = manager._generate_filename(
                "ambical", "english", "question", 200, 100, "method"
            )
            (Path(temporary_directory) / filename).write_text("existing")
            with self.assertRaises(FileExistsError):
                manager.save_results(
                    {}, "ambical", "english", "question", 200, 100, "method"
                )


class TokenUsageTests(unittest.TestCase):
    def test_ollama_connector_records_reported_input_and_output_tokens(self):
        connector = OllamaConnector.__new__(OllamaConnector)
        BaseLLMConnector.__init__(
            connector,
            "fake",
            {"name": "fake", "temperature": 0.0},
        )
        connector.ollama_model_name = "fake"
        connector.ollama_client = Mock()
        connector.ollama_client.generate.return_value = iter(
            [
                SimpleNamespace(
                    response="ans",
                    done=False,
                    done_reason=None,
                    prompt_eval_count=None,
                    eval_count=None,
                ),
                SimpleNamespace(
                    response="wer",
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=7,
                    eval_count=3,
                ),
            ]
        )
        self.assertEqual(connector.invoke("question"), "answer")
        self.assertTrue(connector.ollama_client.generate.call_args.kwargs["stream"])
        self.assertEqual(
            connector.token_usage(),
            {
                "calls": 1,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "unreported_calls": 0,
            },
        )

    def test_ollama_connector_marks_failed_attempt_unreported(self):
        connector = OllamaConnector.__new__(OllamaConnector)
        BaseLLMConnector.__init__(
            connector,
            "fake",
            {"name": "fake", "temperature": 0.0},
        )
        connector.ollama_model_name = "fake"
        connector.ollama_client = Mock()
        connector.ollama_client.generate.side_effect = RuntimeError("provider failed")
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            connector.invoke("question")
        self.assertEqual(connector.token_usage()["calls"], 1)
        self.assertEqual(connector.token_usage()["unreported_calls"], 1)

    def test_ollama_connector_aborts_immediately_when_usage_is_missing(self):
        connector = OllamaConnector.__new__(OllamaConnector)
        BaseLLMConnector.__init__(
            connector,
            "fake",
            {"name": "fake", "temperature": 0.0},
        )
        connector.ollama_model_name = "fake"
        connector.ollama_client = Mock()
        connector.ollama_client.generate.return_value = iter(
            [
                SimpleNamespace(
                    response="answer",
                    done=True,
                    done_reason="stop",
                    prompt_eval_count=None,
                    prompt_eval_cached_count=None,
                    eval_count=3,
                )
            ]
        )
        with self.assertRaisesRegex(
            TokenUsageUnavailableError,
            '"done": true.*"prompt_eval_count": null',
        ):
            connector.invoke("question")
        self.assertEqual(connector.token_usage()["calls"], 1)
        self.assertEqual(connector.token_usage()["unreported_calls"], 1)

    def test_ollama_connector_rejects_nonfinal_http_200_response(self):
        connector = OllamaConnector.__new__(OllamaConnector)
        BaseLLMConnector.__init__(
            connector,
            "translator",
            {"name": "fake", "temperature": 0.0},
        )
        connector.ollama_model_name = "fake"
        connector.ollama_client = Mock()
        connector.ollama_client.generate.return_value = iter(
            [
                SimpleNamespace(
                    response="partial",
                    done=False,
                    done_reason=None,
                    prompt_eval_count=None,
                    prompt_eval_cached_count=None,
                    eval_count=None,
                )
            ]
        )
        with self.assertRaisesRegex(
            TokenUsageUnavailableError,
            'model_alias=.*translator.*"done": false',
        ):
            connector.invoke("question")
        self.assertEqual(connector.token_usage()["calls"], 1)
        self.assertEqual(connector.token_usage()["unreported_calls"], 1)


if __name__ == "__main__":
    unittest.main()
