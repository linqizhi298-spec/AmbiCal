import hashlib
import json
import re
import unittest
from pathlib import Path


RAG_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = RAG_ROOT.parent
DATASET_DIRECTORY = (
    RAG_ROOT / "question_datasets" / "vn_low_emission_rice_english_qa"
)
MANIFEST_PATH = DATASET_DIRECTORY / "ambical_calibration_questions_manifest.json"
FALSE_POSITIVE_PATH = (
    DATASET_DIRECTORY / "question_answers_false_positive_calibration.json"
)
NULL_PATH = DATASET_DIRECTORY / "question_answers_null_calibration.json"
FORMAL_ANSWERABLE_PATH = DATASET_DIRECTORY / "question_answers_pairs.json"
FORMAL_UNANSWERABLE_PATH = (
    DATASET_DIRECTORY / "question_answers_unanswerable.json"
)

EXPECTED_FALSE_POSITIVE_SHA256 = (
    "678255e48af9e08f8ab2d1d2d89e6f6dde686f7f4acb1d850bb8777a8b542362"
)
EXPECTED_NULL_SHA256 = (
    "77fe196880f44d5b2eea920bff7383f568012aba6c79889e997d8545c77393e9"
)
EXPECTED_FORMAL_ANSWERABLE_SHA256 = (
    "6b4d918c8433186703b0c1918170883b3cdeaaae5240ee000c1dacf0ee4ac4eb"
)
EXPECTED_FORMAL_UNANSWERABLE_SHA256 = (
    "23de6833abd42b586b925952002be6f71e8dcbb736a5526beaf8d5fc8da0065c"
)


def _reject_non_finite(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=_reject_non_finite,
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_question(question):
    """Match the manifest's lowercase ASCII-alphanumeric normalization."""

    return " ".join(re.findall(r"[a-z0-9]+", str(question).casefold()))


class VietnameseLowEmissionRiceCalibrationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = _load_json(MANIFEST_PATH)
        cls.false_positive = _load_json(FALSE_POSITIVE_PATH)
        cls.null_questions = _load_json(NULL_PATH)
        cls.formal_answerable = _load_json(FORMAL_ANSWERABLE_PATH)
        cls.formal_unanswerable = _load_json(FORMAL_UNANSWERABLE_PATH)

    def test_current_files_have_expected_counts_and_locked_hashes(self):
        self.assertEqual(len(self.false_positive), 30)
        self.assertEqual(len(self.null_questions), 30)
        self.assertEqual(len(self.formal_answerable), 100)
        self.assertEqual(len(self.formal_unanswerable), 100)

        self.assertEqual(_sha256(FALSE_POSITIVE_PATH), EXPECTED_FALSE_POSITIVE_SHA256)
        self.assertEqual(_sha256(NULL_PATH), EXPECTED_NULL_SHA256)
        self.assertEqual(
            _sha256(FORMAL_ANSWERABLE_PATH), EXPECTED_FORMAL_ANSWERABLE_SHA256
        )
        self.assertEqual(
            _sha256(FORMAL_UNANSWERABLE_PATH), EXPECTED_FORMAL_UNANSWERABLE_SHA256
        )

    def test_manifest_hashes_match_raw_file_bytes(self):
        sources = self.manifest["sources"]
        calibration_sets = self.manifest["calibration_sets"]
        declared = {
            "manual": sources["manual"],
            "formal_answerable_questions": sources["formal_answerable_questions"],
            "formal_unanswerable_questions": sources[
                "formal_unanswerable_questions"
            ],
            "false_positive_prior": calibration_sets["false_positive_prior"],
            "null_distribution": calibration_sets["null_distribution"],
        }
        for label, identity in declared.items():
            with self.subTest(label=label):
                path = (REPOSITORY_ROOT / identity["path"]).resolve(strict=True)
                self.assertTrue(path.is_file())
                self.assertRegex(identity["sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(_sha256(path), identity["sha256"])

        self.assertNotEqual(
            calibration_sets["false_positive_prior"]["sha256"],
            calibration_sets["null_distribution"]["sha256"],
        )

    def test_calibration_question_schema_purpose_and_ids_are_strict(self):
        expected_fields = {
            "question",
            "answer",
            "page",
            "calibration_id",
            "calibration_purpose",
            "topical_anchor",
            "absent_reason",
        }
        expected_sets = (
            (
                self.false_positive,
                "false_positive_prior",
                "AMB-VN-FP-",
            ),
            (
                self.null_questions,
                "null_distribution",
                "AMB-VN-NULL-",
            ),
        )
        all_ids = []
        for questions, purpose, prefix in expected_sets:
            for index, item in enumerate(questions, start=1):
                with self.subTest(calibration_id=item.get("calibration_id")):
                    self.assertEqual(set(item), expected_fields)
                    self.assertIsInstance(item["question"], str)
                    self.assertTrue(item["question"].strip())
                    self.assertEqual(item["answer"], "Not found in context")
                    self.assertIsNone(item["page"])
                    self.assertEqual(item["calibration_purpose"], purpose)
                    self.assertEqual(item["calibration_id"], f"{prefix}{index:03d}")
                    self.assertTrue(item["topical_anchor"].strip())
                    self.assertTrue(item["absent_reason"].strip())
                    all_ids.append(item["calibration_id"])

        self.assertEqual(len(all_ids), 60)
        self.assertEqual(len(set(all_ids)), 60)

    def test_normalized_questions_are_disjoint_from_each_other_and_formal_200(self):
        false_positive = {
            _normalize_question(item["question"]) for item in self.false_positive
        }
        null_questions = {
            _normalize_question(item["question"]) for item in self.null_questions
        }
        formal = {
            _normalize_question(item["question"])
            for item in self.formal_answerable + self.formal_unanswerable
        }

        self.assertEqual(len(false_positive), 30)
        self.assertEqual(len(null_questions), 30)
        self.assertEqual(len(formal), 200)
        self.assertFalse(false_positive & null_questions)
        self.assertFalse(false_positive & formal)
        self.assertFalse(null_questions & formal)

    def test_manifest_independence_counts_are_recomputed_and_zero(self):
        false_positive = {
            _normalize_question(item["question"]) for item in self.false_positive
        }
        null_questions = {
            _normalize_question(item["question"]) for item in self.null_questions
        }
        formal = {
            _normalize_question(item["question"])
            for item in self.formal_answerable + self.formal_unanswerable
        }
        checks = self.manifest["independence_checks"]
        actual = {
            "false_positive_test_overlap_count": len(false_positive & formal),
            "null_test_overlap_count": len(null_questions & formal),
            "false_positive_null_overlap_count": len(
                false_positive & null_questions
            ),
            "duplicate_calibration_id_count": 60
            - len(
                {
                    item["calibration_id"]
                    for item in self.false_positive + self.null_questions
                }
            ),
        }
        for name, value in actual.items():
            with self.subTest(check=name):
                self.assertEqual(value, 0)
                self.assertEqual(checks[name], value)

    def test_manifest_marks_data_ready_without_claiming_profile_or_full_test(self):
        design = self.manifest["design"]
        status = self.manifest["status"]
        self.assertEqual(design["total_questions"], 60)
        self.assertEqual(design["false_positive_prior_questions"], 30)
        self.assertEqual(design["null_distribution_questions"], 30)
        self.assertTrue(status["questions_ready"])
        self.assertFalse(status["calibration_profile_built"])
        self.assertFalse(status["ambical_full_test_started"])


if __name__ == "__main__":
    unittest.main()
