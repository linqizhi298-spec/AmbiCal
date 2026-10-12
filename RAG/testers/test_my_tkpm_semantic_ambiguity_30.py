"""Static data tests for the my_tkpm semantic-ambiguity-30 dataset and parameter lock."""

import json
import re
import unittest
from pathlib import Path

RAG = Path(__file__).resolve().parent.parent
DS = RAG / "question_datasets" / "my_tkpm_english_qa"
FINAL = DS / "question_answers_my_tkpm_semantic_ambiguity_30_20260926_01.json"
CAND = DS / "question_answers_my_tkpm_semantic_ambiguity_candidates_20260926_01.json"
LOCK = RAG / "my_tkpm_v51_method_parameter_lock_20260926_01.json"
CHUNKS = json.loads((RAG.parent / "tmp" / "my_tkpm_chunks_sect.json").read_text(encoding="utf-8"))
CORPUS = {c["id"] for c in CHUNKS}


def norm(t):
    return re.sub(r"\s+", " ", str(t).strip().casefold())


def load(p):
    qs = json.loads(p.read_text(encoding="utf-8-sig"))
    return qs if isinstance(qs, list) else qs.get("questions", [])


class MyTkpmSemanticAmbiguity30Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.final = load(FINAL)
        cls.cand = load(CAND)
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8-sig"))
        cls.orig_a = load(DS / "question_answers_pairs.json")
        cls.orig_u = load(DS / "question_answers_unanswerable.json")

    def test_1_final_30(self):
        self.assertEqual(len(self.final), 30)

    def test_2_20_answerable_10_unanswerable(self):
        a = sum(1 for q in self.final if q["answerability"] == "answerable")
        u = sum(1 for q in self.final if q["answerability"] == "unanswerable")
        self.assertEqual((a, u), (20, 10))

    def test_3_question_ids_unique(self):
        ids = [q["question_id"] for q in self.final]
        self.assertEqual(len(ids), len(set(ids)))

    def test_4_stems_unique(self):
        stems = [norm(q["question"]) for q in self.final]
        self.assertEqual(len(set(stems)), 30)

    def test_5_exact_overlap_vs_original200_zero(self):
        orig = {norm(q["question"]) for q in self.orig_a + self.orig_u}
        self.assertEqual(sum(1 for q in self.final if norm(q["question"]) in orig), 0)

    def test_6_normalized_overlap_vs_original200_zero(self):
        orig_tokens = [set(re.findall(r"[a-z0-9]+", q["question"].lower())) for q in self.orig_a + self.orig_u]
        worst = 0.0
        for q in self.final:
            t = set(re.findall(r"[a-z0-9]+", q["question"].lower()))
            best = max((len(t & o) / max(1, len(t | o)) for o in orig_tokens), default=0.0)
            worst = max(worst, best)
        self.assertLessEqual(worst, 0.60)  # near-duplicate guard; observed max 0.538

    def test_7_all_chunk_ids_exist(self):
        for q in self.final:
            for cid in q["gold_chunk_ids"] + q["equivalent_gold_chunk_ids"] + q["wrong_sense_chunk_ids"]:
                self.assertIn(cid, CORPUS, q["question_id"])

    def test_8_gold_and_wrong_disjoint(self):
        for q in self.final:
            self.assertFalse(set(q["gold_chunk_ids"]) & set(q["wrong_sense_chunk_ids"]), q["question_id"])
            self.assertFalse(set(q["equivalent_gold_chunk_ids"]) & set(q["wrong_sense_chunk_ids"]), q["question_id"])

    def test_9_answerable_gold_nonempty(self):
        for q in self.final:
            if q["answerability"] == "answerable":
                self.assertTrue(q["gold_chunk_ids"] or q["equivalent_gold_chunk_ids"], q["question_id"])

    def test_10_unanswerable_gold_empty(self):
        for q in self.final:
            if q["answerability"] == "unanswerable":
                self.assertEqual(q["gold_chunk_ids"], [], q["question_id"])
                self.assertEqual(q["equivalent_gold_chunk_ids"], [], q["question_id"])

    def test_11_all_quality_10_of_10(self):
        for q in self.final:
            self.assertEqual(sum(q["quality_scores"].values()), 10, q["question_id"])

    def test_12_no_forbidden_phrasing(self):
        for q in self.final:
            low = q["question"].lower()
            for banned in ("rather than", "not a but b", "instead of the wrong", "not the wrong"):
                self.assertNotIn(banned, low, q["question_id"])

    def test_13_no_method_result_fields(self):
        for q in self.final:
            for banned in ("dense_qa_result", "ambical_qa_result", "ew_correct", "dense_correct",
                           "retrieval_result", "qa_judgment"):
                self.assertNotIn(banned, q, q["question_id"])

    def test_14_dense_screening_status_all_not_run(self):
        for q in self.final:
            self.assertEqual(q["dense_screening_status"], "not_run", q["question_id"])

    def test_15_ambical_flag_all_false(self):
        for q in self.final:
            self.assertIs(q["ambical_used_during_construction"], False, q["question_id"])

    def test_17_method_config_signature_unchanged(self):
        import sys
        sys.path.insert(0, str(RAG))
        from utils.ambical_experiment import build_ambical_config
        lock = self.lock
        method = {
            "method_version": lock["method_version"],
            "null_low_threshold": lock["null_low"], "null_high_threshold": lock["null_high"],
            "lambda_positive": lock["lambda_positive"],
            "max_counterfactual_phrases": lock["max_counterfactual_phrases"],
            "counterfactual_promotion_top_k": lock["counterfactual_promotion_top_k"],
            "candidate_top_n": lock["candidate_top_n"], "max_k": lock["max_k"], "gate_top_m": lock["gate_top_m"],
            "lambda_false_positive": lock["lambda_false_positive"], "lambda_redundancy": lock["lambda_redundancy"],
            "branch_temperature": lock["branch_temperature"], "gate_temperature": lock["gate_temperature"],
            "adaptive_temperature": lock["adaptive_temperature"], "consistency_floor_eta": lock["consistency_floor_eta"],
            "z_max": lock["z_max"], "epsilon": lock["epsilon"], "context_gate_threshold": lock["context_gate_threshold"],
            "iqr_normal_consistency": lock["iqr_normal_consistency"],
            "minimum_counterfactual_content_tokens": lock["minimum_counterfactual_content_tokens"],
            "require_normalized_collection_embeddings": lock["require_normalized_collection_embeddings"],
            "collection_norm_tolerance": lock["collection_norm_tolerance"],
            "gate_v2_bias": lock["gate_v2_weights"]["gate_v2_bias"],
            **{k: v for k, v in lock["gate_v2_weights"].items()},
            "switches": lock["switches"],
        }
        cfg = build_ambical_config({"source_language": "English", "method_config": method})
        # 数值参数与冻结锁完全一致；方法标识统一为 ambical-online-v5 后
        # 其 utility_signature 为 efd31b151...
        self.assertEqual(cfg.utility_signature(),
                         "efd31b1515662a5e94a22f69b58b9f430cd43e5578421f5f2b0a170dd3298614")


if __name__ == "__main__":
    unittest.main()
