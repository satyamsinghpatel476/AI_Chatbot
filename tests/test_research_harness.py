import tempfile
import unittest

from evaluator.research.benchmarks import (
    build_context_contamination_suite,
    build_cross_domain_suite,
    build_intent_suite,
    build_knowledge_suite,
    build_memory_suite,
)
from evaluator.research.run_clean_experiment import score_intent_rows
from evaluator.research.scoring import (
    score_knowledge_rows,
    score_memory_rows,
)
from evaluator.research.statistics import (
    bootstrap_ci,
    describe_latency,
    mcnemar_exact,
)


class BenchmarkDesignTests(unittest.TestCase):
    def test_required_suite_sizes(self):
        self.assertEqual(len(build_context_contamination_suite()), 100)
        self.assertEqual(len(build_memory_suite()), 7)
        self.assertEqual(len(build_knowledge_suite()), 10)
        self.assertEqual(len(build_intent_suite()), 60)
        self.assertEqual(len(build_cross_domain_suite()), 50)

    def test_intent_suite_is_balanced(self):
        counts = {}
        for row in build_intent_suite():
            counts[row["gold_intent"]] = counts.get(row["gold_intent"], 0) + 1
        self.assertEqual(set(counts.values()), {10})

    def test_context_ids_are_unique(self):
        rows = build_context_contamination_suite()
        self.assertEqual(len({row["id"] for row in rows}), len(rows))


class ScoringTests(unittest.TestCase):
    def test_personal_contradiction_is_not_exact_recall(self):
        row = {
            "id": "mem-01",
            "system": "A",
            "expected": "Asha",
            "response": "Your name is not Asha; it is Ravi.",
        }
        scored = score_memory_rows([row])[0]
        self.assertEqual(scored["exact_recall"], 0)
        self.assertEqual(scored["incorrect_recall"], 1)

    def test_intent_absence_stays_na(self):
        row = {
            "id": "intent-general-01",
            "system": "A",
            "gold_intent": "general",
            "predicted_intent": None,
        }
        self.assertIsNone(score_intent_rows([row])[0]["intent_correct"])

    def test_knowledge_false_memory(self):
        row = {
            "id": "know-01",
            "system": "A",
            "required_terms": ["grid", "low-energy", "cells"],
            "exact_recall_response": "It is a grid of low-energy cells.",
            "paraphrased_recall_response": "It marks cells with low energy.",
            "unrelated_query_response": "Velora Grid Prime is a faster version.",
            "exact_recall_latency_ms": 1,
            "paraphrased_recall_latency_ms": 2,
            "unrelated_query_latency_ms": 3,
        }
        scored = score_knowledge_rows([row])[0]
        self.assertEqual(scored["knowledge_growth_accuracy"], 1.0)
        self.assertEqual(scored["false_memory"], 1)
        self.assertEqual(scored["latency_ms"], 2.0)


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_and_latency_are_not_rounded_to_zero(self):
        low, high = bootstrap_ci([1.2, 2.4, 3.6], seed=1, samples=100)
        self.assertLess(low, high)
        summary = describe_latency([0.4, 1.2, 7.9], seed=1)
        self.assertGreater(summary["median"], 0)
        self.assertGreater(summary["p95"], summary["median"])

    def test_mcnemar_uses_paired_discordance(self):
        result = mcnemar_exact([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(result["discordant"], 2)
        self.assertEqual(result["n"], 4)


if __name__ == "__main__":
    unittest.main()
