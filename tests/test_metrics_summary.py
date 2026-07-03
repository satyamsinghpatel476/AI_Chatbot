import unittest

from evaluator.metrics_summary import summarize


class MetricsSummaryTests(unittest.TestCase):
    def test_missing_intent_and_mixed_score_are_safe(self):
        entries = [
            {
                "question": "Can Uber improve robot localization?",
                "results": {
                    "A": {
                        "accuracy": 8,
                        "latency": 2,
                        "hallucination": 0,
                        "contamination": 1,
                        "question_type": "mixed",
                        "expected_intent": "mixed",
                        "predicted_intent": None,
                    },
                    "B": {
                        "accuracy": 10,
                        "latency": 1,
                        "hallucination": 0,
                        "contamination": 0,
                        "question_type": "mixed",
                        "expected_intent": "mixed",
                        "predicted_intent": "mixed",
                    },
                    "C": {
                        "accuracy": 6,
                        "latency": 3,
                        "hallucination": 1,
                        "contamination": 1,
                        "question_type": "mixed",
                        "expected_intent": "mixed",
                        "predicted_intent": "daily",
                    },
                },
            },
            {
                "question": "What is SLAM?",
                "results": {
                    "A": {
                        "accuracy": 9,
                        "latency": 2,
                        "hallucination": 0,
                        "contamination": 0,
                        "question_type": "robotics",
                    },
                    "B": {
                        "accuracy": None,
                        "latency": None,
                        "question_type": "robotics",
                    },
                    "C": {
                        "accuracy": 7,
                        "latency": 4,
                        "hallucination": 0,
                        "contamination": 0,
                        "question_type": "robotics",
                    },
                },
            },
        ]

        summary = summarize(entries)

        self.assertEqual(summary["System A"]["intent_accuracy"], "N/A")
        self.assertEqual(summary["System B"]["intent_accuracy"], 1.0)
        self.assertEqual(summary["System C"]["intent_accuracy"], 0.0)
        self.assertEqual(summary["System A"]["context_switching_score"], 7.5)
        self.assertEqual(summary["System B"]["context_switching_score"], 10)
        self.assertEqual(summary["System C"]["context_switching_score"], 5.5)
        self.assertIsNotNone(summary["System A"]["overall_composite_score"])


if __name__ == "__main__":
    unittest.main()
