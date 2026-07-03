import unittest

from benchmark_hygiene import is_valid_benchmark_question
from chatbot_system_c import (
    _deterministic_relationship_response,
    _needs_clarification,
    _relationship_hint,
    _response_mentions_current_question,
    _resolve_domain,
    chatbot_system_c,
)


class SystemCPolicyTests(unittest.TestCase):
    def test_daily_navigation_phone_question_stays_daily(self):
        self.assertEqual(
            _resolve_domain("How do I reduce phone battery drain during navigation?"),
            "daily",
        )

    def test_studying_online_question_stays_daily(self):
        self.assertEqual(
            _resolve_domain("How can I reduce distractions while studying online?"),
            "daily",
        )

    def test_studying_online_question_gets_daily_answer(self):
        result = chatbot_system_c(
            "How can I reduce distractions while studying online?",
            return_metadata=True,
        )
        response = result["response"].lower()

        self.assertEqual(result["resolved_domain"], "daily")
        self.assertTrue(result["llm_called"])
        self.assertIn("daily_digital_assistance.txt", result["retrieved_sources"])
        self.assertIn("distractions", response)
        self.assertIn("studying online", response)
        self.assertTrue(
            result["response_validation"]["mentions_current_question"]
        )

    def test_robotics_and_consumer_app_question_is_mixed(self):
        self.assertEqual(
            _resolve_domain("Can Uber improve robot localization?"),
            "mixed",
        )

    def test_robotics_gps_question_stays_robotics(self):
        self.assertEqual(
            _resolve_domain("How do robots navigate without GPS?"),
            "robotics",
        )

    def test_general_why_question_is_not_forced_to_ambiguity(self):
        self.assertEqual(
            _resolve_domain("Why should an assistant refuse to invent unknown technical terms?"),
            "unknown",
        )
        self.assertFalse(
            _needs_clarification(
                "Why should an assistant refuse to invent unknown technical terms?",
                "unknown",
            )
        )

    def test_mixed_relationship_response_uses_required_structure(self):
        query = "Can restaurant ratings replace robot perception?"
        domain = _resolve_domain(query)
        hint = _relationship_hint(query, domain)
        response = _deterministic_relationship_response(query, domain, hint)

        self.assertIn("Direct Answer", response)
        self.assertIn("Relationship Type:\nUnsupported", response)
        self.assertIn("Robotics Perspective", response)
        self.assertIn("Daily-Life Perspective", response)
        self.assertIn("Important Difference", response)
        self.assertIn("Final Conclusion", response)

    def test_stale_mixed_answer_fails_current_question_validation(self):
        valid, missing = _response_mentions_current_question(
            "Compare SLAM and ride-sharing apps.",
            "Restaurant ratings cannot replace robot perception.",
        )

        self.assertFalse(valid)
        self.assertIn("slam", missing)
        self.assertIn("ride-sharing", missing)

    def test_screen_time_requires_screen_time_in_answer(self):
        valid, missing = _response_mentions_current_question(
            "How can I reduce screen time?",
            "Reduce phone battery drain by lowering brightness.",
        )

        self.assertFalse(valid)
        self.assertEqual(missing, ["screen time"])

    def test_benchmark_question_validation_rejects_fragments(self):
        self.assertFalse(is_valid_benchmark_question("terms."))
        self.assertFalse(is_valid_benchmark_question("without contamination."))
        self.assertTrue(is_valid_benchmark_question("What is SLAM?"))
        self.assertTrue(
            is_valid_benchmark_question("How can a beginner debug SLAM?")
        )

    def test_unknown_general_question_stays_calibrated(self):
        result = chatbot_system_c(
            "How can I make the system more reliable?",
            return_metadata=True,
        )
        response = result["response"].lower()

        self.assertEqual(result["resolved_domain"], "unknown")
        self.assertIn("depends on the system", response)
        self.assertIn("robot", response)
        self.assertIn("software app", response)
        self.assertIn("network/service", response)
        self.assertEqual(result["response"].count("?"), 1)
        self.assertTrue(result["deterministic_path_used"])
        self.assertFalse(result["llm_called"])

    def test_slam_ride_sharing_comparison_uses_current_entities(self):
        result = chatbot_system_c(
            "Compare SLAM and ride-sharing apps.",
            return_metadata=True,
        )
        response = result["response"].lower()

        self.assertIn("slam", response)
        self.assertIn("ride-sharing", response)
        self.assertNotIn("restaurant ratings", response)
        self.assertTrue(
            result["response_validation"]["mentions_current_question"]
        )

    def test_unknown_named_technology_refuses_to_invent(self):
        result = chatbot_system_c(
            "What is ZorplexNav-X in robotics?",
            return_metadata=True,
        )

        self.assertIn("I cannot verify", result["response"])
        self.assertIn("standard or well-established concept", result["response"])
        self.assertIn("unsupported_entity_guardrail", result["pipeline"])

    def test_ambiguous_tracking_question_gets_one_clarification(self):
        result = chatbot_system_c(
            "How can I improve tracking accuracy?",
            return_metadata=True,
        )

        self.assertIn("ambiguous", result["response"].lower())
        self.assertIn("Phone/GPS tracking", result["response"])
        self.assertIn("Robot localization", result["response"])
        self.assertEqual(result["response"].count("?"), 1)
        self.assertIn("ambiguity_guardrail_no_mistral", result["pipeline"])


if __name__ == "__main__":
    unittest.main()
