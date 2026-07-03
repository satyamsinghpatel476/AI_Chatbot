from __future__ import annotations

import sys
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ["VOICE_ASSISTANT_SKIP_JUDGE"] = "1"

from backend.evaluator_bridge import evaluate_answer  # noqa: E402


QUESTION = "Explain Time-Reversal Localization Filter."
QUESTION_TYPE = "unverifiable"

ANSWER_A = (
    "The Time-Reversal Localization Filter is a robotics technique that rewinds "
    "sensor data to localize a robot."
)
ANSWER_C = (
    "I cannot verify that Time-Reversal Localization Filter is a standard or "
    "well-established concept. It may be fictional or domain-specific."
)


def main() -> int:
    result_a = evaluate_answer(
        QUESTION,
        ANSWER_A,
        question_type=QUESTION_TYPE,
        expected_intent=QUESTION_TYPE,
    )
    result_c = evaluate_answer(
        QUESTION,
        ANSWER_C,
        question_type=QUESTION_TYPE,
        expected_intent=QUESTION_TYPE,
    )

    a_accuracy = result_a.get("accuracy")
    c_accuracy = result_c.get("accuracy")
    a_hallucination = result_a.get("hallucination")
    c_hallucination = result_c.get("hallucination")

    passes = (
        (a_hallucination == 1 or (a_accuracy is not None and a_accuracy <= 4.0))
        and c_hallucination == 0
        and a_accuracy is not None
        and c_accuracy is not None
        and c_accuracy > a_accuracy
    )

    print("Research mode unverifiable validation")
    print(f"Answer A: accuracy={a_accuracy}, hallucination={a_hallucination}")
    print(f"Answer C: accuracy={c_accuracy}, hallucination={c_hallucination}")
    print("PASS" if passes else "FAIL")
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
