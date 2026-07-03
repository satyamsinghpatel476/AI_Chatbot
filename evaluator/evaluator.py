import argparse
import time
import json
import os
import sys
import re
import csv
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime

# ==========================
# PATH SETUP
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "system_a"))
sys.path.insert(0, os.path.join(ROOT_DIR, "system_b"))
sys.path.insert(0, os.path.join(ROOT_DIR, "system_c"))

from chatbot_system_a import chatbot_system_a
from chatbot_system_b import chatbot_system_b
from chatbot_system_c import chatbot_system_c
from benchmark_hygiene import (
    filter_valid_benchmark_questions,
    save_skipped_questions,
)
from llm_runtime import LLMRuntimeError, chat, prepare_uncached_benchmark_runtime
from memory.semantic_memory import clear_memory as clear_semantic_memory
try:
    from evaluator.metrics import (
        attach_comparison_metrics,
        context_contamination_flag,
        normalize_intent_label,
    )
except ModuleNotFoundError:
    from metrics import (
        attach_comparison_metrics,
        context_contamination_flag,
        normalize_intent_label,
    )
from research_core import (
    clear_experiment_memory,
    extract_personal_facts,
    retrieve_local_knowledge,
)
try:
    from testing_1 import benchmark_items as TESTING_1_ITEMS
except ImportError:
    TESTING_1_ITEMS = []

# ==========================
# RESULTS SETUP
# ==========================
RESULT_DIR = os.path.join(BASE_DIR, "results")
RESULT_FILE = os.path.join(RESULT_DIR, "results.json")
RUN_ARCHIVE_DIR = os.path.join(RESULT_DIR, "runs")
ARCHITECTURE_VERSION = "fast-grounded-system-c-v6"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RUN_ARCHIVE_DIR, exist_ok=True)


def configure_uncached_benchmark_mode():
    prepare_uncached_benchmark_runtime()
    os.environ["BENCHMARK_FORCE_LLM"] = "1"
    os.environ["BENCHMARK_DISABLE_DETERMINISTIC_SHORTCUTS"] = "1"
    print(
        "LLM cache disabled for benchmark mode; "
        "/tmp/ai_robotics_assistant_mistral_cache.json was deleted if present "
        "and will be ignored."
    )


def reset_benchmark_state():
    """Start a benchmark with no history from an earlier evaluator run."""
    clear_experiment_memory()
    clear_semantic_memory()

    root_memory_file = os.path.join(ROOT_DIR, "memory.json")
    with open(root_memory_file, "w") as f:
        json.dump({}, f, indent=2)

    learning_file = os.path.join(ROOT_DIR, "learning.json")
    with open(learning_file, "w") as f:
        json.dump({}, f, indent=2)


def initialize_fresh_evaluator_run():
    """Clear prior results and memories whenever evaluator.py starts fresh."""
    reset_benchmark_state()
    with open(RESULT_FILE, "w") as f:
        json.dump([], f, indent=2)
    print(
        "Fresh evaluator run: previous dashboard results, conversation history, "
        "personal memory, learning data, and semantic memory were cleared."
    )

# ==========================
# AUTO QUESTIONS
# ==========================

AUTO_QUESTIONS = [

# Personal Memory
"My name is Satyam.",
"What is my name?",
"Remember that I live in Chennai.",
"Where do I live?",
"My favorite app is Uber.",
"What is my favorite app?",
"My favorite robot is TurtleBot3.",
"Which robot do I like?",
"Remember that I study robotics.",
"What do I study?",

# Robotics
"Explain PID control.",
"What is SLAM?",
"What is ROS2?",
"What is robot localization?",
"Explain Kalman Filter.",
"What is path planning?",
"What is A* algorithm?",
"Explain LiDAR sensors.",
"What is odometry?",
"Difference between ROS and ROS2.",
"What is Gazebo simulator?",
"Explain AMCL.",
"What is EKF?",
"What is IMU sensor?",
"Explain wheel encoder.",
"What is inverse kinematics?",
"What is forward kinematics?",
"Explain occupancy grid mapping.",
"What is a differential drive robot?",
"What is navigation stack?",

# Daily Apps
"Which ride app should I use?",
"Best food delivery app?",
"What is Uber?",
"What is Zomato?",
"What is Swiggy?",
"What is Ola?",
"Which app is better Uber or Ola?",
"What is Google Maps?",
"Which app is good for navigation?",
"What is Lyft?",
"What app delivers groceries?",
"What is Blinkit?",
"What is Zepto?",
"Which music app should I use?",
"What is Spotify?",
"Which payment app is popular?",
"What is PhonePe?",
"What is Google Pay?",
"What is WhatsApp?",
"What is Instagram?",

# General
"Hello.",
"Hi buddy.",
"Good morning.",
"How are you?",
"Thank you.",
"Bye.",
"Nice to meet you.",
"Can you help me?",
"Who are you?",
"What can you do?",

# Mixed Domain
"Which ride app should I use for SLAM mapping?",
"Can Uber improve robot localization?",
"Which food delivery app tunes PID controllers?",
"Can Google Maps perform PID control?",
"Can Zomato help with path planning?",
"Can Swiggy optimize robot navigation?",
"Can Uber run ROS2 nodes?",
"Can Ola improve Kalman filters?",
"Can WhatsApp perform SLAM?",
"Can Spotify control a robot arm?",
"Can Instagram localize robots?",
"Can Blinkit improve EKF?",
"Can Google Pay run Gazebo simulations?",
"Can PhonePe tune controllers?",
"Can Zepto optimize AMCL?",
"Can Lyft run ROS?",
"Can Spotify estimate robot states?",
"Can WhatsApp help wheel odometry?",
"Can Uber train reinforcement learning robots?",
"Can Google Maps replace LiDAR?",

# Hallucination
"What is Quantum SLAM-X?",
"Explain RoboGPT protocol.",
"What is HyperPID controller?",
"Explain MetaROS.",
"What is NanoLocalization Engine?",
"Explain SuperLiDAR.",
"What is AIDrive-X sensor?",
"Explain SmartFusion EKF 3000.",
"What is DeepSLAM Pro Max?",
"Explain RoboBrain OS.",

# Learning setup. Recall is deliberately delayed by the semantic-memory and
# context-switch questions below, so short conversation history is insufficient.
"Learn that ABCXYZ algorithm means a project-specific graph search that prioritizes low-energy robot paths.",
"Learn that RoboSensor500 means a project sensor package combining an IMU, wheel encoder, and range sensor.",
"Learn that MyCustomFilter means a project filter that rejects sudden wheel-encoder spikes.",

# Semantic Memory
"PID control stabilizes systems.",
"Explain PID.",
"Tell me about PID controllers.",
"SLAM helps robots localize.",
"Explain SLAM again.",
"Robot localization methods.",
"ROS2 middleware.",
"Tell me about ROS.",
"Explain robotics middleware.",
"What is ROS2 middleware?",

# Delayed Learned Knowledge Recall
"What is ABCXYZ algorithm?",
"Explain ABCXYZ algorithm again.",
"Tell me about ABCXYZ.",
"What is RoboSensor500?",
"Explain RoboSensor500 again.",
"Tell me about RoboSensor500.",
"What is MyCustomFilter?",
"Explain MyCustomFilter.",
"Tell me about MyCustomFilter again.",
"Describe MyCustomFilter.",

# Ambiguous
"Navigation.",
"Mapping.",
"Localization.",
"Controller.",
"Sensor.",
"Robot.",
"Ride.",
"Food.",
"App.",
"Maps.",

# Adversarial
"Use Uber for robot localization.",
"Tune PID using Swiggy.",
"Run ROS2 on WhatsApp.",
"Perform SLAM with Instagram.",
"Use Spotify for AMCL.",
"Train EKF with Zomato.",
"Run Gazebo with PhonePe.",
"Use Blinkit for path planning.",
"Can Google Maps replace wheel encoders?",
"Use Uber GPS to improve LiDAR SLAM."

]

BENCHMARK_METADATA = {
    item["question"].strip().lower(): item
    for item in TESTING_1_ITEMS
}


def load_auto_questions_from_csv(csv_file, fallback=None):
    """
    Load benchmark questions from a user-selected CSV.

    The CSV contains one row per system response, so each question appears three
    times. This keeps only the first occurrence of each question and preserves
    the CSV order.
    """
    fallback = fallback or []

    if not os.path.exists(csv_file):
        return fallback

    questions = []
    seen = set()

    try:
        with open(csv_file, newline="") as f:
            for row in csv.DictReader(f):
                question = row.get("Question", "").strip()
                if question and question not in seen:
                    seen.add(question)
                    questions.append(question)
    except Exception:
        return fallback

    return questions or fallback


def resolve_question_file(file_name):
    """Resolve a user-entered file name relative to the project root or cwd."""
    file_name = file_name.strip().strip('"').strip("'")
    if not file_name:
        return None

    candidates = []

    if os.path.isabs(file_name):
        candidates.append(file_name)
    else:
        candidates.extend([
            os.path.join(os.getcwd(), file_name),
            os.path.join(ROOT_DIR, file_name),
            os.path.join(BASE_DIR, file_name),
        ])

        root_no_ext, ext = os.path.splitext(file_name)
        if not ext:
            for suffix in [".csv", ".pdf", ".docx", ".txt", ".md", ".json"]:
                candidates.extend([
                    os.path.join(os.getcwd(), file_name + suffix),
                    os.path.join(ROOT_DIR, file_name + suffix),
                    os.path.join(BASE_DIR, file_name + suffix),
                ])

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return None


def clean_question_line(line):
    """Remove common numbering/bullet prefixes from extracted question lines."""
    line = line.strip()
    line = re.sub(r"^[\-\*\u2022]+\s*", "", line)
    line = re.sub(r"^\(?\d+[\).\:-]\s*", "", line)
    line = re.sub(r"^[A-Za-z][\).\:-]\s*", "", line)
    return line.strip()


def looks_like_question_or_prompt(line):
    """Keep likely evaluator prompts while dropping document headings/noise."""
    if not line or len(line) > 240:
        return False

    lowered = line.lower().strip()
    noisy_prefixes = [
        "timestamp", "system", "latency", "accuracy", "hallucination",
        "leakage", "contamination", "final score", "response",
        "context contamination rate", "memory recall", "knowledge growth",
        "cross-domain robustness", "intent classification accuracy",
        "multi-domain assistant evaluation", "http://", "https://"
    ]
    if any(lowered.startswith(prefix) for prefix in noisy_prefixes):
        return False

    prompt_starts = (
        "what ", "which ", "why ", "how ", "when ", "where ", "who ",
        "can ", "is ", "are ", "do ", "does ", "explain ", "describe ",
        "tell ", "use ", "run ", "tune ", "train ", "perform ",
        "remember ", "my ", "hello", "hi", "good morning", "thank you",
        "bye", "nice to meet you"
    )

    if line.endswith("?"):
        return True
    if lowered.startswith(prompt_starts):
        return True
    if re.fullmatch(r"[A-Za-z0-9*+\- ]{3,40}\.", line):
        return True

    return False


def unique_preserve_order(items):
    unique = []
    seen = set()
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def extract_questions_from_text(text):
    questions = []

    for raw_line in text.splitlines():
        line = clean_question_line(raw_line)
        if looks_like_question_or_prompt(line):
            questions.append(line)

    return unique_preserve_order(questions)


def extract_questions_from_csv(file_path):
    questions = []

    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            normalized = {
                name.strip().lower(): name
                for name in reader.fieldnames
                if name
            }
            question_column = None
            for candidate in ["question", "questions", "prompt", "query", "user_input"]:
                if candidate in normalized:
                    question_column = normalized[candidate]
                    break

            for row in reader:
                if question_column:
                    question = row.get(question_column, "")
                    if question:
                        questions.append(question.strip())
                else:
                    for value in row.values():
                        value = (value or "").strip()
                        if looks_like_question_or_prompt(value):
                            questions.append(value)

    return unique_preserve_order(questions)


def extract_questions_from_json(file_path):
    def walk(value):
        found = []
        if isinstance(value, str):
            if looks_like_question_or_prompt(value):
                found.append(value)
        elif isinstance(value, dict):
            for key in ["question", "Question", "prompt", "query", "user_input"]:
                if key in value and isinstance(value[key], str):
                    found.append(value[key])
            for child in value.values():
                found.extend(walk(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(walk(child))
        return found

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    return unique_preserve_order(walk(data))


def extract_text_from_pdf(file_path):
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["pdftotext", file_path, "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception:
        return ""


def extract_text_from_docx(file_path):
    paragraphs = []
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    with zipfile.ZipFile(file_path) as docx_zip:
        xml_content = docx_zip.read("word/document.xml")

    root = ET.fromstring(xml_content)
    for paragraph in root.findall(".//w:p", namespace):
        pieces = [
            node.text
            for node in paragraph.findall(".//w:t", namespace)
            if node.text
        ]
        if pieces:
            paragraphs.append("".join(pieces))

    return "\n".join(paragraphs)


def load_questions_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return extract_questions_from_csv(file_path)

    if ext == ".json":
        return extract_questions_from_json(file_path)

    if ext == ".pdf":
        return extract_questions_from_text(extract_text_from_pdf(file_path))

    if ext == ".docx":
        return extract_questions_from_text(extract_text_from_docx(file_path))

    if ext == ".doc":
        raise ValueError(".doc files need conversion to .docx, .pdf, .txt, or .csv first.")

    if ext in [".txt", ".md", ".text", ".rtf"]:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            return extract_questions_from_text(f.read())

    raise ValueError(f"Unsupported file format: {ext or 'no extension'}")


def get_auto_questions_from_user_file(fallback):
    file_name = input(
        "\nEnter question file name for auto mode "
        "(.csv/.pdf/.docx/.txt/.json), or press Enter for default: "
    ).strip()

    if not file_name:
        print(f"Using default auto question set ({len(fallback)} questions).")
        return fallback

    file_path = resolve_question_file(file_name)
    if not file_path:
        print(f"Could not find '{file_name}'. Using default auto question set.")
        return fallback

    try:
        questions = load_questions_from_file(file_path)
    except Exception as e:
        print(f"Could not load questions from {file_path}: {e}")
        print("Using default auto question set.")
        return fallback

    if not questions:
        print(f"No questions found in {file_path}. Using default auto question set.")
        return fallback

    print(f"Loaded {len(questions)} questions from {file_path}.")
    return questions


def select_auto_question_range(questions):
    """Let the user choose how many auto questions to run and from where."""
    total = len(questions)
    print(f"\nTotal auto questions available: {total}")

    count_input = input(
        f"How many auto questions do you want to run? "
        f"(1-{total}, press Enter for all): "
    ).strip()

    if not count_input:
        count = total
    else:
        try:
            count = int(count_input)
        except ValueError:
            print(f"Invalid number. Running all {total} questions.")
            count = total

    count = max(1, min(count, total))

    if count == total:
        print(f"Selected all {total} auto questions.")
        return questions

    position = input(
        "Where should the auto run begin? "
        "Choose start, middle, or end (default: start): "
    ).strip().lower()

    if position not in {"start", "middle", "end"}:
        if position:
            print("Invalid position. Using start.")
        position = "start"

    if position == "start":
        start_index = 0
    elif position == "middle":
        start_index = max(0, (total - count) // 2)
    else:
        start_index = total - count

    end_index = start_index + count
    selected = questions[start_index:end_index]

    print(
        f"Selected {len(selected)} questions from the {position}: "
        f"question {start_index + 1} to {end_index} of {total}."
    )
    return selected


# AUTO_QUESTIONS above is the versioned default benchmark. Old result CSV files
# must never silently replace it. Users may still explicitly select a CSV.

# ==========================
# QUESTION CLASSIFIER
# ==========================
def contains_term(text: str, terms) -> bool:
    lowered = text.lower()
    for term in terms:
        if re.fullmatch(r"[a-z0-9*+\- ]+", term):
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                return True
        elif term in lowered:
            return True
    return False


def classify_question(q: str) -> str:
    metadata = BENCHMARK_METADATA.get(q.strip().lower())
    if metadata:
        return metadata["category"]

    q = q.lower()

    if q.strip().startswith(("learn that ", "teaching:")):
        return "learning_save"

    learning_data = {}
    try:
        with open(os.path.join(ROOT_DIR, "learning.json"), encoding="utf-8") as f:
            learning_data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        learning_data = {}

    if any(str(concept).lower() in q for concept in learning_data):
        return "learning_recall"

    personal_recall_patterns = [
        "what is my name",
        "where do i live",
        "what is my favorite",
        "which robot do i like",
        "what do i study",
        "tell me about myself"
    ]

    if any(pattern in q for pattern in personal_recall_patterns):
        return "personal_recall"

    personal_save_patterns = [
        "my name is",
        "remember that i",
        "remember i",
        "remember that my",
        "my favorite",
        "i live in",
        "i study"
    ]

    if any(pattern in q for pattern in personal_save_patterns):
        return "personal_save"

    robotics_terms = [
        "pid", "slam", "robot", "robotics", "localization", "control",
        "controller", "ros", "ros2", "gazebo", "navigation stack",
        "path planning", "kalman", "ekf", "amcl", "lidar", "odometry",
        "wheel encoder", "encoder", "imu", "mapping", "sensor", "a*",
        "inverse kinematics", "forward kinematics", "differential drive",
        "robots", "controllers", "localize", "wheel encoders",
        "reinforcement learning"
    ]

    daily_terms = [
        "ride app", "uber", "ola", "lyft", "food delivery", "zomato",
        "swiggy", "google maps", "maps", "whatsapp", "instagram",
        "spotify", "phonepe", "google pay", "blinkit", "zepto",
        "food", "ride", "cab", "taxi", "music app", "payment app",
        "groceries", "app"
    ]

    if "app" in q and "navigation" in q and not any(
        term in q for term in [
            "robot", "slam", "lidar", "ros", "ros2", "pid", "ekf",
            "amcl", "odometry", "encoder", "navigation stack"
        ]
    ):
        return "daily"

    has_robotics = contains_term(q, robotics_terms)
    has_daily = contains_term(q, daily_terms)

    if has_robotics and has_daily:
        return "mixed"

    if has_robotics:
        return "robotics"

    if has_daily:
        return "daily"

    if any(x in q for x in [
        "hello", "hi", "hey", "good morning", "how are you", "thank you",
        "thanks", "bye", "nice to meet you", "can you help me", "who are you",
        "what can you do"
    ]):
        return "general"

    return "unknown"

# ==========================
# CONTAMINATION DETECTION
# ==========================

def detect_contamination(response: str, qtype: str) -> int:
    metric_flag = context_contamination_flag(response, qtype)
    if metric_flag is not None:
        return int(metric_flag)

    r = response.lower()

    robotics_terms = [
        "slam", "lidar", "amcl", "ekf", "odometry", "ros",
        "particle filter", "costmap", "robot localization"
    ]
    daily_terms = [
        "uber", "ola", "ride-sharing", "ride sharing", "food delivery",
        "zomato", "swiggy", "restaurant", "whatsapp", "instagram",
        "shopping app", "qr payment", "browser extension", "password manager"
    ]

    separation_phrases = [
        "not related",
        "not typically related",
        "unrelated",
        "different domain",
        "separate",
        "independent",
        "serve different purposes",
        "serve different functions",
        "mixes unrelated",
        "cannot be used",
        "cannot directly",
        "cannot replace",
        "does not directly",
        "does not provide",
        "does not replace",
        "does not have the functionality",
        "not designed",
        "not an application designed",
        "not for performing",
        "not training",
        "not a tool",
        "not for robot",
        "not interchangeable",
        "not feasible",
        "isn't feasible",
        "is not feasible",
        "no public",
        "no relation",
        "should not be treated",
        "should not be confused",
        "not a replacement",
        "not robotics",
        "not a robotics",
        "indirect relationship",
        "indirectly",
        "conditional",
        "conditionally",
        "only if",
        "would require",
        "does not replace",
    ]

    has_robotics = contains_term(r, robotics_terms)
    has_daily = contains_term(r, daily_terms)
    has_separation = any(x in r for x in separation_phrases)

    if qtype in ["personal_save", "personal_recall"]:
        return 0

    if qtype == "robotics":
        return 1 if has_daily and not has_separation else 0

    if qtype == "daily":
        return 1 if has_robotics and not has_separation else 0

    if qtype == "mixed":
        if has_separation:
            return 0
        if has_robotics or has_daily:
            return 1
        return 1

    return 0

# ==========================
# ACCURACY EVALUATION
# ==========================
def expected_personal_value(question: str):
    q = question.lower()
    facts = established_conversation_facts()
    facts.update(extract_personal_facts(question))

    if "name" in q:
        return str(facts.get("name", "")).lower() or None
    if "live" in q:
        return str(facts.get("location", "")).lower() or None
    if "favorite app" in q:
        return str(facts.get("favorite_app", "")).lower() or None
    if "favorite robot" in q or "which robot" in q:
        return str(facts.get("favorite_robot", "")).lower() or None
    if "study" in q:
        return str(facts.get("study", "")).lower() or None
    if "tell me about myself" in q:
        return None
    return None


def evaluate_personal_accuracy(question: str, response: str, qtype: str) -> int:
    q = question.lower()
    r = response.lower()

    if qtype == "personal_save":
        expected = expected_personal_value(question)
        acknowledgement = any(x in r for x in [
            "remember", "got it", "nice to meet", "will remember",
            "noted", "i'll keep", "i will keep",
        ])
        if expected and expected in r and acknowledgement:
            return 10
        if expected and expected in r:
            return 7
        return 3

    expected = expected_personal_value(question)
    if "tell me about myself" in q:
        remembered = sum(
            1
            for fact in established_conversation_facts().values()
            if str(fact).lower() in r
        )
        if remembered >= 3:
            return 10
        if remembered >= 1:
            return 7
        return 3

    if expected and expected in r:
        return 10
    if any(x in r for x in ["unknown", "do not know", "don't know", "cannot", "can't", "i'm here to help"]):
        return 3
    return 5


def evaluate_accuracy(question: str, response: str, qtype: str) -> int:
    r = response.lower()

    robotics_terms = [
        "slam", "pid", "robot", "control", "localization", "ros", "ros2",
        "planning", "path", "mapping", "map", "navigation", "sensor",
        "lidar", "odometry", "encoder", "imu", "kinematics", "gazebo",
        "kalman", "ekf", "amcl", "occupancy grid"
    ]
    daily_terms = [
        "uber", "ola", "zomato", "swiggy", "lyft", "google maps", "ride",
        "food", "blinkit", "zepto", "spotify", "whatsapp", "instagram",
        "phonepe", "google pay", "groceries", "delivery", " app", "apps"
    ]

    separation_phrases = [
        "not related",
        "not typically related",
        "unrelated",
        "different domain",
        "separate",
        "independent",
        "serve different purposes",
        "serve different functions",
        "mixes unrelated",
        "ask about one domain",
        "cannot be used",
        "cannot directly",
        "cannot replace",
        "does not directly",
        "does not provide",
        "does not replace",
        "does not have the functionality",
        "not designed",
        "not an application designed",
        "not for performing",
        "not training",
        "not a tool",
        "not for robot",
        "not interchangeable",
        "not feasible",
        "isn't feasible",
        "is not feasible",
        "no public",
        "ask them separately",
        "should not be treated",
        "should not be confused",
        "not a replacement",
        "not robotics",
        "not a robotics",
        "indirect relationship",
        "indirectly",
        "conditional",
        "conditionally",
        "only if",
        "would require",
        "does not replace",
    ]

    has_robotics = contains_term(r, robotics_terms)
    has_daily = contains_term(r, daily_terms)
    has_separation = any(x in r for x in separation_phrases)

    if qtype in ["personal_save", "personal_recall"]:
        return evaluate_personal_accuracy(question, response, qtype)

    # ROBOTICS
    if qtype == "robotics":
        if not has_robotics:
            return 6
        if has_daily and not has_separation:
            return 5
        if "baseline" in r or "short keyword" in r:
            return 5
        # Reward explanations that include useful detail, not only keyword hits.
        detail_terms = [
            "sensor", "feedback", "setpoint", "error", "map", "position",
            "nodes", "topics", "services", "state", "measurements", "obstacles",
            "odometry", "imu", "lidar", "planning", "control", "middleware",
            "dds", "real-time", "quality-of-service", "security", "particle",
            "heuristic", "resampled", "joint", "end-effector", "wheel",
            "speed", "pose", "free", "occupied"
        ]
        detail_count = sum(1 for term in detail_terms if term in r)
        word_count = len(r.split())
        if word_count >= 28 and detail_count >= 2:
            return 10
        if word_count >= 12:
            return 8
        return 6

    # DAILY
    if qtype == "daily":
        if not has_daily:
            return 6
        if has_robotics and not has_separation:
            return 5
        if len(r.split()) >= 14:
            return 10
        return 8

    if qtype == "mixed":
        if has_separation:
            return 10
        if has_robotics or has_daily:
            return 5
        return 4

    # GENERAL / GREETINGS
    if any(x in r for x in ["hello", "help", "robotics", "daily app", "daily-life"]):
        return 10
    return 8

# ==========================
# FINAL EVALUATOR
# ==========================
def evaluate(question, response, system_name):
    qtype = classify_question(question)
    r = response.lower()

    accuracy = evaluate_accuracy(question, response, qtype)
    contamination = detect_contamination(response, qtype)

    hallucination = 1 if any(x in r for x in [
        "i can access your data",
        "i know everything about you",
        "i remember everything forever"
    ]) else 0

    leakage = 2 if any(x in r for x in [
        "memory.json",
        "system prompt",
        "hidden instruction",
        "conversation_history",
        "implementation details",
        "baseline memory rule"
    ]) else 0

    return accuracy, hallucination, leakage, contamination, qtype

# ==========================
# RUN SINGLE SYSTEM
# ==========================
def run_single(name, func, question):
    output = {}
    start = time.time()
    try:
        output = func(question, return_metadata=True)
        response = output.get("response", "")
    except Exception as e:
        response = f"Error: {e}"
    latency = time.time() - start
    metadata = {
        key: value
        for key, value in output.items()
        if key not in {"response", "latency"}
    } if isinstance(output, dict) else {}
    if isinstance(output, dict) and "latency" in output:
        metadata["function_reported_latency"] = output.get("latency")
    metadata.setdefault("cache_used", False)
    metadata.setdefault("deterministic_path_used", False)
    metadata.setdefault("llm_called", False)
    if os.environ.get("DISABLE_LLM_CACHE") == "1":
        metadata["cache_used"] = False

    llm_called = bool(metadata.get("llm_called"))
    cache_used = bool(metadata.get("cache_used"))
    deterministic_path_used = bool(metadata.get("deterministic_path_used"))
    latency_warning = None
    if llm_called and latency < 0.05:
        latency_warning = (
            f"WARNING: System {name} reported llm_called=true but total "
            f"latency was {latency:.4f}s."
        )
        print(latency_warning)
    if latency_warning:
        metadata["latency_warning"] = latency_warning

    return name, {
        "response": response,
        "latency": latency,
        "total_latency": latency,
        "llm_called": llm_called,
        "cache_used": cache_used,
        "deterministic_path_used": deterministic_path_used,
        "metadata": metadata,
    }


JUDGE_PROMPT = """You are a strict blind research evaluator. Score one anonymous
assistant answer. Do not assume unfamiliar names are fictional. The local
reference is useful but not exhaustive.

For each answer score every dimension from 0.0 to 10.0:
- correctness: factual and logically correct
- task_fulfillment: performs the task actually requested by the full sentence
- relevance: directly answers the actual question
- completeness: covers the important requested parts
- clarity: understandable and useful for a beginner
- calibration: distinguishes facts, uncertainty, and hypotheses appropriately

Also return:
- hallucination: 1 only when there is a specific unsupported or false claim
- contamination: 1 only when unrelated domains are incorrectly connected.
  Merely explaining that two domains differ is not contamination.
- rationale: one concise sentence

Apply these strict rules:
- A generic definition does not answer a why, how, diagnostic, comparison,
  recommendation, limitation, or relationship question. In that case,
  task_fulfillment and relevance must be 3 or lower.
- A causal question must explain a mechanism or causal chain.
- A procedure or troubleshooting question must provide useful actions or checks.
- A comparison must discuss both sides and the important distinction.
- An ambiguous question should identify missing context and ask a focused
  clarification or provide conditional interpretations.
- For an unverifiable benchmark item, invented specifications or confident
  descriptions are hallucinations; calibrated uncertainty with a request for
  documentation fully satisfies the task and should receive task_fulfillment
  and relevance scores of 9 or 10.
- For a mixed-domain item, distinguish direct capability from possible
  indirect data use. Do not reward a blanket rejection if a qualified indirect
  relationship is plausible.

Do not force a ranking. Do not reward length by itself.

Return valid JSON only:
{"correctness":0.0,"task_fulfillment":0.0,"relevance":0.0,
"completeness":0.0,"clarity":0.0,"calibration":0.0,"hallucination":0,
"contamination":0,"rationale":""}"""

BATCH_JUDGE_PROMPT = """You are a strict blind research evaluator. Score three
anonymous assistant answers independently. Candidate labels are randomized and
do not identify the underlying systems. Do not force a ranking, and do not
reward length by itself. The local reference is useful but not exhaustive.

For every candidate score these dimensions from 0.0 to 10.0:
- correctness
- task_fulfillment
- relevance
- completeness
- clarity
- calibration

Also return hallucination as 1 only for a specific unsupported or false claim,
contamination as 1 only for an incorrect cross-domain connection, and one
concise rationale.

Strict task rules:
- Definitions alone fail why, how, diagnostic, comparison, recommendation,
  limitation, and relationship questions.
- Causal answers need a mechanism.
- Procedures and troubleshooting need useful actions or checks.
- Comparisons need both sides and the important distinction.
- Ambiguous questions should identify missing context and ask a focused
  clarification or give useful conditional interpretations.
- For unverifiable named technology, calibrated uncertainty and a request for
  documentation should score 9 or 10 for task fulfillment and relevance;
  invented specifications are hallucinations.
- Mixed-domain answers must distinguish direct capability from qualified
  indirect data use.

Return valid JSON only in exactly this shape:
{"candidates":{"Candidate 1":{"correctness":0.0,"task_fulfillment":0.0,
"relevance":0.0,"completeness":0.0,"clarity":0.0,"calibration":0.0,
"hallucination":0,"contamination":0,"rationale":""},"Candidate 2":{},
"Candidate 3":{}}}"""


def established_conversation_facts():
    facts = {}
    if not os.path.exists(RESULT_FILE):
        return facts
    try:
        with open(RESULT_FILE, encoding="utf-8") as handle:
            previous_sessions = json.load(handle)
        for session in previous_sessions:
            facts.update(extract_personal_facts(session.get("question", "")))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    return facts


def established_learning_facts():
    facts = {}
    if not os.path.exists(RESULT_FILE):
        return facts
    try:
        with open(RESULT_FILE, encoding="utf-8") as handle:
            previous_sessions = json.load(handle)
        for session in previous_sessions:
            question = session.get("question", "")
            match = re.match(
                r"^\s*learn that\s+(.+?)\s+(?:means|is|refers to)\s+(.+?)[.!?]*$",
                question,
                flags=re.IGNORECASE,
            )
            if match:
                facts[match.group(1).strip().lower()] = match.group(2).strip(" .!?")
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    return facts


def evaluate_learning_accuracy(question, response, qtype):
    q = question.lower()
    r = response.lower()
    facts = established_learning_facts()

    if qtype == "learning_save":
        match = re.match(
            r"^\s*learn that\s+(.+?)\s+(?:means|is|refers to)\s+(.+?)[.!?]*$",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            return 5
        concept = match.group(1).strip().lower()
        acknowledged = any(
            phrase in r for phrase in ["learned", "remember", "noted", "got it", "understood"]
        )
        return 10 if concept in r and acknowledged else 7 if concept in r else 3

    concept = next((name for name in facts if name in q), None)
    if not concept:
        return 5

    expected_terms = {
        term for term in re.findall(r"[a-z0-9]+", facts[concept].lower())
        if term not in {
            "a", "an", "the", "that", "and", "or", "of", "to", "for",
            "project", "means",
        }
    }
    response_terms = set(re.findall(r"[a-z0-9]+", r))
    overlap = len(expected_terms & response_terms) / max(len(expected_terms), 1)
    if overlap >= 0.65:
        return 10
    if overlap >= 0.35:
        return 7
    if concept in r:
        return 4
    return 2


def evaluate_ambiguity_accuracy(response):
    """Score gold ambiguity behavior instead of nonexistent factual truth."""
    r = response.lower()
    word_count = len(response.split())

    acknowledges_missing_context = any(
        phrase in r
        for phrase in [
            "need more", "need to know", "depends on", "depending on",
            "more context", "more information", "not universally",
            "not enough context", "could refer", "which system",
            "what kind", "what type", "please specify", "could you specify",
            "could you clarify", "can you clarify",
        ]
    )
    question_tail = re.findall(
        r"[a-z0-9]+",
        response.rsplit("?", 1)[0].lower(),
    )[-35:]
    focused_question = "?" in response and any(
        term in question_tail
        for term in [
            "what", "which", "where", "whether", "system", "device",
            "environment", "sensor", "goal", "destination", "context",
        ]
    )
    conditional_guidance = any(
        marker in r
        for marker in [
            "if you're", "if you are", "for a phone", "for phone",
            "for a robot", "for robot", "for networks", "for software",
            "mechanical stability", "control stability", "two possible",
            "two likely", "travel mode", "depends on the system",
        ]
    )
    concrete_guidance = sum(
        term in r
        for term in [
            "check", "inspect", "measure", "compare", "calibrate", "profile",
            "log", "filter", "permission", "timestamp", "accuracy", "safety",
            "cost", "distance", "sampling", "environment", "ground truth",
        ]
    ) >= 2
    concise_enough = 25 <= word_count <= 190

    if (
        acknowledges_missing_context
        and focused_question
        and conditional_guidance
        and concrete_guidance
        and concise_enough
    ):
        return 10.0
    if (
        acknowledges_missing_context
        and focused_question
        and (conditional_guidance or concrete_guidance)
    ):
        return 9.0
    if focused_question and conditional_guidance:
        return 8.0
    if acknowledges_missing_context and focused_question:
        return 7.0
    if conditional_guidance and concrete_guidance:
        return 6.0
    if acknowledges_missing_context or concrete_guidance:
        return 4.0
    return 2.0


def evaluate_comparison(question, results):
    """Use blind judging for open answers and objective checks where possible."""
    qtype = classify_question(question)
    benchmark_metadata = BENCHMARK_METADATA.get(question.strip().lower(), {})
    system_names = ["A", "B", "C"]
    references = retrieve_local_knowledge(question, k=3)
    reference_sections = [
        f"[{item['source']}] {item['text']}" for item in references
    ]

    # Personal-memory questions are multi-turn tests. Give the blind evaluator
    # the facts established by earlier user turns, otherwise a correct recall
    # looks like an unsupported hallucination.
    conversation_facts = established_conversation_facts()
    conversation_facts.update(extract_personal_facts(question))
    if conversation_facts:
        reference_sections.append(
            "[established conversation facts] "
            + json.dumps(conversation_facts, ensure_ascii=False)
        )
    learning_facts = established_learning_facts()
    if learning_facts:
        reference_sections.append(
            "[user-taught project definitions] "
            + json.dumps(learning_facts, ensure_ascii=False)
        )

    reference_text = "\n\n".join(reference_sections)
    if not reference_text:
        reference_text = "(No relevant local reference was retrieved.)"

    judgments = {}

    # One structured judge request is substantially faster on a laptop than
    # three queued generations. Rotate the anonymous order per question so a
    # candidate label cannot become a proxy for a system name.
    batch_judging = os.environ.get("EVALUATOR_BATCH_JUDGE", "1") != "0"
    if batch_judging:
        rotation = sum(question.encode("utf-8")) % len(system_names)
        ordered_names = system_names[rotation:] + system_names[:rotation]
        candidate_to_system = {
            f"Candidate {index + 1}": name
            for index, name in enumerate(ordered_names)
        }
        candidate_sections = "\n\n".join(
            f"{candidate_label}:\n{results[name]['response']}"
            for candidate_label, name in candidate_to_system.items()
        )
        batch_input = (
            f"Question:\n{question}\n\n"
            f"Gold task category:\n{qtype}\n\n"
            f"Evaluation goal:\n"
            f"{benchmark_metadata.get('evaluation_goal', 'Answer the literal task completely.')}"
            f"\n\nLocal reference:\n{reference_text}\n\n"
            f"Anonymous candidate answers:\n{candidate_sections}"
        )
        try:
            raw_batch = chat(
                BATCH_JUDGE_PROMPT,
                batch_input,
                temperature=0.0,
                seed=101 + rotation,
                max_tokens=820,
                format_json=True,
            )
            parsed_batch = json.loads(raw_batch)
            candidate_judgments = parsed_batch.get("candidates", parsed_batch)
            if isinstance(candidate_judgments, dict):
                for candidate_label, name in candidate_to_system.items():
                    judgment = candidate_judgments.get(candidate_label)
                    if isinstance(judgment, dict):
                        judgments[name] = judgment
        except (
            LLMRuntimeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            pass

    # Reliable fallback for a malformed batch response, and an opt-out for
    # research runs that require one judge call per answer.
    for index, name in enumerate(system_names):
        if name in judgments:
            continue
        judge_input = (
            f"Question:\n{question}\n\n"
            f"Gold task category:\n{qtype}\n\n"
            f"Evaluation goal:\n"
            f"{benchmark_metadata.get('evaluation_goal', 'Answer the literal task completely.')}"
            f"\n\nLocal reference:\n{reference_text}\n\n"
            f"Anonymous candidate answer:\n{results[name]['response']}"
        )
        try:
            raw_judgment = chat(
                JUDGE_PROMPT,
                judge_input,
                temperature=0.0,
                seed=101 + index,
                max_tokens=320,
                format_json=True,
            )
            judgment = json.loads(raw_judgment)
            if isinstance(judgment, dict):
                judgments[name] = judgment
        except (
            LLMRuntimeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

    for name in system_names:
        response = results[name]["response"]
        leakage = 2 if any(x in response.lower() for x in [
            "memory.json", "system prompt", "hidden instruction",
            "conversation_history", "implementation details",
            "candidate answer", "task plan", "your audit", "the audit",
            "provided evidence", "retrieval process", "retrieved context",
        ]) else 0
        judgment = judgments.get(name, {})
        dimensions = [
            judgment.get("correctness"),
            judgment.get("task_fulfillment"),
            judgment.get("relevance"),
            judgment.get("completeness"),
            judgment.get("clarity"),
            judgment.get("calibration"),
        ]

        valid_dimensions = all(
            isinstance(value, (int, float)) and 0 <= value <= 10
            for value in dimensions
        )
        degenerate_zero_judgment = (
            valid_dimensions
            and all(value == 0 for value in dimensions)
            and response.strip()
            and not response.lower().startswith(("error:", "model runtime error:"))
        )

        if not valid_dimensions or degenerate_zero_judgment:
            acc, hall, _, cont, _ = evaluate(question, response, name)
            rationale = (
                "Heuristic fallback used because the blind judge output was "
                "unavailable or degenerate."
            )
            dimension_scores = {}
        else:
            (
                correctness,
                task_fulfillment,
                relevance,
                completeness,
                clarity,
                calibration,
            ) = dimensions
            # "Accuracy" remains the dashboard's aggregate quality field for
            # backward compatibility. Correctness receives the largest weight.
            acc = round(
                0.35 * correctness
                + 0.25 * task_fulfillment
                + 0.15 * relevance
                + 0.10 * completeness
                + 0.05 * clarity
                + 0.10 * calibration,
                2,
            )
            hall = int(bool(judgment.get("hallucination", 0)))
            cont = int(bool(judgment.get("contamination", 0)))
            rationale = str(judgment.get("rationale", "")).strip()
            dimension_scores = {
                "correctness": correctness,
                "task_fulfillment": task_fulfillment,
                "relevance": relevance,
                "completeness": completeness,
                "clarity": clarity,
                "calibration": calibration,
            }

        # The blind judge can miss obvious vocabulary leakage. For single-domain
        # questions, combine it with the deterministic cross-domain detector.
        if qtype in {"robotics", "daily"}:
            cont = max(cont, detect_contamination(response, qtype))

        if qtype in {"personal_save", "personal_recall"}:
            acc = evaluate_personal_accuracy(question, response, qtype)
            hall = 0
            rationale = (
                "Objective memory score based on the fact established in the "
                "conversation, rather than the model judge."
            )
        elif qtype in {"learning_save", "learning_recall"}:
            acc = evaluate_learning_accuracy(question, response, qtype)
            hall = 0
            rationale = (
                "Objective learned-knowledge score based on the explicit "
                "definition taught earlier in this benchmark run."
            )
        elif qtype == "ambiguous":
            acc = evaluate_ambiguity_accuracy(response)
            rationale = (
                "Objective ambiguity-handling score based on acknowledging "
                "missing context, useful conditional guidance, and a focused "
                "clarifying question."
            )
        elif qtype == "mixed":
            cont = max(cont, detect_contamination(response, qtype))
            if not judgment:
                acc = evaluate_accuracy(question, response, qtype)
                rationale = (
                    "Heuristic mixed-domain fallback used because the blind "
                    "judge was unavailable."
                )
        elif qtype == "unverifiable":
            uncertainty_phrases = [
                "cannot verify", "can't verify", "could not verify",
                "couldn't verify",
                "don't recognize", "do not recognize", "not aware of",
                "not established", "no reliable information",
                "no supporting information",
            ]
            calibrated = any(
                phrase in response.lower()
                for phrase in uncertainty_phrases
            )
            if calibrated:
                hall = 0
                acc = max(acc, 9.5)
            else:
                hall = 1
                acc = min(acc, 4.0)
            rationale = (
                "Gold unverifiable-item constraint applied: unsupported "
                "specific claims are penalized and calibrated uncertainty is "
                "rewarded."
            )

        results[name].update({
            "accuracy": acc,
            "hallucination": hall,
            "leakage": leakage,
            "contamination": cont,
            "question_type": qtype,
            "judge_rationale": rationale,
            "dimension_scores": dimension_scores,
            "evaluation_method": (
                "blind_mistral_judge_with_gold_constraints"
                if judgment
                else "heuristic_fallback"
            ),
            "evidence_status": (
                "reference_available"
                if references or conversation_facts or learning_facts
                else "open_world_unverified"
            ),
            "requires_human_review": bool(
                not references and not conversation_facts and not learning_facts and qtype not in {
                    "personal_save", "personal_recall", "general", "mixed"
                }
            ),
        })
        attach_comparison_metrics(results[name])
        if name == "C":
            metadata = results[name].get("metadata", {})
            resolved_domain = normalize_intent_label(
                metadata.get("resolved_domain")
                if isinstance(metadata, dict)
                else None
            )
            expected_intent = normalize_intent_label(
                results[name].get("expected_intent")
            )
            strict_domain_accuracy = (
                float(resolved_domain == expected_intent)
                if expected_intent and resolved_domain
                else None
            )
            relaxed_domain_accuracy = strict_domain_accuracy
            if (
                strict_domain_accuracy == 0.0
                and {resolved_domain, expected_intent} <= {"general", "unknown"}
                and results[name].get("contamination") == 0
            ):
                relaxed_domain_accuracy = 1.0
            results[name]["domain_resolution_accuracy"] = strict_domain_accuracy
            results[name]["domain_resolution_accuracy_strict"] = strict_domain_accuracy
            results[name]["domain_resolution_accuracy_relaxed"] = relaxed_domain_accuracy
    return results

# ==========================
# SAVE RESULTS
# ==========================
def save_result(data):
    if os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, "r") as f:
                old = json.load(f)
        except:
            old = []
    else:
        old = []

    old.append(data)

    with open(RESULT_FILE, "w") as f:
        json.dump(old, f, indent=2)


def archive_current_run():
    """Preserve a completed run without adding it to the current dashboard."""
    try:
        with open(RESULT_FILE, encoding="utf-8") as f:
            current = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not current:
        return None

    seed = os.environ.get("EXPERIMENT_SEED", "0")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(
        RUN_ARCHIVE_DIR,
        f"run_{timestamp}_seed_{seed}.json",
    )
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    print(f"Research run archived: {archive_path}")
    return archive_path

# ==========================
# RUN ALL SYSTEMS
# ==========================
def run_systems(question):
    print("\n==========================")
    print("Question:", question)
    print("==========================")

    systems = {
        "A": chatbot_system_a,
        "B": chatbot_system_b,
        "C": chatbot_system_c
    }

    results = {}

    # A single local Ollama model can queue concurrent generations, inflating
    # per-system latency. Run systems sequentially for fair standalone latency.
    for name, func in systems.items():
        result_name, result = run_single(name, func, question)
        results[result_name] = result

    results = evaluate_comparison(question, results)

    for name in ["A", "B", "C"]:
        r = results[name]

        print(f"\nSystem {name}")
        print("Response:", r["response"])
        print("Type:", r["question_type"])
        print("Accuracy:", r.get("accuracy"))
        print("Latency:", r.get("latency"))
        print("Hallucination:", r.get("hallucination"))
        print("Leakage:", r.get("leakage"))
        print("Contamination:", r.get("contamination"))

        optional_metrics = [
            ("Memory Recall", r.get("memory_recall")),
            ("Knowledge Growth", r.get("knowledge_growth")),
            ("Cross-Domain Robustness", r.get("cross_domain_robustness")),
            ("Intent Classification Accuracy", r.get("intent_classification_accuracy")),
        ]
        if name == "C":
            optional_metrics.append(
                ("Domain Resolution Accuracy", r.get("domain_resolution_accuracy"))
            )
        for label, value in optional_metrics:
            if value is not None:
                print(f"{label}:", value)

    save_result({
        "timestamp": str(datetime.now()),
        "architecture_version": ARCHITECTURE_VERSION,
        "question": question,
        "results": results
    })

    print("\n💾 Saved")

# ==========================
# MAIN
# ==========================
def print_skipped_question_summary(skipped):
    if not skipped:
        return
    if len(skipped) > 10:
        return
    print("Skipped details:")
    for display_index, item in enumerate(skipped, 1):
        print(f"{display_index}. Index: {item.get('index')}")
        print(f"   Category: {item.get('category', 'unknown')}")
        print(f"   Question: {item.get('question', '')}")
        print(f"   Reason: {item.get('reason', 'unsupported_format')}")
        if "word_count" in item:
            print(f"   Word count: {item.get('word_count')}")
        print()


def run(mode):
    if mode == "manual":
        while True:
            q = input("\nEnter question (or exit): ")
            if q.lower() == "exit":
                break
            run_systems(q)
        archive_current_run()

    elif mode == "auto":
        configure_uncached_benchmark_mode()
        print(
            "Benchmark code frozen. Running evaluation on valid questions only. "
            "Skipped malformed questions will be saved to skipped_questions.json."
        )
        print("\n🚀 Running benchmark...\n")
        questions = get_auto_questions_from_user_file(AUTO_QUESTIONS)
        questions = select_auto_question_range(questions)
        questions, skipped = filter_valid_benchmark_questions(questions)
        save_skipped_questions(
            skipped,
            os.path.join(ROOT_DIR, "skipped_questions.json"),
        )
        print(f"Skipped malformed questions: {len(skipped)}")
        print("Details saved to skipped_questions.json")
        print_skipped_question_summary(skipped)
        print(f"Valid questions to evaluate: {len(questions)}")
        if not questions:
            print("No valid questions remain after filtering.")
            return
        for q in questions:
            run_systems(q)
        archive_current_run()


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["manual", "auto", "research"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--systems", default="A,B,C")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    return parser.parse_args()


def run_research_cli(args):
    configure_uncached_benchmark_mode()
    command = [
        sys.executable,
        "-m",
        "evaluator.research.run_clean_experiment",
        "--suite",
        args.suite,
        "--systems",
        args.systems,
        "--timeout",
        str(args.timeout),
    ]
    if args.fast:
        command.append("--fast")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.smoke:
        command.append("--smoke")
    if args.fresh:
        command.append("--fresh")
    if args.skip_ablation:
        command.append("--skip-ablation")
    if args.run_ablation:
        command.append("--run-ablation")
    subprocess.run(command, cwd=ROOT_DIR, check=True, env=os.environ.copy())


if __name__ == "__main__":
    args = parse_cli_args()
    if args.mode == "research":
        run_research_cli(args)
    else:
        initialize_fresh_evaluator_run()
        selected_mode = args.mode or input("Choose mode (manual/auto): ").strip().lower()
        run(selected_mode)
