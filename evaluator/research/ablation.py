import json
import os
import re
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from intent_model import get_intent_classifier
from llm_runtime import LLMRuntimeError, chat, was_last_cache_used
from research_core import retrieve_local_knowledge


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MINILM_PATH = os.path.join(ROOT_DIR, "models", "all-MiniLM-L6-v2")

BASE_PROMPT = """You are a local assistant for beginners. Answer the latest
message directly and concisely. Do not claim live or private access."""

GROUNDED_PROMPT = """You are a local multi-domain assistant for beginners.
Answer the latest message directly. Use supplied facts and references only when
they are relevant. Distinguish direct, indirect, conditional, incompatible, and
uncertain relationships. Consumer services are not robot sensors, controllers,
SLAM systems, or replacements for onboard safety and perception."""

AUDIT_PROMPT = """Audit the draft against the user question and supplied
requirements. Identify incorrect relationships, unsupported claims, missing
required information, unsafe replacement claims, and meta-leakage. Return JSON:
{"pass":true,"issues":[],"revision_instructions":""}"""

REVISION_PROMPT = """Revise the draft using the audit. Produce only the final
answer. Preserve correct content, fix every listed issue, and do not mention
the audit, prompt, retrieval, scoring, or hidden processing."""


def _fast_research_mode():
    if _benchmark_force_llm():
        return False
    return (
        os.environ.get("FULL_RESEARCH_MODE") != "1"
        and os.environ.get("FAST_RESEARCH_MODE", "1") != "0"
    )


def _benchmark_force_llm():
    return (
        os.environ.get("BENCHMARK_FORCE_LLM") == "1"
        or os.environ.get("BENCHMARK_DISABLE_DETERMINISTIC_SHORTCUTS") == "1"
    )


def _load_json(path, default):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _extract_facts(text):
    patterns = [
        ("name", r"\bmy name is\s+(.+?)[.!?]*$"),
        ("city", r"\b(?:remember that )?i live in\s+(.+?)[.!?]*$"),
        ("favorite_app", r"\bmy favorite app is\s+(.+?)[.!?]*$"),
        ("favorite_robot", r"\bmy favorite robot is\s+(.+?)[.!?]*$"),
        ("field_of_study", r"\b(?:remember that )?i study\s+(.+?)[.!?]*$"),
        ("preferred_language", r"\bmy preferred language is\s+(.+?)[.!?]*$"),
        ("operating_system", r"\bmy operating system is\s+(.+?)[.!?]*$"),
    ]
    facts = {}
    for key, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            facts[key] = match.group(1).strip(" .!?")
    return facts


def _requested_fact(text):
    q = text.lower()
    checks = [
        ("name", ["my name"]),
        ("city", ["where do i live", "my city"]),
        ("favorite_app", ["favorite app"]),
        ("favorite_robot", ["favorite robot", "which robot"]),
        ("field_of_study", ["what do i study", "field of study"]),
        ("preferred_language", ["preferred language"]),
        ("operating_system", ["operating system"]),
    ]
    for key, phrases in checks:
        if any(phrase in q for phrase in phrases):
            return key
    return None


class PersistentSemanticMemory:
    def __init__(self, state_dir):
        self.text_path = os.path.join(state_dir, "ablation_semantic_texts.json")
        self.index_path = os.path.join(state_dir, "ablation_semantic.index")
        self.model = SentenceTransformer(MINILM_PATH, local_files_only=True)
        self.texts = _load_json(self.text_path, [])
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(384)
        if self.index.ntotal != len(self.texts):
            self.rebuild()

    def rebuild(self):
        self.index = faiss.IndexFlatIP(384)
        if self.texts:
            vectors = self.model.encode(
                self.texts,
                normalize_embeddings=True,
            ).astype("float32")
            self.index.add(vectors)
        self.save()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        _save_json(self.text_path, self.texts)

    def add(self, text):
        if text in self.texts:
            return
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
        ).astype("float32")
        self.index.add(vector)
        self.texts.append(text)
        self.save()

    def search(self, query, k=3):
        if not self.texts:
            return []
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(vector, min(k, len(self.texts)))
        return [
            {"text": self.texts[idx], "score": float(score)}
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0 and score >= 0.35
        ]

    def clear(self):
        self.texts = []
        self.index = faiss.IndexFlatIP(384)
        self.save()

    def reload(self):
        self.texts = _load_json(self.text_path, [])
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(384)
        if self.index.ntotal != len(self.texts):
            self.rebuild()


class SystemCAblation:
    def __init__(
        self,
        level,
        state_dir,
        *,
        temperature=0.2,
        seed=0,
        timeout=180,
    ):
        if level not in range(7):
            raise ValueError("Ablation level must be C0 through C6.")
        self.level = level
        self.state_dir = state_dir
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.fast_mode = _fast_research_mode()
        os.makedirs(state_dir, exist_ok=True)
        self.facts_path = os.path.join(state_dir, "ablation_facts.json")
        self.learning_path = os.path.join(state_dir, "ablation_learning.json")
        self.classifier = get_intent_classifier() if level >= 1 else None
        self.semantic = PersistentSemanticMemory(state_dir) if level >= 3 else None

    def reset(self):
        _save_json(self.facts_path, {})
        _save_json(self.learning_path, {})
        if self.semantic:
            self.semantic.clear()

    def reload(self):
        if self.semantic:
            self.semantic.reload()

    def _call(self, system_prompt, user_prompt, *, format_json=False, seed_offset=0):
        return chat(
            system_prompt,
            user_prompt,
            temperature=0.0 if format_json else self.temperature,
            seed=self.seed + seed_offset,
            max_tokens=320 if format_json else 220,
            format_json=format_json,
            ensure_complete=False,
            timeout=self.timeout,
        )

    def _memory_response(self, query):
        if self.level < 2:
            return None
        facts = _load_json(self.facts_path, {})
        stated = _extract_facts(query)
        if stated:
            facts.update(stated)
            _save_json(self.facts_path, facts)
            key, value = next(iter(stated.items()))
            return f"Remembered: your {key.replace('_', ' ')} is {value}."
        requested = _requested_fact(query)
        if requested and facts.get(requested):
            return f"Your {requested.replace('_', ' ')} is {facts[requested]}."
        return None

    def _learning_response(self, query):
        if self.level < 3:
            return None
        database = _load_json(self.learning_path, {})
        match = re.match(
            r"^\s*learn that\s+(.+?)\s+(?:means|is|refers to)\s+(.+?)[.!?]*$",
            query,
            flags=re.IGNORECASE,
        )
        if match:
            concept = match.group(1).strip().lower()
            definition = match.group(2).strip(" .!?")
            database[concept] = definition
            _save_json(self.learning_path, database)
            self.semantic.add(f"{concept}: {definition}")
            return f"Learned: {concept} means {definition}."
        q = query.lower()
        for concept, definition in database.items():
            if concept in q:
                return f"{concept} means {definition}."
        return None

    def _needs_extra_audit(self, query):
        q = query.lower()
        consumer = any(
            term in q
            for term in [
                "uber", "zomato", "swiggy", "google maps", "instagram",
                "whatsapp", "spotify", "restaurant", "rating", "social media",
                "food delivery", "ride-hailing", "app permission",
            ]
        )
        robotics = any(
            term in q
            for term in [
                "robot", "slam", "lidar", "localization", "pid", "perception",
                "controller", "sensor", "odometry", "obstacle", "ros",
                "navigation", "path planning",
            ]
        )
        named_unknown = bool(re.search(
            r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9-]*\b",
            query,
        ))
        ambiguous = q.strip() in {
            "navigation", "tracking", "control", "mapping", "permissions",
        }
        return (consumer and robotics) or named_unknown or ambiguous

    def __call__(self, query, return_metadata=False):
        start = time.perf_counter()
        stages = [f"C{self.level}"]
        cache_used = False
        deterministic_path_used = False
        llm_called = False
        memory_answer = self._memory_response(query)
        if memory_answer:
            response = memory_answer
            stages.append("structured_memory")
            deterministic_path_used = True
        else:
            learned_answer = self._learning_response(query)
            if learned_answer:
                response = learned_answer
                stages.append("semantic_learning")
                deterministic_path_used = True
            else:
                sections = [f"Latest user message:\n{query}"]
                classified_intent = None
                confidence = None
                if self.level >= 1:
                    classified_intent, confidence = self.classifier.predict(query)
                    sections.append(
                        f"Intent hint: {classified_intent} ({confidence:.3f})"
                    )
                    stages.append("intent_classifier")
                if self.level >= 2:
                    facts = _load_json(self.facts_path, {})
                    if facts:
                        sections.append(
                            "Relevant personal facts:\n"
                            + json.dumps(facts, ensure_ascii=False)
                        )
                    stages.append("structured_memory")
                if self.level >= 3:
                    semantic = self.semantic.search(query)
                    if semantic:
                        sections.append(
                            "Semantic memory:\n"
                            + "\n".join(item["text"] for item in semantic)
                        )
                    stages.append("minilm_faiss")
                if self.level >= 4:
                    documents = retrieve_local_knowledge(query, k=2)
                    if documents:
                        sections.append(
                            "Domain reference:\n"
                            + "\n\n".join(item["text"] for item in documents)
                        )
                    stages.append("domain_scoped_rag")
                if self.level >= 5:
                    from chatbot_system_c import (
                        _relationship_guidance,
                        _relationship_hint,
                        _resolve_domain,
                    )

                    domain = _resolve_domain(query)
                    hint = _relationship_hint(query, domain)
                    if hint != "none":
                        sections.append(
                            "Relationship requirement:\n"
                            + _relationship_guidance(query, hint)
                        )
                    stages.append("relationship_analyzer")

                prompt = "\n\n".join(sections)
                try:
                    llm_called = True
                    draft = self._call(
                        GROUNDED_PROMPT if self.level >= 4 else BASE_PROMPT,
                        prompt,
                    )
                    cache_used = cache_used or was_last_cache_used()
                    response = draft
                    stages.append("mistral_draft")
                    if self.level >= 6 and (
                        not self.fast_mode or self._needs_extra_audit(query)
                    ):
                        audit_input = (
                            f"Question:\n{query}\n\nDraft:\n{draft}\n\n"
                            f"Available context:\n{prompt}"
                        )
                        raw_audit = self._call(
                            AUDIT_PROMPT,
                            audit_input,
                            format_json=True,
                            seed_offset=1,
                        )
                        cache_used = cache_used or was_last_cache_used()
                        audit = json.loads(raw_audit)
                        stages.append("answer_audit")
                        if not audit.get("pass", False):
                            revision_input = (
                                f"Question:\n{query}\n\nDraft:\n{draft}\n\n"
                                f"Audit issues:\n{json.dumps(audit, ensure_ascii=False)}"
                                f"\n\nAvailable context:\n{prompt}"
                            )
                            response = self._call(
                                REVISION_PROMPT,
                                revision_input,
                                seed_offset=2,
                            )
                            cache_used = cache_used or was_last_cache_used()
                            stages.append("final_revision")
                        response = re.sub(
                            r"(?i).*(?:system prompt|retrieved context|audit says).*(?:\n|$)",
                            "",
                            response,
                        ).strip() or draft
                        stages.append("meta_leakage_cleanup")
                except (LLMRuntimeError, json.JSONDecodeError) as exc:
                    response = f"Model runtime error: {exc}"
                    stages.append("runtime_error")

        result = {
            "response": response,
            "latency": time.perf_counter() - start,
            "pipeline": stages,
            "ablation_level": f"C{self.level}",
            "cache_used": cache_used,
            "deterministic_path_used": deterministic_path_used,
            "llm_called": llm_called,
        }
        return result if return_metadata else response
