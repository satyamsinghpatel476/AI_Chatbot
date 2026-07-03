import json
import os
import shutil

import faiss


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEMANTIC_MODEL_DIR = os.path.join(ROOT_DIR, "models", "all-MiniLM-L6-v2")


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def initialize_state_dir(state_dir):
    os.makedirs(state_dir, exist_ok=True)
    defaults = {
        "system_a_interactions.json": [],
        "system_b_interactions.json": [],
        "system_b_facts.json": {},
        "system_c_facts.json": {},
        "learning.json": {},
        "semantic_texts.json": [],
    }
    for filename, value in defaults.items():
        path = os.path.join(state_dir, filename)
        if not os.path.exists(path):
            write_json(path, value)


def reset_state_dir(state_dir):
    if os.path.isdir(state_dir):
        shutil.rmtree(state_dir)
    initialize_state_dir(state_dir)


def copy_state(source_dir, target_dir):
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _controlled_chat(temperature, seed, timeout):
    from llm_runtime import chat as runtime_chat

    def call(system_prompt, user_prompt, **kwargs):
        kwargs["temperature"] = temperature
        kwargs["seed"] = seed
        kwargs["timeout"] = timeout
        return runtime_chat(system_prompt, user_prompt, **kwargs)

    return call


def _rebuild_semantic_index(semantic):
    semantic.index = faiss.IndexFlatIP(semantic.dimension)
    if semantic.EMBEDDINGS_ENABLED and semantic.texts:
        vectors = semantic.MODEL.encode(
            semantic.texts,
            normalize_embeddings=True,
        ).astype("float32")
        semantic.index.add(vectors)
    if semantic.EMBEDDINGS_ENABLED:
        faiss.write_index(semantic.index, semantic.INDEX_FILE)
    write_json(semantic.TEXT_FILE, semantic.texts)


def _configure_semantic_memory(semantic, state_dir):
    semantic.INDEX_FILE = os.path.join(state_dir, "semantic.index")
    semantic.TEXT_FILE = os.path.join(state_dir, "semantic_texts.json")
    semantic.texts = read_json(semantic.TEXT_FILE, [])
    embeddings_requested = os.environ.get(
        "EXPERIMENT_SEMANTIC_EMBEDDINGS",
        "1",
    ) != "0"

    if embeddings_requested and os.path.isdir(SEMANTIC_MODEL_DIR):
        try:
            from sentence_transformers import SentenceTransformer

            semantic.MODEL = SentenceTransformer(
                SEMANTIC_MODEL_DIR,
                local_files_only=True,
            )
            semantic.EMBEDDINGS_ENABLED = True
        except Exception:
            semantic.MODEL = None
            semantic.EMBEDDINGS_ENABLED = False
    else:
        semantic.MODEL = None
        semantic.EMBEDDINGS_ENABLED = False

    if semantic.EMBEDDINGS_ENABLED and os.path.exists(semantic.INDEX_FILE):
        semantic.index = faiss.read_index(semantic.INDEX_FILE)
    else:
        semantic.index = faiss.IndexFlatIP(semantic.dimension)

    if semantic.index.ntotal != len(semantic.texts):
        _rebuild_semantic_index(semantic)


def configure_stable_system(system_name, state_dir, temperature, seed, timeout):
    """Redirect stable Version 3 modules to experiment-local state."""
    initialize_state_dir(state_dir)

    import research_core

    research_core.MEMORY_DIR = state_dir
    controlled_chat = _controlled_chat(temperature, seed, timeout)

    if system_name == "A":
        import system_a.chatbot_system_a as module

        module.chat = controlled_chat
        return module, module.chatbot_system_a

    if system_name == "B":
        import system_b.chatbot_system_b as module
        import system_b.grounding as grounding

        grounding.INTERACTION_FILE = os.path.join(
            state_dir,
            "system_b_interactions.json",
        )
        module.chat = controlled_chat
        return module, module.chatbot_system_b

    if system_name == "C":
        import chatbot_system_c as module
        import memory.semantic_memory as semantic

        module.LEARNING_FILE = os.path.join(state_dir, "learning.json")
        _configure_semantic_memory(semantic, state_dir)
        module.chat = controlled_chat
        return module, module.chatbot_system_c

    raise ValueError(f"Unknown stable system: {system_name}")


def reset_loaded_system(system_name, state_dir):
    reset_state_dir(state_dir)
    if system_name == "C":
        import memory.semantic_memory as semantic

        semantic.index = faiss.IndexFlatIP(semantic.dimension)
        semantic.texts = []
        if semantic.EMBEDDINGS_ENABLED:
            faiss.write_index(semantic.index, semantic.INDEX_FILE)
        write_json(semantic.TEXT_FILE, [])


def reload_loaded_system(system_name, state_dir):
    if system_name == "C":
        import memory.semantic_memory as semantic

        semantic.texts = read_json(
            os.path.join(state_dir, "semantic_texts.json"),
            [],
        )
        semantic.TEXT_FILE = os.path.join(state_dir, "semantic_texts.json")
        semantic.INDEX_FILE = os.path.join(state_dir, "semantic.index")
        if semantic.EMBEDDINGS_ENABLED and os.path.exists(semantic.INDEX_FILE):
            semantic.index = faiss.read_index(semantic.INDEX_FILE)
        else:
            semantic.index = faiss.IndexFlatIP(semantic.dimension)
        if semantic.index.ntotal != len(semantic.texts):
            _rebuild_semantic_index(semantic)


def seed_context(system_name, state_dir, context, module=None):
    if system_name == "A":
        rows = [
            {"user": item["user"], "assistant": item["assistant"]}
            for item in context
        ]
        write_json(os.path.join(state_dir, "system_a_interactions.json"), rows[-24:])
        return

    if system_name == "B":
        rows = []
        for item in context:
            intent, _ = module.CLASSIFIER.predict(item["user"])
            rows.append({
                "user": item["user"],
                "assistant": item["assistant"],
                "intent": intent,
            })
        write_json(os.path.join(state_dir, "system_b_interactions.json"), rows[-24:])


def inject_filler_history(system_name, state_dir, count):
    if system_name == "A":
        rows = [
            {
                "user": f"Filler daily conversation turn {index}.",
                "assistant": "Acknowledged.",
            }
            for index in range(count)
        ]
        write_json(os.path.join(state_dir, "system_a_interactions.json"), rows[-24:])
    elif system_name == "B":
        rows = [
            {
                "user": f"Filler daily conversation turn {index}.",
                "assistant": "Acknowledged.",
                "intent": "daily",
            }
            for index in range(count)
        ]
        write_json(os.path.join(state_dir, "system_b_interactions.json"), rows[-24:])
