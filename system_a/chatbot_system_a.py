import os
import sys
import time

from llm_runtime import LLMRuntimeError, chat, was_last_cache_used
from research_core import (
    append_interaction,
    complete_sentence_response,
    load_interactions,
)


SYSTEM_PROMPT = """Continue the conversation and answer the latest user message.
Use the recent transcript as ordinary conversational context. Give a helpful
answer in under 120 words."""


def chatbot_system_a(user_query, return_metadata=False):
    start = time.time()
    cache_used = False
    llm_called = False
    interactions = load_interactions("system_a", limit=10)
    transcript = "\n".join(
        f"User: {item.get('user', '')}\nAssistant: {item.get('assistant', '')}"
        for item in interactions
    )
    prompt = f"""Recent conversation transcript:
{transcript or "(none)"}

    Latest user message:
{user_query}"""
    try:
        llm_called = True
        response = chat(
            SYSTEM_PROMPT,
            prompt,
            temperature=0.85,
            seed=int(os.environ.get("EXPERIMENT_SEED", "0")) + 11,
            max_tokens=140,
            ensure_complete=False,
        )
        cache_used = was_last_cache_used()
        response = complete_sentence_response(response, max_words=110)
    except LLMRuntimeError as exc:
        response = f"Model runtime error: {exc}"

    append_interaction("system_a", user_query, response)
    result = {
        "response": response,
        "latency": time.time() - start,
        "memory_turns": len(interactions),
        "cache_used": cache_used,
        "deterministic_path_used": False,
        "llm_called": llm_called,
        "pipeline": ["user", "unfiltered_conversation_memory", "mistral", "answer"],
    }
    return result if return_metadata else response


def _cli_value(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _print_cli_response(result, metadata_mode):
    response = result.get("response", "") if isinstance(result, dict) else result
    print("\nSystem A:")
    print(response)

    if metadata_mode and isinstance(result, dict):
        predicted_intent = result.get("predicted_intent")
        if predicted_intent is None:
            predicted_intent = result.get("classified_intent") or result.get("intent")
        fields = [
            ("latency", result.get("latency")),
            ("resolved_domain", result.get("resolved_domain")),
            ("predicted_intent", predicted_intent),
            ("retrieved_sources", result.get("retrieved_sources")),
            ("pipeline", result.get("pipeline")),
        ]
        for label, value in fields:
            if value not in (None, "", []):
                print(f"{label}: {_cli_value(value)}")
    print()


def run_cli():
    metadata_mode = "--metadata" in sys.argv[1:]

    print("System A Chatbot started.")
    if metadata_mode:
        print("Metadata mode enabled.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            result = chatbot_system_a(
                user_input,
                return_metadata=True,
            ) if metadata_mode else chatbot_system_a(user_input)
            _print_cli_response(result, metadata_mode)
        except Exception as exc:
            print("\nSystem A:")
            print(f"Runtime error: {exc}\n")


if __name__ == "__main__":
    run_cli()
