import json
import hashlib
import os
import time
import urllib.error
import urllib.request


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:latest")
LLM_CACHE_PATH = os.environ.get(
    "LLM_CACHE_PATH",
    "/tmp/ai_robotics_assistant_mistral_cache.json",
)
LLM_CACHE_MAX_ENTRIES = int(os.environ.get("LLM_CACHE_MAX_ENTRIES", "1000"))
_LLM_CACHE = None
LAST_CACHE_USED = False
LAST_CALL_INFO = {
    "cache_used": False,
    "cache_enabled": True,
    "latency": None,
}


class LLMRuntimeError(RuntimeError):
    pass


def _cache_enabled():
    return os.environ.get("DISABLE_LLM_CACHE") != "1"


def _cache_path():
    return os.environ.get("LLM_CACHE_PATH", LLM_CACHE_PATH)


def _load_cache():
    global _LLM_CACHE
    if _LLM_CACHE is not None:
        return _LLM_CACHE
    try:
        with open(_cache_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        _LLM_CACHE = data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        _LLM_CACHE = {}
    return _LLM_CACHE


def _save_cache(cache):
    if not _cache_enabled():
        return
    if len(cache) > LLM_CACHE_MAX_ENTRIES:
        keys = list(cache)
        for key in keys[: len(cache) - LLM_CACHE_MAX_ENTRIES]:
            cache.pop(key, None)
    cache_path = _cache_path()
    directory = os.path.dirname(cache_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False)
    os.replace(tmp_path, cache_path)


def _cache_key(payload):
    stable = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _set_last_call_info(**updates):
    global LAST_CACHE_USED, LAST_CALL_INFO
    updates.pop("response", None)
    LAST_CALL_INFO = {
        "cache_used": False,
        "cache_enabled": _cache_enabled(),
        "latency": None,
        **updates,
    }
    LAST_CACHE_USED = bool(LAST_CALL_INFO["cache_used"])


def reset_llm_cache(delete_file=False):
    global _LLM_CACHE
    _LLM_CACHE = {}
    _set_last_call_info(cache_used=False, cache_enabled=_cache_enabled())
    if delete_file:
        try:
            os.remove(_cache_path())
        except FileNotFoundError:
            pass
        except OSError:
            pass


def prepare_uncached_benchmark_runtime():
    os.environ["DISABLE_LLM_CACHE"] = "1"
    reset_llm_cache(delete_file=True)


def chat(
    system_prompt,
    user_prompt,
    *,
    temperature=0.4,
    seed=42,
    max_tokens=300,
    format_json=False,
    ensure_complete=True,
    timeout=180,
    return_metadata=False,
):
    """Generate one response with the shared local Mistral model."""
    started = time.time()
    _set_last_call_info(cache_used=False, cache_enabled=_cache_enabled())
    cache_payload = {
        "model": OLLAMA_MODEL,
        "url": OLLAMA_URL,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "format_json": format_json,
        "ensure_complete": ensure_complete,
        "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "2048")),
    }
    cache_key = _cache_key(cache_payload)
    if _cache_enabled():
        cached = _load_cache().get(cache_key)
        if isinstance(cached, str) and cached.strip():
            metadata = {
                "response": cached,
                "cache_used": True,
                "cache_enabled": True,
                "latency": time.time() - started,
            }
            _set_last_call_info(**metadata)
            return metadata if return_metadata else cached

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    pieces = []

    # One continuation is enough to repair answers that hit num_predict while
    # avoiding an unbounded generation loop.
    max_passes = 1 if format_json or not ensure_complete else 2
    for pass_index in range(max_passes):
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
            "messages": messages,
            "options": {
                "temperature": temperature,
                "seed": seed + pass_index,
                "num_predict": max_tokens if pass_index == 0 else min(max_tokens, 100),
                "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "2048")),
            },
        }
        if format_json:
            payload["format"] = "json"

        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMRuntimeError(
                f"Could not call {OLLAMA_MODEL} through Ollama at {OLLAMA_URL}: {exc}"
            ) from exc

        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise LLMRuntimeError("Ollama returned an empty Mistral response.")
        pieces.append(content)

        if data.get("done_reason") != "length" or pass_index + 1 >= max_passes:
            break

        messages.extend([
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Continue exactly where you stopped. Finish the current "
                    "sentence and answer concisely. Do not restart or repeat."
                ),
            },
        ])

    final = " ".join(pieces).strip()
    if _cache_enabled() and final:
        cache = _load_cache()
        cache[cache_key] = final
        _save_cache(cache)
    metadata = {
        "response": final,
        "cache_used": False,
        "cache_enabled": _cache_enabled(),
        "latency": time.time() - started,
    }
    _set_last_call_info(**metadata)
    return metadata if return_metadata else final


def was_last_cache_used():
    return LAST_CACHE_USED


def last_call_metadata():
    return dict(LAST_CALL_INFO)
