"""Hosted retrieval providers, with on-disk caching."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time as _time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Sequence
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.env import get

CACHE = ROOT / "results" / ".cache"
CALLS = {"cohere_embed": 0, "cohere_rerank": 0, "groq_chat": 0}

LAST_CALL: dict = {}

_RATE = {"cohere_embed": 60.0 / 5, "cohere_rerank": 60.0 / 10, "groq_chat": 60.0 / 28}
_LAST: dict = {}

GROQ_TPM = 8000
_TOK_WINDOW: list = []


def _tpm_wait(est_tokens: int = 3000) -> None:
    import time
    while True:
        now = time.monotonic()
        _TOK_WINDOW[:] = [(t, n) for t, n in _TOK_WINDOW if now - t < 60.0]
        used = sum(n for _, n in _TOK_WINDOW)
        if used + est_tokens <= GROQ_TPM * 0.9 or not _TOK_WINDOW:
            return
        time.sleep(min(8.0, 61.0 - (now - _TOK_WINDOW[0][0])))


def note_tokens(n: int) -> None:
    import time
    _TOK_WINDOW.append((time.monotonic(), n))

_TEXT_VEC: dict = {}


def _pace(name: str) -> None:
    import time
    gap = _RATE.get(name, 0.0)
    last = _LAST.get(name)
    if last is not None:
        wait = gap - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _LAST[name] = time.monotonic()


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]


class CacheMiss(RuntimeError):
    """Raised instead of a live call when AGENTFAULT_CACHE_ONLY is set."""


def _cached(name: str, payload, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}-{_key(payload)}.json"
    if p.exists():
        return json.loads(p.read_text())
    if os.environ.get("AGENTFAULT_CACHE_ONLY"):
        raise CacheMiss(f"{name}: no cached response for this input and "
                        f"AGENTFAULT_CACHE_ONLY is set ({p.name})")
    _pace(name)
    out = fn()
    p.write_text(json.dumps(out))
    CALLS[name] = CALLS.get(name, 0) + 1
    return out


def _post_json(url: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")


def cohere_embed(texts: Sequence[str], input_type: str,
                 model: str = "embed-english-v3.0") -> List[List[float]]:
    def go():
        out = _post_json("https://api.cohere.com/v2/embed",
                         {"model": model, "texts": list(texts), "input_type": input_type,
                          "embedding_types": ["float"]}, get("COHERE_API_KEY"))
        return out["embeddings"]["float"]
    return _cached("cohere_embed", ("embed", model, input_type, list(texts)), go)


def cohere_rerank(query: str, documents: Sequence[str],
                  model: str = "rerank-v3.5") -> List[int]:
    """Return document indices ordered by Cohere's relevance score."""
    def go():
        out = _post_json("https://api.cohere.com/v2/rerank",
                         {"model": model, "query": query, "documents": list(documents),
                          "top_n": len(documents)}, get("COHERE_API_KEY"))
        return [r["index"] for r in out["results"]]
    return _cached("cohere_rerank", ("rerank", model, query, list(documents)), go)


def embed_into_memo(texts: Sequence[str], input_type: str,
                    model: str = "embed-english-v3.0") -> None:
    """Embed every text not already memoised, in ONE batched call."""
    todo = [t for t in dict.fromkeys(texts) if (input_type, t) not in _TEXT_VEC]
    if not todo:
        return
    vecs = cohere_embed(todo, input_type, model)
    for t, v in zip(todo, vecs):
        _TEXT_VEC[(input_type, t)] = v


def memo_vec(text: str, input_type: str) -> List[float]:
    if (input_type, text) not in _TEXT_VEC:
        embed_into_memo([text], input_type)
    return _TEXT_VEC[(input_type, text)]


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

RATE_LIMIT_WAIT_S = 20.0
MAX_RATE_LIMIT_WAITS = 4
MAX_SERVER_RETRIES = 2
_sleep = _time.sleep


class ProviderError(RuntimeError):
    """A chat call that failed, tagged with WHY it failed."""

    def __init__(self, kind: str, status: int, detail: str):
        super().__init__(f"{kind} (HTTP {status}): {detail[:400]}")
        self.kind, self.status = kind, status


def classify_status(status: int) -> str:
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        return "auth"
    if status >= 500:
        return "server"
    if status == 0:
        return "transport"
    return "client"


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def configure(spec: str) -> None:
    """Point chat() at one provider for this process, by name instead of three"""
    for k in ("AGENTFAULT_CHAT_URL", "AGENTFAULT_CHAT_KEY_ENV", "AGENTFAULT_CHAT_UPSTREAM"):
        os.environ.pop(k, None)
    if spec == "groq":
        return
    if spec.startswith("openrouter:"):
        os.environ.update(AGENTFAULT_CHAT_URL=OPENROUTER_URL,
                          AGENTFAULT_CHAT_KEY_ENV="OPENROUTER_API_KEY",
                          AGENTFAULT_CHAT_UPSTREAM=spec.split(":", 1)[1])
    elif spec.startswith(("http://", "https://")):
        os.environ["AGENTFAULT_CHAT_URL"] = spec
    else:
        raise ValueError(f"unknown provider {spec!r}: use groq, openrouter:<host>, or a URL")


def chat_endpoint() -> tuple:
    """(url, api key or None, provider label) for the chat model."""
    url = os.environ.get("AGENTFAULT_CHAT_URL") or GROQ_URL
    if url == GROQ_URL:
        return url, get("GROQ_API_KEY"), "groq"
    key_env = os.environ.get("AGENTFAULT_CHAT_KEY_ENV")
    label = urlparse(url).netloc or "custom"
    if os.environ.get("AGENTFAULT_CHAT_UPSTREAM"):
        label += "/" + os.environ["AGENTFAULT_CHAT_UPSTREAM"]
    return url, (get(key_env) if key_env else None), label


def chat_cache_payload(provider: str, model: str, temperature: float, max_tokens: int,
                       messages: List[dict], tools: "List[dict] | None",
                       seed: "int | None"):
    """What a cached chat response is keyed by."""
    if seed is None and provider == "groq":
        return ("groq", model, temperature, max_tokens, messages, bool(tools))
    tools_sha = hashlib.sha256(
        json.dumps(tools or [], sort_keys=True).encode()).hexdigest()[:16]
    return ("chat", provider, model, temperature, max_tokens, messages, tools_sha, seed)


def _curl_chat(url: str, key: "str | None", body: dict) -> tuple:
    """One HTTP attempt. Returns (status, body text); status 0 means curl failed."""
    cmd = ["curl", "-s", "--max-time", "120", "-w", "\n%{http_code}",
           "-H", "Content-Type: application/json",
           "-H", "User-Agent: curl/8.4.0"]
    if key:
        cmd += ["-H", f"Authorization: Bearer {key}"]
    cmd += ["-d", json.dumps(body), url]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return 0, p.stderr
    text, _, code = p.stdout.rpartition("\n")
    return (int(code) if code.strip().isdigit() else 0), text


def chat(messages: List[dict], model: str = "openai/gpt-oss-20b",
         temperature: float = 0.0, max_tokens: int = 1024,
         tools: "List[dict] | None" = None, seed: "int | None" = None,
         cache: bool = True) -> dict:
    """One chat completion against the configured endpoint, cached on disk."""
    url, key, provider = chat_endpoint()
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if seed is not None:
        body["seed"] = seed
    upstream = os.environ.get("AGENTFAULT_CHAT_UPSTREAM")
    if upstream and provider != "groq":
        body["provider"] = {"order": [upstream], "allow_fallbacks": False,
                            "require_parameters": True}

    def go():
        LAST_CALL["cached"] = False
        waits = retries = 0
        while True:
            if provider == "groq":
                _tpm_wait()
            status, text = _curl_chat(url, key, body)
            if status == 200:
                try:
                    out = json.loads(text)
                except json.JSONDecodeError:
                    raise ProviderError("transport", status, text)
                if "error" in out:
                    raise ProviderError("server", status, str(out["error"]))
                note_tokens(out.get("usage", {}).get("total_tokens", 0))
                return out
            kind = classify_status(status)
            if kind == "rate_limit" and waits < MAX_RATE_LIMIT_WAITS:
                waits += 1
                _sleep(RATE_LIMIT_WAIT_S)
                continue
            if kind in ("server", "transport") and retries < MAX_SERVER_RETRIES:
                retries += 1
                _sleep(5.0 * retries)
                continue
            raise ProviderError(kind, status, text)

    LAST_CALL.clear()
    LAST_CALL.update({"request": body, "cached": True, "provider": provider})
    _t0 = _time.perf_counter()
    name = "groq_chat" if provider == "groq" else "chat"
    if cache:
        out = _cached(name, chat_cache_payload(provider, model, temperature, max_tokens,
                                               messages, tools, seed), go)
    else:
        _pace(name)
        out = go()
        CALLS[name] = CALLS.get(name, 0) + 1
    LAST_CALL["duration_ms"] = (_time.perf_counter() - _t0) * 1000
    return out


groq_chat = chat
