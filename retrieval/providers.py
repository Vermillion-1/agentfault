"""Hosted retrieval providers, with on-disk caching.

Caching is not an optimisation here, it is a quota control. The Cohere trial key
allows 1,000 calls per MONTH; a careless re-run of the sweep would spend a
meaningful slice of that on identical inputs. Every response is keyed by a hash of
(provider, model, input_type, texts) and replayed from results/.cache/ thereafter,
so repeated sweeps cost zero calls and stay byte-identical.

Batching matters for the same reason: Cohere embed accepts many texts per call, so
the whole 20-chunk corpus is two calls rather than twenty.

Keys come from tools.env and are never logged.
"""
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
from tools.env import get  # noqa: E402

CACHE = ROOT / "results" / ".cache"
CALLS = {"cohere_embed": 0, "cohere_rerank": 0, "groq_chat": 0}

# The exact request body and wall time of the most recent groq_chat, for provenance.
# The trace layer must record what was ACTUALLY SENT, not a reconstruction -- the
# whole point of harness-owned provenance is that it is not re-derived from anything
# the agent could influence. `cached` is recorded because a replayed response has a
# real hash but a meaningless duration, and conflating the two would quietly corrupt
# any latency analysis. Single-threaded by design; do not read this across threads.
LAST_CALL: dict = {}

# Trial-key ceilings: 5/min embed, 10/min rerank. Exceeding them returns 429 and
# wastes quota on a retry, so calls are paced rather than retried.
_RATE = {"cohere_embed": 60.0 / 5, "cohere_rerank": 60.0 / 10, "groq_chat": 60.0 / 28}
_LAST: dict = {}

# Groq's binding constraint is TOKENS per minute, not requests: gpt-oss-20b allows
# ~8k TPM on the free tier and a single debugging episode costs ~8.8k, so request
# pacing alone still 429s. This tracks a rolling token window and waits it out.
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

# Text -> vector memo, so an embedding is paid for once per process regardless of
# how many observation levels reuse it.
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
    # A replay check must never spend quota. On 2026-11-17 one did: the L3 cache
    # entries had been keyed on tracebacks holding raw heap addresses, so none of
    # them could ever match again, and the "zero-call" replay quietly made ~280
    # live Cohere calls against a 1,000-a-month trial key. With this set, a miss
    # is an error that names itself.
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


# ------------------------------------------------------------------ cohere

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
    """Embed every text not already memoised, in ONE batched call.

    Batching is the whole quota strategy: the 20-chunk corpus plus all four
    observation-level queries for a fault is two calls, not twenty-four.
    """
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


# -------------------------------------------------------------------- chat

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# How long to sit out a 429, and how many times. Groq's limit is a rolling
# one-minute token window, so a few 20-second waits always outlast it.
RATE_LIMIT_WAIT_S = 20.0
MAX_RATE_LIMIT_WAITS = 4
MAX_SERVER_RETRIES = 2
_sleep = _time.sleep            # indirection so tests never actually sleep


class ProviderError(RuntimeError):
    """A chat call that failed, tagged with WHY it failed.

    The sweep must react differently to each kind, and a bare RuntimeError cannot
    say which one happened:

      rate_limit  429. Wait and retry -- already done here; raised only once the
                  waits are used up.
      auth        401 / 403. Retrying cannot help. On 2026-09-29 Groq returned 403
                  for a full day; a sweep that kept going would have written every
                  remaining cell as an error row. The runner stops on this.
      server      5xx. Retried a little here, then raised.
      transport   curl itself failed, or the body was not JSON.
      client      any other 4xx. A malformed request is our bug; never retried.
    """

    def __init__(self, kind: str, status: int, detail: str):
        super().__init__(f"{kind} (HTTP {status}): {detail[:200]}")
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


def chat_endpoint() -> tuple:
    """(url, api key or None, provider label) for the chat model.

    Defaults to Groq. Any OpenAI-compatible server works instead -- a local vLLM,
    for example -- by setting two environment variables:

        AGENTFAULT_CHAT_URL      e.g. http://localhost:8000/v1/chat/completions
        AGENTFAULT_CHAT_KEY_ENV  NAME of the variable holding its key (optional)

    The Groq key is only ever sent to Groq. A custom URL gets a key only when
    AGENTFAULT_CHAT_KEY_ENV names one, so pointing the sweep at another server
    cannot leak the Groq key to it.
    """
    url = os.environ.get("AGENTFAULT_CHAT_URL") or GROQ_URL
    if url == GROQ_URL:
        return url, get("GROQ_API_KEY"), "groq"
    key_env = os.environ.get("AGENTFAULT_CHAT_KEY_ENV")
    return url, (get(key_env) if key_env else None), urlparse(url).netloc or "custom"


def chat_cache_payload(provider: str, model: str, temperature: float, max_tokens: int,
                       messages: List[dict], tools: "List[dict] | None",
                       seed: "int | None"):
    """What a cached chat response is keyed by.

    Before 2026-11-17 the key had no seed in it. Seeds 1..9 all run at temperature
    0.7 with the same opening prompt, so every one of them would have replayed the
    FIRST seed's cached response and the sweep would have reported zero variance
    across seeds -- ten copies of one episode. The seed is in the key now.

    Tools are keyed by a hash of the schemas rather than `bool(tools)`: adding or
    rewording a tool changes what the model sees, so it must not hit an old entry.

    A call with no seed on Groq keeps the old key shape, so the responses cached
    before this change still replay.
    """
    if seed is None and provider == "groq":
        return ("groq", model, temperature, max_tokens, messages, bool(tools))
    tools_sha = hashlib.sha256(
        json.dumps(tools or [], sort_keys=True).encode()).hexdigest()[:16]
    return ("chat", provider, model, temperature, max_tokens, messages, tools_sha, seed)


def _curl_chat(url: str, key: "str | None", body: dict) -> tuple:
    """One HTTP attempt. Returns (status, body text); status 0 means curl failed.

    Groq sits behind Cloudflare, which rejects urllib's TLS fingerprint with
    error 1010. curl carries a conventional fingerprint and is accepted, so the
    transport here is deliberately curl rather than the stdlib."""
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
    """One chat completion against the configured endpoint, cached on disk.

    `cache=False` neither reads nor writes the cache. It exists for one purpose:
    measuring the noise floor. A cached seed-0 response replays forever, which
    makes temperature 0 LOOK perfectly repeatable whether or not the provider
    really is. Only a second call that actually goes out can tell."""
    url, key, provider = chat_endpoint()
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if seed is not None:
        body["seed"] = seed

    def go():
        LAST_CALL["cached"] = False
        waits = retries = 0
        while True:
            if provider == "groq":
                _tpm_wait()          # the token window is Groq's limit, not a local server's
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


groq_chat = chat    # the old name, kept so existing callers keep working
