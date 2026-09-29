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
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.env import get  # noqa: E402

CACHE = ROOT / "results" / ".cache"
CALLS = {"cohere_embed": 0, "cohere_rerank": 0, "groq_chat": 0}

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


def _cached(name: str, payload, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}-{_key(payload)}.json"
    if p.exists():
        return json.loads(p.read_text())
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


# -------------------------------------------------------------------- groq

def groq_chat(messages: List[dict], model: str = "openai/gpt-oss-20b",
              temperature: float = 0.0, max_tokens: int = 1024,
              tools: "List[dict] | None" = None) -> dict:
    """Groq sits behind Cloudflare, which rejects urllib's TLS fingerprint with
    error 1010. curl carries a conventional fingerprint and is accepted, so the
    transport here is deliberately curl rather than the stdlib."""
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    def go():
        _tpm_wait()
        p = subprocess.run(
            ["curl", "-s", "--max-time", "120",
             "-H", f"Authorization: Bearer {get('GROQ_API_KEY')}",
             "-H", "Content-Type: application/json",
             "-H", "User-Agent: curl/8.4.0",
             "-d", json.dumps(body),
             "https://api.groq.com/openai/v1/chat/completions"],
            capture_output=True, text=True)
        if p.returncode != 0 or not p.stdout.strip():
            raise RuntimeError(f"groq curl failed rc={p.returncode} {p.stderr[:200]}")
        out = json.loads(p.stdout)
        if "error" in out:
            raise RuntimeError(f"groq: {str(out['error'])[:300]}")
        note_tokens(out.get("usage", {}).get("total_tokens", 0))
        return out
    return _cached("groq_chat", ("groq", model, temperature, max_tokens, messages,
                                 bool(tools)), go)
