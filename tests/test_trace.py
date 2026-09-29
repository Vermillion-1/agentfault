"""Offline test of the loop -> trace integration.

The model is scripted rather than called, for three reasons: it needs no API key or
network, it is deterministic, and it lets us assert on a KNOWN agent behaviour
(read, edit, test, done) instead of whatever a live model happens to do.

The assertions that matter are the publication-safety ones. Reasoning text must
appear ONLY in the sidecar, never in the published trace, because provider terms on
redistributing chain-of-thought differ. A test is the only thing that keeps that
true as the schema evolves.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.loop as loop                                    # noqa: E402
from tools.corpus import load_faults, materialise            # noqa: E402

SECRET_COT = "SENTINEL_REASONING_MUST_NOT_LEAK_INTO_PUBLISHED_TRACE"


def _msg(tool, args, call_id):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "", "reasoning": SECRET_COT,
        "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": tool, "arguments": json.dumps(args)}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                  "completion_tokens_details": {"reasoning_tokens": 7}},
        "id": f"resp_{call_id}"}


def run_episode(tmp: Path):
    fault = [f for f in load_faults() if f["id"] == "intervals-001"][0]
    repo = materialise(fault, tmp / "intervals")

    script = [
        _msg("read", {"path": "intervals/core.py"}, "c1"),
        _msg("search", {"pattern": "def merge"}, "c2"),
        _msg("edit", {"path": "intervals/core.py",
                      "old": "        if cur.start < last.end:",
                      "new": "        if cur.start <= last.end:"}, "c3"),
        _msg("test", {}, "c4"),
        _msg("done", {}, "c5"),
    ]
    calls = {"n": 0}

    def fake_groq(messages, **kw):
        i = min(calls["n"], len(script) - 1)
        calls["n"] += 1
        loop.LAST_CALL.clear()
        loop.LAST_CALL.update({"request": {"model": kw.get("model", "openai/gpt-oss-20b"),
                                           "messages": messages},
                               "duration_ms": 12.3, "cached": False})
        return script[i]

    real = loop.groq_chat
    loop.groq_chat = fake_groq
    try:
        out = tmp / "trace"
        res = loop.Episode(repo, fault, level="L3", seed=0, trace_dir=out).run()
    finally:
        loop.groq_chat = real
    return res, out


def test_trace_layers():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        res, out = run_episode(tmp)

        manifest = json.loads((out / "run.json").read_text())
        events = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
        claims = [json.loads(l) for l in (out / "claims.jsonl").read_text().splitlines()]
        reasoning = [json.loads(l) for l in (out / "reasoning.jsonl").read_text().splitlines()]
        annots = [json.loads(l) for l in (out / "annotations.jsonl").read_text().splitlines()]

        # Layer 0 -- ground truth known at injection time
        inj = manifest["injection"]
        assert inj["injected"] and inj["file"] == "intervals/core.py"
        assert inj["function"] == "merge"
        assert manifest["outcome"]["resolved"] is True, "scripted fix should go green"
        assert manifest["outcome"]["agent_claimed_success"] is True
        assert manifest["outcome"]["silent_failure"] is False
        assert manifest["outcome"]["localised_function"] is True
        assert manifest["harness"]["git_sha"]

        # Layer 1 -- harness-observed, with evidence hashes
        llm = [e for e in events if e["operation"] == "chat"]
        tools = [e for e in events if e["operation"] == "execute_tool"]
        assert llm and tools
        assert all(e["provenance_origin"] == "HARNESS_OBSERVED" for e in events)
        assert all(e["evidence_ref"]["http_request_sha256"] for e in llm)
        assert all(e["evidence_ref"]["http_response_sha256"] for e in llm)
        assert llm[0]["llm"]["reasoning_source_field"] == "reasoning"
        assert llm[0]["llm"]["reasoning_is_raw_cot"] is True, "gpt-oss returns raw CoT"

        # navigation -- how it moved through the repo
        nav = [e["navigation"] for e in tools]
        assert {n["intent"] for n in nav} >= {"read", "search", "edit", "test"}
        assert nav[0]["novel_paths_count"] == 1
        assert any(n["repeat_paths_count"] >= 1 for n in nav), "revisit must be counted"
        assert manifest["navigation_summary"]["lines_consumed"] > 0

        # Layer 2 -- agent self-report, marked untrusted
        assert claims and all(c["provenance_origin"] == "AGENT_REPORTED" for c in claims)
        assert any(c["claimed_status"] == "done" for c in claims)

        # Layer 3 -- free ground-truth label
        assert any(a["label_source"] == "INJECTED_GROUND_TRUTH" for a in annots)

        # sidecar isolation -- the publication-safety property
        assert reasoning and reasoning[0]["text"] == SECRET_COT
        for name in ("run.json", "events.jsonl", "claims.jsonl", "annotations.jsonl"):
            assert SECRET_COT not in (out / name).read_text(), \
                f"reasoning text leaked into {name}"
        assert llm[0]["llm"]["reasoning_sha256"], "hash kept so the sidecar can be re-joined"


if __name__ == "__main__":
    test_trace_layers()
    print("all trace assertions passed")
