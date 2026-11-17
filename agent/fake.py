"""Scripted stand-ins for the chat model, so episodes run with no key and no network.

Each behaviour is a fixed sequence of tool calls that drives the loop to ONE known
outcome. That is what makes the outcome labels testable: a live model does whatever
it does, a script does exactly what the test needs.

    fix          read, apply the true fix, test, done      -> fixed
    give_up      read, give_up                             -> gave_up
    wrong_done   read, done without editing                -> silent_failure
    tamper       rename every failing test away, test, done -> fixed_tampered
    stall        list, forever                             -> exhausted

Used by tests/ and by `runner/agent_sweep.py --fake`, which rehearses a whole sweep
(resume, circuit breaker, CSV, report) before any quota is spent.
"""
from __future__ import annotations

import json
from typing import Callable, List

from retrieval.providers import LAST_CALL, ProviderError

BEHAVIOURS = ("fix", "give_up", "wrong_done", "tamper", "stall")


def _call(tool: str, args: dict, call_id: str, reasoning: str = "") -> dict:
    """One assistant turn holding one tool call, in the provider's response shape."""
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"id": call_id, "type": "function", "function": {
               "name": tool, "arguments": json.dumps(args)}}]}
    if reasoning:
        msg["reasoning"] = reasoning
    return {"choices": [{"finish_reason": "tool_calls", "message": msg}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "completion_tokens_details": {"reasoning_tokens": 7}},
            "id": f"resp_{call_id}"}


def script_for(fault: dict, behaviour: str) -> List[tuple]:
    """The (tool, args) sequence for one behaviour on one fault."""
    read = ("read", {"path": fault["file"]})
    if behaviour == "fix":
        # The fault swapped `anchor` for `replacement`; the fix swaps it back.
        return [read,
                ("edit", {"path": fault["file"], "old": fault["replacement"],
                          "new": fault["anchor"]}),
                ("test", {}), ("done", {})]
    if behaviour == "give_up":
        return [read, ("give_up", {})]
    if behaviour == "wrong_done":
        return [read, ("done", {})]
    if behaviour == "tamper":
        # Rename each failing test so pytest no longer collects it. The source is
        # never touched, so the suite goes green with the fault still in place.
        steps = []
        for node in fault["expected_failing_tests"]:
            path, name = node.split("::")[0], node.split("::")[-1].split("[")[0]
            steps.append(("edit", {"path": path, "old": f"def {name}(",
                                   "new": f"def disabled_{name}("}))
        return steps + [("test", {}), ("done", {})]
    if behaviour == "stall":
        return [("list", {})]
    raise ValueError(f"unknown behaviour {behaviour!r}")


def scripted_model(fault: dict, behaviour: str, reasoning: str = "") -> Callable:
    """A drop-in for `retrieval.providers.chat` that plays one script.

    Past the end of the script it repeats the last step, which is what makes
    `stall` run until the loop's step limit."""
    steps = script_for(fault, behaviour)
    n = {"calls": 0}

    def fake_chat(messages, **kw):
        i = min(n["calls"], len(steps) - 1)
        n["calls"] += 1
        # The loop reads LAST_CALL for provenance, so the fake has to fill it the
        # way the real client does.
        LAST_CALL.clear()
        LAST_CALL.update({"request": {"model": kw.get("model"), "messages": messages,
                                      "seed": kw.get("seed")},
                          "duration_ms": 12.3, "cached": False, "provider": "fake"})
        tool, args = steps[i]
        return _call(tool, args, f"c{n['calls']}", reasoning)

    return fake_chat


def failing_model(kind: str = "server", status: int = 503) -> Callable:
    """A model whose every call fails, for exercising the sweep's circuit breaker."""
    def fake_chat(messages, **kw):
        raise ProviderError(kind, status, "scripted failure")
    return fake_chat
