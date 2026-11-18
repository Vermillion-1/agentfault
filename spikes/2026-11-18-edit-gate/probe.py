"""Spike 2026-11-18: would the edit gate (agent/policy.py) have refused any real edit?

Replays every distinct `edit` call gpt-oss-20b returned in the research repo's
response cache (every live Phase D call, the set-aside Groq runs included; the
trace previews are truncated, the cache holds full responses) and asks the gate
whether the replacement adds anything forbidden relative to the text it replaced.
Run from the agentfault root:

    python3 spikes/2026-11-18-edit-gate/probe.py ../agent-fault-research/results/.cache
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from agent.policy import violations          # noqa: E402

edits, refused = set(), []
for p in sorted(Path(sys.argv[1]).glob("chat-*.json")):
    for ch in json.loads(p.read_text()).get("choices", []):
        for call in ch.get("message", {}).get("tool_calls") or []:
            if call["function"]["name"] != "edit":
                continue
            try:
                a = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            key = (a.get("path", ""), a.get("old", ""), a.get("new", ""))
            if key in edits:
                continue
            edits.add(key)
            bad = violations(key[1], key[2])
            if bad:
                refused.append((key, bad))
print(f"distinct edits replayed: {len(edits)}; refused by the gate: {len(refused)}")
for (path, old, new), bad in refused:
    print(f"  {path}: {bad}\n    old={old[:80]!r}\n    new={new[:80]!r}")
