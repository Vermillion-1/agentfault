"""Spike 2026-11-18: would the edit gate (agent/policy.py) have refused any real edit?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from agent.policy import violations

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
