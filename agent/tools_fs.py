"""The agent's file tools: list, read, search, edit. Standard library only.

They act on one directory, the agent's workspace, and it holds source files only:
the tests are never copied in (agent/sandbox.py). So no tool can show a test,
whatever path the agent guesses, because there is no test file to find.

The same functions run in-process for offline tests and inside the Docker
workspace container for live runs, invoked as a script:

    python tools_fs.py <tool> <root>    # JSON args on stdin, JSON result on stdout

Each call returns {"text": what the agent sees, "seen": [[rel, first, last], ...],
"edited": [rel, line] or null, "policy": [violation, ...] or null}; the episode
turns those into its own bookkeeping, so these functions keep no state.

`edit` refuses any edit that agent/policy.py forbids. The agent is told only that
edits may change program logic; what was attempted goes to `policy`, for the record.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    from agent.policy import violations
except ImportError:            # run as a script in the container, next to policy.py
    from policy import violations

MAX_FILE_CHARS = 6000


def _sources(root: Path):
    return [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts]


def _inside(root: Path, path: str):
    """The resolved path when it is a file inside root, else None."""
    p = (root / path).resolve()
    return p if str(p).startswith(str(root.resolve())) and p.is_file() else None


def t_list(root: Path, a: dict) -> dict:
    return {"text": "\n".join(str(p.relative_to(root)) for p in _sources(root))}


def t_read(root: Path, a: dict) -> dict:
    p = _inside(root, a.get("path", ""))
    if not p:
        return {"text": f"no such file: {a.get('path')}"}
    lines = p.read_text()[:MAX_FILE_CHARS].splitlines()
    rel = str(p.relative_to(root.resolve()))
    return {"text": "\n".join(f"{i:4d}| {ln}" for i, ln in enumerate(lines, 1)),
            "seen": [[rel, 1, len(lines)]]}


def t_search(root: Path, a: dict) -> dict:
    try:
        rx = re.compile(a.get("pattern", ""))
    except re.error as e:
        return {"text": f"bad regex: {e}"}
    hits = []
    for p in _sources(root):
        for i, ln in enumerate(p.read_text().splitlines(), 1):
            if rx.search(ln):
                hits.append((str(p.relative_to(root)), i, ln.strip()[:110]))
    hits = hits[:40]
    return {"text": "\n".join(f"{rel}:{i}: {ln}" for rel, i, ln in hits) or "no matches",
            "seen": [[rel, i, i] for rel, i, _ in hits]}


def t_edit(root: Path, a: dict) -> dict:
    p = _inside(root, a.get("path", ""))
    # Python source only: a config file such as pyproject.toml configures pytest.
    if not p or p.suffix != ".py":
        return {"text": f"no such file: {a.get('path', '')}"}
    # Normalised, so "./intervals/core.py" and "intervals/core.py" count as the
    # same file when the edit is later compared with the fault's location.
    rel = str(p.relative_to(root.resolve()))
    old, new = a.get("old", ""), a.get("new", "")
    text = p.read_text()
    n = text.count(old)
    if not old or n == 0:
        return {"text": "edit rejected: `old` not found verbatim. Read the file and copy the exact text."}
    if n > 1:
        return {"text": f"edit rejected: `old` appears {n} times. Include more context to disambiguate."}
    line_no = text[:text.index(old)].count("\n") + 1
    patched = text.replace(old, new)
    bad = violations(text, patched)
    if bad:
        return {"text": "edit rejected: edits may only change program logic.", "policy": bad}
    p.write_text(patched)
    return {"text": f"edited {rel} at line {line_no}", "edited": [rel, line_no]}


TOOLS = {"list": t_list, "read": t_read, "search": t_search, "edit": t_edit}

if __name__ == "__main__":
    print(json.dumps(TOOLS[sys.argv[1]](Path(sys.argv[2]), json.load(sys.stdin))))
