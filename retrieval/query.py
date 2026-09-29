"""Build the retrieval query from what the agent can actually see when tests fail.

The key structural point: the query IS the observation level. The decision doc's
v0.1 arm (how much does the agent see about the failure) and its v0.3 arm (how is
context retrieved) are the same experiment viewed from two sides, because richer
observation changes only the query string handed to the retriever.

So one harness measures both, and the sweep is a grid rather than two studies:

    L0  "tests failed"                      -- pass/fail only
    L1  + which tests failed                -- node ids
    L2  + assertion text                    -- the `E ...` lines pytest prints
    L3  + full traceback                    -- frames, expected vs actual

Nothing here calls a model. The failure text is ground truth produced by running
the suite, which is why the intrinsic half of this study costs nothing to run.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

LEVELS = ["L0", "L1", "L2", "L3"]


def run_suite(repo: Path) -> str:
    """Full pytest output with long tracebacks. One run serves every level."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-rf",
         "--tb=long", "-p", "no:cacheprovider"],
        cwd=repo, capture_output=True, text=True)
    return proc.stdout


def failing_nodes(output: str) -> List[str]:
    return sorted(set(re.findall(r"^(?:FAILED|ERROR)\s+(\S+)", output, re.M)))


def assertion_lines(output: str) -> List[str]:
    """The `E   ...` lines -- pytest's one-line summary of what went wrong."""
    return [ln.strip() for ln in output.splitlines() if ln.startswith("E ")]


def failures_section(output: str) -> str:
    """Everything between the FAILURES banner and the short summary."""
    m = re.search(r"=+ FAILURES =+\n(.*?)(?:\n=+ (?:short test summary|warnings)|\Z)",
                  output, re.S)
    return m.group(1).strip() if m else ""


def build(output: str, level: str) -> str:
    """Compose the query text for one observation level."""
    if level == "L0":
        return "tests failed"

    nodes = failing_nodes(output)
    # Node ids carry the test's own words (test_adjacent_coalesce), which is real
    # signal an agent would have -- not leakage, since the test name describes the
    # symptom, not the faulty function.
    q = ["tests failed"] + nodes
    if level == "L1":
        return "\n".join(q)

    q += assertion_lines(output)
    if level == "L2":
        return "\n".join(q)

    if level == "L3":
        return "\n".join(q + [failures_section(output)])

    raise ValueError(f"unknown observation level {level!r}")


def build_all(repo: Path) -> Dict[str, str]:
    out = run_suite(repo)
    return {lvl: build(out, lvl) for lvl in LEVELS}
