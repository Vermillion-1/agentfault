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


def scrub_paths(text: str, repo: Path) -> str:
    """Replace machine-specific absolute paths with a stable placeholder.

    Three separate problems, one fix:

      reproducibility -- episodes run in a fresh temp directory, so the traceback
                         embeds a different absolute path every time. That makes the
                         prompt non-deterministic, which silently defeats response
                         caching and means no run can be replayed exactly.
      privacy         -- a published trace would otherwise carry the author's home
                         directory and username in every traceback.
      comparability   -- two runs of the same fault should differ only in what the
                         agent did, not in where the harness happened to put files.

    The agent loses nothing: paths inside the repo are still relative and still
    navigable, which is all it needs to find and edit a file.
    """
    out = text.replace(str(repo.resolve()), "<repo>").replace(str(repo), "<repo>")
    # macOS reports /private/var while tempfile hands out /var; normalise both, then
    # sweep any residual temp path so a stray form cannot reintroduce nondeterminism.
    out = re.sub(r"/private/var/folders/[^\s:,)\"']+", "<tmp>", out)
    out = re.sub(r"/var/folders/[^\s:,)\"']+", "<tmp>", out)
    out = re.sub(r"/tmp/[A-Za-z0-9_]{6,}", "<tmp>", out)
    # CPython prints object identity as a heap address in tracebacks
    # ("<test_core.TestMerge object at 0x10f8c6150>"). It is different on every run
    # and carries no information the agent can use, so it is the last thing standing
    # between two runs of the same fault and a byte-identical prompt.
    out = re.sub(r"0x[0-9a-fA-F]{6,}", "0xADDR", out)
    # Elapsed time varies run to run for the same reason.
    out = re.sub(r"\bin \d+\.\d+s\b", "in <t>s", out)
    return out


def run_suite(repo: Path) -> str:
    """Full pytest output with long tracebacks. One run serves every level."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-rf",
         "--tb=long", "-p", "no:cacheprovider"],
        cwd=repo, capture_output=True, text=True)
    return scrub_paths(proc.stdout, repo)


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
