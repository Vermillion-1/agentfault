"""Build the retrieval query from what the agent can actually see when tests fail."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

LEVELS = ["L0", "L1", "L2", "L3"]


def scrub_paths(text: str, repo: Path) -> str:
    """Replace machine-specific absolute paths with a stable placeholder."""
    out = text.replace(str(repo.resolve()), "<repo>").replace(str(repo), "<repo>")
    out = re.sub(r"/private/var/folders/[^\s:,)\"']+", "<tmp>", out)
    out = re.sub(r"/var/folders/[^\s:,)\"']+", "<tmp>", out)
    out = re.sub(r"/tmp/[A-Za-z0-9_]{6,}", "<tmp>", out)
    out = re.sub(r"0x[0-9a-fA-F]{6,}", "0xADDR", out)
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
