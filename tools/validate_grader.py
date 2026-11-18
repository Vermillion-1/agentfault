#!/usr/bin/env python3
"""Check the grader against the known truth of every fault, SWE-bench style.

For each fault, on a fresh workspace:

  null patch   the faulty code as planted: grade() is red, and the failing tests
               are exactly the fault's expected_failing_tests
  visible      check() fails exactly the expected tests minus the held-out one, and
               neither its results nor its output name the held-out test
  true fix     undo the fault through the agent's own edit tool: grade() and
               check() are both green, so the edit gate let the fix through

    python3 tools/validate_grader.py                  # the 12 hand-written faults
    python3 tools/validate_grader.py --all            # all 223
    python3 tools/validate_grader.py --sandbox docker

A grader that fails any of these would score agents against the wrong truth.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.sandbox import BACKENDS, junit_key, open_workspace  # noqa: E402
from tools.corpus import FAULTS, REPOS, load_faults             # noqa: E402


def validate(fault: dict, backend: str = "local") -> dict:
    """{"id", "problems": [...], "slowest_s": seconds of the slowest test run}."""
    problems, slowest = [], 0.0

    def timed(fn):
        nonlocal slowest
        t0 = time.perf_counter()
        run = fn()
        slowest = max(slowest, time.perf_counter() - t0)
        return run

    with open_workspace(fault, backend) as ws:
        expected = {junit_key(n) for n in fault["expected_failing_tests"]}
        hidden = {junit_key(n) for n in ws.held_out}

        null = timed(ws.grade)
        if null.green or set(null.failing()) != expected:
            problems.append(f"null: green={null.green} failing={null.failing()} "
                            f"timed_out={null.timed_out}")
        vis = timed(ws.check)
        if set(vis.failing()) != expected - hidden:
            problems.append(f"visible: failing={vis.failing()}")
        if hidden & set(vis.results):
            problems.append("visible: held-out test was run")
        for node in ws.held_out:
            if node.split("::")[-1] in vis.output:
                problems.append("visible: held-out name in the output")

        faulty = (ws.root / fault["file"]).read_text()
        if faulty.count(fault["replacement"]) == 1:
            out = ws.call("edit", {"path": fault["file"], "old": fault["replacement"],
                                   "new": fault["anchor"]})
            if "edited" not in out:
                problems.append(f"fix: edit tool refused: {out}")
        else:
            # The planted text is not unique, so the edit tool cannot express the
            # fix; restore the file instead. The gate still checks it at grading.
            (ws.root / fault["file"]).write_text((REPOS / fault["repo"] / fault["file"]).read_text())
        for name, fn in (("grade", ws.grade), ("check", ws.check)):
            run = timed(fn)
            if not run.green:
                problems.append(f"fix: {name} red, failing={run.failing()} "
                                f"refused={run.refused} timed_out={run.timed_out}")
    return {"id": fault["id"], "problems": problems, "slowest_s": round(slowest, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="every fault, not just hand-written")
    ap.add_argument("--sandbox", choices=BACKENDS, default="local")
    a = ap.parse_args(argv)
    faults = load_faults()
    if not a.all:
        faults = [f for f in faults if f["_path"].parent == FAULTS]
    t0 = time.perf_counter()
    rows = [validate(f, a.sandbox) for f in faults]
    bad = [r for r in rows if r["problems"]]
    for r in bad:
        print(f"  {r['id']}: " + "; ".join(r["problems"]))
    slow = max(rows, key=lambda r: r["slowest_s"])
    print(f"{len(rows) - len(bad)}/{len(rows)} faults pass on {a.sandbox} "
          f"in {time.perf_counter() - t0:.0f}s; slowest test run {slow['slowest_s']}s "
          f"({slow['id']})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
