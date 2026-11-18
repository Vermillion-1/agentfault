"""Impossible faults (tools/canaries.py) and the report's integrity sections.

  a canary an honest fix can pass   -> a cheating rate that counts honest work
  canaries inside the fix rate      -> a fix rate lowered by impossible tasks
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import runner.agent_sweep as sweep                 # noqa: E402
from runner.report import agent_report             # noqa: E402
from tools.canaries import load_canaries, validate  # noqa: E402


def test_every_stored_canary_is_impossible():
    canaries = load_canaries()
    assert len(canaries) >= 5
    assert {c["id"]: validate(c) for c in canaries} == {c["id"]: [] for c in canaries}


def test_canaries_are_reported_apart_from_the_fix_rate():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        assert sweep.main(["--fake", "--faults", "intervals-001", "--canaries",
                           "--levels", "L3", "-o", str(tmp / "raw.csv"),
                           "--traces", str(tmp / "traces")]) == 0
        rows = list(csv.DictReader((tmp / "raw.csv").open()))
        canary = [r for r in rows if r["canary"]]
        assert len(canary) == len(load_canaries())
        # The scripted true fix, on an impossible fault: never green, never fixed.
        assert all(r["fix_rate"] == "0" and r["visible_green"] == "0" for r in canary)
        assert all(r["outcome"] == "silent_failure" for r in canary)

        md = tmp / "AGENT_REPORT.md"
        agent_report(tmp / "raw.csv", md)
        text = md.read_text()
        assert "**1 faults" in text, "canaries never count as faults in the headline"
        assert "## 8. Integrity" in text and "## 9. Canaries" in text
        row = [ln for ln in text.split("## 9.")[1].splitlines() if "**L3**" in ln][0]
        assert [c.strip() for c in row.strip("|").split("|")][2:] == [
            str(len(canary)), "0", str(len(canary)), "0"]
