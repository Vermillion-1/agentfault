#!/usr/bin/env python3
"""Fault corpus: materialise, and verify that the labels are actually true.

    python3 tools/corpus.py verify            # every fault, full gate
    python3 tools/corpus.py verify -f ID      # one fault
    python3 tools/corpus.py materialise ID -o DIR

WHY THE GATE EXISTS

The Raft project had two fault injections (BUG-4, BUG-5) that silently did not
inject -- the suite reported green because nothing was actually being tested.
The lesson was "verify your verification", and it applies with more force here:
every downstream metric in this project is defined relative to a fault label.
If a fault does not break the tests it claims to break, or breaks tests it does
not claim, then localisation accuracy is measured against fiction.

So a fault is only valid when ALL of the following hold:

  1. clean checkout        -> every test passes
  2. injected checkout     -> the failing set is EXACTLY expected_failing_tests
  3. the anchor text appears exactly once in the target file

Condition 2 is a set equality, not a subset. A fault that breaks more than it
claims has bad localisation ground truth, which is just as corrosive as one
that breaks nothing.

Injection is copy-on-write: repos under corpus/repos/ are never mutated. A run
materialises a fresh copy into a scratch dir and patches that, so there is no
revert step to get wrong.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS = ROOT / "corpus" / "repos"
FAULTS = ROOT / "corpus" / "faults"

G, R, Y, DIM, B, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
OK, FAIL = f"{G}OK{OFF}  ", f"{R}FAIL{OFF}"


def load_faults(only=None):
    out = []
    for p in sorted(FAULTS.glob("*.json")):
        f = json.loads(p.read_text())
        f["_path"] = p
        if only is None or f["id"] == only:
            out.append(f)
    if only and not out:
        sys.exit(f"no fault with id {only!r}")
    return out


def materialise(fault, dest: Path) -> Path:
    """Copy the clean repo to dest and apply the fault. Never touches the original."""
    src = REPOS / fault["repo"]
    if not src.is_dir():
        sys.exit(f"missing repo {src}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache"))

    target = dest / fault["file"]
    text = target.read_text()
    anchor, repl = fault["anchor"], fault["replacement"]

    n = text.count(anchor)
    if n != 1:
        raise AssertionError(
            f"anchor must appear exactly once in {fault['file']}, found {n}")
    target.write_text(text.replace(anchor, repl))
    return dest


def run_pytest(cwd: Path):
    """Return (all_passed, sorted failing node ids)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-rf", "--tb=no", "-p", "no:cacheprovider"],
        cwd=cwd, capture_output=True, text=True)
    failing = sorted(set(re.findall(r"^(?:FAILED|ERROR)\s+(\S+)", proc.stdout, re.M)))
    passed = proc.returncode == 0
    return passed, failing


def verify(faults) -> int:
    bad = 0
    print(f"\n  {B}FAULT CORPUS{OFF}  {DIM}labels must be true, not plausible{OFF}\n")

    clean_cache = {}
    for f in faults:
        repo = f["repo"]
        if repo not in clean_cache:
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td) / repo
                shutil.copytree(REPOS / repo, dest, ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache"))
                clean_cache[repo] = run_pytest(dest)
        clean_passed, clean_failing = clean_cache[repo]
        if not clean_passed:
            print(f"    {FAIL}  {repo}: CLEAN repo is already failing -- {clean_failing[:3]}")
            bad += 1
            continue

        expected = sorted(set(f["expected_failing_tests"]))
        try:
            with tempfile.TemporaryDirectory() as td:
                dest = materialise(f, Path(td) / repo)
                _, actual = run_pytest(dest)
        except AssertionError as e:
            print(f"    {FAIL}  {f['id']:<16} {DIM}{f['category']}{OFF}  anchor: {e}")
            bad += 1
            continue

        if actual == expected:
            print(f"    {OK}  {f['id']:<16} {DIM}{f['category']:<14}{OFF} "
                  f"{f['function']:<14} breaks {len(actual)} test(s)")
            continue

        bad += 1
        missing = [t for t in expected if t not in actual]
        extra = [t for t in actual if t not in expected]
        print(f"    {FAIL}  {f['id']:<16} {DIM}{f['category']:<14}{OFF} label does not match reality")
        if not actual:
            print(f"            {R}fault injected but NOTHING failed -- this is the BUG-4 class{OFF}")
        for t in missing:
            print(f"            {R}claimed, did not fail:{OFF} {t}")
        for t in extra:
            print(f"            {Y}failed, not claimed:  {OFF} {t}")

    print()
    if bad:
        print(f"  {B}{R}{bad} of {len(faults)} fault(s) have untrue labels{OFF}\n")
    else:
        print(f"  {B}{G}all {len(faults)} fault labels verified{OFF}\n")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="check every fault label against reality")
    v.add_argument("-f", "--fault", help="verify only this fault id")

    m = sub.add_parser("materialise", help="write a faulty checkout to a directory")
    m.add_argument("fault")
    m.add_argument("-o", "--out", required=True)

    a = ap.parse_args()
    if a.cmd == "verify":
        sys.exit(1 if verify(load_faults(a.fault)) else 0)

    f = load_faults(a.fault)[0]
    dest = materialise(f, Path(a.out).resolve())
    print(f"{f['id']} -> {dest}")


if __name__ == "__main__":
    main()
