#!/usr/bin/env python3
"""Generate categorised faults mechanically, and let the gate decide which survive.

    python3 tools/inject.py generate intervals --limit 200
    python3 tools/inject.py generate ledger --out corpus/faults/generated

WHY THIS WORKS WITHOUT A SCHEMA CHANGE

A fault in corpus/faults/*.json is (file, anchor, replacement) plus a label. A
mutation operator emits exactly that triple. And tools/corpus.py already refuses to
TRUST `expected_failing_tests` -- it runs the suite and demands set equality.

So the gate flips role. Hand-authored faults: a human writes the label, the gate
checks it. Generated faults: the operator proposes a change, and THE GATE DISCOVERS
THE LABEL AND FILTERS. The corpus builder and the validator are the same code.

THE FILTERS, AND WHY EACH EXISTS

  0 tests fail        -> equivalent mutant, or the code is untested. No signal, and
                         no way to tell an agent it is wrong. Discard.
  too many fail       -> the fault is everywhere, so "did the agent find it" is
                         trivially yes. Localisation needs a bounded blast radius.
  anchor not unique   -> materialise() requires exactly one occurrence, or an edit
                         could silently land on the wrong line.
  collection ERROR    -> a proxy for Gay & Salahirad (ICST 2023), who found that
                         some operators produce mutants making the WRONG tests fail
                         -- labelled, but with a failure signature unlike any real
                         fault. A pytest ERROR means import/collection broke, which
                         is a syntax-class failure we are not trying to generate.
                         This is a PROXY, not their criterion: the full version
                         needs coverage data to check the failing tests actually
                         exercise the mutated function. Documented, not hidden.

ON CATEGORY BALANCE

cosmic-ray only emits syntactically valid Python, so this corpus is Logic-heavy by
construction. That is deliberate. DebugBench and RepoDebug are dominated by Syntax
and Reference errors, which a linter catches -- a benchmark weighted that way can be
scored well by a linter wrapper.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.chunk import chunk_file          # noqa: E402
from tools.corpus import REPOS, run_pytest      # noqa: E402

G, R, Y, DIM, B, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"

# --- operator -> (ODC Type, ODC Qualifier, our coarse category) ------------------
#
# ODC is Chillarege et al., IEEE TSE 1992. Qualifier is the axis that matters here:
# mutation operators are good at Incorrect and structurally poor at Missing, which
# is why whole ODC cells are unreachable. docs/TAXONOMY.md records which.

BOUNDARY_SWAPS = {("Lt", "LtE"), ("LtE", "Lt"), ("Gt", "GtE"), ("GtE", "Gt")}


def classify(op_name: str) -> Tuple[str, str, str]:
    short = op_name.split("/", 1)[1]

    if short.startswith("ReplaceComparisonOperator"):
        parts = short.split("_")
        if len(parts) >= 3 and (parts[1], parts[2]) in BOUNDARY_SWAPS:
            return ("Checking", "Incorrect", "off_by_one")
        return ("Checking", "Incorrect", "logic_error")

    if short.startswith("ReplaceBinaryOperator"):
        return ("Algorithm", "Incorrect", "logic_error")
    if short.startswith("ReplaceUnaryOperator"):
        return ("Algorithm", "Incorrect", "logic_error")
    if short == "AddNot":
        return ("Checking", "Extraneous", "logic_error")
    if short in ("ReplaceAndWithOr", "ReplaceOrWithAnd",
                 "ReplaceTrueWithFalse", "ReplaceFalseWithTrue"):
        return ("Checking", "Incorrect", "logic_error")
    if short == "NumberReplacer":
        return ("Assignment", "Incorrect", "off_by_one")
    if short in ("ReplaceBreakWithContinue", "ReplaceContinueWithBreak",
                 "ZeroIterationForLoop"):
        return ("Algorithm", "Incorrect", "logic_error")
    if short == "RemoveDecorator":
        return ("Function", "Missing", "wrong_api")
    if short == "ExceptionReplacer":
        return ("Checking", "Incorrect", "wrong_api")
    if short == "VariableReplacer":
        return ("Assignment", "Incorrect", "logic_error")
    if short == "VariableInserter":
        return ("Assignment", "Extraneous", "logic_error")
    return ("Algorithm", "Incorrect", "logic_error")


# --- candidate generation --------------------------------------------------------

def single_hunk(before: str, after: str) -> Optional[Tuple[str, str, int]]:
    """Return (anchor, replacement, 1-based line) when the change is one contiguous
    run of lines. Multi-hunk mutations are skipped: the schema anchors on one snippet,
    and a multi-hunk fault muddies structural-scope labelling anyway."""
    a, b = before.splitlines(), after.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    if len(ops) != 1:
        return None
    tag, i1, i2, j1, j2 = ops[0]
    if tag != "replace" or (i2 - i1) != (j2 - j1):
        return None
    return "\n".join(a[i1:i2]), "\n".join(b[j1:j2]), i1 + 1


def owning_function(repo: Path, rel: str, line: int) -> Optional[str]:
    for c in chunk_file(repo / rel, repo):
        if c.start_line <= line <= c.end_line:
            return c.qualname
    return None


def candidates(repo: Path, rel: str, max_per_operator: int) -> List[dict]:
    from cosmic_ray.mutating import mutate_code
    from cosmic_ray.plugins import get_operator, operator_names

    src = (repo / rel).read_text()
    out: List[dict] = []
    for name in sorted(operator_names()):
        try:
            op = get_operator(name)()
        except Exception:
            continue
        for occ in range(max_per_operator):
            try:
                mutated = mutate_code(src, op, occ)
            except Exception:
                break
            if mutated is None or mutated == src:
                break
            hunk = single_hunk(src, mutated)
            if hunk is None:
                continue
            anchor, replacement, line = hunk
            if not anchor.strip() or src.count(anchor) != 1:
                continue          # materialise() needs exactly one occurrence
            fn = owning_function(repo, rel, line)
            if fn is None:
                continue
            odc_type, odc_qual, category = classify(name)
            out.append({
                "operator": name, "occurrence": occ, "file": rel, "function": fn,
                "line_hint": line, "anchor": anchor, "replacement": replacement,
                "odc_type": odc_type, "odc_qualifier": odc_qual, "category": category,
            })
    return out


# --- the gate, used as a filter --------------------------------------------------

def screen(repo_name: str, cands: List[dict], max_fail_frac: float,
           total_tests: int, timeout: float = 20.0) -> Tuple[List[dict], Dict[str, int]]:
    src_repo = REPOS / repo_name
    kept: List[dict] = []
    reasons = {"no_failures": 0, "too_broad": 0, "collection_error": 0,
               "nonterminating": 0, "kept": 0}

    for i, c in enumerate(cands, 1):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / repo_name
            shutil.copytree(src_repo, dest, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache"))
            target = dest / c["file"]
            text = target.read_text()
            if text.count(c["anchor"]) != 1:
                continue
            target.write_text(text.replace(c["anchor"], c["replacement"]))
            _, failing = run_pytest(dest, timeout=timeout)

        if failing == ["<timeout>"]:
            # The mutant hangs. Real category, not an error in our tooling.
            reasons["nonterminating"] += 1
            continue
        if not failing:
            reasons["no_failures"] += 1
            continue
        if any("::" not in f for f in failing):
            # a bare file path rather than a test node id means collection failed
            reasons["collection_error"] += 1
            continue
        if total_tests and len(failing) / total_tests > max_fail_frac:
            reasons["too_broad"] += 1
            continue

        c["expected_failing_tests"] = sorted(failing)
        c["blast_radius"] = len(failing)
        kept.append(c)
        reasons["kept"] += 1
        print(f"    {G}keep{OFF} [{i}/{len(cands)}] {c['operator'].split('/')[1][:44]:<44} "
              f"{c['function']:<22} breaks {len(failing)}")
    return kept, reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("repo")
    g.add_argument("--limit", type=int, default=0, help="cap candidates screened")
    g.add_argument("--max-per-operator", type=int, default=12)
    g.add_argument("--max-fail-frac", type=float, default=0.5)
    g.add_argument("--timeout", type=float, default=20.0,
                   help="per-candidate pytest timeout; mutants CAN hang")
    g.add_argument("--out", default="corpus/faults/generated")
    a = ap.parse_args()

    repo = REPOS / a.repo
    if not repo.is_dir():
        sys.exit(f"no repo at {repo}")

    with tempfile.TemporaryDirectory() as td:
        clean = Path(td) / a.repo
        shutil.copytree(repo, clean, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache"))
        passed, _ = run_pytest(clean)
        if not passed:
            sys.exit(f"{a.repo}: clean repo is not green; fix that first")
        import subprocess
        r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                            "-p", "no:cacheprovider"], cwd=clean,
                           capture_output=True, text=True)
        m = re.search(r"(\d+) tests? collected", r.stdout)
        total = int(m.group(1)) if m else 0

    sources = [p for p in sorted(repo.rglob("*.py"))
               if "tests" not in p.parts and "__pycache__" not in p.parts
               and not p.name.startswith("test_")]

    cands: List[dict] = []
    for p in sources:
        cands.extend(candidates(repo, str(p.relative_to(repo)), a.max_per_operator))
    if a.limit:
        cands = cands[:a.limit]

    print(f"\n  {B}GENERATE{OFF}  {a.repo}: {len(cands)} candidate mutants, "
          f"{total} tests in suite\n")
    kept, reasons = screen(a.repo, cands, a.max_fail_frac, total, a.timeout)

    outdir = ROOT / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    for n, c in enumerate(kept, 1):
        fid = f"{a.repo}-gen-{n:03d}"
        rec = {
            "id": fid, "repo": a.repo, "category": c["category"],
            "file": c["file"], "function": c["function"], "line_hint": c["line_hint"],
            "anchor": c["anchor"], "replacement": c["replacement"],
            "description": f"Generated by {c['operator']} (occurrence {c['occurrence']}).",
            "expected_failing_tests": c["expected_failing_tests"],
            "provenance": {
                "generator": "cosmic-ray", "operator": c["operator"],
                "occurrence": c["occurrence"],
                "odc_type": c["odc_type"], "odc_qualifier": c["odc_qualifier"],
            },
        }
        (outdir / f"{fid}.json").write_text(json.dumps(rec, indent=2) + "\n")

    print(f"\n  {B}screened {len(cands)}{OFF} -> {G}{reasons['kept']} kept{OFF}")
    print(f"    {DIM}discarded: {reasons['no_failures']} broke nothing, "
          f"{reasons['too_broad']} too broad, "
          f"{reasons['collection_error']} collection error, "
          f"{reasons['nonterminating']} non-terminating{OFF}")
    print(f"  written to {outdir.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
