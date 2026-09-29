#!/usr/bin/env python3
"""Regenerate results/REPORT.md from results/retrieval_raw.csv.

The report is derived, never hand-written, so it cannot drift from the data.
Arm comparisons are reported WITHIN an observation level: pooling across levels
would mix L0's near-zero scores into every arm's distribution, inflating variance
and flattering nothing in particular.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retrieval.metrics import bootstrap_ci  # noqa: E402

RAW = ROOT / "results" / "retrieval_raw.csv"
OUT = ROOT / "results" / "REPORT.md"


def mean(v): return sum(v) / len(v) if v else 0.0


def main() -> int:
    rows = list(csv.DictReader(RAW.open()))
    arms = sorted({r["arm"] for r in rows})
    levels = sorted({r["level"] for r in rows})
    faults = sorted({r["fault"] for r in rows})

    def sel(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    L = []
    A = L.append
    A("# Retrieval arm sweep — results\n")
    A(f"Generated from `retrieval_raw.csv` by `runner/report.py`. "
      f"**{len(faults)} faults, {len(arms)} arms, {len(levels)} observation levels, "
      f"{len(rows)} cells.**\n")
    A("Ground truth is the fault label: the function an edit must touch to fix the bug. "
      "`recall_fn@k` asks whether that exact function appears in the top k. "
      "CIs are percentile bootstrap over the fault population.\n")

    # ---- headline: observation level
    A("\n## 1. Observation level dominates retrieval method\n")
    A("Pooled over all ranking arms. This is the largest effect in the study.\n")
    A("| Level | what the query contains | MRR (fn) | 95% CI | recall_fn@1 | recall_fn@3 |")
    A("|---|---|---|---|---|---|")
    desc = {"L0": "pass/fail only", "L1": "+ which tests failed",
            "L2": "+ assertion text", "L3": "+ full traceback"}
    for lv in levels:
        s = sel(level=lv)
        m = [float(r["mrr_fn"]) for r in s]
        lo, hi = bootstrap_ci(m)
        A(f"| **{lv}** | {desc[lv]} | {mean(m):.3f} | [{lo:.3f}, {hi:.3f}] | "
          f"{mean([float(r['recall_fn@1']) for r in s]):.2f} | "
          f"{mean([float(r['recall_fn@3']) for r in s]):.2f} |")

    # ---- arms, within level
    for lv in ("L2", "L3"):
        if lv not in levels:
            continue
        A(f"\n## 2{'ab'[lv == 'L3']}. Ranking arms at {lv} ({desc[lv]})\n")
        A("| Arm | MRR (fn) | 95% CI | recall_fn@1 | recall_fn@3 | recall_file@3 | median ms |")
        A("|---|---|---|---|---|---|---|")
        ranked = sorted(arms, key=lambda a: -mean([float(r["mrr_fn"]) for r in sel(level=lv, arm=a)]))
        for a in ranked:
            s = sel(level=lv, arm=a)
            m = [float(r["mrr_fn"]) for r in s]
            lo, hi = bootstrap_ci(m)
            lat = sorted(float(r["latency_ms"]) for r in s)
            A(f"| `{a}` | {mean(m):.3f} | [{lo:.3f}, {hi:.3f}] | "
              f"{mean([float(r['recall_fn@1']) for r in s]):.2f} | "
              f"{mean([float(r['recall_fn@3']) for r in s]):.2f} | "
              f"{mean([float(r['recall_file@3']) for r in s]):.2f} | "
              f"{lat[len(lat)//2]:.1f} |")

    # ---- grid
    A("\n## 3. Arm x level grid — recall_fn@3\n")
    A("| Arm | " + " | ".join(levels) + " |")
    A("|---" * (len(levels) + 1) + "|")
    for a in arms:
        cells = [f"{mean([float(r['recall_fn@3']) for r in sel(arm=a, level=lv)]):.2f}"
                 for lv in levels]
        A(f"| `{a}` | " + " | ".join(cells) + " |")

    # ---- category + blast radius
    A("\n## 4. By fault category (best arm per level, L2)\n")
    cats = sorted({r["category"] for r in rows})
    A("| Category | n faults | mean MRR (fn) @ L2 |")
    A("|---|---|---|")
    for c in cats:
        s = [r for r in sel(level="L2") if r["category"] == c]
        n = len({r["fault"] for r in s})
        A(f"| {c} | {n} | {mean([float(r['mrr_fn']) for r in s]):.3f} |")

    A("\n## 5. By blast radius — how many tests the fault breaks\n")
    A("A fault breaking one test is a harder localisation problem than one breaking seven.\n")
    A("| Tests broken | n faults | mean MRR (fn) @ L2 |")
    A("|---|---|---|")
    by = defaultdict(list)
    for r in sel(level="L2"):
        by[int(r["blast_radius"])].append(r)
    for k in sorted(by):
        n = len({r["fault"] for r in by[k]})
        A(f"| {k} | {n} | {mean([float(r['mrr_fn']) for r in by[k]]):.3f} |")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
