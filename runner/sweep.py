#!/usr/bin/env python3
"""The sweep: every retrieval arm x every observation level x every fault.

    python3 runner/sweep.py                 # full grid -> results/retrieval_raw.csv
    python3 runner/sweep.py --arms A2_bm25 A5_embed

WHAT IS AND IS NOT VARIED

Held fixed across every cell: the corpus, the chunker, the fault set, the gold
labels, and k. Varied: the ranking function (arm) and the query (observation
level). Any delta is therefore attributable to one of those two axes and nothing
else -- the property most retrieval comparisons lack.

ON SEEDS

SPEC.md requires >=3 seeds and bootstrap CIs. Seeds do not apply here: every arm
is deterministic given its input, so repeated runs are byte-identical and a seed
axis would be theatre. The real variance is ACROSS FAULTS, so CIs are bootstrapped
over the fault population instead. When the extrinsic (agent) arm lands, that one
IS stochastic and does need seeds.

Raw per-cell rows are committed as CSV. None of the papers in
../ai-harness-research/sources/ do this.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrieval import query as Q                    # noqa: E402
from retrieval.arms import ARMS                     # noqa: E402
from retrieval.chunk import chunk_repo              # noqa: E402
from retrieval.metrics import score                 # noqa: E402
from tools.corpus import load_faults, materialise   # noqa: E402

RESULTS = ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--levels", nargs="*", default=Q.LEVELS)
    ap.add_argument("-o", "--out", default=str(RESULTS / "retrieval_raw.csv"))
    a = ap.parse_args()

    faults = load_faults()
    rows = []
    print(f"\n  sweep: {len(faults)} faults x {len(a.arms)} arms x {len(a.levels)} levels "
          f"= {len(faults) * len(a.arms) * len(a.levels)} cells\n")

    for f in faults:
        with tempfile.TemporaryDirectory() as td:
            repo = materialise(f, Path(td) / f["repo"])
            queries = Q.build_all(repo)
            chunks = chunk_repo(repo)
            gold_uid = f"{f['file']}::{f['function']}"
            if gold_uid not in {c.uid for c in chunks}:
                print(f"    SKIP {f['id']}: gold {gold_uid} is not a chunk")
                continue

            # Pre-warm hosted embeddings in ONE batched call per fault instead of
            # one per observation level. Quota control, not speed.
            if any(x.startswith("A7") for x in a.arms):
                from retrieval.providers import embed_into_memo
                embed_into_memo([c.text for c in chunks], "search_document")
                embed_into_memo([queries[lv] for lv in a.levels], "search_query")

            for level in a.levels:
                for arm in a.arms:
                    t0 = time.perf_counter()
                    ranking = ARMS[arm](chunks, queries[level])
                    ms = (time.perf_counter() - t0) * 1000
                    m = score(ranking, gold_uid, f["file"])
                    rows.append({
                        "fault": f["id"], "repo": f["repo"], "category": f["category"],
                        "gold_fn": f["function"], "gold_file": f["file"],
                        "blast_radius": len(f["expected_failing_tests"]),
                        "level": level, "arm": arm,
                        "query_chars": len(queries[level]), "latency_ms": round(ms, 2),
                        **{k: round(v, 4) for k, v in m.items()},
                    })
            print(f"    {f['id']:<16} {len(chunks)} chunks  "
                  f"query chars L0/L3 = {len(queries['L0'])}/{len(queries['L3'])}")

    RESULTS.mkdir(exist_ok=True)
    out = Path(a.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    try:
        from retrieval.providers import CALLS
        print(f"  billable API calls this run: {CALLS}")
    except Exception:
        pass
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out          # an --out outside the repo is legitimate, e.g. a scratch replay
    print(f"\n  {len(rows)} rows -> {shown}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
