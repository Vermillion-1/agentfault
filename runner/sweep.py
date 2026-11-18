#!/usr/bin/env python3
"""The sweep: every retrieval arm x every observation level x every fault."""
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

from retrieval import query as Q
from retrieval.arms import ARMS
from retrieval.chunk import chunk_repo
from retrieval.metrics import score
from tools.corpus import FAULTS, load_faults, materialise

RESULTS = ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--levels", nargs="*", default=Q.LEVELS)
    ap.add_argument("-o", "--out", default=str(RESULTS / "retrieval_raw.csv"))
    ap.add_argument("--generated", action="store_true",
                    help="also sweep the 211 mechanically generated faults")
    a = ap.parse_args()

    faults = [f for f in load_faults()
              if a.generated or f["_path"].parent == FAULTS]
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
        shown = out
    print(f"\n  {len(rows)} rows -> {shown}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
