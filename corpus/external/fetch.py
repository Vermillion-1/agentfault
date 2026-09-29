#!/usr/bin/env python3
"""Fetch external bug corpora at pinned commits, with licence rules enforced in code.

    python3 corpus/external/fetch.py status
    python3 corpus/external/fetch.py get bugsinpy
    python3 corpus/external/fetch.py get quixbugs

WHY A FETCHER AND NOT A CHECKED-IN COPY

Licence status differs per corpus and it is not cosmetic:

  QuixBugs  -- MIT. Safe to vendor, and small enough that vendoring makes the repo
               self-contained for a smoke test.
  BugsInPy  -- NO LICENCE FILE AT ALL. Default copyright applies. Research use is
               fine; REDISTRIBUTION IS NOT CLEARLY PERMITTED. So it is fetched into
               a gitignored directory and never committed. This is the same posture
               Defects4J takes with its own subject programs: fetch, do not vendor.

`redistributable: False` is therefore not a note, it is a switch -- such a corpus is
written only under corpus/external/, which .gitignore excludes, and `status` fails
loudly if a non-redistributable corpus is found tracked by git.

Also recorded here so it is never accidentally pulled in: BEARS is GPL-3.0, which
would be copyleft-incompatible with a permissively licensed harness.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL = ROOT / "corpus" / "external"
VENDOR = ROOT / "corpus" / "vendor"

G, R, Y, DIM, B, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"

# Ceiling for everything this script pulls, in GB. The agreed budget is 15 GB for the
# whole project; corpora get a slice, leaving room for Docker images later.
DISK_CEILING_GB = 6.0

SOURCES = {
    "quixbugs": {
        "url": "https://github.com/jkoppel/QuixBugs.git",
        "ref": "master",
        "licence": "MIT",
        "redistributable": True,
        "dest": VENDOR / "quixbugs",
        "why": "40 single-line defects, Python+Java, ~1 MB. Vendored as a fast smoke "
               "set. Known limitation: single-function algorithmic programs, not "
               "representative of repo-scale debugging -- do not report it alone.",
    },
    "bugsinpy": {
        "url": "https://github.com/soarsmu/BugsInPy.git",
        "ref": "master",
        "licence": "NONE DETECTED",
        "redistributable": False,
        "dest": EXTERNAL / "bugsinpy",
        "why": "493 real Python bugs across 17 projects, Defects4J-style CLI. The "
               "real-bug anchor for the mutation-validity argument. Harness is ~1 MB; "
               "per-project checkouts are fetched on demand and are the bulk of the "
               "disk cost.",
    },
}

FORBIDDEN = {
    "bears": "GPL-3.0 -- copyleft, incompatible with a permissive harness. Do not pull.",
}


def du_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    out = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True).stdout
    return int(out.split()[0]) / (1024 * 1024)


def free_gb() -> float:
    st = shutil.disk_usage(ROOT)
    return st.free / (1024 ** 3)


def cmd_status(_) -> int:
    print(f"\n  {B}EXTERNAL CORPORA{OFF}  {DIM}ceiling {DISK_CEILING_GB} GB, "
          f"{free_gb():.1f} GB free on disk{OFF}\n")
    total = 0.0
    bad = 0
    for name, s in SOURCES.items():
        size = du_gb(s["dest"])
        total += size
        present = f"{G}present{OFF}" if s["dest"].exists() else f"{DIM}absent{OFF}"
        redis = f"{G}vendorable{OFF}" if s["redistributable"] else f"{Y}fetch-only{OFF}"
        print(f"    {name:<12} {present:<22} {redis:<22} {s['licence']:<16} {size:.2f} GB")

        if not s["redistributable"] and s["dest"].exists():
            r = subprocess.run(["git", "ls-files", "--error-unmatch",
                                str(s["dest"].relative_to(ROOT))],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"      {R}TRACKED BY GIT — must not be committed{OFF}")
                bad += 1
    print(f"\n    total {total:.2f} GB of {DISK_CEILING_GB} GB\n")
    for n, why in FORBIDDEN.items():
        print(f"    {R}never fetch{OFF} {n}: {why}")
    print()
    return 1 if bad else 0


def cmd_get(a) -> int:
    s = SOURCES.get(a.name)
    if not s:
        sys.exit(f"unknown corpus {a.name!r}; known: {', '.join(SOURCES)}")
    if a.name in FORBIDDEN:
        sys.exit(f"refusing: {FORBIDDEN[a.name]}")

    used = sum(du_gb(x["dest"]) for x in SOURCES.values())
    if used >= DISK_CEILING_GB:
        sys.exit(f"disk ceiling reached ({used:.2f} GB of {DISK_CEILING_GB} GB)")
    if free_gb() < 2.0:
        sys.exit(f"refusing: only {free_gb():.1f} GB free on disk")

    dest = s["dest"]
    if dest.exists():
        print(f"  {a.name} already at {dest.relative_to(ROOT)}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {s['url']} @ {s['ref']} -> {dest.relative_to(ROOT)}")
    r = subprocess.run(["git", "clone", "--depth", "1", "--branch", s["ref"],
                        s["url"], str(dest)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"clone failed: {r.stderr[-400:]}")

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest,
                         capture_output=True, text=True).stdout.strip()

    if s["redistributable"]:
        # Strip the nested .git BEFORE anything can stage it. A clone that still has
        # its own .git is committed by the outer repo as a gitlink -- a bare commit
        # pointer -- so the files never actually land, and a fresh clone of this repo
        # gets an EMPTY directory while `git status` looks clean. "Vendored" would be
        # a false claim. Provenance is not lost: the pinned commit is recorded in the
        # .pinned.json written just below.
        shutil.rmtree(dest / ".git", ignore_errors=True)
    (dest.parent / f"{a.name}.pinned.json").write_text(json.dumps({
        "name": a.name, "url": s["url"], "ref": s["ref"], "commit": sha,
        "licence": s["licence"], "redistributable": s["redistributable"],
    }, indent=2) + "\n")
    print(f"  {G}ok{OFF} pinned at {sha[:12]}, {du_gb(dest):.2f} GB")
    if not s["redistributable"]:
        print(f"  {Y}fetch-only{OFF} — gitignored, never commit its contents")
    else:
        print(f"  {G}vendored{OFF} — nested .git stripped so the files are really committed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status"); st.set_defaults(fn=cmd_status)
    g = sub.add_parser("get"); g.add_argument("name"); g.set_defaults(fn=cmd_get)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
