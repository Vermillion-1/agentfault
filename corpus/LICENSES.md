# Corpus licences and redistribution status

Enforced in code, not just recorded here. `corpus/external/fetch.py` carries
`redistributable: True|False` per source; anything `False` is written only under
`corpus/external/`, which `.gitignore` excludes, and `fetch.py status` exits non-zero if such a
corpus is ever found tracked by git.

## In use

| Corpus | Licence | Redistributable | Location | Verified |
|---|---|---|---|---|
| **QuixBugs** | **MIT** (`LICENSE`, "Copyright 2017-2019 James Koppel") | **yes** — vendored | `corpus/vendor/quixbugs`, pinned `4257f44b0ff1` | licence file read directly |
| **BugsInPy** | **NONE** — no `LICENSE`/`COPYING` file exists | **no** — fetch only | `corpus/external/bugsinpy`, pinned `11c5f1eea954` | absence confirmed by listing the repo root |

**BugsInPy carries no licence file**, so default copyright applies. Research use is fine;
redistribution is not clearly permitted. It is therefore fetched and never committed — the same
posture Defects4J takes toward its own subject programs. The harness is ~1 MB; per-project checkouts
are pulled on demand and are the bulk of any disk cost.

Repos written for this project (`corpus/repos/intervals`, `corpus/repos/ledger`) are original work and
carry the repository's own licence.

## Deliberately not used

| Corpus | Why not |
|---|---|
| **BEARS** | **GPL-3.0.** Copyleft, incompatible with a permissively licensed harness. `fetch.py` refuses it by name. |
| **SWE-bench** family | ~120 GB of Docker images at default cache level, up to ~2 TB at instance level, 16 GB RAM recommended — beyond the 15 GB budget. Separately, OpenAI audited 27.6% of SWE-bench Verified and found 59.4% of those had flawed tests; Epoch AI rates it **Flawed**. |
| **Bugs.jar**, **RunBugRun** | No licence file detected, and neither is Python-first. |
| **DebugBench**, **MegaBugFix** | Repo licences are permissive, but the underlying problems derive from LeetCode, whose terms restrict the problem content. The repo licence does not cure that. |

## A caveat that applies to every benchmark here

**A harness licence is not a content licence.** Defects4J is MIT, but the 17 projects it checks out
carry their own licences and it does not vendor them. Any corpus that fetches real repositories
inherits per-repository terms, which is why agent traces over third-party code are published as
previews plus SHA-256 hashes rather than full payloads. See `docs/DATA_STATEMENT.md`.
