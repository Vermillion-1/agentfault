# agentfault

**Inject known faults into an agent's environment, vary how much the agent can see about what went
wrong, and measure whether it notices, whether it diagnoses correctly, and how often it confidently
ships a wrong answer.**

Decided 2026-09-11 in `../ai-harness-research/notes/06-the-decision.md`. Started 2026-09-28.

## The contribution

Fault injection gives **ground-truth failure attribution**. You normally cannot evaluate a
failure-diagnosis method, because in real failures nobody knows the true cause — ASPIRE reports
success rates but not repair precision for exactly this reason. If you inject the bug, you know its
file, its function, its category. So *"did the agent localise the actual cause?"* becomes measurable.

Everything else here is careful engineering.

## Status

| Phase | State |
|---|---|
| Fault corpus + injector + label gate | **working** — 12 hand-written + **211 generated**, 2 repos, labels verified |
| Mechanical fault generation (cosmic-ray) | **working** — `tools/inject.py`, gate used as filter |
| External corpora, licence-gated fetch | **working** — QuixBugs vendored, BugsInPy fetch-only |
| Retrieval sweep (11 arms x 4 observation levels) | **working** — 528 cells, results committed |
| Minimal ReAct agent (native tool calling) | **working** — verified end to end on one fault |
| Batch agent sweep with seeds | not started |
| Chaos layer (v0.2) | not started |

```sh
python3 tools/corpus.py verify     # gate every fault label
python3 tools/corpus.py verify --prefix intervals-gen --sample 12   # sample a generated set
python3 tools/inject.py generate intervals    # mechanically generate + screen
python3 corpus/external/fetch.py status       # external corpora + licence posture
python3 runner/sweep.py            # retrieval grid -> results/retrieval_raw.csv
python3 runner/report.py           # regenerate results/REPORT.md from the CSV
python3 tools/env.py               # confirm keys load (prints presence, never values)
```

Keys live in a gitignored `.env`; `tools/env.py` reports only presence and length.
Hosted responses are disk-cached in `results/.cache/`, so re-running the sweep costs
zero API calls and stays byte-identical.

## Results so far

Full tables in [`results/REPORT.md`](results/REPORT.md), regenerated from the raw CSV.

**1. Observation quality dominates retrieval method.** Pooled over all 11 arms:

| Level | query contains | MRR (fn) | 95% CI |
|---|---|---|---|
| L0 | pass/fail only | 0.220 | [0.192, 0.251] |
| L1 | + which tests failed | 0.552 | [0.500, 0.606] |
| L2 | + assertion text | 0.608 | [0.551, 0.666] |
| L3 | + full traceback | 0.718 | [0.660, 0.776] |

L0 and L3 do not overlap, and that gap is larger than the gap between any two ranking
arms. Telling the retriever *which test failed* buys more than any choice of retriever.

**2. At L3, hosted rerank leads and every arm's CI overlaps its neighbour's.**

| Arm | MRR (fn) | 95% CI | recall_fn@3 |
|---|---|---|---|
| `A9_bm25_then_cohere_rerank` | 0.850 | [0.675, 1.000] | 0.92 |
| `A8_cohere_rerank` | 0.833 | [0.653, 1.000] | 0.92 |
| `A7_cohere_embed` | 0.819 | [0.667, 0.958] | **1.00** |
| `A1_lexical` | 0.794 | [0.617, 0.958] | 0.92 |
| `A5_embed` (MiniLM) | 0.711 | [0.510, 0.896] | 0.75 |
| `A0_none` | 0.198 | [0.142, 0.269] | 0.17 |

Cohere Embed v3 is the only arm that puts the faulty function in the top 3 on **every**
fault. But at n=12 the confidence intervals overlap heavily, so the honest claim is
that hosted retrieval models lead a ranking no arm statistically separates in — not
that any arm is proven better.

**3. A published finding did not replicate.** `../ai-harness-research/README.md` records
RRF hybrid beating both its components by +18.1% MRR. Here `A6_hybrid_rrf` (0.715) lands
*below* both BM25 (0.771) and plain lexical overlap (0.794). Fusing a weak arm with a
strong one dragged the strong one down. Consistent with
`../Context-Selection/SPEC-context-selection.md` §1, which predicted embeddings would
"degrade on exact-symbol queries" — that prediction now has a number.

## Known limitations

Stated because they bound every number above.

- **n = 12 faults across 2 repos.** Adjacent arms are not separated. More faults is the
  single highest-value next step, not more arms.
- **Latency figures for hosted arms are contaminated by the response cache.** Only the
  first call to each endpoint measures real network latency; the medians in the report
  mix cached and uncached calls and should not be quoted as API latency.
- **Fault realism is unvalidated.** Repos were written for this project, which removes
  training-data contamination but means injected bugs may be easier or stranger than
  real ones. Sampling from real commit history is the fix.
- **The extrinsic arm is smoke-tested, not swept.** One episode fixed one fault end to
  end (`intervals-001` at L3: fix_rate 1, edited inside the faulty function, 5 steps).
  Whether observation level changes *agent* success is still unmeasured.
- **`flaky_dependency` faults are uncovered.** They need nondeterminism, which belongs
  in the v0.2 chaos layer rather than in a corpus that must stay reproducible.
- **Groq free tier is token-per-minute bound**, ~8k TPM against ~8.8k per episode, so
  throughput is roughly one episode per minute. `retrieval/providers.py` paces a rolling
  token window against it.

## The label gate, and why it is the first thing built

Every downstream metric is defined relative to a fault label. If a fault does not break the tests it
claims to break, localisation accuracy is measured against fiction.

The Raft project had two fault injections — BUG-4 and BUG-5 — that **silently did not inject**. The
suite reported green because nothing was being tested. `tools/corpus.py verify` is that lesson made
executable. A fault is valid only when:

1. the clean checkout passes every test,
2. the injected checkout's failing set **exactly equals** `expected_failing_tests`, and
3. the anchor text appears exactly once in the target file.

Condition 2 is set equality, not containment. A fault that breaks *more* than it claims has bad
localisation ground truth, which corrupts the measurement just as badly as one that breaks nothing.

The gate is itself tested: a deliberately inert probe fault (dropping a defensive `max()` that
cannot change behaviour, while claiming a test failure) is correctly rejected and reported as the
BUG-4 class.

## Design decisions

**Copy-on-write injection.** `corpus/repos/` is pristine and never mutated. Injection copies a repo
into a scratch dir and patches the copy, so there is no revert step to get wrong and no way for a
crashed run to leave a dirty tree.

**Anchored replacement, not diffs.** A fault names exact original text that must appear exactly once.
Patches drift against edited files and fail silently at the hunk level; an anchor that no longer
matches is a hard error.

**Hand-written repos, not SWE-bench.** `Context-Selection/SPEC-context-selection.md` §5 flags
contamination: SWE-bench repos are widely present in training data, so absolute scores are not
trustworthy there. These repos were written for this project and are in no training set, which means
absolute numbers are interpretable here, not just relative deltas. The cost is fault realism — see
Risks.

## Corpus

`corpus/repos/intervals` — half-open `[start, end)` interval arithmetic, 92 lines, 29 tests. Chosen
because nearly every boundary decision has a plausible-looking wrong version, which is what makes
injected faults resemble real ones.

| Fault | Category | Function | Breaks |
|---|---|---|---|
| `intervals-001` | off_by_one | `merge` | 3 |
| `intervals-002` | logic_error | `total_length` | 2 |
| `intervals-003` | off_by_one | `Interval.overlaps` | 1 |
| `intervals-004` | off_by_one | `find_gaps` | 4 |
| `intervals-005` | wrong_api | `intersect` | 4 |

The 1-to-4 spread in blast radius is deliberate: a fault breaking one test is a much harder
localisation problem than one breaking four, and the metric should be able to see that difference.

`corpus/repos/ledger` — a double-entry ledger over CSV rows, 88 lines, 24 tests. Added because
`intervals` has no imports used by a single function and no parsing, so `missing_import` and
`wrong_api` faults had nowhere to live without taking down the whole suite.

**12 faults across 4 categories**, blast radius 1 to 7 tests. `flaky_dependency` remains uncovered
by design — see Limitations.

## Risks

- **Fault realism.** Injected bugs may be easier or weirder than real ones. Sample some from real
  commit history before claiming generality.
- **Statistical power.** Agent runs are noisy. Make the L0-vs-L3 contrast as stark as possible;
  ASPIRE's was 14% vs 62%, so a large effect is plausible.
- **Scope creep.** The failure mode is building a framework instead of running an experiment. The
  agent is a means, not the product. ~300 lines, then measure.

## Done looks like

One-command reproduction, a chaptered docs site, `existing_issues.md`, **raw run data committed as
CSV** (which none of the papers in `../ai-harness-research/sources/` do), and a short writeup:
*"How much of an agent's success is the harness?"*
