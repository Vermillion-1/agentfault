# Where this project is — resume from here

**Last updated:** 2026-09-29 · **Plan:** `~/.claude/plans/luminous-snuggling-sonnet.md` (agentfault v0.2)

## One-line status

Phases A, B and C are done and pushed. **Phase D is blocked on a Groq network block, nothing else.**

## First thing to do on resume

```sh
cd ~/Desktop/Projects/agentfault
python3 tools/env.py                      # keys load? (prints presence, never values)
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $GROQ_API_KEY" \
     -H 'User-Agent: curl/8.4.0' https://api.groq.com/openai/v1/models
```

**200 → Groq is back, go straight to Phase D.**
**403 → still blocked** (it was blocked all of 2026-09-29 after an earlier burst of calls; Cohere
was unaffected, returning 200 throughout, so this is Groq-side, not the key and not the network).
While blocked, everything except the live sweep still runs — tests, corpus generation, retrieval
sweep, and the whole trace layer are offline-capable.

Sanity check that nothing rotted:

```sh
python3 -m pytest tests/ -q                                  # offline, no key needed
python3 tools/corpus.py verify --prefix intervals-0          # 5 hand-written
python3 tools/corpus.py verify --prefix ledger-gen --sample 4
python3 corpus/external/fetch.py status
```

## What exists

| Phase | State | Where |
|---|---|---|
| A — corpus acquisition, licence-gated | **done** | `corpus/external/fetch.py`, `corpus/LICENSES.md` |
| B — mechanical injection | **done**, 211 generated faults | `tools/inject.py`, `corpus/faults/generated/` |
| C — four-layer provenance | **done**, test-enforced | `agent/trace.py`, `tests/test_trace.py` |
| D — the crossed sweep | **NOT STARTED — needs Groq** | would extend `runner/sweep.py` |
| E — ground-truth reconciliation | not started | short analysis on Phase D output |

**Corpus:** 12 hand-written + 211 generated faults, 2 repos (`intervals`, `ledger`), 16 functions,
53 faults that break exactly one test (the band where localisation discriminates).

**External:** QuixBugs vendored (MIT, 366 files, pinned `4257f44b0ff1`). BugsInPy fetched
(pinned `11c5f1eea954`) into a **gitignored** directory — it has no licence file, verified by
listing the repo root, so it must never be committed.

## Phase D, concretely

The crossed design is the contribution: **injected-fault location × observation level × model**,
scoring localisation against the injection rather than against a gold patch.

- Axes: 12 hand-written faults (start there, they are the best-understood) × L0–L3 × two models
  (`openai/gpt-oss-20b` weak, `openai/gpt-oss-120b` strong) × ≥10 seeds.
- Metrics already implemented in `agent/loop.py`: fix rate, localisation at **function** level,
  **silent-failure rate**, steps, tokens. Report an accuracy–cost Pareto, not a ranking.
- **Do not score file-level localisation alone** — two 2026 papers find it no longer discriminates.
- **Do not use an LLM judge** for process metrics — measured AUROC ≤ 0.65 on false success, worse
  than TF-IDF.
- Budget: Groq's binding limit is tokens/minute (~8k for `gpt-oss-20b`) against ~8.8k per episode,
  so roughly **one episode per minute**. A 48-episode single-seed pass is ~50 minutes unattended.
  `retrieval/providers.py` already paces a rolling token window.

Every episode should pass `trace_dir=` so Phase D produces the trace dataset as a side effect.

## Known-good invariants — if one of these breaks, something regressed

1. `tests/test_trace.py` passes with no API key and no network.
2. Reasoning text appears **only** in `reasoning.jsonl` — the test plants a sentinel and asserts
   its absence from the four published files.
3. Three independent runs of the same fault produce **byte-identical** L0–L3 prompts.
4. The retrieval sweep reproduces from the committed cache with **zero** API calls.
5. `.env` is never tracked; `tools/env.py` prints presence and length, never values.
6. BugsInPy never appears in `git status`.

## Open items

- [ ] **Phase D sweep** — blocked on Groq only.
- [ ] **Phase E** — compare gold-patch vs trajectory-derived vs injected-fault ground truth on the
      same instances. Short analysis, genuinely unoccupied ground.
- [ ] Verification item 8 from the plan: a script asserting every arXiv ID in `docs/` resolves to a
      paper whose title matches. Motivated by ICSE 2027's desk-reject rule for unverifiable
      references, and by the four bad citations already found in `../ai-harness-research/`.
- [ ] `flaky_dependency` fault category still uncovered — belongs in the v0.2 chaos layer, not in a
      corpus that must stay reproducible.
- [ ] Repo is **private**. `gh repo edit Vermillion-1/agentfault --visibility public` when ready.

## Findings so far, so context is not lost

- **Observation level dominates retrieval method.** Pooled over 11 arms: L0 MRR 0.220 → L3 0.718,
  non-overlapping CIs, and that gap exceeds the gap between any two ranking arms.
- **Cohere's retrieval models lead but do not separate.** At L3, `bm25→Cohere rerank` 0.850,
  `Cohere rerank` 0.833, `Cohere embed` 0.819 (the only arm with recall@3 = 1.00), against naive
  lexical 0.794. Every CI overlaps its neighbour at n=12.
- **RRF hybrid did not replicate** its published +18.1% advantage — it landed *below* both of its
  components here, consistent with embeddings degrading on exact-symbol queries.
- **Zero faults in any ODC `Missing` cell**, across both repos, and structurally none are possible.
- **Mutants can hang** — `break`→`continue` in a `while` loop. Screening needs a timeout.

## Context outside this repo

- `../ai-harness-research/` — literature scan, 7 PDFs, and `notes/06-the-decision.md`, which is the
  decision this project executes. **Four citations there were corrected on 2026-09-29** with dated
  `⚠️ CORRECTED` banners; see that folder's own git history.
- `../Commonplace/SPEC-commonplace.md` — a parked project, deliberately blocked behind agentfault
  v0.1 shipping.
- `../Resume/` (`job-apps-2027`) — job-application harness, unrelated except that agentfault is the
  portfolio artifact for those applications.
