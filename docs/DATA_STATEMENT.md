# Data statement

Covers the fault corpus and the agent traces.

## What is in the artifact

| Part | Contents | Licence posture |
|---|---|---|
| `corpus/repos/` | Two Python packages written for this project | This repository's licence |
| `corpus/faults/` | Hand-authored and generated fault definitions | This repository's licence |
| `corpus/vendor/quixbugs` | QuixBugs at pinned commit `4257f44b0ff1` | **MIT**, upstream LICENSE retained |
| `corpus/external/` | Fetched corpora | **Not distributed.** Gitignored. See `corpus/LICENSES.md` |
| `results/traces/` | Agent episode traces | This repository's licence, with the carve-outs below |

## Agent traces

**Payloads are previews plus hashes, not full bodies.** Every prompt, response and tool result is
stored as a truncated preview (240 chars) plus the SHA-256 of the complete payload. This keeps
integrity and tamper-evidence while avoiding republication of whole prompts or of third-party source
code that an agent happened to read. It is also the mechanism that makes provenance harness-owned:
a hash of the actual HTTP response is not something the agent can fabricate.

**Reasoning traces are a separate file.** `reasoning.jsonl` holds raw chain-of-thought and is
withholdable by simply not shipping it. A test asserts reasoning text never appears in the four
published files.

**Publication tier is decided by the model.**

| Tier | Models | Posture |
|---|---|---|
| Publishable | `gpt-oss` (Apache-2.0, open weights), DeepSeek-R1 | Raw CoT may be stored *and* published |
| Store-only | OpenAI o-series, Anthropic Claude | Provider-generated *summaries* under provider terms. Kept for debugging; not republished in bulk |
| Never | Any closed model's hidden CoT | Extraction is prohibited. **Not attempted here.** |

**We did not extract concealed reasoning from any closed model.** Published work shows this is
technically feasible; it is prohibited under provider terms, and no part of this project does it.

## Content warning

Raw chain-of-thought is **not aligned output**. Model providers' stated reason for hiding it is that
it may contain unsafe, deceptive or offensive intermediate text. Anyone using `reasoning.jsonl`
should expect content that would not appear in a final response.

## Faithfulness caveat

A reasoning trace is evidence of what the model **emitted**, not of what **caused** its action. Work
on chain-of-thought faithfulness shows the two can diverge. Any analysis that treats reasoning text
as a causal explanation is overclaiming — which is, pleasingly, the same epistemics this project
applies to agent self-report one level up.

## Known limitations of the corpus

- **No omission bugs.** Zero faults in any ODC `Missing` cell, and structurally none are reachable
  by mutation. See `docs/TAXONOMY.md`.
- **Logic-class only.** No syntax or reference errors, deliberately.
- **Small repos.** Two hand-written packages, 16 functions covered. Findings do not transfer to
  repo-scale debugging without the real-bug anchor.
- **Fault realism is unvalidated.** Written for this project, which removes training-data
  contamination but means injected bugs may be easier or stranger than real ones.
- **The wrong-failure-signature filter is a proxy**, not the published criterion it approximates.

## Reproducibility

Prompts are byte-identical across runs: temp paths, heap addresses and elapsed times are scrubbed
from observation text. Hosted responses are disk-cached, so the retrieval sweep reproduces from a
clean checkout with **zero API calls**. Every fault label is re-derivable by running
`tools/corpus.py verify`.
