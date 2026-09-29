# Related work, and what this project does *not* claim

This file exists to stop the positioning drifting. It is easy, months into a project, to start
describing it as the first of its kind. It is not, and the specific ways it is not are written
down here so that claim can never be made by accident.

## Already done. Do not claim these.

| Claim we might be tempted to make | Who did it first |
|---|---|
| Ablating how much a coding agent can see about a failure | **Debug2Fix** (arXiv 2602.18571, Feb 2026) — debugger access vs none on GitBug-Java and SWE-Bench-Live. **>20% improvement**, and weaker models *with* a debugger match stronger models *without* one. This is our L0→L3 experiment, published first. |
| Silent failure as a headline metric | **Confident and Wrong** (arXiv 2603.25764) — 1,750 trajectories, 4 frontier models. Silent semantic failures are **80%** of Llama 4's failures and **68%** of GPT-5's. |
| Fault injection into agent systems with held-out ground truth | **AgentChaos** (ASE 2026, arXiv 2608.06790) and **AgentChaosBench** (arXiv 2608.14680) — 275 traces, fault type *and* location withheld. Best detector reaches **≤24.8%** top-1. |
| Synthetic code faults at scale with known locations | **SWE-smith** (NeurIPS 2025 D&B **Spotlight**, arXiv 2504.21798) — 50K instances from 128 repos. |
| Categorised fault injection into existing repos | **RepoDebug** (EMNLP 2025 Findings) — Tree-sitter AST injection, 8 languages, ~30,696 instances, 4×22 taxonomy. **DebugBench** (ACL 2024 Findings) — 4,253 LLM-injected bugs. |
| Oracle-localisation granularity ablation | **arXiv 2604.00167** — injects ground-truth localisation at file/function/line into Agentless; function-level wins. |
| Observability as the bottleneck | **The Observability Gap** (CHI 2026 Workshop, arXiv 2603.26942) — 0% success under output-only feedback, restored by minimal code-level knowledge. |

## What is actually unoccupied

1. **The crossing.** Each paper above varies *one* of {injected code fault, observation level,
   context seed}. None crosses injected-fault-location with graded observation level and scores
   **localisation against the injected ground truth**. The factorial is the contribution; neither
   factor alone is.
2. **Injected ground truth for *source-code* localisation.** AgentChaosBench does this for *runtime*
   faults — which component misbehaved. SWE-smith has the corpus but scores by test-pass, not by
   localisation-against-injection.
3. **Reconciling four incompatible definitions of "where the bug is."** Gold patch (SWE-bench
   lineage), regions successful trajectories consulted (SWE-Explore), human annotation
   (ContextBench), and injected fault. They disagree systematically: gold patches under-count
   context *read but not edited*; trajectory-derived truth is circular; annotation is subjective.
   No paper compares them on the same instances, and the injected fault is the only one independent
   of the other three.
4. **`provenance_origin`.** No trace standard records whether a step was harness-observed or
   agent-claimed. See `agent/trace.py`.
5. **Raw per-run data committed as an artifact** — verified by absence across the surveyed papers.

## Honest one-line positioning

> Fault injection for agents exists at the tool boundary; observation ablation exists for
> debuggers. No work crosses them, and none uses injected code faults as an independent ground
> truth for source-level localisation.

## Constraints the literature imposes on our own design

- **Do not score file-level localisation alone.** SWE-Explore and TraceProbe independently find
  file choice no longer separates success from failure. Score function and line.
- **Do not use LLM judges for process metrics.** *What Process Evaluation Actually Measures*
  (arXiv 2608.22960) finds full-trace LLM judges systematically biased toward semantic relevance
  over causal contribution; judges detect false success at **AUROC ≤ 0.65**, worse than TF-IDF.
- **Report an accuracy–cost Pareto, not a ranking.** HAL (ICLR 2026, 21,730 rollouts): the costliest
  model is on the frontier in only **1 of 9** benchmarks.
- **≥10 runs per cell with power analysis**, reporting pass@1, pass@k and pass^k. Not "≥30" — that
  figure is not in the source usually cited for it.
- **Record the full harness config.** *Same Signal, Different Semantics* (64,380 runs, 126 configs):
  the same behavioural signal carries opposite meaning across configurations; framework identity
  explains **64%** of variance in mean turns, LLM family only **10%**.

## Citation hygiene

Four citations in this author's own earlier research notes did not survive checking — a
misattributed run-count recommendation, a robotics paper cited as software, a benchmark name
collision, and a novelty claim already superseded. ICSE 2027 desk-rejects submissions containing
unverifiable references. Every arXiv ID in this directory should be checked against the live
listing before it appears in anything submitted.
