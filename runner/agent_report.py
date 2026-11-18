#!/usr/bin/env python3
"""Regenerate results/AGENT_REPORT.md from results/agent_raw.csv (the Phase D sweep).

    python3 runner/agent_report.py
    python3 runner/agent_report.py --raw some.csv --out some.md

Derived, never hand-written, like the retrieval report. Four rules shape every table:

THE FAULT IS THE UNIT. Ten seeds of one fault are ten looks at the same bug, not
ten independent bugs. Each fault is first reduced to its own rate over seeds, and
the bootstrap then resamples faults. Resampling episodes instead would shrink
every interval for free.

LEVELS ARE COMPARED PAIRED. L0 and L3 run on the same faults, so the contrast is
the per-fault difference, with its own interval (retrieval.metrics.paired_bootstrap_ci).

A DEAD CELL IS MARKED, NOT AVERAGED. When more than 10% of a cell's episodes
errored, the survivors are not a fair sample of the cell: the ones that died are
often the long, hard ones. Such a row is marked with a dagger. A cell with nothing
completed prints a dash, never a zero, because "not measured" is not "scored 0".

REPEATS ARE NOT SEEDS. `rep` > 0 rows re-run a cell with the response cache off,
to measure the noise floor. They appear only in the noise-floor section; every
other table uses rep 0.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retrieval.metrics import bootstrap_ci, paired_bootstrap_ci  # noqa: E402

AGENT_RAW = ROOT / "results" / "agent_raw.csv"
AGENT_OUT = ROOT / "results" / "AGENT_REPORT.md"

DEAD_BAR = 0.10
OUTCOME_COLS = ["fixed", "fixed_elsewhere", "false_success", "visible_only",
                "silent_failure", "gave_up", "exhausted"]
# Harness v1's class, gone since tests left the workspace; shown only for v1 CSVs.
LEGACY_COLS = ["fixed_tampered"]


def mean(v): return sum(v) / len(v) if v else 0.0


def agent_report(raw: Path = AGENT_RAW, out: Path = AGENT_OUT) -> None:
    every = list(csv.DictReader(raw.open()))
    for r in every:
        r.setdefault("rep", "0")            # CSVs written before repeats existed
    done = [r for r in every if not r["error"]]
    rows = [r for r in done if r["rep"] == "0"]
    if not rows:
        print(f"{raw.name}: no completed rows, nothing to report")
        return
    levels = sorted({r["level"] for r in every})
    models = sorted({r["model"] for r in every})
    faults = sorted({r["fault"] for r in rows})
    k = len({r["seed"] for r in rows})

    def sel(src=rows, **kw):
        return [r for r in src if all(r[c] == v for c, v in kw.items())]

    def by_fault(s, col):
        """{fault: mean of `col` over that fault's seeds}."""
        by = defaultdict(list)
        for r in s:
            by[r["fault"]].append(float(r[col]))
        return {f: mean(v) for f, v in by.items()}

    def dead(m, lv):
        """Share of this cell's attempted episodes that errored."""
        s = sel(every, model=m, level=lv)
        return len([r for r in s if r["error"]]) / len(s) if s else 0.0

    def tag(m, lv):
        return " †" if dead(m, lv) > DEAD_BAR else ""

    cells = [(m, lv) for m in models for lv in levels if sel(every, model=m, level=lv)]
    any_dead = any(dead(m, lv) > DEAD_BAR for m, lv in cells)

    L = []
    A = L.append
    A("# Agent sweep — results\n")
    n_err = len(every) - len(done)
    A(f"Generated from `{raw.name}` by `runner/agent_report.py`. **{len(faults)} faults, "
      f"{len(levels)} observation levels, {len(models)} model(s), {k} seed(s), "
      f"{len(rows)} episodes.**"
      + (f" {n_err} errored episodes are excluded and still need a re-run." if n_err else "")
      + "\n")
    A("`fixed` is judged against the original tests, so an agent that edits the tests "
      "cannot score. Intervals are percentile bootstrap over faults.\n")
    if any_dead:
        A(f"`†` More than {DEAD_BAR:.0%} of this cell's episodes errored. Its numbers "
          "describe the survivors only and should not be compared with other rows.\n")

    # ---- 1. headline
    A("\n## 1. Fix rate by observation level\n")
    A(f"pass@1 is the mean over seeds; pass^{k} asks that all {k} seed(s) of a fault "
      "were fixed.\n")
    A(f"| Model | Level | faults | pass@1 | 95% CI | pass^{k} | edited faulty fn | steps | tokens |")
    A("|---|---|---|---|---|---|---|---|---|")
    for m, lv in cells:
        s = sel(model=m, level=lv)
        if not s:
            A(f"| `{m}` | **{lv}**{tag(m, lv)} | 0 | — | — | — | — | — | — |")
            continue
        pf = list(by_fault(s, "fix_rate").values())
        lo, hi = bootstrap_ci(pf)
        A(f"| `{m}` | **{lv}**{tag(m, lv)} | {len(pf)} | {mean(pf):.2f} | [{lo:.2f}, {hi:.2f}] | "
          f"{mean([1.0 if v == 1.0 else 0.0 for v in pf]):.2f} | "
          f"{mean(list(by_fault(s, 'localised_fn').values())):.2f} | "
          f"{mean([float(r['steps']) for r in s]):.1f} | "
          f"{mean([float(r['prompt_tokens']) + float(r['completion_tokens']) for r in s]):.0f} |")

    # ---- 2. paired contrast against the lowest level
    if len(levels) > 1:
        base = levels[0]
        A(f"\n## 2. What more observation buys, against {base}\n")
        A(f"Per-fault difference in fix rate from {base}, on the faults both levels "
          "ran. An interval that excludes 0 is a difference this sample can see.\n")
        A("| Model | Level | faults | fix rate difference | 95% CI (paired) |")
        A("|---|---|---|---|---|")
        for m in models:
            b = by_fault(sel(model=m, level=base), "fix_rate")
            for lv in levels[1:]:
                c = by_fault(sel(model=m, level=lv), "fix_rate")
                both = sorted(set(b) & set(c))
                if not both:
                    A(f"| `{m}` | **{lv}** − {base} | 0 | — | — |")
                    continue
                x, y = [c[f] for f in both], [b[f] for f in both]
                lo, hi = paired_bootstrap_ci(x, y)
                A(f"| `{m}` | **{lv}** − {base}{tag(m, lv) or tag(m, base)} | {len(both)} | "
                  f"{mean(x) - mean(y):+.2f} | [{lo:+.2f}, {hi:+.2f}] |")

    # ---- 3. outcomes
    A("\n## 3. How episodes ended\n")
    A("Share of episodes per outcome. A pass/fail score on the tests the agent could "
      "run would count the first three columns as one number.\n")
    cols = OUTCOME_COLS + [c for c in LEGACY_COLS if any(r["outcome"] == c for r in rows)]
    A("| Model | Level | n | " + " | ".join(cols) + " |")
    A("|---" * (len(cols) + 3) + "|")
    for m, lv in cells:
        s = sel(model=m, level=lv)
        if not s:
            A(f"| `{m}` | **{lv}**{tag(m, lv)} | 0 | " + " | ".join("—" for _ in cols) + " |")
            continue
        A(f"| `{m}` | **{lv}**{tag(m, lv)} | {len(s)} | "
          + " | ".join(f"{mean([1.0 if r['outcome'] == c else 0.0 for r in s]):.2f}"
                       for c in cols) + " |")

    # ---- 4. per fault: the error-analysis view
    A("\n## 4. Fault by fault\n")
    A("The outcome of each fault at each level"
      + (f", as fixed / {k} seeds" if k > 1 else "")
      + ". Read down a column for what a level cannot solve, across a row for what "
      "more observation changes. `—` means not run or errored.\n")
    for m in models:
        if not sel(model=m):
            continue
        A(f"\n`{m}`\n")
        A("| Fault | category | tests broken | " + " | ".join(levels) + " |")
        A("|---" * (len(levels) + 3) + "|")
        for f in faults:
            s = sel(model=m, fault=f)
            if not s:
                continue
            cs = []
            for lv in levels:
                c = [r for r in s if r["level"] == lv]
                if not c:
                    cs.append("—")
                elif k == 1:
                    cs.append(c[0]["outcome"])
                else:
                    cs.append(f"{sum(int(r['fix_rate']) for r in c)}/{len(c)}")
            A(f"| `{f}` | {s[0]['category']} | {s[0]['blast_radius']} | " + " | ".join(cs) + " |")

    # ---- 5. cost
    A("\n## 5. Cost\n")
    A("Latency counts only calls that actually went out; a replayed response has "
      "no meaningful duration.\n")
    A("| Model | Level | model calls | of which cached | ms per uncached call |")
    A("|---|---|---|---|---|")
    for m, lv in cells:
        s = sel(model=m, level=lv)
        if not s:
            continue
        calls = sum(int(r["llm_calls"]) for r in s)
        cached = sum(int(r["llm_calls_cached"]) for r in s)
        ms = sum(float(r["llm_ms_uncached"]) for r in s)
        per = f"{ms / (calls - cached):.0f}" if calls > cached else "—"
        A(f"| `{m}` | **{lv}** | {calls} | {cached} | {per} |")

    # ---- 6. seed variance
    if k > 1:
        A("\n## 6. Faults whose result changes with the seed\n")
        A("Same fault, same level, same model: fixed on some seeds and not on others.\n")
        A("| Model | Level | faults that flip | of |")
        A("|---|---|---|---|")
        for m, lv in cells:
            pf = list(by_fault(sel(model=m, level=lv), "fix_rate").values())
            if pf:
                A(f"| `{m}` | **{lv}** | {sum(1 for v in pf if 0.0 < v < 1.0)} | {len(pf)} |")

    # ---- 7. noise floor
    groups = defaultdict(list)
    for r in sel(done, seed="0"):
        groups[(r["model"], r["level"], r["fault"])].append(r)
    groups = {g: v for g, v in groups.items() if len(v) > 1}
    if groups:
        A("\n## 7. Noise floor at temperature 0\n")
        A("The same cell run more than once at seed 0, with the response cache off "
          "after the first run. Any disagreement here is noise the provider adds on "
          "its own, and a difference between levels has to be larger than it to mean "
          "anything.\n")
        A("| Model | Level | faults repeated | runs each | outcome differs | fixed differs |")
        A("|---|---|---|---|---|---|")
        for m, lv in cells:
            g = [v for (gm, glv, _), v in groups.items() if (gm, glv) == (m, lv)]
            if not g:
                continue
            runs = sorted({len(v) for v in g})
            A(f"| `{m}` | **{lv}** | {len(g)} | {'–'.join(map(str, runs))} | "
              f"{sum(1 for v in g if len({r['outcome'] for r in v}) > 1)} | "
              f"{sum(1 for v in g if len({r['fix_rate'] for r in v}) > 1)} |")

    out.write_text("\n".join(L) + "\n")
    try:
        print(f"wrote {out.relative_to(ROOT)}")
    except ValueError:
        print(f"wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(AGENT_RAW))
    ap.add_argument("--out", default=str(AGENT_OUT))
    a = ap.parse_args(argv)
    agent_report(Path(a.raw), Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
