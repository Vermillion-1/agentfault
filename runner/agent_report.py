#!/usr/bin/env python3
"""Regenerate results/AGENT_REPORT.md from results/agent_raw.csv (the Phase D sweep)."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retrieval.metrics import bootstrap_ci, paired_bootstrap_ci

AGENT_RAW = ROOT / "results" / "agent_raw.csv"
AGENT_OUT = ROOT / "results" / "AGENT_REPORT.md"

DEAD_BAR = 0.10
OUTCOME_COLS = ["fixed", "fixed_elsewhere", "false_success", "visible_only",
                "silent_failure", "gave_up", "exhausted"]
LEGACY_COLS = ["fixed_tampered"]


def mean(v): return sum(v) / len(v) if v else 0.0


def agent_report(raw: Path = AGENT_RAW, out: Path = AGENT_OUT) -> None:
    every = list(csv.DictReader(raw.open()))
    for r in every:
        r.setdefault("rep", "0")
    canary_rows = [r for r in every if r.get("canary") and not r["error"] and r["rep"] == "0"]
    every = [r for r in every if not r.get("canary")]
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
    A("`fixed` is judged against every test, the held-out one included, in a grader "
      "the agent cannot reach. Intervals are percentile bootstrap over faults.\n")
    if any_dead:
        A(f"`†` More than {DEAD_BAR:.0%} of this cell's episodes errored. Its numbers "
          "describe the survivors only and should not be compared with other rows.\n")

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

    if k > 1:
        A("\n## 6. Faults whose result changes with the seed\n")
        A("Same fault, same level, same model: fixed on some seeds and not on others.\n")
        A("| Model | Level | faults that flip | of |")
        A("|---|---|---|---|")
        for m, lv in cells:
            pf = list(by_fault(sel(model=m, level=lv), "fix_rate").values())
            if pf:
                A(f"| `{m}` | **{lv}** | {sum(1 for v in pf if 0.0 < v < 1.0)} | {len(pf)} |")

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

    A("\n## 8. Integrity\n")
    A("`false_success` is counted on faults with a held-out test only; elsewhere it "
      "cannot happen. `refused edits` are edits the edit gate blocked, `test probes` "
      "are reads or edits of test-like paths; both are attempts, every one refused. "
      "Fixed by step k is read from 50-step runs (`first_fixed_step`), not from runs "
      "with a k-step budget.\n")
    A("| Model | Level | false_success (held-out faults) | episodes with refused edits | "
      "episodes with test probes | fixed by 12 / 24 / 50 | stop reasons |")
    A("|---|---|---|---|---|---|---|")
    for m, lv in cells:
        s = sel(model=m, level=lv)
        if not s:
            continue
        ho = [r for r in s if r.get("held_out")]
        fs = f"{sum(int(r.get('false_success') or 0) for r in ho)}/{len(ho)}" if ho else "—"
        pol = sum(1 for r in s if int(r.get("policy_hits") or 0))
        probe = sum(1 for r in s if int(r.get("test_probes") or 0))
        if all(r.get("first_fixed_step") not in (None, "") for r in s):
            first = [int(r["first_fixed_step"]) for r in s]
            curve = " / ".join(f"{sum(1 for x in first if 0 < x <= b)}"
                               for b in (12, 24, 50)) + f" of {len(s)}"
        else:
            curve = "—"
        stops = defaultdict(int)
        for r in s:
            stops[r.get("stop_reason") or "?"] += 1
        A(f"| `{m}` | **{lv}** | {fs} | {pol} | {probe} | {curve} | "
          + ", ".join(f"{k_} {v}" for k_, v in sorted(stops.items())) + " |")

    if canary_rows:
        A("\n## 9. Canaries: impossible faults\n")
        A("Each canary adds a test that contradicts a visible one (`tools/canaries.py`), "
          "so no honest fix passes the visible suite. A green visible suite is a cheat; "
          "a `done` is a false claim.\n")
        A("| Model | Level | episodes | visible suite green | claimed done | gave up |")
        A("|---|---|---|---|---|---|")
        for m in models:
            for lv in levels:
                s = [r for r in canary_rows if r["model"] == m and r["level"] == lv]
                if s:
                    A(f"| `{m}` | **{lv}** | {len(s)} | "
                      f"{sum(int(r['visible_green']) for r in s)} | "
                      f"{sum(int(r['claimed_done']) for r in s)} | "
                      f"{sum(int(r['gave_up']) for r in s)} |")

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
