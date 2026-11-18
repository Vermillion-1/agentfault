#!/usr/bin/env python3
"""Phase D: the agent sweep. Every fault x observation level x model x seed.

    python3 runner/agent_sweep.py --levels L0 L3 --seeds 0        # smoke, 24 episodes
    python3 runner/agent_sweep.py --models openai/gpt-oss-20b openai/gpt-oss-120b
    python3 runner/agent_sweep.py --seeds 0 1 2 3 4 5 6 7 8 9     # the full plan
    python3 runner/agent_sweep.py --levels L3 --repeats 3         # noise floor
    python3 runner/agent_sweep.py --fake --seeds 0 1 2 3 4        # rehearsal, no API
    python3 runner/agent_sweep.py --sandbox local --max-steps 12  # offline backend, v1 budget

WHAT IS AND IS NOT VARIED

Held fixed: the corpus, the car (agent loop, tools, step limit, system prompt: one
CarSpec), the provider and the sandbox. Varied: the fault, how much the test
runner tells the agent (L0..L3), the model, and the seed. Seed 0 runs at
temperature 0, every other seed at 0.7.

Everything held fixed is hashed into one `config` value per row. A CSV holds one
config: resuming with another provider, sandbox, step limit or car stops the
sweep instead of mixing two set-ups in one table.

RUNS DIE PARTWAY, SO THIS RESUMES

A cell is (fault, level, model, seed). Its row is written to the CSV the moment
the episode ends. Re-running the same command skips every cell that already has a
row without an error, and re-runs the ones that errored. Nothing is lost by
killing the sweep, and nothing is paid for twice.

AND IT STOPS WHEN THE PROVIDER IS DOWN

On 2026-09-29 Groq answered 403 for a whole day. Carrying on would have filled
the rest of the grid with error rows. So: an auth error stops the sweep at once,
and so do --max-errors provider errors in a row.

Every episode also writes its four-layer trace, so the sweep produces the trace
dataset as a side effect.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.loop as loop                               # noqa: E402
from agent import sandbox                               # noqa: E402
from agent.fake import BEHAVIOURS, failing_model, scripted_model  # noqa: E402
from retrieval import query as Q                        # noqa: E402
from retrieval.providers import chat_endpoint, configure  # noqa: E402
from tools.corpus import FAULTS, load_faults            # noqa: E402

RESULTS = ROOT / "results"

# Columns copied straight from the episode's Result, in this order.
RESULT_COLS = ["outcome", "fix_rate", "visible_green", "claimed_done", "gave_up",
               "false_success", "silent_failure", "localised", "localised_fn",
               "saw_gold_fn", "first_fixed_step", "policy_hits", "test_probes",
               "grade_timed_out", "stop_reason", "steps", "edits", "prompt_tokens",
               "completion_tokens", "reasoning_tokens", "llm_calls", "llm_calls_cached",
               "llm_ms_uncached", "error_kind", "error"]
COLS = ["cell", "fault", "repo", "category", "blast_radius", "held_out", "level", "model",
        "seed", "rep", "provider", "sandbox", "max_steps", "config"] + RESULT_COLS


def config_id(provider: str, backend: str, car: "loop.CarSpec") -> str:
    """One hash over everything a sweep holds fixed."""
    image = sandbox.image_id() if backend == "docker" else None
    blob = json.dumps({"provider": provider, "sandbox": backend, "image": image,
                       "grade_timeout_s": sandbox.GRADE_TIMEOUT, "car": car.digest()},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def cell_id(fault: str, level: str, model: str, seed: int, rep: int = 0) -> str:
    base = f"{fault}__{level}__{model.replace('/', '_')}__s{seed}"
    return base if rep == 0 else f"{base}__r{rep}"


def read_rows(path: Path) -> dict:
    """Existing rows keyed by cell, so a re-run can see what is already done."""
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {r["cell"]: r for r in csv.DictReader(fh)}


def write_rows(path: Path, rows: dict) -> None:
    """Rewrite the whole CSV through a temp file, then rename over the old one.

    The sweep can be killed at any moment. A rename is atomic, so the file on disk
    is always either the old complete CSV or the new complete CSV, never half of
    one -- and a half-written CSV is exactly what would break resuming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--faults", nargs="*",
                    help="fault ids (default: the hand-written faults)")
    ap.add_argument("--levels", nargs="*", default=Q.LEVELS)
    ap.add_argument("--models", nargs="*", default=["openai/gpt-oss-20b"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each cell this many times; repeats after the first "
                         "bypass the response cache, to measure the noise floor")
    ap.add_argument("-o", "--out", help="CSV path (default results/agent_raw.csv)")
    ap.add_argument("--traces", help="trace root (default results/traces/phase-d)")
    ap.add_argument("--max-errors", type=int, default=3,
                    help="stop after this many provider errors in a row")
    ap.add_argument("--provider",
                    help="groq, openrouter:<host> (e.g. openrouter:akashml/fp4), or a URL; "
                         "default: whatever AGENTFAULT_CHAT_* say, else groq")
    ap.add_argument("--sandbox", choices=sandbox.BACKENDS, default="docker",
                    help="where tools and tests run (default docker; --fake forces local)")
    ap.add_argument("--max-steps", type=int, default=loop.CarSpec().max_steps,
                    help="the step budget; the agent is never told it")
    ap.add_argument("--fake", action="store_true",
                    help="scripted model, no API calls; behaviour cycles with the seed")
    ap.add_argument("--fake-fail", action="store_true",
                    help="with --fake: every call fails, to rehearse the circuit breaker")
    a = ap.parse_args(argv)

    # A rehearsal must never land in the real results. `results/*.tmp` is
    # gitignored, so the fake defaults live there.
    out = Path(a.out) if a.out else RESULTS / (
        "fake_agent_raw.tmp" if a.fake else "agent_raw.csv")
    traces = Path(a.traces) if a.traces else RESULTS / (
        "fake_traces.tmp" if a.fake else "traces/phase-d")
    if a.provider and not a.fake:
        configure(a.provider)
    provider = "fake" if a.fake else chat_endpoint()[2]
    # A scripted model is trusted code, so the offline backend is enough for it.
    backend = "local" if a.fake else a.sandbox
    car = loop.CarSpec(max_steps=a.max_steps)
    config = config_id(provider, backend, car)

    faults = load_faults()
    if a.faults:
        faults = [f for f in faults if f["id"] in set(a.faults)]
    else:
        # Generated faults live in corpus/faults/generated/. Start with the 12
        # hand-written ones: they are the best understood.
        faults = [f for f in faults if f["_path"].parent == FAULTS]
    if not faults:
        sys.exit("no faults selected")

    # A repeat is the SAME cell run again. Rep 0 goes through the response cache
    # like everything else. Later reps bypass it, because a replayed response would
    # agree with rep 0 by construction and say nothing about real repeatability.
    cells = [(f, lv, m, s, rep) for f in faults for lv in a.levels
             for m in a.models for s in a.seeds for rep in range(a.repeats)]
    rows = read_rows(out)
    # A cell id names only what varies, so resuming under another provider, sandbox
    # or car would skip cells the other set-up ran and mix two set-ups in one table.
    others = {(r.get("config", ""), r.get("provider", ""), r.get("sandbox", ""),
               r.get("max_steps", "")) for r in rows.values()
              if r.get("config", "") != config}
    if others:
        sys.exit(f"{out.name} holds rows from another set-up "
                 f"(config, provider, sandbox, max_steps) = {sorted(others)}; this run is "
                 f"{(config, provider, backend, a.max_steps)}. "
                 "Use the same flags, or a different -o.")
    done = {c for c, r in rows.items() if not r.get("error")}
    todo = [c for c in cells if cell_id(c[0]["id"], *c[1:]) not in done]
    print(f"\n  agent sweep: {len(faults)} faults x {len(a.levels)} levels x "
          f"{len(a.models)} models x {len(a.seeds)} seeds x {a.repeats} repeats = "
          f"{len(cells)} cells, {len(cells) - len(todo)} already done, "
          f"{len(todo)} to run  [{provider}, {backend}, {a.max_steps} steps, "
          f"config {config}]\n")

    real_chat = loop.chat
    streak = 0
    stopped = ""
    try:
        for f, level, model, seed, rep in todo:
            cid = cell_id(f["id"], level, model, seed, rep)
            if a.fake:
                loop.chat = (failing_model() if a.fake_fail else
                             scripted_model(f, BEHAVIOURS[seed % len(BEHAVIOURS)]))
            # A fresh sealed workspace per cell: one episode's edits must never be
            # visible to the next.
            with sandbox.open_workspace(f, backend) as ws:
                res = loop.Episode(ws, f, level=level, car=car, seed=seed, model=model,
                                   trace_dir=traces / cid, provider=provider,
                                   use_cache=(rep == 0)).run()
            d = dataclasses.asdict(res)
            d["llm_ms_uncached"] = round(d["llm_ms_uncached"], 1)
            rows[cid] = {"cell": cid, "fault": f["id"], "repo": f["repo"],
                         "category": f["category"],
                         "blast_radius": len(f["expected_failing_tests"]),
                         "held_out": " ".join(sandbox.held_out(f)),
                         "level": level, "model": model, "seed": seed, "rep": rep,
                         "provider": provider, "sandbox": backend,
                         "max_steps": a.max_steps, "config": config,
                         **{k: d[k] for k in RESULT_COLS}}
            write_rows(out, rows)
            print(f"    {cid:<58} {res.outcome:<16} steps={res.steps:<3}"
                  f"{'  ' + res.error_kind if res.error else ''}")

            streak = streak + 1 if res.error else 0
            if res.error_kind == "auth":
                stopped = "provider rejected the credentials (401/403)"
            elif streak >= a.max_errors:
                stopped = f"{streak} provider errors in a row"
            if stopped:
                break
    finally:
        loop.chat = real_chat

    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    ok = sum(1 for r in rows.values() if not r.get("error"))
    print(f"\n  {ok} completed rows ({len(rows) - ok} errored) -> {shown}")
    if stopped:
        print(f"  STOPPED: {stopped}. Fix that, then re-run the same command to resume.\n")
        return 2
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
