"""Offline tests for the things Phase D's numbers depend on.

Each test pins one way the sweep could report a wrong number without failing:

  seeds replaying one cached response   -> zero variance that is not real
  a claim checked only on visible tests -> a "fix" that fixed nothing
  no way to stop except `done`          -> giving up counted as running out of steps
  a dead provider                       -> a grid full of error rows
  a killed sweep                        -> cells paid for twice, or lost

The model is scripted (agent/fake.py), so nothing here needs a key or the network.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.loop as loop                                    # noqa: E402
import runner.agent_sweep as sweep                           # noqa: E402
from agent.fake import BEHAVIOURS, failing_model, scripted_model  # noqa: E402
from retrieval import providers                              # noqa: E402
from agent.sandbox import open_workspace                      # noqa: E402
from runner.report import agent_report                       # noqa: E402
from tools.corpus import REPOS, load_faults                  # noqa: E402

FAULT_ID = "intervals-001"


def _fault():
    return [f for f in load_faults() if f["id"] == FAULT_ID][0]


def _episode(tmp: Path, model_fn, trace: bool = False, fault=None, **kw):
    """Run one episode against a scripted model; return (result, trace dir)."""
    fault = fault or _fault()
    out = tmp / "trace" if trace else None
    real = loop.chat
    loop.chat = model_fn
    try:
        with open_workspace(fault, timeout=kw.pop("timeout", 60)) as ws:
            res = loop.Episode(ws, fault, level="L3", seed=0, trace_dir=out, **kw).run()
    finally:
        loop.chat = real
    return res, out


# ------------------------------------------------------------------ seeds

def test_seed_is_part_of_the_cache_key():
    msgs = [{"role": "user", "content": "x"}]
    tools = loop.TOOL_SCHEMAS
    key = lambda **kw: providers._key(providers.chat_cache_payload(  # noqa: E731
        "groq", "m", 0.7, 100, msgs, tools, **kw))

    assert key(seed=1) != key(seed=2), "two seeds must never share a cached response"
    assert key(seed=1) == key(seed=1), "and one seed must replay itself"

    # A changed tool schema is a different prompt, so it must miss the cache too.
    other = providers._key(providers.chat_cache_payload(
        "groq", "m", 0.7, 100, msgs, tools[:-1], 1))
    assert other != key(seed=1)

    # Calls made before seeds existed keep their old key, so those entries replay.
    legacy = providers._key(("groq", "m", 0.7, 100, msgs, True))
    assert key(seed=None) == legacy


def test_episode_passes_its_seed_to_the_model():
    seen = []
    inner = scripted_model(_fault(), "give_up")

    def spy(messages, **kw):
        seen.append(kw.get("seed"))
        return inner(messages, **kw)

    fault = _fault()
    real = loop.chat
    loop.chat = spy
    try:
        with open_workspace(fault) as ws:
            loop.Episode(ws, fault, level="L0", seed=7).run()
    finally:
        loop.chat = real
    assert seen and set(seen) == {7}


# --------------------------------------------------------------- outcomes

def test_each_scripted_behaviour_gets_its_outcome():
    want = {"fix": "fixed", "give_up": "gave_up", "wrong_done": "silent_failure",
            "stall": "exhausted", "peek_tests": "silent_failure"}
    assert set(want) == set(BEHAVIOURS)
    for behaviour, outcome in want.items():
        with tempfile.TemporaryDirectory() as td:
            res, _ = _episode(Path(td), scripted_model(_fault(), behaviour))
        assert res.outcome == outcome, f"{behaviour}: got {res.outcome}"
        assert res.outcome in loop.OUTCOMES


def test_real_fix_is_clean():
    with tempfile.TemporaryDirectory() as td:
        res, _ = _episode(Path(td), scripted_model(_fault(), "fix"))
    assert (res.fix_rate, res.visible_green, res.policy_hits, res.test_probes) == (1, 1, 0, 0)
    assert res.localised_fn == 1 and res.saw_gold_fn == 1
    assert res.first_fixed_step == 2, "read at step 1, the fixing edit at step 2"
    assert res.stop_reason == "done"
    assert res.llm_calls == 4 and res.llm_calls_cached == 0 and res.llm_ms_uncached > 0


def test_the_step_budget_is_the_cars_and_stall_uses_all_of_it():
    with tempfile.TemporaryDirectory() as td:
        res, _ = _episode(Path(td), scripted_model(_fault(), "stall"),
                          car=loop.CarSpec(max_steps=7))
    assert (res.steps, res.stop_reason, res.first_fixed_step) == (7, "steps", 0)
    assert loop.CarSpec(max_steps=7).digest() != loop.CarSpec().digest()


def test_a_full_context_ends_the_episode_cleanly():
    with tempfile.TemporaryDirectory() as td:
        # The scripted model reports 100 prompt tokens per call.
        res, _ = _episode(Path(td), scripted_model(_fault(), "stall"),
                          car=loop.CarSpec(context_tokens=100, context_frac=0.5))
    assert (res.outcome, res.stop_reason, res.steps, res.error) == ("exhausted", "context", 1, "")


def test_the_test_tool_reports_a_pass_at_every_level():
    fault = _fault()
    with open_workspace(fault) as ws:
        for lv in ("L0", "L1", "L2", "L3"):
            assert loop.Episode(ws, fault, level=lv, seed=0).t_test() != "all tests passed"
        shutil.copy(REPOS / fault["repo"] / fault["file"], ws.root / fault["file"])
        for lv in ("L0", "L1", "L2", "L3"):
            assert loop.Episode(ws, fault, level=lv, seed=0).t_test() == "all tests passed"


def test_give_up_is_not_a_silent_failure():
    with tempfile.TemporaryDirectory() as td:
        res, _ = _episode(Path(td), scripted_model(_fault(), "give_up"))
    assert (res.gave_up, res.claimed_done, res.silent_failure) == (1, 0, 0)


def test_classify_outcome_covers_the_cases_no_script_reaches():
    c = loop.classify_outcome
    base = dict(error=False, visible_green=True, all_green=True, localised_fn=True,
                claimed_done=True, gave_up=False)
    assert c(**base) == "fixed"
    assert c(**{**base, "localised_fn": False}) == "fixed_elsewhere"
    # Every test decides a fix; the agent's own tests decide only what it claimed.
    assert c(**{**base, "all_green": False}) == "false_success"
    assert c(**{**base, "all_green": False, "claimed_done": False}) == "visible_only"
    assert c(**{**base, "all_green": False, "visible_green": False}) == "silent_failure"
    # A provider error outranks everything: a half-run episode is not a result.
    assert c(**{**base, "error": True}) == "error"


# ---------------------------------------------------------------- provider

def test_http_status_classification():
    k = providers.classify_status
    assert k(429) == "rate_limit"
    assert k(401) == "auth" and k(403) == "auth"
    assert k(500) == "server" and k(503) == "server"
    assert k(0) == "transport"
    assert k(400) == "client"


def test_groq_key_is_not_sent_to_a_custom_endpoint(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("AGENTFAULT_CHAT_URL", "http://localhost:8000/v1/chat/completions")
    monkeypatch.delenv("AGENTFAULT_CHAT_KEY_ENV", raising=False)
    url, key, provider = providers.chat_endpoint()
    assert key is None and provider == "localhost:8000"

    monkeypatch.delenv("AGENTFAULT_CHAT_URL")
    assert providers.chat_endpoint() == (providers.GROQ_URL, "groq-secret", "groq")


def test_a_pinned_upstream_is_sent_and_recorded(monkeypatch):
    monkeypatch.setenv("AGENTFAULT_CHAT_URL", "https://openrouter.ai/api/v1/chat/completions")
    monkeypatch.setenv("AGENTFAULT_CHAT_UPSTREAM", "groq")
    monkeypatch.delenv("AGENTFAULT_CHAT_KEY_ENV", raising=False)
    monkeypatch.setattr(providers, "_RATE", {})
    sent = {}

    def fake_curl(url, key, body):
        sent.update(body)
        return 200, json.dumps({"id": "r", "usage": {"total_tokens": 1}})

    monkeypatch.setattr(providers, "_curl_chat", fake_curl)
    providers.chat([{"role": "user", "content": "x"}], seed=0, cache=False)
    assert sent["provider"] == {"order": ["groq"], "allow_fallbacks": False,
                                "require_parameters": True}
    assert providers.chat_endpoint()[2] == "openrouter.ai/groq", \
        "the label, and so the cache key and CSV, must name the host"


def test_provider_error_is_recorded_with_its_kind():
    with tempfile.TemporaryDirectory() as td:
        res, _ = _episode(Path(td), failing_model("auth", 403))
    assert res.outcome == "error" and res.error_kind == "auth"


# ------------------------------------------------------------------- sweep

def _sweep(tmp: Path, *extra):
    return sweep.main(["--fake", "--faults", FAULT_ID, "--levels", "L0", "L3",
                       "-o", str(tmp / "raw.csv"), "--traces", str(tmp / "traces"), *extra])


def _rows(tmp: Path):
    return list(csv.DictReader((tmp / "raw.csv").open()))


def test_a_provider_is_chosen_by_name_and_never_mixed_in_one_csv(monkeypatch):
    for k in ("AGENTFAULT_CHAT_URL", "AGENTFAULT_CHAT_KEY_ENV", "AGENTFAULT_CHAT_UPSTREAM"):
        monkeypatch.delenv(k, raising=False)
    providers.configure("openrouter:akashml/fp4")
    assert providers.chat_endpoint()[2] == "openrouter.ai/akashml/fp4"
    providers.configure("groq")
    assert providers.chat_endpoint()[2] == "groq", "a later spec clears the earlier one"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        assert _sweep(tmp) == 0
        rows = _rows(tmp)
        assert {r["sandbox"] for r in rows} == {"local"}, "a scripted model runs locally"
        for r in rows:
            r["provider"], r["config"] = "openrouter.ai/akashml/fp4", "another"
        sweep.write_rows(tmp / "raw.csv", {r["cell"]: r for r in rows})
        try:
            _sweep(tmp)
            raise AssertionError("resuming under another provider must stop")
        except SystemExit as e:
            assert "akashml" in str(e)


def test_a_csv_never_mixes_step_budgets():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        assert _sweep(tmp, "--max-steps", "12") == 0
        assert {r["max_steps"] for r in _rows(tmp)} == {"12"}
        with pytest.raises(SystemExit):
            _sweep(tmp, "--max-steps", "50")


def test_sweep_resumes_and_seeds_differ():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        assert _sweep(tmp, "--seeds", "0", "1") == 0
        rows = _rows(tmp)
        assert len(rows) == 4
        by_seed = {r["seed"]: r["outcome"] for r in rows if r["level"] == "L3"}
        assert by_seed == {"0": "fixed", "1": "gave_up"}

        # Re-running must skip all four: the trace files are not rewritten.
        stamp = {p: p.stat().st_mtime_ns for p in (tmp / "traces").rglob("run.json")}
        assert len(stamp) == 4
        assert _sweep(tmp, "--seeds", "0", "1") == 0
        assert stamp == {p: p.stat().st_mtime_ns for p in stamp}

        # Widening the grid runs only the new cells.
        assert _sweep(tmp, "--seeds", "0", "1", "2") == 0
        assert len(_rows(tmp)) == 6

        agent_report(tmp / "raw.csv", tmp / "AGENT_REPORT.md")
        assert "pass@1" in (tmp / "AGENT_REPORT.md").read_text()


def test_repeats_are_separate_rows_and_feed_the_noise_floor():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        assert _sweep(tmp, "--seeds", "0", "--repeats", "2") == 0
        rows = _rows(tmp)
        assert sorted((r["level"], r["rep"]) for r in rows) == [
            ("L0", "0"), ("L0", "1"), ("L3", "0"), ("L3", "1")]
        assert len({r["cell"] for r in rows}) == 4

        md = tmp / "AGENT_REPORT.md"
        agent_report(tmp / "raw.csv", md)
        text = md.read_text()
        assert "Noise floor" in text
        # The headline counts rep 0 only: 2 episodes, not 4.
        assert "2 episodes" in text
        # The scripted model is deterministic, so no repeat disagrees...
        assert _noise_row(text, "L3")[-2:] == ["0", "0"]

        # ...and a repeat that DOES disagree is counted.
        for r in rows:
            if r["level"] == "L3" and r["rep"] == "1":
                r["outcome"], r["fix_rate"] = "gave_up", "0"
        _write(tmp / "raw.csv", rows)
        agent_report(tmp / "raw.csv", md)
        assert _noise_row(md.read_text(), "L3")[-2:] == ["1", "1"]


def _noise_row(text: str, level: str):
    section = text.split("Noise floor")[1]
    line = [ln for ln in section.splitlines() if f"**{level}**" in ln][0]
    return [c.strip() for c in line.strip("|").split("|")]


def _write(path: Path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sweep.COLS)
        w.writeheader()
        w.writerows(rows)


def test_a_cell_with_too_many_dead_runs_is_marked():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        assert _sweep(tmp, "--seeds", "0", "1", "2", "3", "4") == 0
        rows = _rows(tmp)
        # One of five L0 episodes errored: 20%, over the 10% bar. L3 is untouched.
        victim = [r for r in rows if r["level"] == "L0"][0]
        victim["error"], victim["error_kind"], victim["outcome"] = "boom", "server", "error"
        _write(tmp / "raw.csv", rows)

        md = tmp / "AGENT_REPORT.md"
        agent_report(tmp / "raw.csv", md)
        table = md.read_text().split("## 1.")[1].split("## 2.")[0]
        l0 = [ln for ln in table.splitlines() if "**L0**" in ln][0]
        l3 = [ln for ln in table.splitlines() if "**L3**" in ln][0]
        assert "†" in l0 and "†" not in l3
        assert "1 errored episodes are excluded" in md.read_text()


def test_paired_interval_sees_a_difference_unpaired_intervals_hide():
    from retrieval.metrics import bootstrap_ci, paired_bootstrap_ci
    # Faults vary a lot in difficulty, and B beats A by 0.1 on every single one.
    a = [0.0, 0.2, 0.4, 0.6, 0.8]
    b = [x + 0.1 for x in a]
    lo, hi = paired_bootstrap_ci(b, a)
    assert abs(lo - 0.1) < 1e-9 and abs(hi - 0.1) < 1e-9, "every fault agrees: no spread"
    # Unpaired, the two columns overlap almost entirely.
    assert bootstrap_ci(a)[1] > bootstrap_ci(b)[0]


def test_cache_off_neither_reads_nor_writes(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(providers, "CACHE", Path(td))
        monkeypatch.setattr(providers, "_RATE", {})          # no pacing sleeps
        monkeypatch.delenv("AGENTFAULT_CHAT_URL", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
        answer = {"n": 0}

        def fake_curl(url, key, body):
            answer["n"] += 1
            return 200, json.dumps({"id": f"r{answer['n']}", "usage": {"total_tokens": 1}})

        monkeypatch.setattr(providers, "_curl_chat", fake_curl)
        msgs = [{"role": "user", "content": "x"}]

        assert providers.chat(msgs, seed=0, cache=False)["id"] == "r1"
        assert not list(Path(td).iterdir()), "cache off must not write"

        assert providers.chat(msgs, seed=0)["id"] == "r2"            # stored
        assert providers.chat(msgs, seed=0)["id"] == "r2"            # replayed
        assert providers.LAST_CALL["cached"] is True
        assert providers.chat(msgs, seed=0, cache=False)["id"] == "r3", \
            "cache off must not read the stored response"
        assert providers.LAST_CALL["cached"] is False


def test_cache_only_mode_refuses_live_calls(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(providers, "CACHE", Path(td))
        monkeypatch.setenv("AGENTFAULT_CACHE_ONLY", "1")
        called = []
        try:
            providers._cached("cohere_embed", ("anything",), lambda: called.append(1))
        except providers.CacheMiss:
            pass
        else:
            raise AssertionError("a cache miss must raise in cache-only mode")
        assert not called, "and the live call must never run"


def test_sweep_stops_on_an_error_streak_then_recovers():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Eight cells, every call failing: it must stop after two, not run all eight.
        rc = _sweep(tmp, "--seeds", "0", "1", "2", "3", "--fake-fail", "--max-errors", "2")
        assert rc == 2
        assert len(_rows(tmp)) == 2 and all(r["outcome"] == "error" for r in _rows(tmp))

        # Provider back: the same command re-runs the errored cells and finishes.
        assert _sweep(tmp, "--seeds", "0", "1", "2", "3") == 0
        rows = _rows(tmp)
        assert len(rows) == 8 and not any(r["error"] for r in rows)
