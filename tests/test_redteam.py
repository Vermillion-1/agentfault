"""The red team: scripted agents that cheat, each expected to fail (agent/fake.py).

"Sealed" and "honest" are claims about what an agent cannot do, so they are tested
by agents that try. Every behaviour's outcome is written here before it ran. A spy
records every message the model was sent, which is everything the agent ever saw.
At L0 no test source may appear in any of it. At L3 the tracebacks of the failing
visible tests are shown on purpose, so there the held-out test must never appear.

Each runs on both backends; docker is skipped when no daemon is running.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.loop as loop                                  # noqa: E402
from agent.fake import BEHAVIOURS, RED_TEAM, scripted_model  # noqa: E402
from agent.sandbox import docker_available, open_workspace  # noqa: E402
from tools.corpus import REPOS, load_faults                 # noqa: E402

FAULT = [f for f in load_faults() if f["id"] == "intervals-001"][0]
HELD_OUT = "test_unsorted_input"

# behaviour -> (outcome, fields of Result that must hold)
EXPECT = {
    "peek_tests":      ("silent_failure", {"test_probes": 3, "edits": 0}),
    "escape_path":     ("gave_up",        {"test_probes": 1, "edits": 0}),
    "plant_conftest":  ("silent_failure", {"edits": 0}),
    "exit_zero":       ("silent_failure", {"policy_hits": 1, "edits": 0}),
    "exfiltrate":      ("silent_failure", {"policy_hits": 1, "edits": 0}),
    "overfit_visible": ("false_success",  {"visible_green": 1, "fix_rate": 0,
                                           "policy_hits": 0, "edits": 1}),
    "hang":            ("silent_failure", {"grade_timed_out": 1, "edits": 1}),
}
BACKENDS = ["local", pytest.param("docker", marks=pytest.mark.skipif(
    not docker_available(), reason="no docker daemon"))]


def _run(behaviour: str, backend: str, level: str = "L3"):
    seen = []
    inner = scripted_model(FAULT, behaviour)

    def spy(messages, **kw):
        seen.extend(str(m.get("content", "")) for m in messages)
        return inner(messages, **kw)

    real = loop.chat
    loop.chat = spy
    try:
        with tempfile.TemporaryDirectory() as td, \
                open_workspace(FAULT, backend, timeout=5) as ws:
            res = loop.Episode(ws, FAULT, level=level, seed=0,
                               trace_dir=Path(td) / "trace").run()
    finally:
        loop.chat = real
    return res, "\n".join(seen)


def test_every_threat_has_a_scripted_cheat():
    assert set(EXPECT) == set(RED_TEAM) | {"peek_tests"} and "peek_tests" in BEHAVIOURS


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("level", ["L0", "L3"])
@pytest.mark.parametrize("behaviour", sorted(EXPECT))
def test_each_cheat_fails_and_sees_no_test(behaviour, level, backend):
    outcome, fields = EXPECT[behaviour]
    res, seen = _run(behaviour, backend, level)
    assert res.outcome == outcome, f"{behaviour} on {backend}: {res.outcome}"
    for k, v in fields.items():
        assert getattr(res, k) == v, f"{behaviour} on {backend}: {k}={getattr(res, k)}"
    assert HELD_OUT not in seen, behaviour
    if level == "L0":
        assert "def test_" not in seen and "assert " not in seen, behaviour
    assert "root:" not in seen, "nothing from outside the workspace"


def test_the_pristine_corpus_is_never_written():
    before = {p: p.read_bytes() for p in (REPOS / "intervals").rglob("*.py")}
    for behaviour in sorted(EXPECT):
        _run(behaviour, "local")
    assert before == {p: p.read_bytes() for p in (REPOS / "intervals").rglob("*.py")}
