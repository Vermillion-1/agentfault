"""The docker backend gives the same episode as the local one (skipped without a daemon).

  backends that differ    -> offline tests that prove nothing about live runs
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.loop as loop                                  # noqa: E402
from agent.fake import scripted_model                      # noqa: E402
from agent.sandbox import docker_available, open_workspace  # noqa: E402
from tools.corpus import FAULTS, load_faults               # noqa: E402
from tools.validate_grader import validate                 # noqa: E402

pytestmark = pytest.mark.skipif(not docker_available(), reason="no docker daemon")
HAND = [f for f in load_faults() if f["_path"].parent == FAULTS]


def _first_prompt_and_grade(fault, backend):
    sent = []
    inner = scripted_model(fault, "give_up")

    def spy(messages, **kw):
        sent.append([dict(m) for m in messages])
        return inner(messages, **kw)

    real = loop.chat
    loop.chat = spy
    try:
        with open_workspace(fault, backend) as ws:
            loop.Episode(ws, fault, level="L3", seed=0).run()
            grade = ws.grade().results
    finally:
        loop.chat = real
    return sent[0], grade


def test_hand_written_faults_grade_against_the_truth_in_docker():
    assert [r for r in (validate(f, "docker") for f in HAND) if r["problems"]] == []


@pytest.mark.parametrize("fault", HAND, ids=[f["id"] for f in HAND])
def test_local_and_docker_send_the_same_prompt_and_grade_the_same(fault):
    assert _first_prompt_and_grade(fault, "local") == _first_prompt_and_grade(fault, "docker")
