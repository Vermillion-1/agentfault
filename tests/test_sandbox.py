"""The sealed workspace and the grader (agent/sandbox.py), attacked on purpose.

Each cheat is tried twice where it can be: through the edit gate, and with the
gate switched off, so each layer is shown to hold on its own.

  tests reachable from the workspace     -> the agent reads the answer
  held-out test on disk during `test`     -> its body printed back at L3
  "green" from pytest's return code       -> os._exit(0) scores as a fix
  a missing test passes                   -> breaking collection scores as a fix
  a hung test run                         -> a sweep that never ends
  pytest config editable                  -> the agent reconfigures the grader
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent.sandbox as sb                       # noqa: E402
from agent.sandbox import junit_key, open_workspace  # noqa: E402
from retrieval import query as Q                 # noqa: E402
from tools.corpus import FAULTS, load_faults     # noqa: E402
from tools.validate_grader import validate       # noqa: E402

DOC = '    """Coalesce overlapping and adjacent intervals into a minimal sorted list."""\n'


def _fault(fid="intervals-001"):
    return [f for f in load_faults() if f["id"] == fid][0]


def _plant(ws, line: str) -> None:
    """Write code into `merge` directly, as if the edit gate had been bypassed."""
    p = ws.root / "intervals" / "core.py"
    p.write_text(p.read_text().replace(DOC, DOC + line))


@pytest.fixture
def no_gate(monkeypatch):
    monkeypatch.setattr(sb, "violations", lambda before, after: [])


def test_workspace_holds_no_test_and_visible_lacks_the_held_out_test():
    f = _fault()
    with open_workspace(f) as ws:
        assert sb.test_files(ws.root) == []
        assert not any(p.name in (".git", ".pytest_cache") for p in ws.root.rglob("*"))
        name = ws.held_out[0].split("::")[-1]
        assert f"def {name}" in (ws.all / "tests" / "test_core.py").read_text()
        assert f"def {name}" not in (ws.visible / "tests" / "test_core.py").read_text()
        assert junit_key(ws.held_out[0]) not in {junit_key(n) for n in ws.expected["visible"]}


def test_hand_written_faults_grade_against_the_truth():
    hand = [f for f in load_faults() if f["_path"].parent == FAULTS]
    bad = [r for r in (validate(f) for f in hand) if r["problems"]]
    assert bad == []


@pytest.mark.skipif(not os.environ.get("AGENTFAULT_FULL"), reason="set AGENTFAULT_FULL=1")
def test_every_fault_grades_against_the_truth():
    bad = [r for r in (validate(f) for f in load_faults()) if r["problems"]]
    assert bad == []


def test_exit_zero_is_refused_by_the_gate_at_grading():
    with open_workspace(_fault()) as ws:
        _plant(ws, "    import os; os._exit(0)\n")
        run = ws.grade()
        assert not run.green and run.refused


def test_exit_zero_is_red_even_without_the_gate(no_gate):
    with open_workspace(_fault()) as ws:
        _plant(ws, "    import os; os._exit(0)\n")
        run = ws.grade()
        assert not run.green and not run.refused and run.results == {}


def test_held_out_body_never_reaches_the_agent_without_the_gate(no_gate):
    with open_workspace(_fault()) as ws:
        name = ws.held_out[0].split("::")[-1]
        _plant(ws, "    import glob; print(''.join(open(p).read() for p in glob.glob('tests/*.py')))\n")
        run = ws.check()
        assert "def test_" in run.output          # the leak itself works...
        for level in Q.LEVELS:                     # ...but the held-out test is not there
            assert name not in Q.build(run.output, level)
            assert "deselected" not in Q.build(run.output, level)


def test_a_missing_test_is_a_failure():
    f = _fault()
    with open_workspace(f) as ws:
        (ws.root / f["file"]).write_text((sb.ROOT / "corpus" / "repos" / f["repo"] / f["file"]).read_text())
        assert ws.grade().green
        ws.expected["all"] = ws.expected["all"] + ("tests/test_core.py::TestGhost::test_ghost",)
        assert not ws.grade().green


def test_a_hung_test_run_times_out():
    # An infinite loop is ordinary logic, so the gate allows it; the timeout stops it.
    with open_workspace(_fault(), timeout=3) as ws:
        out = ws.call("edit", {"path": "intervals/core.py", "old": DOC,
                               "new": DOC + "    while True:\n        pass\n"})
        assert "edited" in out
        run = ws.check()
        assert run.timed_out and not run.green


def test_pytest_config_cannot_be_edited_or_smuggled():
    with open_workspace(_fault()) as ws:
        out = ws.call("edit", {"path": "pyproject.toml", "old": "[", "new": "["})
        assert out["text"].startswith("no such file")
        cfg = ws.root / "pyproject.toml"
        cfg.write_text(cfg.read_text() + '\n[tool.pytest.ini_options]\naddopts = "-k nothing"\n')
        assert set(ws.grade().failing()) == {junit_key(n) for n in _fault()["expected_failing_tests"]}


@pytest.mark.parametrize("path", ["tests/test_core.py", "../tests/test_core.py",
                                  "/etc/passwd", "intervals/../../all/tests/test_core.py"])
def test_guessed_paths_are_refused(path):
    with open_workspace(_fault()) as ws:
        for tool in ("read", "edit"):
            out = ws.call(tool, {"path": path, "old": "a", "new": "b"})
            assert out["text"].startswith("no such file"), (tool, path)


def test_a_planted_symlink_is_refused():
    with open_workspace(_fault()) as ws:
        (ws.root / "intervals" / "link.py").symlink_to(ws.all / "tests" / "test_core.py")
        assert ws.call("read", {"path": "intervals/link.py"})["text"].startswith("no such file")


def test_new_files_cannot_be_created():
    with open_workspace(_fault()) as ws:
        for path in ("conftest.py", "intervals/conftest.py", "sitecustomize.py"):
            out = ws.call("edit", {"path": path, "old": "", "new": "x = 1"})
            assert out["text"].startswith("no such file")
            assert not (ws.root / path).exists()
