"""Where the agent's tools run and where the tests run. The agent never sees a test.

A Workspace is the agent's checkout: the faulty source and nothing else. The test
files are never copied into it, so no path the agent guesses can reach them. Tests
live in a separate folder and run in a separate place, the grader, which joins a
copy of the workspace source with the tests for each run and reports back.

    with open_workspace(fault, "docker") as ws:
        ws.call("read", {"path": "intervals/core.py"})   # the agent's file tools
        ws.run_tests(deselect=held_out(fault))             # what the agent's `test` sees
        ws.run_tests()                                     # grading: every test

Two backends, one behaviour:

  docker  live runs. The workspace is a container with no network that holds only
          the source; the agent's tools run inside it. Each test run is a fresh,
          throwaway grader container. Image: docker/Dockerfile.
  local   offline tests (no Docker, no network). The same source-only workspace in a
          temp folder; the grader in another temp folder.

THE HELD-OUT TEST

For a fault that breaks two or more tests, the last failing node id in sorted
order is held out: the agent's `test` never runs it, grading always does. The rule
looks only at test names, never at outcomes, so it cannot be tuned to a result. A
fix that passes the visible tests but not the held-out one is the measurable false
success the v1 harness could not produce.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from agent import tools_fs
from retrieval import query as Q
from tools.corpus import materialise

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "agentfault-sandbox:1"
BACKENDS = ("docker", "local")

# The flags of retrieval/query.py: run_suite, so test output reads the same as in
# the retrieval sweep and in harness v1.
PYTEST = ["-m", "pytest", "-q", "--no-header", "-rf", "--tb=long", "-p", "no:cacheprovider"]


def is_test_file(rel: Path) -> bool:
    return ("tests" in rel.parts or rel.name.startswith("test_")
            or rel.name == "conftest.py")


def test_files(repo: Path) -> List[Path]:
    return [p for p in sorted(repo.rglob("*.py"))
            if "__pycache__" not in p.parts and is_test_file(p.relative_to(repo))]


def held_out(fault: dict) -> Tuple[str, ...]:
    """The test the agent never runs: the last of two or more failing node ids."""
    failing = sorted(fault["expected_failing_tests"])
    return (failing[-1],) if len(failing) >= 2 else ()


def _split(fault: dict, td: Path) -> Tuple[Path, Path]:
    """Materialise the faulty checkout, then move every test file out of it.

    Returns (workspace, tests): the workspace holds source only, the tests folder
    holds test files only, at their repo-relative paths."""
    full = materialise(fault, td / "full")
    ws, tests = td / "workspace", td / "tests"
    for p in test_files(full):
        dest = tests / p.relative_to(full)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        p.unlink()
    shutil.rmtree(full / ".pytest_cache", ignore_errors=True)
    for d in sorted((d for d in full.rglob("*") if d.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()            # a tests/ folder emptied by the move
    full.rename(ws)
    if test_files(ws):
        raise RuntimeError("a test file is still in the workspace; refusing to run")
    return ws, tests


def _pytest_args(deselect: Sequence[str]) -> List[str]:
    return PYTEST + [a for d in deselect for a in ("--deselect", d)]


class Workspace:
    """The local backend, and the interface the docker backend keeps."""
    backend = "local"

    def __init__(self, fault: dict, root: Path, tests: Path):
        self.fault, self.root, self.tests = fault, root, tests

    def call(self, name: str, args: dict) -> dict:
        return tools_fs.TOOLS[name](self.root, args)

    def run_tests(self, deselect: Sequence[str] = ()) -> Tuple[bool, str]:
        """(green, pytest output with machine paths scrubbed) for the workspace
        source joined with the tests, minus `deselect`."""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "work"
            shutil.copytree(self.root, work, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache"))
            shutil.copytree(self.tests, work, dirs_exist_ok=True)
            p = subprocess.run([sys.executable, *_pytest_args(deselect)],
                               cwd=work, capture_output=True, text=True)
            return p.returncode == 0, Q.scrub_paths(p.stdout, work)

    def manifest(self) -> dict:
        return {"backend": self.backend, "image": None}

    def close(self) -> None:
        pass


class DockerWorkspace(Workspace):
    backend = "docker"

    def __init__(self, fault: dict, root: Path, tests: Path):
        super().__init__(fault, root, tests)
        ensure_image()
        self.cid = _docker(
            "run", "-d", "--rm", "--network", "none",
            "-v", f"{root.resolve()}:/repo",
            "-v", f"{ROOT / 'agent' / 'tools_fs.py'}:/opt/af/tools_fs.py:ro",
            IMAGE, "sleep", "infinity").strip()

    def call(self, name: str, args: dict) -> dict:
        p = subprocess.run(["docker", "exec", "-i", self.cid, "python",
                            "/opt/af/tools_fs.py", name, "/repo"],
                           input=json.dumps(args), capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"tool {name} failed in the workspace: {p.stderr[-300:]}")
        return json.loads(p.stdout)

    def run_tests(self, deselect: Sequence[str] = ()) -> Tuple[bool, str]:
        script = ("cp -r /src/. /work && cp -r /tests/. /work && cd /work && "
                  "exec python " + " ".join(f"'{a}'" for a in _pytest_args(deselect)))
        p = subprocess.run(["docker", "run", "--rm", "--network", "none",
                            "-v", f"{self.root.resolve()}:/src:ro",
                            "-v", f"{self.tests.resolve()}:/tests:ro",
                            IMAGE, "sh", "-c", script],
                           capture_output=True, text=True)
        return p.returncode == 0, Q.scrub_paths(p.stdout, Path("/work"))

    def manifest(self) -> dict:
        return {"backend": self.backend, "image": IMAGE, "image_id": image_id()}

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.cid], capture_output=True)


def _docker(*args: str) -> str:
    p = subprocess.run(["docker", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {p.stderr[-300:]}")
    return p.stdout


def ensure_image() -> None:
    """Build the sandbox image once; later calls find it and return."""
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode:
        _docker("build", "-q", "-t", IMAGE, str(ROOT / "docker"))


def image_id() -> str:
    return _docker("image", "inspect", "-f", "{{.Id}}", IMAGE).strip()


def docker_available() -> bool:
    return subprocess.run(["docker", "version"], capture_output=True).returncode == 0


@contextlib.contextmanager
def open_workspace(fault: dict, backend: str = "local") -> Iterator[Workspace]:
    if backend not in BACKENDS:
        raise ValueError(f"unknown sandbox {backend!r}: use one of {BACKENDS}")
    with tempfile.TemporaryDirectory() as td:
        root, tests = _split(fault, Path(td))
        ws = (DockerWorkspace if backend == "docker" else Workspace)(fault, root, tests)
        try:
            yield ws
        finally:
            ws.close()


def git_state() -> dict:
    """The commit a run used, and whether the checkout had uncommitted changes."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"git_sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def tests_digest(tests: Path) -> str:
    """One hash over the tests a run was graded with."""
    h = hashlib.sha256()
    for p in sorted(tests.rglob("*.py")):
        h.update(str(p.relative_to(tests)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()
