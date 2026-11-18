"""Where the agent's tools run and where the tests run. The agent never sees a test.

A Workspace is the agent's checkout: the faulty source and nothing else. The test
files are never copied into it, so no path the agent guesses can reach them. Tests
live in separate folders and run in a separate place, the grader, which joins a
copy of the workspace source with the tests for each run and reports back.

    with open_workspace(fault, "docker") as ws:
        ws.call("read", {"path": "intervals/core.py"})   # the agent's file tools
        ws.check()                                       # what the agent's `test` runs
        ws.grade()                                       # scoring: every test

Two backends, one behaviour:

  docker  live runs. The workspace is a container with no network that holds only
          the source; the agent's tools run inside it. Each test run is a fresh,
          throwaway, locked-down grader container. Image: docker/Dockerfile.
  local   offline tests (no Docker, no network). The same layout in temp folders.
          It does not contain the code it runs, so it is for trusted, scripted
          agents only.

THE HELD-OUT TEST

For a fault that breaks two or more tests, the last failing node id in sorted
order is held out. The rule looks only at test names, never at outcomes, so it
cannot be tuned to a result. The agent's `check()` runs a copy of the tests with
that test function DELETED, not deselected: a deselected test's file is still on
disk, and source code the agent edits could print it into the L3 output
[spikes/2026-11-17-grader-holes]. Deleting it agrees with --deselect on all 166
faults that have a held-out test [same spike].

WHAT "GREEN" MEANS

Never pytest's return code: `os._exit(0)` in the source makes that 0 with nothing
run [same spike]. Green means pytest wrote a JUnit report, the report lists exactly
the tests collected from the pristine suite, and every one passed. A missing test
is a failure, as in SWE-bench. Only files that were in the workspace at the start
are copied to the grader, only `.py` files are taken from the agent's side (a
pyproject.toml configures pytest), and each is checked against the edit gate again
(agent/policy.py) before any of it runs.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from agent import tools_fs
from agent.policy import violations
from retrieval import query as Q
from tools.corpus import materialise

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "agentfault-sandbox:1"
BACKENDS = ("docker", "local")
GRADE_TIMEOUT = 60.0

# The flags of retrieval/query.py: run_suite, so test output reads the same as in
# the retrieval sweep and in harness v1.
PYTEST = ["-m", "pytest", "-q", "--no-header", "-rf", "--tb=long", "-p", "no:cacheprovider"]

# The grader container: no network, nothing writable but scratch space, no root,
# no capabilities, bounded processes and memory.
LOCKDOWN = ["--network", "none", "--read-only", "--user", "65534:65534",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "256", "--memory", "1g",
            "--tmpfs", "/work:rw,exec,mode=1777", "--tmpfs", "/tmp:rw,mode=1777",
            "-e", "PYTHONDONTWRITEBYTECODE=1"]


@dataclass(frozen=True)
class Run:
    """One test run. `results` maps node id -> passed, from the JUnit report."""
    green: bool
    results: Dict[str, bool] = field(default_factory=dict)
    output: str = ""
    timed_out: bool = False
    refused: Tuple[str, ...] = ()       # edit-gate violations found at grading

    def failing(self) -> List[str]:
        return sorted(k for k, ok in self.results.items() if not ok)


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


def strip_test(tests: Path, node: str) -> None:
    """Delete one test function, decorators included, from its file in `tests`."""
    rel, *names = node.split("::")
    path = tests / rel
    src = path.read_text()
    body = ast.parse(src).body
    for cls in names[:-1]:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == names[-1].split("[")[0])
    lo = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno) - 1
    lines = src.splitlines(keepends=True)
    del lines[lo:fn.end_lineno]
    path.write_text("".join(lines))


def _split(fault: dict, td: Path) -> Tuple[Path, Path, Path, Path]:
    """Lay out one episode: (workspace, start, visible, all).

    workspace  the faulty source, no test file; the agent's tools act on it
    start      a frozen copy of the workspace, what the edit gate compares against
    visible    every test minus the held-out function; the agent's `test` runs it
    all        every test, pristine; grading runs it"""
    full = materialise(fault, td / "full")
    ws, start, vis, alltests = td / "workspace", td / "start", td / "visible", td / "all"
    for p in test_files(full):
        dest = alltests / p.relative_to(full)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        p.unlink()
    shutil.rmtree(full / ".pytest_cache", ignore_errors=True)
    for d in sorted((d for d in full.rglob("*") if d.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()            # a tests/ folder emptied by the move
    full.rename(ws)
    canary = fault.get("canary")
    if canary:
        # An impossible fault (tools/canaries.py): one test is added beside the one
        # it contradicts, in every test set, so the agent's runs include it too.
        path = alltests / canary["test_file"]
        lines = path.read_text().splitlines(keepends=True)
        n = canary["insert_after_line"]
        path.write_text("".join(lines[:n]) + canary["text"] + "".join(lines[n:]))
    if test_files(ws):
        raise RuntimeError("a test file is still in the workspace; refusing to run")
    shutil.copytree(ws, start)
    shutil.copytree(alltests, vis)
    for node in held_out(fault):
        strip_test(vis, node)
    return ws, start, vis, alltests


def _sources(root: Path) -> Tuple[str, ...]:
    return tuple(str(p.relative_to(root)) for p in sorted(root.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts)


def junit_key(node: str) -> str:
    """A pytest node id in the form JUnit reports it: `classname::name`."""
    path, *parts = node.split("::")
    module = path[:-3].replace("/", ".") if path.endswith(".py") else path
    return ".".join([module] + parts[:-1]) + "::" + parts[-1]


def _read_junit(xml: Path) -> Optional[Dict[str, bool]]:
    if not xml.is_file():
        return None
    try:
        cases = ET.parse(xml).iter("testcase")
        return {f'{c.get("classname")}::{c.get("name")}':
                not any(x.tag in ("failure", "error", "skipped") for x in c)
                for c in cases}
    except ET.ParseError:
        return None


def collect(source: Path, tests: Path) -> Tuple[str, ...]:
    """The node ids pytest collects from the pristine source and `tests`, run on the
    host: both are trusted, and the set must not depend on the backend."""
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "work"
        shutil.copytree(source, work)
        shutil.copytree(tests, work, dirs_exist_ok=True)
        p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                            "-p", "no:cacheprovider"], cwd=work, capture_output=True,
                           text=True, timeout=GRADE_TIMEOUT)
    nodes = tuple(sorted(ln.strip() for ln in p.stdout.splitlines() if "::" in ln))
    if not nodes:
        raise RuntimeError(f"collected no tests: {p.stdout[-300:]}")
    return nodes


_JUNIT_LINE = re.compile(r"^-+ generated xml file: .*$\n?", re.M)


class Workspace:
    """The local backend, and the interface the docker backend keeps."""
    backend = "local"

    def __init__(self, fault: dict, td: Path, timeout: float = GRADE_TIMEOUT):
        self.fault, self.timeout = fault, timeout
        self.root, self.start, self.visible, self.all = _split(fault, td)
        self.files = _sources(self.start)
        self.held_out = held_out(fault)
        self.expected = {"visible": collect(self.start, self.visible),
                         "all": collect(self.start, self.all)}

    def call(self, name: str, args: dict) -> dict:
        return tools_fs.TOOLS[name](self.root, args)

    def check(self) -> Run:
        """The agent's `test`: the visible tests, held-out function deleted."""
        return self._run(self.visible, "visible")

    def grade(self) -> Run:
        """Scoring: every pristine test."""
        return self._run(self.all, "all")

    def _run(self, tests: Path, which: str) -> Run:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            refused = []
            for rel in self.files:
                # Only source code comes from the agent's side. Anything else, such
                # as pyproject.toml, configures pytest and comes from the start copy.
                was = self.start / rel
                now = self.root / rel if rel.endswith(".py") else was
                if not now.is_file():
                    continue      # the tools cannot delete a file; nothing to copy
                dest = src / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(now, dest)
                if rel.endswith(".py"):
                    refused += violations(was.read_text(), now.read_text())
            if refused:
                return Run(False, refused=tuple(refused))
            out = Path(td) / "out"
            out.mkdir(mode=0o777)
            out.chmod(0o777)
            stdout, timed_out = self._pytest(src, tests, out)
            got = _read_junit(out / "r.xml")
        stdout = _JUNIT_LINE.sub("", stdout)
        if timed_out or got is None:
            return Run(False, got or {}, stdout, timed_out)
        expected = {junit_key(n) for n in self.expected[which]}
        green = set(got) == expected and all(got.values())
        return Run(green, got, stdout)

    def _pytest(self, src: Path, tests: Path, out: Path) -> Tuple[str, bool]:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "work"
            shutil.copytree(src, work)
            shutil.copytree(tests, work, dirs_exist_ok=True)
            try:
                p = subprocess.run([sys.executable, *PYTEST, f"--junitxml={out / 'r.xml'}"],
                                   cwd=work, capture_output=True, text=True,
                                   timeout=self.timeout)
            except subprocess.TimeoutExpired:
                return "", True
            return Q.scrub_paths(p.stdout, work), False

    def manifest(self) -> dict:
        return {"backend": self.backend, "image": None, "image_id": None,
                "held_out": list(self.held_out),
                "tests_visible_sha256": tests_digest(self.visible),
                "tests_all_sha256": tests_digest(self.all),
                "grade_timeout_s": self.timeout}

    def close(self) -> None:
        pass


class DockerWorkspace(Workspace):
    backend = "docker"

    def __init__(self, fault: dict, td: Path, timeout: float = GRADE_TIMEOUT):
        super().__init__(fault, td, timeout)
        ensure_image()
        # The tools are ours and trusted, but the container gets no more than they
        # need: no network, no capabilities, a read-only root.
        self.cid = _docker(
            "run", "-d", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64",
            "-v", f"{self.root.resolve()}:/repo",
            "-v", f"{ROOT / 'agent' / 'tools_fs.py'}:/opt/af/tools_fs.py:ro",
            "-v", f"{ROOT / 'agent' / 'policy.py'}:/opt/af/policy.py:ro",
            IMAGE, "sleep", "infinity").strip()

    def call(self, name: str, args: dict) -> dict:
        p = subprocess.run(["docker", "exec", "-i", self.cid, "python",
                            "/opt/af/tools_fs.py", name, "/repo"],
                           input=json.dumps(args), capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"tool {name} failed in the workspace: {p.stderr[-300:]}")
        return json.loads(p.stdout)

    def _pytest(self, src: Path, tests: Path, out: Path) -> Tuple[str, bool]:
        name = f"af-grade-{uuid.uuid4().hex[:12]}"
        script = ("cp -r /src/. /work && cp -r /tests/. /work && cd /work && "
                  "exec python " + " ".join(f"'{a}'" for a in PYTEST)
                  + " --junitxml=/out/r.xml")
        try:
            p = subprocess.run(["docker", "run", "--rm", "--name", name, *LOCKDOWN,
                                "-v", f"{src.resolve()}:/src:ro",
                                "-v", f"{tests.resolve()}:/tests:ro",
                                "-v", f"{out.resolve()}:/out",
                                IMAGE, "sh", "-c", script],
                               capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return "", True
        return Q.scrub_paths(p.stdout, Path("/work")), False

    def manifest(self) -> dict:
        return {**super().manifest(), "image": IMAGE, "image_id": image_id()}

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
def open_workspace(fault: dict, backend: str = "local",
                   timeout: float = GRADE_TIMEOUT) -> Iterator[Workspace]:
    if backend not in BACKENDS:
        raise ValueError(f"unknown sandbox {backend!r}: use one of {BACKENDS}")
    with tempfile.TemporaryDirectory() as td:
        ws = (DockerWorkspace if backend == "docker" else Workspace)(fault, Path(td), timeout)
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
