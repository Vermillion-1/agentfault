"""A minimal ReAct agent over a faulty checkout. Deliberately ~300 lines.

The agent is a MEANS, not the product. The decision doc names the failure mode
exactly: "building a framework instead of running an experiment". So this has five
tools, one JSON action per turn, no framework, and no abstraction that is not paid
for by the experiment.

WHAT THIS MEASURES

  fix_rate            did the suite go green
  localised           did it edit the file the fault actually lives in
  localised_fn        did it edit inside the faulty function's line range
  silent_failure      did it CLAIM done while the suite was still red
  steps, tokens       cost

silent_failure is the metric worth caring about. Agent Retrieval Bench found
27-35% of real trajectories never touch a gold file and still emit a patch; an
agent that confidently ships a wrong answer is worse than one that gives up, and
almost nothing measures it.

TWO KNOBS, EVERYTHING ELSE FIXED

  observation level   L0..L3 -- how much the test runner tells the agent
  seeded context      which chunks (if any) are placed in the prompt up front

Those are the same two axes as the retrieval sweep, which is what lets the
intrinsic and extrinsic halves of this study be compared at all.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.trace import TraceWriter        # noqa: E402
from retrieval import query as Q          # noqa: E402
from retrieval.chunk import Chunk         # noqa: E402
from retrieval.providers import LAST_CALL, groq_chat  # noqa: E402

MAX_STEPS = 12
MAX_FILE_CHARS = 6000

SYSTEM = """You are debugging a small Python repository whose test suite is failing.

Exactly one deliberate fault has been introduced. Find it and fix it by calling the
provided tools.

When editing, `old` must appear EXACTLY ONCE in the file, character for character,
including indentation. Prefer the smallest edit that fixes the bug. Re-run the tests
after editing. Call `done` only when you believe the suite passes."""


def _schema(name, desc, props, required):
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


TOOL_SCHEMAS = [
    _schema("list", "List the repository source files.", {}, []),
    _schema("read", "Read a source file with line numbers.",
            {"path": {"type": "string", "description": "repo-relative path"}}, ["path"]),
    _schema("search", "Regex search across source files.",
            {"pattern": {"type": "string"}}, ["pattern"]),
    _schema("edit", "Replace an exact unique snippet in a file.",
            {"path": {"type": "string"},
             "old": {"type": "string", "description": "exact text, must be unique"},
             "new": {"type": "string"}}, ["path", "old", "new"]),
    _schema("test", "Re-run the test suite.", {}, []),
    _schema("done", "Finish, asserting the suite now passes.", {}, []),
]


@dataclass
class Result:
    fault: str
    level: str
    seeded: str
    seed: int
    fix_rate: int = 0
    localised: int = 0
    localised_fn: int = 0
    silent_failure: int = 0
    steps: int = 0
    edits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    error: str = ""
    files_edited: List[str] = field(default_factory=list)


class Episode:
    def __init__(self, repo: Path, fault: dict, level: str,
                 seed_chunks: Optional[List[Chunk]] = None,
                 seeded_label: str = "none", seed: int = 0,
                 model: str = "openai/gpt-oss-20b",
                 trace_dir: Optional[Path] = None):
        self.repo, self.fault, self.level = repo, fault, level
        self.seed_chunks, self.seeded_label = seed_chunks, seeded_label
        self.seed, self.model = seed, model
        self.res = Result(fault["id"], level, seeded_label, seed)
        self.edited_lines: Dict[str, List[int]] = {}
        self.trace = TraceWriter(
            trace_dir, fault=fault, level=level, model=model, seed=seed,
            scaffold={"max_steps": MAX_STEPS, "max_file_chars": MAX_FILE_CHARS,
                      "tools": [t["function"]["name"] for t in TOOL_SCHEMAS],
                      "system_prompt_sha256": __import__("hashlib").sha256(
                          SYSTEM.encode()).hexdigest(),
                      "seeded_context": seeded_label},
            repo_root=repo) if trace_dir else None

    # ------------------------------------------------- tool dispatch + tracing
    def _describe(self, name: str, args: dict, result: str):
        """Map a tool call onto navigation semantics: what was the agent trying to
        do, which paths did it touch, how much context did it consume. This is the
        record that answers 'how did it search the repo', which is the thing almost
        nobody logs and the thing the retrieval literature says actually matters."""
        lines = result.count("\n") + 1 if result else 0
        if name == "read":
            return "read", [args.get("path", "")], lines
        if name == "search":
            paths = sorted({ln.split(":", 1)[0] for ln in result.splitlines()
                            if ":" in ln and not ln.startswith("no matches")})
            return "search", paths, lines
        if name == "edit":
            return "edit", [args.get("path", "")], 1
        if name == "test":
            return "test", [], lines
        return "list", [], lines

    def _run_tool(self, name: str, args: dict, fn) -> str:
        t0 = time.perf_counter()
        obs = str(fn(args))
        ms = (time.perf_counter() - t0) * 1000
        if self.trace:
            intent, paths, lines = self._describe(name, args, obs)
            self.trace.tool_event(name=name, args=args, result=obs, duration_ms=ms,
                                  intent=intent, paths=paths, lines=lines)
        return obs

    # ---------------------------------------------------------------- tools
    def _sources(self) -> List[Path]:
        return [p for p in sorted(self.repo.rglob("*.py"))
                if "__pycache__" not in p.parts and "tests" not in p.parts]

    def t_list(self, _) -> str:
        return "\n".join(str(p.relative_to(self.repo)) for p in self._sources())

    def t_read(self, a) -> str:
        p = (self.repo / a.get("path", "")).resolve()
        if not str(p).startswith(str(self.repo.resolve())) or not p.is_file():
            return f"no such file: {a.get('path')}"
        text = p.read_text()[:MAX_FILE_CHARS]
        return "\n".join(f"{i:4d}| {ln}" for i, ln in enumerate(text.splitlines(), 1))

    def t_search(self, a) -> str:
        pat = a.get("pattern", "")
        try:
            rx = re.compile(pat)
        except re.error as e:
            return f"bad regex: {e}"
        hits = []
        for p in self._sources():
            for i, ln in enumerate(p.read_text().splitlines(), 1):
                if rx.search(ln):
                    hits.append(f"{p.relative_to(self.repo)}:{i}: {ln.strip()[:110]}")
        return "\n".join(hits[:40]) or "no matches"

    def t_edit(self, a) -> str:
        rel = a.get("path", "")
        p = (self.repo / rel).resolve()
        if not str(p).startswith(str(self.repo.resolve())) or not p.is_file():
            return f"no such file: {rel}"
        old, new = a.get("old", ""), a.get("new", "")
        text = p.read_text()
        n = text.count(old)
        if not old or n == 0:
            return "edit rejected: `old` not found verbatim. Read the file and copy the exact text."
        if n > 1:
            return f"edit rejected: `old` appears {n} times. Include more context to disambiguate."
        line_no = text[:text.index(old)].count("\n") + 1
        p.write_text(text.replace(old, new))
        self.edited_lines.setdefault(rel, []).append(line_no)
        self.res.edits += 1
        if rel not in self.res.files_edited:
            self.res.files_edited.append(rel)
        return f"edited {rel} at line {line_no}"

    def t_test(self, _) -> str:
        return Q.build(Q.run_suite(self.repo), self.level)

    # --------------------------------------------------------------- driving
    def _seed_block(self) -> str:
        if not self.seed_chunks:
            return ""
        parts = [f"--- {c.file}:{c.start_line}-{c.end_line}  ({c.qualname})\n{c.text}"
                 for c in self.seed_chunks]
        return ("\n\nThese source locations were retrieved as likely relevant. "
                "They may or may not contain the fault:\n\n" + "\n\n".join(parts))

    def run(self) -> Result:
        first = self.t_test(None)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"Repository root holds these files:\n{self.t_list(None)}\n\n"
                            f"Test run:\n{first}{self._seed_block()}\n\nBegin."}]

        tools = {"list": self.t_list, "read": self.t_read, "search": self.t_search,
                 "edit": self.t_edit, "test": self.t_test}
        claimed_done = False

        for _ in range(MAX_STEPS):
            self.res.steps += 1
            try:
                out = groq_chat(msgs, model=self.model, temperature=0.0 if self.seed == 0 else 0.7,
                                max_tokens=1600, tools=TOOL_SCHEMAS)
            except Exception as e:
                self.res.error = str(e)[:160]
                break
            if self.trace:
                self.trace.llm_event(
                    request_body=LAST_CALL.get("request", {}), response=out,
                    duration_ms=LAST_CALL.get("duration_ms", 0.0), model=self.model)
            u = out.get("usage", {})
            self.res.prompt_tokens += u.get("prompt_tokens", 0)
            self.res.completion_tokens += u.get("completion_tokens", 0)
            self.res.reasoning_tokens += u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

            msg = out["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()

            if not calls:
                # Fallback for a model that describes an action in prose instead of
                # calling it. Keeps one loop working across both conventions.
                action = _extract_json(content)
                msgs.append({"role": "assistant", "content": content or "(empty)"})
                if action is None:
                    msgs.append({"role": "user", "content": "Call one of the provided tools."})
                    continue
                if action.get("tool") == "done":
                    claimed_done = True
                    break
                nm = action.get("tool")
                fn = tools.get(nm)
                obs = self._run_tool(nm, action, fn) if fn else "unknown tool"
                msgs.append({"role": "user", "content": obs[:4000]})
                continue

            msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
            if self.trace:
                self.trace.claim(
                    event_id=f"{self.trace.run_id[:8]}-{self.trace.step:04d}",
                    claimed_actions=[c["function"]["name"] for c in calls],
                    claimed_status="done" if any(
                        c["function"]["name"] == "done" for c in calls) else None,
                    content=content)
            stop = False
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "done":
                    claimed_done, stop = True, True
                    obs = "finished"
                else:
                    fn = tools.get(name)
                    obs = self._run_tool(name, args, fn) if fn else f"unknown tool {name!r}"
                msgs.append({"role": "tool", "tool_call_id": call.get("id", name),
                             "name": name, "content": str(obs)[:4000]})
            if stop:
                break

        # --------------------------------------------------------- scoring
        passed, _ = _suite_green(self.repo)
        self.res.fix_rate = int(passed)
        self.res.silent_failure = int(claimed_done and not passed)
        gold_file = self.fault["file"]
        self.res.localised = int(gold_file in self.edited_lines)
        self.res.localised_fn = int(any(
            _in_gold_fn(self.repo, gold_file, self.fault["function"], ln)
            for ln in self.edited_lines.get(gold_file, [])))

        if self.trace:
            self.trace.finish(
                resolved=bool(passed), agent_claimed_success=claimed_done,
                localised_file=bool(self.res.localised),
                localised_fn=bool(self.res.localised_fn),
                edits=self.res.edits, files_edited=self.res.files_edited,
                error=self.res.error)
        return self.res


def _extract_json(text: str) -> Optional[dict]:
    """Models wrap JSON in fences or prose; take the first balanced object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
            elif ch == '"':
                in_str, esc = True, False
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict) and "tool" in obj:
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _suite_green(repo: Path) -> tuple:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header",
                        "--tb=no", "-p", "no:cacheprovider"],
                       cwd=repo, capture_output=True, text=True)
    return p.returncode == 0, p.stdout


def _in_gold_fn(repo: Path, rel: str, qualname: str, line: int) -> bool:
    from retrieval.chunk import chunk_file
    for c in chunk_file(repo / rel, repo):
        if c.qualname == qualname:
            return c.start_line <= line <= c.end_line
    return False
