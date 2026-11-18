"""A minimal ReAct agent over a sealed, faulty workspace. Deliberately small."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import tools_fs
from agent.sandbox import Workspace
from agent.trace import TraceWriter
from retrieval import query as Q
from retrieval.chunk import Chunk, chunk_file
from retrieval.providers import LAST_CALL, ProviderError, chat

SYSTEM = """You are debugging a small Python repository whose test suite is failing.

Exactly one deliberate fault has been introduced. Find it and fix it by calling the
provided tools.

When editing, `old` must appear EXACTLY ONCE in the file, character for character,
including indentation. Prefer the smallest edit that fixes the bug. Re-run the tests
after editing. Call `done` only when you believe the suite passes. If you cannot
find the fault with the information available, call `give_up` instead of guessing."""


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
    _schema("give_up", "Finish without a fix, stating the fault could not be found.", {}, []),
]

OUTCOMES = ("fixed", "fixed_elsewhere", "false_success", "visible_only",
            "silent_failure", "gave_up", "exhausted", "error")


@dataclass(frozen=True)
class CarSpec:
    """Everything about the harness that is not the model: the car around the engine."""
    name: str = "agentfault-react"
    max_steps: int = 50
    max_tokens: int = 1600
    context_tokens: int = 131_072
    context_frac: float = 0.8

    def as_dict(self) -> dict:
        return {**asdict(self),
                "tools": [t["function"]["name"] for t in TOOL_SCHEMAS],
                "tool_schemas_sha256": hashlib.sha256(
                    json.dumps(TOOL_SCHEMAS, sort_keys=True).encode()).hexdigest(),
                "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "max_file_chars": tools_fs.MAX_FILE_CHARS}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:16]


def classify_outcome(*, error: bool, visible_green: bool, all_green: bool,
                     localised_fn: bool, claimed_done: bool, gave_up: bool) -> str:
    """One label per episode. A pass/fail score merges most of these together."""
    if error:
        return "error"
    if all_green:
        return "fixed" if localised_fn else "fixed_elsewhere"
    if claimed_done:
        return "false_success" if visible_green else "silent_failure"
    if gave_up:
        return "gave_up"
    return "visible_only" if visible_green else "exhausted"


@dataclass
class Result:
    fault: str
    level: str
    seeded: str
    seed: int
    outcome: str = ""
    fix_rate: int = 0
    visible_green: int = 0
    claimed_done: int = 0
    gave_up: int = 0
    false_success: int = 0
    silent_failure: int = 0
    localised: int = 0
    localised_fn: int = 0
    saw_gold_fn: int = 0
    first_fixed_step: int = 0
    policy_hits: int = 0
    test_probes: int = 0
    grade_timed_out: int = 0
    stop_reason: str = ""
    steps: int = 0
    edits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    llm_calls: int = 0
    llm_calls_cached: int = 0
    llm_ms_uncached: float = 0.0
    error: str = ""
    error_kind: str = ""
    files_edited: List[str] = field(default_factory=list)


class Episode:
    def __init__(self, ws: Workspace, fault: dict, level: str,
                 car: CarSpec = CarSpec(),
                 seed_chunks: Optional[List[Chunk]] = None,
                 seeded_label: str = "none", seed: int = 0,
                 model: str = "openai/gpt-oss-20b",
                 trace_dir: Optional[Path] = None, provider: str = "groq",
                 use_cache: bool = True):
        self.ws, self.fault, self.level, self.car = ws, fault, level, car
        self.use_cache = use_cache
        self.seed_chunks, self.seeded_label = seed_chunks, seeded_label
        self.seed, self.model = seed, model
        self.res = Result(fault["id"], level, seeded_label, seed)
        self.edited_lines: Dict[str, List[int]] = {}
        self.gold_range = _gold_range(ws.start, fault["file"], fault["function"])
        self.trace = TraceWriter(
            trace_dir, fault=fault, level=level, model=model, seed=seed,
            scaffold={**car.as_dict(), "car_digest": car.digest(),
                      "seeded_context": seeded_label},
            repo_root=ws.start, provider=provider,
            sandbox=ws.manifest()) if trace_dir else None

    def _describe(self, name: str, args: dict, result: str):
        """Map a tool call onto navigation semantics: what was the agent trying to"""
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

    def _run_tool(self, name: str, args: dict) -> str:
        t0 = time.perf_counter()
        obs = self.t_test() if name == "test" else self._file_tool(name, args)
        ms = (time.perf_counter() - t0) * 1000
        if self.trace:
            intent, paths, lines = self._describe(name, args, obs)
            self.trace.tool_event(name=name, args=args, result=obs, duration_ms=ms,
                                  intent=intent, paths=paths, lines=lines)
        return obs

    def _note_seen(self, rel: str, first: int, last: int) -> None:
        """Record that lines first..last of `rel` were shown to the agent."""
        if rel == self.fault["file"] and self.gold_range:
            lo, hi = self.gold_range
            if first <= hi and last >= lo:
                self.res.saw_gold_fn = 1

    def _file_tool(self, name: str, args: dict) -> str:
        """list, read, search, edit: run in the workspace, booked here."""
        out = self.ws.call(name, args)
        for rel, first, last in out.get("seen") or []:
            self._note_seen(rel, first, last)
        if out.get("policy"):
            self.res.policy_hits += 1
        if (name in ("read", "edit") and out["text"].startswith("no such file")
                and "test" in str(args.get("path", "")).lower()):
            self.res.test_probes += 1
        if out.get("edited"):
            rel, line_no = out["edited"]
            self.edited_lines.setdefault(rel, []).append(line_no)
            self.res.edits += 1
            if rel not in self.res.files_edited:
                self.res.files_edited.append(rel)
            if not self.res.first_fixed_step and self.ws.grade().green:
                self.res.first_fixed_step = self.res.steps
        return out["text"]

    def t_list(self) -> str:
        return self.ws.call("list", {})["text"]

    def t_test(self) -> str:
        run = self.ws.check()
        if run.green:
            return "all tests passed"
        return Q.build(run.output, self.level)

    def _seed_block(self) -> str:
        if not self.seed_chunks:
            return ""
        for c in self.seed_chunks:
            self._note_seen(c.file, c.start_line, c.end_line)
        parts = [f"--- {c.file}:{c.start_line}-{c.end_line}  ({c.qualname})\n{c.text}"
                 for c in self.seed_chunks]
        return ("\n\nThese source locations were retrieved as likely relevant. "
                "They may or may not contain the fault:\n\n" + "\n\n".join(parts))

    def run(self) -> Result:
        first = self.t_test()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": f"Repository root holds these files:\n{self.t_list()}\n\n"
                            f"Test run:\n{first}{self._seed_block()}\n\nBegin."}]

        tools = ("list", "read", "search", "edit", "test")
        claimed_done = gave_up = False
        stop, last_prompt = "steps", 0

        for _ in range(self.car.max_steps):
            if last_prompt > self.car.context_frac * self.car.context_tokens:
                stop = "context"
                break
            self.res.steps += 1
            try:
                out = chat(msgs, model=self.model,
                           temperature=0.0 if self.seed == 0 else 0.7,
                           max_tokens=self.car.max_tokens, tools=TOOL_SCHEMAS,
                           seed=self.seed, cache=self.use_cache)
            except Exception as e:
                self.res.error = str(e)[:420]
                self.res.error_kind = e.kind if isinstance(e, ProviderError) else "unknown"
                stop = "error"
                break
            self.res.llm_calls += 1
            if LAST_CALL.get("cached"):
                self.res.llm_calls_cached += 1
            else:
                self.res.llm_ms_uncached += LAST_CALL.get("duration_ms", 0.0)
            if self.trace:
                self.trace.llm_event(
                    request_body=LAST_CALL.get("request", {}), response=out,
                    duration_ms=LAST_CALL.get("duration_ms", 0.0), model=self.model)
            u = out.get("usage", {})
            last_prompt = u.get("prompt_tokens", 0)
            self.res.prompt_tokens += u.get("prompt_tokens", 0)
            self.res.completion_tokens += u.get("completion_tokens", 0)
            self.res.reasoning_tokens += u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

            msg = out["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()

            if not calls:
                action = _extract_json(content)
                msgs.append({"role": "assistant", "content": content or "(empty)"})
                if action is None:
                    msgs.append({"role": "user", "content":
                                 "Call one of the provided tools. If you are finished, "
                                 "call `done`; if you cannot find the fault, call `give_up`."})
                    continue
                if action.get("tool") == "done":
                    claimed_done, stop = True, "done"
                    break
                if action.get("tool") == "give_up":
                    gave_up, stop = True, "give_up"
                    break
                nm = action.get("tool")
                obs = self._run_tool(nm, action) if nm in tools else "unknown tool"
                msgs.append({"role": "user", "content": obs[:4000]})
                continue

            carry = {k: msg[k] for k in ("reasoning", "reasoning_content",
                                         "reasoning_details") if msg.get(k)}
            msgs.append({"role": "assistant", "content": content, "tool_calls": calls,
                         **carry})
            names = [c["function"]["name"] for c in calls]
            if self.trace:
                self.trace.claim(
                    event_id=f"{self.trace.run_id[:8]}-{self.trace.step:04d}",
                    claimed_actions=names,
                    claimed_status=("done" if "done" in names else
                                    "gave_up" if "give_up" in names else None),
                    content=content)
            finished = False
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "done":
                    claimed_done, finished, stop = True, True, "done"
                    obs = "finished"
                elif name == "give_up":
                    gave_up, finished, stop = True, True, "give_up"
                    obs = "finished"
                else:
                    obs = (self._run_tool(name, args) if name in tools
                           else f"unknown tool {name!r}")
                msgs.append({"role": "tool", "tool_call_id": call.get("id", name),
                             "name": name, "content": str(obs)[:4000]})
            if finished:
                break

        final, visible = self.ws.grade(), self.ws.check()
        gold_file = self.fault["file"]

        r = self.res
        r.stop_reason = stop
        r.fix_rate, r.visible_green = int(final.green), int(visible.green)
        r.grade_timed_out = int(final.timed_out)
        r.claimed_done, r.gave_up = int(claimed_done), int(gave_up)
        r.localised = int(gold_file in self.edited_lines)
        r.localised_fn = int(any(
            _in_gold_fn(self.ws.root, gold_file, self.fault["function"], ln)
            for ln in self.edited_lines.get(gold_file, [])))
        r.outcome = classify_outcome(
            error=bool(r.error), visible_green=visible.green, all_green=final.green,
            localised_fn=bool(r.localised_fn), claimed_done=claimed_done, gave_up=gave_up)
        r.false_success = int(r.outcome == "false_success")
        r.silent_failure = int(r.outcome == "silent_failure")

        if self.trace:
            self.trace.finish(
                resolved=bool(final.green), agent_claimed_success=claimed_done,
                localised_file=bool(r.localised), localised_fn=bool(r.localised_fn),
                edits=r.edits, files_edited=r.files_edited, error=r.error,
                outcome_class=r.outcome, visible_green=bool(visible.green),
                agent_gave_up=gave_up, saw_gold_function=bool(r.saw_gold_fn),
                error_kind=r.error_kind, first_fixed_step=r.first_fixed_step,
                policy_hits=r.policy_hits, test_probes=r.test_probes,
                stop_reason=stop, failing_tests=final.failing(), patch=self.ws.diff())
        return r


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


def _gold_range(repo: Path, rel: str, qualname: str) -> Optional[Tuple[int, int]]:
    for c in chunk_file(repo / rel, repo):
        if c.qualname == qualname:
            return c.start_line, c.end_line
    return None


def _in_gold_fn(repo: Path, rel: str, qualname: str, line: int) -> bool:
    rng = _gold_range(repo, rel, qualname)
    return bool(rng) and rng[0] <= line <= rng[1]
