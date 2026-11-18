#!/usr/bin/env python3
"""Four-layer provenance for an agent episode."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PREVIEW_CHARS = 240
SCHEMA_VERSION = "agentfault-trace-0.1"

REASONING_FIELDS = ("reasoning", "reasoning_content", "thinking")

RAW_COT_MODELS = ("gpt-oss", "deepseek-r1", "qwen")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def preview_hash(value: Any, n: int = PREVIEW_CHARS) -> Dict[str, Any]:
    """The publish-safe representation of any payload."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return {"preview": text[:n], "sha256": sha256(text), "chars": len(text)}


def _git_sha(path: Path) -> Optional[str]:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                       capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None


def _git_dirty(path: Path) -> Optional[bool]:
    """Uncommitted changes in the harness checkout: then the sha alone does not"""
    r = subprocess.run(["git", "status", "--porcelain"], cwd=path,
                       capture_output=True, text=True)
    return bool(r.stdout.strip()) if r.returncode == 0 else None


@dataclass
class Navigation:
    """Cumulative view of how the agent moved through the repository."""
    paths_seen: set = field(default_factory=set)
    lines_consumed: int = 0
    searches: int = 0
    reads: int = 0
    edits: int = 0

    def observe(self, intent: str, paths: List[str], lines: int) -> Dict[str, Any]:
        novel = [p for p in paths if p not in self.paths_seen]
        repeat = [p for p in paths if p in self.paths_seen]
        self.paths_seen.update(paths)
        self.lines_consumed += lines
        if intent == "search":
            self.searches += 1
        elif intent == "read":
            self.reads += 1
        elif intent == "edit":
            self.edits += 1
        return {
            "intent": intent, "paths_touched": paths,
            "novel_paths_count": len(novel), "repeat_paths_count": len(repeat),
            "lines_returned": lines,
            "cumulative_lines_consumed": self.lines_consumed,
            "cumulative_paths_seen": len(self.paths_seen),
        }


class TraceWriter:
    def __init__(self, out_dir: Path, *, fault: dict, level: str, model: str,
                 seed: int, scaffold: Dict[str, Any], repo_root: Path,
                 harness_root: Optional[Path] = None, provider: str = "groq",
                 sandbox: Optional[Dict[str, Any]] = None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        self.t0 = time.time()
        self.step = 0
        self.nav = Navigation()
        self._cost = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        self._reasoning_seen = False

        harness_root = harness_root or Path(__file__).resolve().parent.parent
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "harness": {
                "name": "agentfault", "git_sha": _git_sha(harness_root),
                "dirty": _git_dirty(harness_root),
                "python": platform.python_version(), "platform": platform.platform(),
            },
            "scaffold": scaffold,
            "sandbox": sandbox or {},
            "model": {"provider": provider, "model_id": model, "seed": seed},
            "task": {
                "corpus": "agentfault", "fault_id": fault["id"], "repo": fault["repo"],
                "observation_level": level,
            },
            "injection": {
                "injected": True,
                "fault_id": fault["id"], "category": fault["category"],
                "file": fault["file"], "function": fault["function"],
                "line_hint": fault.get("line_hint"),
                "expected_failing_tests": fault.get("expected_failing_tests", []),
                "generator": fault.get("provenance", {}).get("generator", "hand-authored"),
                "operator": fault.get("provenance", {}).get("operator"),
                "odc_type": fault.get("provenance", {}).get("odc_type"),
                "odc_qualifier": fault.get("provenance", {}).get("odc_qualifier"),
            },
            "repo_base_sha": _git_sha(repo_root),
        }
        self._events = (self.dir / "events.jsonl").open("w")
        self._claims = (self.dir / "claims.jsonl").open("w")
        self._reasoning = (self.dir / "reasoning.jsonl").open("w")
        self._annotations = (self.dir / "annotations.jsonl").open("w")

    def _write(self, fh, rec: dict) -> None:
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()

    def _base(self, operation: str) -> dict:
        self.step += 1
        return {
            "event_id": f"{self.run_id[:8]}-{self.step:04d}",
            "run_id": self.run_id,
            "step_index": self.step,
            "ts_offset_ms": round((time.time() - self.t0) * 1000, 1),
            "operation": operation,
            "provenance_origin": "HARNESS_OBSERVED",
        }

    def llm_event(self, *, request_body: dict, response: dict, duration_ms: float,
                  model: str) -> str:
        """Record one model call from the ACTUAL request and response."""
        ev = self._base("chat")
        msg = (response.get("choices") or [{}])[0].get("message", {}) or {}
        usage = response.get("usage", {}) or {}

        reasoning_field = next((f for f in REASONING_FIELDS if msg.get(f)), None)
        reasoning_text = msg.get(reasoning_field) if reasoning_field else None
        is_raw = any(tag in model.lower() for tag in RAW_COT_MODELS)

        ev["evidence_ref"] = {
            "http_request_sha256": sha256(json.dumps(request_body, sort_keys=True, default=str)),
            "http_response_sha256": sha256(json.dumps(response, sort_keys=True, default=str)),
            "response_id": response.get("id"),
        }
        ev["llm"] = {
            "request_model": model,
            "input_messages": preview_hash(request_body.get("messages", [])),
            "output_content": preview_hash(msg.get("content") or ""),
            "tool_calls": [c.get("function", {}).get("name") for c in (msg.get("tool_calls") or [])],
            "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "reasoning_output_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            },
            "reasoning_source_field": reasoning_field,
            "reasoning_is_raw_cot": bool(reasoning_field) and is_raw,
            "reasoning_sha256": sha256(reasoning_text) if reasoning_text else None,
            "duration_ms": round(duration_ms, 1),
        }
        self._cost["input_tokens"] += ev["llm"]["usage"]["input_tokens"]
        self._cost["output_tokens"] += ev["llm"]["usage"]["output_tokens"]
        self._cost["reasoning_tokens"] += ev["llm"]["usage"]["reasoning_output_tokens"]
        self._write(self._events, ev)

        if reasoning_text:
            self._reasoning_seen = True
            self._write(self._reasoning, {
                "event_id": ev["event_id"], "run_id": self.run_id,
                "source_field": reasoning_field, "is_raw_cot": is_raw,
                "sha256": sha256(reasoning_text), "text": reasoning_text,
            })
        return ev["event_id"]

    def tool_event(self, *, name: str, args: dict, result: str, duration_ms: float,
                   intent: str, paths: List[str], lines: int) -> str:
        ev = self._base("execute_tool")
        ev["tool"] = {
            "name": name,
            "args": preview_hash(args),
            "result": preview_hash(result),
            "duration_ms": round(duration_ms, 1),
        }
        ev["navigation"] = self.nav.observe(intent, paths, lines)
        self._write(self._events, ev)
        return ev["event_id"]

    def claim(self, *, event_id: str, claimed_actions: List[str],
              claimed_status: Optional[str] = None, content: str = "") -> None:
        """What the agent SAYS it did. Never trusted, joined to layer 1 on event_id."""
        self._write(self._claims, {
            "event_id": event_id, "run_id": self.run_id,
            "provenance_origin": "AGENT_REPORTED",
            "claimed_actions": claimed_actions,
            "claimed_status": claimed_status,
            "content": preview_hash(content),
        })

    def annotate(self, *, label: str, label_source: str, event_id: Optional[str] = None,
                 **extra) -> None:
        self._write(self._annotations, {
            "run_id": self.run_id, "event_id": event_id,
            "label": label, "label_source": label_source, **extra,
        })

    def finish(self, *, resolved: bool, agent_claimed_success: bool,
               localised_file: bool, localised_fn: bool, edits: int,
               files_edited: List[str], error: str = "",
               outcome_class: str = "", visible_green: Optional[bool] = None,
               agent_gave_up: bool = False, saw_gold_function: bool = False,
               error_kind: str = "", first_fixed_step: int = 0,
               policy_hits: int = 0, test_probes: int = 0, stop_reason: str = "",
               failing_tests: Optional[List[str]] = None, patch: str = "") -> Path:
        if visible_green is None:
            visible_green = resolved
        self.manifest["outcome"] = {
            "resolved": resolved,
            "visible_green": visible_green,
            "outcome_class": outcome_class,
            "agent_claimed_success": agent_claimed_success,
            "agent_gave_up": agent_gave_up,
            "silent_failure": outcome_class == "silent_failure",
            "false_success": outcome_class == "false_success",
            "localised_file": localised_file,
            "localised_function": localised_fn,
            "saw_gold_function": saw_gold_function,
            "first_fixed_step": first_fixed_step,
            "policy_hits": policy_hits, "test_probes": test_probes,
            "stop_reason": stop_reason,
            "failing_tests": failing_tests or [],
            "edits": edits, "files_edited": files_edited,
            "error": error, "error_kind": error_kind,
        }
        self.manifest["cost"] = {
            **self._cost,
            "wall_clock_ms": round((time.time() - self.t0) * 1000, 1),
            "steps": self.step,
        }
        self.manifest["navigation_summary"] = {
            "paths_seen": sorted(self.nav.paths_seen),
            "lines_consumed": self.nav.lines_consumed,
            "searches": self.nav.searches, "reads": self.nav.reads,
            "edits": self.nav.edits,
        }
        self.manifest["has_reasoning_sidecar"] = self._reasoning_seen

        inj = self.manifest["injection"]
        self.annotate(label="injected_fault_location", label_source="INJECTED_GROUND_TRUTH",
                      file=inj["file"], function=inj["function"],
                      category=inj["category"], odc_qualifier=inj["odc_qualifier"])
        if policy_hits:
            self.annotate(label="capability_edit_refused", label_source="HARNESS_OBSERVED",
                          count=policy_hits)
        if test_probes:
            self.annotate(label="test_path_probe", label_source="HARNESS_OBSERVED",
                          count=test_probes)

        for fh in (self._events, self._claims, self._reasoning, self._annotations):
            fh.close()
        (self.dir / "patch.diff").write_text(patch)
        p = self.dir / "run.json"
        p.write_text(json.dumps(self.manifest, indent=2, default=str) + "\n")
        return p
