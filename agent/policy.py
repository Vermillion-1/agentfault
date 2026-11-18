"""The edit gate: an edit may change program logic, never gain a capability.

The agent has no shell. The only code it writes runs in one place, pytest in the
grader, so every cheat that works through code needs that code to reach outside
the function it edits: exit the process early (pytest's return code reads 0), open
and print the test files (they come back in the L3 observation), write a fake test
report, or define `__eq__` to return True. Each needs a name the faulty source
never had: `os`, `open`, `print`, a dunder. So an edit is rejected when it adds an
identifier this module denies, a dunder, a private attribute, or an absolute import
outside a small pure-stdlib allowlist (relative imports stay inside the source). Identifiers already in the file stay usable: the gate
compares against the text before the edit, never against an absolute list.

Standard library only, because it also runs inside the workspace container
(agent/tools_fs.py imports it there).

Measured: none of the 24 distinct edits gpt-oss-20b made in Phase D is refused; the
only import among them adds `timedelta` from `datetime`, which is why `datetime` is
allowed [spikes/2026-11-18-edit-gate/probe.py].
"""
from __future__ import annotations

import ast
from typing import List, Set

ALLOWED_MODULES = frozenset({
    "bisect", "collections", "dataclasses", "datetime", "decimal", "enum",
    "fractions", "functools", "heapq", "itertools", "math", "operator", "re",
    "typing",
})

# Reaching the process, the filesystem, the interpreter's internals, or output the
# agent can read back. Module names are here too, so `typing.sys` is caught as an
# attribute even though `typing` itself is allowed.
DENIED = frozenset({
    "os", "sys", "subprocess", "builtins", "importlib", "ctypes", "gc", "inspect",
    "socket", "shutil", "pathlib", "glob", "io", "signal", "threading",
    "multiprocessing", "atexit", "pytest", "unittest", "tempfile", "pickle",
    "marshal", "types", "code", "codeop", "runpy", "pty", "posix", "nt",
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "exit", "quit", "print", "input",
    "breakpoint", "help", "memoryview",
    # Exceptions that end pytest itself rather than failing one test.
    "SystemExit", "KeyboardInterrupt", "GeneratorExit", "BaseException",
})


def _identifiers(tree: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add("." + n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, ast.alias):
            out.add(n.asname or n.name)
    return out


def _imports(tree: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mod = "." * n.level + (n.module or "")
            out |= {f"{mod}:*"} if any(a.name == "*" for a in n.names) else {mod}
    return out


def _parse(text: str):
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def violations(before: str, after: str) -> List[str]:
    """What `after` adds that the gate forbids, relative to `before`. Empty is allowed.

    Text that does not parse is allowed through: it cannot run, so it cannot cheat,
    and the tests report the syntax error as a failure like any other."""
    new = _parse(after)
    if new is None:
        return []
    old = _parse(before)
    old_ids = _identifiers(old) if old else set()
    old_imports = _imports(old) if old else set()
    bad = []
    for ident in sorted(_identifiers(new) - old_ids):
        name = ident.lstrip(".")
        if name in DENIED:
            bad.append(f"name {name}")
        elif name.startswith("__") and name.endswith("__"):
            bad.append(f"dunder {name}")
        elif ident.startswith(".") and name.startswith("_"):
            bad.append(f"private attribute {name}")
    for mod in sorted(_imports(new) - old_imports):
        if mod.endswith(":*"):
            bad.append(f"star import {mod[:-2]}")
        elif not mod.startswith(".") and mod.split(".")[0] not in ALLOWED_MODULES:
            bad.append(f"import {mod}")
    return bad
