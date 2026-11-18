"""The edit gate: an edit may change program logic, never gain a capability."""
from __future__ import annotations

import ast
from typing import List, Set

ALLOWED_MODULES = frozenset({
    "bisect", "collections", "dataclasses", "datetime", "decimal", "enum",
    "fractions", "functools", "heapq", "itertools", "math", "operator", "re",
    "typing",
})

DENIED = frozenset({
    "os", "sys", "subprocess", "builtins", "importlib", "ctypes", "gc", "inspect",
    "socket", "shutil", "pathlib", "glob", "io", "signal", "threading",
    "multiprocessing", "atexit", "pytest", "unittest", "tempfile", "pickle",
    "marshal", "types", "code", "codeop", "runpy", "pty", "posix", "nt",
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "exit", "quit", "print", "input",
    "breakpoint", "help", "memoryview",
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
    """What `after` adds that the gate forbids, relative to `before`. Empty is allowed."""
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
