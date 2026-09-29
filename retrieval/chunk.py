"""Split a repo's Python source into retrievable chunks at function granularity.

Function granularity is not an arbitrary choice: corpus/faults/*.json label a fault
with `function`, so a chunk that IS a function makes the ground truth exact rather
than approximate. Retrieval is then scored against "did you return the function that
actually contains the bug", not "did you return a file that happens to contain it".

File-level recall is reported alongside, since that is the weaker claim most
retrieval evaluations actually make.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Chunk:
    file: str            # repo-relative path
    qualname: str        # "merge" or "Interval.overlaps"
    start_line: int
    end_line: int
    text: str

    @property
    def uid(self) -> str:
        return f"{self.file}::{self.qualname}"


def _walk(node: ast.AST, prefix: str = ""):
    """Yield (qualname, node) for every function/method, one level of nesting deep."""
    for child in node.body:  # type: ignore[attr-defined]
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield f"{prefix}{child.name}", child
        elif isinstance(child, ast.ClassDef):
            for sub in child.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{prefix}{child.name}.{sub.name}", sub


def chunk_file(path: Path, repo_root: Path) -> List[Chunk]:
    src = path.read_text()
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    rel = str(path.relative_to(repo_root))
    out: List[Chunk] = []
    covered: set[int] = set()

    for qualname, node in _walk(tree):
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end = node.end_lineno or node.lineno
        covered.update(range(start, end + 1))
        out.append(Chunk(rel, qualname, start, end, "\n".join(lines[start - 1:end])))

    # Module-level code (imports, constants, class bodies outside methods) becomes one
    # chunk so that faults living there are still retrievable rather than invisible.
    leftover = [i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()]
    if leftover:
        out.append(Chunk(rel, "<module>", leftover[0], leftover[-1],
                         "\n".join(lines[i - 1] for i in leftover)))
    return out


def chunk_repo(repo_root: Path, include_tests: bool = False) -> List[Chunk]:
    """All chunks in a repo. Tests are excluded by default.

    Excluding tests is deliberate and worth defending: the failing test name appears
    verbatim in the query, so a test chunk is a trivially perfect lexical match that
    tells the agent nothing about where the bug lives. Including them would inflate
    every lexical arm and flatter the baseline.
    """
    out: List[Chunk] = []
    for p in sorted(repo_root.rglob("*.py")):
        parts = set(p.parts)
        if "__pycache__" in parts:
            continue
        if not include_tests and ("tests" in parts or p.name.startswith("test_")):
            continue
        out.extend(chunk_file(p, repo_root))
    return out
