"""Spike 2026-11-17: probe the v2 grader (agent/sandbox.py, research repo) for two holes."""
import sys, ast, shutil, subprocess, tempfile, xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,'.')
from tools.corpus import load_faults
from agent.sandbox import open_workspace, held_out, PYTEST
from retrieval import query as Q

def strip(tests: Path, node: str, out: Path):
    """Copy tests to out with the held-out test function removed (by AST line range)."""
    shutil.copytree(tests, out)
    rel, *names = node.split("::")
    p = out/rel; src = p.read_text(); tree = ast.parse(src)
    body = tree.body
    for n in names[:-1]:
        body = next(c for c in body if isinstance(c, ast.ClassDef) and c.name == n).body
    fn = next(c for c in body if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and c.name == names[-1])
    lo = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno) - 1
    lines = src.splitlines(True); del lines[lo:fn.end_lineno]
    p.write_text("".join(lines))

def grade(root, tests):
    """Per-test outcomes from junit xml written outside the work dir."""
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)/"work"; shutil.copytree(root, work); shutil.copytree(tests, work, dirs_exist_ok=True)
        xml = Path(td)/"r.xml"
        p = subprocess.run([sys.executable, *PYTEST, f"--junitxml={xml}"], cwd=work, capture_output=True, text=True)
        if not xml.exists(): return None, p.returncode
        res = {}
        for tc in ET.parse(xml).iter("testcase"):
            nid = tc.get("classname").replace(".", "/") 
            ok = not any(c.tag in ("failure","error","skipped") for c in tc)
            res[f'{tc.get("classname")}::{tc.get("name")}'] = ok
        return res, p.returncode

fs = load_faults(); ok_strip = 0; n = 0
for f in fs:
    H = held_out(f)
    if not H: continue
    n += 1
    with open_workspace(f, "local") as ws, tempfile.TemporaryDirectory() as td:
        vis = Path(td)/"vis"; strip(ws.tests, H[0], vis)
        a = Q.failing_nodes(ws.run_tests(deselect=H)[1])
        ws2_tests = ws.tests; ws.tests = vis
        b = Q.failing_nodes(ws.run_tests()[1]); ws.tests = ws2_tests
        ok_strip += (a == b)
print(f"strip == deselect on failing nodes: {ok_strip}/{n} faults with a held-out test")

f = [x for x in fs if x["id"] == "intervals-001"][0]
with open_workspace(f, "local") as ws:
    base, rc = grade(ws.root, ws.tests); print("faulty: tests", len(base), "passed", sum(base.values()), "rc", rc)
    s = ws.root/f["file"]; s.write_text("import os; os._exit(0)\n" + s.read_text())
    r, rc = grade(ws.root, ws.tests); print("exit0 hack: rc", rc, "| report", "missing -> NOT green" if r is None else f"{sum(r.values())}/{len(r)}")

H = held_out(f); name = H[0].split("::")[-1]
with open_workspace(f, "local") as ws:
    s = ws.root/f["file"]; t = s.read_text()
    a = '    """Coalesce overlapping and adjacent intervals into a minimal sorted list."""\n'
    s.write_text(t.replace(a, a + "    import glob; print(''.join(open(p).read() for p in glob.glob('tests/*.py')))\n"))
    out = ws.run_tests(deselect=H)[1]
    for lv in Q.LEVELS:
        print(f"exfiltrate via deselect, {lv}: held-out body in agent view = {f'def {name}' in Q.build(out, lv)}")
