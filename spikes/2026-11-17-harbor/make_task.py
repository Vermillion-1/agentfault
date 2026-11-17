#!/usr/bin/env python3
"""Turn one agentfault fault into a sealed Harbor task (spike, 2026-11-17).

    cd <research checkout>                     # needs agent/sandbox.py (moves to
    python3 ../agentfault/spikes/2026-11-17-harbor/make_task.py intervals-001 OUT_DIR   # public at D1)

OUT_DIR must already hold a scaffold from `harbor task init`. The script fills it in:

  environment/   the faulty source only; tests are moved out (agent/sandbox.py: _split)
  instruction.md the L3 failure text from the VISIBLE tests (held-out deselected)
  tests/         every pristine test, plus test.sh, which writes reward.json with
                 `fixed` (all tests, held-out included) and `visible_pass`
  solution/      the reference fix, for Harbor's `oracle` agent
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
from agent import sandbox                     # noqa: E402
from retrieval import query as Q              # noqa: E402
from tools.corpus import load_faults          # noqa: E402

fault_id, T = sys.argv[1], Path(sys.argv[2])
f = [x for x in load_faults() if x["id"] == fault_id][0]
held = sandbox.held_out(f)

with tempfile.TemporaryDirectory() as td:
    ws, tests = sandbox._split(f, Path(td))
    green, out = sandbox.Workspace(f, ws, tests).run_tests(deselect=held)
    shutil.rmtree(T / "environment" / "app", ignore_errors=True)
    shutil.copytree(ws, T / "environment" / "app")
    for p in (T / "tests").glob("*"):
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    shutil.copytree(tests, T / "tests" / "suite")

(T / "environment" / "Dockerfile").write_text(
    "FROM python:3.11.7-slim\nRUN pip install --no-cache-dir pytest==7.4.0\n"
    "COPY app/ /app/\nWORKDIR /app\n")
(T / "instruction.md").write_text(f"""You are debugging a small Python repository in /app whose test suite is failing.
The test files are not available to you. Exactly one deliberate fault has been introduced in the source. Find it and fix it with the smallest change.

Test run:
{Q.build(out, "L3")}

When you are finished, end with a line that says DONE, or GIVE UP if you cannot find the fault.
""")
deselect = " ".join(f"--deselect '{h}'" for h in held)
(T / "tests" / "test.sh").write_text(f"""#!/bin/bash
# agentfault verifier: pristine tests (incl. the held-out one) on the agent's final /app.
cp -r /tests/suite/. /app/
cd /app
python -m pytest -q -p no:cacheprovider > /logs/verifier/all.txt 2>&1; ALL=$?
python -m pytest -q -p no:cacheprovider {deselect} > /logs/verifier/visible.txt 2>&1; VIS=$?
echo "{{\\"fixed\\": $([ $ALL -eq 0 ] && echo 1 || echo 0), \\"visible_pass\\": $([ $VIS -eq 0 ] && echo 1 || echo 0)}}" > /logs/verifier/reward.json
""")
(T / "solution" / "solve.sh").write_text(f"""#!/bin/bash
cd /app && python - <<'PY'
from pathlib import Path
p = Path({f['file']!r}); t = p.read_text()
assert t.count({f['replacement']!r}) == 1
p.write_text(t.replace({f['replacement']!r}, {f['anchor']!r}))
PY
""")
print("held out:", held, "| visible suite green on faulty code:", green)
