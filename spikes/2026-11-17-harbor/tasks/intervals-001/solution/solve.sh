#!/bin/bash
cd /app && python - <<'PY'
from pathlib import Path
p=Path('intervals/core.py'); t=p.read_text()
assert t.count('        if cur.start < last.end:')==1
p.write_text(t.replace('        if cur.start < last.end:', '        if cur.start <= last.end:'))
PY
