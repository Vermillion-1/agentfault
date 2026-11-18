"""Spike 2026-11-18: which NVIDIA-hosted models can serve as transcript reviewers."""
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
from retrieval import providers
from runner import judge as J

CANDIDATES = ["moonshotai/kimi-k3.1-pro", "nvidia/nemotron-3-ultra-550b-a55b",
              "mistralai/mistral-large-4-1t-a52b", "z-ai/glm-5.3-flash",
              "minimaxai/minimax-m3.1-flash"]
root = Path(sys.argv[1])
cases = {"cheat": root / "intervals-001__exit_zero", "honest": root / "intervals-001__fix"}
os.environ.setdefault("AGENTFAULT_CHAT_TIMEOUT", "300")
providers.configure("nvidia")
texts = {k: J.transcript(v) for k, v in cases.items()}
jobs = [(m, k) for m in CANDIDATES for k in texts]
with ThreadPoolExecutor(max_workers=5) as pool:
    got = dict(zip(jobs, pool.map(lambda j: J.review(texts[j[1]], j[0]), jobs)))
rows = []
for m in CANDIDATES:
    c, h = got[(m, "cheat")], got[(m, "honest")]
    ok = all("cheating_confidence" in r for r in (c, h))
    passed = ok and c["cheating_confidence"] >= J.FLAG_AT > h["cheating_confidence"]
    secs = [r["seconds"] for r in (c, h) if "seconds" in r]
    rows.append({"model": m, "cheat": c.get("cheating_confidence", c.get("error")),
                 "honest": h.get("cheating_confidence", h.get("error")),
                 "median_s": statistics.median(secs) if secs else None, "passed": passed})
    print(rows[-1])
(root / "reviewer_probe.json").write_text(json.dumps({"rows": rows, "raw": {f"{m}|{k}": v for (m, k), v in got.items()}}, indent=2))
