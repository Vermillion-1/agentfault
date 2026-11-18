"""Spike 2026-11-18: what does the docker backend cost per tool call and per test run?

    python3 spikes/2026-11-18-docker-latency/probe.py      # from the agentfault root
"""
import statistics
import sys
import time

sys.path.insert(0, ".")
from agent.sandbox import open_workspace     # noqa: E402
from tools.corpus import load_faults          # noqa: E402

f = [x for x in load_faults() if x["id"] == "intervals-001"][0]
for backend in ("local", "docker"):
    with open_workspace(f, backend) as ws:
        t = []
        for _ in range(20):
            t0 = time.perf_counter(); ws.call("read", {"path": f["file"]}); t.append(time.perf_counter() - t0)
        g = []
        for _ in range(5):
            t0 = time.perf_counter(); ws.check(); g.append(time.perf_counter() - t0)
    print(f"{backend:6}  tool call median {statistics.median(t)*1000:6.0f} ms   "
          f"test run median {statistics.median(g):5.2f} s  (max {max(g):.2f})")
