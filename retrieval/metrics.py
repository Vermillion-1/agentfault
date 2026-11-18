"""Ranking metrics, scored against the fault label."""
from __future__ import annotations

from typing import Dict, List, Sequence

KS = (1, 3, 5, 10)


def _rank_of(ranking: Sequence[str], targets: set) -> int:
    """1-based rank of the first hit, or 0 when absent."""
    for i, uid in enumerate(ranking, start=1):
        if uid in targets:
            return i
    return 0


def score(ranking: Sequence[str], gold_fn_uid: str, gold_file: str) -> Dict[str, float]:
    fn_targets = {gold_fn_uid}
    file_targets = {uid for uid in ranking if uid.split("::")[0] == gold_file}

    r_fn = _rank_of(ranking, fn_targets)
    r_file = _rank_of(ranking, file_targets)

    out: Dict[str, float] = {
        "rank_fn": r_fn,
        "rank_file": r_file,
        "mrr_fn": 1.0 / r_fn if r_fn else 0.0,
        "mrr_file": 1.0 / r_file if r_file else 0.0,
        "n_candidates": len(ranking),
    }
    for k in KS:
        out[f"recall_fn@{k}"] = 1.0 if r_fn and r_fn <= k else 0.0
        out[f"recall_file@{k}"] = 1.0 if r_file and r_file <= k else 0.0
        hits = sum(1 for uid in ranking[:k] if uid in file_targets)
        out[f"precision_file@{k}"] = hits / k
    return out


def bootstrap_ci(values: List[float], iters: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap 95% CI. Non-negotiable per SPEC statistical requirements."""
    import random
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def paired_bootstrap_ci(a: List[float], b: List[float], iters: int = 10000,
                        seed: int = 0, alpha: float = 0.05):
    """95% CI for mean(a - b), where a[i] and b[i] come from the SAME item."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    return bootstrap_ci([x - y for x, y in zip(a, b)], iters=iters, seed=seed, alpha=alpha)
