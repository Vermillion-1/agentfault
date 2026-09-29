"""Half-open interval arithmetic: [start, end).

Half-open is the whole reason this module is interesting to inject faults into --
almost every boundary decision here has a plausible-looking wrong version.
"""
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True, order=True)
class Interval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start {self.start} exceeds end {self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def is_empty(self) -> bool:
        return self.start == self.end

    def overlaps(self, other: "Interval") -> bool:
        # Half-open: [0,5) and [5,9) touch but do not overlap.
        return self.start < other.end and other.start < self.end

    def touches(self, other: "Interval") -> bool:
        return self.overlaps(other) or self.end == other.start or other.end == self.start


def merge(intervals: Iterable[Interval]) -> List[Interval]:
    """Coalesce overlapping and adjacent intervals into a minimal sorted list."""
    items = sorted(i for i in intervals if not i.is_empty())
    if not items:
        return []

    out = [items[0]]
    for cur in items[1:]:
        last = out[-1]
        if cur.start <= last.end:
            out[-1] = Interval(last.start, max(last.end, cur.end))
        else:
            out.append(cur)
    return out


def intersect(a: Iterable[Interval], b: Iterable[Interval]) -> List[Interval]:
    """Pairwise intersection of two interval sets, each merged first."""
    left, right = merge(a), merge(b)
    out: List[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        lo = max(left[i].start, right[j].start)
        hi = min(left[i].end, right[j].end)
        if lo < hi:
            out.append(Interval(lo, hi))
        if left[i].end < right[j].end:
            i += 1
        else:
            j += 1
    return out


def subtract(a: Iterable[Interval], b: Iterable[Interval]) -> List[Interval]:
    """Everything in `a` that is not covered by `b`."""
    out: List[Interval] = []
    holes = merge(b)
    for iv in merge(a):
        cursor = iv.start
        for hole in holes:
            if hole.end <= cursor or hole.start >= iv.end:
                continue
            if hole.start > cursor:
                out.append(Interval(cursor, hole.start))
            cursor = max(cursor, hole.end)
        if cursor < iv.end:
            out.append(Interval(cursor, iv.end))
    return out


def total_length(intervals: Iterable[Interval]) -> int:
    """Measure of the union -- double-counted overlap is the classic bug here."""
    return sum(iv.length for iv in merge(intervals))


def find_gaps(intervals: Iterable[Interval]) -> List[Interval]:
    """Uncovered space strictly between the first and last covered point."""
    merged = merge(intervals)
    return [Interval(merged[k].end, merged[k + 1].start) for k in range(len(merged) - 1)]
