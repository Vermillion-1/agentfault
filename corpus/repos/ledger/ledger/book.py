"""A tiny double-entry ledger over CSV rows.

Chosen as a second corpus repo because it has the things `intervals` lacks:
module imports used by exactly one function, string/date parsing, dict aggregation,
and an error path -- which is where missing-import and wrong-API faults can live
without taking down the whole suite.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Entry:
    day: date
    account: str
    cents: int          # signed; credits positive, debits negative
    memo: str = ""


def parse_entries(text: str) -> List[Entry]:
    """Read `day,account,cents,memo` rows. Blank lines and comments are skipped."""
    out: List[Entry] = []
    body = "\n".join(ln for ln in text.splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#"))
    for row in csv.DictReader(StringIO(body)):
        y, m, d = (int(p) for p in row["day"].split("-"))
        out.append(Entry(date(y, m, d), row["account"].strip(),
                         int(row["cents"]), (row.get("memo") or "").strip()))
    return out


def validate(entries: Iterable[Entry]) -> List[str]:
    """Return human-readable problems. Empty list means the book is sound."""
    problems: List[str] = []
    items = list(entries)
    if not items:
        return ["ledger is empty"]
    total = sum(e.cents for e in items)
    if total != 0:
        problems.append(f"ledger does not balance: off by {total} cents")
    for e in items:
        if not e.account:
            problems.append(f"entry on {e.day} has no account")
        if e.cents == 0:
            problems.append(f"entry on {e.day} for {e.account} moves nothing")
    return problems


def balance(entries: Iterable[Entry]) -> Dict[str, int]:
    """Net position per account."""
    out: Dict[str, int] = defaultdict(int)
    for e in entries:
        out[e.account] += e.cents
    return dict(out)


def running_balance(entries: Iterable[Entry], account: str) -> List[int]:
    """Cumulative position for one account, in date order."""
    rows = sorted((e for e in entries if e.account == account), key=lambda e: e.day)
    total = 0
    out: List[int] = []
    for e in rows:
        total += e.cents
        out.append(total)
    return out


def largest_gap_days(entries: Iterable[Entry]) -> int:
    """Longest run of days with no activity at all, between first and last entry."""
    days = sorted({e.day for e in entries})
    if len(days) < 2:
        return 0
    return max((days[i + 1] - days[i]).days - 1 for i in range(len(days) - 1))


def summarise(entries: Iterable[Entry]) -> str:
    items = list(entries)
    if not items:
        return "empty ledger"
    span = (max(e.day for e in items) - min(e.day for e in items)) + timedelta(days=1)
    return (f"{len(items)} entries across {len(balance(items))} accounts "
            f"over {span.days} days")
