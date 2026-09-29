from datetime import date

import pytest

from ledger import (Entry, parse_entries, validate, balance, running_balance,
                    largest_gap_days, summarise)

BOOK = """\
# opening
day,account,cents,memo
2026-01-01,cash,50000,seed
2026-01-01,equity,-50000,seed
2026-01-05,rent,-120000,jan
2026-01-05,cash,120000,jan
"""


def E(d, acct, cents, memo=""):
    return Entry(date(2026, 1, d), acct, cents, memo)


class TestParse:
    def test_count(self):
        assert len(parse_entries(BOOK)) == 4

    def test_skips_comments(self):
        assert all(e.account for e in parse_entries(BOOK))

    def test_types(self):
        e = parse_entries(BOOK)[0]
        assert e.day == date(2026, 1, 1) and e.cents == 50000

    def test_memo_kept(self):
        assert parse_entries(BOOK)[0].memo == "seed"

    def test_blank_input(self):
        assert parse_entries("day,account,cents,memo\n") == []


class TestValidate:
    def test_balanced_book_is_clean(self):
        assert validate(parse_entries(BOOK)) == []

    def test_empty(self):
        assert validate([]) == ["ledger is empty"]

    def test_unbalanced_detected(self):
        out = validate([E(1, "cash", 100)])
        assert any("does not balance" in p for p in out)

    def test_missing_account_detected(self):
        out = validate([E(1, "", 100), E(1, "x", -100)])
        assert any("no account" in p for p in out)

    def test_zero_move_detected(self):
        out = validate([E(1, "a", 0), E(1, "b", 0)])
        assert any("moves nothing" in p for p in out)


class TestBalance:
    def test_nets_per_account(self):
        assert balance(parse_entries(BOOK)) == {
            "cash": 170000, "equity": -50000, "rent": -120000}

    def test_empty(self):
        assert balance([]) == {}

    def test_accumulates(self):
        assert balance([E(1, "a", 10), E(2, "a", 5)]) == {"a": 15}


class TestRunningBalance:
    def test_cumulative(self):
        got = running_balance([E(1, "a", 10), E(2, "a", 5), E(3, "a", -3)], "a")
        assert got == [10, 15, 12]

    def test_filters_account(self):
        assert running_balance([E(1, "a", 10), E(1, "b", 99)], "a") == [10]

    def test_sorts_by_date(self):
        assert running_balance([E(3, "a", 1), E(1, "a", 10)], "a") == [10, 11]

    def test_absent_account(self):
        assert running_balance([E(1, "a", 10)], "zzz") == []


class TestLargestGap:
    def test_single_entry(self):
        assert largest_gap_days([E(1, "a", 1)]) == 0

    def test_consecutive_days_no_gap(self):
        assert largest_gap_days([E(1, "a", 1), E(2, "a", 1)]) == 0

    def test_one_gap(self):
        assert largest_gap_days([E(1, "a", 1), E(5, "a", 1)]) == 3

    def test_largest_of_several(self):
        got = largest_gap_days([E(1, "a", 1), E(3, "a", 1), E(10, "a", 1)])
        assert got == 6


class TestSummarise:
    def test_empty(self):
        assert summarise([]) == "empty ledger"

    def test_shape(self):
        s = summarise(parse_entries(BOOK))
        assert "4 entries" in s and "3 accounts" in s

    def test_inclusive_span(self):
        assert "5 days" in summarise([E(1, "a", 1), E(5, "a", -1)])
