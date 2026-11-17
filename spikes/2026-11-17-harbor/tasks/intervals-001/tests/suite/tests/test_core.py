import pytest

from intervals import Interval, merge, intersect, subtract, total_length, find_gaps


def I(a, b):
    return Interval(a, b)


class TestInterval:
    def test_length(self):
        assert I(2, 7).length == 5

    def test_empty(self):
        assert I(3, 3).is_empty()
        assert not I(3, 4).is_empty()

    def test_rejects_inverted(self):
        with pytest.raises(ValueError):
            I(9, 2)

    def test_adjacent_do_not_overlap(self):
        # Half-open: [0,5) and [5,9) share no point.
        assert not I(0, 5).overlaps(I(5, 9))

    def test_real_overlap(self):
        assert I(0, 6).overlaps(I(5, 9))

    def test_touching(self):
        assert I(0, 5).touches(I(5, 9))
        assert not I(0, 5).touches(I(6, 9))


class TestMerge:
    def test_empty(self):
        assert merge([]) == []

    def test_disjoint_preserved(self):
        assert merge([I(0, 2), I(5, 7)]) == [I(0, 2), I(5, 7)]

    def test_overlapping_coalesce(self):
        assert merge([I(0, 5), I(3, 9)]) == [I(0, 9)]

    def test_adjacent_coalesce(self):
        assert merge([I(0, 5), I(5, 9)]) == [I(0, 9)]

    def test_unsorted_input(self):
        assert merge([I(5, 9), I(0, 5)]) == [I(0, 9)]

    def test_nested_absorbed(self):
        assert merge([I(0, 10), I(3, 4)]) == [I(0, 10)]

    def test_empties_dropped(self):
        assert merge([I(2, 2), I(4, 6)]) == [I(4, 6)]


class TestIntersect:
    def test_disjoint(self):
        assert intersect([I(0, 3)], [I(5, 8)]) == []

    def test_partial(self):
        assert intersect([I(0, 6)], [I(4, 9)]) == [I(4, 6)]

    def test_adjacent_yields_nothing(self):
        assert intersect([I(0, 5)], [I(5, 9)]) == []

    def test_multiple(self):
        got = intersect([I(0, 5), I(8, 12)], [I(3, 10)])
        assert got == [I(3, 5), I(8, 10)]


class TestSubtract:
    def test_no_overlap(self):
        assert subtract([I(0, 5)], [I(7, 9)]) == [I(0, 5)]

    def test_punch_hole(self):
        assert subtract([I(0, 10)], [I(3, 6)]) == [I(0, 3), I(6, 10)]

    def test_trim_front(self):
        assert subtract([I(0, 10)], [I(0, 4)]) == [I(4, 10)]

    def test_trim_back(self):
        assert subtract([I(0, 10)], [I(6, 20)]) == [I(0, 6)]

    def test_fully_removed(self):
        assert subtract([I(2, 5)], [I(0, 9)]) == []


class TestTotalLength:
    def test_disjoint_sums(self):
        assert total_length([I(0, 3), I(5, 9)]) == 7

    def test_overlap_not_double_counted(self):
        assert total_length([I(0, 6), I(4, 10)]) == 10

    def test_nested_not_double_counted(self):
        assert total_length([I(0, 10), I(2, 3)]) == 10


class TestFindGaps:
    def test_single_has_no_gap(self):
        assert find_gaps([I(0, 5)]) == []

    def test_one_gap(self):
        assert find_gaps([I(0, 3), I(7, 9)]) == [I(3, 7)]

    def test_adjacent_has_no_gap(self):
        assert find_gaps([I(0, 5), I(5, 9)]) == []

    def test_two_gaps(self):
        assert find_gaps([I(0, 2), I(4, 6), I(9, 11)]) == [I(2, 4), I(6, 9)]
