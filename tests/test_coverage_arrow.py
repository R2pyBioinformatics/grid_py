"""Tests for grid_py._arrow -- Arrow specification."""

from __future__ import annotations

import numpy as np
import pytest

from grid_py._arrow import Arrow, arrow, _recycle_unit
from grid_py._units import Unit


# ---------------------------------------------------------------------------
# Arrow construction
# ---------------------------------------------------------------------------

class TestArrowConstruction:

    def test_defaults(self):
        a = Arrow()
        np.testing.assert_array_equal(a.angle, [30.0])
        assert len(a.length) == 1
        np.testing.assert_array_equal(a.ends, [2])  # "last" -> 2
        np.testing.assert_array_equal(a.type, [1])  # "open" -> 1

    def test_custom_angle(self):
        a = Arrow(angle=45)
        np.testing.assert_array_equal(a.angle, [45.0])

    def test_angle_sequence(self):
        a = Arrow(angle=[10, 20, 30])
        np.testing.assert_array_equal(a.angle, [10.0, 20.0, 30.0])

    def test_custom_length(self):
        u = Unit(0.5, "inches")
        a = Arrow(length=u)
        assert a.length is u

    def test_invalid_length_raises(self):
        with pytest.raises(TypeError, match="Unit"):
            Arrow(length=0.5)

    def test_ends_first(self):
        a = Arrow(ends="first")
        np.testing.assert_array_equal(a.ends, [1])

    def test_ends_both(self):
        a = Arrow(ends="both")
        np.testing.assert_array_equal(a.ends, [3])

    def test_ends_sequence(self):
        a = Arrow(ends=["first", "last"])
        np.testing.assert_array_equal(a.ends, [1, 2])

    def test_invalid_ends_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            Arrow(ends="middle")

    def test_type_closed(self):
        a = Arrow(type="closed")
        np.testing.assert_array_equal(a.type, [2])

    def test_type_sequence(self):
        a = Arrow(type=["open", "closed"])
        np.testing.assert_array_equal(a.type, [1, 2])

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            Arrow(type="half")

    def test_numpy_integer_angle(self):
        a = Arrow(angle=np.int64(45))
        np.testing.assert_array_equal(a.angle, [45.0])


# ---------------------------------------------------------------------------
# Arrow length (__len__)
# ---------------------------------------------------------------------------

class TestArrowLen:

    def test_scalar(self):
        a = Arrow()
        assert len(a) == 1

    def test_vector(self):
        a = Arrow(angle=[10, 20, 30], ends=["first", "last"])
        assert len(a) == 3


# ---------------------------------------------------------------------------
# Arrow subscript (__getitem__)
# ---------------------------------------------------------------------------

class TestArrowGetItem:

    def test_integer_index(self):
        a = Arrow(angle=[10, 20, 30])
        sub = a[1]
        np.testing.assert_array_equal(sub.angle, [20.0])
        assert len(sub) == 1

    def test_slice(self):
        a = Arrow(angle=[10, 20, 30])
        sub = a[0:2]
        np.testing.assert_array_equal(sub.angle, [10.0, 20.0])

    def test_list_index(self):
        a = Arrow(angle=[10, 20, 30])
        sub = a[[0, 2]]
        np.testing.assert_array_equal(sub.angle, [10.0, 30.0])


# ---------------------------------------------------------------------------
# Arrow rep
# ---------------------------------------------------------------------------

class TestArrowRep:

    def test_rep_times(self):
        a = Arrow(angle=45)
        r = a.rep(times=3)
        assert len(r) == 3
        np.testing.assert_array_equal(r.angle, [45.0, 45.0, 45.0])

    def test_rep_length_out(self):
        a = Arrow(angle=[10, 20])
        r = a.rep(length_out=5)
        assert len(r) == 5

    def test_rep_default(self):
        a = Arrow()
        r = a.rep()
        assert len(r) == 1


# ---------------------------------------------------------------------------
# Arrow repr
# ---------------------------------------------------------------------------

class TestArrowRepr:

    def test_repr(self):
        a = Arrow()
        s = repr(a)
        assert "Arrow" in s
        assert "angle" in s
        assert "length" in s
        assert "ends" in s
        assert "type" in s


# ---------------------------------------------------------------------------
# arrow factory
# ---------------------------------------------------------------------------

class TestArrowFactory:

    def test_defaults(self):
        a = arrow()
        assert isinstance(a, Arrow)
        np.testing.assert_array_equal(a.angle, [30.0])

    def test_custom(self):
        a = arrow(angle=45, type="closed")
        np.testing.assert_array_equal(a.angle, [45.0])
        np.testing.assert_array_equal(a.type, [2])


# ---------------------------------------------------------------------------
# _recycle_unit
# ---------------------------------------------------------------------------

class TestRecycleUnit:

    def test_same_length(self):
        u = Unit([1, 2, 3], "cm")
        result = _recycle_unit(u, 3)
        assert len(result) == 3

    def test_longer(self):
        u = Unit([1, 2], "cm")
        result = _recycle_unit(u, 5)
        assert len(result) == 5
