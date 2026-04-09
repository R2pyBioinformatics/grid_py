"""Tests for grid_py._utils -- utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from grid_py._utils import (
    _auto_name_counter,
    _is_finite,
    _recycle,
    depth,
    explode,
    grid_pretty,
    n2mfrow,
)


# ---------------------------------------------------------------------------
# _auto_name_counter
# ---------------------------------------------------------------------------

class TestAutoNameCounter:

    def test_yields_sequential_names(self):
        gen = _auto_name_counter("TEST")
        assert next(gen) == "TEST.1"
        assert next(gen) == "TEST.2"
        assert next(gen) == "TEST.3"

    def test_default_prefix(self):
        gen = _auto_name_counter()
        assert next(gen) == "GRID.1"


# ---------------------------------------------------------------------------
# _recycle
# ---------------------------------------------------------------------------

class TestRecycle:

    def test_recycle_longer(self):
        result = _recycle([1, 2, 3], 7)
        assert list(result) == [1, 2, 3, 1, 2, 3, 1]

    def test_recycle_shorter(self):
        result = _recycle([1, 2, 3, 4, 5], 3)
        assert list(result) == [1, 2, 3]

    def test_recycle_same_length(self):
        result = _recycle([10, 20], 2)
        assert list(result) == [10, 20]

    def test_recycle_empty_input(self):
        result = _recycle([], 5)
        assert len(result) == 0

    def test_recycle_zero_length(self):
        result = _recycle([1, 2], 0)
        assert len(result) == 0

    def test_recycle_numpy_array(self):
        result = _recycle(np.array([10, 20, 30]), 5)
        assert list(result) == [10, 20, 30, 10, 20]


# ---------------------------------------------------------------------------
# _is_finite
# ---------------------------------------------------------------------------

class TestIsFinite:

    def test_none_is_not_finite(self):
        assert _is_finite(None) is False

    def test_finite_scalar(self):
        assert _is_finite(1.0) == True

    def test_inf_not_finite(self):
        assert _is_finite(float("inf")) == False

    def test_nan_not_finite(self):
        assert _is_finite(float("nan")) == False

    def test_array(self):
        result = _is_finite([1.0, float("inf"), float("nan"), 2.0])
        expected = [True, False, False, True]
        np.testing.assert_array_equal(result, expected)

    def test_non_numeric_dtype(self):
        result = _is_finite("hello")
        assert result == False

    def test_non_numeric_array(self):
        result = _is_finite(["a", "b"])
        np.testing.assert_array_equal(result, [False, False])


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------

class TestDepth:

    def test_string_single(self):
        assert depth("ROOT") == 1

    def test_string_multi(self):
        assert depth("A::B::C") == 3

    def test_dict_with_n(self):
        assert depth({"n": 5}) == 5

    def test_object_with_n_attr(self):
        class FakePath:
            n = 4
        assert depth(FakePath()) == 4

    def test_object_with_depth_method(self):
        class FakePath:
            def depth(self):
                return 7
        assert depth(FakePath()) == 7

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Cannot compute depth"):
            depth(42)


# ---------------------------------------------------------------------------
# explode
# ---------------------------------------------------------------------------

class TestExplode:

    def test_string_single(self):
        assert explode("ROOT") == ["ROOT"]

    def test_string_multi(self):
        assert explode("A::B::C") == ["A", "B", "C"]

    def test_dict_single(self):
        result = explode({"n": 1, "name": "leaf"})
        assert result == ["leaf"]

    def test_dict_multi(self):
        result = explode({"n": 2, "path": "parent", "name": "child"})
        assert result == ["parent", "child"]

    def test_object_with_explode_method(self):
        class FakePath:
            def explode(self):
                return ["x", "y"]
        assert explode(FakePath()) == ["x", "y"]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Cannot explode"):
            explode(42)


# ---------------------------------------------------------------------------
# grid_pretty
# ---------------------------------------------------------------------------

class TestGridPretty:

    def test_basic(self):
        ticks = grid_pretty([0.0, 1.0])
        assert len(ticks) > 0
        assert ticks[0] <= 0.0
        assert ticks[-1] >= 1.0

    def test_negative(self):
        ticks = grid_pretty([-10.0, 10.0])
        assert ticks[0] <= -10.0
        assert ticks[-1] >= 10.0

    def test_single_point(self):
        ticks = grid_pretty([5.0, 5.0])
        assert len(ticks) == 1
        assert ticks[0] == 5.0

    def test_custom_n(self):
        ticks = grid_pretty([0.0, 100.0], n=10)
        assert len(ticks) > 0

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="exactly two"):
            grid_pretty([1.0, 2.0, 3.0])

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            grid_pretty([0.0, float("inf")])


# ---------------------------------------------------------------------------
# n2mfrow
# ---------------------------------------------------------------------------

class TestN2mfrow:

    def test_zero(self):
        assert n2mfrow(0) == (0, 0)

    def test_one(self):
        assert n2mfrow(1) == (1, 1)

    def test_two(self):
        assert n2mfrow(2) == (2, 1)

    def test_three(self):
        assert n2mfrow(3) == (3, 1)

    def test_four(self):
        assert n2mfrow(4) == (2, 2)

    def test_five(self):
        assert n2mfrow(5) == (3, 2)

    def test_six(self):
        assert n2mfrow(6) == (3, 2)

    def test_seven(self):
        r, c = n2mfrow(7)
        assert r * c >= 7

    def test_twelve(self):
        r, c = n2mfrow(12)
        assert r * c >= 12
        assert c == 3

    def test_large(self):
        r, c = n2mfrow(25)
        assert r * c >= 25

    def test_negative(self):
        assert n2mfrow(-1) == (0, 0)
