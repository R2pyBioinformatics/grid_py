"""Tests to improve coverage of grid_py._mask."""

import pytest

from grid_py._mask import GridMask, as_mask, is_mask


class TestGridMask:
    """Tests for the GridMask class."""

    def test_init_alpha(self):
        m = GridMask("grob", type="alpha")
        assert m.ref == "grob"
        assert m.type == "alpha"

    def test_init_luminance(self):
        m = GridMask("grob", type="luminance")
        assert m.type == "luminance"

    def test_init_invalid_type(self):
        with pytest.raises(ValueError, match="must be one of"):
            GridMask("grob", type="invalid")

    def test_repr(self):
        m = GridMask("g", type="alpha")
        r = repr(m)
        assert "GridMask" in r
        assert "alpha" in r
        assert "'g'" in r

    def test_eq_same(self):
        m1 = GridMask("a", type="alpha")
        m2 = GridMask("a", type="alpha")
        assert m1 == m2

    def test_eq_different_ref(self):
        m1 = GridMask("a", type="alpha")
        m2 = GridMask("b", type="alpha")
        assert m1 != m2

    def test_eq_different_type(self):
        m1 = GridMask("a", type="alpha")
        m2 = GridMask("a", type="luminance")
        assert m1 != m2

    def test_eq_not_implemented(self):
        m = GridMask("a")
        assert m.__eq__("not a mask") is NotImplemented

    def test_hash(self):
        m = GridMask("x")
        assert isinstance(hash(m), int)

    def test_bool_ref_true(self):
        m = GridMask(True, type="alpha")
        assert m.ref is True

    def test_bool_ref_false(self):
        m = GridMask(False, type="alpha")
        assert m.ref is False


class TestAsMask:
    """Tests for as_mask factory."""

    def test_wraps_grob(self):
        m = as_mask("grob")
        assert isinstance(m, GridMask)
        assert m.ref == "grob"
        assert m.type == "alpha"

    def test_luminance_type(self):
        m = as_mask("grob", type="luminance")
        assert m.type == "luminance"

    def test_none_raises(self):
        with pytest.raises(TypeError, match="only a grob"):
            as_mask(None)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            as_mask("grob", type="invalid")

    def test_bool_true(self):
        m = as_mask(True)
        assert m.ref is True

    def test_bool_false(self):
        m = as_mask(False)
        assert m.ref is False


class TestIsMask:
    """Tests for is_mask predicate."""

    def test_true_for_mask(self):
        assert is_mask(GridMask("g")) is True

    def test_false_for_string(self):
        assert is_mask("not_mask") is False

    def test_false_for_none(self):
        assert is_mask(None) is False

    def test_false_for_int(self):
        assert is_mask(42) is False
