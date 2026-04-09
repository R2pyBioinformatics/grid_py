"""Tests to improve coverage of grid_py._clippath."""

import pytest

from grid_py._clippath import GridClipPath, as_clip_path, is_clip_path


class TestGridClipPath:
    """Tests for the GridClipPath class."""

    def test_init_stores_grob(self):
        cp = GridClipPath("my_grob")
        assert cp.grob == "my_grob"

    def test_init_none_raises(self):
        with pytest.raises(TypeError, match="non-None grob"):
            GridClipPath(None)

    def test_repr(self):
        cp = GridClipPath("g")
        r = repr(cp)
        assert "GridClipPath" in r
        assert "'g'" in r

    def test_eq_same(self):
        cp1 = GridClipPath("a")
        cp2 = GridClipPath("a")
        assert cp1 == cp2

    def test_eq_different(self):
        cp1 = GridClipPath("a")
        cp2 = GridClipPath("b")
        assert cp1 != cp2

    def test_eq_not_implemented(self):
        cp = GridClipPath("a")
        assert cp.__eq__("not a clip path") is NotImplemented

    def test_hash(self):
        cp = GridClipPath("x")
        assert isinstance(hash(cp), int)


class TestAsClipPath:
    """Tests for as_clip_path factory."""

    def test_wraps_grob(self):
        cp = as_clip_path("some_grob")
        assert isinstance(cp, GridClipPath)
        assert cp.grob == "some_grob"

    def test_passthrough_existing(self):
        cp = GridClipPath("g")
        cp2 = as_clip_path(cp)
        assert cp2 is cp

    def test_none_raises(self):
        with pytest.raises(TypeError, match="only a grob"):
            as_clip_path(None)

    def test_with_int_grob(self):
        cp = as_clip_path(42)
        assert cp.grob == 42


class TestIsClipPath:
    """Tests for is_clip_path predicate."""

    def test_true_for_clip_path(self):
        assert is_clip_path(GridClipPath("g")) is True

    def test_false_for_string(self):
        assert is_clip_path("not_cp") is False

    def test_false_for_none(self):
        assert is_clip_path(None) is False

    def test_false_for_int(self):
        assert is_clip_path(42) is False
