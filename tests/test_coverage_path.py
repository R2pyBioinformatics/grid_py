"""Tests to improve coverage of grid_py._path."""

import pytest

from grid_py._path import GPath, VpPath, GridPath, as_path, is_closed, PATH_SEP


class TestGPath:
    """Tests for GPath."""

    def test_basic(self):
        p = GPath("a", "b", "c")
        assert str(p) == "a::b::c"
        assert p.name == "c"
        assert p.path == "a::b"
        assert p.n == 3

    def test_single_component(self):
        p = GPath("leaf")
        assert p.name == "leaf"
        assert p.path is None
        assert p.n == 1

    def test_auto_split(self):
        p = GPath("a::b", "c")
        assert p.components == ("a", "b", "c")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            GPath()

    def test_non_string_raises(self):
        with pytest.raises(TypeError, match="expected str"):
            GPath(123)

    def test_empty_string_component_raises(self):
        with pytest.raises(ValueError, match="empty string"):
            GPath("a", "", "c")

    def test_repr(self):
        p = GPath("x", "y")
        assert "GPath" in repr(p)

    def test_eq(self):
        p1 = GPath("a", "b")
        p2 = GPath("a", "b")
        assert p1 == p2

    def test_neq(self):
        p1 = GPath("a", "b")
        p2 = GPath("a", "c")
        assert p1 != p2

    def test_eq_not_implemented(self):
        p = GPath("a")
        assert p.__eq__("not_gpath") is NotImplemented

    def test_hash(self):
        p1 = GPath("a", "b")
        p2 = GPath("a", "b")
        assert hash(p1) == hash(p2)

    def test_len(self):
        p = GPath("a", "b", "c")
        assert len(p) == 3

    def test_components_property(self):
        p = GPath("x", "y", "z")
        assert p.components == ("x", "y", "z")


class TestVpPath:
    """Tests for VpPath."""

    def test_basic(self):
        vp = VpPath("root", "panel", "strip")
        assert str(vp) == "root::panel::strip"
        assert vp.name == "strip"
        assert vp.path == "root::panel"
        assert vp.n == 3

    def test_single_component(self):
        vp = VpPath("root")
        assert vp.name == "root"
        assert vp.path is None

    def test_auto_split(self):
        vp = VpPath("a::b", "c")
        assert vp.components == ("a", "b", "c")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            VpPath()

    def test_non_string_raises(self):
        with pytest.raises(TypeError, match="expected str"):
            VpPath(123)

    def test_empty_component_raises(self):
        with pytest.raises(ValueError, match="empty string"):
            VpPath("a", "")

    def test_getitem_int(self):
        vp = VpPath("a", "b", "c")
        sub = vp[1]
        assert str(sub) == "b"

    def test_getitem_slice(self):
        vp = VpPath("a", "b", "c")
        sub = vp[0:2]
        assert str(sub) == "a::b"

    def test_getitem_empty_raises(self):
        vp = VpPath("a", "b")
        with pytest.raises(IndexError, match="empty"):
            vp[5:5]

    def test_repr(self):
        vp = VpPath("x", "y")
        assert "VpPath" in repr(vp)

    def test_eq(self):
        vp1 = VpPath("a", "b")
        vp2 = VpPath("a", "b")
        assert vp1 == vp2

    def test_neq(self):
        vp1 = VpPath("a", "b")
        vp2 = VpPath("a", "c")
        assert vp1 != vp2

    def test_eq_not_implemented(self):
        vp = VpPath("a")
        assert vp.__eq__("not_vppath") is NotImplemented

    def test_hash(self):
        vp1 = VpPath("a", "b")
        vp2 = VpPath("a", "b")
        assert hash(vp1) == hash(vp2)

    def test_len(self):
        vp = VpPath("a", "b", "c")
        assert len(vp) == 3


class TestGridPath:
    """Tests for GridPath."""

    def test_winding_rule(self):
        gp = GridPath("grob", rule="winding")
        assert gp.rule == "winding"
        assert gp.grob == "grob"
        assert gp.gp is None

    def test_evenodd_rule(self):
        gp = GridPath("grob", rule="evenodd")
        assert gp.rule == "evenodd"

    def test_invalid_rule(self):
        with pytest.raises(ValueError, match="must be one of"):
            GridPath("grob", rule="invalid")

    def test_with_gp(self):
        gp = GridPath("grob", gp="some_gpar")
        assert gp.gp == "some_gpar"

    def test_repr(self):
        gp = GridPath("grob")
        assert "GridPath" in repr(gp)


class TestAsPath:
    """Tests for as_path factory."""

    def test_basic(self):
        gp = as_path("grob")
        assert isinstance(gp, GridPath)
        assert gp.grob == "grob"

    def test_none_raises(self):
        with pytest.raises(TypeError, match="only a grob"):
            as_path(None)

    def test_with_rule(self):
        gp = as_path("grob", rule="evenodd")
        assert gp.rule == "evenodd"

    def test_with_gp(self):
        gp = as_path("grob", gp="my_gpar")
        assert gp.gp == "my_gpar"


class TestIsClosed:
    """Tests for is_closed."""

    def test_open_types(self):
        class FakeGrob:
            pass
        for open_type in ("move.to", "line.to", "lines", "polyline", "segments", "beziergrob"):
            g = FakeGrob()
            g._grid_class = open_type
            assert is_closed(g) is False

    def test_closed_types(self):
        class FakeGrob:
            pass
        for closed_type in ("rect", "circle", "polygon"):
            g = FakeGrob()
            g._grid_class = closed_type
            assert is_closed(g) is True

    def test_default_is_closed(self):
        # Object without _grid_class uses class name
        class SomeClass:
            pass
        assert is_closed(SomeClass()) is True

    def test_unknown_type_is_closed(self):
        class FakeGrob:
            _grid_class = "unknown_type"
        assert is_closed(FakeGrob()) is True
