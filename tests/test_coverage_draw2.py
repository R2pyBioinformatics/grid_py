"""Tests to improve coverage of grid_py._draw (supplementing test_draw.py)."""

import warnings

import numpy as np
import pytest

from grid_py._draw import (
    _merge_gpar,
    _render_grob,
    _unit_to_float,
    _unit_to_array,
    _vp_depth,
    grid_draw,
    grid_locator,
    grid_newpage,
    grid_record,
    grid_delay,
)
from grid_py._gpar import Gpar
from grid_py._grob import GList, GTree, Grob
from grid_py._state import get_state
from grid_py._units import Unit
from grid_py.renderer import CairoRenderer


@pytest.fixture(autouse=True)
def _reset():
    state = get_state()
    state.reset()
    yield
    state.reset()


class TestUnitToFloat:
    """Tests for _unit_to_float."""

    def test_plain_float(self):
        assert _unit_to_float(3.14) == pytest.approx(3.14)

    def test_unit(self):
        u = Unit(2.5, "cm")
        assert _unit_to_float(u) == pytest.approx(2.5)

    def test_int(self):
        assert _unit_to_float(7) == 7.0


class TestUnitToArray:
    """Tests for _unit_to_array."""

    def test_plain_list(self):
        arr = _unit_to_array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_unit(self):
        u = Unit([1, 2, 3], "cm")
        arr = _unit_to_array(u)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_scalar(self):
        arr = _unit_to_array(5.0)
        assert arr.shape == (1,)
        assert arr[0] == 5.0

    def test_numpy_array(self):
        arr = _unit_to_array(np.array([1, 2]))
        np.testing.assert_array_equal(arr, [1.0, 2.0])


class TestMergeGpar:
    """Tests for _merge_gpar."""

    def test_both_none(self):
        gp = _merge_gpar(None, None)
        assert isinstance(gp, Gpar)

    def test_context_none(self):
        grob_gp = Gpar(col="red")
        result = _merge_gpar(None, grob_gp)
        assert result is grob_gp

    def test_grob_none(self):
        ctx_gp = Gpar(col="blue")
        result = _merge_gpar(ctx_gp, None)
        assert result is ctx_gp

    def test_merge(self):
        ctx = Gpar(col="blue", lwd=1)
        grob = Gpar(col="red")
        result = _merge_gpar(ctx, grob)
        assert result.get("col") == "red"


class TestRenderGrob:
    """Tests for _render_grob with various grob types."""

    def _make_renderer(self):
        return CairoRenderer(width=4, height=3, dpi=72)

    def test_none_renderer(self):
        g = Grob(name="test", _grid_class="rect")
        _render_grob(g, None)  # should not raise

    def test_rect(self):
        r = self._make_renderer()
        g = Grob(name="r", _grid_class="rect")
        g.x = 0.5; g.y = 0.5; g.width = 0.4; g.height = 0.4
        g.hjust = 0.5; g.vjust = 0.5
        _render_grob(g, r, gp=Gpar())

    def test_circle(self):
        r = self._make_renderer()
        g = Grob(name="c", _grid_class="circle")
        g.x = 0.5; g.y = 0.5; g.r = 0.3
        _render_grob(g, r, gp=Gpar())

    def test_text(self):
        r = self._make_renderer()
        g = Grob(name="t", _grid_class="text")
        g.x = 0.5; g.y = 0.5; g.label = "hello"
        g.rot = 0; g.hjust = 0.5; g.vjust = 0.5
        _render_grob(g, r, gp=Gpar())

    def test_lines(self):
        r = self._make_renderer()
        g = Grob(name="l", _grid_class="lines")
        g.x = [0.1, 0.9]; g.y = [0.1, 0.9]; g.id = None
        _render_grob(g, r, gp=Gpar())

    def test_polyline_with_id(self):
        r = self._make_renderer()
        g = Grob(name="pl", _grid_class="polyline")
        g.x = [0.1, 0.5, 0.9]; g.y = [0.1, 0.5, 0.9]; g.id = [1, 1, 2]
        _render_grob(g, r, gp=Gpar())

    def test_segments(self):
        r = self._make_renderer()
        g = Grob(name="s", _grid_class="segments")
        g.x0 = [0.1]; g.y0 = [0.1]; g.x1 = [0.9]; g.y1 = [0.9]
        _render_grob(g, r, gp=Gpar())

    def test_polygon(self):
        r = self._make_renderer()
        g = Grob(name="pg", _grid_class="polygon")
        g.x = [0.1, 0.5, 0.9]; g.y = [0.1, 0.9, 0.1]
        _render_grob(g, r, gp=Gpar())

    def test_points(self):
        r = self._make_renderer()
        g = Grob(name="p", _grid_class="points")
        g.x = [0.2, 0.8]; g.y = [0.3, 0.7]
        g.size = 1.0; g.pch = 19
        _render_grob(g, r, gp=Gpar())

    def test_points_pch_list(self):
        r = self._make_renderer()
        g = Grob(name="p", _grid_class="points")
        g.x = [0.5]; g.y = [0.5]
        g.size = 1.0; g.pch = [1, 2]
        _render_grob(g, r, gp=Gpar())

    def test_null_grob(self):
        r = self._make_renderer()
        g = Grob(name="n", _grid_class="null")
        _render_grob(g, r, gp=Gpar())

    def test_move_to(self):
        r = self._make_renderer()
        g = Grob(name="m", _grid_class="move.to")
        g.x = 0.2; g.y = 0.3
        _render_grob(g, r, gp=Gpar())

    def test_line_to(self):
        r = self._make_renderer()
        g = Grob(name="lt", _grid_class="line.to")
        g.x = 0.8; g.y = 0.9
        # Need move_to first
        r.move_to(0.1, 0.1)
        _render_grob(g, r, gp=Gpar())

    def test_pathgrob(self):
        r = self._make_renderer()
        g = Grob(name="path", _grid_class="pathgrob")
        g.x = [0.1, 0.5, 0.9]; g.y = [0.1, 0.9, 0.1]
        g.pathId = None; g.rule = "winding"
        _render_grob(g, r, gp=Gpar())

    def test_pathgrob_with_id(self):
        r = self._make_renderer()
        g = Grob(name="path", _grid_class="pathgrob")
        g.x = [0.1, 0.5, 0.9]; g.y = [0.1, 0.9, 0.1]
        g.pathId = [1, 1, 1]; g.rule = "evenodd"
        _render_grob(g, r, gp=Gpar())

    def test_roundrect(self):
        r = self._make_renderer()
        g = Grob(name="rr", _grid_class="roundrect")
        g.x = 0.5; g.y = 0.5; g.width = 0.4; g.height = 0.3
        g.r = 0.05; g.hjust = 0.5; g.vjust = 0.5
        _render_grob(g, r, gp=Gpar())

    def test_rastergrob(self):
        r = self._make_renderer()
        g = Grob(name="rast", _grid_class="rastergrob")
        g.image = np.zeros((10, 10, 4), dtype=np.uint8)
        g.x = 0.5; g.y = 0.5; g.width = 0.5; g.height = 0.5
        g.interpolate = True
        _render_grob(g, r, gp=Gpar())

    def test_rastergrob_no_image(self):
        r = self._make_renderer()
        g = Grob(name="rast", _grid_class="rastergrob")
        g.image = None
        _render_grob(g, r, gp=Gpar())

    def test_unknown_class_warns(self):
        r = self._make_renderer()
        g = Grob(name="u", _grid_class="unknown_grob_type")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _render_grob(g, r, gp=Gpar())
            assert len(w) >= 1
            assert "unknown grob class" in str(w[0].message)


class TestVpDepth:
    """Tests for _vp_depth."""

    def test_vppath(self):
        from grid_py._path import VpPath
        vp = VpPath("a", "b", "c")
        assert _vp_depth(vp) == 3

    def test_object_with_depth(self):
        class FakeVP:
            def depth(self):
                return 5
        assert _vp_depth(FakeVP()) == 5

    def test_default(self):
        assert _vp_depth("plain_object") == 1


class TestGridDrawAdditional:
    """Additional tests for grid_draw."""

    def test_draw_numeric_is_noop(self):
        grid_newpage()
        grid_draw(42)  # numeric is silently ignored

    def test_draw_unknown_warns(self):
        grid_newpage()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_draw(object(), recording=False)
            assert any("don't know how to draw" in str(x.message) for x in w)


class TestGridLocator:
    """Tests for grid_locator."""

    def test_returns_none_and_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = grid_locator()
            assert result is None
            assert len(w) >= 1
            assert "not supported" in str(w[0].message)


class TestGridRecord:
    """Tests for grid_record."""

    def test_records_and_draws(self):
        grid_newpage()
        called = {"flag": False}

        def my_expr():
            called["flag"] = True

        grid_record(my_expr)
        state = get_state()
        assert len(state.get_display_list()) >= 1


class TestGridDelay:
    """Tests for grid_delay."""

    def test_delays_and_draws(self):
        grid_newpage()

        def my_expr():
            return Grob(name="delayed", _grid_class="null")

        grid_delay(my_expr)
        state = get_state()
        assert len(state.get_display_list()) >= 1
