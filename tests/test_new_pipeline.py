"""Tests for new pipeline functions added during R grid alignment.

Covers:
- _vp_calc.py inverse transforms (7 functions)
- _renderer_base.py evaluateGrobUnit pipeline (3 functions)
- _units.py convert_theta, summary, device_loc, device_dim
- _patterns.py resolve pipeline
- _coords.py per-primitive grob_points
- _size.py rewritten widthDetails/heightDetails
- _group.py draw_details with real renderer
- _grob.py grep/global name matching
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from grid_py._units import (
    Unit,
    convert_theta,
    convert_unit,
    convert_x,
    convert_y,
    convert_width,
    convert_height,
    device_loc,
    device_dim,
    unit_summary_min,
    unit_summary_max,
    unit_summary_sum,
    unit_rep,
    unit_type,
    unit_c,
)
from grid_py._vp_calc import (
    _transform_from_inches,
    _transform_xy_from_inches,
    _transform_wh_from_inches,
    _transform_xy_to_npc,
    _transform_wh_to_npc,
    _transform_xy_from_npc,
    _transform_wh_from_npc,
)


# ===================================================================
# _vp_calc.py inverse transforms
# ===================================================================


class TestTransformFromInches:
    """Port of R unit.c:1226-1333 transformFromINCHES."""

    def test_inches_identity(self):
        assert _transform_from_inches(2.0, "inches", 12, 1, 1.2, 10, 8) == 2.0

    def test_cm(self):
        assert abs(_transform_from_inches(1.0, "cm", 12, 1, 1.2, 10, 8) - 2.54) < 1e-10

    def test_mm(self):
        assert abs(_transform_from_inches(1.0, "mm", 12, 1, 1.2, 10, 8) - 25.4) < 1e-10

    def test_points(self):
        assert abs(_transform_from_inches(1.0, "points", 12, 1, 1.2, 10, 8) - 72.27) < 1e-6

    def test_bigpts(self):
        assert abs(_transform_from_inches(1.0, "bigpts", 12, 1, 1.2, 10, 8) - 72.0) < 1e-10

    def test_npc(self):
        # 1 inch in a 5.08cm (= 2 inch) wide viewport = 0.5 NPC
        result = _transform_from_inches(1.0, "npc", 12, 1, 1.2, 5.08, 5.08)
        assert abs(result - 0.5) < 1e-6

    def test_char(self):
        # (1 * 72) / (12 * 1) = 6.0
        result = _transform_from_inches(1.0, "char", 12, 1, 1.2, 10, 8)
        assert abs(result - 6.0) < 1e-10

    def test_lines(self):
        # (1 * 72) / (12 * 1 * 1.2) = 5.0
        result = _transform_from_inches(1.0, "lines", 12, 1, 1.2, 10, 8)
        assert abs(result - 5.0) < 1e-10

    def test_snpc(self):
        # min(10, 8) cm = 8cm = 3.1496 inches → 1 inch = 1/3.1496 snpc
        result = _transform_from_inches(1.0, "snpc", 12, 1, 1.2, 10, 8)
        expected = 1.0 / (8.0 / 2.54)
        assert abs(result - expected) < 1e-6

    def test_gss_scale_applied(self):
        # Physical units are divided by scale
        result = _transform_from_inches(1.0, "cm", 12, 1, 1.2, 10, 8, scale=2.0)
        assert abs(result - 2.54 / 2.0) < 1e-10

    def test_zero_viewport_raises(self):
        with pytest.raises(ValueError, match="zero dimension"):
            _transform_from_inches(1.0, "npc", 12, 1, 1.2, 0.0, 10)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError):
            _transform_from_inches(1.0, "bogus_unit", 12, 1, 1.2, 10, 8)


class TestTransformXYFromInches:
    """Port of R unit.c:1348-1377."""

    def test_native_location(self):
        # scalemin=10, scalemax=20, thisCM=5.08 (2 inches)
        # 1 inch → scalemin + (1/2)*(20-10) = 15.0
        result = _transform_xy_from_inches(
            1.0, "native", 10.0, 20.0, 12, 1, 1.2, 5.08, 5.08)
        assert abs(result - 15.0) < 1e-6

    def test_npc_passthrough(self):
        result = _transform_xy_from_inches(
            0.5, "npc", 0, 1, 12, 1, 1.2, 5.08, 5.08)
        expected = _transform_from_inches(0.5, "npc", 12, 1, 1.2, 5.08, 5.08)
        assert abs(result - expected) < 1e-10


class TestTransformWHFromInches:
    """Port of R unit.c:1379-1408."""

    def test_native_dimension(self):
        # scalemin=10, scalemax=20, thisCM=5.08 (2 inches)
        # 1 inch → (1/2)*(20-10) = 5.0  (no scalemin offset for dimensions)
        result = _transform_wh_from_inches(
            1.0, "native", 10.0, 20.0, 12, 1, 1.2, 5.08, 5.08)
        assert abs(result - 5.0) < 1e-6


class TestTransformRelativeSpecialCase:
    """Port of R unit.c:1418-1475, zero-dimension special case."""

    def test_xy_to_npc_from_native(self):
        # native 15 in [10,20] → NPC 0.5
        result = _transform_xy_to_npc(15.0, "native", 10.0, 20.0)
        assert abs(result - 0.5) < 1e-10

    def test_xy_to_npc_from_npc(self):
        result = _transform_xy_to_npc(0.7, "npc", 0, 1)
        assert result == 0.7

    def test_wh_to_npc_from_native(self):
        # 5 native units in [10,20] → 0.5 NPC
        result = _transform_wh_to_npc(5.0, "native", 10.0, 20.0)
        assert abs(result - 0.5) < 1e-10

    def test_xy_from_npc_to_native(self):
        # NPC 0.5 in [10,20] → native 15
        result = _transform_xy_from_npc(0.5, "native", 10.0, 20.0)
        assert abs(result - 15.0) < 1e-10

    def test_wh_from_npc_to_native(self):
        result = _transform_wh_from_npc(0.5, "native", 10.0, 20.0)
        assert abs(result - 5.0) < 1e-10


# ===================================================================
# _units.py: convert_theta
# ===================================================================


class TestConvertTheta:
    def test_east(self):
        assert convert_theta("east") == 0.0

    def test_north(self):
        assert convert_theta("north") == 90.0

    def test_west(self):
        assert convert_theta("west") == 180.0

    def test_south(self):
        assert convert_theta("south") == 270.0

    def test_numeric_passthrough(self):
        assert convert_theta(45) == 45.0

    def test_negative_wrap(self):
        assert convert_theta(-90) == 270.0

    def test_over_360(self):
        assert convert_theta(450) == 90.0

    def test_case_insensitive(self):
        assert convert_theta("North") == 90.0

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="invalid theta"):
            convert_theta("northeast")


# ===================================================================
# _units.py: Summary.unit (min/max/sum)
# ===================================================================


class TestUnitSummary:
    def test_sum_simple(self):
        u = Unit([1, 2, 3], "cm")
        result = unit_summary_sum(u)
        assert len(result) == 1
        assert abs(result._values[0] - 6.0) < 1e-10
        assert result._units[0] == "cm"

    def test_min_simple(self):
        u = Unit([3, 1, 4], "cm")
        result = unit_summary_min(u)
        assert abs(result._values[0] - 1.0) < 1e-10

    def test_max_simple(self):
        u = Unit([3, 1, 4], "cm")
        result = unit_summary_max(u)
        assert abs(result._values[0] - 4.0) < 1e-10

    def test_multi_arg_sum(self):
        u1 = Unit([1, 2], "inches")
        u2 = Unit([3], "inches")
        result = unit_summary_sum(u1, u2)
        assert abs(result._values[0] - 6.0) < 1e-10

    def test_mixed_units_produce_compound(self):
        u1 = Unit(2, "cm")
        u2 = Unit(1, "inches")
        result = unit_summary_sum(u1, u2)
        assert result._units[0] == "sum"

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            unit_summary_sum()


# ===================================================================
# _units.py: unit_type with recurse
# ===================================================================


class TestUnitTypeRecurse:
    def test_simple_no_recurse(self):
        u = Unit([1, 2], "cm")
        assert unit_type(u) == ["cm", "cm"]

    def test_simple_recurse(self):
        u = Unit([1, 2], "cm")
        assert unit_type(u, recurse=True) == ["cm", "cm"]

    def test_compound_no_recurse(self):
        u = Unit(1, "cm") + Unit(2, "inches")
        assert unit_type(u) == "sum"

    def test_compound_recurse(self):
        u = Unit(1, "cm") + Unit(2, "inches")
        result = unit_type(u, recurse=True)
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert "cm" in result[0]
        assert "inches" in result[0]


# ===================================================================
# _units.py: unit_rep with length_out/each
# ===================================================================


class TestUnitRepExtended:
    def test_each(self):
        u = Unit([1, 2], "cm")
        r = unit_rep(u, each=3)
        assert len(r) == 6
        assert list(r._values) == [1, 1, 1, 2, 2, 2]

    def test_length_out_truncate(self):
        u = Unit([1, 2, 3], "cm")
        r = unit_rep(u, length_out=2)
        assert len(r) == 2

    def test_length_out_recycle(self):
        u = Unit([1, 2], "cm")
        r = unit_rep(u, length_out=5)
        assert len(r) == 5
        assert list(r._values) == [1, 2, 1, 2, 1]

    def test_each_plus_length_out(self):
        u = Unit([1, 2], "cm")
        r = unit_rep(u, each=2, length_out=3)
        assert len(r) == 3
        assert list(r._values) == [1, 1, 2]

    def test_times_zero(self):
        u = Unit([1, 2], "cm")
        r = unit_rep(u, times=0)
        assert len(r) == 0


# ===================================================================
# _units.py: device_loc / device_dim (with renderer)
# ===================================================================


class TestDeviceLocDim:
    @pytest.fixture(autouse=True)
    def setup_renderer(self):
        from grid_py import grid_newpage
        grid_newpage(width=7, height=5, dpi=100)

    def test_device_loc_inches(self):
        result = device_loc(Unit(1, "inches"), Unit(2, "inches"))
        assert "x" in result and "y" in result
        # Should be Unit objects in inches
        assert isinstance(result["x"], Unit)

    def test_device_loc_value_only(self):
        result = device_loc(
            Unit(1, "inches"), Unit(2, "inches"), value_only=True)
        assert isinstance(result["x"], np.ndarray)
        assert isinstance(result["y"], np.ndarray)

    def test_device_loc_device_coords(self):
        result = device_loc(
            Unit(1, "inches"), Unit(2, "inches"),
            value_only=True, device=True)
        # Device coords at 100 DPI: 1 inch = 100 px
        assert abs(result["x"][0] - 100.0) < 1.0

    def test_device_dim_inches(self):
        result = device_dim(Unit(2, "inches"), Unit(3, "inches"))
        assert "w" in result and "h" in result

    def test_device_dim_value_only(self):
        result = device_dim(
            Unit(2, "inches"), Unit(3, "inches"), value_only=True)
        assert abs(result["w"][0] - 2.0) < 0.01
        assert abs(result["h"][0] - 3.0) < 0.01


# ===================================================================
# _units.py: convertUnit with renderer context
# ===================================================================


class TestConvertUnitWithContext:
    @pytest.fixture(autouse=True)
    def setup_renderer(self):
        from grid_py import grid_newpage
        grid_newpage(width=7, height=5, dpi=100)

    def test_npc_to_inches(self):
        # 0.5 NPC in 7-inch wide viewport = 3.5 inches
        result = convert_x(Unit(0.5, "npc"), "inches", valueOnly=True)
        assert abs(result[0] - 3.5) < 0.01

    def test_inches_to_npc(self):
        result = convert_x(Unit(3.5, "inches"), "npc", valueOnly=True)
        assert abs(result[0] - 0.5) < 0.01

    def test_native_to_inches(self):
        from grid_py._viewport import Viewport, push_viewport, pop_viewport
        vp = Viewport(name="v", xscale=(0, 100), yscale=(0, 50))
        push_viewport(vp)
        # native 50 in [0,100] = 0.5 NPC * 7 inches = 3.5 inches
        result = convert_x(Unit(50, "native"), "inches", valueOnly=True)
        assert abs(result[0] - 3.5) < 0.01
        pop_viewport()

    def test_cm_to_native(self):
        from grid_py._viewport import Viewport, push_viewport, pop_viewport
        vp = Viewport(name="v", xscale=(0, 100), yscale=(0, 50))
        push_viewport(vp)
        result = convert_width(Unit(2.54, "cm"), "native", valueOnly=True)
        # 2.54cm = 1 inch, viewport is 7 inches wide, range 100
        # 1 inch / 7 inches * 100 = 14.286
        assert abs(result[0] - 100.0 / 7.0) < 0.1
        pop_viewport()


# ===================================================================
# _renderer_base.py: evaluateGrobUnit
# ===================================================================


class TestEvaluateGrobUnit:
    @pytest.fixture(autouse=True)
    def setup_renderer(self):
        from grid_py import grid_newpage
        grid_newpage(width=7, height=5, dpi=100)

    def test_grobwidth_rect_inches(self):
        from grid_py._grob import Grob
        from grid_py._state import get_state
        rect = Grob(name="r", _grid_class="rect",
                     x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                     width=Unit(3, "inches"), height=Unit(2, "inches"))
        renderer = get_state().get_renderer()
        result = renderer._evaluate_grob_unit(rect, "grobwidth")
        assert abs(result - 3.0) < 0.01

    def test_grobheight_rect_mm(self):
        from grid_py._grob import Grob
        from grid_py._state import get_state
        rect = Grob(name="r", _grid_class="rect",
                     x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                     width=Unit(50, "mm"), height=Unit(30, "mm"))
        renderer = get_state().get_renderer()
        result = renderer._evaluate_grob_unit(rect, "grobheight")
        assert abs(result - 30.0 / 25.4) < 0.01

    def test_grobwidth_unit_resolves(self):
        from grid_py._grob import Grob
        from grid_py._state import get_state
        rect = Grob(name="r", _grid_class="rect",
                     x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                     width=Unit(2, "inches"), height=Unit(1, "inches"))
        renderer = get_state().get_renderer()
        u = Unit(1, "grobwidth", data=rect)
        val = renderer._resolve_to_inches(u, "x", True)
        assert abs(val - 2.0) < 0.01

    def test_grobwidth_string_lookup(self):
        """Test that gPath string lookup works via _find_grob_for_metric."""
        from grid_py._grob import Grob, GTree, GList
        from grid_py._state import get_state
        state = get_state()
        child = Grob(name="myrect", _grid_class="rect",
                      x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                      width=Unit(4, "inches"), height=Unit(3, "inches"))
        tree = GTree(children=GList(child), name="tree")
        state._current_grob = tree
        renderer = state.get_renderer()
        result = renderer._evaluate_grob_unit("myrect", "grobwidth")
        assert abs(result - 4.0) < 0.01
        state._current_grob = None

    def test_null_unit_returns_zero(self):
        """GTree without widthDetails → unit(1,"null") → 0 inches."""
        from grid_py._grob import GTree, GList, Grob
        from grid_py._state import get_state
        tree = GTree(children=GList(Grob(name="c")), name="t")
        renderer = get_state().get_renderer()
        result = renderer._evaluate_grob_unit(tree, "grobwidth")
        assert result == 0.0


# ===================================================================
# _size.py: widthDetails/heightDetails rewrite
# ===================================================================


class TestWidthHeightDetails:
    @pytest.fixture(autouse=True)
    def setup_renderer(self):
        from grid_py import grid_newpage
        grid_newpage(width=7, height=5, dpi=100)

    def test_rect_width_inches(self):
        from grid_py._grob import Grob
        from grid_py._size import width_details
        rect = Grob(name="r", _grid_class="rect",
                     x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                     width=Unit(3, "inches"), height=Unit(2, "inches"))
        w = width_details(rect)
        assert abs(w._values[0] - 3.0) < 0.01

    def test_rect_height_mm(self):
        from grid_py._grob import Grob
        from grid_py._size import height_details
        rect = Grob(name="r", _grid_class="rect",
                     x=Unit(10, "mm"), y=Unit(10, "mm"),
                     width=Unit(50, "mm"), height=Unit(30, "mm"))
        h = height_details(rect)
        assert abs(h._values[0] - 30.0 / 25.4) < 0.01

    def test_circle_width(self):
        from grid_py._grob import Grob
        from grid_py._size import width_details
        c = Grob(name="c", _grid_class="circle",
                  x=Unit(3, "inches"), y=Unit(2, "inches"),
                  r=Unit(1, "inches"))
        w = width_details(c)
        assert abs(w._values[0] - 2.0) < 0.01  # diameter

    def test_lines_width(self):
        from grid_py._grob import Grob
        from grid_py._size import width_details
        ln = Grob(name="l", _grid_class="lines",
                   x=Unit([1, 4], "inches"), y=Unit([1, 3], "inches"))
        w = width_details(ln)
        assert abs(w._values[0] - 3.0) < 0.01  # 4 - 1

    def test_segments_width(self):
        from grid_py._grob import Grob
        from grid_py._size import width_details
        seg = Grob(name="s", _grid_class="segments",
                    x0=Unit(1, "inches"), y0=Unit(1, "inches"),
                    x1=Unit(5, "inches"), y1=Unit(3, "inches"))
        w = width_details(seg)
        assert abs(w._values[0] - 4.0) < 0.01  # 5 - 1

    def test_points_height(self):
        from grid_py._grob import Grob
        from grid_py._size import height_details
        pts = Grob(name="p", _grid_class="points",
                    x=Unit([1, 2], "inches"), y=Unit([1, 4], "inches"))
        h = height_details(pts)
        assert abs(h._values[0] - 3.0) < 0.01  # 4 - 1


# ===================================================================
# _coords.py: per-primitive grob_points
# ===================================================================


class TestGrobPointsPrimitives:
    @pytest.fixture(autouse=True)
    def setup_renderer(self):
        from grid_py import grid_newpage
        grid_newpage(width=7, height=5, dpi=100)

    def test_rect_closed(self):
        from grid_py._grob import Grob
        from grid_py._coords import grob_points
        rect = Grob(name="r", _grid_class="rect",
                     x=Unit(3, "inches"), y=Unit(2, "inches"),
                     width=Unit(2, "inches"), height=Unit(1, "inches"),
                     just="centre")
        pts = grob_points(rect, closed=True)
        assert not pts.is_empty()
        # 4 corners
        assert len(pts[0]) == 4

    def test_rect_open_is_empty(self):
        from grid_py._grob import Grob
        from grid_py._coords import grob_points
        rect = Grob(name="r", _grid_class="rect",
                     x=Unit(3, "inches"), y=Unit(2, "inches"),
                     width=Unit(2, "inches"), height=Unit(1, "inches"))
        pts = grob_points(rect, closed=False)
        assert pts.is_empty()

    def test_circle_closed(self):
        from grid_py._grob import Grob
        from grid_py._coords import grob_points
        c = Grob(name="c", _grid_class="circle",
                  x=Unit(3, "inches"), y=Unit(2, "inches"),
                  r=Unit(1, "inches"))
        pts = grob_points(c, closed=True)
        assert not pts.is_empty()
        assert len(pts[0]) == 100  # default n=100

    def test_lines_open(self):
        from grid_py._grob import Grob
        from grid_py._coords import grob_points
        ln = Grob(name="l", _grid_class="lines",
                   x=Unit([1, 3], "inches"), y=Unit([1, 2], "inches"))
        pts = grob_points(ln, closed=False)
        assert not pts.is_empty()
        assert len(pts[0]) == 2

    def test_segments_open(self):
        from grid_py._grob import Grob
        from grid_py._coords import grob_points
        seg = Grob(name="s", _grid_class="segments",
                    x0=Unit(1, "inches"), y0=Unit(1, "inches"),
                    x1=Unit(3, "inches"), y1=Unit(2, "inches"))
        pts = grob_points(seg, closed=False)
        assert not pts.is_empty()
        # 1 segment → 2 points
        assert len(pts[0]) == 2

    def test_polygon_closed(self):
        from grid_py._grob import Grob
        from grid_py._coords import grob_points
        pg = Grob(name="p", _grid_class="polygon",
                   x=Unit([0, 1, 0.5], "inches"),
                   y=Unit([0, 0, 1], "inches"))
        pts = grob_points(pg, closed=True)
        assert not pts.is_empty()
        assert len(pts[0]) == 3

    def test_null_always_empty(self):
        from grid_py._grob import Grob
        from grid_py._coords import grob_points
        n = Grob(name="n", _grid_class="null")
        pts = grob_points(n, closed=True)
        assert pts.is_empty()

    def test_text_closed(self):
        from grid_py._grob import Grob
        from grid_py._coords import grob_points
        t = Grob(name="t", _grid_class="text",
                  label="Hello", x=Unit(3, "inches"), y=Unit(2, "inches"),
                  just="centre")
        pts = grob_points(t, closed=True)
        # Text returns bounding box (4 corners) if label is non-empty
        assert not pts.is_empty()


# ===================================================================
# _patterns.py: resolve pipeline
# ===================================================================


class TestPatternResolution:
    def test_is_pattern(self):
        from grid_py._patterns import (
            LinearGradient, RadialGradient, is_pattern,
        )
        assert is_pattern(LinearGradient(colours=["red", "blue"]))
        assert is_pattern(RadialGradient(colours=["white", "black"]))
        assert not is_pattern("red")
        assert not is_pattern(None)

    def test_is_pattern_list(self):
        from grid_py._patterns import (
            LinearGradient, is_pattern_list,
        )
        assert is_pattern_list([LinearGradient(colours=["r", "b"]),
                                LinearGradient(colours=["r", "b"])])
        assert not is_pattern_list([])
        assert not is_pattern_list(["red", "blue"])

    def test_resolved_pattern(self):
        from grid_py._patterns import ResolvedPattern, LinearGradient
        grad = LinearGradient(colours=["red", "blue"])
        rp = ResolvedPattern(grad, ref=42)
        assert rp.ref == 42
        assert rp.pattern is grad
        assert rp._resolved is True

    def test_resolve_fill_string_passthrough(self):
        from grid_py._patterns import resolve_fill
        assert resolve_fill("red") == "red"
        assert resolve_fill(None) is None

    def test_resolve_fill_already_resolved(self):
        from grid_py._patterns import (
            resolve_fill, ResolvedPattern, LinearGradient,
        )
        rp = ResolvedPattern(LinearGradient(colours=["r", "b"]), ref=1)
        assert resolve_fill(rp) is rp

    def test_resolve_pattern_linear_gradient(self):
        from grid_py import grid_newpage
        from grid_py._patterns import (
            LinearGradient, resolve_pattern, ResolvedPattern,
        )
        grid_newpage(width=7, height=5, dpi=100)
        grad = LinearGradient(
            colours=["red", "blue"],
            x1=Unit(0, "npc"), y1=Unit(0, "npc"),
            x2=Unit(1, "npc"), y2=Unit(1, "npc"),
        )
        result = resolve_pattern(grad)
        assert isinstance(result, ResolvedPattern)
        assert result.ref["type"] == "linear_gradient"

    def test_resolve_pattern_radial_gradient(self):
        from grid_py import grid_newpage
        from grid_py._patterns import (
            RadialGradient, resolve_pattern, ResolvedPattern,
        )
        grid_newpage(width=7, height=5, dpi=100)
        grad = RadialGradient(
            colours=["white", "black"],
            cx1=Unit(0.5, "npc"), cy1=Unit(0.5, "npc"), r1=Unit(0, "npc"),
            cx2=Unit(0.5, "npc"), cy2=Unit(0.5, "npc"), r2=Unit(0.5, "npc"),
        )
        result = resolve_pattern(grad)
        assert isinstance(result, ResolvedPattern)
        assert result.ref["type"] == "radial_gradient"

    def test_record_grob_for_pattern_noop_on_string_fill(self):
        """record_grob_for_pattern_resolution is a no-op for string fills."""
        from grid_py import grid_newpage
        from grid_py._grob import Grob
        from grid_py._patterns import record_grob_for_pattern_resolution
        grid_newpage(width=7, height=5, dpi=100)
        g = Grob(name="r", _grid_class="rect")
        # Should not raise
        record_grob_for_pattern_resolution(g)


# ===================================================================
# _group.py: draw_details with real renderer
# ===================================================================


class TestGroupDrawDetailsWithRenderer:
    @pytest.fixture(autouse=True)
    def setup_renderer(self):
        from grid_py import grid_newpage
        grid_newpage(width=7, height=5, dpi=100)

    def test_group_grob_defines_and_uses(self):
        from grid_py._grob import Grob
        from grid_py._group import GroupGrob
        from grid_py._draw import grid_draw
        from grid_py._state import get_state
        src = Grob(name="src", _grid_class="rect",
                    x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                    width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        grp = GroupGrob(src=src, op="over", name="mygrp")
        grid_draw(grp)
        state = get_state()
        data = state.lookup_group("mygrp")
        assert data is not None
        assert data["ref"] is not None

    def test_define_then_use(self):
        from grid_py._grob import Grob
        from grid_py._group import DefineGrob, UseGrob
        from grid_py._draw import grid_draw
        from grid_py._state import get_state
        src = Grob(name="src", _grid_class="circle",
                    x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                    r=Unit(0.3, "npc"))
        defn = DefineGrob(src=src, name="mycircle")
        grid_draw(defn)
        state = get_state()
        assert state.lookup_group("mycircle") is not None
        # Now use it
        use = UseGrob(group="mycircle", name="use1")
        grid_draw(use)  # should not raise


# ===================================================================
# _grob.py: grep/global name matching
# ===================================================================


class TestGrepGlobalMatching:
    def test_name_match_exact(self):
        from grid_py._grob import _name_match
        assert _name_match("foo", "foo", grep=False)
        assert not _name_match("foo", "foobar", grep=False)

    def test_name_match_grep(self):
        from grid_py._grob import _name_match
        assert _name_match("foo", "foobar", grep=True)
        assert _name_match("^rect", "rect.1", grep=True)
        assert not _name_match("^circle", "rect.1", grep=True)

    def test_name_positions(self):
        from grid_py._grob import _name_positions
        names = ["a", "b", "c", "b"]
        assert _name_positions("b", names, grep=False) == [1]
        assert _name_positions("z", names, grep=False) == []
        assert _name_positions("[abc]", names, grep=True) == [0, 1, 2, 3]

    def test_get_grob_grep(self):
        from grid_py._grob import Grob, GTree, GList, get_grob
        c1 = Grob(name="rect.1")
        c2 = Grob(name="rect.2")
        c3 = Grob(name="circle.1")
        tree = GTree(children=GList(c1, c2, c3), name="tree")
        found = get_grob(tree, "rect.1", grep=True)
        assert found.name == "rect.1"

    def test_get_grob_global(self):
        from grid_py._grob import Grob, GTree, GList, get_grob
        c1 = Grob(name="rect.1", _grid_class="rect")
        c2 = Grob(name="rect.2", _grid_class="rect")
        c3 = Grob(name="circle.1", _grid_class="circle")
        tree = GTree(children=GList(c1, c2, c3), name="tree")
        found = get_grob(tree, "rect", grep=True, global_=True)
        # global returns GList of all matches
        assert len(found) == 2
