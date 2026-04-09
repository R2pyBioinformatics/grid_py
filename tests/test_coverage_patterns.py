"""Tests for grid_py._patterns -- gradients and patterns."""

from __future__ import annotations

import pytest

from grid_py._patterns import (
    LinearGradient,
    Pattern,
    RadialGradient,
    _ensure_unit,
    _make_stops,
    _resolve_just,
    _validate_extend,
    linear_gradient,
    pattern,
    radial_gradient,
)
from grid_py._units import Unit


# ---------------------------------------------------------------------------
# _validate_extend
# ---------------------------------------------------------------------------

class TestValidateExtend:

    def test_valid_values(self):
        for v in ("pad", "repeat", "reflect", "none"):
            assert _validate_extend(v) == v

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="extend"):
            _validate_extend("invalid")


# ---------------------------------------------------------------------------
# _ensure_unit
# ---------------------------------------------------------------------------

class TestEnsureUnit:

    def test_already_unit(self):
        u = Unit(1, "cm")
        result = _ensure_unit(u, "npc")
        assert result is u

    def test_numeric_to_unit(self):
        result = _ensure_unit(0.5, "npc")
        assert isinstance(result, Unit)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _make_stops
# ---------------------------------------------------------------------------

class TestMakeStops:

    def test_auto_stops(self):
        colours, stops = _make_stops(["red", "blue"], None)
        assert colours == ["red", "blue"]
        assert len(stops) == 2
        assert stops[0] == 0.0
        assert stops[1] == 1.0

    def test_explicit_stops(self):
        colours, stops = _make_stops(["red", "blue"], [0.2, 0.8])
        assert stops == [0.2, 0.8]

    def test_recycle_colours(self):
        colours, stops = _make_stops(["red"], [0.0, 0.5, 1.0])
        assert len(colours) == 3
        assert colours == ["red", "red", "red"]

    def test_recycle_stops(self):
        colours, stops = _make_stops(["red", "green", "blue"], [0.0])
        assert len(stops) == 3

    def test_three_colours_auto(self):
        colours, stops = _make_stops(["a", "b", "c"], None)
        assert len(stops) == 3
        assert stops[0] == 0.0
        assert stops[-1] == 1.0


# ---------------------------------------------------------------------------
# _resolve_just
# ---------------------------------------------------------------------------

class TestResolveJust:

    def test_centre(self):
        h, v = _resolve_just("centre")
        assert h == 0.5
        assert v == 0.5

    def test_center(self):
        h, v = _resolve_just("center")
        assert h == 0.5
        assert v == 0.5

    def test_left(self):
        h, v = _resolve_just("left")
        assert h == 0.0

    def test_right(self):
        h, v = _resolve_just("right")
        assert h == 1.0

    def test_top(self):
        h, v = _resolve_just("top")
        assert v == 1.0

    def test_bottom(self):
        h, v = _resolve_just("bottom")
        assert v == 0.0

    def test_numeric_pair(self):
        h, v = _resolve_just((0.3, 0.7))
        assert h == 0.3
        assert v == 0.7

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            _resolve_just("unknown")


# ---------------------------------------------------------------------------
# LinearGradient
# ---------------------------------------------------------------------------

class TestLinearGradient:

    def test_defaults(self):
        lg = LinearGradient(["black", "white"])
        assert lg.colours == ["black", "white"]
        assert len(lg.stops) == 2
        assert lg.extend == "pad"
        assert lg.group is True

    def test_custom_coords(self):
        lg = LinearGradient(["red", "blue"], x1=0.1, y1=0.2, x2=0.9, y2=0.8)
        assert len(lg.x1) == 1
        assert len(lg.x2) == 1

    def test_unit_coords(self):
        lg = LinearGradient(
            ["red", "blue"],
            x1=Unit(0, "npc"),
            y1=Unit(0, "npc"),
            x2=Unit(1, "npc"),
            y2=Unit(1, "npc"),
        )
        assert len(lg.x1) == 1

    def test_invalid_extend_raises(self):
        with pytest.raises(ValueError):
            LinearGradient(["black", "white"], extend="bad")

    def test_repr(self):
        lg = LinearGradient(["red", "blue"])
        s = repr(lg)
        assert "LinearGradient" in s
        assert "red" in s


# ---------------------------------------------------------------------------
# RadialGradient
# ---------------------------------------------------------------------------

class TestRadialGradient:

    def test_defaults(self):
        rg = RadialGradient(["black", "white"])
        assert rg.colours == ["black", "white"]
        assert len(rg.stops) == 2
        assert rg.extend == "pad"
        assert rg.group is True

    def test_custom_params(self):
        rg = RadialGradient(
            ["red", "green", "blue"],
            cx1=0.3, cy1=0.3, r1=0.0,
            cx2=0.5, cy2=0.5, r2=0.8,
        )
        assert len(rg.cx1) == 1
        assert len(rg.r2) == 1

    def test_extend_reflect(self):
        rg = RadialGradient(["a", "b"], extend="reflect")
        assert rg.extend == "reflect"

    def test_repr(self):
        rg = RadialGradient(["a", "b"])
        s = repr(rg)
        assert "RadialGradient" in s


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

class TestPattern:

    def test_defaults(self):
        p = Pattern(grob="dummy")
        assert p.grob == "dummy"
        assert p.hjust == 0.5
        assert p.vjust == 0.5
        assert p.extend == "pad"
        assert p.group is True

    def test_custom_dims(self):
        p = Pattern("grob", x=0.1, y=0.2, width=0.5, height=0.5)
        assert len(p.x) == 1
        assert len(p.width) == 1

    def test_custom_just(self):
        p = Pattern("grob", just="left")
        assert p.hjust == 0.0

    def test_numeric_just(self):
        p = Pattern("grob", just=(0.3, 0.7))
        assert p.hjust == 0.3
        assert p.vjust == 0.7

    def test_repr(self):
        p = Pattern("grob")
        s = repr(p)
        assert "Pattern" in s


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactoryFunctions:

    def test_linear_gradient_defaults(self):
        lg = linear_gradient()
        assert lg.colours == ["black", "white"]

    def test_linear_gradient_custom(self):
        lg = linear_gradient(["red", "blue"], stops=[0.0, 1.0])
        assert lg.colours == ["red", "blue"]

    def test_radial_gradient_defaults(self):
        rg = radial_gradient()
        assert rg.colours == ["black", "white"]

    def test_radial_gradient_custom(self):
        rg = radial_gradient(["red", "blue"], r2=0.8)
        assert len(rg.r2) == 1

    def test_pattern_factory(self):
        p = pattern("grob_obj", x=0.5, y=0.5)
        assert isinstance(p, Pattern)
        assert p.grob == "grob_obj"

    def test_linear_gradient_group_false(self):
        lg = linear_gradient(group=False)
        assert lg.group is False

    def test_radial_gradient_extend_none(self):
        rg = radial_gradient(extend="none")
        assert rg.extend == "none"

    def test_pattern_extend_repeat(self):
        p = pattern("g", extend="repeat")
        assert p.extend == "repeat"
