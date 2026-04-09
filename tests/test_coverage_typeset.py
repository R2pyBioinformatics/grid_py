"""Tests for grid_py._typeset -- glyph/typesetting support."""

from __future__ import annotations

import pytest

from grid_py._typeset import (
    GlyphInfo,
    GlyphJust,
    _valid_glyph_grob,
    glyph_grob,
    glyph_just,
)
from grid_py._gpar import Gpar
from grid_py._grob import Grob
from grid_py._units import Unit


# ---------------------------------------------------------------------------
# GlyphJust
# ---------------------------------------------------------------------------

class TestGlyphJust:

    def test_numeric(self):
        gj = GlyphJust(0.5)
        assert gj.value == 0.5
        assert gj.is_numeric is True

    def test_string(self):
        gj = GlyphJust("left")
        assert gj.value == "left"
        assert gj.is_numeric is False

    def test_with_name(self):
        gj = GlyphJust(0.3, name="width")
        assert gj.name == "width"

    def test_repr_without_name(self):
        gj = GlyphJust(0.5)
        assert "0.5" in repr(gj)

    def test_repr_with_name(self):
        gj = GlyphJust(0.5, name="w")
        assert "name=" in repr(gj)

    def test_int_is_numeric(self):
        gj = GlyphJust(1)
        assert gj.is_numeric is True


# ---------------------------------------------------------------------------
# glyph_just
# ---------------------------------------------------------------------------

class TestGlyphJustFactory:

    def test_pass_through_glyph_just(self):
        original = GlyphJust(0.5)
        result = glyph_just(original)
        assert result is original

    def test_string_centre(self):
        result = glyph_just("centre")
        assert result.value == "centre"

    def test_string_center_aliased(self):
        result = glyph_just("center")
        assert result.value == "centre"

    def test_string_left(self):
        result = glyph_just("left")
        assert result.value == "left"

    def test_numeric_float(self):
        result = glyph_just(0.3)
        assert result.value == 0.3
        assert result.is_numeric is True

    def test_numeric_int(self):
        result = glyph_just(0)
        assert result.value == 0.0
        assert result.is_numeric is True

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="numeric or a string"):
            glyph_just([1, 2])


# ---------------------------------------------------------------------------
# GlyphInfo
# ---------------------------------------------------------------------------

class TestGlyphInfo:

    def test_basic(self):
        info = GlyphInfo(
            glyphs={"x": [0, 10], "y": [0, 10]},
            width=100.0,
            height=50.0,
        )
        assert info.width == 100.0
        assert info.height == 50.0
        assert info.h_anchor == {"left": 0.0}
        assert info.v_anchor == {"bottom": 0.0}

    def test_custom_anchors(self):
        info = GlyphInfo(
            glyphs={"x": [0]},
            width=10.0,
            height=12.0,
            h_anchor={"left": 0, "right": 100},
            v_anchor={"bottom": 0, "top": 80},
        )
        assert info.h_anchor == {"left": 0, "right": 100}
        assert info.v_anchor == {"bottom": 0, "top": 80}

    def test_repr(self):
        info = GlyphInfo({"x": [1, 2, 3], "y": [4, 5, 6]}, 10.0, 12.0)
        s = repr(info)
        assert "n_glyphs=3" in s


# ---------------------------------------------------------------------------
# _valid_glyph_grob
# ---------------------------------------------------------------------------

class TestValidGlyphGrob:

    def _make_valid_grob(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        g = Grob(
            _grid_class="glyphgrob",
            glyphInfo=info,
            x=Unit(0.5, "npc"),
            y=Unit(0.5, "npc"),
            hjust=GlyphJust("centre"),
            vjust=GlyphJust("centre"),
        )
        return g

    def test_valid_passes(self):
        g = self._make_valid_grob()
        result = _valid_glyph_grob(g)
        assert result is g

    def test_missing_glyph_info_raises(self):
        g = Grob(_grid_class="glyphgrob", x=Unit(0.5, "npc"), y=Unit(0.5, "npc"))
        with pytest.raises(TypeError, match="GlyphInfo"):
            _valid_glyph_grob(g)

    def test_non_unit_x_raises(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        g = Grob(
            _grid_class="glyphgrob",
            glyphInfo=info,
            x=0.5,
            y=Unit(0.5, "npc"),
            hjust=GlyphJust("centre"),
            vjust=GlyphJust("centre"),
        )
        with pytest.raises(TypeError, match="Unit"):
            _valid_glyph_grob(g)

    def test_non_glyph_just_raises(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        g = Grob(
            _grid_class="glyphgrob",
            glyphInfo=info,
            x=Unit(0.5, "npc"),
            y=Unit(0.5, "npc"),
            hjust=0.5,
            vjust=0.5,
        )
        with pytest.raises(TypeError, match="GlyphJust"):
            _valid_glyph_grob(g)


# ---------------------------------------------------------------------------
# glyph_grob
# ---------------------------------------------------------------------------

class TestGlyphGrob:

    def test_basic(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        g = glyph_grob(info)
        assert g._grid_class == "glyphgrob"
        assert isinstance(g.hjust, GlyphJust)
        assert isinstance(g.vjust, GlyphJust)

    def test_custom_position(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        g = glyph_grob(info, x=0.3, y=0.7, default_units="npc")
        assert isinstance(g.x, Unit)
        assert isinstance(g.y, Unit)

    def test_with_unit_position(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        g = glyph_grob(info, x=Unit(1, "cm"), y=Unit(2, "cm"))
        assert isinstance(g.x, Unit)

    def test_custom_just_string(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        g = glyph_grob(info, hjust="left", vjust="bottom")
        assert g.hjust.value == "left"
        assert g.vjust.value == "bottom"

    def test_custom_just_numeric(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        g = glyph_grob(info, hjust=0.0, vjust=1.0)
        assert g.hjust.is_numeric
        assert g.vjust.is_numeric

    def test_with_gp(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        gp = Gpar(col="red")
        g = glyph_grob(info, gp=gp)
        assert g.gp is gp

    def test_custom_name(self):
        info = GlyphInfo({"x": [0], "y": [0]}, 10.0, 12.0)
        g = glyph_grob(info, name="myglyphs")
        assert g.name == "myglyphs"
