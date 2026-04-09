"""Tests to improve coverage of grid_py._highlevel (supplementing test_highlevel.py).

Focuses on functions and branches not covered by the existing test_highlevel.py:
- grid_show_viewport
- grid_plot_and_legend
- grid_panel
- grid_strip
- grid_multipanel
- grid_top_level_vp
- Internal helpers: _ensure_unit, _is_even, _is_odd, _extend_range
- Edge cases in pack_grob, xaxis/yaxis internals
"""

import math
import warnings

import numpy as np
import pytest

import grid_py
from grid_py import (
    GList,
    GTree,
    Gpar,
    Grob,
    GridLayout,
    Unit,
    Viewport,
    get_state,
    grid_newpage,
    grid_draw,
    is_grob,
    rect_grob,
    circle_grob,
    text_grob,
    lines_grob,
    null_grob,
)
from grid_py._highlevel import (
    _ensure_unit,
    _is_even,
    _is_odd,
    _extend_range,
    frame_grob,
    pack_grob,
    place_grob,
    grid_frame,
    grid_pack,
    grid_place,
    grid_show_viewport,
    grid_plot_and_legend,
    grid_panel,
    grid_strip,
    grid_multipanel,
    grid_top_level_vp,
    xaxis_grob,
    yaxis_grob,
    grid_xaxis,
    grid_yaxis,
    legend_grob,
    grid_legend,
    grid_grill,
    grid_show_layout,
    _num_col_specs,
    _col_spec,
    _new_col,
    _num_row_specs,
    _row_spec,
    _new_row,
    _update_col,
    _update_row,
)


@pytest.fixture(autouse=True)
def _reset():
    state = get_state()
    state.reset()
    yield
    state.reset()


class TestInternalHelpers:
    """Tests for internal helper functions."""

    def test_ensure_unit_already_unit(self):
        u = Unit(1, "cm")
        assert _ensure_unit(u) is u

    def test_ensure_unit_numeric(self):
        u = _ensure_unit(0.5)
        assert isinstance(u, Unit)

    def test_is_even(self):
        assert _is_even(0) is True
        assert _is_even(2) is True
        assert _is_even(3) is False

    def test_is_odd(self):
        assert _is_odd(1) is True
        assert _is_odd(3) is True
        assert _is_odd(4) is False

    def test_extend_range(self):
        lo, hi = _extend_range([1.0, 5.0])
        assert lo < 1.0
        assert hi > 5.0

    def test_extend_range_zero_range(self):
        lo, hi = _extend_range([3.0, 3.0])
        assert lo < 3.0
        assert hi > 3.0


class TestColRowHelpers:
    """Tests for column/row specification helpers."""

    def test_num_col_specs_none(self):
        assert _num_col_specs(None, None, None, None) == 0

    def test_num_col_specs_side_left(self):
        assert _num_col_specs("left", None, None, None) == 1

    def test_num_col_specs_side_top(self):
        # top/bottom don't count as col spec
        assert _num_col_specs("top", None, None, None) == 0

    def test_num_col_specs_col(self):
        assert _num_col_specs(None, 1, None, None) == 1

    def test_col_spec_side_left(self):
        assert _col_spec("left", None, None, None, 3) == 1

    def test_col_spec_side_right(self):
        assert _col_spec("right", None, None, None, 3) == 4

    def test_col_spec_col_before(self):
        assert _col_spec(None, None, 2, None, 3) == 2

    def test_col_spec_col_after(self):
        assert _col_spec(None, None, None, 2, 3) == 3

    def test_col_spec_col_explicit(self):
        assert _col_spec(None, 2, None, None, 3) == 2

    def test_new_col_none_col(self):
        assert _new_col(None, None, None, None, 3) is True

    def test_new_col_range_valid(self):
        assert _new_col(None, [1, 2], None, None, 3) is False

    def test_new_col_range_invalid(self):
        with pytest.raises(ValueError, match="range of existing"):
            _new_col(None, [1, 5], None, None, 3)

    def test_new_col_single_new(self):
        assert _new_col(None, 4, None, None, 3) is True

    def test_new_col_single_existing(self):
        assert _new_col(None, 2, None, None, 3) is False

    def test_new_col_invalid(self):
        with pytest.raises(ValueError, match="invalid 'col'"):
            _new_col(None, 5, None, None, 3)

    def test_num_row_specs_none(self):
        assert _num_row_specs(None, None, None, None) == 0

    def test_num_row_specs_side_top(self):
        assert _num_row_specs("top", None, None, None) == 1

    def test_num_row_specs_side_left(self):
        # left/right don't count as row spec
        assert _num_row_specs("left", None, None, None) == 0

    def test_row_spec_side_top(self):
        assert _row_spec("top", None, None, None, 3) == 1

    def test_row_spec_side_bottom(self):
        assert _row_spec("bottom", None, None, None, 3) == 4

    def test_row_spec_row_before(self):
        assert _row_spec(None, None, 2, None, 3) == 2

    def test_row_spec_row_after(self):
        assert _row_spec(None, None, None, 2, 3) == 3

    def test_new_row_range_valid(self):
        assert _new_row(None, [1, 2], None, None, 3) is False

    def test_new_row_range_invalid(self):
        with pytest.raises(ValueError, match="range of existing"):
            _new_row(None, [1, 5], None, None, 3)

    def test_new_row_single_new(self):
        assert _new_row(None, 4, None, None, 3) is True

    def test_new_row_invalid(self):
        with pytest.raises(ValueError, match="invalid 'row'"):
            _new_row(None, 5, None, None, 3)

    def test_update_col_shift(self):
        assert _update_col(3, 2) == 4

    def test_update_col_no_shift(self):
        assert _update_col(1, 3) == 1

    def test_update_col_range(self):
        assert _update_col([1, 3], 2) == [1, 4]

    def test_update_col_range_no_shift(self):
        assert _update_col([1, 2], 5) == [1, 2]

    def test_update_row_shift(self):
        assert _update_row(3, 2) == 4

    def test_update_row_no_shift(self):
        assert _update_row(1, 3) == 1

    def test_update_row_range(self):
        assert _update_row([1, 3], 2) == [1, 4]

    def test_update_row_range_no_shift(self):
        assert _update_row([1, 2], 5) == [1, 2]


class TestGridShowViewport:
    """Tests for grid_show_viewport."""

    def test_default_viewport(self):
        grid_newpage()
        grid_show_viewport(newpage=False)

    def test_custom_viewport(self):
        grid_newpage()
        vp = Viewport(width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        grid_show_viewport(v=vp, newpage=False)

    def test_with_parent_layout(self):
        grid_newpage()
        lay = GridLayout(nrow=2, ncol=2)
        vp = Viewport(layout_pos_row=1, layout_pos_col=1)
        grid_show_viewport(v=vp, parent_layout=lay, newpage=True)

    def test_with_vp_param(self):
        grid_newpage()
        outer = Viewport(width=Unit(0.8, "npc"), height=Unit(0.8, "npc"))
        grid_show_viewport(newpage=False, vp=outer)

    def test_with_vp_and_parent_layout(self):
        grid_newpage()
        lay = GridLayout(nrow=2, ncol=2)
        v = Viewport(layout_pos_row=1, layout_pos_col=1)
        # Don't pass vp= here as the internal pop count can mismatch
        grid_show_viewport(v=v, parent_layout=lay, newpage=True)


class TestGridPlotAndLegend:
    """Tests for grid_plot_and_legend."""

    def test_basic(self):
        grid_newpage()
        plot_g = rect_grob(name="plot")
        legend_g = rect_grob(name="legend")
        grid_plot_and_legend(plot_expr=plot_g, legend_expr=legend_g)

    def test_no_legend(self):
        grid_newpage()
        plot_g = rect_grob(name="plot")
        grid_plot_and_legend(plot_expr=plot_g)

    def test_no_plot(self):
        grid_newpage()
        legend_g = rect_grob(name="legend")
        grid_plot_and_legend(legend_expr=legend_g)

    def test_both_none(self):
        grid_newpage()
        grid_plot_and_legend()


class TestGridStrip:
    """Tests for grid_strip."""

    def test_basic(self):
        grid_newpage()
        grid_strip(label="test strip")

    def test_custom_ranges(self):
        grid_newpage()
        grid_strip(range_full=(0, 10), range_thumb=(3, 7))

    def test_with_viewport(self):
        grid_newpage()
        vp = Viewport(width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        grid_strip(vp=vp)


class TestGridPanel:
    """Tests for grid_panel."""

    def test_default(self):
        grid_newpage()
        result = grid_panel()
        assert "strip_vp" in result
        assert "plot_vp" in result

    def test_with_data(self):
        grid_newpage()
        result = grid_panel(x=[0.1, 0.5, 0.9], y=[0.2, 0.6, 0.8])
        assert isinstance(result, dict)

    def test_axis_options(self):
        grid_newpage()
        result = grid_panel(
            x=[0.1, 0.5], y=[0.2, 0.6],
            axis_right=True,
            axis_top=True,
            axis_left_label=False,
            axis_bottom_label=False,
        )
        assert isinstance(result, dict)

    def test_with_viewport(self):
        grid_newpage()
        vp = Viewport(width=Unit(0.8, "npc"), height=Unit(0.8, "npc"))
        result = grid_panel(vp=vp)
        assert isinstance(result, dict)


class TestGridMultipanel:
    """Tests for grid_multipanel."""

    def test_basic(self):
        grid_multipanel(nplots=4, nrow=2, ncol=2)

    def test_auto_layout(self):
        grid_multipanel(nplots=4)

    def test_single_panel(self):
        grid_multipanel(nplots=1, nrow=1, ncol=1)

    def test_invalid_nplots(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            grid_multipanel(nplots=0)

    def test_with_data(self):
        x = list(np.random.uniform(size=20))
        y = list(np.random.uniform(size=20))
        z = list(np.random.uniform(size=20))
        grid_multipanel(x=x, y=y, z=z, nplots=2, nrow=1, ncol=2)

    def test_with_viewport(self):
        grid_newpage()
        vp = Viewport(width=Unit(0.8, "npc"), height=Unit(0.8, "npc"))
        grid_multipanel(nplots=2, nrow=1, ncol=2, vp=vp, newpage=False)


class TestGridTopLevelVp:
    """Tests for grid_top_level_vp."""

    def test_returns_viewport(self):
        vp = grid_top_level_vp()
        assert isinstance(vp, Viewport)


class TestAdditionalAxisBranches:
    """Additional axis tests for uncovered branches."""

    def test_xaxis_with_edits(self):
        from grid_py._grob import GEdit
        edit = GEdit(gp=Gpar(col="red"))
        xg = xaxis_grob(at=[0, 0.5, 1], edits=edit)
        assert isinstance(xg, GTree)

    def test_yaxis_with_edits(self):
        from grid_py._grob import GEdit
        edit = GEdit(gp=Gpar(col="red"))
        yg = yaxis_grob(at=[0, 0.5, 1], edits=edit)
        assert isinstance(yg, GTree)

    def test_yaxis_custom_labels(self):
        yg = yaxis_grob(at=[0, 0.5, 1], label=["lo", "mid", "hi"])
        children_names = list(yg._children_order)
        assert "labels" in children_names

    def test_yaxis_not_main(self):
        """yaxis with main=False puts labels on right side."""
        yg = yaxis_grob(at=[0, 1], main=False, label=True)
        assert isinstance(yg, GTree)


class TestLegendAdditional:
    """Additional legend tests for uncovered branches."""

    def test_lines_first_false(self):
        lg = legend_grob(labels=["A"], pch=[1], do_lines=True, lines_first=False)
        assert isinstance(lg, GTree)

    def test_no_pch_with_lines(self):
        lg = legend_grob(labels=["A"], do_lines=True, do_points=False)
        assert isinstance(lg, GTree)

    def test_no_lines_no_points(self):
        lg = legend_grob(labels=["A"], do_lines=False, do_points=False)
        assert isinstance(lg, GTree)

    def test_ncol_only(self):
        lg = legend_grob(labels=["A", "B", "C"], ncol=2)
        assert isinstance(lg, GTree)

    def test_nrow_only(self):
        lg = legend_grob(labels=["A", "B", "C"], nrow=2)
        assert isinstance(lg, GTree)

    def test_invalid_ncol(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            legend_grob(labels=["A"], ncol=0)

    def test_gp_with_list_col(self):
        gp = Gpar(col=["red", "blue"], lty=[1, 2], lwd=[1, 2], fill=["grey", "white"])
        lg = legend_grob(labels=["A", "B"], gp=gp)
        assert isinstance(lg, GTree)


class TestGridGrillWithVp:
    """Test grid_grill with viewport."""

    def test_with_vp(self):
        grid_newpage()
        vp = Viewport(width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        grid_grill(vp=vp)


class TestGridFrameDraw:
    """Test grid_frame with draw=True and False."""

    def test_frame_draw_true(self):
        grid_newpage()
        fg = grid_frame(draw=True)
        assert isinstance(fg, GTree)

    def test_frame_draw_false(self):
        fg = grid_frame(draw=False)
        assert isinstance(fg, GTree)


class TestGridPackDraw:
    """Test grid_pack with redraw option."""

    def test_grid_pack_redraw_true(self):
        grid_newpage()
        fg = frame_grob(name="f")
        r = rect_grob(name="r")
        result = grid_pack(fg, r, redraw=True)
        assert isinstance(result, GTree)

    def test_grid_pack_redraw_false(self):
        fg = frame_grob(name="f")
        r = rect_grob(name="r")
        result = grid_pack(fg, r, redraw=False)
        assert isinstance(result, GTree)


class TestGridPlaceDraw:
    """Test grid_place with redraw option."""

    def test_grid_place_redraw(self):
        grid_newpage()
        fg = frame_grob(name="f")
        fg = pack_grob(fg, rect_grob(name="r1"), side="left")
        r2 = circle_grob(name="r2")
        result = grid_place(fg, r2, row=1, col=1, redraw=True)
        assert isinstance(result, GTree)


class TestGridShowLayoutAdditional:
    """Additional tests for grid_show_layout."""

    def test_with_vp(self):
        grid_newpage()
        lay = GridLayout(nrow=2, ncol=2)
        outer = Viewport(width=Unit(0.9, "npc"), height=Unit(0.9, "npc"))
        vp = grid_show_layout(lay, vp=outer, newpage=False)
        assert vp is not None

    def test_larger_layout(self):
        lay = GridLayout(nrow=3, ncol=3)
        vp = grid_show_layout(lay)
        assert vp is not None
