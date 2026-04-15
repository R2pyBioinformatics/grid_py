"""Tests for gradient rendering, pattern tiling, mask, and grid_locator.

Covers all grid_py features implemented in the gradient/pattern session:
- LinearGradient / RadialGradient fill rendering
- group=True vs group=False gradient resolution
- PatternList vectorized fills
- Pattern tiling fill
- grid_locator coordinate conversion
- Mask rendering (render_mask / apply_mask)
"""

import math
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Gradient rendering
# ---------------------------------------------------------------------------

class TestGradientRendering:
    """Test LinearGradient and RadialGradient fill rendering."""

    def test_linear_gradient_rect(self):
        from grid_py import rect_grob, Gpar, grid_draw, grid_newpage
        from grid_py._patterns import linear_gradient
        from grid_py._state import get_state

        grad = linear_gradient(colours=["red", "blue"])
        r = rect_grob(x=0.5, y=0.5, width=0.8, height=0.8,
                       gp=Gpar(fill=grad, col="black"))
        grid_newpage(width=3, height=3, dpi=100)
        grid_draw(r)
        # If no error, the gradient rendered successfully
        renderer = get_state().get_renderer()
        assert renderer is not None

    def test_radial_gradient_circle(self):
        from grid_py import circle_grob, Gpar, grid_draw, grid_newpage
        from grid_py._patterns import radial_gradient
        from grid_py._state import get_state

        grad = radial_gradient(colours=["yellow", "red", "black"])
        c = circle_grob(x=0.5, y=0.5, r=0.3,
                         gp=Gpar(fill=grad, col="black"))
        grid_newpage(width=3, height=3, dpi=100)
        grid_draw(c)
        renderer = get_state().get_renderer()
        assert renderer is not None

    def test_group_true_viewport_resolution(self):
        """group=True: gradient resolved relative to viewport."""
        from grid_py import rect_grob, Gpar, grid_draw, grid_newpage
        from grid_py._patterns import linear_gradient

        grad = linear_gradient(colours=["red", "blue"],
                                x1=0.0, y1=0.5, x2=1.0, y2=0.5,
                                group=True)
        # Two rects at different x positions should show different slices
        r = rect_grob(x=[0.25, 0.75], y=[0.5, 0.5],
                       width=[0.3, 0.3], height=[0.8, 0.8],
                       gp=Gpar(fill=grad, col="black"))
        grid_newpage(width=4, height=2, dpi=100)
        grid_draw(r)  # Should not raise

    def test_group_false_per_shape_resolution(self):
        """group=False: gradient resolved per shape."""
        from grid_py import rect_grob, Gpar, grid_draw, grid_newpage
        from grid_py._patterns import linear_gradient

        grad = linear_gradient(colours=["red", "blue"],
                                x1=0.0, y1=0.5, x2=1.0, y2=0.5,
                                group=False)
        r = rect_grob(x=[0.25, 0.75], y=[0.5, 0.5],
                       width=[0.3, 0.3], height=[0.8, 0.8],
                       gp=Gpar(fill=grad, col="black"))
        grid_newpage(width=4, height=2, dpi=100)
        grid_draw(r)  # Each rect gets its own full gradient

    def test_pattern_list(self):
        """PatternList: each vectorized element gets a different gradient."""
        from grid_py import rect_grob, Gpar, grid_draw, grid_newpage
        from grid_py._patterns import linear_gradient

        g1 = linear_gradient(colours=["red", "white"])
        g2 = linear_gradient(colours=["blue", "yellow"])
        r = rect_grob(x=[0.25, 0.75], y=[0.5, 0.5],
                       width=[0.3, 0.3], height=[0.8, 0.8],
                       gp=Gpar(fill=[g1, g2], col="black"))
        grid_newpage(width=4, height=2, dpi=100)
        grid_draw(r)  # First rect red→white, second blue→yellow


# ---------------------------------------------------------------------------
# Pattern tiling
# ---------------------------------------------------------------------------

class TestPatternTiling:
    """Test Pattern tiling fill."""

    def test_pattern_tiling_renders(self):
        from grid_py import rect_grob, circle_grob, Gpar, grid_draw, grid_newpage
        from grid_py._patterns import pattern
        from grid_py._grob import grob_tree

        tile = circle_grob(x=0.5, y=0.5, r=0.3,
                            gp=Gpar(fill="red", col="blue"))
        pat = pattern(tile, width=0.1, height=0.1, extend="repeat")
        r = rect_grob(x=0.5, y=0.5, width=0.8, height=0.8,
                       gp=Gpar(fill=pat, col="black"))
        grid_newpage(width=3, height=3, dpi=100)
        grid_draw(r)  # Should render tiled circles


# ---------------------------------------------------------------------------
# grid_locator
# ---------------------------------------------------------------------------

class TestGridLocator:
    """Test grid_locator coordinate conversion."""

    def test_npc_center(self):
        from grid_py import grid_newpage, grid_locator, Viewport, push_viewport
        from grid_py._state import get_state

        grid_newpage(width=4, height=3, dpi=100)
        renderer = get_state().get_renderer()
        cx = renderer._width_px / 2
        cy = renderer._height_px / 2

        result = grid_locator("npc", x_device=cx, y_device=cy)
        assert result is not None
        assert abs(result["x"] - 0.5) < 0.01
        assert abs(result["y"] - 0.5) < 0.01

    def test_native_with_viewport(self):
        from grid_py import grid_newpage, grid_locator, Viewport, push_viewport
        from grid_py._state import get_state

        grid_newpage(width=4, height=3, dpi=100)
        vp = Viewport(xscale=[0, 100], yscale=[0, 50])
        push_viewport(vp)

        renderer = get_state().get_renderer()
        cx = renderer._width_px / 2
        cy = renderer._height_px / 2

        result = grid_locator("native", x_device=cx, y_device=cy)
        assert result is not None
        assert abs(result["x"] - 50.0) < 1.0
        assert abs(result["y"] - 25.0) < 1.0

    def test_none_returns_none(self):
        import pytest
        from grid_py import grid_newpage, grid_locator
        grid_newpage(width=3, height=3, dpi=100)
        with pytest.warns(UserWarning, match="not supported in non-interactive"):
            assert grid_locator("npc") is None

    def test_cm_conversion(self):
        from grid_py import grid_newpage, grid_locator
        from grid_py._state import get_state

        grid_newpage(width=4, height=3, dpi=100)
        renderer = get_state().get_renderer()
        result = grid_locator("cm", x_device=0, y_device=0)
        assert result is not None
        assert result["x"] == 0.0


# ---------------------------------------------------------------------------
# Mask rendering
# ---------------------------------------------------------------------------

class TestMaskRendering:
    """Test mask rendering infrastructure."""

    def test_render_mask(self):
        import cairo
        from grid_py import circle_grob, Gpar, grid_newpage
        from grid_py._state import get_state

        grid_newpage(width=3, height=3, dpi=100)
        mask_grob = circle_grob(x=0.5, y=0.5, r=0.3,
                                 gp=Gpar(fill="white", col=None))
        renderer = get_state().get_renderer()
        surface = renderer.render_mask(mask_grob)
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() > 0
        assert surface.get_height() > 0

    def test_apply_mask(self):
        import cairo
        from grid_py import rect_grob, circle_grob, Gpar, grid_draw, grid_newpage
        from grid_py._state import get_state

        grid_newpage(width=3, height=3, dpi=100)
        # Draw background
        bg = rect_grob(x=0.5, y=0.5, width=0.9, height=0.9,
                        gp=Gpar(fill="red", col=None))
        grid_draw(bg)

        # Render and apply mask
        mask_grob = circle_grob(x=0.5, y=0.5, r=0.3,
                                 gp=Gpar(fill="white", col=None))
        renderer = get_state().get_renderer()
        mask_surface = renderer.render_mask(mask_grob)
        assert mask_surface is not None
        # apply_mask should not raise
        renderer.apply_mask(mask_surface, mask_type="alpha")
