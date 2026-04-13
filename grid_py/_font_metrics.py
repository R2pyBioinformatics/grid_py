"""Pluggable text measurement backends for grid_py.

Provides font metrics implementations that renderers can use when they
lack native text measurement capabilities (e.g. the WebRenderer).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

__all__ = [
    "FontMetricsBackend",
    "CairoFontMetrics",
    "HeuristicMetrics",
    "get_font_backend",
]

# Module-level cache for the default backend.
_cached_backend: Optional[FontMetricsBackend] = None


class FontMetricsBackend(ABC):
    """Abstract interface for text measurement."""

    @abstractmethod
    def measure(self, text: str, gp: Any = None) -> Dict[str, float]:
        """Return ``{'ascent', 'descent', 'width'}`` in inches."""


# ---------------------------------------------------------------------------
# Helpers shared by concrete backends
# ---------------------------------------------------------------------------

def _extract_font_params(gp: Any) -> tuple:
    """Pull fontfamily, fontsize, fontface, and cex from a Gpar-like object.

    Returns (family, fontsize, fontface, cex).
    """
    family: Optional[str] = None
    fontsize: float = 12.0
    fontface: Any = None
    cex: float = 1.0

    if gp is not None:
        val = gp.get("fontfamily", None)
        if val is not None:
            family = str(val)
        val = gp.get("fontsize", None)
        if val is not None:
            fontsize = float(val)
        val = gp.get("fontface", None)
        if val is not None:
            fontface = val
        val = gp.get("cex", None)
        if val is not None:
            cex = float(val)

    return family, fontsize, fontface, cex


# ---------------------------------------------------------------------------
# Cairo backend
# ---------------------------------------------------------------------------

class CairoFontMetrics(FontMetricsBackend):
    """Text measurement using a Cairo context."""

    def __init__(self) -> None:
        import cairo  # noqa: F811 – runtime import
        self._cairo = cairo
        self._surface = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
        self._ctx = cairo.Context(self._surface)

    def measure(self, text: str, gp: Any = None) -> Dict[str, float]:
        cairo = self._cairo
        ctx = self._ctx

        family, fontsize, fontface, cex = _extract_font_params(gp)

        slant = cairo.FONT_SLANT_NORMAL
        weight = cairo.FONT_WEIGHT_NORMAL

        if fontface is not None:
            if fontface == 2 or fontface == "bold":
                weight = cairo.FONT_WEIGHT_BOLD
            elif fontface == 3 or fontface == "italic":
                slant = cairo.FONT_SLANT_ITALIC
            elif fontface == 4 or fontface == "bold.italic":
                weight = cairo.FONT_WEIGHT_BOLD
                slant = cairo.FONT_SLANT_ITALIC

        ctx.select_font_face(family or "sans-serif", slant, weight)
        ctx.set_font_size(fontsize * cex)

        fe = ctx.font_extents()  # (ascent, descent, height, …)
        te = ctx.text_extents(text)  # (x_bearing, y_bearing, width, height, …)

        return {
            "ascent": fe[0] / 72.0,
            "descent": fe[1] / 72.0,
            "width": te[2] / 72.0,
        }


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

class HeuristicMetrics(FontMetricsBackend):
    """Rough estimates based on character count and font size."""

    def measure(self, text: str, gp: Any = None) -> Dict[str, float]:
        _family, fontsize, _fontface, cex = _extract_font_params(gp)
        effective = fontsize * cex
        avg_char_width = effective * 0.6 / 72.0

        return {
            "ascent": effective * 0.75 / 72.0,
            "descent": effective * 0.25 / 72.0,
            "width": len(text) * avg_char_width,
        }


# ---------------------------------------------------------------------------
# Factory / singleton accessor
# ---------------------------------------------------------------------------

def get_font_backend() -> FontMetricsBackend:
    """Return a cached :class:`FontMetricsBackend` instance.

    Prefers :class:`CairoFontMetrics` when PyCairo is available, otherwise
    falls back to :class:`HeuristicMetrics`.
    """
    global _cached_backend
    if _cached_backend is not None:
        return _cached_backend

    try:
        _cached_backend = CairoFontMetrics()
    except ImportError:
        _cached_backend = HeuristicMetrics()

    return _cached_backend
