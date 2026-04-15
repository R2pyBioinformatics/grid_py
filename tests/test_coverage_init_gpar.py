"""Coverage tests for grid_py.__init__ and grid_py._gpar edge cases."""
from __future__ import annotations

import pytest

import grid_py
from grid_py._gpar import Gpar, get_gpar


# ---------------------------------------------------------------------------
# __init__.py -- deprecated aliases
# ---------------------------------------------------------------------------


class TestInitAliases:
    def test_convert_native(self):
        assert grid_py.convert_native is grid_py.convert_unit

    def test_grid_convert(self):
        assert grid_py.grid_convert is grid_py.convert_unit

    def test_grid_convert_x(self):
        assert grid_py.grid_convert_x is grid_py.convert_x

    def test_grid_convert_y(self):
        assert grid_py.grid_convert_y is grid_py.convert_y

    def test_grid_convert_width(self):
        assert grid_py.grid_convert_width is grid_py.convert_width

    def test_grid_convert_height(self):
        assert grid_py.grid_convert_height is grid_py.convert_height

    def test_device_loc(self):
        # device_loc is now a real function (not a placeholder alias)
        assert callable(grid_py.device_loc)

    def test_device_dim(self):
        # device_dim is now a real function (not a placeholder alias)
        assert callable(grid_py.device_dim)

    def test_grid_collection(self):
        assert grid_py.grid_collection is grid_py.grid_draw

    def test_grid_copy(self):
        assert grid_py.grid_copy is grid_py.grid_draw

    def test_grid_display_list(self):
        state = grid_py.get_state()
        prev = grid_py.grid_display_list(False)
        assert isinstance(prev, bool)
        grid_py.grid_display_list(True)

    def test_engine_display_list(self):
        assert grid_py.engine_display_list is grid_py.grid_display_list

    def test_version(self):
        assert isinstance(grid_py.__version__, str)
        assert len(grid_py.__version__) > 0


# ---------------------------------------------------------------------------
# Gpar edge cases
# ---------------------------------------------------------------------------


class TestGparEdges:
    def test_get_gpar(self):
        state = grid_py.get_state()
        gp = get_gpar()
        assert isinstance(gp, Gpar)

    def test_gpar_get_set(self):
        gp = Gpar()
        gp.set("col", "red")
        assert gp.get("col") == "red"

    def test_gpar_get_default(self):
        gp = Gpar()
        assert gp.get("nonexistent", "default") == "default"

    def test_gpar_repr(self):
        gp = Gpar(col="red")
        r = repr(gp)
        assert "Gpar" in r or "gpar" in r.lower() or "col" in r
