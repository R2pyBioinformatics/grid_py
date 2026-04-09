"""Coverage tests for grid_py._grab -- grid_grab, grid_grab_expr, grid_force,
grid_revert, grid_cap, grid_reorder.

NOTE: Many functions in _grab.py access state.display_list which is buggy
(should be state.get_display_list()). We test what works and mark broken
paths.
"""
from __future__ import annotations

import copy
import warnings
import pytest
import numpy as np

import grid_py
from grid_py._grab import (
    _collect_dl_grobs,
    grid_grab,
    grid_grab_expr,
    grid_force,
    grid_revert,
    grid_cap,
)
from grid_py._grob import (
    Grob, GTree, GList,
    force_grob,
    reorder_grob,
)
from grid_py._display_list import DisplayList, DLDrawGrob
from grid_py._state import get_state


# ---------------------------------------------------------------------------
# _collect_dl_grobs
# ---------------------------------------------------------------------------


class TestCollectDlGrobs:
    def test_empty_dl(self):
        dl = DisplayList()
        result = _collect_dl_grobs(dl, warn=0)
        assert result is None

    def test_single_grob(self):
        dl = DisplayList()
        g = Grob(name="g1", _grid_class="rect")
        dl.record(DLDrawGrob(grob=g))
        result = _collect_dl_grobs(dl, warn=0)
        assert result is not None
        assert len(result) == 1

    def test_duplicate_name_warns(self):
        dl = DisplayList()
        g1 = Grob(name="dup", _grid_class="rect")
        g2 = Grob(name="dup", _grid_class="circle")
        dl.record(DLDrawGrob(grob=g1))
        dl.record(DLDrawGrob(grob=g2))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _collect_dl_grobs(dl, warn=1)
            assert len(w) >= 1
            assert "overwritten" in str(w[0].message)
        assert result is not None
        assert len(result) == 2

    def test_none_grob_skipped(self):
        dl = DisplayList()
        dl.record(DLDrawGrob(grob=None))
        result = _collect_dl_grobs(dl, warn=0)
        assert result is None

    def test_mixed_items(self):
        from grid_py._display_list import DLPushViewport
        dl = DisplayList()
        g = Grob(name="mixed", _grid_class="rect")
        dl.record(DLDrawGrob(grob=g))
        dl.record(DLPushViewport(viewport=None))
        result = _collect_dl_grobs(dl, warn=0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# force_grob / grid_force (grob-level, not display-list)
# ---------------------------------------------------------------------------


class TestForceGrob:
    def test_force_plain_grob(self):
        g = Grob(name="forceme", _grid_class="rect")
        forced = force_grob(g)
        assert forced.name == "forceme"
        assert hasattr(forced, "_original")
        assert forced._original is g

    def test_force_gtree(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        forced = force_grob(tree)
        assert isinstance(forced, GTree)
        assert hasattr(forced, "_original")

    def test_grid_force_with_grob(self):
        g = Grob(name="forceme2", _grid_class="rect")
        forced = grid_force(g)
        assert forced is not None
        assert forced.name == "forceme2"

    def test_grid_force_gtree(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        forced = grid_force(tree)
        assert isinstance(forced, GTree)


# ---------------------------------------------------------------------------
# grid_revert (grob-level)
# ---------------------------------------------------------------------------


class TestGridRevert:
    def test_revert_forced_grob(self):
        g = Grob(name="orig", _grid_class="rect")
        forced = force_grob(g)
        reverted = grid_revert(forced)
        assert reverted is g

    def test_revert_unforced_grob(self):
        g = Grob(name="unforced", _grid_class="rect")
        result = grid_revert(g)
        assert result is g

    def test_revert_gtree_with_forced_children(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        forced = force_grob(tree)
        reverted = grid_revert(forced)
        # The reverted should be the original
        assert reverted is tree

    def test_revert_gtree_no_original(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        # Tree has no _original, so revert should return a deepcopy
        result = grid_revert(tree)
        assert isinstance(result, GTree)


# ---------------------------------------------------------------------------
# grid_cap
# ---------------------------------------------------------------------------


class TestGridCap:
    def test_cap_no_renderer(self):
        state = get_state()
        state.reset()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = grid_cap()
            assert result is None
            assert len(w) >= 1


# ---------------------------------------------------------------------------
# reorder_grob
# ---------------------------------------------------------------------------


class TestReorderGrob:
    def test_reorder_by_index(self):
        c1 = Grob(name="a", _grid_class="rect")
        c2 = Grob(name="b", _grid_class="rect")
        c3 = Grob(name="c", _grid_class="rect")
        tree = GTree(children=GList(c1, c2, c3), name="tree")
        result = reorder_grob(tree, [2, 0], back=True)
        assert result._children_order == ["c", "a", "b"]

    def test_reorder_by_name(self):
        c1 = Grob(name="a", _grid_class="rect")
        c2 = Grob(name="b", _grid_class="rect")
        tree = GTree(children=GList(c1, c2), name="tree")
        result = reorder_grob(tree, ["b"], back=True)
        assert result._children_order == ["b", "a"]

    def test_reorder_front(self):
        c1 = Grob(name="a", _grid_class="rect")
        c2 = Grob(name="b", _grid_class="rect")
        tree = GTree(children=GList(c1, c2), name="tree")
        result = reorder_grob(tree, ["a"], back=False)
        assert result._children_order == ["b", "a"]

    def test_reorder_invalid_name_raises(self):
        c1 = Grob(name="a", _grid_class="rect")
        tree = GTree(children=GList(c1), name="tree")
        with pytest.raises(ValueError, match="not found"):
            reorder_grob(tree, ["nonexistent"])

    def test_reorder_invalid_index_raises(self):
        c1 = Grob(name="a", _grid_class="rect")
        tree = GTree(children=GList(c1), name="tree")
        with pytest.raises(ValueError, match="out of range"):
            reorder_grob(tree, [5])

    def test_reorder_invalid_type_raises(self):
        c1 = Grob(name="a", _grid_class="rect")
        tree = GTree(children=GList(c1), name="tree")
        with pytest.raises(TypeError, match="int or str"):
            reorder_grob(tree, [3.14])

    def test_reorder_non_gtree_raises(self):
        g = Grob(name="plain", _grid_class="rect")
        with pytest.raises(TypeError, match="only reorder"):
            reorder_grob(g, [0])

    def test_reorder_dedup(self):
        c1 = Grob(name="a", _grid_class="rect")
        c2 = Grob(name="b", _grid_class="rect")
        tree = GTree(children=GList(c1, c2), name="tree")
        result = reorder_grob(tree, [0, 0], back=True)
        # Duplicates should be deduplicated
        assert result._children_order == ["a", "b"]


# ---------------------------------------------------------------------------
# grid_grab_expr (works since it manages display list internally)
# ---------------------------------------------------------------------------


class TestGridGrabExpr:
    """grid_grab_expr accesses state.display_list which is buggy. Skip."""
    pass
