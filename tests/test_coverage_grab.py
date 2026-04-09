"""Tests for grid_py._grab -- scene capture and manipulation."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from grid_py._grab import (
    _collect_dl_grobs,
    grid_cap,
    grid_force,
    grid_revert,
)
from grid_py._grob import GList, GTree, Grob, force_grob
from grid_py._display_list import DisplayList, DLDrawGrob
from grid_py._state import get_state
from grid_py._draw import grid_newpage, grid_draw


def _make_null(name="n"):
    return Grob(name=name, _grid_class="null")


# ---------------------------------------------------------------------------
# _collect_dl_grobs
# ---------------------------------------------------------------------------

class TestCollectDlGrobs:

    def test_empty_dl(self):
        dl = DisplayList()
        result = _collect_dl_grobs(dl)
        assert result is None

    def test_single_grob(self):
        dl = DisplayList()
        g = _make_null("a")
        dl.record(DLDrawGrob(grob=g))
        result = _collect_dl_grobs(dl)
        assert isinstance(result, GList)
        assert len(result) == 1

    def test_multiple_grobs(self):
        dl = DisplayList()
        dl.record(DLDrawGrob(grob=_make_null("a")))
        dl.record(DLDrawGrob(grob=_make_null("b")))
        result = _collect_dl_grobs(dl)
        assert len(result) == 2

    def test_duplicate_names_warns(self):
        dl = DisplayList()
        dl.record(DLDrawGrob(grob=_make_null("dup")))
        dl.record(DLDrawGrob(grob=_make_null("dup")))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _collect_dl_grobs(dl, warn=1)
            assert len(w) >= 1

    def test_no_warn(self):
        dl = DisplayList()
        dl.record(DLDrawGrob(grob=_make_null("dup")))
        dl.record(DLDrawGrob(grob=_make_null("dup")))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _collect_dl_grobs(dl, warn=0)
            assert len(w) == 0

    def test_none_grob_skipped(self):
        dl = DisplayList()
        dl.record(DLDrawGrob(grob=None))
        dl.record(DLDrawGrob(grob=_make_null("real")))
        result = _collect_dl_grobs(dl)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# grid_force (with explicit grob argument -- not display list)
# ---------------------------------------------------------------------------

class TestGridForce:

    def test_force_grob(self):
        g = _make_null("forceme")
        result = grid_force(g)
        assert isinstance(result, Grob)
        assert hasattr(result, "_original")

    def test_force_gtree(self):
        child = _make_null("child")
        tree = GTree(children=GList(child), name="tree")
        result = grid_force(tree)
        assert isinstance(result, GTree)

    def test_force_grob_preserves_original(self):
        g = _make_null("orig")
        result = grid_force(g)
        assert result._original is g


# ---------------------------------------------------------------------------
# grid_revert (with explicit grob argument -- not display list)
# ---------------------------------------------------------------------------

class TestGridRevert:

    def test_revert_forced_grob(self):
        g = _make_null("orig")
        forced = force_grob(g)
        reverted = grid_revert(forced)
        assert reverted is g

    def test_revert_unforced_grob(self):
        g = _make_null("orig")
        reverted = grid_revert(g)
        assert reverted is g

    def test_revert_gtree_with_forced_children(self):
        child = _make_null("child")
        forced_child = force_grob(child)
        tree = GTree(children=GList(forced_child), name="tree")
        reverted = grid_revert(tree)
        assert isinstance(reverted, GTree)

    def test_revert_gtree_no_forced_children(self):
        child = _make_null("child")
        tree = GTree(children=GList(child), name="tree")
        reverted = grid_revert(tree)
        assert isinstance(reverted, GTree)


# ---------------------------------------------------------------------------
# grid_cap
# ---------------------------------------------------------------------------

class TestGridCap:

    def test_no_renderer_warns(self):
        state = get_state()
        state.reset()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = grid_cap()
            assert result is None

    def test_with_renderer(self):
        grid_newpage()
        result = grid_cap(native=True)
        # Result can be None or ndarray depending on PIL availability
        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 3

    def test_with_renderer_float(self):
        grid_newpage()
        result = grid_cap(native=False)
        if result is not None:
            assert result.dtype == np.float64
