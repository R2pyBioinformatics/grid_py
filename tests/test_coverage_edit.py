"""Coverage tests for grid_py._edit -- grid_edit, grid_get, grid_set,
grid_add, grid_remove, and convenience aliases.

NOTE: All public functions in _edit.py access state.display_list which is
buggy (should be state.get_display_list()). We test the internal helpers
that do NOT depend on that, and test the public API by manually setting up
state._display_list.
"""
from __future__ import annotations

import warnings
import pytest

import grid_py
from grid_py._edit import (
    _ensure_gpath,
    _name_match,
    _find_dl_grobs,
    _match_grob_path,
    grid_edit,
    grid_get,
    grid_set,
    grid_add,
    grid_remove,
    grid_gedit,
    grid_gget,
    grid_gremove,
)
from grid_py._grob import Grob, GTree, GList
from grid_py._gpar import Gpar
from grid_py._path import GPath
from grid_py._display_list import DisplayList, DLDrawGrob
from grid_py._state import get_state


# ---------------------------------------------------------------------------
# _ensure_gpath
# ---------------------------------------------------------------------------


class TestEnsureGpath:
    def test_string(self):
        result = _ensure_gpath("mygrob")
        assert isinstance(result, GPath)
        assert result.name == "mygrob"

    def test_gpath(self):
        gp = GPath("a", "b")
        result = _ensure_gpath(gp)
        assert result is gp

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="expected str or GPath"):
            _ensure_gpath(42)


# ---------------------------------------------------------------------------
# _name_match
# ---------------------------------------------------------------------------


class TestNameMatch:
    def test_exact_match(self):
        result = _name_match("abc", ["abc", "def"], grep=False)
        assert result == 0

    def test_exact_no_match(self):
        result = _name_match("xyz", ["abc", "def"], grep=False)
        assert result is None

    def test_grep_match(self):
        result = _name_match("ab", ["abc", "def"], grep=True)
        assert result == 0

    def test_grep_no_match(self):
        result = _name_match("xyz", ["abc", "def"], grep=True)
        assert result is None


# ---------------------------------------------------------------------------
# _match_grob_path
# ---------------------------------------------------------------------------


class TestMatchGrobPath:
    def test_single_level_exact(self):
        g = Grob(name="target", _grid_class="rect")
        result = _match_grob_path(g, GPath("target"), strict=False, grep_flags=[False])
        assert result is g

    def test_single_level_no_match(self):
        g = Grob(name="other", _grid_class="rect")
        result = _match_grob_path(g, GPath("target"), strict=False, grep_flags=[False])
        assert result is None

    def test_single_level_grep(self):
        g = Grob(name="myRect123", _grid_class="rect")
        result = _match_grob_path(g, GPath("myRect"), strict=False, grep_flags=[True])
        assert result is g

    def test_single_level_strict_no_recurse(self):
        child = Grob(name="child", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = _match_grob_path(tree, GPath("child"), strict=True, grep_flags=[False])
        assert result is None  # strict prevents recursion

    def test_single_level_non_strict_recurse_into_gtree(self):
        child = Grob(name="child", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = _match_grob_path(tree, GPath("child"), strict=False, grep_flags=[False])
        assert result is not None
        assert result.name == "child"

    def test_multi_level(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = _match_grob_path(tree, GPath("tree", "c1"), strict=False, grep_flags=[False, False])
        assert result is not None
        assert result.name == "c1"

    def test_multi_level_no_match(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = _match_grob_path(tree, GPath("tree", "nonexistent"), strict=False, grep_flags=[False, False])
        assert result is None

    def test_non_grob_returns_none(self):
        result = _match_grob_path("not_a_grob", GPath("x"), strict=False, grep_flags=[False])
        assert result is None

    def test_multi_level_non_strict_skip(self):
        inner_child = Grob(name="target", _grid_class="rect")
        inner_tree = GTree(children=GList(inner_child), name="mid")
        outer = GTree(children=GList(inner_tree), name="outer")
        result = _match_grob_path(outer, GPath("mid", "target"), strict=False, grep_flags=[False, False])
        assert result is not None
        assert result.name == "target"


# ---------------------------------------------------------------------------
# _find_dl_grobs
# ---------------------------------------------------------------------------


class TestFindDlGrobs:
    def test_find_in_display_list(self):
        dl = DisplayList()
        g = Grob(name="target", _grid_class="rect")
        dl.record(DLDrawGrob(grob=g))
        result = _find_dl_grobs(dl, GPath("target"), strict=False,
                                grep=False, global_=False)
        assert len(result) == 1
        assert result[0][1].name == "target"

    def test_find_global(self):
        dl = DisplayList()
        g1 = Grob(name="match1", _grid_class="rect")
        g2 = Grob(name="match2", _grid_class="rect")
        dl.record(DLDrawGrob(grob=g1))
        dl.record(DLDrawGrob(grob=g2))
        result = _find_dl_grobs(dl, GPath("match.*"), strict=False,
                                grep=True, global_=True)
        assert len(result) == 2

    def test_find_no_match(self):
        dl = DisplayList()
        g = Grob(name="other", _grid_class="rect")
        dl.record(DLDrawGrob(grob=g))
        result = _find_dl_grobs(dl, GPath("target"), strict=False,
                                grep=False, global_=False)
        assert len(result) == 0

    def test_skip_none_grobs(self):
        dl = DisplayList()
        dl.record(DLDrawGrob(grob=None))
        result = _find_dl_grobs(dl, GPath("any"), strict=False,
                                grep=True, global_=True)
        assert len(result) == 0

    def test_grep_flag_list(self):
        dl = DisplayList()
        g = Grob(name="target", _grid_class="rect")
        dl.record(DLDrawGrob(grob=g))
        result = _find_dl_grobs(dl, GPath("target"), strict=False,
                                grep=[False], global_=False)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Public API -- using monkeypatch to set state._display_list
# We patch state to have a display_list attribute that's a DisplayList
# ---------------------------------------------------------------------------


def _setup_state_with_dl(grobs):
    """Helper to set up state with a display list containing given grobs.
    We need to patch state to have display_list attribute since the code
    accesses state.display_list directly."""
    state = get_state()
    state.reset()
    dl = DisplayList()
    for g in grobs:
        dl.record(DLDrawGrob(grob=g))
    # Patch: the code accesses state.display_list
    state.display_list = dl
    return state


class TestGridGet:
    def test_get_existing(self):
        g = Grob(name="findme", _grid_class="rect")
        _setup_state_with_dl([g])
        result = grid_get("findme")
        assert result is not None
        assert result.name == "findme"

    def test_get_not_found(self):
        g = Grob(name="other", _grid_class="rect")
        _setup_state_with_dl([g])
        result = grid_get("nonexistent")
        assert result is None

    def test_get_global(self):
        g1 = Grob(name="match1", _grid_class="rect")
        g2 = Grob(name="match2", _grid_class="rect")
        _setup_state_with_dl([g1, g2])
        result = grid_get("match.*", grep=True, global_=True)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_all_devices_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(NotImplementedError):
            grid_get("test", allDevices=True)

    def test_get_invalid_grep_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(TypeError, match="invalid 'grep'"):
            grid_get("test", grep="not_bool")

    def test_gget(self):
        g = Grob(name="findme", _grid_class="rect")
        _setup_state_with_dl([g])
        result = grid_gget("find.*")
        assert isinstance(result, list)
        assert len(result) >= 1


class TestGridEdit:
    def test_edit_attribute(self):
        g = Grob(name="editable", _grid_class="rect", custom=1)
        _setup_state_with_dl([g])
        grid_edit("editable", custom=99, redraw=False)
        assert g.custom == 99

    def test_edit_gp(self):
        g = Grob(name="editable", _grid_class="rect")
        _setup_state_with_dl([g])
        grid_edit("editable", gp=Gpar(col="blue"), redraw=False)

    def test_edit_name(self):
        g = Grob(name="editable", _grid_class="rect")
        _setup_state_with_dl([g])
        grid_edit("editable", name="new_name", redraw=False)
        assert g.name == "new_name"

    def test_edit_vp(self):
        g = Grob(name="editable", _grid_class="rect")
        _setup_state_with_dl([g])
        grid_edit("editable", vp="some_path", redraw=False)

    def test_edit_unknown_slot_warns(self):
        g = Grob(name="editable", _grid_class="rect")
        _setup_state_with_dl([g])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_edit("editable", nonexistent=42, redraw=False)
            assert any("not found" in str(x.message) for x in w)

    def test_edit_all_devices_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(NotImplementedError):
            grid_edit("test", allDevices=True)

    def test_edit_invalid_grep_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(TypeError, match="invalid 'grep'"):
            grid_edit("test", grep="not_bool")

    def test_gedit(self):
        g = Grob(name="editable", _grid_class="rect", custom=1)
        _setup_state_with_dl([g])
        grid_gedit("edit.*", custom=42, redraw=False)
        assert g.custom == 42


class TestGridSet:
    def test_set_grob(self):
        g = Grob(name="replaceable", _grid_class="rect")
        _setup_state_with_dl([g])
        new_g = Grob(name="replacement", _grid_class="circle")
        grid_set("replaceable", new_g, redraw=False)

    def test_set_not_found_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(ValueError, match="does not specify"):
            grid_set("nonexistent", Grob(name="new"), redraw=False)

    def test_set_invalid_grep_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(TypeError, match="invalid 'grep'"):
            grid_set("test", Grob(name="new"), grep="not_bool")


class TestGridAdd:
    def test_add_top_level(self):
        state = _setup_state_with_dl([])
        new_g = Grob(name="added", _grid_class="rect")
        grid_add(new_g, redraw=False)

    def test_add_to_gtree(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        _setup_state_with_dl([tree])
        new_g = Grob(name="c2", _grid_class="circle")
        grid_add(new_g, gPath="tree", redraw=False)
        assert tree.n_children() == 2

    def test_add_to_non_gtree_warns(self):
        g = Grob(name="plain", _grid_class="rect")
        _setup_state_with_dl([g])
        new_g = Grob(name="c1", _grid_class="circle")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_add(new_g, gPath="plain", redraw=False)
            assert any("non-gTree" in str(x.message) for x in w)

    def test_add_all_devices_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(NotImplementedError):
            grid_add(Grob(), allDevices=True)

    def test_add_invalid_grep_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(TypeError, match="invalid 'grep'"):
            grid_add(Grob(), gPath="test", grep="not_bool")


class TestGridRemove:
    def test_remove_single(self):
        g = Grob(name="removable", _grid_class="rect")
        state = _setup_state_with_dl([g])
        grid_remove("removable", redraw=False)
        # Check removed
        dl = state.display_list
        remaining = [item for item in dl if isinstance(item, DLDrawGrob) and item.grob is not None]
        assert len(remaining) == 0

    def test_remove_not_found_warns(self):
        _setup_state_with_dl([])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_remove("nonexistent", warn=True, redraw=False)
            assert any("not found" in str(x.message) for x in w)

    def test_remove_multi_level(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        _setup_state_with_dl([tree])
        grid_remove(GPath("tree", "c1"), redraw=False)
        assert tree.n_children() == 0

    def test_remove_multi_level_not_found_warns(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        _setup_state_with_dl([tree])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_remove(GPath("tree", "nonexistent"), warn=True, redraw=False)
            assert any("not found" in str(x.message) for x in w)

    def test_remove_grep(self):
        g1 = Grob(name="match1", _grid_class="rect")
        g2 = Grob(name="match2", _grid_class="rect")
        state = _setup_state_with_dl([g1, g2])
        grid_remove("match.*", grep=True, global_=True, redraw=False)

    def test_remove_all_devices_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(NotImplementedError):
            grid_remove("test", allDevices=True)

    def test_remove_invalid_grep_raises(self):
        _setup_state_with_dl([])
        with pytest.raises(TypeError, match="invalid 'grep'"):
            grid_remove("test", grep="not_bool")

    def test_gremove(self):
        g = Grob(name="removable", _grid_class="rect")
        state = _setup_state_with_dl([g])
        grid_gremove("remov.*", redraw=False)

    def test_remove_multi_level_grep(self):
        child1 = Grob(name="child_a", _grid_class="rect")
        child2 = Grob(name="child_b", _grid_class="rect")
        tree = GTree(children=GList(child1, child2), name="tree")
        _setup_state_with_dl([tree])
        grid_remove(GPath("tree", "child_.*"), grep=True, global_=True, redraw=False)
        assert tree.n_children() == 0

    def test_remove_multi_level_non_gtree_parent(self):
        """Multi-level remove where parent is not a GTree -- should skip."""
        g = Grob(name="plain", _grid_class="rect")
        _setup_state_with_dl([g])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_remove(GPath("plain", "child"), warn=True, redraw=False)
            # Should warn about not found
            assert any("not found" in str(x.message) for x in w)


# Cleanup: remove patched display_list after each test
@pytest.fixture(autouse=True)
def _cleanup_display_list():
    yield
    state = get_state()
    if hasattr(state, "display_list"):
        try:
            delattr(state, "display_list")
        except AttributeError:
            pass
