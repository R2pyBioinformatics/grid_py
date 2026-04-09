"""Coverage tests for grid_py._ls -- grid_ls, grid_grep, nested_listing,
path_listing, grob_path_listing, show_grob, get_names, child_names.
"""
from __future__ import annotations

import warnings
import pytest

import grid_py
from grid_py._ls import (
    FlatGridListing,
    GridListingEntry,
    grid_ls,
    grid_grep,
    nested_listing,
    path_listing,
    grob_path_listing,
    show_grob,
    get_names,
    child_names,
    _inc_path,
    _list_grob,
)
from grid_py._grob import Grob, GTree, GList, GEdit, GEditList
from grid_py._gpar import Gpar
from grid_py._path import GPath


# ---------------------------------------------------------------------------
# FlatGridListing
# ---------------------------------------------------------------------------


class TestFlatGridListing:
    def test_empty(self):
        fl = FlatGridListing()
        assert len(fl) == 0
        assert repr(fl) == "FlatGridListing(n=0)"

    def test_append_entry(self):
        fl = FlatGridListing()
        entry = GridListingEntry(name="test", g_depth=1, vp_depth=2,
                                 g_path="p", vp_path="vp", entry_type="grobListing")
        fl.append_entry(entry)
        assert len(fl) == 1
        assert fl.names[0] == "test"
        assert fl.g_depths[0] == 1
        assert fl.vp_depths[0] == 2
        assert fl.g_paths[0] == "p"
        assert fl.vp_paths[0] == "vp"
        assert fl.types[0] == "grobListing"

    def test_extend(self):
        fl1 = FlatGridListing()
        fl1.append_entry(GridListingEntry(name="a"))
        fl2 = FlatGridListing()
        fl2.append_entry(GridListingEntry(name="b"))
        fl1.extend(fl2)
        assert len(fl1) == 2
        assert fl1.names == ["a", "b"]

    def test_subset(self):
        fl = FlatGridListing()
        fl.append_entry(GridListingEntry(name="a"))
        fl.append_entry(GridListingEntry(name="b"))
        fl.append_entry(GridListingEntry(name="c"))
        sub = fl.subset([0, 2])
        assert len(sub) == 2
        assert sub.names == ["a", "c"]


# ---------------------------------------------------------------------------
# _inc_path
# ---------------------------------------------------------------------------


class TestIncPath:
    def test_empty_path(self):
        assert _inc_path("", "child") == "child"

    def test_nonempty_path(self):
        result = _inc_path("parent", "child")
        assert "parent" in result
        assert "child" in result


# ---------------------------------------------------------------------------
# _list_grob
# ---------------------------------------------------------------------------


class TestListGrob:
    def test_none_returns_empty(self):
        fl = _list_grob(None, True, False, False, False)
        assert len(fl) == 0

    def test_non_grob_returns_empty(self):
        fl = _list_grob("not_a_grob", True, False, False, False)
        assert len(fl) == 0

    def test_plain_grob(self):
        g = Grob(name="rect1", _grid_class="rect")
        fl = _list_grob(g, grobs=True, viewports=False,
                        full_names=False, recursive=False)
        assert len(fl) == 1
        assert fl.names[0] == "rect1"
        assert fl.types[0] == "grobListing"

    def test_plain_grob_full_names(self):
        g = Grob(name="rect1", _grid_class="rect")
        fl = _list_grob(g, grobs=True, viewports=False,
                        full_names=True, recursive=False)
        assert "rect1" in fl.names[0]

    def test_plain_grob_no_grobs(self):
        g = Grob(name="rect1", _grid_class="rect")
        fl = _list_grob(g, grobs=False, viewports=False,
                        full_names=False, recursive=False)
        assert len(fl) == 0

    def test_gtree_with_children(self):
        child1 = Grob(name="c1", _grid_class="rect")
        child2 = Grob(name="c2", _grid_class="circle")
        tree = GTree(children=GList(child1, child2), name="tree1")
        fl = _list_grob(tree, grobs=True, viewports=False,
                        full_names=False, recursive=True)
        assert len(fl) == 3  # tree + 2 children
        assert fl.types[0] == "gTreeListing"

    def test_gtree_no_recursive(self):
        child1 = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child1), name="tree1")
        fl = _list_grob(tree, grobs=True, viewports=False,
                        full_names=False, recursive=False)
        assert len(fl) == 1  # just the tree
        assert fl.types[0] == "gTreeListing"

    def test_glist(self):
        g1 = Grob(name="g1", _grid_class="rect")
        g2 = Grob(name="g2", _grid_class="circle")
        gl = GList(g1, g2)
        fl = _list_grob(gl, grobs=True, viewports=False,
                        full_names=False, recursive=False)
        assert len(fl) == 2


# ---------------------------------------------------------------------------
# grid_ls (with grob argument -- no display list needed)
# ---------------------------------------------------------------------------


class TestGridLs:
    def test_ls_grob(self):
        g = Grob(name="mygrob", _grid_class="rect")
        fl = grid_ls(g, print_=False)
        assert len(fl) == 1
        assert fl.names[0] == "mygrob"

    def test_ls_gtree_recursive(self):
        child = Grob(name="child", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        fl = grid_ls(tree, recursive=True, print_=False)
        assert len(fl) == 2

    def test_ls_gtree_full_names(self):
        child = Grob(name="child", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        fl = grid_ls(tree, fullNames=True, print_=False)
        assert len(fl) == 2

    def test_ls_print_true(self, capsys):
        g = Grob(name="printed", _grid_class="rect")
        fl = grid_ls(g, print_=True)
        captured = capsys.readouterr()
        assert "printed" in captured.out

    def test_ls_print_callable(self):
        results = []
        g = Grob(name="callable_test", _grid_class="rect")
        fl = grid_ls(g, print_=lambda x: results.append(x))
        assert len(results) == 1

    def test_ls_print_invalid_raises(self):
        g = Grob(name="test", _grid_class="rect")
        with pytest.raises(TypeError, match="invalid 'print_'"):
            grid_ls(g, print_=42)


# ---------------------------------------------------------------------------
# grid_grep (with grob argument)
# ---------------------------------------------------------------------------


class TestGridGrep:
    def test_grep_simple_match(self):
        child = Grob(name="myRect", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = grid_grep("myRect", x=tree, grep=False, global_=True)
        assert len(result) >= 1

    def test_grep_regex_match(self):
        child = Grob(name="myRect", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = grid_grep("my.*", x=tree, grep=True, global_=True)
        assert len(result) >= 1

    def test_grep_no_match(self):
        child = Grob(name="myRect", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = grid_grep("nonexistent", x=tree, grep=False, global_=True)
        assert result == []

    def test_grep_first_only(self):
        child1 = Grob(name="match1", _grid_class="rect")
        child2 = Grob(name="match2", _grid_class="rect")
        tree = GTree(children=GList(child1, child2), name="tree")
        result = grid_grep("match", x=tree, grep=True, global_=False)
        assert isinstance(result, GPath)

    def test_grep_strict(self):
        child = Grob(name="strictmatch", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = grid_grep("tree", x=tree, grep=False,
                           global_=True, strict=True)
        # tree is listed at depth 1
        assert isinstance(result, list)

    def test_grep_invalid_path_type_raises(self):
        with pytest.raises(TypeError, match="expected str or GPath"):
            grid_grep(42)

    def test_grep_all_devices_raises(self):
        with pytest.raises(NotImplementedError):
            grid_grep("test", allDevices=True)

    def test_grep_gpath_argument(self):
        child = Grob(name="foo", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = grid_grep(GPath("foo"), x=tree, grep=False, global_=True)
        assert len(result) >= 1

    def test_grep_empty_listing(self):
        result = grid_grep("test", x=GList(), grep=False, global_=True)
        assert result == []


# ---------------------------------------------------------------------------
# Formatting functions
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_nested_listing(self, capsys):
        fl = FlatGridListing()
        fl.append_entry(GridListingEntry(name="root", g_depth=0, vp_depth=0))
        fl.append_entry(GridListingEntry(name="child", g_depth=1, vp_depth=0))
        nested_listing(fl)
        captured = capsys.readouterr()
        assert "root" in captured.out
        assert "child" in captured.out

    def test_nested_listing_custom_indent(self, capsys):
        fl = FlatGridListing()
        fl.append_entry(GridListingEntry(name="a", g_depth=0, vp_depth=1))
        nested_listing(fl, gindent="--", vpindent="..")
        captured = capsys.readouterr()
        assert ".." in captured.out

    def test_nested_listing_invalid_type_raises(self):
        with pytest.raises(TypeError, match="expected FlatGridListing"):
            nested_listing("not_a_listing")

    def test_path_listing(self, capsys):
        fl = FlatGridListing()
        fl.append_entry(GridListingEntry(
            name="g1", g_depth=0, vp_depth=0,
            g_path="", vp_path="", entry_type="grobListing"))
        path_listing(fl)
        captured = capsys.readouterr()
        assert "g1" in captured.out

    def test_path_listing_with_vp(self, capsys):
        fl = FlatGridListing()
        fl.append_entry(GridListingEntry(
            name="vp1", g_depth=0, vp_depth=0,
            g_path="", vp_path="", entry_type="vpListing"))
        fl.append_entry(GridListingEntry(
            name="g1", g_depth=0, vp_depth=1,
            g_path="", vp_path="vp1", entry_type="grobListing"))
        path_listing(fl)
        captured = capsys.readouterr()
        assert "vp1" in captured.out

    def test_path_listing_no_align(self, capsys):
        fl = FlatGridListing()
        fl.append_entry(GridListingEntry(
            name="g1", g_depth=0, vp_depth=0,
            g_path="", vp_path="", entry_type="grobListing"))
        path_listing(fl, gAlign=False)
        captured = capsys.readouterr()
        assert "g1" in captured.out

    def test_path_listing_empty(self, capsys):
        fl = FlatGridListing()
        path_listing(fl)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_path_listing_invalid_type_raises(self):
        with pytest.raises(TypeError, match="expected FlatGridListing"):
            path_listing("bad")

    def test_grob_path_listing(self, capsys):
        fl = FlatGridListing()
        fl.append_entry(GridListingEntry(
            name="g1", g_depth=0, vp_depth=0,
            g_path="", vp_path="", entry_type="grobListing"))
        fl.append_entry(GridListingEntry(
            name="vp1", g_depth=0, vp_depth=0,
            g_path="", vp_path="", entry_type="vpListing"))
        grob_path_listing(fl)
        captured = capsys.readouterr()
        assert "g1" in captured.out

    def test_grob_path_listing_no_grobs(self, capsys):
        fl = FlatGridListing()
        fl.append_entry(GridListingEntry(
            name="vp1", g_depth=0, vp_depth=0,
            g_path="", vp_path="", entry_type="vpListing"))
        grob_path_listing(fl)
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# show_grob
# ---------------------------------------------------------------------------


class TestShowGrob:
    def test_show_grob_with_x(self, capsys):
        g = Grob(name="show_test", _grid_class="rect")
        result = show_grob(x=g)
        assert result is g
        captured = capsys.readouterr()
        assert "show_test" in captured.out

    def test_show_grob_gtree_with_gpath(self, capsys):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = show_grob(x=tree, gPath="c1")
        assert result is not None
        assert result.name == "c1"

    def test_show_grob_gtree_gpath_not_found(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = show_grob(x=tree, gPath="nonexistent")
        assert result is None

    def test_show_grob_non_gtree_with_gpath(self):
        g = Grob(name="plain", _grid_class="rect")
        result = show_grob(x=g, gPath="anything")
        assert result is None

    def test_show_grob_gpath_as_string(self, capsys):
        child = Grob(name="s1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        result = show_grob(x=tree, gPath="s1")
        assert result.name == "s1"


# ---------------------------------------------------------------------------
# get_names / child_names (deprecated)
# ---------------------------------------------------------------------------


class TestDeprecatedHelpers:
    def test_get_names_grob(self):
        g = Grob(name="single", _grid_class="rect")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = get_names(g)
        assert names == ["single"]

    def test_get_names_gtree(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = get_names(tree)
        assert "c1" in names

    def test_get_names_non_grob(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = get_names("not_a_grob")
        assert names == []

    def test_get_names_warns(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            get_names(Grob(name="test"))

    def test_child_names_gtree(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = child_names(tree)
        assert "c1" in names

    def test_child_names_non_gtree(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = child_names(Grob(name="plain"))
        assert names == []

    def test_child_names_warns(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            child_names(Grob(name="test"))


# ---------------------------------------------------------------------------
# _list_display_list (via patching state.display_list)
# ---------------------------------------------------------------------------

from grid_py._ls import _list_display_list
from grid_py._display_list import (
    DisplayList, DLDrawGrob, DLPushViewport, DLPopViewport,
    DLUpViewport, DLDownViewport,
)
from grid_py._state import get_state
from grid_py._viewport import Viewport


def _setup_dl_state(items):
    """Patch state with a DisplayList containing given items."""
    state = get_state()
    state.reset()
    dl = DisplayList()
    for item in items:
        dl.record(item)
    state.display_list = dl
    return state


class TestListDisplayList:
    def test_empty_dl(self):
        _setup_dl_state([])
        fl = _list_display_list(grobs=True, viewports=True,
                                full_names=False, recursive=True)
        assert len(fl) == 0

    def test_grobs_only(self):
        g = Grob(name="g1", _grid_class="rect")
        _setup_dl_state([DLDrawGrob(grob=g)])
        fl = _list_display_list(grobs=True, viewports=False,
                                full_names=False, recursive=True)
        assert len(fl) == 1
        assert fl.names[0] == "g1"

    def test_viewports_push(self):
        vp = Viewport(name="vp1")
        _setup_dl_state([DLPushViewport(viewport=vp)])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=False, recursive=True)
        assert len(fl) == 1
        assert fl.names[0] == "vp1"
        assert fl.types[0] == "vpListing"

    def test_viewports_push_full_names(self):
        vp = Viewport(name="vp1")
        _setup_dl_state([DLPushViewport(viewport=vp)])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=True, recursive=True)
        assert "viewport[" in fl.names[0]

    def test_viewports_pop(self):
        vp = Viewport(name="vp1")
        _setup_dl_state([
            DLPushViewport(viewport=vp),
            DLPopViewport(n=1),
        ])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=False, recursive=True)
        assert len(fl) == 2
        assert fl.types[1] == "vpPopListing"

    def test_viewports_pop_full_names(self):
        vp = Viewport(name="vp1")
        _setup_dl_state([
            DLPushViewport(viewport=vp),
            DLPopViewport(n=1),
        ])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=True, recursive=True)
        assert "popViewport[" in fl.names[1]

    def test_viewports_up(self):
        vp = Viewport(name="vp1")
        _setup_dl_state([
            DLPushViewport(viewport=vp),
            DLUpViewport(n=1),
        ])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=False, recursive=True)
        assert len(fl) == 2
        assert fl.types[1] == "vpUpListing"

    def test_viewports_up_full_names(self):
        vp = Viewport(name="vp1")
        _setup_dl_state([
            DLPushViewport(viewport=vp),
            DLUpViewport(n=1),
        ])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=True, recursive=True)
        assert "upViewport[" in fl.names[1]

    def test_viewports_down(self):
        from grid_py._path import VpPath
        path = VpPath("target")
        _setup_dl_state([DLDownViewport(path=path)])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=False, recursive=True)
        assert len(fl) == 1
        assert fl.types[0] == "vpNameListing"

    def test_viewports_down_full_names(self):
        from grid_py._path import VpPath
        path = VpPath("target")
        _setup_dl_state([DLDownViewport(path=path)])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=True, recursive=True)
        assert "downViewport[" in fl.names[0]

    def test_viewports_down_no_path(self):
        _setup_dl_state([DLDownViewport(path=None)])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=False, recursive=True)
        assert len(fl) == 1
        assert fl.names[0] == "?"

    def test_grobs_full_names(self):
        g = Grob(name="g1", _grid_class="rect")
        _setup_dl_state([DLDrawGrob(grob=g)])
        fl = _list_display_list(grobs=True, viewports=False,
                                full_names=True, recursive=True)
        assert len(fl) == 1
        assert "rect[g1]" in fl.names[0]

    def test_grobs_with_viewport_context(self):
        vp = Viewport(name="vp1")
        g = Grob(name="g1", _grid_class="rect")
        _setup_dl_state([
            DLPushViewport(viewport=vp),
            DLDrawGrob(grob=g),
        ])
        fl = _list_display_list(grobs=True, viewports=True,
                                full_names=False, recursive=True)
        assert len(fl) == 2

    def test_none_grob_skipped(self):
        _setup_dl_state([DLDrawGrob(grob=None)])
        fl = _list_display_list(grobs=True, viewports=False,
                                full_names=False, recursive=True)
        assert len(fl) == 0

    def test_pop_adjusts_depth(self):
        vp1 = Viewport(name="a")
        vp2 = Viewport(name="b")
        _setup_dl_state([
            DLPushViewport(viewport=vp1),
            DLPushViewport(viewport=vp2),
            DLPopViewport(n=2),
        ])
        fl = _list_display_list(grobs=False, viewports=True,
                                full_names=False, recursive=True)
        assert len(fl) == 3


# Cleanup after test
@pytest.fixture(autouse=True)
def _cleanup_dl_state():
    yield
    state = get_state()
    if hasattr(state, "display_list"):
        try:
            delattr(state, "display_list")
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# grid_ls with display list (patched)
# ---------------------------------------------------------------------------


class TestGridLsDisplayList:
    def test_ls_display_list(self):
        g = Grob(name="dlgrob", _grid_class="rect")
        _setup_dl_state([DLDrawGrob(grob=g)])
        fl = grid_ls(x=None, print_=False)
        assert len(fl) == 1
        assert fl.names[0] == "dlgrob"


# ---------------------------------------------------------------------------
# grid_grep with display list (patched)
# ---------------------------------------------------------------------------


class TestGridGrepDisplayList:
    def test_grep_display_list(self):
        g = Grob(name="dlgrob", _grid_class="rect")
        _setup_dl_state([DLDrawGrob(grob=g)])
        result = grid_grep("dlgrob", x=None, grep=False, global_=True)
        assert len(result) >= 1

    def test_grep_with_viewports(self):
        vp = Viewport(name="myvp")
        _setup_dl_state([DLPushViewport(viewport=vp)])
        result = grid_grep("myvp", x=None, grep=False,
                           global_=True, viewports=True)
        assert len(result) >= 1

    def test_grep_strict_display_list(self):
        g = Grob(name="strictgrob", _grid_class="rect")
        _setup_dl_state([DLDrawGrob(grob=g)])
        result = grid_grep("strictgrob", x=None, grep=False,
                           global_=True, strict=True)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# show_grob with display list (patched)
# ---------------------------------------------------------------------------


class TestShowGrobDisplayList:
    def test_show_grob_none_none(self, capsys):
        g = Grob(name="showdl", _grid_class="rect")
        _setup_dl_state([DLDrawGrob(grob=g)])
        result = show_grob(x=None, gPath=None)
        captured = capsys.readouterr()
        assert "showdl" in captured.out

    def test_show_grob_none_with_gpath(self):
        g = Grob(name="findable", _grid_class="rect")
        _setup_dl_state([DLDrawGrob(grob=g)])
        result = show_grob(x=None, gPath="findable")
