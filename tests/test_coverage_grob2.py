"""Coverage tests for grid_py._grob -- Grob methods, GTree manipulation,
GEdit/GEditList, free functions.
"""
from __future__ import annotations

import copy
import warnings
import pytest

import grid_py
from grid_py._grob import (
    Grob,
    GTree,
    GList,
    GEdit,
    GEditList,
    grob_tree,
    grob_name,
    is_grob,
    get_grob,
    set_grob,
    add_grob,
    remove_grob,
    edit_grob,
    force_grob,
    set_children,
    reorder_grob,
    apply_edit,
    apply_edits,
    _auto_name,
    _reset_auto_name,
    _edit_this_grob,
)
from grid_py._gpar import Gpar
from grid_py._path import GPath


# ---------------------------------------------------------------------------
# Auto-name
# ---------------------------------------------------------------------------


class TestAutoName:
    def test_auto_name(self):
        _reset_auto_name()
        n1 = _auto_name()
        n2 = _auto_name()
        assert n1 == "GRID.GROB.1"
        assert n2 == "GRID.GROB.2"

    def test_grob_name_none(self):
        _reset_auto_name()
        name = grob_name()
        assert name.startswith("GRID.GROB.")

    def test_grob_name_with_grob(self):
        _reset_auto_name()
        g = Grob(name="test", _grid_class="rect")
        name = grob_name(g)
        assert "rect" in name

    def test_grob_name_non_grob_raises(self):
        with pytest.raises(TypeError, match="invalid 'grob'"):
            grob_name("not_a_grob")


# ---------------------------------------------------------------------------
# Grob
# ---------------------------------------------------------------------------


class TestGrob:
    def test_basic_construction(self):
        g = Grob(name="test", _grid_class="rect")
        assert g.name == "test"
        assert g._grid_class == "rect"

    def test_auto_name_when_none(self):
        g = Grob()
        assert g.name.startswith("GRID.")

    def test_name_setter(self):
        g = Grob(name="old")
        g.name = "new"
        assert g.name == "new"

    def test_gp_property(self):
        gp = Gpar(col="red")
        g = Grob(gp=gp)
        assert g.gp is gp

    def test_gp_setter_invalid_raises(self):
        g = Grob()
        with pytest.raises(TypeError, match="expected Gpar"):
            g.gp = "not_gpar"

    def test_gp_setter_none(self):
        g = Grob(gp=Gpar(col="red"))
        g.gp = None
        assert g.gp is None

    def test_vp_property(self):
        g = Grob()
        assert g.vp is None

    def test_vp_setter_string(self):
        g = Grob()
        g.vp = "some_path"
        from grid_py._path import VpPath
        assert isinstance(g.vp, VpPath)

    def test_gp_constructor_invalid_raises(self):
        with pytest.raises(TypeError, match="expected Gpar"):
            Grob(gp="bad")

    def test_repr(self):
        g = Grob(name="r1", _grid_class="rect")
        assert "rect[r1]" in repr(g)

    def test_str(self):
        g = Grob(name="r1", _grid_class="rect")
        assert "rect[r1]" in str(g)

    def test_hook_methods_defaults(self):
        g = Grob()
        g.draw_details()
        g.pre_draw_details()
        g.post_draw_details()
        g.valid_details()
        assert g.make_content() is g
        assert g.make_context() is g
        assert g.edit_details() is g
        assert g.width_details() is None
        assert g.height_details() is None
        assert g.x_details() is None
        assert g.y_details() is None
        assert g.ascent_details() is None
        assert g.descent_details() is None
        assert g.grob_coords() is None
        assert g.grob_points() is None

    def test_kwargs_stored(self):
        g = Grob(name="test", custom_attr="hello")
        assert g.custom_attr == "hello"


# ---------------------------------------------------------------------------
# GList
# ---------------------------------------------------------------------------


class TestGList:
    def test_empty(self):
        gl = GList()
        assert len(gl) == 0

    def test_with_grobs(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        gl = GList(g1, g2)
        assert len(gl) == 2

    def test_none_skipped(self):
        g = Grob(name="a")
        gl = GList(g, None)
        assert len(gl) == 1

    def test_nested_glist_flattened(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        inner = GList(g1)
        gl = GList(inner, g2)
        assert len(gl) == 2

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="only Grob"):
            GList("not_a_grob")

    def test_getitem_int(self):
        g = Grob(name="a")
        gl = GList(g)
        assert gl[0].name == "a"

    def test_getitem_slice(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        gl = GList(g1, g2)
        sub = gl[0:1]
        assert isinstance(sub, GList)
        assert len(sub) == 1

    def test_setitem(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        gl = GList(g1)
        gl[0] = g2
        assert gl[0].name == "b"

    def test_setitem_invalid_raises(self):
        gl = GList(Grob(name="a"))
        with pytest.raises(TypeError, match="only Grob"):
            gl[0] = "bad"

    def test_append(self):
        gl = GList()
        gl.append(Grob(name="a"))
        assert len(gl) == 1

    def test_append_invalid_raises(self):
        gl = GList()
        with pytest.raises(TypeError, match="only Grob"):
            gl.append("bad")

    def test_repr(self):
        g = Grob(name="x", _grid_class="rect")
        gl = GList(g)
        assert "rect[x]" in repr(gl)

    def test_iter(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        gl = GList(g1, g2)
        names = [g.name for g in gl]
        assert names == ["a", "b"]


# ---------------------------------------------------------------------------
# GTree
# ---------------------------------------------------------------------------


class TestGTree:
    def test_basic(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        assert tree.n_children() == 1

    def test_get_children(self):
        child = Grob(name="c1")
        tree = GTree(children=GList(child), name="tree")
        gl = tree.get_children()
        assert len(gl) == 1

    def test_set_children(self):
        child = Grob(name="c1")
        tree = GTree(name="tree")
        tree.set_children(GList(child))
        assert tree.n_children() == 1

    def test_add_child(self):
        tree = GTree(name="tree")
        child = Grob(name="c1")
        tree.add_child(child)
        assert tree.n_children() == 1

    def test_add_child_replace(self):
        child1 = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child1), name="tree")
        child2 = Grob(name="c1", _grid_class="circle")
        tree.add_child(child2)
        assert tree.n_children() == 1
        assert tree.get_child("c1")._grid_class == "circle"

    def test_add_child_invalid_raises(self):
        tree = GTree(name="tree")
        with pytest.raises(TypeError, match="can only add"):
            tree.add_child("not_a_grob")

    def test_remove_child(self):
        child = Grob(name="c1")
        tree = GTree(children=GList(child), name="tree")
        tree.remove_child("c1")
        assert tree.n_children() == 0

    def test_remove_child_not_found_raises(self):
        tree = GTree(name="tree")
        with pytest.raises(KeyError, match="not found"):
            tree.remove_child("nonexistent")

    def test_get_child(self):
        child = Grob(name="c1")
        tree = GTree(children=GList(child), name="tree")
        assert tree.get_child("c1").name == "c1"

    def test_get_child_not_found_raises(self):
        tree = GTree(name="tree")
        with pytest.raises(KeyError, match="not found"):
            tree.get_child("nonexistent")

    def test_set_child(self):
        child1 = Grob(name="c1")
        tree = GTree(children=GList(child1), name="tree")
        child2 = Grob(name="c1", _grid_class="circle")
        tree.set_child("c1", child2)
        assert tree.get_child("c1")._grid_class == "circle"

    def test_set_child_not_found_raises(self):
        tree = GTree(name="tree")
        with pytest.raises(KeyError, match="not found"):
            tree.set_child("c1", Grob(name="c1"))

    def test_set_child_name_mismatch_raises(self):
        child = Grob(name="c1")
        tree = GTree(children=GList(child), name="tree")
        with pytest.raises(ValueError, match="does not match"):
            tree.set_child("c1", Grob(name="c2"))

    def test_children_order(self):
        c1 = Grob(name="a")
        c2 = Grob(name="b")
        tree = GTree(children=GList(c1, c2), name="tree")
        assert tree._children_order == ["a", "b"]

    def test_explicit_children_order(self):
        c1 = Grob(name="a")
        c2 = Grob(name="b")
        tree = GTree(children=GList(c1, c2), name="tree",
                     children_order=["b", "a"])
        assert tree._children_order == ["b", "a"]

    def test_edit_details_forbidden(self):
        tree = GTree(name="tree")
        with pytest.raises(ValueError, match="invalid to directly edit"):
            tree.edit_details(children=GList())

    def test_repr(self):
        c1 = Grob(name="a")
        tree = GTree(children=GList(c1), name="tree")
        r = repr(tree)
        assert "tree" in r
        assert "a" in r

    def test_set_children_internal_invalid_raises(self):
        tree = GTree(name="tree")
        with pytest.raises(TypeError, match="GList or None"):
            tree._set_children_internal("bad")


# ---------------------------------------------------------------------------
# GEdit / GEditList
# ---------------------------------------------------------------------------


class TestGEdit:
    def test_basic(self):
        e = GEdit(col="red", lwd=2)
        assert e.specs == {"col": "red", "lwd": 2}

    def test_repr(self):
        e = GEdit(col="blue")
        assert "GEdit" in repr(e)
        assert "col" in repr(e)


class TestGEditList:
    def test_basic(self):
        e1 = GEdit(col="red")
        e2 = GEdit(lwd=3)
        el = GEditList(e1, e2)
        assert len(el) == 2

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="can only contain"):
            GEditList("not_an_edit")

    def test_iter(self):
        e1 = GEdit(col="red")
        el = GEditList(e1)
        items = list(el)
        assert len(items) == 1

    def test_repr(self):
        e1 = GEdit(col="red")
        el = GEditList(e1)
        assert "GEditList" in repr(el)


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


class TestFreeFunctions:
    def test_is_grob(self):
        assert is_grob(Grob()) is True
        assert is_grob("not_a_grob") is False

    def test_grob_tree(self):
        c1 = Grob(name="a")
        c2 = Grob(name="b")
        tree = grob_tree(c1, c2, name="mytree")
        assert isinstance(tree, GTree)
        assert tree.name == "mytree"
        assert tree.n_children() == 2

    def test_get_grob_single(self):
        child = Grob(name="c1")
        tree = GTree(children=GList(child), name="tree")
        result = get_grob(tree, "c1")
        assert result.name == "c1"

    def test_get_grob_multi_depth(self):
        inner_child = Grob(name="inner")
        inner_tree = GTree(children=GList(inner_child), name="mid")
        outer = GTree(children=GList(inner_tree), name="outer")
        result = get_grob(outer, GPath("mid", "inner"))
        assert result.name == "inner"

    def test_get_grob_non_gtree_raises(self):
        with pytest.raises(TypeError, match="can only get"):
            get_grob("not_gtree", "path")

    def test_get_grob_multi_depth_non_gtree_raises(self):
        child = Grob(name="c1")
        tree = GTree(children=GList(child), name="tree")
        with pytest.raises(KeyError, match="non-GTree"):
            get_grob(tree, GPath("c1", "subchild"))

    def test_set_grob_single(self):
        child = Grob(name="c1", _grid_class="rect")
        tree = GTree(children=GList(child), name="tree")
        new_child = Grob(name="c1", _grid_class="circle")
        result = set_grob(tree, "c1", new_child)
        assert result._children["c1"]._grid_class == "circle"

    def test_set_grob_multi_depth(self):
        inner_child = Grob(name="inner", _grid_class="rect")
        inner_tree = GTree(children=GList(inner_child), name="mid")
        outer = GTree(children=GList(inner_tree), name="outer")
        new_inner = Grob(name="inner", _grid_class="circle")
        result = set_grob(outer, GPath("mid", "inner"), new_inner)
        # Verify the update propagated
        mid = result._children["mid"]
        assert isinstance(mid, GTree)
        assert mid._children["inner"]._grid_class == "circle"

    def test_set_grob_non_gtree_raises(self):
        with pytest.raises(TypeError, match="can only set"):
            set_grob("not_gtree", "path", Grob())

    def test_set_grob_not_found_raises(self):
        tree = GTree(name="tree")
        with pytest.raises(KeyError, match="not found"):
            set_grob(tree, "nonexistent", Grob(name="nonexistent"))

    def test_set_grob_name_mismatch_raises(self):
        child = Grob(name="c1")
        tree = GTree(children=GList(child), name="tree")
        with pytest.raises(ValueError, match="does not match"):
            set_grob(tree, "c1", Grob(name="c2"))

    def test_set_grob_non_grob_raises(self):
        child = Grob(name="c1")
        tree = GTree(children=GList(child), name="tree")
        with pytest.raises(TypeError, match="replacement must be"):
            set_grob(tree, "c1", "not_a_grob")

    def test_add_grob(self):
        tree = GTree(name="tree")
        child = Grob(name="c1")
        result = add_grob(tree, child)
        assert result.n_children() == 1

    def test_add_grob_with_name(self):
        tree = GTree(name="tree")
        child = Grob(name="old_name")
        result = add_grob(tree, child, name="new_name")
        assert "new_name" in result._children_order

    def test_add_grob_non_gtree_raises(self):
        with pytest.raises(TypeError, match="can only add"):
            add_grob("not_gtree", Grob())

    def test_add_grob_non_grob_raises(self):
        tree = GTree(name="tree")
        with pytest.raises(TypeError, match="child must be"):
            add_grob(tree, "not_a_grob")

    def test_remove_grob(self):
        child = Grob(name="c1")
        tree = GTree(children=GList(child), name="tree")
        result = remove_grob(tree, "c1")
        assert result.n_children() == 0

    def test_remove_grob_non_gtree_raises(self):
        with pytest.raises(TypeError, match="can only remove"):
            remove_grob("not_gtree", "name")

    def test_edit_grob(self):
        g = Grob(name="orig", _grid_class="rect", custom=42)
        edited = edit_grob(g, custom=99)
        assert edited.custom == 99
        assert g.custom == 42  # original unchanged

    def test_edit_grob_gp(self):
        g = Grob(name="test", gp=Gpar(col="red"))
        edited = edit_grob(g, gp=Gpar(col="blue"))
        # gp should be merged
        assert edited.gp is not None

    def test_edit_grob_gp_none(self):
        g = Grob(name="test", gp=Gpar(col="red"))
        edited = edit_grob(g, gp=None)
        assert edited.gp is None

    def test_edit_grob_name(self):
        g = Grob(name="old")
        edited = edit_grob(g, name="new")
        assert edited.name == "new"

    def test_edit_grob_vp(self):
        g = Grob(name="test")
        edited = edit_grob(g, vp="some_path")
        from grid_py._path import VpPath
        assert isinstance(edited.vp, VpPath)

    def test_edit_grob_unknown_slot_warns(self):
        g = Grob(name="test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            edited = _edit_this_grob(copy.deepcopy(g), {"nonexistent_slot": 42})
            assert len(w) >= 1
            assert "not found" in str(w[0].message)

    def test_set_children(self):
        tree = GTree(name="tree")
        c = Grob(name="c1")
        result = set_children(tree, GList(c))
        assert result.n_children() == 1

    def test_set_children_non_gtree_raises(self):
        with pytest.raises(TypeError, match="can only set children"):
            set_children("not_gtree", GList())

    def test_apply_edit(self):
        g = Grob(name="test", custom=1)
        e = GEdit(custom=2)
        result = apply_edit(g, e)
        assert result.custom == 2

    def test_apply_edit_none(self):
        g = Grob(name="test")
        result = apply_edit(g, None)
        assert result is g

    def test_apply_edit_invalid_raises(self):
        g = Grob(name="test")
        with pytest.raises(TypeError, match="expected GEdit"):
            apply_edit(g, "not_an_edit")

    def test_apply_edits_single(self):
        g = Grob(name="test", custom=1)
        e = GEdit(custom=2)
        result = apply_edits(g, e)
        assert result.custom == 2

    def test_apply_edits_list(self):
        g = Grob(name="test", custom=1)
        e1 = GEdit(custom=2)
        e2 = GEdit(custom=3)
        el = GEditList(e1, e2)
        result = apply_edits(g, el)
        assert result.custom == 3

    def test_apply_edits_none(self):
        g = Grob(name="test")
        result = apply_edits(g, None)
        assert result is g

    def test_apply_edits_invalid_raises(self):
        g = Grob(name="test")
        with pytest.raises(TypeError, match="expected GEdit"):
            apply_edits(g, 42)

    def test_set_grob_multi_depth_non_gtree_child_raises(self):
        child = Grob(name="leaf")
        tree = GTree(children=GList(child), name="tree")
        with pytest.raises(TypeError, match="not a GTree"):
            set_grob(tree, GPath("leaf", "sub"), Grob(name="sub"))

    def test_set_grob_multi_depth_child_not_found_raises(self):
        tree = GTree(name="tree")
        with pytest.raises(KeyError, match="not found"):
            set_grob(tree, GPath("missing", "sub"), Grob(name="sub"))
