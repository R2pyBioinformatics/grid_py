"""Tests for grid_py._group -- group, define, and use grob system."""

from __future__ import annotations

import numpy as np
import pytest

from grid_py._group import (
    COMPOSITING_OPERATORS,
    DefineGrob,
    GroupGrob,
    UseGrob,
    _validate_op,
    _validate_transform,
    define_grob,
    grid_define,
    grid_group,
    grid_use,
    group_grob,
    use_grob,
)
from grid_py._gpar import Gpar
from grid_py._grob import GList, GTree, Grob


# ---------------------------------------------------------------------------
# _validate_op
# ---------------------------------------------------------------------------

class TestValidateOp:

    def test_valid_ops(self):
        for op in COMPOSITING_OPERATORS:
            assert _validate_op(op) == op

    def test_case_insensitive(self):
        assert _validate_op("OVER") == "over"
        assert _validate_op("Source") == "source"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid compositing"):
            _validate_op("invalid_op")


# ---------------------------------------------------------------------------
# _validate_transform
# ---------------------------------------------------------------------------

class TestValidateTransform:

    def test_none_is_ok(self):
        _validate_transform(None)  # should not raise

    def test_valid_identity(self):
        _validate_transform(np.eye(3, dtype=np.float64))

    def test_not_ndarray_raises(self):
        with pytest.raises(TypeError, match="numpy ndarray"):
            _validate_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="3x3"):
            _validate_transform(np.eye(4, dtype=np.float64))

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="numeric"):
            _validate_transform(np.array([["a", "b", "c"]] * 3))

    def test_bad_last_column_raises(self):
        mat = np.eye(3, dtype=np.float64)
        mat[0, 2] = 1.0
        with pytest.raises(ValueError, match="Invalid transform"):
            _validate_transform(mat)

    def test_bad_bottom_right_raises(self):
        mat = np.eye(3, dtype=np.float64)
        mat[2, 2] = 2.0
        with pytest.raises(ValueError, match="Invalid transform"):
            _validate_transform(mat)


# ---------------------------------------------------------------------------
# GroupGrob
# ---------------------------------------------------------------------------

class TestGroupGrob:

    def test_basic_creation(self):
        src = Grob(name="src1")
        g = GroupGrob(src=src)
        assert g.src is src
        assert g.op == "over"
        assert g.dst is None
        assert g._grid_class == "GridGroup"

    def test_with_dst(self):
        src = Grob(name="src1")
        dst = Grob(name="dst1")
        g = GroupGrob(src=src, dst=dst)
        assert g.src is src
        assert g.dst is dst
        assert g.n_children() == 2

    def test_custom_op(self):
        g = GroupGrob(src=Grob(name="s"), op="xor")
        assert g.op == "xor"

    def test_invalid_op_raises(self):
        with pytest.raises(ValueError):
            GroupGrob(src=Grob(name="s"), op="bad")

    def test_no_src_no_dst(self):
        g = GroupGrob()
        assert g.src is None
        assert g.dst is None

    def test_repr(self):
        g = GroupGrob(src=Grob(name="s"), name="grp1")
        s = repr(g)
        assert "GroupGrob" in s
        assert "grp1" in s

    def test_valid_details(self):
        g = GroupGrob(src=Grob(name="s"), op="over")
        g.valid_details()  # should not raise

    def test_draw_details(self):
        g = GroupGrob(src=Grob(name="s"))
        g.draw_details()  # placeholder, should not raise


# ---------------------------------------------------------------------------
# DefineGrob
# ---------------------------------------------------------------------------

class TestDefineGrob:

    def test_basic_creation(self):
        src = Grob(name="src1")
        d = DefineGrob(src=src)
        assert d.src is src
        assert d.op == "over"
        assert d._grid_class == "GridDefine"

    def test_with_dst(self):
        src = Grob(name="src1")
        dst = Grob(name="dst1")
        d = DefineGrob(src=src, dst=dst)
        assert d.n_children() == 2

    def test_no_src(self):
        d = DefineGrob()
        assert d.src is None

    def test_repr(self):
        d = DefineGrob(src=Grob(name="s"), name="def1")
        s = repr(d)
        assert "DefineGrob" in s

    def test_valid_details(self):
        d = DefineGrob(src=Grob(name="s"))
        d.valid_details()

    def test_draw_details(self):
        d = DefineGrob(src=Grob(name="s"))
        d.draw_details()  # placeholder


# ---------------------------------------------------------------------------
# UseGrob
# ---------------------------------------------------------------------------

class TestUseGrob:

    def test_basic_creation(self):
        u = UseGrob(group="mygroup")
        assert u.group == "mygroup"
        assert u.transform is None
        assert u._grid_class == "GridUse"

    def test_with_transform(self):
        mat = np.eye(3, dtype=np.float64)
        u = UseGrob(group="g", transform=mat)
        np.testing.assert_array_equal(u.transform, mat)

    def test_invalid_transform_raises(self):
        bad = np.eye(3, dtype=np.float64)
        bad[0, 2] = 5.0
        with pytest.raises(ValueError):
            UseGrob(group="g", transform=bad)

    def test_repr(self):
        u = UseGrob(group="g", name="use1")
        s = repr(u)
        assert "UseGrob" in s
        assert "use1" in s

    def test_repr_with_transform(self):
        u = UseGrob(group="g", transform=np.eye(3, dtype=np.float64))
        s = repr(u)
        assert "set" in s

    def test_valid_details(self):
        u = UseGrob(group="g")
        u.valid_details()

    def test_draw_details(self):
        u = UseGrob(group="g")
        with pytest.warns(UserWarning, match="Unknown group"):
            u.draw_details()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactoryFunctions:

    def test_group_grob(self):
        src = Grob(name="s")
        g = group_grob(src=src, op="source")
        assert isinstance(g, GroupGrob)
        assert g.op == "source"

    def test_grid_group_draw_true(self):
        src = Grob(name="s")
        g = grid_group(src=src, draw=True)
        assert isinstance(g, GroupGrob)

    def test_grid_group_draw_false(self):
        src = Grob(name="s")
        g = grid_group(src=src, draw=False)
        assert isinstance(g, GroupGrob)

    def test_define_grob(self):
        src = Grob(name="s")
        d = define_grob(src=src)
        assert isinstance(d, DefineGrob)

    def test_grid_define_draw_true(self):
        src = Grob(name="s")
        d = grid_define(src=src, draw=True)
        assert isinstance(d, DefineGrob)

    def test_grid_define_draw_false(self):
        src = Grob(name="s")
        d = grid_define(src=src, draw=False)
        assert isinstance(d, DefineGrob)

    def test_use_grob(self):
        u = use_grob(group="g")
        assert isinstance(u, UseGrob)
        assert u.group == "g"

    def test_grid_use_draw_true(self):
        with pytest.warns(UserWarning, match="Unknown group"):
            u = grid_use(group="g", draw=True)
        assert isinstance(u, UseGrob)

    def test_grid_use_draw_false(self):
        u = grid_use(group="g", draw=False)
        assert isinstance(u, UseGrob)

    def test_group_grob_with_gp(self):
        gp = Gpar(col="blue")
        g = group_grob(src=Grob(name="s"), gp=gp)
        assert g.gp is gp

    def test_use_grob_with_transform(self):
        mat = np.eye(3, dtype=np.float64)
        mat[2, 0] = 10
        u = use_grob(group="g", transform=mat)
        assert u.transform[2, 0] == 10
