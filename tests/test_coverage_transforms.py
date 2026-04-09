"""Tests for grid_py._transforms -- affine transformation matrices."""

from __future__ import annotations

import math

import numpy as np
import pytest

from grid_py._transforms import (
    defn_rotate,
    defn_scale,
    defn_translate,
    group_flip,
    group_rotate,
    group_scale,
    group_shear,
    group_translate,
    use_rotate,
    use_scale,
    use_translate,
    viewport_rotate,
    viewport_scale,
    viewport_transform,
    viewport_translate,
)


# ---------------------------------------------------------------------------
# group_translate
# ---------------------------------------------------------------------------

class TestGroupTranslate:

    def test_identity(self):
        mat = group_translate(0, 0)
        np.testing.assert_array_equal(mat, np.eye(3))

    def test_translate(self):
        mat = group_translate(10, 20)
        assert mat[2, 0] == 10
        assert mat[2, 1] == 20
        assert mat[2, 2] == 1

    def test_point_transform(self):
        mat = group_translate(5, 3)
        pt = np.array([1, 2, 1]) @ mat
        np.testing.assert_array_almost_equal(pt, [6, 5, 1])


# ---------------------------------------------------------------------------
# group_rotate
# ---------------------------------------------------------------------------

class TestGroupRotate:

    def test_zero_rotation(self):
        mat = group_rotate(0)
        np.testing.assert_array_almost_equal(mat, np.eye(3))

    def test_90_degrees(self):
        mat = group_rotate(90)
        pt = np.array([1, 0, 1]) @ mat
        np.testing.assert_array_almost_equal(pt, [0, 1, 1], decimal=10)

    def test_180_degrees(self):
        mat = group_rotate(180)
        pt = np.array([1, 0, 1]) @ mat
        np.testing.assert_array_almost_equal(pt, [-1, 0, 1], decimal=10)

    def test_360_identity(self):
        mat = group_rotate(360)
        np.testing.assert_array_almost_equal(mat, np.eye(3), decimal=10)


# ---------------------------------------------------------------------------
# group_scale
# ---------------------------------------------------------------------------

class TestGroupScale:

    def test_identity(self):
        mat = group_scale(1, 1)
        np.testing.assert_array_equal(mat, np.eye(3))

    def test_scale(self):
        mat = group_scale(2, 3)
        assert mat[0, 0] == 2
        assert mat[1, 1] == 3

    def test_point_transform(self):
        mat = group_scale(2, 0.5)
        pt = np.array([4, 6, 1]) @ mat
        np.testing.assert_array_almost_equal(pt, [8, 3, 1])


# ---------------------------------------------------------------------------
# group_shear
# ---------------------------------------------------------------------------

class TestGroupShear:

    def test_identity(self):
        mat = group_shear(0, 0)
        np.testing.assert_array_equal(mat, np.eye(3))

    def test_shear_x(self):
        mat = group_shear(0.5, 0)
        assert mat[1, 0] == 0.5
        assert mat[0, 1] == 0

    def test_shear_y(self):
        mat = group_shear(0, 0.5)
        assert mat[0, 1] == 0.5
        assert mat[1, 0] == 0


# ---------------------------------------------------------------------------
# group_flip
# ---------------------------------------------------------------------------

class TestGroupFlip:

    def test_no_flip(self):
        mat = group_flip(False, False)
        np.testing.assert_array_equal(mat, np.eye(3))

    def test_flip_x(self):
        mat = group_flip(True, False)
        assert mat[0, 0] == -1
        assert mat[1, 1] == 1

    def test_flip_y(self):
        mat = group_flip(False, True)
        assert mat[0, 0] == 1
        assert mat[1, 1] == -1

    def test_flip_both(self):
        mat = group_flip(True, True)
        assert mat[0, 0] == -1
        assert mat[1, 1] == -1


# ---------------------------------------------------------------------------
# Definition transforms
# ---------------------------------------------------------------------------

class TestDefnTransforms:

    def test_defn_translate(self):
        mat = defn_translate(5, 10)
        expected = group_translate(5, 10)
        np.testing.assert_array_equal(mat, expected)

    def test_defn_rotate(self):
        mat = defn_rotate(45)
        expected = group_rotate(45, device=True)
        np.testing.assert_array_equal(mat, expected)

    def test_defn_scale(self):
        mat = defn_scale(2, 3)
        expected = group_scale(2, 3)
        np.testing.assert_array_equal(mat, expected)


# ---------------------------------------------------------------------------
# Use transforms
# ---------------------------------------------------------------------------

class TestUseTransforms:

    def test_use_translate(self):
        mat = use_translate(3, 7)
        expected = group_translate(3, 7)
        np.testing.assert_array_equal(mat, expected)

    def test_use_rotate(self):
        mat = use_rotate(90)
        expected = group_rotate(90, device=True)
        np.testing.assert_array_equal(mat, expected)

    def test_use_scale(self):
        mat = use_scale(0.5, 0.5)
        expected = group_scale(0.5, 0.5)
        np.testing.assert_array_equal(mat, expected)


# ---------------------------------------------------------------------------
# Viewport transforms
# ---------------------------------------------------------------------------

class TestViewportTransforms:

    def test_viewport_translate(self):
        mat = viewport_translate(1, 2)
        expected = group_translate(1, 2)
        np.testing.assert_array_equal(mat, expected)

    def test_viewport_rotate(self):
        mat = viewport_rotate(30)
        expected = group_rotate(30, device=True)
        np.testing.assert_array_equal(mat, expected)

    def test_viewport_scale(self):
        mat = viewport_scale(4, 5)
        expected = group_scale(4, 5)
        np.testing.assert_array_equal(mat, expected)


# ---------------------------------------------------------------------------
# viewport_transform (combined)
# ---------------------------------------------------------------------------

class TestViewportTransform:

    def test_identity(self):
        mat = viewport_transform()
        np.testing.assert_array_almost_equal(mat, np.eye(3))

    def test_translate_only(self):
        mat = viewport_transform(dx=10, dy=20)
        expected = group_translate(10, 20)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_scale_only(self):
        mat = viewport_transform(sx=2, sy=3)
        expected = group_scale(2, 3)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_combined(self):
        mat = viewport_transform(dx=10, dy=20, rotation=0, sx=2, sy=3)
        expected = group_translate(10, 20) @ group_scale(2, 3)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_with_rotation(self):
        mat = viewport_transform(dx=0, dy=0, rotation=90, sx=1, sy=1)
        expected = group_rotate(90, device=True)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_full_composition(self):
        mat = viewport_transform(dx=5, dy=10, rotation=45, sx=2, sy=3)
        T = group_translate(5, 10)
        R = group_rotate(45, device=True)
        S = group_scale(2, 3)
        expected = T @ R @ S
        np.testing.assert_array_almost_equal(mat, expected)
