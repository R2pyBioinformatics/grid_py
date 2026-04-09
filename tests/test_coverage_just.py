"""Tests to improve coverage of grid_py._just."""

import numpy as np
import pytest

from grid_py._just import (
    valid_just,
    resolve_hjust,
    resolve_vjust,
    resolve_raster_size,
    _valid_charjust,
    _valid_numjust,
    _match_just_string,
)


class TestMatchJustString:
    """Tests for _match_just_string."""

    def test_valid_strings(self):
        assert _match_just_string("left") == 0
        assert _match_just_string("right") == 1
        assert _match_just_string("bottom") == 2
        assert _match_just_string("top") == 3
        assert _match_just_string("centre") == 4
        assert _match_just_string("center") == 5

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="invalid justification string"):
            _match_just_string("invalid")


class TestValidCharjust:
    """Tests for _valid_charjust."""

    def test_empty(self):
        assert _valid_charjust([]) == (0.5, 0.5)

    def test_single_left(self):
        assert _valid_charjust(["left"]) == (0.0, 0.5)

    def test_single_right(self):
        assert _valid_charjust(["right"]) == (1.0, 0.5)

    def test_single_bottom(self):
        assert _valid_charjust(["bottom"]) == (0.5, 0.0)

    def test_single_top(self):
        assert _valid_charjust(["top"]) == (0.5, 1.0)

    def test_single_centre(self):
        assert _valid_charjust(["centre"]) == (0.5, 0.5)

    def test_single_center(self):
        assert _valid_charjust(["center"]) == (0.5, 0.5)

    def test_two_strings(self):
        assert _valid_charjust(["left", "top"]) == (0.0, 1.0)
        assert _valid_charjust(["right", "bottom"]) == (1.0, 0.0)
        assert _valid_charjust(["centre", "centre"]) == (0.5, 0.5)

    def test_invalid_hjust_string(self):
        with pytest.raises(ValueError, match="invalid horizontal"):
            _valid_charjust(["bottom", "top"])

    def test_invalid_vjust_string(self):
        with pytest.raises(ValueError, match="invalid vertical"):
            _valid_charjust(["left", "right"])


class TestValidNumjust:
    """Tests for _valid_numjust."""

    def test_empty(self):
        assert _valid_numjust([]) == (0.5, 0.5)

    def test_single(self):
        assert _valid_numjust([0.3]) == (0.3, 0.5)

    def test_two(self):
        assert _valid_numjust([0.1, 0.9]) == (0.1, 0.9)


class TestValidJust:
    """Tests for valid_just."""

    def test_string_centre(self):
        assert valid_just("centre") == (0.5, 0.5)

    def test_string_left(self):
        assert valid_just("left") == (0.0, 0.5)

    def test_numeric_scalar(self):
        assert valid_just(0.25) == (0.25, 0.5)

    def test_numpy_integer(self):
        assert valid_just(np.int32(1)) == (1.0, 0.5)

    def test_numpy_floating(self):
        assert valid_just(np.float64(0.7)) == (0.7, 0.5)

    def test_list_strings(self):
        assert valid_just(["right", "top"]) == (1.0, 1.0)

    def test_list_numbers(self):
        assert valid_just([0.1, 0.9]) == (0.1, 0.9)

    def test_empty_sequence(self):
        assert valid_just([]) == (0.5, 0.5)

    def test_all_numeric_np(self):
        assert valid_just([np.float64(0.2), np.float64(0.8)]) == (0.2, 0.8)

    def test_mixed_raises(self):
        # Mixed types that can't be converted to float
        with pytest.raises(ValueError, match="invalid justification"):
            valid_just(["left", 0.5])  # won't convert "left" to float


class TestResolveHjust:
    """Tests for resolve_hjust."""

    def test_explicit_override(self):
        assert resolve_hjust("centre", hjust=0.3) == 0.3

    def test_from_just(self):
        assert resolve_hjust("left") == 0.0

    def test_from_just_right(self):
        assert resolve_hjust("right") == 1.0


class TestResolveVjust:
    """Tests for resolve_vjust."""

    def test_explicit_override(self):
        assert resolve_vjust("centre", vjust=0.8) == 0.8

    def test_from_just_top(self):
        assert resolve_vjust("top") == 1.0

    def test_from_just_bottom(self):
        assert resolve_vjust("bottom") == 0.0


class TestResolveRasterSize:
    """Tests for resolve_raster_size."""

    def test_both_none_wider_raster(self):
        img = np.zeros((200, 400, 3))
        w, h = resolve_raster_size(img, (6.0, 4.0), (None, None))
        assert w == pytest.approx(6.0)
        assert h == pytest.approx(3.0)

    def test_both_none_taller_raster(self):
        img = np.zeros((400, 200, 3))
        w, h = resolve_raster_size(img, (6.0, 4.0), (None, None))
        assert h == pytest.approx(4.0)
        assert w == pytest.approx(2.0)

    def test_width_none(self):
        img = np.zeros((200, 400, 3))
        w, h = resolve_raster_size(img, (6.0, 4.0), (None, 3.0))
        assert h == pytest.approx(3.0)
        assert w == pytest.approx(6.0)

    def test_height_none(self):
        img = np.zeros((200, 400, 3))
        w, h = resolve_raster_size(img, (6.0, 4.0), (5.0, None))
        assert w == pytest.approx(5.0)
        assert h == pytest.approx(2.5)

    def test_both_specified(self):
        img = np.zeros((200, 400, 3))
        w, h = resolve_raster_size(img, (6.0, 4.0), (3.0, 2.0))
        assert w == pytest.approx(3.0)
        assert h == pytest.approx(2.0)
