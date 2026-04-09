"""Tests to improve coverage of grid_py._display_list."""

import pytest

from grid_py._display_list import (
    DisplayList,
    DLOperation,
    DLDrawGrob,
    DLPushViewport,
    DLPopViewport,
    DLUpViewport,
    DLDownViewport,
    DLEditGrob,
    DLSetGpar,
)


class TestDisplayList:
    """Tests for DisplayList."""

    def test_init_empty(self):
        dl = DisplayList()
        assert len(dl) == 0
        assert dl.is_enabled() is True

    def test_record_adds_item(self):
        dl = DisplayList()
        op = DLOperation("test")
        dl.record(op)
        assert len(dl) == 1

    def test_record_disabled(self):
        dl = DisplayList()
        dl.set_enabled(False)
        dl.record(DLOperation("test"))
        assert len(dl) == 0
        assert dl.is_enabled() is False

    def test_get_items_returns_copy(self):
        dl = DisplayList()
        dl.record(DLOperation("a"))
        items = dl.get_items()
        assert len(items) == 1
        items.append(DLOperation("b"))
        assert len(dl) == 1  # original unchanged

    def test_clear(self):
        dl = DisplayList()
        dl.record(DLOperation("a"))
        dl.record(DLOperation("b"))
        dl.clear()
        assert len(dl) == 0

    def test_iter(self):
        dl = DisplayList()
        dl.record(DLOperation("a"))
        dl.record(DLOperation("b"))
        items = list(dl)
        assert len(items) == 2

    def test_getitem(self):
        dl = DisplayList()
        op = DLOperation("first")
        dl.record(op)
        assert dl[0] is op

    def test_apply(self):
        dl = DisplayList()
        dl.record(DLOperation("a"))
        dl.record(DLOperation("b"))
        results = dl.apply(lambda item: item.op_type)
        assert results == ["a", "b"]

    def test_replay_calls_replay(self):
        called = []

        class MockOp(DLOperation):
            def replay(self, state):
                called.append(state)

        dl = DisplayList()
        dl.record(MockOp("test"))
        dl.replay("mock_state")
        assert called == ["mock_state"]

    def test_replay_skips_no_replay(self):
        """Items without replay attr are skipped (base DLOperation has noop)."""
        dl = DisplayList()
        dl.record(DLOperation("noop"))
        # Should not raise
        dl.replay("state")


class TestDLOperation:
    """Tests for DLOperation base class."""

    def test_init(self):
        op = DLOperation("test_op", x=1, y=2)
        assert op.op_type == "test_op"
        assert op.params == {"x": 1, "y": 2}

    def test_replay_noop(self):
        op = DLOperation("noop")
        op.replay("state")  # should not raise


class TestDLDrawGrob:
    """Tests for DLDrawGrob."""

    def test_init(self):
        op = DLDrawGrob(grob="my_grob")
        assert op.op_type == "draw_grob"
        assert op.grob == "my_grob"

    def test_replay_calls_draw(self):
        drawn = []

        class FakeGrob:
            def draw(self, state):
                drawn.append(state)

        op = DLDrawGrob(grob=FakeGrob())
        op.replay("st")
        assert drawn == ["st"]

    def test_replay_none_grob(self):
        op = DLDrawGrob(grob=None)
        op.replay("state")  # should not raise

    def test_replay_grob_without_draw(self):
        op = DLDrawGrob(grob="no_draw_method")
        op.replay("state")  # should not raise


class TestDLPushViewport:
    """Tests for DLPushViewport."""

    def test_init(self):
        op = DLPushViewport(viewport="vp")
        assert op.op_type == "push_vp"
        assert op.viewport == "vp"

    def test_replay(self):
        pushed = []

        class MockState:
            def push_viewport(self, vp):
                pushed.append(vp)

        op = DLPushViewport(viewport="vp1")
        op.replay(MockState())
        assert pushed == ["vp1"]

    def test_replay_none_state(self):
        op = DLPushViewport(viewport="vp")
        op.replay(None)  # should not raise


class TestDLPopViewport:
    """Tests for DLPopViewport."""

    def test_init(self):
        op = DLPopViewport(n=3)
        assert op.n == 3
        assert op.op_type == "pop_vp"

    def test_replay(self):
        popped = []

        class MockState:
            def pop_viewport(self, n):
                popped.append(n)

        op = DLPopViewport(n=2)
        op.replay(MockState())
        assert popped == [2]

    def test_replay_none_state(self):
        op = DLPopViewport()
        op.replay(None)


class TestDLUpViewport:
    """Tests for DLUpViewport."""

    def test_init(self):
        op = DLUpViewport(n=2)
        assert op.n == 2
        assert op.op_type == "up_vp"

    def test_replay(self):
        navigated = []

        class MockState:
            def up_viewport(self, n):
                navigated.append(n)

        op = DLUpViewport(n=1)
        op.replay(MockState())
        assert navigated == [1]

    def test_replay_none_state(self):
        op = DLUpViewport()
        op.replay(None)


class TestDLDownViewport:
    """Tests for DLDownViewport."""

    def test_init(self):
        op = DLDownViewport(path="panel")
        assert op.path == "panel"
        assert op.op_type == "down_vp"

    def test_replay(self):
        navigated = []

        class MockState:
            def down_viewport(self, path):
                navigated.append(path)

        op = DLDownViewport(path="p")
        op.replay(MockState())
        assert navigated == ["p"]

    def test_replay_none_state(self):
        op = DLDownViewport(path="p")
        op.replay(None)


class TestDLEditGrob:
    """Tests for DLEditGrob."""

    def test_init(self):
        op = DLEditGrob(grob_name="g", specs={"col": "red"})
        assert op.grob_name == "g"
        assert op.specs == {"col": "red"}
        assert op.op_type == "edit_grob"

    def test_replay(self):
        edits = []

        class MockState:
            def edit_grob(self, name, specs):
                edits.append((name, specs))

        op = DLEditGrob(grob_name="g", specs={"x": 1})
        op.replay(MockState())
        assert edits == [("g", {"x": 1})]

    def test_replay_none_state(self):
        op = DLEditGrob()
        op.replay(None)


class TestDLSetGpar:
    """Tests for DLSetGpar."""

    def test_init(self):
        op = DLSetGpar(gpar="gp")
        assert op.gpar == "gp"
        assert op.op_type == "set_gpar"

    def test_replay(self):
        gpars = []

        class MockState:
            def set_gpar(self, gp):
                gpars.append(gp)

        op = DLSetGpar(gpar="my_gpar")
        op.replay(MockState())
        assert gpars == ["my_gpar"]

    def test_replay_none_state(self):
        op = DLSetGpar(gpar="gp")
        op.replay(None)
