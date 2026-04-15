"""Tests for WebRenderer, scene graph, and font metrics."""

import json
import numpy as np
import pytest

from grid_py._font_metrics import (
    CairoFontMetrics, HeuristicMetrics, get_font_backend,
)
from grid_py._gpar import Gpar
from grid_py._renderer_base import GridRenderer
from grid_py._scene_graph import (
    DefsCollection, GrobNode, SceneGraph, SceneNode, ViewportNode, _IdGenerator,
)
from grid_py.renderer_web import WebRenderer


# ===================================================================== #
# _IdGenerator                                                          #
# ===================================================================== #

class TestIdGenerator:
    def test_sequential(self):
        gen = _IdGenerator()
        assert gen.next("n") == "n-0"
        assert gen.next("n") == "n-1"
        assert gen.next("x") == "x-2"

    def test_reset(self):
        gen = _IdGenerator()
        gen.next("a")
        gen.next("a")
        gen.reset()
        assert gen.next("a") == "a-0"


# ===================================================================== #
# Scene graph data model                                                #
# ===================================================================== #

class TestSceneNode:
    def test_to_dict_basic(self):
        node = SceneNode(node_id="n1", node_type="test")
        d = node.to_dict()
        assert d["id"] == "n1"
        assert d["type"] == "test"
        assert d["children"] == []
        assert "metadata" not in d  # empty metadata omitted

    def test_to_dict_with_children(self):
        child = SceneNode(node_id="c1", node_type="child")
        parent = SceneNode(node_id="p1", node_type="parent", children=[child])
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["id"] == "c1"

    def test_to_dict_with_metadata(self):
        node = SceneNode(node_id="n1", node_type="test", metadata={"key": "val"})
        d = node.to_dict()
        assert d["metadata"] == {"key": "val"}


class TestViewportNode:
    def test_to_dict(self):
        vp = ViewportNode(
            node_id="vp1", node_type="viewport", name="panel",
            transform={"x0": 10, "y0": 20, "w": 100, "h": 80},
            clip=True, clip_id="clip-0",
        )
        d = vp.to_dict()
        assert d["type"] == "viewport"
        assert d["name"] == "panel"
        assert d["clip"] is True
        assert d["clip_id"] == "clip-0"
        assert "mask_id" not in d  # None omitted

    def test_mask_fields(self):
        vp = ViewportNode(
            node_id="vp1", node_type="viewport",
            mask_id="mask-0", mask_type="alpha",
        )
        d = vp.to_dict()
        assert d["mask_id"] == "mask-0"
        assert d["mask_type"] == "alpha"


class TestGrobNode:
    def test_to_dict(self):
        grob = GrobNode(
            node_id="g1", node_type="rect",
            props={"x": 10, "y": 20, "w": 30, "h": 40},
            gpar={"fill": "red", "col": "black"},
            render_hint="svg",
        )
        d = grob.to_dict()
        assert d["type"] == "rect"
        assert d["props"]["x"] == 10
        assert d["gpar"]["fill"] == "red"
        assert d["render_hint"] == "svg"


class TestSceneGraph:
    def test_to_json(self):
        root = ViewportNode(
            node_id="vp-0", node_type="viewport", name="ROOT",
            transform={"x0": 0, "y0": 0, "w": 700, "h": 500},
        )
        sg = SceneGraph(width=700, height=500, dpi=150, root=root)
        j = sg.to_json()
        parsed = json.loads(j)
        assert parsed["version"] == 1
        assert parsed["width"] == 700
        assert parsed["root"]["type"] == "viewport"


class TestDefsCollection:
    def test_empty(self):
        defs = DefsCollection()
        d = defs.to_dict()
        assert d == {"gradients": [], "patterns": [], "clip_paths": [], "masks": []}

    def test_with_entries(self):
        defs = DefsCollection(
            gradients=[{"id": "g1", "type": "linear"}],
            clip_paths=[{"id": "c1", "x": 0, "y": 0, "w": 10, "h": 10}],
        )
        d = defs.to_dict()
        assert len(d["gradients"]) == 1
        assert len(d["clip_paths"]) == 1


# ===================================================================== #
# Font metrics                                                          #
# ===================================================================== #

class TestFontMetrics:
    def test_heuristic_basic(self):
        m = HeuristicMetrics()
        result = m.measure("Hello")
        assert "ascent" in result
        assert "descent" in result
        assert "width" in result
        assert result["width"] > 0
        assert result["ascent"] > 0

    def test_heuristic_empty(self):
        m = HeuristicMetrics()
        result = m.measure("")
        assert result["width"] == 0.0

    def test_heuristic_with_gpar(self):
        m = HeuristicMetrics()
        gp = Gpar(fontsize=24)
        r1 = m.measure("X")
        r2 = m.measure("X", gp=gp)
        assert r2["width"] > r1["width"]  # Larger font = wider

    def test_cairo_backend_available(self):
        m = CairoFontMetrics()
        result = m.measure("Hello")
        assert result["width"] > 0
        assert result["ascent"] > 0

    def test_get_font_backend(self):
        backend = get_font_backend()
        assert hasattr(backend, "measure")
        result = backend.measure("test")
        assert isinstance(result, dict)


# ===================================================================== #
# WebRenderer — core functionality                                      #
# ===================================================================== #

class TestWebRendererInheritance:
    def test_is_grid_renderer(self):
        r = WebRenderer()
        assert isinstance(r, GridRenderer)

    def test_has_required_methods(self):
        r = WebRenderer()
        for method in ("draw_rect", "draw_circle", "draw_text", "draw_points",
                       "draw_line", "draw_polyline", "draw_segments",
                       "draw_polygon", "draw_path", "draw_raster",
                       "draw_roundrect", "push_viewport", "pop_viewport",
                       "resolve_x", "resolve_y", "resolve_w", "resolve_h",
                       "text_extents", "to_html", "to_scene_json", "save"):
            assert hasattr(r, method), f"Missing method: {method}"


class TestWebRendererDrawPrimitives:
    def setup_method(self):
        self.r = WebRenderer(width=7, height=5, dpi=150)

    def _children(self):
        return self.r._scene_root.children

    def test_draw_rect(self):
        self.r.draw_rect(0.5, 0.5, 0.3, 0.2)
        assert len(self._children()) == 1
        d = self._children()[0].to_dict()
        assert d["type"] == "rect"
        assert "x" in d["props"]
        assert "w" in d["props"]

    def test_draw_circle(self):
        self.r.draw_circle(0.5, 0.5, 0.1)
        d = self._children()[0].to_dict()
        assert d["type"] == "circle"
        assert "r" in d["props"]

    def test_draw_text(self):
        self.r.draw_text(0.5, 0.5, "Hello", rot=45.0)
        d = self._children()[0].to_dict()
        assert d["type"] == "text"
        assert d["props"]["label"] == "Hello"
        assert d["props"]["rot"] == 45.0
        assert d["render_hint"] == "svg"  # Text always SVG

    def test_draw_points(self):
        self.r.draw_points(np.array([0.1, 0.5, 0.9]),
                           np.array([0.2, 0.4, 0.6]))
        d = self._children()[0].to_dict()
        assert d["type"] == "points"
        assert len(d["props"]["x"]) == 3
        assert len(d["props"]["y"]) == 3

    def test_draw_polyline(self):
        self.r.draw_polyline(np.array([0.0, 0.5, 1.0]),
                             np.array([0.0, 1.0, 0.0]))
        d = self._children()[0].to_dict()
        assert d["type"] == "polyline"

    def test_draw_polyline_with_id(self):
        self.r.draw_polyline(
            np.array([0.0, 0.5, 0.5, 1.0]),
            np.array([0.0, 1.0, 0.0, 1.0]),
            id_=np.array([1, 1, 2, 2]),
        )
        d = self._children()[0].to_dict()
        assert d["type"] == "polyline"
        assert "groups" in d["props"]
        assert len(d["props"]["groups"]) == 2

    def test_draw_segments(self):
        self.r.draw_segments(
            np.array([0.0, 0.5]), np.array([0.0, 0.5]),
            np.array([1.0, 1.0]), np.array([1.0, 0.0]),
        )
        d = self._children()[0].to_dict()
        assert d["type"] == "segments"
        assert len(d["props"]["x0"]) == 2

    def test_draw_polygon(self):
        self.r.draw_polygon(np.array([0.0, 0.5, 1.0]),
                            np.array([0.0, 1.0, 0.0]))
        d = self._children()[0].to_dict()
        assert d["type"] == "polygon"

    def test_draw_path(self):
        self.r.draw_path(
            np.array([0.0, 0.5, 1.0, 0.2, 0.8, 0.5]),
            np.array([0.0, 1.0, 0.0, 0.3, 0.3, 0.8]),
            np.array([1, 1, 1, 2, 2, 2]),
        )
        d = self._children()[0].to_dict()
        assert d["type"] == "path"
        assert len(d["props"]["groups"]) == 2

    def test_draw_roundrect(self):
        self.r.draw_roundrect(0.5, 0.5, 0.6, 0.4, r=0.05)
        d = self._children()[0].to_dict()
        assert d["type"] == "roundrect"
        assert d["props"]["r"] > 0

    def test_draw_raster(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        self.r.draw_raster(img, 0.0, 0.0, 1.0, 1.0)
        d = self._children()[0].to_dict()
        assert d["type"] == "raster"
        assert d["props"]["src"].startswith("data:image/png;base64,")

    def test_move_to_line_to(self):
        self.r.move_to(0.1, 0.2)
        self.r.line_to(0.5, 0.8)
        d = self._children()[0].to_dict()
        assert d["type"] == "polyline"
        assert len(d["props"]["x"]) == 2


class TestWebRendererGpar:
    def test_gpar_serialisation(self):
        r = WebRenderer()
        gp = Gpar(fill="red", col="blue", lwd=2, fontsize=14, alpha=0.8)
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=gp)
        d = r._scene_root.children[0].to_dict()
        # Colours are now normalised through the shared R colour table
        assert d["gpar"]["fill"] == "rgb(255,0,0)"
        assert d["gpar"]["col"] == "rgb(0,0,255)"
        assert d["gpar"]["lwd"] == 2.0
        assert d["gpar"]["fontsize"] == 14.0
        assert d["gpar"]["alpha"] == 0.8


class TestWebRendererPathCollection:
    def test_path_collect_stroke(self):
        r = WebRenderer()
        r.save_state()
        r.begin_path_collect()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        r.end_path_stroke(Gpar(col="red"))
        r.restore_state()
        d = r._scene_root.children[0].to_dict()
        assert d["type"] == "compound_stroke"
        assert len(d["props"]["sub_paths"]) == 1

    def test_path_collect_fill(self):
        r = WebRenderer()
        r.begin_path_collect(rule="evenodd")
        r.draw_circle(0.5, 0.5, 0.2)
        r.end_path_fill(Gpar(fill="blue"))
        d = r._scene_root.children[0].to_dict()
        assert d["type"] == "compound_fill"
        assert d["props"]["rule"] == "evenodd"

    def test_path_collect_fill_stroke(self):
        r = WebRenderer()
        r.begin_path_collect()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        r.draw_circle(0.3, 0.3, 0.1)
        r.end_path_fill_stroke()
        d = r._scene_root.children[0].to_dict()
        assert d["type"] == "compound_fill_stroke"
        assert len(d["props"]["sub_paths"]) == 2


class TestWebRendererViewport:
    def test_push_pop_viewport(self):
        from grid_py._viewport import Viewport
        r = WebRenderer(width=7, height=5)
        vp = Viewport(name="test", x=0.5, y=0.5, width=0.5, height=0.5)
        r.push_viewport(vp)
        assert len(r._node_stack) == 2
        vp_dict = r._scene_root.children[0].to_dict()
        assert vp_dict["type"] == "viewport"
        assert vp_dict["name"] == "test"
        r.pop_viewport()
        assert len(r._node_stack) == 1

    def test_nested_viewports(self):
        from grid_py._viewport import Viewport
        r = WebRenderer()
        vp1 = Viewport(name="outer", x=0.5, y=0.5, width=0.8, height=0.8)
        vp2 = Viewport(name="inner", x=0.5, y=0.5, width=0.5, height=0.5)
        r.push_viewport(vp1)
        r.draw_rect(0.5, 0.5, 1.0, 1.0)
        r.push_viewport(vp2)
        r.draw_circle(0.5, 0.5, 0.3)
        r.pop_viewport()
        r.pop_viewport()

        outer = r._scene_root.children[0]
        assert outer.node_type == "viewport"
        assert outer.name == "outer"
        assert len(outer.children) == 2  # rect + inner viewport
        inner = outer.children[1]
        assert inner.node_type == "viewport"
        assert inner.name == "inner"
        assert len(inner.children) == 1  # circle


class TestWebRendererOutput:
    def test_to_scene_json(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        j = r.to_scene_json()
        parsed = json.loads(j)
        assert parsed["version"] == 1
        assert "root" in parsed
        assert "defs" in parsed

    def test_to_html(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        html = r.to_html()
        assert "<!DOCTYPE html>" in html
        assert "gridpy.render" in html
        assert "d3.v7.min.js" in html
        assert "gridpy-container" in html  # CSS class

    def test_new_page(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        assert len(r._scene_root.children) == 1
        r.new_page()
        assert len(r._scene_root.children) == 0

    def test_text_extents(self):
        r = WebRenderer()
        m = r.text_extents("Hello")
        assert "ascent" in m
        assert "descent" in m
        assert "width" in m
        assert m["width"] > 0

    def test_save(self, tmp_path):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        path = tmp_path / "test.html"
        r.save(str(path))
        assert path.exists()
        content = path.read_text()
        assert "gridpy.render" in content


class TestWebRendererCoordinateAccuracy:
    """Verify that WebRenderer draw_* methods pass through device coordinates.

    draw_* methods receive device coordinates (already resolved by _draw.py).
    They must record those values directly in the scene graph without
    applying any additional transform.
    """

    def test_rect_device_passthrough(self):
        """draw_rect stores device coords directly (with hjust/vjust applied)."""
        for x, y, w, h in [(350, 250, 200, 100), (0, 0, 700, 500),
                            (100, 50, 30, 30)]:
            web_r = WebRenderer(width=7, height=5, dpi=150)
            web_r.draw_rect(x, y, w, h, hjust=0.5, vjust=0.5)
            d = web_r._scene_root.children[0].to_dict()
            # hjust=0.5, vjust=0.5 → x0 = x - w*0.5, y0 = y - h*0.5
            assert abs(d["props"]["x"] - (x - w * 0.5)) < 1e-9
            assert abs(d["props"]["y"] - (y - h * 0.5)) < 1e-9
            assert abs(d["props"]["w"] - w) < 1e-9
            assert abs(d["props"]["h"] - h) < 1e-9
            web_r.new_page()

    def test_circle_device_passthrough(self):
        web_r = WebRenderer(width=7, height=5, dpi=150)
        web_r.draw_circle(300.0, 200.0, 50.0)
        d = web_r._scene_root.children[0].to_dict()
        assert abs(d["props"]["x"] - 300.0) < 1e-9
        assert abs(d["props"]["y"] - 200.0) < 1e-9
        assert abs(d["props"]["r"] - 50.0) < 1e-9

    def test_text_device_passthrough(self):
        web_r = WebRenderer(width=7, height=5, dpi=150)
        web_r.draw_text(350.0, 100.0, "hello")
        d = web_r._scene_root.children[0].to_dict()
        assert abs(d["props"]["x"] - 350.0) < 1e-9
        assert abs(d["props"]["y"] - 100.0) < 1e-9

    def test_points_device_passthrough(self):
        xs = np.array([100.0, 350.0, 600.0])
        ys = np.array([50.0, 250.0, 450.0])
        web_r = WebRenderer(width=7, height=5, dpi=150)
        web_r.draw_points(xs, ys)
        d = web_r._scene_root.children[0].to_dict()
        for i in range(3):
            assert abs(d["props"]["x"][i] - float(xs[i])) < 1e-9
            assert abs(d["props"]["y"][i] - float(ys[i])) < 1e-9

    def test_end_to_end_rect_matches_cairo(self):
        """Full pipeline: grid_draw(rect_grob) produces same device coords for both renderers."""
        from grid_py import CairoRenderer, get_state, grid_draw, rect_grob, Gpar

        cairo_r = CairoRenderer(width=7, height=5, dpi=100)
        get_state().init_device(cairo_r)
        grid_draw(rect_grob(x=0.5, y=0.5, width=0.8, height=0.6,
                             gp=Gpar(fill='red')))

        web_r = WebRenderer(width=7, height=5, dpi=100)
        get_state().init_device(web_r)
        grid_draw(rect_grob(x=0.5, y=0.5, width=0.8, height=0.6,
                             gp=Gpar(fill='red')))

        d = web_r._scene_root.children[0].to_dict()
        # Both renderers should place the rect at the same device position
        # 0.8 npc wide on 700px = 560px; 0.6 npc tall on 500px = 300px
        assert abs(d["props"]["w"] - 560.0) < 1.0
        assert abs(d["props"]["h"] - 300.0) < 1.0


class TestWebRendererVectorizedGpar:
    """Test that per-element gpar (vectorized colors, lwd, etc.) serializes correctly."""

    def test_multi_color_serialization(self):
        r = WebRenderer()
        gp = Gpar(col=["red", "blue", "green"])
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=gp)
        d = r._scene_root.children[0].to_dict()
        assert isinstance(d["gpar"]["col"], list)
        assert len(d["gpar"]["col"]) == 3
        assert d["gpar"]["col"][0] == "rgb(255,0,0)"

    def test_multi_fill_serialization(self):
        r = WebRenderer()
        gp = Gpar(fill=["#FF0000", "#00FF00", "#0000FF"])
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=gp)
        d = r._scene_root.children[0].to_dict()
        assert isinstance(d["gpar"]["fill"], list)
        assert len(d["gpar"]["fill"]) == 3

    def test_multi_lwd_serialization(self):
        r = WebRenderer()
        gp = Gpar(lwd=[1.0, 2.0, 3.0])
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=gp)
        d = r._scene_root.children[0].to_dict()
        assert d["gpar"]["lwd"] == [1.0, 2.0, 3.0]

    def test_single_element_list_unwrapped(self):
        r = WebRenderer()
        gp = Gpar(col=["red"])
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=gp)
        d = r._scene_root.children[0].to_dict()
        assert d["gpar"]["col"] == "rgb(255,0,0)"  # Unwrapped and normalised

    def test_empty_list_gpar_rejected_by_gpar(self):
        """Gpar rejects empty lists — this is correct R behaviour."""
        with pytest.raises(ValueError, match="must not be length 0"):
            Gpar(col=[])


class TestWebRendererGradientPattern:
    """Test gradient and pattern serialization into scene graph defs."""

    def test_linear_gradient_in_fill(self):
        from grid_py._patterns import LinearGradient
        from grid_py._units import Unit
        grad = LinearGradient(
            colours=["red", "blue"], stops=[0.0, 1.0],
            x1=Unit(0, "npc"), y1=Unit(0, "npc"),
            x2=Unit(1, "npc"), y2=Unit(1, "npc"),
        )
        r = WebRenderer()
        gp = Gpar(fill=grad)
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=gp)
        d = r._scene_root.children[0].to_dict()
        # Fill should be a url(#grad-N) reference
        assert d["gpar"]["fill"].startswith("url(#grad-")
        # Defs should have the gradient
        assert len(r._defs.gradients) == 1
        g = r._defs.gradients[0]
        assert g["type"] == "linear"
        assert g["colours"] == ["red", "blue"]
        assert g["stops"] == [0.0, 1.0]

    def test_radial_gradient_in_fill(self):
        from grid_py._patterns import RadialGradient
        from grid_py._units import Unit
        grad = RadialGradient(
            colours=["white", "black"], stops=[0.0, 1.0],
            cx1=Unit(0.5, "npc"), cy1=Unit(0.5, "npc"), r1=Unit(0, "npc"),
            cx2=Unit(0.5, "npc"), cy2=Unit(0.5, "npc"), r2=Unit(0.5, "npc"),
        )
        r = WebRenderer()
        gp = Gpar(fill=grad)
        r.draw_circle(0.5, 0.5, 0.3, gp=gp)
        assert len(r._defs.gradients) == 1
        assert r._defs.gradients[0]["type"] == "radial"


class TestWebRendererClipping:
    """Test that viewport clipping generates correct clip paths in defs."""

    def test_clip_viewport_creates_def(self):
        from grid_py._viewport import Viewport
        r = WebRenderer(width=7, height=5)
        vp = Viewport(name="clipped", x=0.5, y=0.5, width=0.5, height=0.5, clip=True)
        r.push_viewport(vp)
        # Should have a clip path in defs
        assert len(r._defs.clip_paths) == 1
        cp = r._defs.clip_paths[0]
        assert "id" in cp
        assert cp["w"] > 0
        assert cp["h"] > 0
        # Viewport node should reference it
        vp_node = r._scene_root.children[0]
        assert vp_node.clip_id == cp["id"]
        r.pop_viewport()

    def test_no_clip_no_def(self):
        from grid_py._viewport import Viewport
        r = WebRenderer(width=7, height=5)
        vp = Viewport(name="unclipped", x=0.5, y=0.5, width=0.5, height=0.5)
        r.push_viewport(vp)
        assert len(r._defs.clip_paths) == 0
        r.pop_viewport()


class TestWebRendererNewPage:
    """Test that new_page() completely resets all state."""

    def test_new_page_resets_children(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        r.draw_circle(0.5, 0.5, 0.1)
        assert len(r._scene_root.children) == 2
        r.new_page()
        assert len(r._scene_root.children) == 0

    def test_new_page_resets_defs(self):
        from grid_py._patterns import LinearGradient
        from grid_py._units import Unit
        r = WebRenderer()
        grad = LinearGradient(colours=["red", "blue"], stops=[0.0, 1.0],
                              x1=Unit(0, "npc"), y1=Unit(0, "npc"),
                              x2=Unit(1, "npc"), y2=Unit(1, "npc"))
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=Gpar(fill=grad))
        assert len(r._defs.gradients) == 1
        r.new_page()
        assert len(r._defs.gradients) == 0

    def test_new_page_resets_viewport_stack(self):
        from grid_py._viewport import Viewport
        r = WebRenderer(width=7, height=5)
        vp = Viewport(name="test", x=0.5, y=0.5, width=0.5, height=0.5)
        r.push_viewport(vp)
        assert len(r._vp_transform_stack) == 2
        r.new_page()
        assert len(r._vp_transform_stack) == 1
        assert len(r._node_stack) == 1

    def test_new_page_resets_id_generator(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        first_id = r._scene_root.children[0].node_id
        r.new_page()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        second_id = r._scene_root.children[0].node_id
        # After reset, IDs should restart (vp-0 takes first, grob gets next)
        assert second_id == first_id  # both should be grob-1


class TestWebRendererHTMLOutput:
    """Verify HTML output structure and correctness."""

    def test_html_contains_valid_json(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=Gpar(fill="red"))
        html = r.to_html()
        # Extract the JSON from the HTML
        start = html.index("gridpy.render(")
        # Find the JSON object (second argument)
        bracket_start = html.index("{", start)
        # Find matching closing — simple approach: use json_str from to_scene_json
        json_str = r.to_scene_json()
        assert json_str in html, "Scene graph JSON must be embedded in HTML"
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["version"] == 1

    def test_html_contains_css(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        html = r.to_html()
        assert "gridpy-container" in html
        assert "gridpy-tooltip" in html
        assert "gridpy-canvas" in html

    def test_html_contains_js_runtime(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        html = r.to_html()
        assert "var gridpy" in html
        assert "function render" in html
        assert "CANVAS_THRESHOLD" in html

    def test_theme_dark_in_html(self):
        r = WebRenderer(theme="dark")
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        html = r.to_html()
        assert 'theme: "dark"' in html

    def test_non_interactive_mode(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        html = r.to_html(interactive=False)
        assert "interactive: false" in html


class TestWebRendererEdgeCases:
    """Edge cases that should not crash."""

    def test_empty_scene_graph(self):
        r = WebRenderer()
        j = r.to_scene_json()
        parsed = json.loads(j)
        assert parsed["root"]["children"] == []

    def test_draw_polygon_too_few_points(self):
        r = WebRenderer()
        r.draw_polygon(np.array([0.1, 0.2]), np.array([0.3, 0.4]))
        assert len(r._scene_root.children) == 0  # Silently skipped

    def test_draw_line_too_few_points(self):
        r = WebRenderer()
        r.draw_line(np.array([0.1]), np.array([0.2]))
        assert len(r._scene_root.children) == 0  # Silently skipped

    def test_draw_with_none_gpar(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=None)
        d = r._scene_root.children[0].to_dict()
        assert d["gpar"] == {}

    def test_unicode_text(self):
        r = WebRenderer()
        r.draw_text(0.5, 0.5, "Hello 世界 🌍")
        d = r._scene_root.children[0].to_dict()
        assert d["props"]["label"] == "Hello 世界 🌍"
        # Verify JSON roundtrip
        j = r.to_scene_json()
        parsed = json.loads(j)
        assert parsed["root"]["children"][0]["props"]["label"] == "Hello 世界 🌍"

    def test_zero_size_viewport(self):
        from grid_py._viewport import Viewport
        r = WebRenderer(width=7, height=5)
        vp = Viewport(name="zero", x=0.5, y=0.5, width=0.0, height=0.0)
        r.push_viewport(vp)
        r.draw_rect(0.5, 0.5, 1.0, 1.0)
        r.pop_viewport()
        # Should not crash
        j = r.to_scene_json()
        json.loads(j)  # Valid JSON

    def test_many_grobs_performance(self):
        """Verify we can handle a large number of grobs without issue."""
        r = WebRenderer()
        xs = np.random.rand(5000)
        ys = np.random.rand(5000)
        r.draw_points(xs, ys)
        d = r._scene_root.children[0].to_dict()
        assert len(d["props"]["x"]) == 5000
        # JSON should be serializable
        j = r.to_scene_json()
        assert len(j) > 10000


class TestWebRendererFullPipeline:
    """Test WebRenderer through the standard grid_py drawing pipeline."""

    def test_grid_draw_integration(self):
        from grid_py._draw import grid_draw
        from grid_py._primitives import rect_grob, circle_grob, text_grob
        from grid_py._state import get_state

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)

        grid_draw(rect_grob(x=0.5, y=0.5, width=0.8, height=0.6,
                  gp=Gpar(fill="lightblue", col="navy")))
        grid_draw(circle_grob(x=0.3, y=0.3, r=0.1, gp=Gpar(fill="red")))
        grid_draw(text_grob(label="Test", x=0.5, y=0.9))

        sg = json.loads(r.to_scene_json())
        assert len(sg["root"]["children"]) == 3
        types = [c["type"] for c in sg["root"]["children"]]
        assert "rect" in types
        assert "circle" in types
        assert "text" in types

    def test_viewport_integration(self):
        from grid_py._draw import grid_draw
        from grid_py._primitives import rect_grob
        from grid_py._state import get_state
        from grid_py._viewport import Viewport, push_viewport, pop_viewport

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)

        vp = Viewport(name="panel", x=0.5, y=0.5, width=0.8, height=0.8)
        push_viewport(vp, recording=False)
        r.push_viewport(vp)
        grid_draw(rect_grob(x=0.5, y=0.5, width=1.0, height=1.0,
                  gp=Gpar(fill="white", col="black")))
        r.pop_viewport()

        sg = json.loads(r.to_scene_json())
        # Root should have a viewport child
        root_children = sg["root"]["children"]
        vp_nodes = [c for c in root_children if c["type"] == "viewport"]
        assert len(vp_nodes) >= 1
