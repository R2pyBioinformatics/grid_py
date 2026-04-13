"""Scene graph data model for the WebRenderer.

Provides a JSON-serializable tree representation of the entire grid scene,
including viewports, grobs, and shared definitions (gradients, patterns,
clip paths, masks).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "SceneNode",
    "ViewportNode",
    "GrobNode",
    "DefsCollection",
    "SceneGraph",
]


# ---------------------------------------------------------------------------
# Auto-incrementing ID generator
# ---------------------------------------------------------------------------

class _IdGenerator:
    """Thread-safe auto-incrementing ID generator.

    Each call to ``next(prefix)`` returns ``"{prefix}-{counter}"`` where
    *counter* is a monotonically increasing integer starting at 0.
    """

    def __init__(self) -> None:
        self._counter: int = 0
        self._lock = threading.Lock()

    def next(self, prefix: str = "node") -> str:
        """Return the next unique ID string with the given *prefix*."""
        with self._lock:
            node_id = f"{prefix}-{self._counter}"
            self._counter += 1
            return node_id

    def reset(self) -> None:
        """Reset the counter to zero."""
        with self._lock:
            self._counter = 0


# Module-level singleton shared by factory helpers.
_id_gen = _IdGenerator()


# ---------------------------------------------------------------------------
# Scene nodes
# ---------------------------------------------------------------------------

@dataclass
class SceneNode:
    """Base node in the scene graph tree.

    Parameters
    ----------
    node_id : str
        Unique identifier for this node.
    node_type : str
        Discriminator tag persisted in the serialized form (e.g.
        ``"viewport"``, ``"rect"``, ``"text"``).
    children : list[SceneNode]
        Ordered child nodes.
    metadata : dict
        Arbitrary key/value pairs attached to this node.
    """

    node_id: str
    node_type: str
    children: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Recursively convert the node tree to a plain dict.

        The returned structure is directly suitable for
        ``json.dumps(node.to_dict())``.  Keys use the short names
        expected by the JavaScript runtime (``id``, ``type``).
        """
        d: dict = {
            "id": self.node_id,
            "type": self.node_type,
            "children": [child.to_dict() for child in self.children],
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class ViewportNode(SceneNode):
    """A viewport (coordinate-system container) in the scene tree.

    Parameters
    ----------
    name : str
        The viewport name (mirrors the grid viewport name).
    transform : dict
        Viewport placement expressed as ``{"x0": ..., "y0": ...,
        "w": ..., "h": ...}`` in the parent coordinate system.
    clip : bool
        Whether drawing is clipped to the viewport boundary.
    clip_id : str or None
        Reference to a ``<clipPath>`` element in *DefsCollection*.
    mask_id : str or None
        Reference to a ``<mask>`` element in *DefsCollection*.
    mask_type : str or None
        Mask compositing type (e.g. ``"luminance"``, ``"alpha"``).
    """

    name: str = ""
    transform: dict = field(default_factory=lambda: {"x0": 0.0, "y0": 0.0, "w": 1.0, "h": 1.0})
    clip: bool = False
    clip_id: Optional[str] = None
    mask_id: Optional[str] = None
    mask_type: Optional[str] = None

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["name"] = self.name
        d["transform"] = self.transform
        d["clip"] = self.clip
        if self.clip_id is not None:
            d["clip_id"] = self.clip_id
        if self.mask_id is not None:
            d["mask_id"] = self.mask_id
        if self.mask_type is not None:
            d["mask_type"] = self.mask_type
        return d


@dataclass
class GrobNode(SceneNode):
    """A graphical object (grob) leaf or branch in the scene tree.

    Parameters
    ----------
    props : dict
        Grob-specific geometric properties (e.g. ``x``, ``y``,
        ``width``, ``height`` for a rect grob).
    gpar : dict
        Serialized graphical parameters (fill, colour, lwd, ...).
    render_hint : str
        Hint for the web renderer: ``"auto"``, ``"svg"``, or
        ``"canvas"``.
    """

    props: dict = field(default_factory=dict)
    gpar: dict = field(default_factory=dict)
    render_hint: str = "auto"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["props"] = self.props
        d["gpar"] = self.gpar
        d["render_hint"] = self.render_hint
        # Pass through metadata as "data" for JS tooltip consumption
        if self.metadata:
            d["data"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# Shared definitions (gradients, patterns, clip paths, masks)
# ---------------------------------------------------------------------------

@dataclass
class DefsCollection:
    """Container for shared definitions referenced by ID from scene nodes.

    Each list stores plain dicts that must contain at least an ``"id"``
    key together with type-specific fields.
    """

    gradients: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    clip_paths: list = field(default_factory=list)
    masks: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gradients": list(self.gradients),
            "patterns": list(self.patterns),
            "clip_paths": list(self.clip_paths),
            "masks": list(self.masks),
        }


# ---------------------------------------------------------------------------
# Top-level scene graph
# ---------------------------------------------------------------------------

@dataclass
class SceneGraph:
    """Top-level scene graph container.

    Parameters
    ----------
    version : int
        Schema version (currently ``1``).
    width : float
        Device width in user units.
    height : float
        Device height in user units.
    dpi : float
        Dots-per-inch used for unit conversion.
    root : ViewportNode
        The root viewport of the scene tree.
    defs : DefsCollection
        Shared definitions (gradients, patterns, clips, masks).
    """

    width: float
    height: float
    dpi: float
    root: ViewportNode
    version: int = 1
    defs: DefsCollection = field(default_factory=DefsCollection)

    def to_dict(self) -> dict:
        """Return the full scene graph as a plain nested dict."""
        return {
            "version": self.version,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "root": self.root.to_dict(),
            "defs": self.defs.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the scene graph to a JSON string.

        Parameters
        ----------
        indent : int or None
            JSON indentation level. Pass ``None`` for compact output.
        """
        return json.dumps(self.to_dict(), indent=indent)
