"""Scene graph data model for the WebRenderer.

Provides a JSON-serializable tree representation of the entire grid scene,
including viewports, grobs, and shared definitions (gradients, patterns,
clip paths, masks).

Schema v1.1 (backward compatible with v1):
    - Adds ``schema_version`` (string, e.g. ``"1.1"``).
    - Adds optional ``interaction_modules`` list of JS module names that
      should be loaded client-side.
    - Adds optional ``data_grids`` for large-matrix DataGrid bypass —
      consumers (upper-layer packages) register compressed value arrays
      keyed by id; JS runtime resolves tooltips against these.
    - Adds optional ``entity_index`` built in WebRenderer.finalize():
      ``{row: {id: [grob_id, ...]}, col: {id: [grob_id, ...]}}`` for
      fast hover-highlight and cross-panel linkage.

All added fields are optional — an older client handling a v1 payload
will simply ignore them.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "SceneNode",
    "ViewportNode",
    "GrobNode",
    "DataGridNode",
    "DefsCollection",
    "SceneGraph",
]


# ---------------------------------------------------------------------------
# Auto-incrementing ID generator
# ---------------------------------------------------------------------------

class _IdGenerator:
    """Thread-safe auto-incrementing ID generator."""

    def __init__(self) -> None:
        self._counter: int = 0
        self._lock = threading.Lock()

    def next(self, prefix: str = "node") -> str:
        with self._lock:
            node_id = f"{prefix}-{self._counter}"
            self._counter += 1
            return node_id

    def reset(self) -> None:
        with self._lock:
            self._counter = 0


_id_gen = _IdGenerator()


# ---------------------------------------------------------------------------
# Scene nodes
# ---------------------------------------------------------------------------

@dataclass
class SceneNode:
    """Base node in the scene graph tree."""

    node_id: str
    node_type: str
    children: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
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
    """A viewport (coordinate-system container) in the scene tree."""

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
    """A graphical object (grob) leaf or branch in the scene tree."""

    props: dict = field(default_factory=dict)
    gpar: dict = field(default_factory=dict)
    render_hint: str = "auto"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["props"] = self.props
        d["gpar"] = self.gpar
        d["render_hint"] = self.render_hint
        if self.metadata:
            d["data"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# DataGrid — compact representation of a large matrix body for tooltips
# ---------------------------------------------------------------------------

@dataclass
class DataGridNode:
    """Compact matrix-value payload registered alongside the scene graph.

    Upper-layer packages (complexheatmap-py) emit one DataGridNode per
    rasterized heatmap body so the JS runtime can resolve tooltips by
    pixel → (row_idx, col_idx) lookup instead of needing one grob per
    cell.  The encoded ``values`` field is a base64-encoded typed array
    whose dtype is declared by ``value_dtype``.

    Parameters
    ----------
    id : str
        Stable identifier matching the raster grob's ``data_grid_ref``.
    row_ids, col_ids : list[int]
        Original-matrix global indices for each row/column, in draw order.
    row_names, col_names : list[str]
        Display names aligned with *row_ids* / *col_ids*.
    values : str
        Base64-encoded typed array (row-major).
    value_shape : [int, int]
        ``[nrow, ncol]`` describing the decoded array.
    value_dtype : str
        ``"float32"`` (default) or ``"quantile8"`` (8-bit quantile codes
        accompanied by a lookup table in *value_lut*).
    value_lut : list[float] or None
        Required when ``value_dtype == "quantile8"`` — the 256-entry
        lookup table mapping code → float value.
    annotations : dict
        Optional per-row / per-column annotation payload, accessible from
        the tooltip template under the ``{annotations.*}`` namespace.
    viewport_name : str
        Viewport under which this grid is drawn — used to resolve pixel
        coordinates into (row, col) indices on hover.
    """

    id: str
    row_ids: List[int] = field(default_factory=list)
    col_ids: List[int] = field(default_factory=list)
    row_names: List[str] = field(default_factory=list)
    col_names: List[str] = field(default_factory=list)
    values: str = ""
    value_shape: List[int] = field(default_factory=lambda: [0, 0])
    value_dtype: str = "float32"
    value_lut: Optional[List[float]] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    viewport_name: str = ""

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "id": self.id,
            "row_ids": list(self.row_ids),
            "col_ids": list(self.col_ids),
            "row_names": list(self.row_names),
            "col_names": list(self.col_names),
            "values": self.values,
            "value_shape": list(self.value_shape),
            "value_dtype": self.value_dtype,
            "viewport_name": self.viewport_name,
        }
        if self.value_lut is not None:
            d["value_lut"] = list(self.value_lut)
        if self.annotations:
            d["annotations"] = self.annotations
        return d


# ---------------------------------------------------------------------------
# Shared definitions
# ---------------------------------------------------------------------------

@dataclass
class DefsCollection:
    """Container for shared definitions referenced by ID from scene nodes."""

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

SCHEMA_VERSION = "1.1"


@dataclass
class SceneGraph:
    """Top-level scene graph container.

    Schema v1.1 fields (all optional):
        - ``interaction_modules``: list of JS module names to load.
        - ``data_grids``: list of DataGridNode payloads.
        - ``entity_index``: {row: {id: [grob_id]}, col: {id: [grob_id]}}.
        - ``tooltip_templates``: registry of compiled tooltip templates,
          keyed by template name.
    """

    width: float
    height: float
    dpi: float
    root: ViewportNode
    version: int = 1
    defs: DefsCollection = field(default_factory=DefsCollection)
    # v1.1 additions
    schema_version: str = SCHEMA_VERSION
    interaction_modules: List[str] = field(default_factory=list)
    data_grids: List[DataGridNode] = field(default_factory=list)
    entity_index: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    tooltip_templates: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return the full scene graph as a plain nested dict."""
        d: Dict[str, Any] = {
            "version": self.version,
            "schema_version": self.schema_version,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "root": self.root.to_dict(),
            "defs": self.defs.to_dict(),
        }
        if self.interaction_modules:
            d["interaction_modules"] = list(self.interaction_modules)
        if self.data_grids:
            d["data_grids"] = [g.to_dict() for g in self.data_grids]
        if self.entity_index:
            d["entity_index"] = {
                axis: {str(k): list(v) for k, v in entries.items()}
                for axis, entries in self.entity_index.items()
            }
        if self.tooltip_templates:
            d["tooltip_templates"] = self.tooltip_templates
        return d

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
