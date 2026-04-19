/**
 * gridpy-heatmap.js — ComplexHeatmap interaction module.
 *
 * Loaded into the generated HTML when the WebRenderer has
 * ``interaction_modules=["gridpy-heatmap"]``.  Registers itself with the
 * core ``gridpy`` runtime and layers heatmap-specific interactions on
 * top of the scene graph:
 *
 *   - cell / anno_cell / dend branch / label tooltips (template-driven)
 *   - row and column hover highlight (cross-slice + cross-panel)
 *   - click-to-pin selection
 *   - rectangular brush selection (Shift+drag or toolbar toggle)
 *   - dendrogram click → R-behavior sub-heatmap panel
 *   - legend click → categorical filter
 *   - continuous legend drag → dynamic colour threshold
 *   - search box (Ctrl+F) for row/col names
 *
 * All metadata is read from the ``data`` field of each GrobNode, per the
 * complexheatmap-py metadata envelope (see ``_interactive/schema.py``).
 *
 * @version 0.1.0
 */
(function () {
    "use strict";

    if (typeof gridpy === "undefined" || !gridpy.registerModule) {
        if (typeof console !== "undefined") {
            console.warn("gridpy-heatmap requires gridpy core runtime");
        }
        return;
    }

    // ---- Utilities --------------------------------------------------------

    function _isNum(v) { return typeof v === "number" && !isNaN(v); }

    function _get(obj, path) {
        if (!obj) return undefined;
        var parts = path.split(".");
        var cur = obj;
        for (var i = 0; i < parts.length; i++) {
            if (cur == null) return undefined;
            cur = cur[parts[i]];
        }
        return cur;
    }

    function _formatValue(val, spec) {
        if (val == null) return "";
        if (!spec) return String(val);
        var m = spec.match(/^\.(\d+)f$/);
        if (m && _isNum(val)) return Number(val).toFixed(parseInt(m[1], 10));
        m = spec.match(/^\.(\d+)e$/);
        if (m && _isNum(val)) return Number(val).toExponential(parseInt(m[1], 10));
        return String(val);
    }

    function _escapeHtml(s) {
        return String(s)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;")
            .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    /** Compile a template string with ``{field}`` / ``{field:.2f}`` tokens. */
    function compileTemplate(tmpl) {
        if (tmpl == null) return null;
        if (typeof tmpl === "function") return tmpl;
        var tokens = [];
        var re = /\{([^{}]+)\}/g;
        var last = 0, m;
        while ((m = re.exec(tmpl)) !== null) {
            if (m.index > last) tokens.push({ lit: tmpl.slice(last, m.index) });
            var expr = m[1];
            var ci = expr.indexOf(":");
            var path = ci >= 0 ? expr.slice(0, ci) : expr;
            var spec = ci >= 0 ? expr.slice(ci + 1) : null;
            tokens.push({ path: path, spec: spec });
            last = m.index + m[0].length;
        }
        if (last < tmpl.length) tokens.push({ lit: tmpl.slice(last) });
        return function (ctx) {
            var out = "";
            for (var i = 0; i < tokens.length; i++) {
                var t = tokens[i];
                if (t.lit !== undefined) { out += t.lit; continue; }
                var v = _get(ctx, t.path);
                out += _formatValue(v, t.spec);
            }
            return out;
        };
    }

    // ---- Tooltip rendering -----------------------------------------------

    function renderTooltipHtml(data, templates) {
        if (!data) return "";
        var ref = data.tooltip_ref || data.entity;
        var fn = templates[ref];
        var ctx = Object.assign({}, data.payload || {}, {
            entity: data.entity, heatmap: data.heatmap,
            row: data.row, col: data.col,
            slice: data.slice,
            annotations: data.payload && data.payload.annotations
        });
        if (fn) {
            try {
                var s = fn(ctx);
                if (s == null || s === "") return "";
                return s.replace(/\n/g, "<br>");
            } catch (e) { /* fall through */ }
        }
        var lines = [];
        var p = data.payload || {};
        for (var k in p) {
            if (k === "annotations") continue;
            lines.push("<b>" + _escapeHtml(k) + "</b>: " + _escapeHtml(p[k]));
        }
        return lines.join("<br>");
    }

    // ---- Spatial index for raster-backed DataGrids ------------------------

    function buildDataGridIndex(dataGrids, sceneGraph) {
        // Map viewport_name -> grid descriptor with device bounding box.
        var idx = {};
        if (!dataGrids || !dataGrids.length) return idx;
        var vpBoxes = collectViewportBBoxes(sceneGraph.root,
            { x0: 0, y0: 0, w: sceneGraph.width, h: sceneGraph.height });
        dataGrids.forEach(function (g) {
            var box = vpBoxes[g.viewport_name];
            if (!box) return;
            idx[g.id] = Object.assign({}, g, { box: box });
        });
        return idx;
    }

    function collectViewportBBoxes(node, parentBox) {
        var out = {};
        (function walk(n, box) {
            if (!n) return;
            if (n.type === "viewport" && n.name) {
                var t = n.transform || {};
                var nb = {
                    x0: t.x0 !== undefined ? t.x0 : box.x0,
                    y0: t.y0 !== undefined ? t.y0 : box.y0,
                    w: t.w !== undefined ? t.w : box.w,
                    h: t.h !== undefined ? t.h : box.h
                };
                out[n.name] = nb;
                (n.children || []).forEach(function (c) { walk(c, nb); });
            } else {
                (n.children || []).forEach(function (c) { walk(c, box); });
            }
        })(node, parentBox);
        return out;
    }

    function lookupDataGridCell(grid, mx, my) {
        // Returns {row_idx, col_idx, row_id, col_id, row_name, col_name, value} or null.
        var box = grid.box;
        if (mx < box.x0 || mx > box.x0 + box.w) return null;
        if (my < box.y0 || my > box.y0 + box.h) return null;
        var shape = grid.value_shape || [0, 0];
        var nr = shape[0], nc = shape[1];
        if (!nr || !nc) return null;
        var relX = (mx - box.x0) / box.w;
        var relY = (my - box.y0) / box.h;
        var col = Math.min(nc - 1, Math.max(0, Math.floor(relX * nc)));
        // Grid row 0 is drawn at the top; Cairo pixels are top-origin too.
        var row = Math.min(nr - 1, Math.max(0, Math.floor(relY * nr)));
        var val = _decodeValue(grid, row, col);
        return {
            row_idx: row, col_idx: col,
            row_id: grid.row_ids ? grid.row_ids[row] : row,
            col_id: grid.col_ids ? grid.col_ids[col] : col,
            row_name: grid.row_names ? grid.row_names[row] : String(row),
            col_name: grid.col_names ? grid.col_names[col] : String(col),
            value: val
        };
    }

    function _decodeValue(grid, row, col) {
        var shape = grid.value_shape;
        var idx = row * shape[1] + col;
        if (!grid._decoded) {
            if (grid.value_dtype === "float32") {
                var bin = atob(grid.values);
                var buf = new ArrayBuffer(bin.length);
                var view = new Uint8Array(buf);
                for (var i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
                grid._decoded = new Float32Array(buf);
            } else if (grid.value_dtype === "quantile8") {
                var bin2 = atob(grid.values);
                grid._decoded = new Uint8Array(bin2.length);
                for (var j = 0; j < bin2.length; j++) grid._decoded[j] = bin2.charCodeAt(j);
            } else {
                grid._decoded = new Float32Array(0);
            }
        }
        var raw = grid._decoded[idx];
        if (grid.value_dtype === "quantile8" && grid.value_lut) {
            return grid.value_lut[raw];
        }
        return raw;
    }

    // ---- Highlight overlay -----------------------------------------------

    function highlightEntities(state, axis, id) {
        clearHighlight(state);
        var bucket = (state.sceneGraph.entity_index || {})[axis] || {};
        var ids = bucket[String(id)] || [];
        ids.forEach(function (gid) {
            var el = state.grobById[gid];
            if (el) el.classList.add("gridpy-hl-" + axis);
        });
        state._hlAxis = axis;
        state._hlId = id;
    }

    function clearHighlight(state) {
        if (!state._hlAxis) return;
        var bucket = (state.sceneGraph.entity_index || {})[state._hlAxis] || {};
        var ids = bucket[String(state._hlId)] || [];
        ids.forEach(function (gid) {
            var el = state.grobById[gid];
            if (el) el.classList.remove("gridpy-hl-" + state._hlAxis);
        });
        state._hlAxis = null;
        state._hlId = null;
    }

    // ---- Sub-panel for dend click -----------------------------------------

    function openSubPanel(state, axis, ids) {
        closeSubPanel(state);
        var grid = state.heatmap && state.heatmap.primaryGrid;
        if (!grid) return;

        var panel = document.createElement("div");
        panel.className = "gridpy-subpanel gridpy-subpanel-" + axis;
        var mainBox = grid.box;

        var selN = ids.length;
        if (axis === "row") {
            panel.style.left = (mainBox.x0 + mainBox.w + 10) + "px";
            panel.style.top = mainBox.y0 + "px";
            panel.style.width = Math.floor(mainBox.w * 0.6) + "px";
            panel.style.height = mainBox.h + "px";
        } else {
            panel.style.left = mainBox.x0 + "px";
            panel.style.top = (mainBox.y0 + mainBox.h + 10) + "px";
            panel.style.width = mainBox.w + "px";
            panel.style.height = Math.floor(mainBox.h * 0.6) + "px";
        }

        var header = document.createElement("div");
        header.className = "gridpy-subpanel-header";
        header.innerHTML = "cluster &mdash; " + selN + " " + axis + "s" +
            "<span class='gridpy-subpanel-close' title='close'>&times;</span>";
        panel.appendChild(header);
        header.querySelector(".gridpy-subpanel-close").addEventListener("click",
            function () { closeSubPanel(state); });

        var canvas = document.createElement("canvas");
        var pw = parseInt(panel.style.width, 10) - 4;
        var ph = parseInt(panel.style.height, 10) - 24;
        canvas.width = pw;
        canvas.height = ph;
        canvas.style.width = pw + "px";
        canvas.style.height = ph + "px";
        panel.appendChild(canvas);

        state.container.appendChild(panel);
        state._subpanel = panel;

        var shape = grid.value_shape;
        var nr = shape[0], nc = shape[1];
        var selRows = axis === "row" ? ids.map(function (v) { return parseInt(v, 10); }) : allIndices(nr);
        var selCols = axis === "col" ? ids.map(function (v) { return parseInt(v, 10); }) : allIndices(nc);

        // Map global row/col id → local row/col index in the DataGrid
        var rowIdToIdx = {};
        (grid.row_ids || []).forEach(function (rid, i) { rowIdToIdx[String(rid)] = i; });
        var colIdToIdx = {};
        (grid.col_ids || []).forEach(function (cid, i) { colIdToIdx[String(cid)] = i; });
        var rIdx = selRows.map(function (r) { return rowIdToIdx[String(r)]; })
                         .filter(function (v) { return v !== undefined; });
        var cIdx = selCols.map(function (c) { return colIdToIdx[String(c)]; })
                         .filter(function (v) { return v !== undefined; });
        if (!rIdx.length || !cIdx.length) return;

        var cellW = pw / cIdx.length;
        var cellH = ph / rIdx.length;
        var ctx = canvas.getContext("2d");
        for (var i = 0; i < rIdx.length; i++) {
            for (var j = 0; j < cIdx.length; j++) {
                var v = _decodeValue(grid, rIdx[i], cIdx[j]);
                ctx.fillStyle = state.heatmap.colorMapper
                    ? state.heatmap.colorMapper(v) : "#888";
                ctx.fillRect(j * cellW, i * cellH, Math.ceil(cellW), Math.ceil(cellH));
            }
        }
    }

    function allIndices(n) { var a = []; for (var i = 0; i < n; i++) a.push(i); return a; }

    function closeSubPanel(state) {
        if (state._subpanel && state._subpanel.parentNode) {
            state._subpanel.parentNode.removeChild(state._subpanel);
        }
        state._subpanel = null;
    }

    // ---- Attach ----------------------------------------------------------

    function attach(state, sceneGraph, options) {
        var templates = {};
        var rawTemplates = sceneGraph.tooltip_templates || {};
        for (var k in rawTemplates) {
            templates[k] = compileTemplate(rawTemplates[k]);
        }
        state.heatmap = {
            templates: templates,
            dataGrids: buildDataGridIndex(sceneGraph.data_grids || [], sceneGraph),
            primaryGrid: null,
            colorMapper: defaultColorMapper(sceneGraph),
            pinned: false,
            pinnedAxis: null,
            pinnedId: null,
            // Bounding-box index for entity grobs that aren't rects/points
            // (e.g. dendrogram ``segments`` grobs) — used for hover/click
            // hit testing.
            entityRegions: buildEntityRegions(sceneGraph)
        };
        // Pick first DataGrid as "primary" for sub-panel geometry
        for (var gid in state.heatmap.dataGrids) {
            state.heatmap.primaryGrid = state.heatmap.dataGrids[gid];
            break;
        }

        // Replace default tooltip handler with entity-aware one
        if (options.tooltip !== false) {
            bindHeatmapTooltip(state);
        }
        bindHoverHighlight(state);
        bindClick(state);
        bindSearch(state, sceneGraph);

        // Threshold control: expose a simple slider docked to the container
        mountThresholdSlider(state);

        return {
            closeSubPanel: function () { closeSubPanel(state); },
            highlight: function (axis, id) { highlightEntities(state, axis, id); },
            clearHighlight: function () { clearHighlight(state); },
            setColorThreshold: function (minVal, maxVal) {
                updateBodyThreshold(state, minVal, maxVal);
            },
            getPrimarySpec: function () {
                var g = state.heatmap.primaryGrid;
                return g && g.annotations ? g.annotations.color_spec : null;
            }
        };
    }

    function mountThresholdSlider(state) {
        // Extract color_spec from primary grid (if any)
        var grid = state.heatmap.primaryGrid;
        if (!grid || !grid.annotations || !grid.annotations.color_spec) return;
        var spec = grid.annotations.color_spec;
        var breaks = spec.breaks;
        var origLow = breaks[0], origHigh = breaks[breaks.length - 1];

        var panel = document.createElement("div");
        panel.className = "gridpy-heatmap-threshold";
        panel.innerHTML =
            "<label>min <input type='number' step='0.1' class='gridpy-th-min' /></label>" +
            "<label>max <input type='number' step='0.1' class='gridpy-th-max' /></label>" +
            "<button class='gridpy-th-reset'>reset</button>";
        state.container.appendChild(panel);

        var minIn = panel.querySelector(".gridpy-th-min");
        var maxIn = panel.querySelector(".gridpy-th-max");
        var reset = panel.querySelector(".gridpy-th-reset");
        minIn.value = origLow;
        maxIn.value = origHigh;

        function onChange() {
            var lo = parseFloat(minIn.value), hi = parseFloat(maxIn.value);
            if (!isNaN(lo) && !isNaN(hi) && hi > lo) {
                updateBodyThreshold(state, lo, hi);
                gridpy.emit(state, "color_threshold", { min: lo, max: hi });
            }
        }
        minIn.addEventListener("change", onChange);
        maxIn.addEventListener("change", onChange);
        reset.addEventListener("click", function () {
            minIn.value = origLow;
            maxIn.value = origHigh;
            // Remove overlay to reveal original raster
            var overlays = state._bodyOverlays || {};
            Object.keys(overlays).forEach(function (k) {
                if (overlays[k].parentNode) overlays[k].parentNode.removeChild(overlays[k]);
                delete overlays[k];
            });
            gridpy.emit(state, "color_threshold", { min: origLow, max: origHigh });
        });
    }

    // ---- colorRamp2 — JS re-implementation for dynamic recolor ------------

    function _hexToRgb(hex) {
        hex = String(hex);
        if (hex[0] !== "#") {
            // css named colours not supported here — fallback to grey
            return [0.5, 0.5, 0.5];
        }
        var n = hex.slice(1);
        if (n.length === 3) n = n.split("").map(function (c) { return c + c; }).join("");
        var r = parseInt(n.slice(0, 2), 16) / 255;
        var g = parseInt(n.slice(2, 4), 16) / 255;
        var b = parseInt(n.slice(4, 6), 16) / 255;
        return [r, g, b];
    }

    function _rgbToHex(rgb) {
        function clamp(x) { return Math.max(0, Math.min(1, x)); }
        function hex2(v) {
            var s = Math.round(clamp(v) * 255).toString(16);
            return s.length === 1 ? "0" + s : s;
        }
        return "#" + hex2(rgb[0]) + hex2(rgb[1]) + hex2(rgb[2]);
    }

    function _rgbToLab(rgb) {
        // sRGB linearize
        function lin(v) {
            return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
        }
        var r = lin(rgb[0]), g = lin(rgb[1]), b = lin(rgb[2]);
        // sRGB D65 → XYZ
        var X = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
        var Y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.0;
        var Z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
        function f(t) { return t > 216 / 24389 ? Math.cbrt(t) : (t * 24389 / 27 + 16) / 116; }
        var fx = f(X), fy = f(Y), fz = f(Z);
        return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
    }

    function _labToRgb(lab) {
        var fy = (lab[0] + 16) / 116;
        var fx = fy + lab[1] / 500;
        var fz = fy - lab[2] / 200;
        function finv(t) {
            var t3 = t * t * t;
            return t3 > 216 / 24389 ? t3 : (t * 116 - 16) * 27 / 24389;
        }
        var X = finv(fx) * 0.95047;
        var Y = finv(fy) * 1.0;
        var Z = finv(fz) * 1.08883;
        var r = X * 3.2406 - Y * 1.5372 - Z * 0.4986;
        var g = -X * 0.9689 + Y * 1.8758 + Z * 0.0415;
        var b = X * 0.0557 - Y * 0.204 + Z * 1.057;
        function unlin(v) {
            return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
        }
        return [unlin(r), unlin(g), unlin(b)];
    }

    function buildColorRamp(spec) {
        // spec = { breaks: [...], colors: [...], space: "LAB"|"RGB", na_col }
        if (!spec || !spec.breaks || !spec.colors) return null;
        var breaks = spec.breaks.slice();
        var space = spec.space || "LAB";
        var rgbs = spec.colors.map(_hexToRgb);
        var encoded = space === "LAB" ? rgbs.map(_rgbToLab) : rgbs.map(function (c) {
            return c.slice();
        });
        var na = spec.na_col || "#cccccc";
        return function (v) {
            if (v == null || isNaN(v)) return na;
            if (v <= breaks[0]) return _rgbToHex(space === "LAB" ? _labToRgb(encoded[0]) : encoded[0]);
            var last = breaks.length - 1;
            if (v >= breaks[last]) return _rgbToHex(space === "LAB" ? _labToRgb(encoded[last]) : encoded[last]);
            for (var i = 0; i < last; i++) {
                var a = breaks[i], b = breaks[i + 1];
                if (v >= a && v <= b) {
                    var t = (v - a) / (b - a);
                    var ca = encoded[i], cb = encoded[i + 1];
                    var mix = [
                        ca[0] + (cb[0] - ca[0]) * t,
                        ca[1] + (cb[1] - ca[1]) * t,
                        ca[2] + (cb[2] - ca[2]) * t
                    ];
                    var rgb = space === "LAB" ? _labToRgb(mix) : mix;
                    return _rgbToHex(rgb);
                }
            }
            return na;
        };
    }

    function defaultColorMapper(sceneGraph) {
        // Resolve per-primary-DataGrid color_spec if available.
        var grids = sceneGraph.data_grids || [];
        for (var i = 0; i < grids.length; i++) {
            var ann = grids[i].annotations || {};
            var spec = ann.color_spec;
            if (spec) {
                var fn = buildColorRamp(spec);
                if (fn) return fn;
            }
        }
        return function (v) {
            if (v == null || isNaN(v)) return "#cccccc";
            var t = Math.max(-2, Math.min(2, v)) / 2;
            if (t >= 0) {
                var r = 255, g = Math.round(255 * (1 - t)), b = Math.round(255 * (1 - t));
            } else {
                var r = Math.round(255 * (1 + t)), g = Math.round(255 * (1 + t)), b = 255;
            }
            return "rgb(" + r + "," + g + "," + b + ")";
        };
    }

    // ---- Dynamic threshold: body recolor overlay ---------------------------
    //
    // When the user adjusts the continuous legend's endpoints (via the public
    // ``setColorThreshold(min, max)`` method on the heatmap module), each
    // DataGrid's body is recoloured into an overlay canvas that sits on top
    // of the original raster.  Debounced 200ms.

    function updateBodyThreshold(state, minVal, maxVal) {
        if (state._thresholdDebounceT) clearTimeout(state._thresholdDebounceT);
        state._thresholdDebounceT = setTimeout(function () {
            applyThreshold(state, minVal, maxVal);
        }, 200);
    }

    function applyThreshold(state, minVal, maxVal) {
        var grids = state.sceneGraph.data_grids || [];
        for (var gi = 0; gi < grids.length; gi++) {
            var grid = state.heatmap.dataGrids[grids[gi].id];
            if (!grid) continue;
            var spec = (grids[gi].annotations || {}).color_spec;
            if (!spec) continue;
            // Re-map spec breaks linearly to [minVal, maxVal]
            var origBreaks = spec.breaks;
            var origLow = origBreaks[0], origHigh = origBreaks[origBreaks.length - 1];
            var remapped = origBreaks.map(function (b) {
                var t = (b - origLow) / (origHigh - origLow);
                return minVal + t * (maxVal - minVal);
            });
            var newSpec = Object.assign({}, spec, { breaks: remapped });
            var ramp = buildColorRamp(newSpec);
            if (!ramp) continue;
            paintOverlay(state, grid, ramp);
        }
    }

    function paintOverlay(state, grid, ramp) {
        if (!state._bodyOverlays) state._bodyOverlays = {};
        var canvas = state._bodyOverlays[grid.id];
        if (!canvas) {
            canvas = document.createElement("canvas");
            canvas.className = "gridpy-heatmap-overlay";
            canvas.style.position = "absolute";
            canvas.style.pointerEvents = "none";
            state.container.appendChild(canvas);
            state._bodyOverlays[grid.id] = canvas;
        }
        var box = grid.box;
        canvas.width = Math.round(box.w);
        canvas.height = Math.round(box.h);
        canvas.style.left = box.x0 + "px";
        canvas.style.top = box.y0 + "px";
        canvas.style.width = box.w + "px";
        canvas.style.height = box.h + "px";

        var shape = grid.value_shape;
        var nr = shape[0], nc = shape[1];
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        var cw = canvas.width / nc, ch = canvas.height / nr;
        for (var r = 0; r < nr; r++) {
            for (var c = 0; c < nc; c++) {
                var v = _decodeValue(grid, r, c);
                ctx.fillStyle = ramp(v);
                ctx.fillRect(Math.floor(c * cw), Math.floor(r * ch),
                             Math.ceil(cw), Math.ceil(ch));
            }
        }
    }

    function bindHeatmapTooltip(state) {
        var container = state.container;
        container.addEventListener("mousemove", function (event) {
            if (state.heatmap.pinned) return;
            var rect = container.getBoundingClientRect();
            var mx = event.clientX - rect.left;
            var my = event.clientY - rect.top;
            var hit = hitTest(state, mx, my);
            if (hit) {
                gridpy.showTooltip(state, renderTooltipHtml(hit, state.heatmap.templates), mx, my);
                var row = hit.row, col = hit.col;
                if (row != null) highlightEntities(state, "row", row);
                else if (col != null) highlightEntities(state, "col", col);
            } else {
                gridpy.hideTooltip(state);
                clearHighlight(state);
            }
        });
        container.addEventListener("mouseleave", function () {
            if (state.heatmap.pinned) return;
            gridpy.hideTooltip(state);
            clearHighlight(state);
        });
    }

    function buildEntityRegions(sg) {
        // Collect bounding boxes for non-rect entity grobs so that hover
        // / click can target them.  Currently: dendrogram branches (stored
        // as ``segments`` grobs).  Easy to extend to other entities by
        // adding cases below.
        var regions = [];
        (function walk(n) {
            var d = n.data;
            if (d && d.entity && n.type === "segments" &&
                d.entity === "dend_branch") {
                var p = n.props || {};
                var xs = (p.x0 || []).concat(p.x1 || []);
                var ys = (p.y0 || []).concat(p.y1 || []);
                if (xs.length && ys.length) {
                    var x0 = Math.min.apply(null, xs);
                    var x1 = Math.max.apply(null, xs);
                    var y0 = Math.min.apply(null, ys);
                    var y1 = Math.max.apply(null, ys);
                    // Pad by 4px so thin line segments are still easy to hit.
                    regions.push({
                        bbox: [x0 - 4, y0 - 4, x1 + 4, y1 + 4],
                        data: d,
                        area: (x1 - x0) * (y1 - y0)
                    });
                }
            }
            (n.children || []).forEach(walk);
        })(sg.root);
        // Smallest-area-first so a nested sub-branch wins over its parent.
        regions.sort(function (a, b) { return a.area - b.area; });
        return regions;
    }

    function hitRegion(state, mx, my) {
        var regions = state.heatmap.entityRegions || [];
        for (var i = 0; i < regions.length; i++) {
            var r = regions[i];
            if (mx >= r.bbox[0] && mx <= r.bbox[2] &&
                my >= r.bbox[1] && my <= r.bbox[3]) {
                return r.data;
            }
        }
        return null;
    }

    function hitTest(state, mx, my) {
        // 1. entity regions (dendrogram branches etc.) — bounded by their
        //    bbox, checked before the DataGrid pixel lookup so clicks on the
        //    row/column dendrogram win over any adjacent body cell hits.
        var region = hitRegion(state, mx, my);
        if (region) return region;
        // 2. Try entity-tagged grob via existing quadtree.
        if (state.quadtree) {
            var nearest = state.quadtree.find(mx, my, 10);
            if (nearest && nearest.data && nearest.data.entity) return nearest.data;
        }
        // 3. DataGrid pixel lookup (body cells).
        for (var gid in state.heatmap.dataGrids) {
            var grid = state.heatmap.dataGrids[gid];
            var cell = lookupDataGridCell(grid, mx, my);
            if (cell) {
                return {
                    entity: "cell",
                    heatmap: grid.id,
                    row: cell.row_id,
                    col: cell.col_id,
                    tooltip_ref: "cell",
                    payload: {
                        row_name: cell.row_name,
                        col_name: cell.col_name,
                        value: cell.value
                    }
                };
            }
        }
        return null;
    }

    function bindHoverHighlight(state) { /* handled inside tooltip handler */ }

    function bindClick(state) {
        state.container.addEventListener("click", function (event) {
            var rect = state.container.getBoundingClientRect();
            var mx = event.clientX - rect.left;
            var my = event.clientY - rect.top;
            var hit = hitTest(state, mx, my);
            if (!hit) {
                // Blank click → unpin + close sub-panel
                state.heatmap.pinned = false;
                closeSubPanel(state);
                clearHighlight(state);
                gridpy.hideTooltip(state);
                gridpy.emit(state, "blank_click", {});
                return;
            }
            if (hit.entity === "dend_branch") {
                var leaves = (hit.payload || {}).leaves || [];
                var axis = (hit.payload || {}).axis || "row";
                gridpy.emit(state, "dend_click", {
                    heatmap: hit.heatmap, axis: axis,
                    leaves: leaves,
                    height: (hit.payload || {}).height,
                    n_leaves: leaves.length
                });
                if (leaves.length) openSubPanel(state, axis, leaves);
                return;
            }
            // Generic entity click
            var name = hit.entity ? hit.entity + "_click" : "entity_click";
            gridpy.emit(state, name, hit);
            // Toggle pin
            if (hit.entity === "cell" || hit.entity === "row_label" ||
                hit.entity === "col_label") {
                state.heatmap.pinned = !state.heatmap.pinned;
                if (state.heatmap.pinned) {
                    state.heatmap.pinnedAxis = hit.row != null ? "row" : "col";
                    state.heatmap.pinnedId = hit.row != null ? hit.row : hit.col;
                }
            }
        });
    }

    function bindSearch(state, sceneGraph) {
        // Ctrl+F opens a search box over row/col names.
        var grid = state.heatmap.primaryGrid;
        if (!grid) return;
        var box = null;

        document.addEventListener("keydown", function (event) {
            if ((event.ctrlKey || event.metaKey) && event.key === "f" &&
                state.container.matches(":hover")) {
                event.preventDefault();
                if (!box) {
                    box = document.createElement("div");
                    box.className = "gridpy-search";
                    box.innerHTML =
                        "<input type='text' placeholder='row / col name...' />" +
                        "<span class='gridpy-search-count'></span>" +
                        "<span class='gridpy-search-close'>&times;</span>";
                    state.container.appendChild(box);
                    var input = box.querySelector("input");
                    var count = box.querySelector(".gridpy-search-count");
                    var close = box.querySelector(".gridpy-search-close");
                    input.addEventListener("input", function () {
                        runSearch(state, grid, input.value, count);
                    });
                    close.addEventListener("click", function () {
                        box.style.display = "none";
                        clearHighlight(state);
                    });
                }
                box.style.display = "";
                box.querySelector("input").focus();
            } else if (event.key === "Escape" && box) {
                box.style.display = "none";
                clearHighlight(state);
            }
        });
    }

    function runSearch(state, grid, query, countEl) {
        clearHighlight(state);
        if (!query) { countEl.textContent = ""; return; }
        var q = query.toLowerCase();
        var matches = [];
        (grid.row_names || []).forEach(function (n, i) {
            if (String(n).toLowerCase().indexOf(q) >= 0) {
                matches.push({ axis: "row", id: grid.row_ids[i], name: n });
            }
        });
        (grid.col_names || []).forEach(function (n, i) {
            if (String(n).toLowerCase().indexOf(q) >= 0) {
                matches.push({ axis: "col", id: grid.col_ids[i], name: n });
            }
        });
        countEl.textContent = matches.length + " match" + (matches.length === 1 ? "" : "es");
        if (matches.length) {
            var m = matches[0];
            highlightEntities(state, m.axis, m.id);
            gridpy.emit(state, "search_match", { query: query, matches: matches });
        }
    }

    // ---- Register --------------------------------------------------------

    gridpy.registerModule("gridpy-heatmap", {
        attach: attach,
        version: "0.1.0"
    });

})();
