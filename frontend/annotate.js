/* AI Platform — label annotation tools.
 *
 * Geometry lives in raster pixel coordinates (exact for training data). The map works in
 * lon/lat, so pixel ↔ lon/lat uses cubic polynomials fitted to a grid of control points that
 * the server computes with PROJ for the raster's CRS (residuals are far below a pixel).
 *
 * Relies on globals from app.js: map, mapReady, state, api, $, esc, setStatus, PALETTE.
 */
"use strict";

(() => {
  const TOOLS = {
    detection: ["select", "bbox", "point"],
    instance: ["select", "polygon", "freehand", "bbox"],
    semantic: ["select", "brush", "eraser", "polygon", "freehand"],
  };
  const TOOL_INFO = {
    select: { key: "V", label: "Select", icon: "➤", help: "Click to select · drag to move · drag handles to resize or edit vertices · Alt+click a vertex to remove it" },
    bbox: { key: "B", label: "Box", icon: "▭", help: "Drag to draw a bounding box" },
    polygon: { key: "P", label: "Polygon", icon: "⬠", help: "Click to add vertices · click the first vertex, double-click or Enter to finish · Backspace removes the last · Esc cancels" },
    freehand: { key: "F", label: "Freehand", icon: "✎", help: "Hold and drag to draw an outline; release to close it" },
    point: { key: "O", label: "Point", icon: "•", help: "Click to place a point on each object" },
    brush: { key: "R", label: "Brush", icon: "🖌", help: "Paint pixels with the active class · [ and ] change the brush size" },
    eraser: { key: "E", label: "Eraser", icon: "⌫", help: "Erase painted pixels back to unlabelled" },
  };
  const TASK_HINT = {
    detection: "Boxes and points for object detection. Export as YOLO, YOLO OBB or COCO.",
    instance: "Exact outlines for instance segmentation. Export as YOLO-seg or COCO polygons.",
    semantic: "Paint every pixel by class: brush, eraser and filled polygons. Export class-index masks.",
  };
  const PRESETS = [["Tree", "#3fb68b"], ["Building", "#e5534b"], ["Water", "#4c9aff"], ["Bare Soil", "#c9a36b"]];
  const POLY_TYPES = ["bbox", "polygon", "freehand"];
  const MAX_HISTORY = 200;

  const A = {
    on: false, raster: null, doc: null, geo: null, tool: "select", cls: null, sel: null,
    undo: [], redo: [], clip: null, draft: null, drag: null, hidden: null,
    saveTimer: null, saving: false, saveAgain: false, pending: new Set(), brush: 24, cursorPx: null, layersReady: false,
  };

  const clone = (o) => JSON.parse(JSON.stringify(o));
  const nextId = () => A.doc.items.reduce((m, i) => Math.max(m, i.id), 0) + 1;
  const classOf = (id) => A.doc.classes.find((c) => c.id === id);
  const typing = (e) => /INPUT|TEXTAREA|SELECT/.test(e.target.tagName);

  // ---------------------------------------------------------------- georeferencing

  function polyFit(src, dst) {
    // least-squares cubic fit dst = f(src) on normalised inputs; returns a function
    const n = src.length;
    const mu = [0, 1].map((k) => src.reduce((a, p) => a + p[k], 0) / n);
    const sc = [0, 1].map((k) => Math.max(...src.map((p) => Math.abs(p[k] - mu[k]))) || 1);
    const dmu = [0, 1].map((k) => dst.reduce((a, p) => a + p[k], 0) / n);
    const terms = (u, v) => [1, u, v, u * u, u * v, v * v, u * u * u, u * u * v, u * v * v, v * v * v];
    const T = src.map((p) => terms((p[0] - mu[0]) / sc[0], (p[1] - mu[1]) / sc[1]));
    const m = T[0].length;
    const coef = [0, 1].map((k) => {
      const M = Array.from({ length: m }, () => new Array(m + 1).fill(0));
      for (let r = 0; r < n; r++) {
        const y = dst[r][k] - dmu[k];
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < m; j++) M[i][j] += T[r][i] * T[r][j];
          M[i][m] += T[r][i] * y;
        }
      }
      for (let c = 0; c < m; c++) {  // Gauss-Jordan with partial pivoting
        let p = c;
        for (let r = c + 1; r < m; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
        [M[c], M[p]] = [M[p], M[c]];
        for (let r = 0; r < m; r++) {
          if (r === c || !M[c][c]) continue;
          const f = M[r][c] / M[c][c];
          for (let j = c; j <= m; j++) M[r][j] -= f * M[c][j];
        }
      }
      return M.map((row, i) => (row[i] ? row[m] / row[i] : 0));
    });
    return (p) => {
      const t = terms((p[0] - mu[0]) / sc[0], (p[1] - mu[1]) / sc[1]);
      return [0, 1].map((k) => dmu[k] + t.reduce((a, v, i) => a + v * coef[k][i], 0));
    };
  }

  async function loadGeoref(r) {
    const g = await api(`/api/rasters/${r.id}/georef`);
    const px = g.points.map((p) => [p[0], p[1]]);
    const ll = g.points.map((p) => [p[2], p[3]]);
    const toLL = polyFit(px, ll);
    const toPx = polyFit(ll, px);
    const err = Math.max(...px.map((p) => { const q = toPx(toLL(p)); return Math.hypot(q[0] - p[0], q[1] - p[1]); }));
    const lat = (r.bounds_wgs84[1] + r.bounds_wgs84[3]) / 2;
    // screen px per raster px at zoom 0 (MapLibre's 512 px world)
    const k0 = (g.res_m * 512) / (40075016.686 * Math.cos((lat * Math.PI) / 180));
    console.info(`[annotate] georeference fit: max round-trip error ${err.toExponential(2)} px`);
    return { toLL, toPx, width: g.width, height: g.height, k0, resM: g.res_m };
  }

  const toPx = (lngLat) => A.geo.toPx([lngLat.lng, lngLat.lat]);
  const toLL = (p) => A.geo.toLL(p);
  const screenToPx = (pt) => toPx(map.unproject(pt));
  const pxPerScreen = () => A.geo ? 1 / (A.geo.k0 * Math.pow(2, map.getZoom())) : 1;
  const clampPx = ([x, y]) => [Math.min(Math.max(x, 0), A.geo.width), Math.min(Math.max(y, 0), A.geo.height)];

  // ---------------------------------------------------------------- rendering

  function ring(it) {
    if (it.type === "bbox") {
      const [[x0, y0], [x1, y1]] = it.pts;
      return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    }
    return it.pts;
  }

  function feature(it, extra = {}) {
    const c = classOf(it.cls);
    const semantic = A.doc.task === "semantic" && it.type !== "point";
    const props = { id: it.id, cls: it.cls, type: it.type, color: it.erase ? "#ffffff" : c ? c.color : "#ffffff",
      src: it.source, sem: semantic, pend: A.pending.has(it.id), erase: !!it.erase, w: it.width || 0, ...extra };
    let geometry;
    if (it.type === "point") geometry = { type: "Point", coordinates: toLL(it.pts[0]) };
    else if (it.type === "stroke") {
      const pts = it.pts.length > 1 ? it.pts : [it.pts[0], [it.pts[0][0] + 0.01, it.pts[0][1]]];
      geometry = { type: "LineString", coordinates: pts.map(toLL) };
    } else {
      const r = ring(it).map(toLL);
      geometry = { type: "Polygon", coordinates: [[...r, r[0]]] };
    }
    return { type: "Feature", id: it.id, properties: props, geometry };
  }

  const fc = (features) => ({ type: "FeatureCollection", features });

  function render() {
    if (!A.layersReady) return;
    map.getSource("ann").setData(fc(A.doc.items.map((it) => feature(it))));
    renderHandles();
    renderClasses();
  }

  function renderHandles() {
    const it = A.sel != null && A.doc.items.find((i) => i.id === A.sel);
    const hs = [];
    if (it && A.tool === "select") {
      if (it.type === "bbox") ring(it).forEach((p, i) => hs.push(handle(p, "corner", i)));
      else if (it.type === "polygon" || it.type === "freehand") {
        if (it.pts.length <= 400) {
          it.pts.forEach((p, i) => {
            hs.push(handle(p, "vertex", i));
            const q = it.pts[(i + 1) % it.pts.length];
            hs.push(handle([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], "mid", i));
          });
        }
      }
    }
    map.getSource("ann-handles").setData(fc(hs));
    map.setFilter("ann-sel", ["==", ["get", "id"], A.sel ?? -1]);
    map.setFilter("ann-sel-point", ["all", ["==", ["get", "id"], A.sel ?? -1], ["==", ["get", "type"], "point"]]);
  }

  function handle(p, kind, idx) {
    return { type: "Feature", properties: { kind, idx }, geometry: { type: "Point", coordinates: toLL(p) } };
  }

  function renderDraft() {
    const feats = [];
    const d = A.draft;
    if (d) {
      if (d.item) feats.push(feature(d.item, { draft: true }));
      if (d.type === "polygon" && d.pts.length) {
        const line = [...d.pts, d.cursor || d.pts[d.pts.length - 1]];
        feats.push({ type: "Feature", properties: { kind: "rubber", color: d.color }, geometry: { type: "LineString", coordinates: line.map(toLL) } });
        d.pts.forEach((p, i) => feats.push({ type: "Feature", properties: { kind: i === 0 ? "first" : "vtx", color: d.color }, geometry: { type: "Point", coordinates: toLL(p) } }));
      }
    }
    if (A.cursorPx && (A.tool === "brush" || A.tool === "eraser")) {
      feats.push({ type: "Feature", properties: { kind: "cursor" }, geometry: { type: "Point", coordinates: toLL(A.cursorPx) } });
    }
    map.getSource("ann-draft").setData(fc(feats));
  }

  function strokeWidth(prop = ["get", "w"]) {
    // screen width of a brush stroke whose diameter is `w` raster pixels, exact at every zoom
    const k = A.geo ? A.geo.k0 : 0;
    return ["interpolate", ["exponential", 2], ["zoom"], 0, ["*", prop, k], 24, ["*", prop, k * 2 ** 24]];
  }

  function setupLayers() {
    if (A.layersReady) return;
    const empty = fc([]);
    map.addSource("ann", { type: "geojson", data: empty });
    map.addSource("ann-handles", { type: "geojson", data: empty });
    map.addSource("ann-draft", { type: "geojson", data: empty });
    const notHidden = ["!=", ["get", "id"], -1];
    const showSem = ["any", ["!", ["get", "sem"]], ["get", "pend"]];
    map.addLayer({ id: "ann-fill", type: "fill", source: "ann", filter: ["all", ["==", ["geometry-type"], "Polygon"], notHidden],
      paint: { "fill-color": ["get", "color"], "fill-opacity": ["case", ["get", "sem"], ["case", ["get", "pend"], 0.45, 0.0], 0.18] } });
    map.addLayer({ id: "ann-line", type: "line", source: "ann", filter: ["all", ["==", ["geometry-type"], "Polygon"], notHidden],
      paint: { "line-color": ["get", "color"], "line-width": ["case", ["get", "sem"], 1, 1.8],
        "line-opacity": ["case", ["get", "sem"], 0.55, ["==", ["get", "src"], "auto"], 0.75, 1] } });
    map.addLayer({ id: "ann-stroke", type: "line", source: "ann", filter: ["all", ["==", ["geometry-type"], "LineString"], notHidden, showSem],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": ["get", "color"], "line-width": 1, "line-opacity": 0.45 } });
    map.addLayer({ id: "ann-stroke-axis", type: "line", source: "ann", filter: ["all", ["==", ["geometry-type"], "LineString"], notHidden],
      paint: { "line-color": ["get", "color"], "line-width": 1, "line-opacity": 0.5, "line-dasharray": [2, 2] } });
    map.addLayer({ id: "ann-point", type: "circle", source: "ann", filter: ["all", ["==", ["geometry-type"], "Point"], notHidden],
      paint: { "circle-radius": 5, "circle-color": ["get", "color"], "circle-stroke-color": "#ffffff", "circle-stroke-width": 1.5 } });
    map.addLayer({ id: "ann-sel", type: "line", source: "ann", filter: ["==", ["get", "id"], -1],
      paint: { "line-color": "#ffffff", "line-width": 2.5, "line-dasharray": [2, 1] } });
    map.addLayer({ id: "ann-sel-point", type: "circle", source: "ann", filter: ["==", ["get", "id"], -1],
      paint: { "circle-radius": 9, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": "#ffffff", "circle-stroke-width": 2 } });
    map.addLayer({ id: "ann-handle", type: "circle", source: "ann-handles",
      paint: { "circle-radius": ["case", ["==", ["get", "kind"], "mid"], 3.5, 5], "circle-color": ["case", ["==", ["get", "kind"], "mid"], "rgba(255,255,255,0.5)", "#ffffff"],
        "circle-stroke-color": "#0f1419", "circle-stroke-width": 1.5 } });
    // draft (in-progress) geometry
    map.addLayer({ id: "ann-draft-fill", type: "fill", source: "ann-draft", filter: ["==", ["geometry-type"], "Polygon"],
      paint: { "fill-color": ["get", "color"], "fill-opacity": 0.3 } });
    map.addLayer({ id: "ann-draft-line", type: "line", source: "ann-draft", filter: ["any", ["==", ["geometry-type"], "Polygon"], ["==", ["get", "kind"], "rubber"]],
      paint: { "line-color": ["coalesce", ["get", "color"], "#ffffff"], "line-width": 2, "line-dasharray": [2, 1] } });
    map.addLayer({ id: "ann-draft-stroke", type: "line", source: "ann-draft", filter: ["all", ["==", ["geometry-type"], "LineString"], ["has", "w"]],
      layout: { "line-cap": "round", "line-join": "round" }, paint: { "line-color": ["get", "color"], "line-width": 1, "line-opacity": 0.55 } });
    map.addLayer({ id: "ann-draft-pt", type: "circle", source: "ann-draft", filter: ["all", ["==", ["geometry-type"], "Point"], ["!=", ["get", "kind"], "cursor"]],
      paint: { "circle-radius": ["case", ["==", ["get", "kind"], "first"], 6, 4], "circle-color": ["coalesce", ["get", "color"], "#ffffff"],
        "circle-stroke-color": "#ffffff", "circle-stroke-width": 1.5 } });
    map.addLayer({ id: "ann-cursor", type: "circle", source: "ann-draft", filter: ["==", ["get", "kind"], "cursor"],
      paint: { "circle-radius": 10, "circle-color": "rgba(255,255,255,0.12)", "circle-stroke-color": "#ffffff", "circle-stroke-width": 1.5 } });
    A.layersReady = true;
  }

  function updateGeoDependentPaint() {
    // stroke widths depend on the raster's ground resolution
    map.setPaintProperty("ann-stroke", "line-width", strokeWidth());
    map.setPaintProperty("ann-draft-stroke", "line-width", strokeWidth());
    updateCursorSize();
  }

  function updateCursorSize() {
    if (!A.layersReady || !A.geo) return;
    map.setPaintProperty("ann-cursor", "circle-radius", Math.max(2, (A.brush / 2) / pxPerScreen()));
  }

  function applyHidden() {
    const f = ["!=", ["get", "id"], A.hidden ?? -1];
    const base = {
      "ann-fill": ["==", ["geometry-type"], "Polygon"], "ann-line": ["==", ["geometry-type"], "Polygon"],
      "ann-stroke-axis": ["==", ["geometry-type"], "LineString"], "ann-point": ["==", ["geometry-type"], "Point"],
    };
    for (const [id, g] of Object.entries(base)) map.setFilter(id, ["all", g, f]);
    map.setFilter("ann-stroke", ["all", ["==", ["geometry-type"], "LineString"], f, ["any", ["!", ["get", "sem"]], ["get", "pend"]]]);
  }

  function setLayersVisible(v) {
    if (!A.layersReady) return;
    for (const l of map.getStyle().layers) {
      if (l.id.startsWith("ann-")) map.setLayoutProperty(l.id, "visibility", v ? "visible" : "none");
    }
  }

  // semantic mask preview tiles (rendered by the server from the saved annotations)
  function refreshMask() {
    const want = A.on && A.doc && A.doc.task === "semantic";
    if (!want) {
      if (map.getLayer("ann-mask")) map.removeLayer("ann-mask");
      if (map.getSource("ann-mask")) map.removeSource("ann-mask");
      return;
    }
    const url = `${location.origin}/api/rasters/${A.raster.id}/annotations/mask/{z}/{x}/{y}.png?v=${A.doc.version}`;
    if (map.getSource("ann-mask")) map.getSource("ann-mask").setTiles([url]);
    else {
      map.addSource("ann-mask", { type: "raster", tiles: [url], tileSize: 256, bounds: A.raster.bounds_wgs84, maxzoom: Math.min(24, A.raster.maxzoom + 2) });
      map.addLayer({ id: "ann-mask", type: "raster", source: "ann-mask", paint: { "raster-opacity": 0.65, "raster-fade-duration": 0, "raster-resampling": "nearest" } }, "ann-fill");
    }
    const ids = [...A.pending];
    setTimeout(() => { ids.forEach((id) => A.pending.delete(id)); if (A.on) render(); }, 900);
  }

  // ---------------------------------------------------------------- history & persistence

  function commit(changes) {
    if (!changes.length) return;
    for (const c of changes) {
      c.at = A.doc.items.findIndex((i) => i.id === c.id);
      if (c.after && A.doc.task === "semantic") A.pending.add(c.id);
    }
    apply(changes, "after");
    A.undo.push(changes);
    if (A.undo.length > MAX_HISTORY) A.undo.shift();
    A.redo = [];
    changed();
  }

  function apply(changes, key) {
    const list = key === "before" ? [...changes].reverse() : changes;
    for (const c of list) {
      const idx = A.doc.items.findIndex((i) => i.id === c.id);
      const v = c[key];
      if (!v) { if (idx >= 0) A.doc.items.splice(idx, 1); continue; }
      if (idx >= 0) A.doc.items[idx] = clone(v);
      else if (c.at >= 0 && key === "before") A.doc.items.splice(Math.min(c.at, A.doc.items.length), 0, clone(v));
      else A.doc.items.push(clone(v));
    }
  }

  function undo() {
    const ch = A.undo.pop();
    if (!ch) return;
    apply(ch, "before");
    A.redo.push(ch);
    if (A.sel != null && !A.doc.items.some((i) => i.id === A.sel)) A.sel = null;
    changed();
  }

  function redo() {
    const ch = A.redo.pop();
    if (!ch) return;
    apply(ch, "after");
    A.undo.push(ch);
    changed();
  }

  function changed() {
    render();
    setSaveState("dirty");
    clearTimeout(A.saveTimer);
    A.saveTimer = setTimeout(save, 700);
  }

  function setSaveState(s, msg) {
    const el = $("saveState");
    el.className = `save-state ${s}`;
    el.textContent = msg || { dirty: "Unsaved…", saving: "Saving…", saved: "Saved", error: "Save failed" }[s];
  }

  async function save() {
    if (!A.doc || !A.raster) return;
    if (A.saving) { A.saveAgain = true; return; }
    A.saving = true;
    setSaveState("saving");
    const doc = A.doc;  // the raster may change while the request is in flight
    try {
      const res = await api(`/api/rasters/${A.raster.id}/annotations`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(doc) });
      doc.version = res.version;
      if (doc !== A.doc) return;
      setSaveState("saved", `Saved · v${res.version}`);
      refreshMask();
    } catch (err) {
      setSaveState("error", err.message.slice(0, 60));
      if (/changed elsewhere/.test(err.message) && confirm("Annotations were changed elsewhere. Reload them? Unsaved edits are lost.")) {
        await loadDoc();
      }
    } finally {
      A.saving = false;
      if (A.saveAgain) { A.saveAgain = false; save(); }
    }
  }

  async function loadDoc() {
    A.doc = await api(`/api/rasters/${A.raster.id}/annotations`);
    A.undo = []; A.redo = []; A.sel = null; A.pending.clear();
    if (!A.doc.classes.length) A.doc.classes.push({ id: 1, name: "object", color: PALETTE[0] });
    if (!classOf(A.cls)) A.cls = A.doc.classes[0].id;
    setTask(A.doc.task, false);
    render();
    refreshMask();
    setSaveState("saved", `Saved · v${A.doc.version}`);
  }

  // ---------------------------------------------------------------- task, tools, classes UI

  function setTask(task, persist = true) {
    if (persist && A.doc.task !== task) { A.doc.task = task; changed(); }
    document.querySelectorAll("#taskSwitch button").forEach((b) => b.classList.toggle("active", b.dataset.task === task));
    $("taskHint").textContent = TASK_HINT[task];
    if (!TOOLS[task].includes(A.tool)) A.tool = TOOLS[task][1];
    renderToolbar();
    setTool(A.tool);
    render();
    refreshMask();
  }

  function renderToolbar() {
    const box = $("toolButtons");
    box.innerHTML = "";
    for (const t of TOOLS[A.doc.task]) {
      const b = document.createElement("button");
      b.dataset.tool = t;
      b.title = `${TOOL_INFO[t].label} (${TOOL_INFO[t].key})`;
      b.innerHTML = `${TOOL_INFO[t].icon}<kbd>${TOOL_INFO[t].key}</kbd>`;
      b.addEventListener("click", () => setTool(t));
      box.appendChild(b);
    }
    $("brushRow").classList.toggle("hidden", A.doc.task !== "semantic");
  }

  function setTool(t) {
    if (!TOOLS[A.doc.task].includes(t)) return;
    cancelDraft();
    A.tool = t;
    document.querySelectorAll("#toolButtons button").forEach((b) => b.classList.toggle("active", b.dataset.tool === t));
    $("annHelp").textContent = TOOL_INFO[t].help;
    const wrap = document.querySelector(".map-wrap");
    wrap.classList.toggle("drawing", t !== "select" && t !== "brush" && t !== "eraser");
    wrap.classList.toggle("brushing", t === "brush" || t === "eraser");
    if (t !== "select") A.sel = null;
    if (A.layersReady) { renderHandles(); renderDraft(); }
  }

  function renderClasses() {
    const counts = {};
    for (const it of A.doc.items) if (!it.erase) counts[it.cls] = (counts[it.cls] || 0) + 1;
    const ul = $("annClasses");
    ul.innerHTML = "";
    A.doc.classes.forEach((c, i) => {
      const li = document.createElement("li");
      li.classList.toggle("active", c.id === A.cls);
      li.innerHTML = `<span class="key">${i < 9 ? i + 1 : ""}</span><input type="color" value="${esc(c.color)}" title="Colour">
        <span class="cname" title="Double-click to rename">${esc(c.name)}</span><span class="count">${counts[c.id] || 0}</span>
        <button class="del" title="Delete class">✕</button>`;
      li.addEventListener("click", (e) => { if (!e.target.closest("input,button")) pickClass(c.id); });
      li.querySelector(".cname").addEventListener("dblclick", () => {
        const name = prompt("Class name", c.name);
        if (name && name.trim()) { c.name = name.trim().slice(0, 80); changed(); }
      });
      li.querySelector("input").addEventListener("change", (e) => { c.color = e.target.value; changed(); });
      li.querySelector(".del").addEventListener("click", () => deleteClass(c));
      ul.appendChild(li);
    });
    const total = A.doc.items.length;
    setStatus($("annStatus"), `${total} annotation${total === 1 ? "" : "s"} · ${A.doc.classes.length} classes`);
  }

  function pickClass(id) {
    A.cls = id;
    const it = A.sel != null && A.doc.items.find((i) => i.id === A.sel);
    if (it && it.cls !== id) commit([{ id: it.id, before: clone(it), after: { ...clone(it), cls: id } }]);
    renderClasses();
  }

  function addClass(name, color) {
    name = name.trim().slice(0, 80);
    if (!name) return;
    const existing = A.doc.classes.find((c) => c.name.toLowerCase() === name.toLowerCase());
    if (existing) { A.cls = existing.id; renderClasses(); return; }
    const id = A.doc.classes.reduce((m, c) => Math.max(m, c.id), 0) + 1;
    A.doc.classes.push({ id, name, color: color || PALETTE[(id - 1) % PALETTE.length] });
    A.cls = id;
    changed();
  }

  function deleteClass(c) {
    if (A.doc.classes.length <= 1) return alert("Keep at least one class.");
    const items = A.doc.items.filter((i) => i.cls === c.id && !i.erase);
    if (!confirm(`Delete class "${c.name}"${items.length ? ` and its ${items.length} annotation(s)` : ""}?`)) return;
    if (items.length) commit(items.map((i) => ({ id: i.id, before: clone(i), after: null })));
    A.doc.classes = A.doc.classes.filter((x) => x.id !== c.id);
    if (A.cls === c.id) A.cls = A.doc.classes[0].id;
    changed();
  }

  // ---------------------------------------------------------------- drawing & editing

  function cancelDraft() {
    A.draft = null;
    if (A.drag) endDrag(false);
    if (A.layersReady) renderDraft();
  }

  function color() { return (classOf(A.cls) || {}).color || "#ffffff"; }

  function addItem(item) {
    item.id = nextId();
    item.cls = item.cls ?? A.cls;
    item.source = item.source || "manual";
    commit([{ id: item.id, before: null, after: item }]);
    return item;
  }

  function rdp(pts, eps) {
    // Ramer–Douglas–Peucker simplification
    if (pts.length < 3) return pts;
    const [a, b] = [pts[0], pts[pts.length - 1]];
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1e-9;
    let idx = 0, dmax = 0;
    for (let i = 1; i < pts.length - 1; i++) {
      const d = Math.abs(dy * pts[i][0] - dx * pts[i][1] + b[0] * a[1] - b[1] * a[0]) / len;
      if (d > dmax) { dmax = d; idx = i; }
    }
    if (dmax <= eps) return [a, b];
    return [...rdp(pts.slice(0, idx + 1), eps).slice(0, -1), ...rdp(pts.slice(idx), eps)];
  }

  function finishPolygon() {
    const d = A.draft;
    if (!d || d.type !== "polygon") return;
    const tol = 2 * pxPerScreen();
    const pts = d.pts.filter((p, i, arr) => i === 0 || Math.hypot(p[0] - arr[i - 1][0], p[1] - arr[i - 1][1]) > tol);
    A.draft = null;
    renderDraft();
    if (pts.length >= 3) addItem({ type: "polygon", pts });
  }

  function hit(point) {
    const b = [[point.x - 5, point.y - 5], [point.x + 5, point.y + 5]];
    const h = map.queryRenderedFeatures(b, { layers: ["ann-handle"] });
    if (h.length) return { handle: h[0].properties };
    const layers = ["ann-point", "ann-stroke-axis", "ann-line", "ann-fill"].filter((l) => map.getLayer(l));
    const f = map.queryRenderedFeatures(b, { layers });
    if (f.length) return { id: f[0].properties.id };
    // semantic fills are transparent: fall back to point-in-polygon in pixel space
    if (A.doc.task === "semantic") {
      const p = screenToPx(point);
      for (let i = A.doc.items.length - 1; i >= 0; i--) {
        const it = A.doc.items[i];
        if (POLY_TYPES.includes(it.type) && inside(p, ring(it))) return { id: it.id };
      }
    }
    return null;
  }

  function inside([x, y], poly) {
    let c = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
    }
    return c;
  }

  function startDrag(kind, it, p, extra = {}) {
    A.drag = { kind, id: it.id, start: p, before: clone(it), item: clone(it), moved: false, ...extra };
    A.hidden = it.id;
    applyHidden();
    map.dragPan.disable();
  }

  function endDrag(commitIt = true) {
    const d = A.drag;
    A.drag = null;
    A.hidden = null;
    map.dragPan.enable();
    A.draft = null;
    if (A.layersReady) { applyHidden(); renderDraft(); }
    if (commitIt && d && d.moved) commit([{ id: d.id, before: d.before, after: d.item }]);
    else if (A.layersReady) render();
  }

  function dragTo(p) {
    const d = A.drag;
    const it = d.item;
    d.moved = true;
    if (d.kind === "move") {
      const dx = p[0] - d.start[0], dy = p[1] - d.start[1];
      it.pts = d.before.pts.map(([x, y]) => [x + dx, y + dy]);
    } else if (d.kind === "corner") {
      const [[x0, y0], [x1, y1]] = d.before.pts;
      const c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
      const opp = c[(d.idx + 2) % 4];
      it.pts = [[Math.min(opp[0], p[0]), Math.min(opp[1], p[1])], [Math.max(opp[0], p[0]), Math.max(opp[1], p[1])]];
    } else if (d.kind === "vertex") {
      it.pts[d.idx] = p;
    }
    A.draft = { item: it };
    renderDraft();
  }

  function onMouseDown(e) {
    if (!A.on || e.originalEvent.button !== 0) return;
    const p = clampPx(toPx(e.lngLat));
    if (A.tool === "select") {
      const h = hit(e.point);
      const it = h && h.id != null ? A.doc.items.find((i) => i.id === h.id) : h && h.handle ? A.doc.items.find((i) => i.id === A.sel) : null;
      if (h && h.handle && it) {
        const { kind, idx } = h.handle;
        if (kind === "vertex" && e.originalEvent.altKey) {
          if (it.pts.length > 3) commit([{ id: it.id, before: clone(it), after: { ...clone(it), pts: it.pts.filter((_, i) => i !== idx) } }]);
          return;
        }
        if (kind === "mid") {
          const copy = clone(it);
          copy.pts.splice(idx + 1, 0, p);
          startDrag("vertex", it, p, { idx: idx + 1 });
          A.drag.item = copy;
          A.drag.moved = true;
        } else startDrag(kind === "corner" ? "corner" : "vertex", it, p, { idx });
        e.preventDefault();
        return;
      }
      if (it) {
        A.sel = it.id;
        renderHandles();
        renderClasses();
        startDrag("move", it, p);
        e.preventDefault();
        return;
      }
      A.sel = null;
      renderHandles();
      return;
    }
    if (A.tool === "bbox") {
      map.dragPan.disable();
      A.draft = { type: "bbox", start: p, item: { id: -1, cls: A.cls, type: "bbox", pts: [p, p] } };
    } else if (A.tool === "freehand") {
      map.dragPan.disable();
      A.draft = { type: "freehand", pts: [p], item: null, color: color() };
    } else if (A.tool === "brush" || A.tool === "eraser") {
      map.dragPan.disable();
      A.draft = { type: "stroke", pts: [p], item: { id: -1, cls: A.cls, type: "stroke", pts: [p], width: A.brush, erase: A.tool === "eraser" } };
      renderDraft();
    }
  }

  function onMouseMove(e) {
    if (!A.on) return;
    const p = clampPx(toPx(e.lngLat));
    if (A.tool === "brush" || A.tool === "eraser") { A.cursorPx = p; if (!A.draft) renderDraft(); }
    if (A.drag) { dragTo(p); return; }
    const d = A.draft;
    if (!d) {
      if (A.tool === "select") {
        const h = hit(e.point);
        map.getCanvas().style.cursor = h ? (h.handle ? "grab" : "move") : "";
      }
      return;
    }
    const step = 2 * pxPerScreen();
    if (d.type === "bbox") d.item.pts = [d.start, p];
    else if (d.type === "polygon") d.cursor = p;
    else if (d.type === "freehand" || d.type === "stroke") {
      const last = d.pts[d.pts.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) < step) return;
      d.pts.push(p);
      d.item = d.type === "stroke" ? { ...d.item, pts: d.pts } :
        { id: -1, cls: A.cls, type: d.pts.length >= 3 ? "polygon" : "stroke", pts: d.pts, width: 1 };
    }
    renderDraft();
  }

  function onMouseUp(e) {
    if (!A.on) return;
    if (A.drag) { endDrag(true); return; }
    const d = A.draft;
    if (!d) return;
    map.dragPan.enable();
    const minSize = 3 * pxPerScreen();
    if (d.type === "bbox") {
      const [[x0, y0], [x1, y1]] = [d.start, clampPx(toPx(e.lngLat))];
      A.draft = null;
      renderDraft();
      if (Math.abs(x1 - x0) >= minSize && Math.abs(y1 - y0) >= minSize) addItem({ type: "bbox", pts: [[Math.min(x0, x1), Math.min(y0, y1)], [Math.max(x0, x1), Math.max(y0, y1)]] });
    } else if (d.type === "freehand") {
      const pts = rdp(d.pts, 0.7 * pxPerScreen());
      A.draft = null;
      renderDraft();
      if (pts.length >= 3) addItem({ type: "freehand", pts });
    } else if (d.type === "stroke") {
      const pts = d.pts.length > 2 ? rdp(d.pts, Math.min(A.brush * 0.1, 0.7 * pxPerScreen())) : d.pts;
      A.draft = null;
      renderDraft();
      addItem({ type: "stroke", pts, width: A.brush, erase: A.tool === "eraser" });
    }
  }

  function onClick(e) {
    if (!A.on) return;
    const p = clampPx(toPx(e.lngLat));
    if (A.tool === "point") addItem({ type: "point", pts: [p] });
    else if (A.tool === "polygon") {
      if (!A.draft) A.draft = { type: "polygon", pts: [], color: color() };
      const d = A.draft;
      if (d.pts.length >= 3) {
        const first = map.project(toLL(d.pts[0]));
        if (Math.hypot(first.x - e.point.x, first.y - e.point.y) < 9) return finishPolygon();
      }
      d.pts.push(p);
      d.item = d.pts.length >= 3 ? { id: -1, cls: A.cls, type: "polygon", pts: d.pts } : null;
      renderDraft();
    }
  }

  function onDblClick(e) {
    if (!A.on) return;
    e.preventDefault();
    if (A.tool === "polygon") finishPolygon();
  }

  // ---------------------------------------------------------------- clipboard & actions

  function selected() { return A.sel != null ? A.doc.items.find((i) => i.id === A.sel) : null; }

  function offsetCopy(it, k = 12) {
    const off = k * pxPerScreen();
    const copy = clone(it);
    copy.pts = copy.pts.map(([x, y]) => clampPx([x + off, y + off]));
    copy.source = "manual";
    return copy;
  }

  const actions = {
    undo, redo,
    copy() { const it = selected(); if (it) A.clip = clone(it); },
    paste() { if (!A.clip) return; const it = addItem(offsetCopy(A.clip)); A.clip = clone(it); A.sel = it.id; setTool("select"); render(); },
    duplicate() { const it = selected(); if (!it) return; const c = addItem(offsetCopy(it)); A.sel = c.id; render(); },
    delete() { const it = selected(); if (!it) return; A.sel = null; commit([{ id: it.id, before: clone(it), after: null }]); },
  };

  document.querySelectorAll("#annToolbar [data-act]").forEach((b) => b.addEventListener("click", () => A.on && actions[b.dataset.act]()));

  document.addEventListener("keydown", (e) => {
    if (!A.on || typing(e)) return;
    const k = e.key.toLowerCase();
    const mod = e.ctrlKey || e.metaKey;
    if (mod && k === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (mod && k === "y") { e.preventDefault(); redo(); return; }
    if (mod && k === "c") { actions.copy(); return; }
    if (mod && k === "v") { e.preventDefault(); actions.paste(); return; }
    if (mod && k === "d") { e.preventDefault(); actions.duplicate(); return; }
    if (mod && k === "s") { e.preventDefault(); clearTimeout(A.saveTimer); save(); return; }
    if (mod) return;
    if (k === "escape") { if (A.draft) cancelDraft(); else { A.sel = null; renderHandles(); } return; }
    if (k === "enter") { finishPolygon(); return; }
    if (k === "delete" || k === "backspace") {
      if (A.draft && A.draft.type === "polygon") {
        A.draft.pts.pop();
        if (!A.draft.pts.length) A.draft = null;
        renderDraft();
      } else actions.delete();
      e.preventDefault();
      return;
    }
    if (k === "[" || k === "]") { setBrush(A.brush * (k === "]" ? 1.25 : 0.8)); return; }
    if (/^[1-9]$/.test(k)) { const c = A.doc.classes[+k - 1]; if (c) pickClass(c.id); return; }
    const tool = Object.keys(TOOL_INFO).find((t) => TOOL_INFO[t].key.toLowerCase() === k);
    if (tool) setTool(tool);
  });

  function setBrush(v) {
    A.brush = Math.round(Math.min(200, Math.max(2, v)));
    $("brushSize").value = A.brush;
    $("brushOut").textContent = `${A.brush}px`;
    updateCursorSize();
  }

  $("brushSize").addEventListener("input", (e) => setBrush(+e.target.value));
  document.querySelectorAll("#taskSwitch button").forEach((b) => b.addEventListener("click", () => A.doc && setTask(b.dataset.task)));
  $("classForm").addEventListener("submit", (e) => { e.preventDefault(); addClass($("className").value, $("classColor").value); $("className").value = ""; });
  $("presetClasses").addEventListener("click", () => {
    const keep = A.cls;
    for (const [n, c] of PRESETS) addClass(n, c);
    A.cls = keep;
    renderClasses();
  });

  // ---------------------------------------------------------------- auto annotation

  for (const [id, fmt] of [["autoConf", (v) => v.toFixed(2)], ["autoScale", (v) => `${v.toFixed(2).replace(/0$/, "")}×`]]) {
    $(id).addEventListener("input", (e) => ($(id + "Out").textContent = fmt(+e.target.value)));
  }

  function fillModels() {
    const sel = $("autoModel");
    sel.innerHTML = "";
    for (const m of state.models.filter((x) => x.available)) sel.appendChild(new Option(m.name, m.id));
    if (!sel.options.length) sel.appendChild(new Option("No model installed", ""));
  }

  function viewWindow() {
    const c = map.getCanvas();
    const corners = [[0, 0], [c.clientWidth, 0], [c.clientWidth, c.clientHeight], [0, c.clientHeight]].map((pt) => screenToPx(pt));
    const xs = corners.map((p) => p[0]), ys = corners.map((p) => p[1]);
    const w = [Math.max(0, Math.min(...xs)), Math.max(0, Math.min(...ys)), Math.min(A.geo.width, Math.max(...xs)), Math.min(A.geo.height, Math.max(...ys))];
    return w[2] > w[0] && w[3] > w[1] ? w : null;
  }

  $("autoRun").addEventListener("click", async () => {
    const st = $("autoStatus");
    const model = $("autoModel").value;
    if (!A.on || !model) return setStatus(st, "Install or add a model first (Detect mode → GeoAI model).", true);
    const region = document.querySelector('input[name="autoRegion"]:checked').value;
    const win = region === "view" ? viewWindow() : null;
    if (region === "view" && !win) return setStatus(st, "The current view does not overlap the image.", true);
    clearTimeout(A.saveTimer);
    await save();
    $("autoRun").disabled = true;
    try {
      const conf = +$("autoConf").value;
      const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raster_id: A.raster.id, model_id: model, conf, scale: +$("autoScale").value, window: win }) });
      let j = job;
      while (j.status === "queued" || j.status === "running") {
        setStatus(st, `Running model… ${Math.round(j.progress * 100)}%`);
        await new Promise((r) => setTimeout(r, 600));
        j = await api(`/api/jobs/${job.id}`);
      }
      if (j.status !== "done") throw new Error(j.error || j.status);
      const res = await api(`/api/rasters/${A.raster.id}/annotations/from-job`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: job.id, min_score: conf, replace_auto: $("autoReplace").checked }) });
      A.doc = res.doc;
      A.undo = []; A.redo = []; A.sel = null;
      render();
      refreshMask();
      setSaveState("saved", `Saved · v${A.doc.version}`);
      setStatus(st, `${res.imported} suggested labels added. Review and edit them, then export.`);
    } catch (err) {
      setStatus(st, err.message, true);
    } finally {
      $("autoRun").disabled = false;
    }
  });

  // ---------------------------------------------------------------- import / export

  $("exportBtn").addEventListener("click", async () => {
    const st = $("ioStatus");
    if (!A.on) return;
    clearTimeout(A.saveTimer);
    await save();
    const body = { format: $("exportFormat").value, chip_size: +$("chipSize").value, overlap: +$("chipOverlap").value,
      val_ratio: +$("valRatio").value, image_format: $("imgFormat").value, include_empty: $("includeEmpty").checked };
    setStatus(st, "Building dataset…");
    try {
      const r = await safeFetch(`/api/rasters/${A.raster.id}/annotations/export`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
      const blob = await r.blob();
      const name = (r.headers.get("content-disposition") || "").match(/filename="?([^";]+)/)?.[1] || "dataset.zip";
      const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      setStatus(st, `Downloaded ${name} (${(blob.size / 1e6).toFixed(1)} MB)`);
    } catch (err) {
      setStatus(st, `Export failed: ${err.message}`, true);
    }
  });

  $("importForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const st = $("ioStatus");
    if (!A.on) return;
    clearTimeout(A.saveTimer);
    await save();
    try {
      const res = await api(`/api/rasters/${A.raster.id}/annotations/import`, { method: "POST", body: new FormData(e.target) });
      A.doc = res.doc;
      A.undo = []; A.redo = []; A.sel = null;
      render();
      refreshMask();
      e.target.reset();
      setStatus(st, `Imported ${res.imported} labels.`);
    } catch (err) {
      setStatus(st, err.message, true);
    }
  });

  // ---------------------------------------------------------------- activation

  async function activate() {
    const r = state.raster;
    $("annToolbar").classList.toggle("hidden", !r || r.status !== "ready");
    $("annHelp").classList.toggle("hidden", !r || r.status !== "ready");
    if (!r || r.status !== "ready") {
      A.on = false;
      setStatus($("annStatus"), r ? "Fix the image's CRS first (see Imagery)." : "Load a georeferenced image to start labelling.", !!r);
      $("annClasses").innerHTML = "";
      setLayersVisible(false);
      return;
    }
    await mapReady;
    setupLayers();
    if (!A.raster || A.raster.id !== r.id || !A.geo) {
      A.raster = r;
      A.geo = null;
      A.geo = await loadGeoref(r);
      updateGeoDependentPaint();
      await loadDoc();
    }
    A.on = true;
    fillModels();
    setLayersVisible(true);
    map.doubleClickZoom.disable();
    map.boxZoom.disable();
    setTool(A.tool);
    render();
    refreshMask();
  }

  function deactivate() {
    if (A.on) { clearTimeout(A.saveTimer); save(); }
    A.on = false;
    cancelDraft();
    setLayersVisible(false);
    refreshMask();
    $("annToolbar").classList.add("hidden");
    $("annHelp").classList.add("hidden");
    document.querySelector(".map-wrap").classList.remove("drawing", "brushing");
    map.doubleClickZoom.enable();
    map.boxZoom.enable();
    map.dragPan.enable();
  }

  map.on("mousedown", onMouseDown);
  map.on("mousemove", onMouseMove);
  map.on("mouseup", onMouseUp);
  map.on("click", onClick);
  map.on("dblclick", onDblClick);
  map.on("zoom", updateCursorSize);
  map.getCanvas().addEventListener("mouseleave", () => { A.cursorPx = null; if (A.on && A.layersReady) renderDraft(); });

  document.addEventListener("mode-changed", (e) => (e.detail === "annotate" ? activate() : deactivate()));
  document.addEventListener("raster-changed", () => {
    if (A.on) save();
    A.raster = null; A.geo = null;
    if (state.mode === "annotate") activate();
  });
  window.addEventListener("beforeunload", (e) => {
    if ($("saveState").classList.contains("dirty")) { save(); e.preventDefault(); }
  });

  window.annotator = A;  // for debugging from the console
})();
