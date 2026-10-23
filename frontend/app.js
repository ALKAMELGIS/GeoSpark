/* AI Platform — GeoAI web client */
"use strict";

const $ = (id) => document.getElementById(id);
const PALETTE = ["#3fb68b", "#e5534b", "#4c9aff", "#f2b84b", "#b97cf3", "#2ec4d6", "#f47b3c", "#9bd25b", "#ef6fa8",
  "#c9a36b", "#5ad1a8", "#7b8cff", "#e9d45b", "#ff8f7a", "#7fc8f8", "#d38fff", "#a3e635", "#f59e0b"];
const color = (i) => PALETTE[i % PALETTE.length];

const state = { models: [], rasters: [], raster: null, model: null, job: null, fc: null, hidden: new Set(), poll: null };

// ------------------------------------------------------------------ server connection

const OFFLINE_HELP = "Start the server with run.bat (double-click) or run.ps1 in the AI_Platfrom folder, keep its window open, and open the address it prints.";
let offlineTimer = null;

function showOffline(msg) {
  const el = $("offline");
  el.querySelector(".msg").textContent = msg;
  el.classList.remove("hidden");
  $("health").textContent = "API offline";
  $("health").classList.remove("ok");
  if (!offlineTimer && location.protocol !== "file:") offlineTimer = setInterval(checkServer, 3000);
}

async function checkServer() {
  try {
    const r = await fetch("/api/health", { cache: "no-store" });
    if (!r.ok) throw new Error();
    clearInterval(offlineTimer);
    offlineTimer = null;
    $("offline").classList.add("hidden");
    boot();  // reload models and rasters after the server comes back
  } catch (_) { /* still offline */ }
}

$("offlineRetry").addEventListener("click", checkServer);

let ourServer = null;
async function isOurServer() {
  if (ourServer !== null) return ourServer;
  try {
    const h = await (await fetch("/api/health", { cache: "no-store" })).json();
    ourServer = h.status === "ok" && "providers" in h;
  } catch (_) {
    ourServer = false;
  }
  return ourServer;
}

// "Failed to fetch" is the browser's message when no server answers: replace it with what to do.
async function safeFetch(path, opts) {
  try {
    return await fetch(path, opts);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    const msg = `Cannot reach the GeoAI server at ${location.origin}. ${OFFLINE_HELP}`;
    showOffline(msg);
    throw new Error(`Cannot reach the GeoAI server — is run.bat / run.ps1 running?`);
  }
}

async function api(path, opts = {}) {
  const r = await safeFetch(path, opts);
  if (r.status === 404 && path.startsWith("/api/") && !(await isOurServer())) {
    showOffline(`${location.origin} is answered by a different program, not the GeoAI server. ${OFFLINE_HELP}`);
    throw new Error("Another program is using this address.");
  }
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

function setStatus(el, text, isError = false) {
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ------------------------------------------------------------------ map

const map = new maplibregl.Map({
  container: "map",
  center: [20, 25],
  zoom: 2,
  attributionControl: { compact: true },
  style: {
    version: 8,
    sources: {
      esri: { type: "raster", tileSize: 256, maxzoom: 19,
        tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
        attribution: "Imagery © Esri, Maxar, Earthstar Geographics" },
      osm: { type: "raster", tileSize: 256, maxzoom: 19,
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        attribution: "© OpenStreetMap contributors" },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#0f1419" } },
      { id: "esri", type: "raster", source: "esri" },
      { id: "osm", type: "raster", source: "osm", layout: { visibility: "none" } },
    ],
  },
});
// The style becomes usable long before basemap tiles finish (or never do, offline),
// so map edits wait on the style alone rather than on the "load" event.
const mapReady = new Promise((resolve) => {
  const check = () => (map.style && map.style._loaded ? resolve() : setTimeout(check, 50));
  check();
});
map.addControl(new maplibregl.NavigationControl(), "top-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");

$("basemap").addEventListener("change", async (e) => {
  await mapReady;
  for (const id of ["esri", "osm"]) map.setLayoutProperty(id, "visibility", e.target.value === id ? "visible" : "none");
});

function bandKey() {
  const r = state.raster;
  if (!r || r.count < 2) return r ? r.default_bands.join("-") : "1-2-3";
  return [$("bandR").value, $("bandG").value, $("bandB").value].join("-");
}

function rasterTilesUrl(r) {
  return `${location.origin}/api/rasters/${r.id}/tiles/{z}/{x}/{y}.png?bands=${bandKey()}`;
}

// First overlay layer: imagery is inserted below it so vectors always draw on top.
function overlayAnchor() {
  return ["det-fill", "ann-mask", "ann-fill"].find((id) => map.getLayer(id));
}

// The imagery is served as exact Web Mercator XYZ tiles rendered from the GeoTIFF at native
// resolution. (A single image overlay loses accuracy: it is resampled, and MapLibre snaps its
// corners to 1/8192 of the covering tile, shifting it by metres against the basemap.)
async function showRaster(r) {
  await mapReady;
  for (const id of ["raster", "raster-footprint"]) if (map.getLayer(id)) map.removeLayer(id);
  for (const id of ["raster", "raster-footprint"]) if (map.getSource(id)) map.removeSource(id);
  if (!r || r.status !== "ready") return;
  map.addSource("raster", { type: "raster", tiles: [rasterTilesUrl(r)], tileSize: 256, bounds: r.bounds_wgs84,
    minzoom: r.minzoom, maxzoom: r.maxzoom, attribution: esc(r.name) });
  map.addLayer({ id: "raster", type: "raster", source: "raster",
    paint: { "raster-opacity": +$("opacity").value, "raster-fade-duration": 0, "raster-resampling": "nearest" } }, overlayAnchor());
  map.addSource("raster-footprint", { type: "geojson",
    data: { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [r.footprint] } } });
  map.addLayer({ id: "raster-footprint", type: "line", source: "raster-footprint",
    paint: { "line-color": "#f2b84b", "line-width": 1.5, "line-dasharray": [3, 2] } }, overlayAnchor());
  const [w, s, e, n] = r.bounds_wgs84;
  map.fitBounds([[w, s], [e, n]], { padding: 40, duration: 600 });
}

for (const id of ["bandR", "bandG", "bandB"]) {
  $(id).addEventListener("change", () => state.raster && map.getSource("raster") && map.getSource("raster").setTiles([rasterTilesUrl(state.raster)]));
}

function renderCrs(r) {
  const panel = $("crsPanel");
  if (!r) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  const c = r.crs_info;
  $("crsInfo").innerHTML = c
    ? `<b>${esc(c.name)}</b>${c.epsg ? ` · EPSG:${c.epsg}` : ""}<br>Datum ${esc(c.datum || "?")} · ${esc(c.type)} · ${esc(c.units || "")}`
    : "<b>No CRS</b>";
  $("crsWarnings").innerHTML = (r.warnings || []).map((w) => `<li class="${esc(w.level)}">${esc(w.message)}</li>`).join("");
  $("crsForm").classList.toggle("hidden", !(r.status !== "ready" || (r.warnings || []).some((w) => w.level !== "info") || panel.dataset.edit));
}

$("crsInfo").addEventListener("dblclick", () => { $("crsPanel").dataset.edit = "1"; renderCrs(state.raster); });

$("crsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const crs = $("crsInput").value.trim();
  if (!crs || !state.raster) return;
  try {
    const r = await api(`/api/rasters/${state.raster.id}/crs`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ crs }) });
    $("crsInput").value = "";
    delete $("crsPanel").dataset.edit;
    await loadRasters(r.id);
  } catch (err) {
    alert(err.message);
  }
});

$("opacity").addEventListener("input", (e) => map.getLayer("raster") && map.setPaintProperty("raster", "raster-opacity", +e.target.value));

function classColorExpr(names) {
  const expr = ["match", ["get", "class_id"]];
  names.forEach((_, i) => expr.push(i, color(i)));
  expr.push("#ffffff");
  return expr;
}

async function showDetections(fc) {
  await mapReady;
  state.fc = fc;
  state.hidden.clear();
  const names = fc.metadata.classes;
  if (map.getSource("det")) {
    map.getSource("det").setData(fc);
  } else {
    map.addSource("det", { type: "geojson", data: fc });
    map.addLayer({ id: "det-fill", type: "fill", source: "det", paint: { "fill-opacity": 0.25 } });
    map.addLayer({ id: "det-line", type: "line", source: "det", paint: { "line-width": ["interpolate", ["linear"], ["zoom"], 12, 1, 18, 2.5] } });
    map.addLayer({ id: "det-hl", type: "line", source: "det", paint: { "line-color": "#ffffff", "line-width": 3 }, filter: ["==", ["id"], -1] });
    map.on("click", "det-fill", (e) => openPopup(e.features[0], e.lngLat));
    map.on("mouseenter", "det-fill", () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", "det-fill", () => (map.getCanvas().style.cursor = ""));
  }
  map.setPaintProperty("det-fill", "fill-color", classColorExpr(names));
  map.setPaintProperty("det-line", "line-color", classColorExpr(names));
  applyClassFilter();
  renderSummary();
  renderTable();
}

function clearDetections() {
  state.fc = null;
  if (map.getSource("det")) map.getSource("det").setData({ type: "FeatureCollection", features: [] });
  $("summary").innerHTML = '<span class="muted">Run a model to create a detection layer.</span>';
  $("exportRow").classList.add("hidden");
  $("tablePanel").classList.add("hidden");
}

function applyClassFilter() {
  const hidden = [...state.hidden];
  const f = hidden.length ? ["!", ["in", ["get", "class_id"], ["literal", hidden]]] : null;
  for (const id of ["det-fill", "det-line"]) map.setFilter(id, f);
}

function openPopup(feature, lngLat) {
  const p = feature.properties;
  const rows = Object.entries(p).filter(([k]) => k !== "class_id")
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
  new maplibregl.Popup({ maxWidth: "280px" }).setLngLat(lngLat).setHTML(`<div class="popup"><table>${rows}</table></div>`).addTo(map);
  map.setFilter("det-hl", ["==", ["id"], feature.id]);
}

// ------------------------------------------------------------------ models

async function loadModels(selectId) {
  state.models = await api("/api/models");
  const sel = $("modelSelect");
  sel.innerHTML = "";
  const groups = {};
  for (const m of state.models) (groups[m.category] ||= []).push(m);
  const labels = { aerial: "Aerial object detection", trees: "Tree detection", buildings: "Building segmentation",
    "segment-anything": "Segment Anything (coming)", custom: "Custom models" };
  for (const [cat, items] of Object.entries(groups)) {
    const og = document.createElement("optgroup");
    og.label = labels[cat] || cat;
    for (const m of items) {
      const o = new Option(m.name + (m.planned ? " — planned" : m.available ? "" : " — not installed"), m.id);
      o.disabled = !m.available;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  const first = state.models.find((m) => m.id === selectId && m.available) || state.models.find((m) => m.available);
  if (first) sel.value = first.id;
  selectModel(first ? first.id : null);
}

function selectModel(id) {
  const m = state.models.find((x) => x.id === id) || null;
  state.model = m;
  const info = $("modelInfo");
  if (!m) {
    info.innerHTML = 'No model installed. Run <code>python scripts/export_models.py yolo26s-obb-dota2</code> or add a custom ONNX model.';
  } else {
    info.innerHTML = `<b>${esc(m.task.toUpperCase())}</b> · input ${m.imgsz}px · ${m.names.length} classes<br>${esc(m.description)}
      <span class="lic">License: ${esc(m.license)}</span>${m.source ? `<a href="${esc(m.source)}" target="_blank" rel="noopener">Source</a>` : ""}`;
  }
  const list = $("classList");
  list.innerHTML = "";
  (m ? m.names : []).forEach((n, i) => {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" value="${i}" checked><span class="swatch" style="background:${color(i)}"></span>${esc(n)}`;
    list.appendChild(l);
  });
  updateRunButton();
}

$("modelSelect").addEventListener("change", (e) => selectModel(e.target.value));
$("allClasses").addEventListener("click", () => document.querySelectorAll("#classList input").forEach((c) => (c.checked = true)));
$("noClasses").addEventListener("click", () => document.querySelectorAll("#classList input").forEach((c) => (c.checked = false)));

$("modelForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  try {
    const m = await api("/api/models", { method: "POST", body: fd });
    e.target.reset();
    await loadModels(m.id);
  } catch (err) {
    alert(`Model registration failed: ${err.message}`);
  }
});

// ------------------------------------------------------------------ rasters

async function loadRasters(selectId) {
  state.rasters = await api("/api/rasters");
  const sel = $("rasterSelect");
  sel.innerHTML = '<option value="">— no imagery loaded —</option>';
  for (const r of state.rasters) sel.appendChild(new Option(r.name, r.id));
  if (selectId) { sel.value = selectId; selectRaster(selectId); }
}

function fmtRes(r) {
  return r.is_geographic ? `${(r.res[0] * 111320).toFixed(2)} m/px (≈)` : `${r.res[0].toFixed(3)} m/px`;
}

function selectRaster(id) {
  const r = state.rasters.find((x) => x.id === id) || null;
  state.raster = r;
  clearDetections();
  const info = $("rasterInfo");
  const bandRow = $("bandRow");
  renderCrs(r);
  if (!r) { info.innerHTML = ""; bandRow.classList.add("hidden"); showRaster(null); updateRunButton(); renderJobs(); notifyRaster(); return; }
  info.innerHTML = `<b>${r.width} × ${r.height}</b> px · ${r.count} band(s) · ${esc(r.dtype)}<br>
    ${r.status === "ready" ? fmtRes(r) + " · " : ""}${(r.size_bytes / 1e6).toFixed(1)} MB${r.overviews?.length ? ` · ${r.overviews.length} overviews` : ""}`;
  for (const [k, sid] of [[0, "bandR"], [1, "bandG"], [2, "bandB"]]) {
    const s = $(sid);
    s.innerHTML = "";
    r.band_descriptions.forEach((d, i) => s.appendChild(new Option(`${i + 1}: ${d}`, i + 1)));
    s.value = r.default_bands[k];
  }
  bandRow.classList.toggle("hidden", r.count < 2);
  showRaster(r);
  updateRunButton();
  renderJobs();
  notifyRaster();
}

// Lets the annotation module react to raster changes without a hard dependency.
function notifyRaster() {
  document.dispatchEvent(new CustomEvent("raster-changed", { detail: state.raster }));
}

$("rasterSelect").addEventListener("change", (e) => selectRaster(e.target.value));

async function uploadRaster(files) {
  files = [...files];
  if (!files.length) return;
  const st = $("uploadStatus");
  const mb = files.reduce((a, f) => a + f.size, 0) / 1e6;
  setStatus(st, `Uploading ${files.map((f) => f.name).join(", ")} (${mb.toFixed(1)} MB) — building overviews…`);
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  try {
    const r = await api("/api/rasters", { method: "POST", body: fd });
    setStatus(st, r.status === "ready" ? "" : "Uploaded, but the CRS needs attention — see below.", r.status !== "ready");
    await loadRasters(r.id);
  } catch (err) {
    setStatus(st, err.message, true);
  }
}

const dz = $("dropzone");
$("rasterFile").addEventListener("change", (e) => { uploadRaster(e.target.files); e.target.value = ""; });
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("over");
  uploadRaster(e.dataTransfer.files);
});

// ------------------------------------------------------------------ jobs

for (const [id, fmt] of [["conf", (v) => v.toFixed(2)], ["iou", (v) => v.toFixed(2)], ["overlap", (v) => `${Math.round(v * 100)}%`], ["scale", (v) => `${v.toFixed(2).replace(/0$/, "")}×`]]) {
  $(id).addEventListener("input", (e) => ($(id + "Out").textContent = fmt(+e.target.value)));
}

function updateRunButton() {
  $("runBtn").disabled = !(state.raster && state.raster.status === "ready" && state.model && state.model.available) || !!state.poll;
}

$("runBtn").addEventListener("click", async () => {
  const classes = [...document.querySelectorAll("#classList input:checked")].map((c) => +c.value);
  if (!classes.length) return setStatus($("jobStatus"), "Select at least one class.", true);
  const all = classes.length === state.model.names.length;
  const body = {
    raster_id: state.raster.id, model_id: state.model.id,
    conf: +$("conf").value, iou: +$("iou").value, overlap: +$("overlap").value,
    classes: all ? null : classes, class_agnostic_nms: $("agnostic").checked, scale: +$("scale").value,
    bands: state.raster.count >= 2 ? [+$("bandR").value, +$("bandG").value, +$("bandB").value] : null,
  };
  try {
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    trackJob(job.id);
  } catch (err) {
    setStatus($("jobStatus"), err.message, true);
  }
});

$("cancelBtn").addEventListener("click", () => state.job && api(`/api/jobs/${state.job}/cancel`, { method: "POST" }));

function trackJob(id) {
  state.job = id;
  $("progressWrap").classList.remove("hidden");
  $("cancelBtn").classList.remove("hidden");
  $("progressBar").style.width = "0";
  const tick = async () => {
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch (err) { return finish(err.message, true); }
    $("progressBar").style.width = `${Math.round(job.progress * 100)}%`;
    setStatus($("jobStatus"), `${job.status} · ${job.message || ""}`);
    if (job.status === "done") { finish(`Done · ${job.summary.count} objects in ${job.summary.elapsed_s}s`); await loadResult(id); }
    else if (job.status === "error") finish(job.error, true);
    else if (job.status === "cancelled") finish("Cancelled");
    else state.poll = setTimeout(tick, 600);
  };
  const finish = (msg, err = false) => {
    clearTimeout(state.poll);
    state.poll = null;
    $("cancelBtn").classList.add("hidden");
    $("progressWrap").classList.add("hidden");
    setStatus($("jobStatus"), msg, err);
    updateRunButton();
    renderJobs();
  };
  state.poll = setTimeout(tick, 200);
  updateRunButton();
}

async function loadResult(id) {
  const fc = await api(`/api/jobs/${id}/result.geojson`);
  state.job = id;
  showDetections(fc);
  $("exportGeojson").href = `/api/jobs/${id}/export?format=geojson`;
  $("exportCsv").href = `/api/jobs/${id}/export?format=csv`;
  $("exportRow").classList.remove("hidden");
  renderJobs();
}

async function renderJobs() {
  const ul = $("jobList");
  if (!state.raster) { ul.innerHTML = '<li class="muted">No imagery selected.</li>'; return; }
  const jobs = (await api("/api/jobs")).filter((j) => j.raster_id === state.raster.id);
  ul.innerHTML = jobs.length ? "" : '<li class="muted">No runs yet.</li>';
  for (const j of jobs) {
    const li = document.createElement("li");
    const m = state.models.find((x) => x.id === j.model_id);
    const when = new Date(j.created * 1000).toLocaleTimeString();
    li.innerHTML = `<span>${esc(m ? m.name.split("—")[0] : j.model_id)} · ${when}</span><span class="muted">${j.status === "done" ? j.summary.count : esc(j.status)}</span>`;
    li.classList.toggle("active", j.id === state.job && !!state.fc);
    if (j.status === "done") li.addEventListener("click", () => loadResult(j.id));
    ul.appendChild(li);
  }
}

// ------------------------------------------------------------------ results

function renderSummary() {
  const fc = state.fc;
  const names = fc.metadata.classes;
  const counts = {};
  for (const f of fc.features) counts[f.properties.class_id] = (counts[f.properties.class_id] || 0) + 1;
  const el = $("summary");
  el.innerHTML = `<div class="total">${fc.features.length} objects · ${fc.metadata.tiles_processed} tiles · ${fc.metadata.elapsed_s}s · click a class to toggle</div>`;
  Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([cid, n]) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span class="swatch" style="background:${color(+cid)}"></span>${esc(names[cid])} <b>${n}</b>`;
    chip.addEventListener("click", () => {
      state.hidden.has(+cid) ? state.hidden.delete(+cid) : state.hidden.add(+cid);
      chip.classList.toggle("off", state.hidden.has(+cid));
      applyClassFilter();
    });
    el.appendChild(chip);
  });
}

const TABLE_LIMIT = 2000;
let sortKey = "score", sortDir = -1;

function renderTable() {
  const feats = state.fc.features.slice().sort((a, b) => {
    const x = a.properties[sortKey], y = b.properties[sortKey];
    return (x > y ? 1 : x < y ? -1 : 0) * sortDir;
  });
  const cols = Object.keys(feats[0]?.properties || {}).filter((k) => k !== "class_id");
  const head = cols.map((c) => `<th data-k="${esc(c)}">${esc(c)}${c === sortKey ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>`).join("");
  const body = feats.slice(0, TABLE_LIMIT).map((f) =>
    `<tr data-id="${f.id}">${cols.map((c) => `<td>${esc(f.properties[c])}</td>`).join("")}</tr>`).join("");
  $("attrTable").innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
  $("tableCount").textContent = feats.length > TABLE_LIMIT ? `showing ${TABLE_LIMIT} of ${feats.length}` : `${feats.length} rows`;
}

$("attrTable").addEventListener("click", (e) => {
  const th = e.target.closest("th");
  if (th) { const k = th.dataset.k; sortDir = k === sortKey ? -sortDir : -1; sortKey = k; return renderTable(); }
  const tr = e.target.closest("tr[data-id]");
  if (!tr) return;
  const f = state.fc.features.find((x) => x.id === +tr.dataset.id);
  const p = f.properties;
  map.flyTo({ center: [p.center_lon, p.center_lat], zoom: Math.max(map.getZoom(), 18), duration: 500 });
  map.setFilter("det-hl", ["==", ["id"], f.id]);
});

$("tableBtn").addEventListener("click", () => $("tablePanel").classList.toggle("hidden"));
$("closeTable").addEventListener("click", () => $("tablePanel").classList.add("hidden"));
$("aboutBtn").addEventListener("click", () => $("about").showModal());

// ------------------------------------------------------------------ modes

state.mode = "detect";

function setMode(mode) {
  state.mode = mode;
  $("modeDetect").classList.toggle("active", mode === "detect");
  $("modeAnnotate").classList.toggle("active", mode === "annotate");
  $("detectPanel").classList.toggle("hidden", mode !== "detect");
  $("annotatePanel").classList.toggle("hidden", mode !== "annotate");
  for (const id of ["det-fill", "det-line", "det-hl"]) {
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", mode === "detect" ? "visible" : "none");
  }
  if (mode !== "detect") $("tablePanel").classList.add("hidden");
  document.dispatchEvent(new CustomEvent("mode-changed", { detail: mode }));
}

$("modeDetect").addEventListener("click", () => setMode("detect"));
$("modeAnnotate").addEventListener("click", () => setMode("annotate"));

// ------------------------------------------------------------------ boot

async function boot() {
  if (location.protocol === "file:") {
    showOffline(`This page was opened as a file, so it cannot talk to the GeoAI server. ${OFFLINE_HELP}`);
    return;
  }
  try {
    const h = await api("/api/health");
    const gpu = h.providers.some((p) => /CUDA|Dml|TensorRT|ROCm|CoreML|OpenVINO/.test(p));
    $("health").textContent = `ONNX Runtime · ${gpu ? "GPU" : "CPU"}`;
    $("health").classList.add("ok");
    await loadModels(state.model?.id);
    await loadRasters(state.raster?.id);
    renderJobs();
  } catch (err) {
    console.warn("[boot]", err.message);  // the offline banner explains what to do
  }
}

boot();
