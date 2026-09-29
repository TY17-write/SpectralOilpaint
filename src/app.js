import { OilEngine } from "./engine.js";
const $ = (id) => document.getElementById(id),
  canvas = $("canvas");
let engine,
  busy = false,
  mode = "paint",
  failed = false,
  previous = 0,
  accumulator = 0,
  statsTime = 0,
  frames = 0;
const tables = await fetch("reference/tables.json").then((r) => {
  if (!r.ok) throw Error("顔料テーブルを読み込めません");
  return r.json();
});
function error(message) {
  failed = true;
  $("overlay").classList.remove("hidden");
  $("overlay").textContent =
    "開始できませんでした。下の詳細を確認してください。";
  $("status").textContent = String(message);
}
async function start() {
  busy = true;
  failed = false;
  $("overlay").classList.remove("hidden");
  $("overlay").textContent = "絵具とキャンバスを準備しています…";
  if (engine) {
    engine.destroy();
    engine = null;
  }
  try {
    const { nx, ny } = canvasSize();
    engine = await OilEngine.create(canvas, tables, {
      nx,
      ny,
      nl: Number($("layers").value),
    });
    engine.onError = error;
    window.oilpaint = engine;
    sync();
    fitCanvas();
    $("dimensions").textContent = `${engine.config.nx} × ${engine.config.ny}`;
    $("overlay").classList.add("hidden");
    $("status").textContent =
      "描画できます。混ぜるモードは絵具を装填しない筆です。";
    previous = performance.now();
    accumulator = 0;
  } catch (e) {
    console.error(e);
    error(e.stack || e.message);
  } finally {
    busy = false;
  }
}
function sync() {
  if (!engine) return;
  const c = engine.config;
  engine.color = $("color").value;
  $("hex").textContent = engine.color.toUpperCase();
  c.radius = Number($("radius").value);
  c.phi = Number($("phi").value) / 100;
  c.tiltX = Number($("tiltX").value);
  c.tiltY = Number($("tiltY").value);
  c.dryTime = Number($("dryTime").value);
  c.replenish = $("refill").checked ? 0.05 : 0;
  c.shading = $("shading").checked;
  c.pauseFluidDuringStroke = $("pauseFluid").checked;
  for (const [id, suffix] of [
    ["radius", " px"],
    ["phi", "%"],
    ["tiltX", "°"],
    ["tiltY", "°"],
  ])
    $(id + "Out").textContent = $(id).value + suffix;
}
for (const id of [
  "color",
  "radius",
  "phi",
  "tiltX",
  "tiltY",
  "dryTime",
  "refill",
  "shading",
  "pauseFluid",
])
  $(id).addEventListener("input", sync);
const baseColors = [
  "#c83228",
  "#d89332",
  "#f0d254",
  "#3b6653",
  "#2c6496",
  "#534379",
  "#fffaf0",
  "#252923",
];
function readSetting(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function saveSetting(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch {}
}
let savedColors = readSetting("oilpaint.palette", []);
if (!Array.isArray(savedColors)) savedColors = [];
savedColors = [...new Set(savedColors.filter(c => typeof c === "string" && /^#[0-9a-f]{6}$/i.test(c)).map(c => c.toLowerCase()))].filter(c => !baseColors.includes(c)).slice(-16);
function registerColor() {
  const color = $("color").value.toLowerCase();
  if (!baseColors.includes(color) && !savedColors.includes(color)) {
    savedColors.push(color);
    savedColors = savedColors.slice(-16);
    saveSetting("oilpaint.palette", savedColors);
  }
  renderPalette();
}
function renderPalette() {
  $("swatches").replaceChildren();
  for (const color of [...baseColors, ...savedColors]) {
  const b = document.createElement("button");
  b.style.background = color;
  b.setAttribute("aria-label", "絵具 " + color);
  b.title = color.toUpperCase();
  b.setAttribute("aria-pressed", String($("color").value.toLowerCase() === color));
  b.addEventListener("click", () => {
    $("color").value = color;
    sync();
    renderPalette();
  });
  $("swatches").append(b);
  }
  for (let i = baseColors.length + savedColors.length; i < 24; i++) {
    const b = document.createElement("button");
    b.className = "empty";
    b.textContent = "+";
    b.title = "色を選んで登録";
    b.setAttribute("aria-label", "色を選んで登録");
    b.onclick = () => $("color").click();
    $("swatches").append(b);
  }
}
$("color").addEventListener("change", registerColor);
renderPalette();
$("theme").value = readSetting("oilpaint.theme", "dark") === "light" ? "light" : "dark";
function setTheme() {
  document.documentElement.dataset.theme = $("theme").value;
  saveSetting("oilpaint.theme", $("theme").value);
}
$("theme").onchange = setTheme;
setTheme();
function canvasSize() {
  const ratios = { "4:3": 4 / 3, F: 530 / 455, P: 530 / 410, M: 530 / 333, S: 1 };
  let ratio = ratios[$("format").value] || 4 / 3;
  if ($("orientation").value === "portrait") ratio = 1 / ratio;
  const edge = Number($("resolution").value);
  const scale = Math.min(edge / Math.max(ratio, 1), 2048 / ratio, 1536);
  return { nx: Math.round(scale * ratio), ny: Math.round(scale) };
}
function nextSize() {
  const { nx, ny } = canvasSize();
  $("nextSize").textContent = `作成時に適用: ${nx} × ${ny} px（上限 2048 × 1536）`;
}
for (const id of ["format", "orientation", "resolution"]) $(id).onchange = nextSize;
nextSize();
function fitCanvas() {
  const stage = $("canvasstage"), wrap = canvas.parentElement;
  const ratio = engine ? engine.config.nx / engine.config.ny : 4 / 3;
  const width = Math.min(stage.clientWidth, stage.clientHeight * ratio);
  wrap.style.width = `${width}px`;
  wrap.style.height = `${width / ratio}px`;
}
new ResizeObserver(fitCanvas).observe($("canvasstage"));
// Fit the tool panel as a unit on desktop; narrow screens retain readable,
// scrollable controls below the canvas instead of shrinking touch targets.
const aside = document.querySelector("aside");
const controls = document.createElement("div");
controls.className = "controls";
controls.append(...aside.childNodes);
aside.append(controls);
function fitControls() {
  controls.style.zoom = "1";
  if (innerWidth <= 700) return;
  const style = getComputedStyle(aside);
  const available = aside.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
  controls.style.zoom = String(Math.min(1, available / controls.offsetHeight));
}
new ResizeObserver(fitControls).observe(aside);
fitControls();
for (const id of ["paint", "smudge"])
  $(id).onclick = () => {
    mode = id;
    $("paint").classList.toggle("selected", id === "paint");
    $("smudge").classList.toggle("selected", id === "smudge");
  };
function point(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) / r.width) * canvas.width,
    y: ((e.clientY - r.top) / r.height) * canvas.height,
    pressure: e.pointerType === "pen" ? Math.max(0.001, e.pressure) : 0.75,
    time: e.timeStamp,
  };
}
let pointer = null;
canvas.addEventListener("pointerdown", (e) => {
  if (!engine || failed || pointer !== null) return;
  e.preventDefault();
  pointer = e.pointerId;
  canvas.setPointerCapture(pointer);
  engine.beginStroke(point(e), { empty: mode === "smudge" });
});
canvas.addEventListener("pointermove", (e) => {
  const r = canvas.getBoundingClientRect(),
    cursor = $("cursor");
  cursor.style.display = "block";
  cursor.style.left = e.clientX - r.left + "px";
  cursor.style.top = e.clientY - r.top + "px";
  const size = (2 * Number($("radius").value) * r.width) / canvas.width;
  cursor.style.width = cursor.style.height = size + "px";
  if (e.pointerId === pointer && engine?.stroking) {
    for (const ev of e.getCoalescedEvents?.() || [e])
      engine.moveStroke(point(ev));
  }
});
function end(e) {
  if (e.pointerId !== pointer) return;
  engine?.endStroke();
  pointer = null;
}
canvas.addEventListener("pointerup", end);
canvas.addEventListener("pointercancel", end);
canvas.addEventListener("lostpointercapture", end);
canvas.addEventListener("pointerleave", () => {
  $("cursor").style.display = "none";
});
$("dry").onclick = () => {
  if (engine) engine.dryRequested = true;
};
$("restart").onclick = () => {
  if (engine && !confirm("現在の絵を消して、新しいキャンバスを作りますか？"))
    return;
  start();
};
$("export").onclick = async () => {
  if (!engine || failed) return;
  try {
    const blob = await engine.exportPNG();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download =
      "oilpaint-" + new Date().toISOString().replace(/[:.]/g, "-") + ".png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  } catch (e) {
    $("status").textContent = "保存に失敗しました: " + e.message;
  }
};
async function tick(now) {
  requestAnimationFrame(tick);
  if (!engine || busy || failed || document.hidden) return;
  accumulator += Math.min((now - previous) / 1000, 0.1);
  previous = now;
  if (
    accumulator < engine.config.dt &&
    !engine.pending.length &&
    !engine.dryRequested
  )
    return;
  accumulator = Math.max(0, accumulator - engine.config.dt);
  busy = true;
  const started = performance.now();
  try {
    engine.step();
    await engine.device.queue.onSubmittedWorkDone();
    frames++;
    if (now - statsTime > 1000) {
      $("stats").textContent =
        `${Math.round((frames * 1000) / (now - statsTime))} fps · ${Math.round(performance.now() - started)} ms`;
      statsTime = now;
      frames = 0;
    }
  } catch (e) {
    error(e.message);
  } finally {
    busy = false;
  }
}
await start();
requestAnimationFrame(tick);
