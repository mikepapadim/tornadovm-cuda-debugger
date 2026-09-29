// tcd web UI: a thin client over the tcd server (which drives cuda-gdb over GDB/MI).
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  status: "idle",
  kernel: null,       // kernel whose source is shown
  line: null,         // current stop line (when kernel is the stopped one)
  stoppedKernel: null,
  breakpoints: [],
  prevLocals: {},
  watches: JSON.parse(localStorageGet("tcd.watches") || "[]"),
};

function localStorageGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } }

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch("/api/" + path, opts);
  return r.json();
}

// ---------------------------------------------------------------- console

function log(text, cls) {
  const el = $(cls === "program" ? "program" : "console");
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 4;
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = text.endsWith("\n") ? text : text + "\n";
  el.appendChild(span);
  if (el.childNodes.length > 4000) el.removeChild(el.firstChild);
  if (atBottom) el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------- status & controls

function setStatus(s) {
  state.status = s;
  const el = $("status");
  el.textContent = s;
  el.className = "status " + s;
  const stopped = s === "stopped";
  const running = s === "running";
  for (const b of document.querySelectorAll("[data-action]")) {
    const a = b.dataset.action;
    b.disabled = (a === "run" && (running || stopped)) ||
      (["continue", "next", "step", "finish"].includes(a) && !stopped) ||
      (a === "interrupt" && !running) ||
      (a === "kill" && !(running || stopped));
  }
}

async function control(action) {
  const r = await api("control", { action });
  if (!r.ok && r.message) log("error: " + r.message, "err");
}

document.querySelectorAll("[data-action]").forEach((b) => b.addEventListener("click", () => control(b.dataset.action)));
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "F5") { e.preventDefault(); control(state.status === "idle" || state.status === "exited" ? "run" : "continue"); }
  if (e.key === "F10") { e.preventDefault(); control("next"); }
  if (e.key === "F11") { e.preventDefault(); control(e.shiftKey ? "finish" : "step"); }
});

// ---------------------------------------------------------------- source view

const KW = /\b(if|else|for|while|return|break|continue|int|long|float|double|char|unsigned|bool|void|short|half|extern|const)\b/g;
const CUDA = /\b(__global__|__shared__|__device__|__syncthreads|blockIdx|blockDim|threadIdx|gridDim|atomicAdd|fma)\b/g;

function highlight(line) {
  const esc = line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const c = esc.indexOf("//");
  const code = c >= 0 ? esc.slice(0, c) : esc;
  const com = c >= 0 ? `<span class="tok-com">${esc.slice(c)}</span>` : "";
  return code
    .replace(CUDA, '<span class="tok-cuda">$1</span>')
    .replace(KW, '<span class="tok-kw">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?[FLf]?)\b/g, '<span class="tok-num">$1</span>') + com;
}

function renderSource(kernel, text, currentLine) {
  state.kernel = kernel;
  const src = $("source");
  src.innerHTML = "";
  const lines = text.split("\n");
  const bpLines = new Set(state.breakpoints.filter((b) => b.kernel === kernel && b.line).map((b) => b.line));
  lines.forEach((l, i) => {
    const n = i + 1;
    const row = document.createElement("div");
    row.className = "ln" + (n === currentLine ? " cur" : "") + (bpLines.has(n) ? " bp" : "");
    row.dataset.line = n;
    row.innerHTML = `<span class="gut"></span><span class="num">${n}</span><span class="code">${highlight(l) || " "}</span>`;
    row.querySelector(".gut").onclick = row.querySelector(".num").onclick = () => toggleBreakpoint(kernel, n);
    src.appendChild(row);
  });
  $("src-title").textContent = kernel + "  ·  tornado_kernel.cu (generated CUDA C)";
  document.querySelectorAll("#kernels li").forEach((li) => li.classList.toggle("active", li.dataset.kernel === kernel));
  const cur = src.querySelector(".ln.cur");
  if (cur) cur.scrollIntoView({ block: "center" });
}

async function showKernel(kernel) {
  const r = await api("source?kernel=" + encodeURIComponent(kernel));
  if (r.text) renderSource(kernel, r.text, kernel === state.stoppedKernel ? state.line : null);
}

// ---------------------------------------------------------------- breakpoints

function renderBreakpoints(list) {
  if (!Array.isArray(list)) return;
  state.breakpoints = list;
  const ul = $("breakpoints");
  ul.innerHTML = "";
  for (const b of list) {
    const li = document.createElement("li");
    const label = b.kernel ? b.kernel + (b.line ? ":" + b.line : " (entry)") : b.location;
    li.innerHTML = `<span>#${b.number} ${label}</span><span class="muted">${b.hits ? b.hits + "×" : ""} <span class="x" title="delete">✕</span></span>`;
    li.querySelector(".x").onclick = async () => renderBreakpoints((await api("break", { delete: b.number })).breakpoints);
    ul.appendChild(li);
  }
  document.querySelectorAll("#source .ln").forEach((row) => {
    const n = +row.dataset.line;
    row.classList.toggle("bp", list.some((b) => b.kernel === state.kernel && b.line === n));
  });
}

async function toggleBreakpoint(kernel, line) {
  const existing = state.breakpoints.find((b) => b.kernel === kernel && b.line === line);
  const r = existing ? await api("break", { delete: existing.number }) : await api("break", { kernel, line });
  if (!r.ok) log("error: " + r.message, "err");
  renderBreakpoints(r.breakpoints);
}

$("bp-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const v = $("bp-input").value.trim();
  if (!v) return;
  const [kernel, line] = v.split(":");
  const r = await api("break", line ? { kernel, line: +line } : { kernel });
  if (r.message) log(r.message.trim(), r.ok ? "" : "err");
  renderBreakpoints(r.breakpoints);
  $("bp-input").value = "";
});

// ---------------------------------------------------------------- state after a stop

function renderKV(table, obj, prev, removable) {
  table.innerHTML = "";
  const keys = Object.keys(obj || {}).filter((k) => !k.startsWith("_"))
    .sort((a, b) => (+(a.match(/_(\d+)$/) || [0, -1])[1]) - (+(b.match(/_(\d+)$/) || [0, -1])[1]) || a.localeCompare(b));
  for (const k of keys) {
    const tr = document.createElement("tr");
    if (prev && k in prev && prev[k] !== obj[k]) tr.className = "changed";
    tr.innerHTML = `<td></td><td></td>`;
    tr.children[0].textContent = k;
    tr.children[1].textContent = obj[k];
    if (removable) {
      const x = document.createElement("span");
      x.className = "x"; x.textContent = "✕"; x.onclick = () => removable(k);
      tr.children[1].appendChild(x);
    }
    table.appendChild(tr);
  }
  if (!keys.length) table.innerHTML = '<tr><td class="muted">—</td></tr>';
}

async function refresh() {
  const s = await api("state");
  setStatus(s.state);
  renderKernelList(s.kernels);
  renderBreakpoints(s.breakpoints);
  if (s.state !== "stopped") return;
  const w = s.where || {};
  if (w.device) {
    state.stoppedKernel = w.function;
    state.line = w.line;
    if (s.source) renderSource(w.function, s.source, w.line);
    $("src-where").textContent = `line ${w.line}`;
    const f = w.focus;
    $("focus-now").textContent = `focused: block (${f.block}) thread (${f.thread})`;
    ["bx", "by", "bz"].forEach((id, i) => ($(id).value = f.block[i]));
    ["tx", "ty", "tz"].forEach((id, i) => ($(id).value = f.thread[i]));
    const t = Array.isArray(s.threads) && s.threads[0] ? s.threads[0] : {};
    renderKV($("locals"), t.locals, state.prevLocals);
    renderKV($("shared"), t.shared);
    state.prevLocals = t.locals || {};
    $("warps").textContent = s.warps || "";
    $("cudakernels").textContent = s.cudaKernels || "";
    refreshWatches();
  } else {
    state.stoppedKernel = null;
    $("src-where").textContent = "stopped in host code";
    log(s.backtrace || "", "muted");
  }
}

function renderKernelList(kernels) {
  const ul = $("kernels");
  if (!kernels || !kernels.length) return;
  ul.innerHTML = "";
  for (const k of kernels) {
    const li = document.createElement("li");
    li.className = "clickable" + (k === state.kernel ? " active" : "");
    li.dataset.kernel = k;
    li.innerHTML = `<span>${k}</span>`;
    li.onclick = () => showKernel(k);
    ul.appendChild(li);
  }
}

// ---------------------------------------------------------------- focus, watches, arrays

$("focus-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const v = (ids) => ids.map((id) => $(id).value || "0").join(",");
  const r = await api("focus", { block: v(["bx", "by", "bz"]), thread: v(["tx", "ty", "tz"]) });
  if (!r.ok) { log("focus: " + r.message, "err"); return; }
  state.prevLocals = {};
  refresh();
});

async function refreshWatches() {
  const values = {};
  for (const w of state.watches) {
    const r = await api("exec", { cmd: "output " + w });
    values[w] = r.ok ? r.output.trim() : r.output;
  }
  renderKV($("watches"), values, null, (k) => {
    state.watches = state.watches.filter((x) => x !== k);
    localStorageSet("tcd.watches", JSON.stringify(state.watches));
    refreshWatches();
  });
}

$("watch-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("watch-input").value.trim();
  if (!v || state.watches.includes(v)) return;
  state.watches.push(v);
  localStorageSet("tcd.watches", JSON.stringify(state.watches));
  $("watch-input").value = "";
  if (state.status === "stopped") refreshWatches();
});

$("array-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const r = await api("array", { expr: $("arr-expr").value || "arg1", type: $("arr-type").value, start: +$("arr-start").value, count: +$("arr-count").value });
  $("array-out").textContent = r.values ? r.values.map((v, i) => `[${r.start + i}] ${v}`).join("\n") : (r.error || JSON.stringify(r));
});

// ---------------------------------------------------------------- console input & tabs

$("cmd-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const cmd = $("cmd-input").value.trim();
  if (!cmd) return;
  log("(cuda-gdb) " + cmd, "cmd");
  $("cmd-input").value = "";
  const r = await api("exec", { cmd });
  if (!r.ok) log(r.output, "err");
  if (state.status === "stopped" && /^(cuda|up|down|frame|f|set)\b/.test(cmd)) refresh();
});

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
}));

// ---------------------------------------------------------------- events

let refreshTimer = null;
function scheduleRefresh() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(refresh, 60);
}

const es = new EventSource("/api/events");
es.onmessage = (m) => {
  const { type, payload } = JSON.parse(m.data);
  switch (type) {
    case "console":
      if (!String(payload).startsWith("TCD-JSON:")) log(payload);
      break;
    case "program":
      log(payload, "program");
      break;
    case "running":
      setStatus("running");
      break;
    case "stopped":
      setStatus("stopped");
      scheduleRefresh();
      break;
    case "exited":
      setStatus("exited");
      log("program exited" + (payload && payload["exit-code"] ? " with code " + payload["exit-code"] : ""), "muted");
      scheduleRefresh();
      break;
  }
};
es.onerror = () => setStatus("gone");

setStatus("idle");
refresh();
setInterval(() => { if (state.status === "running") api("state").then((s) => { renderKernelList(s.kernels); renderBreakpoints(s.breakpoints); }); }, 2000);
