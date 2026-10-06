/* ANPR Console (WORKFLOW step 10). Plain JS, no build step, no external assets.
 * All server data reaches the DOM via textContent / attributes only — never innerHTML. */
"use strict";

(function () {
  const PER_PAGE = 50;
  const HEALTH_FALLBACK_MS = 10000;
  const POLL_FALLBACK_MS = 5000;
  const STATS_REFRESH_MS = 60000;
  const TOKEN_KEY = "anpr.token";
  const THEME_KEY = "anpr.theme";
  const SOUND_KEY = "anpr.sound";
  const RING_LEN = 2 * Math.PI * 18;
  const RECENT_N = 6; // "Recent vehicles" panel
  const TOAST_MS = 7000; // a new-plate pop-up stays this long
  const TOAST_MAX = 3;

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  };
  const SVG_NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs) => {
    const n = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, String(v));
    return n;
  };

  const state = {
    page: 1,
    src: null, // last /api/v1/source reply
    srcKey: "", // stream fields of the last health reply
    q: "",
    from: "",
    to: "",
    range: "all", // quick range buttons: all | hour | today | week | custom (dates typed in)
    recent: [], // newest confirmed plates, newest first (at most RECENT_N)
    total: 0,
    totalPages: 0,
    rows: [],
    maxId: 0,
    latest: null,
    pendingNew: 0,
    lastStatusAt: 0,
    ws: null,
    wsOpen: false,
    wsRetry: 0,
    wsTimer: null,
    loadSeq: 0,
    authBlocked: false,
    statsTimer: null,
    lastToday: null,
    snapReady: false, // the "Last vehicle" snapshot has a picture loaded
    health: null, // last /health (or WebSocket status) reply
  };

  // Capture panel "Live" view: the engine's latest frame (with plate boxes), pulled ~once a second.
  const VIEW_KEY = "anpr.capview";
  const LIVE_MS = 1000;
  const LIVE_IDLE_MS = 5000; // video ended / no signal: the picture can't change, ask less often
  const LIVE_MAX_BACKOFF_MS = 15000;
  const LIVE_TIMEOUT_MS = 10000; // a picture request hanging this long is dropped (else polling stalls)
  const live = {
    view: "live", // "live" | "last"
    interval: 1, // runtime.preview_interval_s from /api/v1/info (0 = switched off)
    timer: null,
    loading: false,
    seq: 0, // id of the request in flight; late answers of dropped requests are ignored
    watchdog: null,
    errors: 0,
    has: false, // a live picture is on screen
    kind: "wait", // badge state: on | wait | warn | bad | off
  };

  // ---- token / URLs -----------------------------------------------------------------------------

  const getToken = () => {
    try { return localStorage.getItem(TOKEN_KEY) || ""; } catch (_) { return ""; }
  };
  let memToken = getToken();
  // A shared link may carry the access token (?token=...): keep it, then take it out of the address bar.
  try {
    const u = new URL(location.href);
    const t = u.searchParams.get("token");
    if (t) {
      memToken = t;
      try { localStorage.setItem(TOKEN_KEY, t); } catch (_) { /* private mode */ }
      u.searchParams.delete("token");
      history.replaceState(null, "", u.pathname + u.search + u.hash);
    }
  } catch (_) { /* old browser */ }
  const setToken = (t) => {
    try {
      if (t) localStorage.setItem(TOKEN_KEY, t); else localStorage.removeItem(TOKEN_KEY);
    } catch (_) { /* private mode: token lives for this page only */ }
    memToken = t;
  };

  function withToken(path, params) {
    const u = new URL(path, location.origin);
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        if (v !== "" && v !== null && v !== undefined) u.searchParams.set(k, String(v));
      }
    }
    if (memToken) u.searchParams.set("token", memToken);
    return u.pathname + u.search;
  }

  // Quick ranges become a `from` value the API understands (ISO date or unix seconds).
  const isoDay = (d) => d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
  function rangeFrom(r) {
    if (r === "hour") return String(Math.floor(Date.now() / 1000) - 3600);
    if (r === "today") return isoDay(new Date());
    if (r === "week") { const d = new Date(); d.setDate(d.getDate() - 6); return isoDay(d); }
    return "";
  }
  const filterParams = () => ({ q: state.q, from: state.from || rangeFrom(state.range), to: state.to });
  const unfiltered = () => !state.q && !state.from && !state.to && state.range === "all";
  // Page 1 of a list that runs up to "now": a new plate belongs at its top.
  const liveListing = () => state.page === 1 && !state.q && !state.from && !state.to;

  class AuthError extends Error {}

  async function api(path, params) {
    const res = await fetch(withToken(path, params), { cache: "no-store", headers: { Accept: "application/json" } });
    if (res.status === 401) throw new AuthError("unauthorized");
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok && res.status !== 503) {
      const msg = body && body.error && body.error.message ? body.error.message : "HTTP " + res.status;
      throw new Error(msg);
    }
    return body;
  }

  // ---- admin password (web.admin_token) ----------------------------------------------------------
  // Viewing needs only the access token; changing things (licence, settings, camera) also needs the
  // admin password when the station has one. Asked once, kept for this browser tab only.

  const ADMIN_KEY = "anpr.admin";
  const adminPass = () => { try { return sessionStorage.getItem(ADMIN_KEY) || ""; } catch (_) { return memAdmin; } };
  let memAdmin = "";
  function setAdmin(p) {
    memAdmin = p;
    try { if (p) sessionStorage.setItem(ADMIN_KEY, p); else sessionStorage.removeItem(ADMIN_KEY); } catch (_) { /* private mode */ }
  }
  const adminHeaders = () => (adminPass() ? { "X-Admin-Token": adminPass() } : {});

  let adminWaiter = null;
  function askAdmin(wrong) {
    // Resolves with true when a password was entered (call again), false when cancelled.
    if (adminWaiter) return adminWaiter.promise;
    const dlg = $("admindlg");
    let resolve;
    const promise = new Promise((r) => { resolve = r; });
    adminWaiter = { promise, resolve };
    $("admininput").value = "";
    $("admin-why").textContent = wrong
      ? "That admin password is not right. Try again."
      : "Changing this station (licence, settings, camera) needs the admin password. Viewing does not.";
    if (typeof dlg.showModal === "function") dlg.showModal();
    else {
      const p = window.prompt("Admin password");
      adminWaiter = null;
      if (p) setAdmin(p.trim());
      resolve(!!p);
    }
    return promise;
  }
  $("admindlg").addEventListener("close", () => {
    const w = adminWaiter;
    adminWaiter = null;
    const p = $("admininput").value.trim();
    const ok = $("admindlg").returnValue === "ok" && !!p;
    if (ok) setAdmin(p);
    if (w) w.resolve(ok);
  });

  // fetch() for admin actions: adds the admin password, asks for it (again) when the station says so.
  async function adminFetch(path, init) {
    for (let attempt = 0; attempt < 3; attempt++) {
      const opts = Object.assign({}, init, { headers: Object.assign({}, init.headers || {}, adminHeaders()) });
      const res = await fetch(withToken(path), opts);
      if (res.status !== 403) return res;
      let body = null;
      try { body = await res.clone().json(); } catch (_) { /* non-JSON */ }
      if (!(body && body.error && body.error.code === "admin_required")) return res;
      const wrong = !!adminPass();
      setAdmin("");
      if (!(await askAdmin(wrong))) {
        const e = new Error("The admin password is needed to change this station.");
        e.adminCancelled = true;
        throw e;
      }
    }
    throw new Error("The admin password was not accepted.");
  }

  // ---- token prompt -----------------------------------------------------------------------------

  let tokenPromptOpen = false;
  function askToken() {
    state.authBlocked = true;
    $("changetoken").hidden = false;
    showMsg("Access token required.");
    if (tokenPromptOpen) return;
    const dlg = $("tokendlg");
    tokenPromptOpen = true;
    $("tokeninput").value = "";
    if (typeof dlg.showModal === "function") {
      dlg.showModal();
    } else {
      const t = window.prompt("Access token required");
      tokenPromptOpen = false;
      if (t) acceptToken(t.trim());
    }
  }
  function acceptToken(t) {
    setToken(t);
    state.authBlocked = false;
    hideMsg();
    loadAll();
    reconnectNow();
    live.errors = 0;
    scheduleLive(0);
  }
  $("tokendlg").addEventListener("close", () => {
    tokenPromptOpen = false;
    const dlg = $("tokendlg");
    const t = $("tokeninput").value.trim();
    if (dlg.returnValue === "ok" && t) acceptToken(t);
  });
  $("changetoken").addEventListener("click", () => askToken());

  // ---- messages ---------------------------------------------------------------------------------

  function showMsg(text) { const m = $("msg"); m.textContent = text; m.hidden = false; }
  function hideMsg() { $("msg").hidden = true; }

  // ---- formatting -------------------------------------------------------------------------------

  const pad = (n) => String(n).padStart(2, "0");
  const fmtTime = (ts) => {
    const d = new Date(ts * 1000);
    return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
  };
  const fmtDate = (ts) => new Date(ts * 1000).toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
  const sameDay = (a, b) => a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
  function fmtDay(ts) {
    const d = new Date(ts * 1000);
    const now = new Date();
    if (sameDay(d, now)) return "Today";
    const y = new Date(now); y.setDate(now.getDate() - 1);
    if (sameDay(d, y)) return "Yesterday";
    return fmtDate(ts);
  }
  function fmtAgo(ts) {
    const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
    if (s < 5) return "just now";
    if (s < 60) return s + " s ago";
    if (s < 3600) return Math.floor(s / 60) + " min ago";
    if (s < 86400) return Math.floor(s / 3600) + " h " + Math.floor((s % 3600) / 60) + " min ago";
    return Math.floor(s / 86400) + " d ago";
  }
  function fmtDuration(s) {
    s = Math.max(0, s);
    if (s < 60) return s.toFixed(1) + " s";
    return Math.floor(s / 60) + " min " + Math.round(s % 60) + " s";
  }
  const pct = (c) => Math.round(Math.max(0, Math.min(1, c)) * 100) + "%";
  const confClass = (c) => (c >= 0.85 ? "c-hi" : c >= 0.6 ? "c-mid" : "c-lo");
  const kindLabel = (k) => (k === "bh" ? "BH series" : k === "manual" ? "2-line · manual" : "Standard");
  // HSRP check (engine): hologram + "IND" left of the number. null = not checked (older event).
  const HSRP_UI = { hsrp: ["HSRP", " tag-ok"], non_hsrp: ["Non-HSRP", " tag-bad"], unsure: ["Not sure", ""] };
  const hsrpLabel = (h) => (HSRP_UI[h] ? HSRP_UI[h][0] : "—");
  const hsrpTag = (h) => el("span", "tag" + (HSRP_UI[h] ? HSRP_UI[h][1] : " tag-none"), hsrpLabel(h));
  const num = (n) => Number(n).toLocaleString();

  // ---- building blocks --------------------------------------------------------------------------

  function plateNode(text, size) {
    const p = el("div", "plate" + (size ? " plate-" + size : ""));
    const band = el("span", "plate-band");
    band.append(el("i"), el("b", "", "IND"));
    p.append(band, el("span", "plate-txt", text));
    return p;
  }
  function setPlate(node, text) {
    node.classList.remove("is-empty");
    node.querySelector(".plate-txt").textContent = text;
  }
  const displayOf = (ev) => ev.plate_display || ev.plate;

  function chevron() {
    const s = svgEl("svg", { viewBox: "0 0 20 20", "aria-hidden": "true" });
    s.append(svgEl("path", { d: "M8 5l5 5-5 5" }));
    return s;
  }

  function cropCell(ev) {
    if (!ev.crop_url) return el("div", "t-crop-missing", "no image");
    const img = el("img", "t-crop");
    img.loading = "lazy";
    img.decoding = "async";
    img.alt = "Plate crop " + displayOf(ev);
    img.src = withToken(ev.crop_url);
    img.addEventListener("error", () => img.replaceWith(el("div", "t-crop-missing", "image gone")), { once: true });
    return img;
  }

  // Small plate picture for the Recent panel and the new-plate pop-up.
  function cropThumb(ev) {
    if (!ev.crop_url) return el("span", "r-crop-missing", "no image");
    const img = el("img", "r-crop");
    img.loading = "lazy";
    img.decoding = "async";
    img.alt = "Plate image " + displayOf(ev);
    img.src = withToken(ev.crop_url);
    img.addEventListener("error", () => img.replaceWith(el("span", "r-crop-missing", "image gone")), { once: true });
    return img;
  }
  const pctPill = (c) => el("span", "pct " + confClass(c), pct(c));

  function confCell(c) {
    const wrap = el("div", "conf " + confClass(c));
    const meter = el("span", "meter");
    const fill = el("i");
    fill.style.width = pct(c);
    meter.append(fill);
    wrap.append(el("span", "", pct(c)), meter);
    return wrap;
  }

  function td(cls, ...kids) {
    const c = el("td", cls);
    c.append(...kids);
    return c;
  }

  function rowNode(ev, fresh) {
    const tr = el("tr", fresh ? "fresh" : "");
    tr.tabIndex = 0;
    tr.dataset.id = String(ev.id);
    tr.setAttribute("aria-label", displayOf(ev) + " at " + fmtTime(ev.last_seen) + ", open details");
    const time = el("div", "t-time");
    time.append(el("b", "", fmtTime(ev.last_seen)), el("small", "", fmtDay(ev.last_seen)));
    time.title = ev.last_seen_iso || "";
    tr.append(
      td("", time),
      td("", plateNode(displayOf(ev), "sm")),
      td("c-crop", cropCell(ev)),
      td("", confCell(ev.confidence)),
      td("c-num", ev.votes + " frames"),
      td("c-kind", el("span", "tag" + (ev.kind === "bh" ? " tag-bh" : ""), kindLabel(ev.kind))),
      td("c-hsrp", hsrpTag(ev.hsrp)),
      td("c-go", chevron()),
    );
    tr.addEventListener("click", () => openDrawer(ev));
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openDrawer(ev); }
    });
    return tr;
  }

  // ---- rendering: recent vehicles ---------------------------------------------------------------

  function renderRecent(freshId) {
    const frag = document.createDocumentFragment();
    if (!state.recent.length) {
      frag.append(el("li", "recent-empty", "No vehicles yet. Confirmed plates will appear here."));
    } else {
      state.recent.forEach((ev, i) => {
        const li = el("li", ev.id === freshId ? "fresh" : "");
        const b = el("button", "r-item");
        b.type = "button";
        b.setAttribute("aria-label", displayOf(ev) + " at " + fmtTime(ev.last_seen) + ", open details");
        const main = el("span", "r-main");
        const sub = el("span", "r-sub");
        const ago = el("span", "", fmtAgo(ev.last_seen));
        ago.dataset.ago = String(ev.last_seen);
        sub.append(el("b", "", fmtTime(ev.last_seen)), ago);
        if (i === 0) sub.append(el("span", "r-new", "Newest"));
        main.append(plateNode(displayOf(ev), "md"), sub);
        const side = el("span", "r-side");
        side.append(cropThumb(ev), pctPill(ev.confidence));
        b.append(main, side);
        b.addEventListener("click", () => openDrawer(ev));
        li.append(b);
        frag.append(li);
      });
    }
    $("recent").replaceChildren(frag);
  }

  function addRecent(ev) {
    if (state.recent.some((r) => r.id === ev.id)) return false;
    state.recent.unshift(ev);
    state.recent.sort((a, b) => b.last_seen - a.last_seen || b.id - a.id);
    if (state.recent.length > RECENT_N) state.recent.length = RECENT_N;
    renderRecent(ev.id);
    return true;
  }

  // ---- new-plate pop-up + sound -----------------------------------------------------------------

  let soundOn = false;
  try { soundOn = localStorage.getItem(SOUND_KEY) === "1"; } catch (_) { /* private mode */ }
  let audio = null;
  function audioCtx() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    if (!audio) audio = new Ctx();
    if (audio.state === "suspended") audio.resume().catch(() => {});
    return audio;
  }
  function chime() {
    if (!soundOn) return;
    try {
      const ctx = audioCtx();
      if (!ctx) return;
      const t0 = ctx.currentTime + 0.02;
      for (const [f, dt] of [[880, 0], [1318.5, 0.13]]) {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = "sine";
        o.frequency.value = f;
        g.gain.setValueAtTime(0.0001, t0 + dt);
        g.gain.exponentialRampToValueAtTime(0.16, t0 + dt + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, t0 + dt + 0.38);
        o.connect(g);
        g.connect(ctx.destination);
        o.start(t0 + dt);
        o.stop(t0 + dt + 0.42);
      }
    } catch (_) { /* no audio on this device */ }
  }
  function renderSound() {
    const b = $("sound");
    b.setAttribute("aria-pressed", String(soundOn));
    b.title = "Sound for new plates: " + (soundOn ? "on" : "off") + " — click to turn " + (soundOn ? "off" : "on");
  }
  $("sound").addEventListener("click", () => {
    soundOn = !soundOn;
    try { localStorage.setItem(SOUND_KEY, soundOn ? "1" : "0"); } catch (_) { /* private mode */ }
    renderSound();
    chime(); // the click also unlocks audio in the browser
  });
  // Browsers only allow sound after a click on the page; unlock it on the first one.
  document.addEventListener("pointerdown", () => { if (soundOn) audioCtx(); }, { once: true, passive: true });

  function toast(ev) {
    const box = $("toasts");
    const t = el("button", "toast");
    t.type = "button";
    t.setAttribute("aria-label", "New vehicle " + displayOf(ev) + ", open details");
    const k = el("span", "toast-k", "New vehicle");
    k.append(el("span", "", fmtTime(ev.last_seen)));
    const main = el("span", "r-main");
    const sub = el("span", "r-sub");
    sub.append(pctPill(ev.confidence), el("span", "", "seen in " + ev.votes + " frames"));
    main.append(plateNode(displayOf(ev), "md"), sub);
    t.append(k, main, cropThumb(ev));
    let gone = false;
    const close = () => {
      if (gone) return;
      gone = true;
      t.classList.add("out");
      setTimeout(() => t.remove(), 320);
    };
    t.addEventListener("click", () => { close(); openDrawer(ev); });
    box.prepend(t);
    while (box.children.length > TOAST_MAX) box.lastElementChild.remove();
    setTimeout(close, TOAST_MS);
  }

  function announce(ev) {
    toast(ev);
    chime();
  }

  // ---- rendering: log ---------------------------------------------------------------------------

  function renderRows(freshIds) {
    const body = $("rows");
    const frag = document.createDocumentFragment();
    if (!state.rows.length) {
      const tr = el("tr", "empty");
      const c = el("td", "", unfiltered() ? "No plates recorded yet." : "No plates match these filters.");
      c.colSpan = 7;
      tr.append(c);
      frag.append(tr);
    } else {
      for (const ev of state.rows) frag.append(rowNode(ev, freshIds && freshIds.has(ev.id)));
    }
    body.replaceChildren(frag);
    renderPager();
  }

  const RANGE_TXT = { hour: "in the last hour", today: "today", week: "in the last 7 days" };
  function summaryText() {
    const n = num(state.total) + (state.total === 1 ? " plate" : " plates");
    if (unfiltered()) return n + " confirmed on record";
    let s = n;
    if (state.q) s += " matching “" + state.q + "”";
    if (state.from || state.to) s += " in the chosen dates";
    else if (RANGE_TXT[state.range]) s += " " + RANGE_TXT[state.range];
    return s;
  }

  function renderPager() {
    const tp = Math.max(1, state.totalPages);
    $("pageinfo").textContent = "Page " + state.page + " of " + tp;
    $("prev").disabled = state.page <= 1;
    $("next").disabled = state.page >= tp;
    $("summary").textContent = summaryText();
    const nb = $("newbadge");
    if (state.pendingNew > 0) {
      nb.textContent = state.pendingNew + " new plate" + (state.pendingNew > 1 ? "s" : "") + " — show latest";
      nb.hidden = false;
    } else {
      nb.hidden = true;
    }
    $("export").href = withToken("/api/v1/plates/export.csv", filterParams());
  }

  // ---- rendering: live capture ------------------------------------------------------------------

  function renderCapture(flash) {
    const ev = state.latest;
    if (!ev) return;
    const plate = $("cap-plate");
    setPlate(plate, displayOf(ev));
    $("cap-time").textContent = fmtTime(ev.last_seen);
    $("cap-votes").textContent = ev.votes + " frames";
    $("cap-kind").textContent = kindLabel(ev.kind);
    $("cap-hsrp").textContent = hsrpLabel(ev.hsrp);
    $("cap-conf").textContent = pct(ev.confidence);
    $("cap-ring").style.strokeDashoffset = String(RING_LEN * (1 - Math.max(0, Math.min(1, ev.confidence))));

    const snap = $("cap-snap");
    const src = ev.snapshot_url || ev.crop_url;
    state.snapReady = !!src && snap.hasAttribute("src"); // the old picture stays up until the new one loads
    if (src) {
      const url = withToken(src);
      if (snap.getAttribute("src") !== url) {
        snap.classList.add("fading");
        const pre = new Image();
        pre.onload = pre.onerror = () => {
          snap.src = url;
          snap.alt = "Snapshot of " + displayOf(ev);
          state.snapReady = true;
          syncViewfinder();
          requestAnimationFrame(() => snap.classList.remove("fading"));
        };
        pre.src = url;
      }
    } else {
      snap.removeAttribute("src");
    }
    syncViewfinder();

    const cw = $("cap-cropwrap");
    if (ev.crop_url && ev.snapshot_url) {
      // An image removed by retention would show as a broken picture: hide the box instead.
      $("cap-crop").onerror = () => { cw.hidden = true; };
      $("cap-crop").src = withToken(ev.crop_url);
      $("cap-crop").alt = "Plate crop " + displayOf(ev);
      cw.hidden = false;
    } else {
      cw.hidden = true;
    }
    $("cap-plate").onclick = () => openDrawer(ev);
    tickAges();
    if (flash) {
      plate.classList.remove("stamp");
      void plate.offsetWidth; // restart the animation
      plate.classList.add("stamp");
    }
  }

  function tickAges() {
    const now = new Date();
    $("clock-time").textContent = pad(now.getHours()) + ":" + pad(now.getMinutes()) + ":" + pad(now.getSeconds());
    $("clock-date").textContent = now.toLocaleDateString(undefined, { weekday: "short", day: "2-digit", month: "short" });
    const age = $("cap-age");
    if (state.latest) {
      const s = Date.now() / 1000 - state.latest.last_seen;
      age.textContent = fmtAgo(state.latest.last_seen);
      age.className = "chip " + (s < 60 ? "chip-signal" : "chip-muted");
    }
    for (const n of document.querySelectorAll("[data-ago]")) n.textContent = fmtAgo(Number(n.dataset.ago));
  }

  // ---- live view --------------------------------------------------------------------------------

  const EMPTY_LAST = [
    "Waiting for the first confirmed vehicle",
    "Plates appear here once several frames agree and the Indian format check passes.",
  ];

  // -> [badge kind, badge text, detail line for the empty viewfinder, show the "Add stream" button]
  function liveStatus() {
    const h = state.health;
    const d = state.src;
    if (live.interval === 0) return ["off", "Live off", "The live view is switched off (runtime.preview_interval_s: 0 in config.yaml)."];
    if (!h) return ["wait", "Connecting", "Connecting to the station…"];
    if (h.status === "no_engine" || !h.engine_ok) return ["bad", "Engine off", "The ANPR engine is not running. Start it to see the camera."];
    // A server started before the live view existed serves this page (read from disk) but no pictures.
    if (!("preview_fresh" in h)) return ["warn", "Restart needed", "Restart the ANPR engine and web services to turn on the live picture."];
    if (d && d.pending) return ["wait", "Switching", "Switching to the new stream…"];
    if (h.source_state === "ended") return ["warn", "Video ended", "The video has finished. Add a stream to play another one or go back to the camera.", true];
    if (h.source_state === "error") return ["bad", "Can't open", "The engine couldn't open this stream. Check the address with Add stream.", true];
    if (h.camera_ok === false) return ["bad", "No signal", "No picture from the camera. The engine keeps reconnecting.", true];
    if (h.preview_fresh === false || !live.has || live.errors >= 2) return ["wait", "Waiting for picture", "The engine is reading the stream; the first picture appears in a moment."];
    return ["on", "Live", h.fps === null || h.fps === undefined ? "" : h.fps.toFixed(1) + " fps"];
  }

  function syncViewfinder() {
    const isLive = live.view === "live";
    const ev = state.latest;
    const lastHas = !!(ev && (ev.snapshot_url || ev.crop_url));
    $("cap-snap").hidden = isLive || !state.snapReady;
    $("cap-live").hidden = !isLive || !live.has;
    $("cap-badge").hidden = !isLive;
    $("cap-empty").hidden = isLive ? live.has : lastHas;
    if (isLive) {
      const [kind, text, detail] = liveStatus();
      $("cap-empty-t").textContent = kind === "on" || kind === "wait" ? "Waiting for the camera picture" : text;
      $("cap-empty-s").textContent = detail;
    } else {
      $("cap-empty-t").textContent = EMPTY_LAST[0];
      $("cap-empty-s").textContent = EMPTY_LAST[1];
    }
  }

  function renderLive() {
    const [kind, text, detail, cta] = liveStatus();
    live.kind = kind;
    $("cap-cta").hidden = !(cta && live.view === "live");
    const b = $("cap-badge");
    b.className = "vf-badge is-" + kind;
    $("cap-badge-t").textContent = text;
    $("cap-badge-s").textContent = kind === "on" ? detail : "";
    b.title = kind === "on" ? "Latest camera picture, refreshed every second" : detail;
    // An old picture stays up (dimmed) so the operator still sees where the camera points.
    $("capture").classList.toggle("is-stale", live.has && kind !== "on");
    syncViewfinder();
  }

  const liveWanted = () => live.view === "live" && !document.hidden && !state.authBlocked && live.interval !== 0;

  function scheduleLive(ms) {
    clearTimeout(live.timer);
    live.timer = null;
    if (liveWanted()) live.timer = setTimeout(pullLive, ms);
  }

  function liveDelay() {
    const every = Math.max(LIVE_MS, live.interval * 1000);
    return live.kind === "on" || live.kind === "wait" ? every : Math.max(every, LIVE_IDLE_MS);
  }

  // Double-buffered: the next picture loads off-screen and replaces the shown <img> only once decoded.
  function pullLive() {
    live.timer = null;
    if (!liveWanted() || live.loading) return;
    live.loading = true; // never more than one picture request in flight
    const seq = ++live.seq;
    const img = new Image();
    const settle = () => {
      if (seq !== live.seq || !live.loading) return false; // dropped by the watchdog: ignore
      clearTimeout(live.watchdog);
      live.loading = false;
      return true;
    };
    const show = () => {
      if (!settle()) return;
      live.errors = 0;
      const cur = $("cap-live");
      img.id = "cap-live";
      img.className = cur.className;
      img.alt = "Live camera picture";
      img.hidden = cur.hidden;
      cur.replaceWith(img);
      live.has = true;
      renderLive();
      scheduleLive(liveDelay());
    };
    // Swap only once decoded, so the old picture never blinks out.
    img.onload = () => { if (img.decode) img.decode().then(show, show); else show(); };
    const fail = () => {
      // 404 = no picture yet, or the server/network is down. The image can't tell us about a 401,
      // so ask the API once: it opens the token prompt when the token is wrong.
      if (!settle()) return;
      live.errors += 1;
      if (live.errors === 2 && !state.authBlocked) {
        api("/api/v1/source").catch((e) => { if (e instanceof AuthError) askToken(); });
      }
      renderLive();
      scheduleLive(Math.min(LIVE_MAX_BACKOFF_MS, liveDelay() * Math.pow(2, Math.min(live.errors, 4))));
    };
    img.onerror = fail;
    // A stalled request (slow Pi, dead connection) would otherwise block every later picture.
    live.watchdog = setTimeout(() => {
      img.onload = img.onerror = null;
      img.removeAttribute("src"); // lets the browser drop the request
      fail();
    }, LIVE_TIMEOUT_MS);
    img.src = withToken("/api/v1/live.jpg", { t: Date.now() }); // cache-buster
  }

  function setView(view, focus) {
    live.view = view === "last" ? "last" : "live";
    try { localStorage.setItem(VIEW_KEY, live.view); } catch (_) { /* private mode */ }
    for (const b of document.querySelectorAll(".vt-b")) {
      const on = b.dataset.view === live.view;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
      if (on && focus) b.focus();
    }
    $("capture").classList.toggle("is-live", live.view === "live");
    $("cap-title").textContent = live.view === "live" ? "Live camera" : "Last confirmed vehicle";
    renderLive();
    scheduleLive(0); // "Live": fetch now; "Last vehicle": just stops the polling
  }

  for (const b of document.querySelectorAll(".vt-b")) {
    b.addEventListener("click", () => setView(b.dataset.view, false));
    b.addEventListener("keydown", (e) => {
      if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) {
        e.preventDefault();
        const next = e.key === "Home" ? "live" : e.key === "End" ? "last" : live.view === "live" ? "last" : "live";
        setView(next, true);
      }
    });
  }

  // ---- rendering: figures -----------------------------------------------------------------------

  // Numbers roll up to their value the first time, and bump when they change later.
  const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  function countUp(n, txt) {
    const m = /^([\d,]+)(\.\d+)?(%?)$/.exec(txt);
    if (!m || reduceMotion()) { n.textContent = txt; return; }
    const target = Number((m[1] + (m[2] || "")).replace(/,/g, ""));
    const dec = m[2] ? m[2].length - 1 : 0;
    const t0 = performance.now(), dur = 900;
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur);
      const v = target * (1 - Math.pow(1 - k, 3));
      n.textContent = (dec ? v.toFixed(dec) : Math.round(v).toLocaleString()) + m[3];
      if (k < 1 && n.dataset.target === txt) requestAnimationFrame(step);
      else if (k >= 1) n.textContent = txt;
    };
    n.dataset.target = txt;
    requestAnimationFrame(step);
  }
  function setKpi(id, value) {
    const n = $(id);
    const txt = String(value);
    if (n.textContent === "—" && txt !== "—") { countUp(n, txt); return; }
    delete n.dataset.target; // stop a running count-up; the real value is set below
    if (n.textContent !== txt) {
      n.classList.remove("bump");
      void n.offsetWidth;
      n.classList.add("bump");
    }
    n.textContent = txt;
  }

  function renderSpark(buckets) {
    const svg = $("k-spark");
    if (!svg || buckets.length < 2) return;
    const W = 120, H = 36, max = Math.max(1, ...buckets.map((b) => b.count));
    const pts = buckets.map((b, i) => [(i * W) / (buckets.length - 1), H - 3 - ((H - 6) * b.count) / max]);
    const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
    const last = pts[pts.length - 1];
    svg.replaceChildren(
      svgEl("path", { class: "sp-area", d: d + ` L${W} ${H} L0 ${H} Z` }),
      svgEl("path", { class: "sp-line", d }),
      svgEl("circle", { class: "sp-dot", cx: last[0].toFixed(1), cy: last[1].toFixed(1), r: 2.6 }),
    );
  }

  function renderStats(d) {
    setKpi("k-today", num(d.today_total));
    setKpi("k-unique", num(d.today_unique));
    setKpi("k-hour", num(d.last_hour_total));
    setKpi("k-conf", d.avg_confidence === null ? "—" : (d.avg_confidence * 100).toFixed(1) + "%");
    const kr = $("k-ring");
    if (kr) kr.style.strokeDashoffset = String(RING_LEN * (1 - Math.max(0, Math.min(1, d.avg_confidence || 0))));
    $("k-today-s").textContent = d.window_total ? num(d.window_total) + " in the last 24 h" : "no vehicles in 24 h";
    renderChart(d.buckets);
    renderTop(d.top);
  }

  function renderChart(buckets) {
    const svg = $("chart");
    const W = 480, H = 140, top = 14, bottom = 4;
    const n = buckets.length || 1;
    const max = Math.max(1, ...buckets.map((b) => b.count));
    const gap = 4;
    const bw = (W - gap * (n - 1)) / n;
    const frag = document.createDocumentFragment();
    // Gradients: amber for the current hour, a cool steel-blue fade for the rest, a glow for the line.
    const defs = svgEl("defs");
    const grad = (id, stops) => {
      const g = svgEl("linearGradient", { id, x1: 0, y1: 0, x2: 0, y2: 1 });
      for (const [o, cls] of stops) g.append(svgEl("stop", { offset: o, class: cls })); // colours in style.css
      defs.append(g);
    };
    grad("g-bar", [[0, "st-bar-t"], [1, "st-bar-b"]]);
    grad("g-now", [[0, "st-now-t"], [1, "st-now-b"]]);
    grad("g-area", [[0, "st-area-t"], [1, "st-area-b"]]);
    frag.append(defs);
    for (const f of [0.25, 0.5, 0.75, 1]) {
      const y = top + (H - top - bottom) * (1 - f);
      frag.append(svgEl("line", { class: "grid", x1: 0, x2: W, y1: y, y2: y }));
    }
    const nowTs = Date.now() / 1000;
    const pts = [];
    let peakI = -1;
    buckets.forEach((b, i) => {
      if (b.count && (peakI < 0 || b.count > buckets[peakI].count)) peakI = i;
    });
    buckets.forEach((b, i) => {
      const h = b.count ? Math.max(4, ((H - top - bottom) * b.count) / max) : 2;
      const isNow = nowTs >= b.start && nowTs < b.start + 3600;
      const x = i * (bw + gap);
      const y = H - bottom - h;
      pts.push([x + bw / 2, b.count ? y : H - bottom]);
      const r = svgEl("rect", {
        class: "bar" + (isNow ? " now" : "") + (b.count ? "" : " zero"),
        x: x.toFixed(2), y: y.toFixed(2), width: bw.toFixed(2), height: h.toFixed(2),
        rx: Math.min(4, bw / 2).toFixed(2),
      });

      const t = svgEl("title");
      const d = new Date(b.start * 1000);
      t.textContent = pad(d.getHours()) + ":00 – " + pad((d.getHours() + 1) % 24) + ":00 · " + b.count + " vehicle" + (b.count === 1 ? "" : "s");
      r.append(t);
      frag.append(r);
      if (i === peakI) {
        const lbl = svgEl("text", { class: "peak-lbl", x: (x + bw / 2).toFixed(2), y: Math.max(10, y - 4).toFixed(2), "text-anchor": "middle" });
        lbl.textContent = String(b.count);
        frag.append(lbl);
      }
    });
    if (pts.length > 1) {
      const line = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
      frag.append(svgEl("path", { class: "area", d: line + ` L${pts[pts.length - 1][0].toFixed(1)} ${H} L${pts[0][0].toFixed(1)} ${H} Z`, fill: "url(#g-area)" }));
      frag.append(svgEl("path", { class: "trend", d: line }));
    }
    svg.replaceChildren(frag);
    renderSpark(buckets.slice(-12));

    const axis = $("chart-axis");
    const labels = document.createDocumentFragment();
    const step = Math.max(1, Math.round(n / 4));
    for (let i = 0; i < n; i += step) {
      const d = new Date(buckets[i].start * 1000);
      labels.append(el("span", "", pad(d.getHours()) + ":00"));
    }
    labels.append(el("span", "", "now"));
    axis.replaceChildren(labels);

    const peak = buckets.reduce((a, b) => (b.count > a.count ? b : a), { count: 0, start: 0 });
    $("chart-peak").textContent = peak.count
      ? "peak " + peak.count + " @ " + pad(new Date(peak.start * 1000).getHours()) + ":00"
      : "no traffic";
  }

  function renderTop(top) {
    const ol = $("toplist");
    const frag = document.createDocumentFragment();
    if (!top.length) {
      frag.append(el("li", "muted", "No vehicles yet."));
    } else {
      top.forEach((t, i) => {
        const li = el("li");
        const b = el("button");
        b.type = "button";
        b.title = "Show all sightings of " + t.plate_display;
        const left = el("span", "top-left");
        left.append(el("span", "rank", String(i + 1).padStart(2, "0")), plateNode(t.plate_display, "sm"));
        const cnt = el("span", "count", t.count + (t.count === 1 ? " visit" : " visits"));
        cnt.append(el("small", "", " · " + fmtAgo(t.last_seen)));
        b.append(left, cnt);
        b.addEventListener("click", () => searchPlate(t.plate));
        li.append(b);
        frag.append(li);
      });
    }
    ol.replaceChildren(frag);
  }

  // ---- rendering: health ------------------------------------------------------------------------

  function setGauge(id, value, level, sub, barPct) {
    const g = $(id);
    g.className = "gauge" + (level ? " " + level : "");
    g.querySelector(".g-v").textContent = value;
    const s = g.querySelector(".g-s");
    if (s && sub !== undefined) s.textContent = sub;
    const bar = g.querySelector(".g-bar i");
    if (bar) bar.style.width = barPct === undefined || barPct === null ? "0" : Math.max(0, Math.min(100, barPct)) + "%";
  }

  const GAUGES = ["g-engine", "g-camera", "g-fps", "g-temp", "g-mem", "g-disk"];
  function setHealthSum(text, chip) {
    const c = $("health-sum");
    c.className = "chip " + chip;
    c.textContent = text;
  }
  // One plain-words line for the whole strip, so nobody has to read six tiles to know "all fine".
  function renderHealthSummary(running) {
    const levels = GAUGES.map((id) => $(id).classList);
    const bad = levels.filter((c) => c.contains("bad")).length;
    const warn = levels.filter((c) => c.contains("warn")).length;
    if (!running) setHealthSum("Engine not running", "chip-bad");
    else if (bad) setHealthSum(bad === 1 ? "1 problem" : bad + " problems", "chip-bad");
    else if (warn) setHealthSum(warn === 1 ? "1 thing to check" : warn + " things to check", "chip-warn");
    else setHealthSum("All systems normal", "chip-ok");
  }

  function renderHealth(h) {
    if (!h) return;
    state.lastStatusAt = Date.now();
    state.health = h;
    renderLive();
    // Pictures are back (new stream, engine restarted): don't sit out the rest of the error back-off.
    if (h.preview_fresh && live.errors > 0 && !live.loading) scheduleLive(0);
    const running = h.status !== "no_engine" && h.engine_ok;

    if (h.status === "no_engine") setGauge("g-engine", "Offline", "bad", "never started");
    else if (!h.engine_ok) setGauge("g-engine", "Stopped", "bad", "last beat " + fmtDuration(h.engine_age_s) + " ago");
    else if (h.last_error) setGauge("g-engine", "Running", "warn", "error: " + h.last_error);
    else setGauge("g-engine", "Running", "ok", num(h.frames || 0) + " frames processed");
    $("g-engine").title = h.last_error ? "Last error: " + h.last_error : "";

    if (!running || h.camera_ok === null || h.camera_ok === undefined) setGauge("g-camera", "Unknown", "", "engine not running");
    else if (h.source_state === "ended") setGauge("g-camera", "Video ended", "warn", "add a stream to continue");
    else if (h.source_state === "error") setGauge("g-camera", "Can't open", "bad", "check the stream address");
    else setGauge("g-camera", h.camera_ok ? "Online" : "No signal", h.camera_ok ? "ok" : "bad", h.camera_ok ? "receiving frames" : "reconnecting…");

    if (!running || h.fps === null || h.fps === undefined) setGauge("g-fps", "—", "", "frames per second");
    else setGauge("g-fps", h.fps.toFixed(1), h.fps < 1 ? "warn" : "ok", "frames per second");

    if (h.cpu_temp_c === null || h.cpu_temp_c === undefined) setGauge("g-temp", "n/a", "", undefined, 0);
    else {
      const t = h.cpu_temp_c;
      setGauge("g-temp", t.toFixed(1) + " °C", t >= 80 ? "bad" : t >= 70 ? "warn" : "ok", undefined, ((t - 30) / 55) * 100);
    }

    if (!running || h.rss_mb === null || h.rss_mb === undefined) setGauge("g-mem", "—", "", "by the engine");
    else setGauge("g-mem", Math.round(h.rss_mb) + " MB", h.rss_mb > 550 ? "warn" : "ok", "by the engine");

    if (h.disk_free_mb === null || h.disk_free_mb === undefined) setGauge("g-disk", "n/a", "", undefined, 0);
    else {
      const gb = h.disk_free_mb / 1000;
      const p = h.disk_free_pct;
      const txt = (gb >= 10 ? gb.toFixed(0) : gb.toFixed(1)) + " GB" + (p !== null ? " · " + Math.round(p) + "%" : "");
      setGauge("g-disk", txt, p !== null && p < 5 ? "bad" : p !== null && p < 15 ? "warn" : "ok", undefined, p);
    }
    document.querySelector(".brand-dot").classList.toggle("is-bad", !h.ok);
    renderHealthSummary(running);

    // The stream address is not in /health (it's open); fetch it when something changed.
    const key = [h.source_rev, h.source_state, running].join("|");
    if (key !== state.srcKey) { state.srcKey = key; loadSource(); }
  }

  // ---- stream input (Add stream) ----------------------------------------------------------------

  const STREAM_UI = {
    file: {
      label: "Video file path",
      placeholder: "/Users/you/Videos/gate-camera.mp4",
      hint: [
        "Full path of a video on the station (this Mac now, the Pi later). In Finder, right-click the file, hold ",
        ["kbd", "⌥ Option"], " and choose ", ["b", "Copy as Pathname"], ". MP4, MOV, AVI or MKV.",
      ],
      empty: "Enter the full path of a video file.",
    },
    rtsp: {
      label: "RTSP link",
      placeholder: "rtsp://user:password@192.168.1.64:554/stream1",
      hint: [
        "The camera's RTSP address, for example ",
        ["code", "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"],
        ". The password stays on the station and is never shown on this page.",
      ],
      empty: "Enter the camera's RTSP link.",
    },
  };
  let streamKind = "file";
  let streamBusy = false;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const capitalise = (t) => (t ? t.charAt(0).toUpperCase() + t.slice(1) : t);

  function srcLabel(d) {
    const v = d && d.current;
    if (!v) return d && d.pending ? "Starting…" : "No stream";
    switch (d.current_kind) {
      case "file": return v.split(/[\\/]/).pop();
      case "rtsp":
        try { const u = new URL(v); return "RTSP · " + u.hostname + (u.port ? ":" + u.port : ""); } catch (_) { return "RTSP camera"; }
      case "webcam": return "Webcam " + v;
      case "picamera": return "Pi camera";
      default: return v;
    }
  }

  // -> [status text, chip class, pill class]
  function srcStatus(d) {
    if (!d) return ["—", "chip-muted", ""];
    if (d.pending) return d.engine_running ? ["Switching…", "chip-signal", "is-pending"] : ["Starts with engine", "chip-muted", "is-pending"];
    if (!d.engine_running) return ["Engine off", "chip-muted", ""];
    if (d.state === "ended") return ["Video ended", "chip-warn", "is-warn"];
    if (d.state === "error") return ["Can't open", "chip-bad", "is-bad"];
    if (d.camera_ok === false) return ["No signal", "chip-bad", "is-bad"];
    return ["Live", "chip-ok", "is-ok"];
  }

  function renderSource(d) {
    state.src = d;
    renderLive();
    const [txt, chip, pill] = srcStatus(d);
    const p = $("streampill");
    p.className = "streampill" + (pill ? " " + pill : "");
    $("sp-v").textContent = srcLabel(d);
    p.title = (d && d.current ? d.current : "No stream") + " · " + txt + " — click to change";
    $("s-now-v").textContent = d && d.current ? d.current : "Nothing yet";
    $("s-now-v").title = (d && d.current) || "";
    const c = $("s-now-state");
    c.className = "chip " + chip;
    c.textContent = txt;
  }

  async function loadSource() {
    try {
      renderSource((await api("/api/v1/source")).data);
      return state.src;
    } catch (e) {
      if (e instanceof AuthError) askToken();
      return null;
    }
  }

  function hintNodes(parts) {
    const frag = document.createDocumentFragment();
    for (const part of parts) frag.append(Array.isArray(part) ? el(part[0], null, part[1]) : part);
    return frag;
  }

  function setStreamKind(kind) {
    streamKind = kind;
    const ui = STREAM_UI[kind];
    for (const b of document.querySelectorAll(".seg-b")) b.setAttribute("aria-checked", String(b.dataset.kind === kind));
    $("s-label").textContent = ui.label;
    $("s-input").placeholder = ui.placeholder;
    $("s-hint").replaceChildren(hintNodes(ui.hint));
  }

  function streamMsg(kind, text) {
    $("s-err").hidden = kind !== "err";
    $("s-ok").hidden = kind !== "ok";
    if (kind === "err") $("s-err").textContent = text;
    if (kind === "ok") $("s-ok").textContent = text;
    $("s-input").setAttribute("aria-invalid", String(kind === "err"));
  }

  function setStreamBusy(busy) {
    streamBusy = busy;
    $("s-submit").disabled = busy;
    $("s-default").disabled = busy;
    $("s-input").readOnly = busy;
    $("s-submit").textContent = busy ? "Starting…" : "Start stream";
  }

  function openStream() {
    if (state.readOnly) return;
    const d = state.src;
    const dlg = $("streamdlg");
    // A video that just ended can be replayed with one click: prefill it (selected, so typing replaces it).
    $("s-input").value = d && d.current_kind === "file" && d.current ? d.current : "";
    setStreamKind(d && d.current_kind === "rtsp" ? "rtsp" : "file");
    streamMsg(null);
    setStreamBusy(false);
    if (typeof dlg.showModal === "function") { if (!dlg.open) dlg.showModal(); } else dlg.setAttribute("open", "");
    $("s-input").focus();
    $("s-input").select();
    loadSource();
  }
  function closeStream() {
    const dlg = $("streamdlg");
    if (typeof dlg.close === "function") { if (dlg.open) dlg.close(); } else dlg.removeAttribute("open");
  }

  // Poll until the engine reports it applied request `rev` (and, for cameras, got a first frame).
  async function waitApplied(rev) {
    const until = Date.now() + 9000;
    let d = null;
    while (Date.now() < until) {
      d = await loadSource();
      if (!d || !d.engine_running) return d;
      const applied = !d.pending && (d.requested_rev === null || d.requested_rev >= rev);
      if (applied && (d.state !== "running" || d.camera_ok)) return d;
      await sleep(600);
    }
    return d;
  }

  async function submitStream(raw) {
    if (streamBusy) return;
    const value = raw.trim();
    if (raw !== "" && !value) { streamMsg("err", STREAM_UI[streamKind].empty); $("s-input").focus(); return; }
    setStreamBusy(true);
    streamMsg(null);
    try {
      const res = await adminFetch("/api/v1/source", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ source: value }),
      });
      if (res.status === 401) { closeStream(); askToken(); return; }
      let body = null;
      try { body = await res.json(); } catch (_) { /* non-JSON */ }
      if (!res.ok) {
        const m = body && body.error && body.error.message;
        streamMsg("err", m ? capitalise(m) + "." : "Could not save the stream (HTTP " + res.status + ").");
        $("s-input").focus();
        return;
      }
      const target = body.data.source;
      streamMsg("ok", "Saved. Switching to " + target + " …");
      const d = await waitApplied(body.data.rev);
      if (!d) streamMsg("ok", "Saved. The dashboard lost contact with the station; it will switch when it can.");
      else if (!d.engine_running) streamMsg("ok", "Saved. The engine isn't running; it will open this stream as soon as it starts.");
      else if (d.state === "error") streamMsg("err", "The engine couldn't open this stream" + (d.error ? ": " + d.error : "."));
      else if (d.pending) streamMsg("ok", "Saved. The engine will switch within a few seconds.");
      else if (d.camera_ok === false) streamMsg("err", "Switched, but no picture yet from " + target + ". Check the address, user name and password — the engine keeps retrying.");
      else {
        streamMsg("ok", "Now reading " + d.current + ".");
        setTimeout(closeStream, 900);
      }
    } catch (e) {
      streamMsg("err", e.adminCancelled ? e.message : "Can't reach the station. Check the connection and try again.");
    } finally {
      setStreamBusy(false);
      loadHealth();
    }
  }

  $("streampill").addEventListener("click", openStream);
  $("addstream").addEventListener("click", openStream);
  $("s-close").addEventListener("click", closeStream);
  $("s-cancel").addEventListener("click", closeStream);
  for (const b of document.querySelectorAll(".seg-b")) {
    b.addEventListener("click", () => { setStreamKind(b.dataset.kind); $("s-input").focus(); });
    b.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        const next = b.dataset.kind === "file" ? "rtsp" : "file";
        setStreamKind(next);
        $("s-type-" + next).focus();
      }
    });
  }
  $("s-input").addEventListener("input", () => {
    const v = $("s-input").value.trim().toLowerCase();
    if (v.startsWith("rtsp")) { if (streamKind !== "rtsp") setStreamKind("rtsp"); } else if (/^['"]?[/~]/.test(v) && streamKind !== "file") setStreamKind("file");
    if (!$("s-err").hidden) streamMsg(null);
  });
  $("streamform").addEventListener("submit", (e) => { e.preventDefault(); submitStream($("s-input").value); });
  $("s-default").addEventListener("click", () => submitStream(""));

  // ---- settings: send plates to the client's API --------------------------------------------------

  let setBusy = false;
  let setTimer = null;

  function setMsg(kind, text) {
    $("set-err").hidden = kind !== "err";
    $("set-ok").hidden = kind !== "ok";
    if (kind === "err") $("set-err").textContent = text;
    if (kind === "ok") $("set-ok").textContent = text;
    if (!kind) showResp(null);
  }

  // The API's answer to Save / Send test, shown under the buttons.
  function showResp(r) {
    const box = $("set-resp");
    if (!r) { box.hidden = true; return; }
    box.hidden = false;
    box.className = "set-resp " + (r.ok ? "is-ok" : "is-bad");
    $("set-resp-k").textContent = "API response" + (r.status ? " · HTTP " + r.status : "") + " · " +
      (r.ok ? "accepted" : r.message);
    $("set-resp-v").textContent = r.reply || "(empty answer)";
  }

  async function sendTest(f) {
    const r = await pushCall("POST", "/api/v1/push/test", f);
    showResp(r);
    return r;
  }

  async function pushCall(method, path, body) {
    const res = await adminFetch(path, {
      method,
      cache: "no-store",
      headers: body ? { "Content-Type": "application/json", Accept: "application/json" } : { Accept: "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 401) throw new AuthError("unauthorized");
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) {
      const m = data && data.error && data.error.message;
      throw new Error(m ? capitalise(m) + "." : "The station answered HTTP " + res.status + ".");
    }
    return data.data;
  }

  function renderPushStatus(st, enabled) {
    let text, chip, cls;
    if (!enabled) { text = "Sending is switched off."; chip = "Off"; cls = "chip-muted"; }
    else if (!st.engine_running) { text = "The engine isn't running: plates are sent once it runs again."; chip = "Paused"; cls = "chip-warn"; }
    else if (st.last_error) {
      text = st.last_error + (st.last_error_ts ? " (" + fmtAgo(st.last_error_ts) + ")" : "") + " · " + st.waiting + " waiting, retrying";
      chip = "Problem"; cls = "chip-bad";
    } else {
      text = st.sent_total + " sent" + (st.last_ok_ts ? ", last " + fmtAgo(st.last_ok_ts) : "") + " · " + st.waiting + " waiting";
      if (st.skipped_total) text += " · " + st.skipped_total + " refused by the server";
      chip = "On"; cls = "chip-ok";
    }
    $("set-reply").hidden = !(enabled && st.last_reply_ts);
    if (st.last_reply_ts) {
      $("set-reply-when").textContent = "(real plates · " + (st.last_reply_status ? "HTTP " + st.last_reply_status + " · " : "") +
        fmtAgo(st.last_reply_ts) + ")";
      $("set-reply-v").textContent = st.last_reply || "(empty answer)";
    }
    $("set-now").textContent = text;
    $("set-now").title = text;
    $("set-chip").textContent = chip;
    $("set-chip").className = "chip " + cls;
  }

  function fillPush(d, fields) {
    const s = d.settings;
    if (fields) {
      $("set-enabled").checked = s.enabled;
      $("set-url").value = s.url;
      $("set-hname").value = s.header_name;
      $("set-hvalue").value = "";
      $("set-device").value = s.device_id;
      $("set-images").checked = s.include_images;
    }
    $("set-hvalue").placeholder = s.header_value_set ? "Saved " + (s.header_value_hint || "") + " (empty = keep)" : "Bearer abc123…";
    $("set-device").placeholder = s.device_id_default;
    renderPushStatus(d.status, s.enabled);
  }

  const pushForm = () => ({
    enabled: $("set-enabled").checked,
    url: $("set-url").value.trim(),
    header_name: $("set-hname").value.trim(),
    header_value: $("set-hvalue").value.trim() || null,
    device_id: $("set-device").value.trim(),
    include_images: $("set-images").checked,
  });

  async function refreshPush(fields) {
    try { fillPush(await pushCall("GET", "/api/v1/push"), fields); }
    catch (e) {
      if (e instanceof AuthError) { closeSettings(); askToken(); return; }
      if (e.adminCancelled) { closeSettings(); return; }  // no password: no settings (and no re-asking)
      if (fields) setMsg("err", e.message || "Can't reach the station.");
    }
  }

  function setBusyUI(busy, label) {
    setBusy = busy;
    for (const id of ["set-save", "set-test"]) $(id).disabled = busy;
    $("set-save").textContent = busy && label === "save" ? "Saving…" : "Save";
    $("set-test").textContent = busy && label === "test" ? "Sending…" : "Send test";
  }

  function openSettings() {
    if (state.readOnly) return;
    const dlg = $("setdlg");
    setMsg(null);
    setBusyUI(false);
    $("set-now").textContent = "Loading…";
    if (typeof dlg.showModal === "function") { if (!dlg.open) dlg.showModal(); } else dlg.setAttribute("open", "");
    refreshPush(true);
    clearInterval(setTimer);
    setTimer = setInterval(() => refreshPush(false), 3000);
  }
  function closeSettings() {
    clearInterval(setTimer);
    const dlg = $("setdlg");
    if (typeof dlg.close === "function") { if (dlg.open) dlg.close(); } else dlg.removeAttribute("open");
  }

  async function savePush() {
    if (setBusy) return;
    const f = pushForm();
    if (f.enabled && !f.url) { setMsg("err", "Enter the client's API address to switch sending on."); $("set-url").focus(); return; }
    setBusyUI(true, "save");
    setMsg(null);
    try {
      const d = await pushCall("PUT", "/api/v1/push", f);
      fillPush(d, true);
      if (!d.settings.enabled || !d.settings.url) {
        setMsg("ok", "Saved. Sending is off.");
        return;
      }
      // Check the API right away: one test plate (marked "test": true), and show what it answered.
      $("set-save").textContent = "Checking the API…";
      const r = await sendTest(pushForm());
      if (r.ok) setMsg("ok", "Saved. The API accepted a test plate, so new plates go to " + d.settings.url + " within a few seconds.");
      else setMsg("err", "Saved, but the API did not accept the test plate (" + r.message + "). Plates wait and are sent again; check the address and key.");
      showResp(r);
    } catch (e) {
      if (e instanceof AuthError) { closeSettings(); askToken(); return; }
      setMsg("err", e.message || "Can't reach the station. Try again.");
    } finally {
      setBusyUI(false);
    }
  }

  async function testPush() {
    if (setBusy) return;
    const f = pushForm();
    if (!f.url) { setMsg("err", "Enter the client's API address first."); $("set-url").focus(); return; }
    setBusyUI(true, "test");
    setMsg(null);
    try {
      const r = await sendTest(f);
      if (r.ok) setMsg("ok", "Test plate sent: the API accepted it.");
      else setMsg("err", "Test failed: " + r.message + ".");
      showResp(r);
    } catch (e) {
      if (e instanceof AuthError) { closeSettings(); askToken(); return; }
      setMsg("err", e.message || "Can't reach the station. Try again.");
    } finally {
      setBusyUI(false);
    }
  }

  $("settingsbtn").addEventListener("click", openSettings);
  $("set-close").addEventListener("click", closeSettings);
  $("set-cancel").addEventListener("click", closeSettings);
  $("setdlg").addEventListener("close", () => clearInterval(setTimer));
  $("set-test").addEventListener("click", testPush);
  $("setform").addEventListener("submit", (e) => { e.preventDefault(); savePush(); });
  $("setform").addEventListener("input", () => { if (!$("set-err").hidden) setMsg(null); });

  // ---- drawer -----------------------------------------------------------------------------------

  let drawerEvent = null;
  function openDrawer(ev) {
    drawerEvent = ev;
    const dlg = $("drawer");
    $("d-eyebrow").textContent = "Event #" + ev.id + " · " + fmtDay(ev.last_seen) + " " + fmtTime(ev.last_seen);
    setPlate($("d-plate"), displayOf(ev));
    $("d-title").textContent = "Details for " + displayOf(ev);

    const snap = $("d-snap");
    if (ev.snapshot_url) {
      snap.src = withToken(ev.snapshot_url);
      snap.alt = "Snapshot of " + displayOf(ev);
      snap.hidden = false;
      $("d-nosnap").hidden = true;
      $("d-open-snap").href = withToken(ev.snapshot_url);
      $("d-open-snap").hidden = false;
    } else {
      snap.hidden = true;
      snap.removeAttribute("src");
      $("d-nosnap").hidden = false;
      $("d-open-snap").hidden = true;
    }
    const crop = $("d-crop");
    if (ev.crop_url) {
      crop.src = withToken(ev.crop_url);
      crop.alt = "Plate crop " + displayOf(ev);
      crop.hidden = false;
      $("d-open-crop").href = withToken(ev.crop_url);
      $("d-open-crop").hidden = false;
    } else {
      crop.hidden = true;
      crop.removeAttribute("src");
      $("d-open-crop").hidden = true;
    }

    const fields = [
      ["Plate number", ev.plate],
      ["Plate type", kindLabel(ev.kind)],
      ["HSRP", hsrpLabel(ev.hsrp)],
      ["Confidence", pct(ev.confidence)],
      ["Seen in", ev.votes + " frames"],
      ["First seen", fmtTime(ev.first_seen) + " · " + fmtDate(ev.first_seen)],
      ["Confirmed", fmtTime(ev.last_seen) + " · " + fmtDate(ev.last_seen)],
      ["Time to confirm", fmtDuration(ev.last_seen - ev.first_seen)],
      ["Track ID", String(ev.track_id)],
    ];
    const dl = $("d-fields");
    const frag = document.createDocumentFragment();
    for (const [k, v] of fields) {
      const d = el("div");
      d.append(el("dt", "", k), el("dd", "", v));
      frag.append(d);
    }
    dl.replaceChildren(frag);

    if (typeof dlg.showModal === "function") { if (!dlg.open) dlg.showModal(); } else dlg.setAttribute("open", "");
  }
  function closeDrawer() {
    const dlg = $("drawer");
    if (dlg.open) dlg.close();
  }
  $("d-close").addEventListener("click", closeDrawer);
  $("drawer").addEventListener("click", (e) => { if (e.target === $("drawer")) closeDrawer(); }); // backdrop
  $("d-search").addEventListener("click", () => {
    if (!drawerEvent) return;
    closeDrawer();
    searchPlate(drawerEvent.plate);
  });

  // The Clipboard API needs https or localhost; on the Pi's plain-http LAN address fall back to execCommand.
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_) {
      const ta = el("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.className = "sr";
      $("drawer").append(ta); // inside the open dialog, so the modal focus trap allows selecting it
      ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch (_) { /* unsupported */ }
      ta.remove();
      return ok;
    }
  }
  let copyTimer = null;
  $("d-copy").addEventListener("click", async () => {
    if (!drawerEvent) return;
    const label = $("d-copy").querySelector("span");
    label.textContent = (await copyText(drawerEvent.plate)) ? "Copied" : "Copy failed";
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => { label.textContent = "Copy"; }, 1600);
  });

  // ---- data loading -----------------------------------------------------------------------------

  async function loadList() {
    const seq = ++state.loadSeq;
    try {
      const body = await api("/api/v1/plates", { ...filterParams(), page: state.page, per_page: PER_PAGE });
      if (seq !== state.loadSeq) return;
      hideMsg();
      state.rows = body.data;
      state.total = body.meta.total;
      state.totalPages = body.meta.total_pages;
      if (state.page > 1 && state.page > state.totalPages && state.totalPages > 0) {
        state.page = state.totalPages;
        return loadList();
      }
      if (liveListing()) state.pendingNew = 0;
      for (const ev of state.rows) state.maxId = Math.max(state.maxId, ev.id);
      renderRows(null);
    } catch (e) {
      if (e instanceof AuthError) return askToken();
      showMsg("Could not load plates: " + e.message);
    }
  }

  async function loadLatest() {
    try {
      const body = await api("/api/v1/plates", { page: 1, per_page: RECENT_N });
      const key = (list) => list.map((e) => e.id).join(",");
      if (key(body.data) !== key(state.recent)) {
        const before = state.recent.length ? state.recent[0].id : null;
        state.recent = body.data.slice(0, RECENT_N);
        renderRecent(before !== null && state.recent.length && state.recent[0].id > before ? state.recent[0].id : null);
      }
      if (body.data.length) {
        const ev = body.data[0];
        state.maxId = Math.max(state.maxId, ev.id);
        if (!state.latest || ev.id !== state.latest.id) {
          const isNew = state.latest !== null && ev.id > state.latest.id;
          state.latest = ev;
          renderCapture(isNew);
        }
      }
    } catch (e) {
      if (e instanceof AuthError) askToken();
    }
  }

  async function loadHealth() {
    try {
      renderHealth(await api("/health"));
    } catch (e) {
      setGauge("g-engine", "No server", "bad", "dashboard can't reach the station");
      setHealthSum("Station unreachable", "chip-bad");
      state.health = { status: "no_engine", engine_ok: false }; // live badge: "Engine off"
      renderLive();
    }
  }

  async function loadStats() {
    try {
      const body = await api("/api/v1/stats", { hours: 24 });
      renderStats(body.data);
    } catch (e) {
      if (e instanceof AuthError) askToken();
    }
  }
  function refreshStatsSoon() {
    clearTimeout(state.statsTimer);
    state.statsTimer = setTimeout(loadStats, 1200);
  }

  // Footer: who the device is licensed to and until when; red when there is no valid licence, amber
  // in the last 30 days.
  function renderLicence(l) {
    const box = $("licence");
    if (!l) {
      box.hidden = state.readOnly || !lic.canActivate;
      box.classList.remove("is-warn");
      box.classList.add("is-bad");
      box.textContent = "Not activated · click to activate";
      return;
    }
    box.hidden = false;
    box.classList.toggle("is-bad", !l.ok);
    box.classList.toggle("is-warn", !!l.ok && typeof l.days_left === "number" && l.days_left <= 30);
    if (!l.ok && /^no licence file/.test(l.error || "")) {
      box.textContent = "Not activated · click to activate (the engine is stopped)";
    } else if (!l.ok) {
      box.textContent = "Licence problem: " + (l.error || "not valid") + " (the engine is stopped)";
    } else if (l.expires) {
      box.textContent = "Licensed to " + l.customer + " · valid until " + l.expires +
        (l.days_left <= 30 ? " (" + l.days_left + " day" + (l.days_left === 1 ? "" : "s") + " left)" : "");
    } else {
      box.textContent = "Licensed to " + l.customer;
    }
    box.title = (l.licence_id ? "Licence " + l.licence_id + " · " : "") + "click for licence and activation";
  }

  // ---- licence activation ----------------------------------------------------------------------
  // Opens by itself when this device has no valid licence (not on a view-only dashboard).

  const lic = { canActivate: false, autoOpened: false, busy: false, mid: null, timer: null };

  function licMsg(kind, text) {
    $("lic-err").hidden = kind !== "err";
    $("lic-ok").hidden = kind !== "ok";
    if (kind === "err") $("lic-err").textContent = text;
    if (kind === "ok") $("lic-ok").textContent = text;
  }

  function fillLicence(d) {
    lic.canActivate = !!d.can_activate;
    $("lic-admin").hidden = !d.admin_required;
    lic.mid = d.machine_id;
    $("lic-mid").textContent = d.machine_id || "cannot be read on this device";
    const st = d.status;
    let text, chip, cls;
    if (!st) { text = "No licence on this device yet."; chip = "Not activated"; cls = "chip-bad"; }
    else if (!st.ok && /^no licence file/.test(st.error || "")) {
      text = "This device is not activated yet."; chip = "Not activated"; cls = "chip-bad";
    } else if (!st.ok) { text = st.error || "The licence is not valid."; chip = "Not valid"; cls = "chip-bad"; }
    else {
      text = "Licensed to " + st.customer + (st.expires ? " · valid until " + st.expires : " · no end date");
      const soon = typeof st.days_left === "number" && st.days_left <= 30;
      chip = soon ? st.days_left + " days left" : "Active";
      cls = soon ? "chip-warn" : "chip-ok";
    }
    $("lic-now").textContent = text;
    $("lic-now").title = text;
    $("lic-chip").textContent = chip;
    $("lic-chip").className = "chip " + cls;
    $("lic-title").textContent = st && st.ok ? "Licence" : "Activate this device";
    $("lic-activate").textContent = st && st.ok ? "Replace licence" : "Activate";
    $("lic-activate").hidden = !lic.canActivate;
    $("lic-key").disabled = !lic.canActivate;
    renderLicence(st);
    return st;
  }

  async function loadLicence(autoOpen) {
    try {
      const d = (await api("/api/v1/licence")).data;
      const st = fillLicence(d);
      if (autoOpen && !lic.autoOpened && lic.canActivate && !state.readOnly && (!st || !st.ok)) {
        lic.autoOpened = true;
        openLicence();
      }
    } catch (e) {
      if (e instanceof AuthError) askToken();
    }
  }

  function openLicence() {
    const dlg = $("licdlg");
    licMsg(null);
    if (typeof dlg.showModal === "function") { if (!dlg.open) dlg.showModal(); } else dlg.setAttribute("open", "");
    loadLicence(false);
    clearInterval(lic.timer);
    lic.timer = setInterval(() => loadLicence(false), 4000);
    if (lic.canActivate) $("lic-key").focus();
  }
  function closeLicence() {
    clearInterval(lic.timer);
    const dlg = $("licdlg");
    if (typeof dlg.close === "function") { if (dlg.open) dlg.close(); } else dlg.removeAttribute("open");
  }

  async function activate() {
    if (lic.busy || !lic.canActivate) return;
    const key = $("lic-key").value.trim();
    if (!key) { licMsg("err", "Paste the licence key first."); $("lic-key").focus(); return; }
    lic.busy = true;
    licMsg(null);
    $("lic-activate").disabled = true;
    $("lic-activate").textContent = "Checking…";
    try {
      const d = await pushCall("POST", "/api/v1/licence", { key });
      const st = fillLicence(d);
      $("lic-key").value = "";
      licMsg("ok", "Activated" + (st && st.customer ? " for " + st.customer : "") +
        ". The engine starts within a few seconds.");
    } catch (e) {
      if (e instanceof AuthError) { closeLicence(); askToken(); return; }
      licMsg("err", e.message || "Can't reach the station. Try again.");
    } finally {
      lic.busy = false;
      $("lic-activate").disabled = false;
      if ($("lic-activate").textContent === "Checking…") $("lic-activate").textContent = "Activate";
    }
  }

  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch (_) { /* http page */ }
    const t = document.createElement("textarea");
    t.value = text; document.body.append(t); t.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch (_) { /* no clipboard */ }
    t.remove();
    return ok;
  }

  // The footer follows the engine's hourly licence check (and activation from another browser).
  setInterval(() => { if (!$("licdlg").open && !document.hidden) loadLicence(false); }, 60000);
  $("licence").addEventListener("click", openLicence);
  $("lic-close").addEventListener("click", closeLicence);
  $("lic-cancel").addEventListener("click", closeLicence);
  $("licdlg").addEventListener("close", () => clearInterval(lic.timer));
  $("licform").addEventListener("submit", (e) => { e.preventDefault(); activate(); });
  $("lic-key").addEventListener("input", () => { if (!$("lic-err").hidden) licMsg(null); });
  $("lic-copy").addEventListener("click", async () => {
    if (!lic.mid) return;
    const ok = await copyText(lic.mid);
    $("lic-copy").textContent = ok ? "Copied" : "Select and copy";
    setTimeout(() => { $("lic-copy").textContent = "Copy"; }, 1600);
  });

  async function loadInfo() {
    try {
      const d = (await api("/api/v1/info")).data;
      $("station").textContent = d.station;
      state.readOnly = !!d.read_only;
      document.documentElement.classList.toggle("is-readonly", state.readOnly);
      $("streampill").disabled = state.readOnly;
      if (state.readOnly) $("streampill").title = "Camera input (view-only dashboard)";
      if (typeof d.preview_interval_s === "number" && d.preview_interval_s !== live.interval) {
        live.interval = d.preview_interval_s;
        renderLive();
        scheduleLive(0);
      }
      document.title = "ANPR Console · " + d.station + " · LogicClutch Software LLP";
      $("rules").textContent =
        "Confirmed = at least " + d.min_votes + " agreeing frames, average confidence ≥ " +
        Math.round(d.min_confidence * 100) + "% and a strict Indian format check · images kept " +
        d.retention_days + " days.";
      renderLicence(d.licence);
      loadLicence(true);
    } catch (e) {
      if (e instanceof AuthError) askToken();
    }
  }

  function loadAll() {
    loadInfo();
    loadLatest();
    loadList();
    loadHealth();
    loadStats();
  }

  // ---- live events ------------------------------------------------------------------------------

  function onLivePlate(ev) {
    if (state.rows.some((r) => r.id === ev.id) && state.latest && ev.id <= state.latest.id) return;
    state.maxId = Math.max(state.maxId, ev.id);
    if (!state.latest || ev.id > state.latest.id) {
      state.latest = ev;
      renderCapture(true);
    }
    if (addRecent(ev)) announce(ev);
    refreshStatsSoon();
    if (liveListing()) {
      if (state.rows.some((r) => r.id === ev.id)) return;
      state.rows.unshift(ev);
      state.rows.sort((a, b) => b.last_seen - a.last_seen || b.id - a.id);
      if (state.rows.length > PER_PAGE) state.rows.length = PER_PAGE;
      state.total += 1;
      state.totalPages = Math.ceil(state.total / PER_PAGE);
      renderRows(new Set([ev.id]));
    } else {
      state.pendingNew += 1;
      renderPager();
    }
  }

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const params = {};
    if (state.maxId > 0) params.since_id = state.maxId;
    return proto + "//" + location.host + withToken("/ws/plates", params);
  }

  function setLink(text, cls) {
    const l = $("link");
    l.querySelector(".link-text").textContent = text;
    l.className = "link " + cls;
  }

  function connect() {
    clearTimeout(state.wsTimer);
    if (state.authBlocked) return;
    let ws;
    try { ws = new WebSocket(wsUrl()); } catch (_) { return scheduleReconnect(); }
    state.ws = ws;
    let opened = false;
    ws.onopen = () => {
      opened = true;
      state.wsOpen = true;
      state.wsRetry = 0;
      setLink("Live", "link-live");
    };
    ws.onmessage = (m) => {
      let msg;
      try { msg = JSON.parse(m.data); } catch (_) { return; }
      if (msg.type === "plate" && msg.data) onLivePlate(msg.data);
      else if (msg.type === "status") renderHealth(msg.data);
    };
    ws.onclose = (e) => {
      if (state.ws !== ws) return;
      state.ws = null;
      state.wsOpen = false;
      if (e.code === 1008) {
        setLink("Locked", "link-bad");
        loadHealth();
        api("/api/v1/plates", { per_page: 1 }).catch((err) => { if (err instanceof AuthError) askToken(); });
        return;
      }
      setLink(opened ? "Reconnecting" : "Polling", "link-warn");
      scheduleReconnect();
    };
    ws.onerror = () => { /* onclose follows */ };
  }

  function scheduleReconnect() {
    clearTimeout(state.wsTimer);
    const base = Math.min(30000, 1000 * Math.pow(2, state.wsRetry));
    state.wsRetry = Math.min(state.wsRetry + 1, 6);
    state.wsTimer = setTimeout(connect, base / 2 + (Math.random() * base) / 2);
  }

  function reconnectNow() {
    state.wsRetry = 0;
    if (state.ws) {
      const old = state.ws;
      state.ws = null;
      try { old.close(); } catch (_) { /* ignore */ }
    }
    connect();
  }

  async function pollFallback() {
    if (state.wsOpen || state.authBlocked || document.hidden) return;
    const before = state.latest ? state.latest.id : 0;
    await loadLatest();
    if (state.latest && state.latest.id > before && before > 0) {
      refreshStatsSoon();
      announce(state.latest);
      if (liveListing()) {
        const known = new Set(state.rows.map((r) => r.id));
        await loadList();
        renderRows(new Set(state.rows.filter((r) => !known.has(r.id)).map((r) => r.id)));
      } else {
        state.pendingNew += 1;
        renderPager();
      }
    }
  }

  // ---- UI wiring --------------------------------------------------------------------------------

  function setRange(r) {
    state.range = r;
    for (const b of document.querySelectorAll(".rg-b")) {
      const on = b.dataset.range === r;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on || (r === "custom" && b.dataset.range === "all") ? 0 : -1;
    }
  }
  function readFilters() {
    state.q = $("q").value.trim().toUpperCase();
    state.from = $("from").value;
    state.to = $("to").value;
    if (state.from && state.to && state.from > state.to) {
      [state.from, state.to] = [state.to, state.from];
      $("from").value = state.from;
      $("to").value = state.to;
    }
    // Typed dates win over the quick range buttons.
    if (state.from || state.to) setRange("custom");
    else if (state.range === "custom") setRange("all");
  }
  function applyFilters() { readFilters(); state.page = 1; loadList(); }
  function resetFilters() {
    $("q").value = ""; $("from").value = ""; $("to").value = "";
    setRange("all");
    applyFilters();
  }
  function searchPlate(plate) {
    $("q").value = plate;
    $("from").value = "";
    $("to").value = "";
    setRange("all");
    applyFilters();
    scrollToLog();
  }
  function scrollToLog() {
    $("log").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  $("filters").addEventListener("submit", (e) => { e.preventDefault(); applyFilters(); });
  let typing = null;
  $("q").addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(applyFilters, 300); });
  for (const id of ["from", "to"]) $(id).addEventListener("change", applyFilters);
  $("clear").addEventListener("click", resetFilters);
  for (const b of document.querySelectorAll(".rg-b")) {
    b.addEventListener("click", () => {
      $("from").value = ""; $("to").value = "";
      setRange(b.dataset.range);
      applyFilters();
    });
    b.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const all = [...document.querySelectorAll(".rg-b")];
      const next = all[(all.indexOf(b) + (e.key === "ArrowRight" ? 1 : all.length - 1)) % all.length];
      next.click();
      next.focus();
    });
  }
  $("recent-all").addEventListener("click", () => { resetFilters(); scrollToLog(); });
  $("cap-cta").addEventListener("click", openStream);

  // ---- theme ------------------------------------------------------------------------------------

  const curTheme = () => (document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark");
  function renderTheme() {
    const t = curTheme();
    $("theme").title = t === "dark" ? "Switch to light theme" : "Switch to dark theme";
    $("theme-color").setAttribute("content", t === "dark" ? "#0a0d12" : "#ffffff");
  }
  $("theme").addEventListener("click", () => {
    const t = curTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem(THEME_KEY, t); } catch (_) { /* private mode */ }
    renderTheme();
  });
  $("prev").addEventListener("click", () => { if (state.page > 1) { state.page -= 1; loadList(); scrollToLog(); } });
  $("next").addEventListener("click", () => { if (state.page < state.totalPages) { state.page += 1; loadList(); scrollToLog(); } });
  $("newbadge").addEventListener("click", () => {
    state.pendingNew = 0;
    resetFilters();
    scrollToLog();
  });

  document.addEventListener("keydown", (e) => {
    const typingInField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement && document.activeElement.tagName);
    if (e.key === "/" && !typingInField && !$("drawer").open && !$("streamdlg").open && !$("setdlg").open && !$("licdlg").open) {
      e.preventDefault();
      $("q").focus();
      $("q").select();
    } else if (e.key === "Escape" && document.activeElement === $("q") && $("q").value) {
      $("q").value = "";
      applyFilters();
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !state.wsOpen) { reconnectNow(); loadAll(); }
    scheduleLive(0); // hidden: stops the live picture; visible again: fresh picture right away
  });

  // ---- start ------------------------------------------------------------------------------------

  $("changetoken").hidden = !memToken;
  renderTheme();
  renderSound();
  setRange("all");
  let savedView = "live";
  try { savedView = localStorage.getItem(VIEW_KEY) || "live"; } catch (_) { /* private mode */ }
  setView(savedView, false);
  renderPager();
  tickAges();
  setInterval(tickAges, 1000);
  setInterval(() => {
    if (Date.now() - state.lastStatusAt >= HEALTH_FALLBACK_MS - 500) loadHealth();
  }, HEALTH_FALLBACK_MS);
  setInterval(pollFallback, POLL_FALLBACK_MS);
  setInterval(() => { if (!document.hidden) loadStats(); }, STATS_REFRESH_MS);
  // Re-label "Today/Yesterday" and chart hours after midnight or long idle.
  // "Last hour" is a moving window: reload it so plates older than an hour drop out.
  setInterval(() => {
    if (document.hidden) return;
    if (state.range === "hour") loadList();
    else if (state.rows.length) renderRows(null);
  }, 5 * 60000);
  loadAll();
  if ("WebSocket" in window) connect(); else setLink("Polling", "link-warn");
})();
