/* Plate Labeller — plain JS, no dependencies. DOM text is only ever set with textContent. */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const el = {
    labeler: $("labeler"), fStatus: $("f-status"), fSource: $("f-source"), fLines: $("f-lines"),
    reload: $("reload"), prev: $("prev"), next: $("next"), pos: $("pos"),
    empty: $("empty"), work: $("work"), imgMain: $("img-main"), imgRaw: $("img-raw"), prep: $("prep"),
    plate: $("plate"), save: $("save"), hint: $("hint"),
    confirm: $("confirm"), confirmMsg: $("confirm-msg"), confirmYes: $("confirm-yes"), confirmNo: $("confirm-no"),
    trackApply: $("track-apply"), unreadable: $("unreadable"), skip: $("skip"), reset: $("reset"),
    mOcr: $("m-ocr"), mStatus: $("m-status"), mSource: $("m-source"), mVideo: $("m-video"),
    mTrack: $("m-track"), mSize: $("m-size"), trackTitle: $("track-title"), track: $("track"),
    pct: $("pct"), toast: $("toast"),
  };
  const STATUSES = ["unverified", "verified", "unreadable", "skip"];

  let rows = [];
  let idx = 0;
  let busy = false;
  let pending = null; // action waiting for "Save anyway"
  let validateTimer = 0;
  let validateSeq = 0;
  let toastTimer = 0;

  // ---- helpers -------------------------------------------------------------------------------

  function clean(text) {
    return (text || "").toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 12);
  }

  function toast(msg, isErr) {
    el.toast.textContent = msg;
    el.toast.classList.toggle("err", !!isErr);
    el.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.toast.hidden = true; }, isErr ? 5000 : 1800);
  }

  async function api(path, body) {
    const opts = body === undefined ? {} : {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    };
    const res = await fetch(path, opts);
    let data = null;
    try { data = await res.json(); } catch (_e) { data = null; }
    if (!res.ok) {
      const err = new Error((data && data.error && data.error.message) || `HTTP ${res.status}`);
      err.status = res.status;
      err.info = data && data.error;
      throw err;
    }
    return data;
  }

  function labelerName() {
    return el.labeler.value.trim();
  }

  function current() {
    return rows[idx] || null;
  }

  // ---- counts / progress -----------------------------------------------------------------

  function showCounts(data) {
    if (!data || !data.counts) return;
    const c = data.counts;
    for (const s of STATUSES) $("n-" + s).textContent = String(c[s] || 0);
    const total = data.total || 0;
    const done = total - (c.unverified || 0);
    el.pct.textContent = total ? `${Math.floor((100 * done) / total)}%` : "0%";
    for (const s of ["verified", "unreadable", "skip"]) {
      $("bar-" + s).style.width = total ? `${(100 * (c[s] || 0)) / total}%` : "0";
    }
    if (Array.isArray(data.sources)) {
      const keep = el.fSource.value;
      const have = Array.from(el.fSource.options).map((o) => o.value);
      const want = ["", ...data.sources];
      if (have.join("\n") !== want.join("\n")) {
        el.fSource.replaceChildren();
        for (const s of want) {
          const o = document.createElement("option");
          o.value = s;
          o.textContent = s || "All sources";
          el.fSource.append(o);
        }
        el.fSource.value = want.includes(keep) ? keep : "";
      }
    }
  }

  // ---- validation hint ---------------------------------------------------------------------

  function setHint(cls, text) {
    el.hint.className = "hint" + (cls ? " " + cls : "");
    el.hint.textContent = text;
  }

  function scheduleValidate() {
    clearTimeout(validateTimer);
    const text = clean(el.plate.value);
    if (!text) { setHint("", "Type the plate exactly as printed (A-Z, 0-9)"); return; }
    validateTimer = setTimeout(async () => {
      const seq = ++validateSeq;
      try {
        const v = await api("/api/validate?text=" + encodeURIComponent(text));
        if (seq !== validateSeq) return;
        if (v.valid) setHint("ok", `${v.display} · valid ${v.kind === "bh" ? "BH-series" : "Indian"} format`);
        else setHint("warn", v.message + " — saving needs a confirm click");
      } catch (e) {
        if (seq === validateSeq) setHint("warn", "Validation unavailable: " + e.message);
      }
    }, 90);
  }

  // ---- rendering ---------------------------------------------------------------------------

  function hideConfirm() {
    pending = null;
    el.confirm.hidden = true;
  }

  function render(focus) {
    hideConfirm();
    const r = current();
    el.pos.textContent = rows.length ? `${idx + 1} / ${rows.length}` : "0 / 0";
    el.prev.disabled = idx <= 0;
    el.next.disabled = idx >= rows.length - 1;
    el.empty.hidden = !!r;
    el.work.hidden = !r;
    if (!r) {
      el.track.replaceChildren();
      el.trackTitle.textContent = "Track";
      return;
    }
    el.imgMain.src = r.image_url;
    el.imgRaw.src = r.raw_url || r.image_url;
    el.prep.textContent = r.prep || "";
    el.plate.value = clean(r.status === "verified" ? r.plate_text : r.suggestion);
    const conf = r.ocr_conf ? ` · conf ${Number(r.ocr_conf).toFixed(2)}` : "";
    const verdict = r.ocr_valid === "1" ? ` · valid → ${r.ocr_plate}` : " · not valid";
    el.mOcr.textContent = (r.ocr_text || "(no read)") + conf + verdict;
    el.mOcr.title = el.mOcr.textContent;
    const who = r.labeler ? ` · ${r.labeler}` : "";
    el.mStatus.textContent = r.status + (r.status === "verified" ? ` “${r.plate_text}”` : "") + who;
    el.mSource.textContent = r.source;
    const vname = r.video.split(/[\\/]/).pop();
    el.mVideo.textContent = vname;
    el.mVideo.title = r.video;
    el.mTrack.textContent = `#${r.track} · ${r.frame}`;
    el.mSize.textContent = `${r.width_px}px · ${r.two_line === "1" ? "2-line?" : "1-line?"}`;
    scheduleValidate();
    loadTrack(r);
    if (focus !== false) {
      el.plate.focus();
      el.plate.select();
    }
  }

  let trackSeq = 0;
  async function loadTrack(r) {
    const seq = ++trackSeq;
    let data;
    try {
      data = await api("/api/track?id=" + encodeURIComponent(r.id));
    } catch (e) {
      if (seq === trackSeq) toast("Track: " + e.message, true);
      return;
    }
    if (seq !== trackSeq) return;
    const items = data.rows || [];
    el.trackTitle.textContent = `Track #${r.track} · ${items.length} crop${items.length === 1 ? "" : "s"}`;
    el.track.replaceChildren();
    for (const t of items) {
      const li = document.createElement("li");
      if (t.id === r.id) li.className = "current";
      const b = document.createElement("button");
      b.type = "button";
      b.title = `frame ${t.frame} · ${t.status === "verified" ? t.plate_text : t.ocr_text || "no read"}`;
      const img = document.createElement("img");
      img.src = t.image_url;
      img.alt = `frame ${t.frame}`;
      img.loading = "lazy";
      const line = document.createElement("span");
      line.className = "t-line";
      const txt = document.createElement("span");
      txt.className = "t-text";
      txt.textContent = t.status === "verified" ? t.plate_text : (t.ocr_text || "—");
      const st = document.createElement("span");
      st.className = "st st-" + t.status;
      st.textContent = t.status;
      line.append(txt, st);
      b.append(img, line);
      b.addEventListener("click", () => jumpTo(t));
      li.append(b);
      el.track.append(li);
    }
  }

  function jumpTo(t) {
    let i = rows.findIndex((x) => x.id === t.id);
    if (i < 0) {
      // not in the filtered list (e.g. already verified while showing "unverified only"): insert it
      i = Math.min(idx + 1, rows.length);
      rows.splice(i, 0, t);
    }
    idx = i;
    render();
  }

  function go(delta) {
    const n = Math.max(0, Math.min(rows.length - 1, idx + delta));
    if (n !== idx) { idx = n; render(); }
  }

  async function load(keepId) {
    const q = new URLSearchParams({ status: el.fStatus.value, source: el.fSource.value, two_line: el.fLines.value });
    let data;
    try {
      data = await api("/api/rows?" + q.toString());
    } catch (e) {
      toast("Load failed: " + e.message, true);
      return;
    }
    rows = data.rows || [];
    showCounts(data);
    const i = keepId ? rows.findIndex((r) => r.id === keepId) : -1;
    idx = i >= 0 ? i : 0;
    render();
  }

  // ---- saving ------------------------------------------------------------------------------

  function mergeRow(r) {
    const i = rows.findIndex((x) => x.id === r.id);
    if (i >= 0) rows[i] = r;
  }

  function nextAfterSave(trackKey) {
    // next row in the list that is still unverified (and, after a track save, not in that track)
    for (let i = idx + 1; i < rows.length; i++) {
      const r = rows[i];
      if (r.status !== "unverified") continue;
      if (trackKey && `${r.source}|${r.video}|${r.track}` === trackKey) continue;
      idx = i;
      return;
    }
    idx = Math.min(idx + 1, rows.length - 1);
  }

  async function submit(action, confirm) {
    const r = current();
    if (!r || busy) return;
    const name = labelerName();
    if (!name) {
      toast("Enter your name in “Labeller” (top right) first", true);
      el.labeler.focus();
      return;
    }
    const text = clean(el.plate.value);
    const isTrack = action === "track";
    const status = isTrack ? "verified" : action;
    if (status === "verified" && !text) {
      toast("Type the plate text first (or mark it unreadable)", true);
      el.plate.focus();
      return;
    }
    const body = {
      id: r.id, status, plate_text: status === "verified" ? text : "", labeler: name, confirm: !!confirm,
      rev: r.rev || "",
    };
    busy = true;
    try {
      const data = await api(isTrack ? "/api/label-track" : "/api/label", body);
      hideConfirm();
      showCounts(data);
      if (isTrack) {
        for (const x of data.rows) mergeRow(x);
        toast(`Saved ${text} on ${data.updated} crop${data.updated === 1 ? "" : "s"} of track #${r.track}`);
        nextAfterSave(`${r.source}|${r.video}|${r.track}`);
      } else {
        mergeRow(data.row);
        if (status === "unverified") { render(); return; }
        toast(status === "verified" ? `Saved ${text}` : `Marked ${status}`);
        nextAfterSave(null);
      }
      render();
    } catch (e) {
      if (e.status === 409 && e.info && e.info.code === "needs_confirm") {
        pending = action;
        el.confirmMsg.textContent = `${text}: ${e.message}. Save it anyway?`;
        el.confirm.hidden = false;
      } else if (e.status === 409 && e.info && e.info.code === "conflict" && e.info.row) {
        // someone else saved this crop meanwhile: show their version, never overwrite it blindly
        mergeRow(e.info.row);
        render();
        toast(e.message, true);
      } else {
        toast("Save failed: " + e.message, true);
      }
    } finally {
      busy = false;
    }
  }

  // ---- events ------------------------------------------------------------------------------

  el.labeler.value = localStorage.getItem("labeler") || "";
  el.labeler.addEventListener("input", () => localStorage.setItem("labeler", el.labeler.value.trim()));

  el.plate.addEventListener("input", () => {
    const pos = el.plate.selectionStart;
    const before = el.plate.value;
    const after = clean(before);
    if (after !== before) {
      el.plate.value = after;
      const p = Math.max(0, Math.min(after.length, (pos || 0) - (before.length - after.length)));
      el.plate.setSelectionRange(p, p);
    }
    hideConfirm();
    scheduleValidate();
  });

  el.save.addEventListener("click", () => submit("verified"));
  el.trackApply.addEventListener("click", () => submit("track"));
  el.unreadable.addEventListener("click", () => submit("unreadable"));
  el.skip.addEventListener("click", () => submit("skip"));
  el.reset.addEventListener("click", () => submit("unverified"));
  el.confirmYes.addEventListener("click", () => { if (pending) submit(pending, true); });
  el.confirmNo.addEventListener("click", () => { hideConfirm(); el.plate.focus(); });
  el.prev.addEventListener("click", () => go(-1));
  el.next.addEventListener("click", () => go(1));
  el.reload.addEventListener("click", () => load(current() && current().id));
  for (const f of [el.fStatus, el.fSource, el.fLines]) f.addEventListener("change", () => load(null));
  el.imgMain.addEventListener("error", () => toast("Image missing: " + (current() ? current().image_path : ""), true));

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    const inPlate = t === el.plate;
    const inField = t instanceof HTMLInputElement || t instanceof HTMLSelectElement;
    const onButton = t instanceof HTMLButtonElement;
    if (t === el.labeler) {
      if (e.key === "Enter" || e.key === "Escape") { e.preventDefault(); el.plate.focus(); }
      return;
    }
    if (e.key === "Enter" && !e.altKey && !e.shiftKey) {
      if (onButton && !(e.ctrlKey || e.metaKey)) return; // let Enter "click" the focused button
      if (t instanceof HTMLSelectElement) return;
      e.preventDefault();
      submit(e.ctrlKey || e.metaKey ? "unreadable" : "verified");
      return;
    }
    if (e.altKey && !e.ctrlKey && !e.metaKey) {
      if (e.code === "KeyS") { e.preventDefault(); submit("skip"); return; }
      if (e.code === "KeyT") { e.preventDefault(); submit("track"); return; }
      if (e.code === "KeyU") { e.preventDefault(); submit("unreadable"); return; }
    }
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      if (t instanceof HTMLSelectElement) return;
      e.preventDefault();
      go(e.key === "ArrowUp" ? -1 : 1);
      return;
    }
    if (e.key === "Escape") {
      if (!el.confirm.hidden) { hideConfirm(); return; }
      if (inPlate) el.plate.blur();
      return;
    }
    if (inField || e.ctrlKey || e.metaKey || e.altKey) return;
    // single-key shortcuts when the text box is not focused
    const k = e.key.toLowerCase();
    if (k === "s") { e.preventDefault(); submit("skip"); }
    else if (k === "t") { e.preventDefault(); submit("track"); }
    else if (k === "u") { e.preventDefault(); submit("unreadable"); }
    else if (k === "e" || k === "/") { e.preventDefault(); el.plate.focus(); el.plate.select(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
  });

  load(null);
})();
