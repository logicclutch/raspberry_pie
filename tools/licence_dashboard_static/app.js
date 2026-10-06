/* Licence Manager page. Plain JS; server data reaches the DOM through textContent only. */
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  };

  // The secret link is http://127.0.0.1:8090/#token=...: keep it for this tab, take it out of the address bar.
  const TOKEN_KEY = "lm.token";
  const m = location.hash.match(/token=([A-Za-z0-9_-]+)/);
  if (m) {
    sessionStorage.setItem(TOKEN_KEY, m[1]);
    history.replaceState(null, "", location.pathname);
  }
  const token = () => sessionStorage.getItem(TOKEN_KEY) || "";

  async function call(method, path, body) {
    const res = await fetch(path, {
      method,
      cache: "no-store",
      headers: Object.assign({ "X-Token": token(), Accept: "application/json" },
        body ? { "Content-Type": "application/json" } : {}),
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch (_) { /* not JSON */ }
    if (!res.ok) {
      const msg = data && (data.error || (data.detail && data.detail[0] && data.detail[0].msg));
      const e = new Error(msg || "The Licence Manager answered " + res.status);
      e.status = res.status;
      throw e;
    }
    return data;
  }

  function alertBox(text) {
    $("alert").hidden = !text;
    $("alert").textContent = text || "";
  }

  let toastTimer = null;
  function toast(text) {
    $("toast").textContent = text;
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 2200);
  }

  async function copy(text, what) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const t = el("textarea"); t.value = text; document.body.append(t); t.select();
      document.execCommand("copy"); t.remove();
    }
    toast(what + " copied");
  }

  async function download(licenceId) {
    try {
      const res = await fetch("/api/licences/" + encodeURIComponent(licenceId) + "/file",
        { headers: { "X-Token": token() }, cache: "no-store" });
      if (!res.ok) throw new Error("download failed (" + res.status + ")");
      const url = URL.createObjectURL(await res.blob());
      const a = el("a"); a.href = url; a.download = "licence.json"; document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch (e) { alertBox(e.message); }
  }

  const fmtDate = (iso) => {
    if (!iso) return "No end date";
    const d = new Date(iso + "T00:00:00");
    return d.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
  };
  const STATUS = { active: ["Active", "chip-ok"], expiring: ["Ending soon", "chip-warn"], ended: ["Ended", "chip-bad"] };
  function statusChip(l) {
    const [label, cls] = STATUS[l.status] || ["—", ""];
    let text = label;
    if (l.status === "expiring") text += " · " + l.days_left + " day" + (l.days_left === 1 ? "" : "s");
    return el("span", "chip " + cls, text);
  }
  function facts(dl, pairs) {
    dl.replaceChildren();
    for (const [k, v] of pairs) {
      const d = el("div");
      d.append(el("dt", "", k), el("dd", "", v));
      dl.append(d);
    }
  }

  // ---- signing key status ----------------------------------------------------------------------
  let info = null;
  async function loadInfo() {
    try {
      info = await call("GET", "/api/info");
      const chip = $("keychip");
      if (!info.key_ok) {
        chip.className = "chip chip-bad"; chip.textContent = "No signing key";
        alertBox(info.key_error + " — licences cannot be made.");
      } else if (!info.key_matches_software) {
        chip.className = "chip chip-bad"; chip.textContent = "Wrong signing key";
        alertBox("This signing key does not match the key built into the software: devices would reject its licences.");
      } else {
        chip.className = "chip chip-ok"; chip.textContent = "Signing key OK";
        chip.title = info.key_path;
      }
      $("generate").disabled = !(info.key_ok && info.key_matches_software);
      validHint();
    } catch (e) {
      alertBox(e.status === 401 ? "Open the link printed in the Terminal when the Licence Manager started." : e.message);
      $("generate").disabled = true;
    }
  }

  // ---- new licence form ------------------------------------------------------------------------
  let validity = "1y";
  function setValidity(v) {
    validity = v;
    for (const b of $("validity").querySelectorAll("button")) b.setAttribute("aria-checked", String(b.dataset.v === v));
    $("date-row").hidden = v !== "date";
    validHint();
  }
  function addYears(iso, n) {
    const d = new Date(iso + "T00:00:00");
    const end = new Date(d); end.setFullYear(d.getFullYear() + n);
    if (end.getDate() !== d.getDate()) end.setDate(0); // 29 Feb -> 28 Feb
    end.setDate(end.getDate() - 1);
    return end.getFullYear() + "-" + String(end.getMonth() + 1).padStart(2, "0") + "-" +
      String(end.getDate()).padStart(2, "0");
  }
  function validHint() {
    const today = info ? info.today : null;
    let t = "";
    if (validity === "never") t = "The licence never ends.";
    else if (validity === "date") t = $("expires").value ? "Works up to and including " + fmtDate($("expires").value) + "." : "";
    else if (today) t = "Works up to and including " + fmtDate(addYears(today, parseInt(validity, 10))) + ".";
    $("valid-hint").textContent = t;
  }
  for (const b of $("validity").querySelectorAll("button")) b.addEventListener("click", () => setValidity(b.dataset.v));

  let keyType = "customer";
  function setKeyType(k) {
    keyType = k;
    for (const b of $("keytype").querySelectorAll("button")) b.setAttribute("aria-checked", String(b.dataset.k === k));
    $("generate").textContent = k === "machine" ? "Generate one key per machine" : "Generate licence key";
  }
  function busyText() {
    const n = parseInt(($("count").textContent.match(/^\d+/) || ["0"])[0], 10);
    return keyType === "machine" && n > 200 ? "Generating " + n + " keys… (about " + Math.max(1, Math.round(n / 150)) + " s)" : "Generating…";
  }
  for (const b of $("keytype").querySelectorAll("button")) b.addEventListener("click", () => setKeyType(b.dataset.k));
  $("expires").addEventListener("input", validHint);

  let countTimer = null;
  let countSeq = 0;
  function liveCount() {
    clearTimeout(countTimer);
    countTimer = setTimeout(async () => {
      const seq = ++countSeq;
      const text = $("machines").value;
      if (!text.trim()) { setCount(0, null); return; }
      try {
        const r = await call("POST", "/api/machines", { machines: text });
        if (seq === countSeq) setCount(r.count, r.error, r.labelled);
      } catch (_) { /* shown on submit */ }
    }, 250);
  }
  function setCount(n, error, labelled) {
    const c = $("count");
    c.className = "count" + (error ? " is-bad" : n ? " is-ok" : "");
    c.textContent = error ? error : n + " machine" + (n === 1 ? "" : "s") + (labelled ? " · " + labelled + " labelled" : "");
    $("machines").setAttribute("aria-invalid", String(!!error));
  }
  $("machines").addEventListener("input", liveCount);

  function formErr(text) {
    $("form-err").hidden = !text;
    $("form-err").textContent = text || "";
  }

  let busy = false;
  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (busy) return;
    const customer = $("customer").value.trim();
    if (!customer) { formErr("Enter the customer name."); $("customer").focus(); return; }
    if (!$("machines").value.trim()) { formErr("Paste at least one machine ID."); $("machines").focus(); return; }
    if (validity === "date" && !$("expires").value) { formErr("Choose the last valid day."); $("expires").focus(); return; }
    busy = true; formErr(null);
    $("generate").disabled = true; $("generate").textContent = busyText();
    try {
      const r = await call("POST", "/api/licences", {
        customer, machines: $("machines").value, validity, key_type: keyType,
        expires: validity === "date" ? $("expires").value : null, note: $("note").value,
      });
      showResult(r.licence);
      loadList();
    } catch (err) {
      formErr(err.message);
    } finally {
      busy = false; setKeyType(keyType);
      $("generate").disabled = !(info && info.key_ok && info.key_matches_software);
    }
  });

  function clearForm() {
    $("form").reset(); setValidity("1y"); setKeyType("customer"); setCount(0, null); formErr(null);
  }
  $("clear").addEventListener("click", clearForm);

  // Rows of per-machine keys (result panel and detail dialog).
  function keyRows(tbody, l) {
    tbody.replaceChildren();
    for (const it of l.items) {
      const tr = el("tr");
      const cp = el("button", "btn btn-sm", "Copy"); cp.type = "button";
      cp.addEventListener("click", (e) => { e.stopPropagation(); copy(it.key, "Key for " + it.device); });
      const act = el("td", "act"); act.append(cp);
      const k = el("td", "k", it.key); k.title = it.key;
      tr.append(el("td", "mono", it.device), el("td", "", it.label || "—"), k, act);
      tbody.append(tr);
    }
  }

  async function downloadCsv(batch, kind) {
    const ext = kind === "csv" ? "csv" : "xlsx";
    try {
      const res = await fetch("/api/batches/" + encodeURIComponent(batch) + "/" + ext,
        { headers: { "X-Token": token() }, cache: "no-store" });
      if (!res.ok) throw new Error("download failed (" + res.status + ")");
      const cd = res.headers.get("Content-Disposition") || "";
      const star = /filename\*=UTF-8''([^;]+)/.exec(cd);
      const name = star ? decodeURIComponent(star[1]) : ((/filename="([^"]+)"/.exec(cd) || [])[1] || "licence-keys." + ext);
      const url = URL.createObjectURL(await res.blob());
      const a = el("a"); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
      toast(ext === "csv" ? "CSV list downloaded" : "Excel file downloaded");
    } catch (e) { alertBox(e.message); }
  }

  let current = null;
  function showResult(l) {
    current = l;
    const many = l.key_type === "machine";
    facts($("res-facts"), [
      ["Customer", l.customer], [many ? "Batch" : "Licence ID", l.licence_id],
      ["Machines", String(l.devices.length) + (many ? " · one key each" : " · one key")],
      ["Valid until", fmtDate(l.expires)],
    ]);
    $("res-one").hidden = many;
    $("res-many").hidden = !many;
    if (many) {
      keyRows($("res-keys"), l);
      $("res-many-n").textContent = l.items.length + " key" + (l.items.length === 1 ? "" : "s");
    } else {
      $("res-key").value = l.key;
    }
    $("result").classList.remove("is-empty");
    $("res-empty").hidden = true;
    $("res-full").hidden = false;
    $("res-chip").hidden = false;
    $("res-title").textContent = many ? "Licence keys ready (one per machine)" : "Licence key ready";
    $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
    if (!many) { $("res-key").focus(); $("res-key").select(); } else $("res-csv").focus();
  }
  $("res-copy").addEventListener("click", () => current && copy(current.key, "Licence key"));
  $("res-file").addEventListener("click", (e) => { e.preventDefault(); if (current) download(current.licence_id); });
  $("res-csv").addEventListener("click", () => current && downloadCsv(current.licence_id, "xlsx"));
  $("res-csv2").addEventListener("click", () => current && downloadCsv(current.licence_id, "csv"));

  // ---- issued list -----------------------------------------------------------------------------
  let all = [];
  async function loadList() {
    try {
      all = (await call("GET", "/api/licences")).licences;
      renderList();
    } catch (e) { if (e.status !== 401) alertBox(e.message); }
  }
  function renderList() {
    const q = $("search").value.trim().toLowerCase().replace(/^0+/, "");
    const rows = all.filter((l) => {
      if (!l.valid) return !q;
      if (!q) return true;
      const labels = l.labels ? Object.values(l.labels).join(" ") : "";
      const ids = l.items ? l.items.map((i) => i.licence_id).join(" ") : "";
      return (l.customer + " " + l.licence_id + " " + ids + " " + (l.note || "") + " " + labels).toLowerCase().includes(q) ||
        l.devices.some((d) => d.includes(q));
    });
    $("list-count").textContent = all.length ? "(" + all.length + ")" : "";
    const tb = $("rows");
    tb.replaceChildren();
    for (const l of rows) {
      const tr = el("tr");
      if (!l.valid) {
        const td = el("td", "", "Damaged record " + l.file + ": " + l.error);
        td.colSpan = 8; tr.append(td); tb.append(tr); continue;
      }
      const cust = el("td", "cust", l.customer);
      if (l.note) cust.append(el("small", "", l.note));
      const many = l.key_type === "machine";
      const act = el("td", "act");
      const cp = el("button", "btn btn-sm", many ? "Excel" : "Copy key"); cp.type = "button";
      cp.addEventListener("click", (e) => {
        e.stopPropagation();
        if (many) downloadCsv(l.licence_id, "xlsx"); else copy(l.key, "Licence key");
      });
      act.append(cp);
      const st = el("td"); st.append(statusChip(l));
      const kt = el("td"); kt.append(el("span", "ktag" + (many ? " is-machine" : ""), many ? "Per machine" : "One for all"));
      tr.append(cust, el("td", "lid", l.licence_id), kt, el("td", "num", l.devices.length), el("td", "", fmtDate(l.issued)),
        el("td", "", fmtDate(l.expires)), st, act);
      tr.addEventListener("click", () => openDetail(l));
      tb.append(tr);
    }
    $("empty").hidden = rows.length > 0;
    $("empty").textContent = all.length ? "No licence matches the search." : "No licences issued yet.";
  }
  $("search").addEventListener("input", renderList);

  let detail = null;
  function openDetail(l) {
    detail = l;
    $("d-title").textContent = l.customer;
    $("d-eyebrow").textContent = l.licence_id;
    facts($("d-facts"), [
      ["Issued", fmtDate(l.issued)], ["Valid until", fmtDate(l.expires)],
      ["Status", (STATUS[l.status] || ["—"])[0] + (l.days_left !== null && l.status !== "ended" ? " · " + l.days_left + " days left" : "")],
      ["Note", l.note || "—"],
    ]);
    const many = l.key_type === "machine";
    $("d-one").hidden = many;
    $("d-many").hidden = !many;
    $("d-copy").hidden = many;
    $("d-file").hidden = many;
    $("d-csv").hidden = !many;
    $("d-csv2").hidden = !many;
    if (many) {
      $("d-kcount").textContent = l.items.length;
      keyRows($("d-keys"), l);
    } else {
      $("d-mcount").textContent = l.devices.length;
      $("d-machines").textContent = l.devices.join("\n");
      $("d-key").value = l.key;
    }
    $("d-eyebrow").textContent = (many ? "Batch · one key per machine · " : "One key for all machines · ") + l.licence_id;
    $("detail").showModal();
  }
  $("d-copy").addEventListener("click", () => detail && copy(detail.key, "Licence key"));
  $("d-file").addEventListener("click", (e) => { e.preventDefault(); if (detail) download(detail.licence_id); });
  $("d-csv").addEventListener("click", () => detail && downloadCsv(detail.licence_id, "xlsx"));
  $("d-csv2").addEventListener("click", () => detail && downloadCsv(detail.licence_id, "csv"));
  $("d-renew").addEventListener("click", () => {
    if (!detail) return;
    $("detail").close();
    clearForm();
    $("customer").value = detail.customer;
    const labels = detail.labels || {};
    $("machines").value = detail.devices.map((d) => labels[d] ? d + "  # " + labels[d] : d).join("\n") + "\n";
    $("note").value = detail.note || "";
    setKeyType(detail.key_type === "machine" ? "machine" : "customer");
    liveCount();
    window.scrollTo({ top: 0, behavior: "smooth" });
    $("machines").focus();
    toast("Customer and machines copied — add machines or change the dates, then Generate");
  });

  // ---- check a key -----------------------------------------------------------------------------
  $("checkform").addEventListener("submit", async (e) => {
    e.preventDefault();
    const out = $("check-out");
    if (!$("check-key").value.trim()) { $("check-key").focus(); return; }
    try {
      const r = await call("POST", "/api/check", { key: $("check-key").value, machine_id: $("check-machine").value });
      out.replaceChildren();
      out.hidden = false;
      if (!r.valid) {
        out.className = "check-out is-bad";
        out.append(el("span", "verdict bad", "Not valid: "), document.createTextNode(r.error));
        return;
      }
      out.className = "check-out is-ok";
      const dl = el("dl", "facts");
      const rows = [["Customer", r.customer], ["Licence ID", r.licence_id],
        ["Key type", r.key_type === "machine" ? "One key per machine" : "One key for all machines"],
        ["Machines", r.key_type === "machine" ? r.devices.join(", ") + (r.label ? " (" + r.label + ")" : "") : String(r.devices.length)],
        ["Valid until", fmtDate(r.expires)]];
      facts(dl, rows);
      out.append(el("div", "verdict ok", "Genuine licence"), dl);
      if (r.machine) {
        out.append(el("div", "verdict " + (r.machine.ok ? "ok" : "bad"),
          (r.machine.ok ? "✓ " : "✗ ") + "Machine " + $("check-machine").value.trim() + ": " + r.machine.message));
      }
    } catch (err) {
      out.hidden = false; out.className = "check-out is-bad"; out.textContent = err.message;
    }
  });

  loadInfo();
  loadList();
})();
