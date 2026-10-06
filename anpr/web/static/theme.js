/* Applies the light/dark theme before the page paints (loaded without `defer`), so a light-theme
 * user never sees a dark flash. Order: ?theme=light|dark in the URL (handy for a wall display),
 * then the choice saved by the toggle, then dark. The toggle itself lives in app.js. */
"use strict";
(function () {
  var t = null;
  try { t = new URLSearchParams(location.search).get("theme"); } catch (_) { /* old browser */ }
  if (t !== "light" && t !== "dark") {
    try { t = localStorage.getItem("anpr.theme"); } catch (_) { /* private mode */ }
  }
  document.documentElement.setAttribute("data-theme", t === "light" ? "light" : "dark");
})();
