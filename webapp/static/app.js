// ============================================================
// DOM refs
// ============================================================
const search         = document.getElementById("search");
const suggestions    = document.getElementById("suggestions");
const results        = document.getElementById("results");
const statusBox      = document.getElementById("status");
const topNInput      = document.getElementById("top_n");
const diffTopKInput  = document.getElementById("diff_topk");
const goBtn          = document.getElementById("go");
const profileFullEl  = document.getElementById("profile-full");
const sidebarEl      = document.getElementById("profile-sidebar");
const sectionProfile = document.getElementById("section-profile");
const sectionResults = document.getElementById("section-results");

let selected      = null;
let debounceTimer = null;

// Bulk trait metadata cache: trait key → {display_name, description, category, higher_means}
const TRAIT_META = new Map();

// ============================================================
// View management
// ============================================================
function setView(name) {
  // 'landing' | 'profile' | 'results'
  const showProfile  = name === "profile" || name === "results";
  const showResults  = name === "results";
  sectionProfile.classList.toggle("hidden", !showProfile);
  sectionResults.classList.toggle("hidden", !showResults);
}

// ============================================================
// Math
// ============================================================
function percentileFromZ(z) {
  const x = Math.abs(z);
  const t = 1 / (1 + 0.2316419 * x);
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const tail = Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * poly;
  const pct  = z >= 0 ? (1 - tail) * 100 : tail * 100;
  return Math.round(Math.max(1, Math.min(99, pct)));
}

// ============================================================
// Formatting
// ============================================================
function formatZ(z) {
  const n = Number(z);
  if (!Number.isFinite(n)) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}σ`;
}

function formatVal(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(3) : "—";
}

function formatDelta(feature, delta) {
  const d    = Number(delta);
  const sign = d >= 0 ? "+" : "";
  const hint = featureUnitHint(feature);
  return `${sign}${d.toFixed(3)}${hint ? " " + hint : ""}`;
}

function featureUnitHint(name) {
  const n = String(name || "");
  if (n.includes("angle"))                            return "deg";
  if (n.startsWith("pct_") || n.includes("pct"))     return "%";
  if (n.includes("per90") || n.includes("per_90"))   return "p90";
  return "";
}

function prettyFeatureName(raw) {
  let s = String(raw || "")
    .replace(/^pct_/, "")
    .replace(/_per90$|_per_90$/g, " per90")
    .replace(/_/g, " ")
    .replace("def third", "defensive third")
    .replace("mid third", "middle third")
    .replace("att third", "attacking third");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, m =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m])
  );
}

// ============================================================
// Z-score bar HTML  (centred at 50%, clamped ±3σ)
// ============================================================
function zBarHtml(z) {
  const clamped = Math.max(-3, Math.min(3, Number(z) || 0));
  const pct     = Math.abs(clamped) / 3 * 50;
  const isPos   = clamped >= 0;
  const left    = isPos ? "50%" : `${(50 - pct).toFixed(1)}%`;
  const cls     = isPos ? "z-fill-pos" : "z-fill-neg";
  return `
    <div class="z-bar">
      <div class="z-center"></div>
      <div class="z-fill ${cls}" style="left:${left};width:${pct.toFixed(1)}%;"></div>
    </div>`;
}

// ============================================================
// Trait meta cache helpers
// ============================================================
async function ensureTraitMeta(traits) {
  const missing = traits
    .map(t => String(t || "").trim())
    .filter(t => t && !TRAIT_META.has(t));
  if (!missing.length) return;

  const res = await safeJsonFetch("/api/traits", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ traits: [...new Set(missing)].slice(0, 500) }),
  });
  for (const [k, v] of Object.entries(res?.traits || {})) TRAIT_META.set(k, v);
}

function labelForTrait(key) {
  const m = TRAIT_META.get(String(key || "").trim());
  return (m && m.display_name) ? m.display_name : prettyFeatureName(key);
}

function tooltipForTrait(key) {
  const m = TRAIT_META.get(String(key || "").trim());
  if (!m) return null;
  const parts = [];
  if (m.description)  parts.push(m.description);
  if (m.higher_means) parts.push(`Higher means: ${m.higher_means}`);
  parts.push(`Key: ${key}`);
  return parts.join("\n\n");
}

function patchRenderedTraitElements() {
  document.querySelectorAll("[data-trait]").forEach(el => {
    const key = String(el.dataset.trait || "").trim();
    if (!key) return;
    const label = labelForTrait(key);
    if (label && el.textContent !== label) el.textContent = label;
    const tip = tooltipForTrait(key);
    if (tip) { el.title = tip; el.dataset.tooltipLoaded = "1"; }
  });
}

// Lazy hover fallback for traits not in bulk cache
document.addEventListener("mouseover", async e => {
  const el = e.target.closest("[data-trait]");
  if (!el || el.dataset.tooltipLoaded === "1") return;
  const key = String(el.dataset.trait || "").trim();
  if (!key) return;
  const cached = tooltipForTrait(key);
  if (cached) { el.title = cached; el.dataset.tooltipLoaded = "1"; return; }
  el.title = "Loading…";
  try {
    const data = await safeJsonFetch(`/api/trait?trait=${encodeURIComponent(key)}`);
    TRAIT_META.set(key, data);
    const tip = tooltipForTrait(key);
    if (tip) el.title = tip;
    el.dataset.tooltipLoaded = "1";
  } catch { el.title = "No glossary entry."; }
});

// ============================================================
// Fetch helper
// ============================================================
async function safeJsonFetch(url, opts = {}) {
  const res = await fetch(url, opts);
  const ct  = res.headers.get("content-type") || "";
  if (!ct.includes("application/json")) {
    const text = await res.text();
    throw new Error(`Non-JSON response (${res.status}): ${text.slice(0, 120)}`);
  }
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error || `Request failed (${res.status})`);
  return data;
}

// ============================================================
// Category summary bar row
// ============================================================
function categorySummaryHtml(cats) {
  if (!cats || !cats.length) return "";
  return `
    <div class="cat-summary">
      ${cats.map(c => {
        const z      = Number(c.mean_z) || 0;
        const pct    = Math.min(100, Math.abs(z) / 3 * 100).toFixed(1);
        const isPos  = z >= 0;
        const cls    = isPos ? "cat-pos" : "cat-neg";
        const fillCls= isPos ? "z-fill-pos" : "z-fill-neg";
        return `
          <div class="cat-card ${cls}">
            <div class="cat-card-name">${escapeHtml(c.category)}</div>
            <div class="cat-card-z ${isPos ? "z-pos" : "z-neg"}">${isPos ? "+" : ""}${z.toFixed(2)}σ</div>
            <div class="cat-mini-track">
              <div class="cat-mini-fill ${fillCls}" style="width:${pct}%;"></div>
            </div>
            <div class="cat-feat-count">${c.feat_count} features</div>
          </div>`;
      }).join("")}
    </div>`;
}

// ============================================================
// Trait row renderers — full (profile page) and compact (sidebar)
// ============================================================
function traitRowHtml(t, mode) {
  const key    = t.trait;
  const label  = labelForTrait(key);
  const z      = Number(t.z);
  const isPos  = z >= 0;
  const tip    = escapeHtml(tooltipForTrait(key) || "Loading…");
  const pctile = percentileFromZ(z);

  const roleMean = (t.role_mean != null)
    ? `<span class="role-mean">Role avg: ${formatVal(t.role_mean)}</span>` : "";

  if (mode === "full") {
    const higherMeans = t.higher_means
      ? `<div class="trait-higher-means">${escapeHtml(t.higher_means)}</div>` : "";
    return `
      <div class="trait-row">
        <div class="trait-row-left">
          <span class="trait" data-trait="${escapeHtml(key)}" title="${tip}">${escapeHtml(label)}</span>
          ${higherMeans}
          <div class="trait-val-line">
            <span class="trait-val">${formatVal(t.value)}</span>
            ${roleMean}
          </div>
        </div>
        <div class="trait-row-right">
          <div class="trait-z-line">
            <span class="${isPos ? "z-pos" : "z-neg"} z-score-num">${formatZ(z)}</span>
            <span class="z-pct-label">${pctile}th %ile</span>
          </div>
          ${zBarHtml(z)}
        </div>
      </div>`;
  }

  // sidebar: compact
  return `
    <div class="sidebar-trait-row">
      <div class="sidebar-trait-header">
        <span class="trait" data-trait="${escapeHtml(key)}" title="${tip}">${escapeHtml(label)}</span>
        <span class="${isPos ? "z-pos" : "z-neg"} z-score-num">${formatZ(z)}</span>
      </div>
      ${zBarHtml(z)}
      <div class="sidebar-trait-footer">
        <span class="role-mean">${escapeHtml(t.direction_vs_role || (isPos ? "higher" : "lower"))} vs role</span>
        <span class="z-pct-label">${pctile}th %ile</span>
      </div>
    </div>`;
}

function traitSectionHtml(title, traits, mode) {
  return `
    <div class="trait-section">
      <div class="trait-section-title">${escapeHtml(title)}</div>
      <div class="trait-section-items">${traits.map(t => traitRowHtml(t, mode)).join("")}</div>
    </div>`;
}

function groupByCategory(traits) {
  const byCategory  = {};
  const uncategorized = [];
  for (const t of traits) {
    t.category ? (byCategory[t.category] = byCategory[t.category] || []).push(t)
               : uncategorized.push(t);
  }
  return { byCategory, uncategorized };
}

// ============================================================
// Full-width source player profile
// ============================================================
async function loadAndRenderFullProfile(player_key, dataset) {
  profileFullEl.innerHTML = `<div class="profile-loading">Loading profile…</div>`;
  sectionProfile.classList.remove("hidden");

  try {
    const prof = await safeJsonFetch(
      `/api/player_profile?player_key=${encodeURIComponent(player_key)}&dataset=${encodeURIComponent(dataset)}&topk=12`
    );
    await ensureTraitMeta((prof.top_traits || []).map(t => t.trait));
    renderFullProfile(prof);
  } catch (err) {
    profileFullEl.innerHTML = `<div class="profile-error">Could not load profile: ${escapeHtml(err.message)}</div>`;
  }
}

function renderFullProfile(prof) {
  const top = prof.top_traits || [];
  const { byCategory, uncategorized } = groupByCategory(top);

  const sections = [
    ...Object.entries(byCategory).map(([cat, traits]) => traitSectionHtml(cat, traits, "full")),
    ...(uncategorized.length ? [traitSectionHtml("Other", uncategorized, "full")] : []),
  ].join("");

  profileFullEl.innerHTML = `
    <div class="profile-full-header">
      <div>
        <div class="profile-full-name">${escapeHtml(prof.player || "—")}</div>
        <div class="profile-full-chips">
          <span class="chip">${escapeHtml(prof.dataset || "—")}</span>
          <span class="chip">${escapeHtml(prof.player_position || "—")}</span>
          <span class="chip">${escapeHtml(prof.role_name || "—")}</span>
        </div>
      </div>
      <div class="profile-baseline-note">Traits ranked by deviation from role peers</div>
    </div>
    ${categorySummaryHtml(prof.category_summary || [])}
    <div class="profile-section-label">Standout traits vs role peers</div>
    <div class="trait-sections">
      ${sections || `<div class="profile-note">No trait data available.</div>`}
    </div>`;

  patchRenderedTraitElements();
}

// ============================================================
// Sidebar comparison profile (clicked result)
// ============================================================
async function loadAndRenderSidebarProfile(player_key, dataset) {
  sidebarEl.classList.remove("muted");
  sidebarEl.innerHTML = `<div class="profile-loading">Loading…</div>`;

  try {
    const prof = await safeJsonFetch(
      `/api/player_profile?player_key=${encodeURIComponent(player_key)}&dataset=${encodeURIComponent(dataset)}&topk=10`
    );
    await ensureTraitMeta((prof.top_traits || []).map(t => t.trait));
    renderSidebarProfile(prof);
  } catch (err) {
    sidebarEl.classList.add("muted");
    sidebarEl.textContent = `Could not load: ${err.message}`;
  }
}

function renderSidebarProfile(prof) {
  const top = prof.top_traits || [];
  const { byCategory, uncategorized } = groupByCategory(top);

  const sections = [
    ...Object.entries(byCategory).map(([cat, traits]) => traitSectionHtml(cat, traits, "sidebar")),
    ...(uncategorized.length ? [traitSectionHtml("Other", uncategorized, "sidebar")] : []),
  ].join("");

  sidebarEl.classList.remove("muted");
  sidebarEl.innerHTML = `
    <div class="sidebar-profile-header">
      <div class="sidebar-profile-name">${escapeHtml(prof.player || "—")}</div>
      <div class="sidebar-profile-sub">${escapeHtml(prof.dataset || "—")} · ${escapeHtml(prof.player_position || "—")}</div>
      <div class="sidebar-profile-role">${escapeHtml(prof.role_name || "—")}</div>
    </div>
    ${categorySummaryHtml(prof.category_summary || [])}
    ${sections || `<div class="profile-note">No traits returned.</div>`}`;

  patchRenderedTraitElements();
}

// ============================================================
// Results cards
// ============================================================
function simBucket(sim) {
  const s = Number(sim);
  if (!Number.isFinite(s)) return "sim-unk";
  if (s >= 0.92) return "sim-elite";
  if (s >= 0.85) return "sim-hi";
  if (s >= 0.78) return "sim-good";
  if (s >= 0.70) return "sim-mid";
  if (s >= 0.62) return "sim-ok";
  if (s >= 0.55) return "sim-low";
  return "sim-vlow";
}

function deltaItemHtml(d) {
  const cls   = Number(d.delta) >= 0 ? "delta-pos" : "delta-neg";
  const label = labelForTrait(d.feature);
  const tip   = escapeHtml(tooltipForTrait(d.feature) || "Loading…");
  return `
    <div class="item">
      <span class="trait" data-trait="${escapeHtml(d.feature)}" title="${tip}">${escapeHtml(label)}</span>
      <span class="${cls}">${escapeHtml(formatDelta(d.feature, d.delta))}</span>
    </div>`;
}

function renderResults(payload) {
  results.innerHTML = "";
  const recs = payload.results || [];

  if (!recs.length) {
    results.innerHTML = `<div class="card">No recommendations found.</div>`;
    return;
  }

  recs.forEach((r, idx) => {
    const diffs  = r.biggest_differences || [];
    const sims   = r.greatest_similarities || [];
    const shared = Array.isArray(r.why_similar) ? r.why_similar.slice(0, 3) : [];

    const simVal  = Number(r.similarity);
    const simPct  = Number.isFinite(simVal) ? Math.max(0, Math.min(1, simVal)) * 100 : 0;
    const bucket  = simBucket(simVal);

    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="rank-badge">#${idx + 1}</div>
      <div class="card-top">
        <div>
          <h3 class="player-name">${escapeHtml(r.player)}</h3>
          <div class="role-name">${escapeHtml(r.role_name || "—")}</div>
        </div>
        <div class="chip">Sim: ${Number.isFinite(simVal) ? simVal.toFixed(3) : "—"}</div>
      </div>
      <div class="simbar ${bucket}"><div style="width:${simPct.toFixed(1)}%"></div></div>
      <div class="meta">
        <span class="chip">${escapeHtml(r.dataset)}</span>
        <span class="chip">${escapeHtml(r.player_position || "—")}</span>
        ${shared.length ? `<span class="chip">Role match</span>` : ""}
      </div>
      <div class="card-profile-hint">Click to view profile →</div>
      <details open>
        <summary>
          <span class="summary-title">Similarities</span>
          <span class="kicker">${sims.length ? `Top ${sims.length} closest features` : "No data"}</span>
        </summary>
        <div class="list" style="margin-top:10px;">
          ${sims.length ? sims.map(deltaItemHtml).join("") : shared.map(t => `<span class="chip">${escapeHtml(t)}</span>`).join("")}
        </div>
      </details>
      <details>
        <summary>
          <span class="summary-title">Biggest differences</span>
          <span class="kicker">Top ${diffs.length}</span>
        </summary>
        <div class="list" style="margin-top:10px;">
          ${diffs.length ? diffs.map(deltaItemHtml).join("") : `<div class="item"><span>No data</span><span></span></div>`}
        </div>
      </details>`;

    card.addEventListener("click", e => {
      if (e.target.closest("summary") || e.target.closest(".trait")) return;
      document.querySelectorAll(".card.active").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      loadAndRenderSidebarProfile(r.player_key, r.dataset);
    });

    results.appendChild(card);
  });
}

// ============================================================
// Search + suggestions
// ============================================================
function renderSuggestions(players) {
  suggestions.innerHTML = "";
  if (!players?.length) { clearSuggestions(); return; }

  players.forEach(p => {
    const div = document.createElement("div");
    div.className = "sugg-item";
    div.innerHTML = `
      <div class="sugg-title">${escapeHtml(p.player)}</div>
      <div class="sugg-sub">${escapeHtml(p.dataset)} · ${escapeHtml(p.player_position || "—")} · ${escapeHtml(p.role_name || "—")}</div>`;
    div.onclick = () => {
      selected = p;
      search.value = p.player;
      clearSuggestions();
      clearStatus();
      // Reset results pane and collapse it
      setView("landing");
      results.innerHTML = "";
      resetSidebar();
      // Eagerly load the full profile
      loadAndRenderFullProfile(p.player_key, p.dataset);
      setView("profile");
    };
    suggestions.appendChild(div);
  });

  suggestions.classList.remove("hidden");
}

function resetSidebar() {
  sidebarEl.classList.add("muted");
  sidebarEl.textContent = "Click a result card to compare profiles.";
  document.querySelectorAll(".card.active").forEach(c => c.classList.remove("active"));
}

function clearSuggestions() { suggestions.innerHTML = ""; suggestions.classList.add("hidden"); }

function showStatus(msg, isError = false) {
  statusBox.textContent = msg;
  statusBox.classList.remove("hidden");
  statusBox.style.color = isError ? "rgba(255,120,120,0.95)" : "rgba(255,255,255,0.70)";
}
function clearStatus() { statusBox.classList.add("hidden"); statusBox.textContent = ""; }

document.addEventListener("click", e => {
  if (!suggestions.contains(e.target) && e.target !== search) clearSuggestions();
});

search.addEventListener("input", () => {
  // If user edits after selecting, clear selection + collapse profile
  if (selected && search.value.trim() !== selected.player) {
    selected = null;
    setView("landing");
    profileFullEl.innerHTML = "";
  }
});

search.addEventListener("input", async () => {
  const q = search.value.trim();
  clearStatus();
  if (debounceTimer) clearTimeout(debounceTimer);
  if (q.length < 2) { clearSuggestions(); return; }
  debounceTimer = setTimeout(async () => {
    try {
      const data = await safeJsonFetch(`/api/players?q=${encodeURIComponent(q)}&limit=10`);
      renderSuggestions(data);
    } catch (err) {
      showStatus(err.message, true);
      clearSuggestions();
    }
  }, 180);
});

// ============================================================
// Find Similar button
// ============================================================
goBtn.addEventListener("click", async () => {
  clearStatus();
  if (!selected) { showStatus("Pick a player from the dropdown first", true); return; }

  const top_n     = parseInt(topNInput?.value || "10", 10);
  const diff_topk = parseInt(diffTopKInput?.value || "8", 10);
  showStatus("Computing recommendations…");
  resetSidebar();

  try {
    const payload = await safeJsonFetch("/api/recommend", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ player_key: selected.player_key, dataset: selected.dataset, top_n, diff_topk }),
    });

    // Bulk-fetch trait meta for all features about to render
    const featSet = new Set();
    (payload.results || []).forEach(r => {
      (r.biggest_differences || []).forEach(x => featSet.add(x.feature));
      (r.greatest_similarities || []).forEach(x => featSet.add(x.feature));
    });
    await ensureTraitMeta([...featSet]);

    renderResults(payload);
    setView("results");
    patchRenderedTraitElements();
    showStatus(`Found ${payload.results?.length || 0} similar players.`);
  } catch (err) {
    showStatus(err.message, true);
  }
});
