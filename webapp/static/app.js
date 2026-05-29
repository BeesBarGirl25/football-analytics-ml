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

const RADAR_CATS = ["Volume", "Progression", "Creativity", "Width", "Distribution", "Style"];

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

// Human-readable delta: unit-aware, no raw decimals
function formatDeltaReadable(feature, delta) {
  const d    = Number(delta);
  const sign = d >= 0 ? "+" : "";
  const n    = String(feature || "");

  if (n.startsWith("pct_"))
    return `${sign}${(d * 100).toFixed(1)}pp`;
  if (n.endsWith("_per90") || n.includes("_per90") || n.endsWith("per_90"))
    return `${sign}${d.toFixed(2)} per 90`;
  if (n.includes("angle"))
    return `${sign}${d.toFixed(1)}°`;
  if (n.startsWith("ttl_"))
    return `${sign}${Math.round(d)}`;
  if (n.startsWith("avg_") || n.startsWith("std_"))
    return `${sign}${d.toFixed(2)}`;
  return `${sign}${d.toFixed(3)}`;
}

// Scouting tier from z-score vs role average
function tierFromZ(z) {
  const n = Number(z);
  if (n >=  2.0) return { label: "Elite",         cls: "tier-elite" };
  if (n >=  1.0) return { label: "Above average",  cls: "tier-high"  };
  if (n >=  0.3) return { label: "Slightly above", cls: "tier-good"  };
  if (n >= -0.3) return { label: "Average",        cls: "tier-avg"   };
  if (n >= -1.0) return { label: "Slightly below", cls: "tier-low"   };
  if (n >= -2.0) return { label: "Below average",  cls: "tier-poor"  };
  return           { label: "Weak",                cls: "tier-weak"  };
}

// How significant is the gap between two players (in role σ units)
function deltaSignificance(deltaZ) {
  if (deltaZ == null) return null;
  const abs = Math.abs(Number(deltaZ));
  if (abs >= 2.0) return { label: "Major",      cls: "sig-major"    };
  if (abs >= 1.0) return { label: "Significant", cls: "sig-notable"  };
  if (abs >= 0.5) return { label: "Noticeable",  cls: "sig-mild"     };
  return                  { label: "Marginal",   cls: "sig-marginal" };
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
    const tier = tierFromZ(z);
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
            <span class="tier-badge ${tier.cls}">${tier.label}</span>
            <span class="z-pct-label">${pctile}th %ile</span>
          </div>
          ${zBarHtml(z)}
          <div class="z-score-secondary">${formatZ(z)}</div>
        </div>
      </div>`;
  }

  // sidebar: compact
  const tier = tierFromZ(z);
  return `
    <div class="sidebar-trait-row">
      <div class="sidebar-trait-header">
        <span class="trait" data-trait="${escapeHtml(key)}" title="${tip}">${escapeHtml(label)}</span>
        <span class="tier-badge ${tier.cls} tier-sm">${tier.label}</span>
      </div>
      ${zBarHtml(z)}
      <div class="sidebar-trait-footer">
        <span class="role-mean">${escapeHtml(t.direction_vs_role || (isPos ? "higher" : "lower"))} vs role · ${pctile}th %ile</span>
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
    </div>
    <div class="profile-full-footer">
      <a class="cmp-profile-link"
         href="/player/${encodeURIComponent(prof.player_key)}/${encodeURIComponent(prof.dataset)}">
        View full profile page →
      </a>
    </div>`;

  patchRenderedTraitElements();
}

// Render a full profile into any element (used by player.html)
function renderFullProfileInto(el, prof) {
  const top = prof.top_traits || [];

  el.innerHTML = `
    <div class="profile-full-header">
      <div>
        <div class="profile-full-name">${escapeHtml(prof.player || "—")}</div>
        <div class="profile-full-chips">
          <span class="chip">${escapeHtml(prof.dataset || "—")}</span>
          <span class="chip">${escapeHtml(prof.player_position || "—")}</span>
          <span class="chip">${escapeHtml(prof.role_name || "—")}</span>
        </div>
      </div>
      <div class="profile-baseline-note">Percentiles vs role peers</div>
    </div>
    <div class="profile-radar-wrap">
      ${buildSinglePlayerRadarSvg(prof.category_summary || [])}
    </div>
    ${buildHighlightsHtml(top)}
    <div class="profile-section-label" style="margin-top:24px;">Detail by category</div>
    ${buildCategoryAccordionHtml(top, prof.category_summary || [])}`;

  patchRenderedTraitElements();
  attachCategoryAccordionHandlers(el);
}

// ============================================================
// Single-player radar SVG
// ============================================================
function buildSinglePlayerRadarSvg(categorySummary) {
  const N = RADAR_CATS.length;
  const CX = 200, CY = 200, R = 140, SIZE = 400, LABEL_PAD = 28;
  const STROKE = "rgba(0,170,255,0.90)";
  const FILL   = "rgba(0,170,255,0.15)";

  const angles = RADAR_CATS.map((_, i) => -Math.PI / 2 + (2 * Math.PI * i) / N);
  const catMap = {};
  for (const c of (categorySummary || [])) catMap[c.category] = c;

  function pt(v, i) { return { x: CX + R * v * Math.cos(angles[i]), y: CY + R * v * Math.sin(angles[i]) }; }
  function polyStr(vals) {
    return vals.map((v, i) => { const p = pt(v, i); return `${p.x.toFixed(1)},${p.y.toFixed(1)}`; }).join(" ");
  }

  const grid = [0.25, 0.5, 0.75, 1.0].map(lvl => {
    const pts = angles.map((_, i) => { const p = pt(lvl, i); return `${p.x.toFixed(1)},${p.y.toFixed(1)}`; }).join(" ");
    const op  = lvl === 1.0 ? "0.18" : lvl === 0.5 ? "0.14" : "0.08";
    return `<polygon points="${pts}" fill="none" stroke="rgba(255,255,255,${op})" stroke-width="${lvl === 0.5 ? 1.5 : 1}"/>`;
  }).join("");

  const spokes = angles.map(a =>
    `<line x1="${CX}" y1="${CY}" x2="${(CX + R * Math.cos(a)).toFixed(1)}" y2="${(CY + R * Math.sin(a)).toFixed(1)}" stroke="rgba(255,255,255,0.10)" stroke-width="1"/>`
  ).join("");

  const labels = RADAR_CATS.map((cat, i) => {
    const a = angles[i];
    const x = (CX + (R + LABEL_PAD) * Math.cos(a)).toFixed(1);
    const y = (CY + (R + LABEL_PAD) * Math.sin(a)).toFixed(1);
    const anchor = Math.cos(a) > 0.15 ? "start" : Math.cos(a) < -0.15 ? "end" : "middle";
    const dy = Math.sin(a) > 0.15 ? "1em" : Math.sin(a) < -0.15 ? "-0.2em" : "0.35em";
    return `<text x="${x}" y="${y}" text-anchor="${anchor}" dy="${dy}" fill="rgba(255,255,255,0.60)" font-size="12" font-family="ui-sans-serif,system-ui">${escapeHtml(cat)}</text>`;
  }).join("");

  const values = RADAR_CATS.map(cat => {
    const d = catMap[cat];
    return d ? Math.max(0.05, Math.min(1, percentileFromZ(d.mean_z) / 100)) : 0.5;
  });

  const dots = values.map((v, i) => {
    const p = pt(v, i);
    return `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.5" fill="${STROKE}"/>`;
  }).join("");

  return `
    <svg viewBox="0 0 ${SIZE} ${SIZE}" xmlns="http://www.w3.org/2000/svg" class="radar-svg">
      ${grid}${spokes}
      <polygon points="${polyStr(values)}" fill="${FILL}" stroke="${STROKE}" stroke-width="2" stroke-linejoin="round"/>
      ${dots}${labels}
    </svg>`;
}

// ============================================================
// Profile highlights — top 3 strengths / weaknesses
// ============================================================
function buildHighlightsHtml(top_traits) {
  const traits = top_traits || [];
  const pos = traits.filter(t => Number(t.z) >= 0).slice(0, 3);
  const neg = traits.filter(t => Number(t.z) <  0).slice(0, 3);

  function item(t, cls) {
    const label = (t.display_name && t.display_name.trim()) ? t.display_name : prettyFeatureName(t.trait);
    const pct   = percentileFromZ(Number(t.z));
    const tip   = escapeHtml(tooltipForTrait(t.trait) || "");
    return `
      <div class="hl-item ${cls}">
        <span class="hl-label trait" data-trait="${escapeHtml(t.trait)}" title="${tip}">${escapeHtml(label)}</span>
        <span class="hl-pct">${pct}th</span>
      </div>`;
  }

  return `
    <div class="highlights-block">
      <div class="highlights-col">
        <div class="highlights-heading hl-pos-heading">▲ Strengths</div>
        ${pos.length ? pos.map(t => item(t, "hl-strength")).join("") : `<div class="hl-none">—</div>`}
      </div>
      <div class="highlights-col">
        <div class="highlights-heading hl-neg-heading">▼ Weaknesses</div>
        ${neg.length ? neg.map(t => item(t, "hl-weakness")).join("") : `<div class="hl-none">—</div>`}
      </div>
    </div>`;
}

// ============================================================
// Category accordion (full profile page)
// ============================================================
function buildCategoryAccordionHtml(top_traits, categorySummary) {
  const traitsByCat = {};
  for (const t of (top_traits || [])) {
    const key = t.category || "Other";
    (traitsByCat[key] = traitsByCat[key] || []).push(t);
  }
  const cats = [...(categorySummary || [])].sort((a, b) => Math.abs(b.mean_z) - Math.abs(a.mean_z));

  return `<div class="cat-accordion">${cats.map(c => {
    const z    = Number(c.mean_z);
    const pct  = percentileFromZ(z);
    const tier = tierFromZ(z);
    const fill = pct >= 65 ? "cat-fill-high" : pct >= 35 ? "cat-fill-mid" : "cat-fill-low";
    const rows = traitsByCat[c.category] || [];
    return `
      <div class="cat-acc-section">
        <button class="cat-acc-header" aria-expanded="false">
          <div class="cat-acc-left">
            <span class="cat-acc-name">${escapeHtml(c.category)}</span>
            <span class="tier-badge ${tier.cls} tier-sm">${tier.label}</span>
          </div>
          <div class="cat-acc-right">
            <div class="cat-acc-track">
              <div class="cat-acc-fill ${fill}" style="width:${Math.max(1, pct)}%;"></div>
            </div>
            <span class="cat-acc-pct">${pct}th</span>
            <span class="cat-acc-chevron" aria-hidden="true">↓</span>
          </div>
        </button>
        <div class="cat-acc-body hidden">
          <div class="trait-section-items">
            ${rows.map(t => traitRowHtml(t, "full")).join("") || `<div class="profile-note">No trait data for this category.</div>`}
          </div>
        </div>
      </div>`;
  }).join("")}</div>`;
}

function attachCategoryAccordionHandlers(root) {
  (root || document).querySelectorAll(".cat-acc-header").forEach(btn => {
    btn.addEventListener("click", () => {
      const section = btn.closest(".cat-acc-section");
      const body    = section.querySelector(".cat-acc-body");
      const open    = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", String(!open));
      section.querySelector(".cat-acc-chevron").textContent = open ? "↓" : "↑";
      body.classList.toggle("hidden", open);
    });
  });
}

// ============================================================
// Reusable player search (profile + compare pages)
// ============================================================
function initPlayerSearch(inputEl, suggestionsEl, onSelect) {
  if (!inputEl || !suggestionsEl) return;
  let timer = null;

  function hide() { suggestionsEl.innerHTML = ""; suggestionsEl.classList.add("hidden"); }

  inputEl.addEventListener("input", () => {
    const q = inputEl.value.trim();
    if (timer) clearTimeout(timer);
    if (q.length < 2) { hide(); return; }
    timer = setTimeout(async () => {
      try {
        const data = await safeJsonFetch(`/api/players?q=${encodeURIComponent(q)}&limit=10`);
        if (!data || !data.length) { hide(); return; }
        suggestionsEl.innerHTML = "";
        data.forEach(p => {
          const div = document.createElement("div");
          div.className = "sugg-item";
          div.innerHTML = `
            <div class="sugg-title">${escapeHtml(p.player)}</div>
            <div class="sugg-sub">${escapeHtml(p.dataset)} · ${escapeHtml(p.player_position || "—")} · ${escapeHtml(p.role_name || "—")}</div>`;
          div.onclick = () => { inputEl.value = p.player; hide(); onSelect(p); };
          suggestionsEl.appendChild(div);
        });
        suggestionsEl.classList.remove("hidden");
      } catch { hide(); }
    }, 180);
  });

  document.addEventListener("click", e => {
    if (!suggestionsEl.contains(e.target) && e.target !== inputEl) hide();
  });
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
  const isPos  = Number(d.delta) >= 0;
  const cls    = isPos ? "delta-pos" : "delta-neg";
  const label  = labelForTrait(d.feature);
  const tip    = escapeHtml(tooltipForTrait(d.feature) || "Loading…");
  const valStr = formatDeltaReadable(d.feature, d.delta);
  const meta   = TRAIT_META.get(String(d.feature || "").trim());
  const sig    = deltaSignificance(d.delta_z);

  const higherMeansHtml = meta?.higher_means
    ? `<div class="delta-higher-means">${escapeHtml(meta.higher_means)}</div>` : "";
  const sigHtml = sig
    ? `<span class="delta-sig ${sig.cls}">${sig.label}</span>` : "";

  return `
    <div class="item">
      <div class="item-left">
        <span class="trait" data-trait="${escapeHtml(d.feature)}" title="${tip}">${escapeHtml(label)}</span>
        ${higherMeansHtml}
      </div>
      <div class="item-right">
        <span class="${cls} delta-val">${escapeHtml(valStr)}</span>
        ${sigHtml}
      </div>
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
      <div class="card-profile-hint">Click card to compare →</div>
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
      if (e.target.closest("summary") || e.target.closest(".trait") || e.target.closest("a")) return;
      const url = `/compare?src_key=${encodeURIComponent(selected.player_key)}&src_dataset=${encodeURIComponent(selected.dataset)}&dst_key=${encodeURIComponent(r.player_key)}&dst_dataset=${encodeURIComponent(r.dataset)}`;
      window.location.href = url;
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
  if (suggestions && !suggestions.contains(e.target) && e.target !== search) clearSuggestions();
});

if (search) {
  search.addEventListener("input", () => {
    if (selected && search.value.trim() !== selected.player) {
      selected = null;
      setView("landing");
      if (profileFullEl) profileFullEl.innerHTML = "";
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

  // Pre-populate from URL params (linked from player profile "Find similar" button)
  const _p = new URLSearchParams(window.location.search);
  const _k = _p.get("src_key"), _d = _p.get("src_dataset");
  if (_k && _d) {
    safeJsonFetch(`/api/player_profile?player_key=${encodeURIComponent(_k)}&dataset=${encodeURIComponent(_d)}&topk=1`)
      .then(prof => {
        if (!prof || !prof.player) return;
        selected = { player_key: _k, dataset: _d, player: prof.player, role_name: prof.role_name, player_position: prof.player_position };
        search.value = prof.player;
        loadAndRenderFullProfile(_k, _d);
        setView("profile");
      }).catch(() => {});
  }
}

// ============================================================
// Find Similar button
// ============================================================
if (goBtn) {
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
}
