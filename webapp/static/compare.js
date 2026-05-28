// ── Constants ─────────────────────────────────────────────────────────────────
// Clockwise from top — must match fill_trait_dictionary RADAR_CATEGORY_ORDER
const RADAR_CATS = ["Volume", "Progression", "Creativity", "Width", "Distribution", "Style"];

const COLOR_A = { fill: "rgba(0,170,255,0.18)",  stroke: "rgba(0,170,255,0.90)",  text: "#00aaff" };
const COLOR_B = { fill: "rgba(255,185,0,0.15)",   stroke: "rgba(255,185,0,0.90)",  text: "#ffb900" };

// ── Boot ──────────────────────────────────────────────────────────────────────
(async () => {
  const root = document.getElementById("compare-root");
  if (!SRC_KEY || !DST_KEY) {
    root.innerHTML = `<div class="profile-error">Missing player parameters.</div>`;
    return;
  }

  try {
    // Fetch both profiles in parallel (topk=200 to get all features)
    const [profA, profB] = await Promise.all([
      safeJsonFetch(`/api/player_profile?player_key=${encodeURIComponent(SRC_KEY)}&dataset=${encodeURIComponent(SRC_DATASET)}&topk=200`),
      safeJsonFetch(`/api/player_profile?player_key=${encodeURIComponent(DST_KEY)}&dataset=${encodeURIComponent(DST_DATASET)}&topk=200`),
    ]);

    // Bulk-load trait meta for everything we'll render
    const allTraits = [
      ...(profA.top_traits || []).map(t => t.trait),
      ...(profB.top_traits || []).map(t => t.trait),
    ];
    await ensureTraitMeta(allTraits);

    document.title = `${profA.player} vs ${profB.player}`;

    root.innerHTML = buildCompareHtml(profA, profB);
    attachExpandHandlers();
    patchRenderedTraitElements();

  } catch (err) {
    root.innerHTML = `<div class="profile-error">Could not load comparison: ${escapeHtml(err.message)}</div>`;
  }
})();

// ── Main HTML builder ─────────────────────────────────────────────────────────
function buildCompareHtml(profA, profB) {
  const catMapA = buildCatMap(profA.category_summary || []);
  const catMapB = buildCatMap(profB.category_summary || []);
  const traitMapA = buildTraitMap(profA.top_traits || []);
  const traitMapB = buildTraitMap(profB.top_traits || []);

  const sameRole = profA.role_id === profB.role_id;
  const roleNote = sameRole
    ? `Both players: <strong>${escapeHtml(profA.role_name || "—")}</strong>`
    : `<span style="color:${COLOR_A.text}">${escapeHtml(profA.role_name || "—")}</span>
       vs
       <span style="color:${COLOR_B.text}">${escapeHtml(profB.role_name || "—")}</span>
       — percentiles are within each player's own role`;

  return `
    <!-- Player headers -->
    <div class="cmp-player-header">
      <div class="cmp-player-card" style="border-top:3px solid ${COLOR_A.stroke}">
        <div class="cmp-player-name">${escapeHtml(profA.player || "—")}</div>
        <div class="cmp-player-meta">
          <span class="chip">${escapeHtml(profA.dataset || "—")}</span>
          <span class="chip">${escapeHtml(profA.player_position || "—")}</span>
          <span class="chip">${escapeHtml(profA.role_name || "—")}</span>
        </div>
        <a href="/player/${encodeURIComponent(SRC_KEY)}/${encodeURIComponent(SRC_DATASET)}"
           class="cmp-profile-link">View full profile →</a>
      </div>

      <div class="cmp-vs">vs</div>

      <div class="cmp-player-card" style="border-top:3px solid ${COLOR_B.stroke}">
        <div class="cmp-player-name">${escapeHtml(profB.player || "—")}</div>
        <div class="cmp-player-meta">
          <span class="chip">${escapeHtml(profB.dataset || "—")}</span>
          <span class="chip">${escapeHtml(profB.player_position || "—")}</span>
          <span class="chip">${escapeHtml(profB.role_name || "—")}</span>
        </div>
        <a href="/player/${encodeURIComponent(DST_KEY)}/${encodeURIComponent(DST_DATASET)}"
           class="cmp-profile-link">View full profile →</a>
      </div>
    </div>

    <div class="cmp-role-note">${roleNote}</div>

    <!-- Radar -->
    <div class="cmp-radar-wrap">
      ${buildRadarSvg(catMapA, catMapB)}
      <div class="cmp-radar-legend">
        <span class="legend-dot" style="background:${COLOR_A.stroke}"></span>
        <span style="color:${COLOR_A.text}">${escapeHtml(profA.player || "A")}</span>
        <span class="legend-dot" style="background:${COLOR_B.stroke};margin-left:16px;"></span>
        <span style="color:${COLOR_B.text}">${escapeHtml(profB.player || "B")}</span>
      </div>
    </div>

    <!-- Category bars -->
    <div class="cmp-categories">
      ${RADAR_CATS.map(cat => buildCategorySection(cat, catMapA[cat], catMapB[cat], traitMapA, traitMapB, profA, profB)).join("")}
    </div>
  `;
}

// ── Radar SVG ────────────────────────────────────────────────────────────────
function buildRadarSvg(catMapA, catMapB) {
  const N = RADAR_CATS.length;
  const CX = 220, CY = 220, R = 150, SIZE = 440;
  const LABEL_PAD = 28;

  // Angles: start at top (−90°), clockwise
  const angles = RADAR_CATS.map((_, i) => -Math.PI / 2 + (2 * Math.PI * i) / N);

  function pt(v, i) {
    return { x: CX + R * v * Math.cos(angles[i]), y: CY + R * v * Math.sin(angles[i]) };
  }

  function polyStr(values) {
    return values.map((v, i) => { const p = pt(v, i); return `${p.x.toFixed(1)},${p.y.toFixed(1)}`; }).join(" ");
  }

  // Background grid (25 / 50 / 75 / 100%)
  const gridLines = [0.25, 0.5, 0.75, 1.0].map(level => {
    const pts = angles.map((_, i) => { const p = pt(level, i); return `${p.x.toFixed(1)},${p.y.toFixed(1)}`; }).join(" ");
    const opacity = level === 1.0 ? "0.18" : "0.09";
    return `<polygon points="${pts}" fill="none" stroke="rgba(255,255,255,${opacity})" stroke-width="1"/>`;
  }).join("");

  // Spokes
  const spokes = angles.map(a => {
    const x = (CX + R * Math.cos(a)).toFixed(1);
    const y = (CY + R * Math.sin(a)).toFixed(1);
    return `<line x1="${CX}" y1="${CY}" x2="${x}" y2="${y}" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>`;
  }).join("");

  // Labels
  const labels = RADAR_CATS.map((cat, i) => {
    const a = angles[i];
    const x = (CX + (R + LABEL_PAD) * Math.cos(a)).toFixed(1);
    const y = (CY + (R + LABEL_PAD) * Math.sin(a)).toFixed(1);
    const anchor = Math.cos(a) > 0.15 ? "start" : Math.cos(a) < -0.15 ? "end" : "middle";
    const dy = Math.sin(a) > 0.15 ? "1em" : Math.sin(a) < -0.15 ? "-0.2em" : "0.35em";
    return `<text x="${x}" y="${y}" text-anchor="${anchor}" dy="${dy}"
              fill="rgba(255,255,255,0.62)" font-size="12.5" font-family="ui-sans-serif,system-ui">${cat}</text>`;
  }).join("");

  // Player polygons
  function catValue(catMap, cat) {
    const d = catMap[cat];
    if (!d) return 0.5; // no data → centre
    return Math.max(0.05, Math.min(1, percentileFromZ(d.mean_z) / 100));
  }

  const valuesA = RADAR_CATS.map(c => catValue(catMapA, c));
  const valuesB = RADAR_CATS.map(c => catValue(catMapB, c));

  // Dot markers at each vertex
  const dotsA = valuesA.map((v, i) => {
    const p = pt(v, i);
    return `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.5" fill="${COLOR_A.stroke}"/>`;
  }).join("");
  const dotsB = valuesB.map((v, i) => {
    const p = pt(v, i);
    return `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.5" fill="${COLOR_B.stroke}"/>`;
  }).join("");

  return `
    <svg viewBox="0 0 ${SIZE} ${SIZE}" xmlns="http://www.w3.org/2000/svg" class="radar-svg">
      ${gridLines}
      ${spokes}
      <polygon points="${polyStr(valuesA)}" fill="${COLOR_A.fill}" stroke="${COLOR_A.stroke}" stroke-width="2" stroke-linejoin="round"/>
      <polygon points="${polyStr(valuesB)}" fill="${COLOR_B.fill}" stroke="${COLOR_B.stroke}" stroke-width="2" stroke-linejoin="round"/>
      ${dotsA}${dotsB}
      ${labels}
    </svg>`;
}

// ── Category section ──────────────────────────────────────────────────────────
function buildCategorySection(cat, catA, catB, traitMapA, traitMapB, profA, profB) {
  const pctA = catA ? percentileFromZ(catA.mean_z) : null;
  const pctB = catB ? percentileFromZ(catB.mean_z) : null;
  const tierA = catA ? tierFromZ(catA.mean_z) : null;
  const tierB = catB ? tierFromZ(catB.mean_z) : null;

  const barA = pctA !== null ? playerBarHtml(pctA, tierA, profA.player, "bar-a") : missingBarHtml(profA.player);
  const barB = pctB !== null ? playerBarHtml(pctB, tierB, profB.player, "bar-b") : missingBarHtml(profB.player);

  // Shared traits in this category
  const sharedTraits = getSharedTraits(cat, traitMapA, traitMapB);
  const detailRows = sharedTraits.map(key => buildTraitCompareRow(key, traitMapA[key], traitMapB[key])).join("");

  return `
    <div class="cmp-category">
      <div class="cmp-cat-header">
        <span class="cmp-cat-name">${escapeHtml(cat)}</span>
        <button class="cmp-expand-btn" aria-expanded="false">Show traits ↓</button>
      </div>
      <div class="cmp-cat-bars">
        ${barA}
        ${barB}
      </div>
      <div class="cmp-trait-detail hidden">
        ${detailRows || `<div class="profile-note">No individual trait data available.</div>`}
      </div>
    </div>`;
}

function playerBarHtml(pct, tier, playerName, cls) {
  const w = Math.max(2, pct);
  return `
    <div class="cmp-bar-row">
      <div class="cmp-bar-label">${escapeHtml(playerName || "—")}</div>
      <div class="cmp-bar-track">
        <div class="cmp-bar-fill ${cls}" style="width:${w}%"></div>
      </div>
      <div class="cmp-bar-meta">
        <span class="tier-badge ${tier.cls} tier-sm">${tier.label}</span>
        <span class="cmp-pct">${pct}th %ile</span>
      </div>
    </div>`;
}

function missingBarHtml(playerName) {
  return `
    <div class="cmp-bar-row">
      <div class="cmp-bar-label">${escapeHtml(playerName || "—")}</div>
      <div class="cmp-bar-track"><div class="cmp-bar-fill bar-missing" style="width:2%"></div></div>
      <div class="cmp-bar-meta"><span class="cmp-pct muted">No data</span></div>
    </div>`;
}

// ── Trait-level comparison row ────────────────────────────────────────────────
function buildTraitCompareRow(key, tA, tB) {
  const label = labelForTrait(key);
  const tip   = escapeHtml(tooltipForTrait(key) || "");
  const meta  = TRAIT_META.get(String(key).trim());

  function traitBar(t, cls) {
    if (!t) return `<div class="trait-cmp-bar-wrap"><div class="trait-cmp-track"><div class="cmp-bar-fill bar-missing" style="width:2%"></div></div><span class="cmp-pct muted">—</span></div>`;
    const pct = percentileFromZ(Number(t.z));
    const w   = Math.max(2, pct);
    return `
      <div class="trait-cmp-bar-wrap">
        <div class="trait-cmp-track">
          <div class="cmp-bar-fill ${cls}" style="width:${w}%"></div>
        </div>
        <span class="cmp-pct">${pct}th %ile · ${formatVal(t.value)}</span>
      </div>`;
  }

  return `
    <div class="trait-cmp-row">
      <div class="trait-cmp-name">
        <span class="trait" data-trait="${escapeHtml(key)}" title="${tip}">${escapeHtml(label)}</span>
        ${meta?.higher_means ? `<div class="delta-higher-means">${escapeHtml(meta.higher_means)}</div>` : ""}
      </div>
      ${traitBar(tA, "bar-a")}
      ${traitBar(tB, "bar-b")}
    </div>`;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function buildCatMap(categorySummary) {
  const m = {};
  for (const c of categorySummary) m[c.category] = c;
  return m;
}

function buildTraitMap(traits) {
  const m = {};
  for (const t of traits) m[t.trait] = t;
  return m;
}

function getSharedTraits(cat, traitMapA, traitMapB) {
  // traits present in either player that belong to this category
  const keys = new Set([
    ...Object.values(traitMapA).filter(t => t.category === cat).map(t => t.trait),
    ...Object.values(traitMapB).filter(t => t.category === cat).map(t => t.trait),
  ]);
  // Sort by average |z| descending so most distinguishing traits appear first
  return [...keys].sort((a, b) => {
    const zA = Math.abs(traitMapA[a]?.z || 0);
    const zB = Math.abs(traitMapB[b]?.z || 0);
    return (Math.abs(traitMapA[b]?.z || 0) + Math.abs(traitMapB[b]?.z || 0)) -
           (Math.abs(traitMapA[a]?.z || 0) + Math.abs(traitMapB[a]?.z || 0));
  });
}

// ── Expand/collapse handlers ──────────────────────────────────────────────────
function attachExpandHandlers() {
  document.querySelectorAll(".cmp-expand-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const section = btn.closest(".cmp-category");
      const detail  = section.querySelector(".cmp-trait-detail");
      const open    = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", !open);
      btn.textContent = open ? "Show traits ↓" : "Hide traits ↑";
      detail.classList.toggle("hidden", open);
    });
  });
}

// renderFullProfileInto is defined in app.js (shared with player.html)
