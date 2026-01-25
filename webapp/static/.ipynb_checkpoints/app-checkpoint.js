const search = document.getElementById("search");
const suggestions = document.getElementById("suggestions");
const results = document.getElementById("results");
const selectedBox = document.getElementById("selected");
const statusBox = document.getElementById("status");
const topNInput = document.getElementById("top_n");
const diffTopKInput = document.getElementById("diff_topk");
const goBtn = document.getElementById("go");

let selected = null;
let debounceTimer = null;

/**
 * Glossary cache.
 * We cache the *full meta* per trait so we can show nice labels + tooltip text.
 * Shape:
 *  TRAIT_META.get("pct_pass_mid_to_att") -> {trait, display_name, description, category, higher_means}
 */
const TRAIT_META = new Map();

/**
 * Fetch trait meta in bulk (fast).
 * Backend: POST /api/traits { traits: ["a","b"] } -> { traits: { "a": {...}, "b": {...} } }
 */
async function ensureTraitMeta(traits) {
  const missing = [];
  for (const t of traits) {
    const key = String(t || "").trim();
    if (!key) continue;
    if (!TRAIT_META.has(key)) missing.push(key);
  }
  if (!missing.length) return;

  const res = await safeJsonFetch("/api/traits", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ traits: [...new Set(missing)].slice(0, 500) })
  });

  const meta = res?.traits || {};
  for (const [k, v] of Object.entries(meta)) {
    TRAIT_META.set(k, v);
  }
}

/**
 * Single-trait fallback (kept for robustness).
 * If the bulk endpoint fails, we can still lazily fetch per-trait.
 */
const traitTooltipCache = new Map();
async function getTraitTooltipTextFallback(trait) {
  const key = String(trait || "").trim();
  if (!key) return null;

  if (traitTooltipCache.has(key)) return traitTooltipCache.get(key);

  const p = (async () => {
    const data = await safeJsonFetch(`/api/trait?trait=${encodeURIComponent(key)}`);
    const parts = [];
    if (data?.description) parts.push(data.description);
    if (data?.higher_means) parts.push(`Higher means: ${data.higher_means}`);
    parts.push(`Key: ${key}`);
    return parts.join("\n\n") || "No glossary entry yet.";
  })();

  traitTooltipCache.set(key, p);

  try {
    const text = await p;
    traitTooltipCache.set(key, text);
    return text;
  } catch (e) {
    traitTooltipCache.delete(key);
    return "Could not load glossary entry.";
  }
}

function tooltipTextForTrait(traitKey) {
  const key = String(traitKey || "").trim();
  const m = TRAIT_META.get(key);
  if (!m) return null;

  const parts = [];
  if (m.description) parts.push(m.description);
  if (m.higher_means) parts.push(`Higher means: ${m.higher_means}`);
  parts.push(`Key: ${key}`);
  return parts.join("\n\n");
}

function labelForTrait(traitKey) {
  const key = String(traitKey || "").trim();
  const m = TRAIT_META.get(key);
  if (m && m.display_name) return m.display_name;
  return prettyFeatureName(key);
}

/**
 * Hover handler:
 * - if we already have meta from bulk fetch, use it immediately
 * - else fallback to the per-trait endpoint (still works)
 */
document.addEventListener("mouseover", async (e) => {
  const el = e.target.closest("[data-trait]");
  if (!el) return;

  // don't refetch if already set
  if (el.dataset.tooltipLoaded === "1") return;

  const key = String(el.dataset.trait || "").trim();
  if (!key) return;

  // 1) if bulk meta exists, use it immediately
  const bulkTip = tooltipTextForTrait(key);
  if (bulkTip) {
    el.title = bulkTip;
    el.dataset.tooltipLoaded = "1";
    return;
  }

  // 2) fallback: lazy fetch single trait
  el.title = "Loading…";
  const text = await getTraitTooltipTextFallback(key);
  if (text) el.title = text;
  el.dataset.tooltipLoaded = "1";
});

function prettyFeatureName(raw) {
  const s = String(raw || "");

  // quick humaniser
  let out = s
    .replace(/^pct_/, "")
    .replace(/_per90$|_per_90$/g, " per90")
    .replace(/_/g, " ");

  // nicer phrases
  out = out
    .replace("pass angle mean", "Pass angle (mean)")
    .replace("pass angle var", "Pass angle (variance)")
    .replace("ttl passes", "Total passes")
    .replace("pct passes", "% passes");

  // thirds/channels
  out = out
    .replace("def third", "defensive third")
    .replace("mid third", "middle third")
    .replace("att third", "attacking third")
    .replace("centre channel", "central channel")
    .replace("left channel", "left channel")
    .replace("right channel", "right channel");

  // title-case-ish
  return out.charAt(0).toUpperCase() + out.slice(1);
}

function featureUnitHint(name) {
  const n = String(name || "");
  if (n.includes("angle")) return "deg";
  if (n.startsWith("pct_") || n.includes("pct")) return "%/share";
  if (n.includes("per90") || n.includes("per_90")) return "per90";
  return "";
}

function formatDeltaWithHint(feature, delta) {
  const d = Number(delta);
  const sign = d >= 0 ? "+" : "";
  const hint = featureUnitHint(feature);
  return `${sign}${d.toFixed(3)}${hint ? " " + hint : ""}`;
}

function showStatus(msg, isError = false) {
  statusBox.textContent = msg;
  statusBox.classList.remove("hidden");
  statusBox.style.color = isError ? "rgba(255,120,120,0.95)" : "rgba(255,255,255,0.70)";
}

function clearStatus() {
  statusBox.classList.add("hidden");
  statusBox.textContent = "";
}

function showSelected(p) {
  selectedBox.classList.remove("hidden");
  selectedBox.innerHTML = `
    <b>Selected:</b> ${escapeHtml(p.player)}
    <span style="opacity:.75">(${escapeHtml(p.dataset)} · ${escapeHtml(p.player_position || "—")} · ${escapeHtml(p.role_name || "—")})</span>
  `;
}

function clearSelected() {
  selectedBox.classList.add("hidden");
  selectedBox.innerHTML = "";
}

function clearSuggestions() {
  suggestions.innerHTML = "";
  suggestions.classList.add("hidden");
}

function openSuggestions() {
  suggestions.classList.remove("hidden");
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[m]));
}

async function safeJsonFetch(url, options = {}) {
  const res = await fetch(url, options);

  // If backend crashes, Heroku might return HTML. Don’t try JSON.parse on that.
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    const text = await res.text();
    throw new Error(`Non-JSON response (${res.status}). First 120 chars: ${text.slice(0, 120)}`);
  }

  const data = await res.json();
  if (!res.ok) {
    const msg = data?.error ? `${data.error}` : `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

function renderSuggestions(players) {
  suggestions.innerHTML = "";
  if (!players || !players.length) {
    clearSuggestions();
    return;
  }

  players.forEach((p) => {
    const div = document.createElement("div");
    div.className = "sugg-item";

    div.innerHTML = `
      <div class="sugg-title">${escapeHtml(p.player)}</div>
      <div class="sugg-sub">${escapeHtml(p.dataset)} · ${escapeHtml(p.player_position || "—")} · ${escapeHtml(p.role_name || "—")}</div>
    `;

    div.onclick = () => {
      selected = p;
      search.value = p.player;
      clearSuggestions();
      clearStatus();
      showSelected(p);
    };

    suggestions.appendChild(div);
  });

  openSuggestions();
}

function renderResults(payload) {
  const recs = payload.results || [];
  results.innerHTML = "";

  if (!recs.length) {
    results.innerHTML = `<div class="card">No recommendations found.</div>`;
    return;
  }

  recs.forEach((r, idx) => {
    const diffs = r.biggest_differences || [];
    const sims = r.greatest_similarities || [];

    // backend might send why_similar (array of strings) OR shared_roles
    const shared =
      Array.isArray(r.shared_roles) ? r.shared_roles.slice(0, 3)
      : Array.isArray(r.why_similar) ? r.why_similar.slice(0, 3)
      : [];

    const sharedHtml = shared.length
      ? shared.map(txt => `<span class="chip">${escapeHtml(txt)}</span>`).join("")
      : `<span class="chip">No role overlap</span>`;

    const diffsHtml = diffs.map(d => {
      const cls = Number(d.delta) >= 0 ? "delta-pos" : "delta-neg";
      const label = labelForTrait(d.feature);
      return `
        <div class="item">
          <span class="trait" data-trait="${escapeHtml(d.feature)}" title="${escapeHtml(tooltipTextForTrait(d.feature) || "Loading…")}">
            ${escapeHtml(label)}
          </span>
          <span class="${cls}">${escapeHtml(formatDeltaWithHint(d.feature, d.delta))}</span>
        </div>
      `;
    }).join("");

    const simsHtml = sims.map(s => {
      const cls = Number(s.delta) >= 0 ? "delta-pos" : "delta-neg";
      const label = labelForTrait(s.feature);
      return `
        <div class="item">
          <span class="trait" data-trait="${escapeHtml(s.feature)}" title="${escapeHtml(tooltipTextForTrait(s.feature) || "Loading…")}">
            ${escapeHtml(label)}
          </span>
          <span class="${cls}">${escapeHtml(formatDeltaWithHint(s.feature, s.delta))}</span>
        </div>
      `;
    }).join("");

    const simVal = Number(r.similarity);
    const simShown = Number.isFinite(simVal) ? simVal.toFixed(3) : "—";
    const simPct = Number.isFinite(simVal)
      ? (Math.max(0, Math.min(1, simVal)) * 100)
      : 0;

    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="rank-badge">#${idx + 1}</div>

      <div class="card-top">
        <div>
          <h3 class="player-name">${escapeHtml(r.player)}</h3>
          <div class="role-name">${escapeHtml(r.role_name || r.role || "—")}</div>
        </div>
        <div class="chip">Similarity: ${simShown}</div>
      </div>

      <div class="simbar"><div style="width:${simPct.toFixed(1)}%"></div></div>

      <div class="meta">
        <span class="chip">${escapeHtml(r.dataset)}</span>
        <span class="chip">${escapeHtml(r.player_position || "—")}</span>
        ${(shared.length ? `<span class="chip">Role match</span>` : ``)}
      </div>

      <details open>
        <summary>
          <span class="summary-title">Similarities</span>
          <span class="kicker">
            ${sims.length ? `Closest features (Top ${sims.length})` : (shared.length ? "Shared role fingerprint" : "No similarity info")}
          </span>
        </summary>

        ${sims.length ? `
          <div class="list" style="margin-top:10px;">
            ${simsHtml}
          </div>
        ` : `
          <div class="meta" style="margin-top:10px;">${sharedHtml}</div>
        `}
      </details>

      <details>
        <summary>
          <span class="summary-title">Biggest differences</span>
          <span class="kicker">Top ${diffs.length || 0}</span>
        </summary>
        <div class="list" style="margin-top:10px;">
          ${diffsHtml || `<div class="item"><span>No diffs available</span><span></span></div>`}
        </div>
      </details>
    `;

    results.appendChild(card);
  });
}

// Close suggestions when clicking elsewhere
document.addEventListener("click", (e) => {
  if (!suggestions.contains(e.target) && e.target !== search) {
    clearSuggestions();
  }
});

// If user types after selecting, unset selection
search.addEventListener("input", () => {
  if (selected && search.value.trim() !== selected.player) {
    selected = null;
    clearSelected();
  }
});

// Debounced search
search.addEventListener("input", async () => {
  const q = search.value.trim();
  clearStatus();

  if (debounceTimer) clearTimeout(debounceTimer);

  if (q.length < 2) {
    clearSuggestions();
    return;
  }

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

goBtn.addEventListener("click", async () => {
  clearStatus();

  if (!selected) {
    showStatus("Pick a player from the dropdown first 👆", true);
    return;
  }

  const top_n = parseInt(topNInput?.value || "10", 10);
  const diff_topk = parseInt(diffTopKInput?.value || "8", 10);

  showStatus("Computing recommendations…");

  try {
    const payload = await safeJsonFetch("/api/recommend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        player_key: selected.player_key,
        dataset: selected.dataset,
        top_n,
        diff_topk
      })
    });

    if (payload.source) showSelected(payload.source);

    // Bulk fetch glossary meta for everything we’re about to render
    const featSet = new Set();
    (payload.results || []).forEach(r => {
      (r.biggest_differences || []).forEach(x => featSet.add(x.feature));
      (r.greatest_similarities || []).forEach(x => featSet.add(x.feature));
    });
    await ensureTraitMeta([...featSet]);

    renderResults(payload);
    showStatus(`Found ${payload.results?.length || 0} similar players.`);
  } catch (err) {
    showStatus(err.message, true);
  }
});
