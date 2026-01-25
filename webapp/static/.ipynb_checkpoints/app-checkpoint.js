const search = document.getElementById("search");
const suggestions = document.getElementById("suggestions");
const results = document.getElementById("results");
const selectedBox = document.getElementById("selected");
const statusBox = document.getElementById("status");
const topNInput = document.getElementById("top_n");
const diffTopKInput = document.getElementById("diff_topk");
const goBtn = document.getElementById("go");

// ✅ NEW: right-hand profile container from index.html
const profileBox = document.getElementById("profile");

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

function simBucket(sim) {
  const s = Number(sim);
  if (!Number.isFinite(s)) return "sim-unk";

  if (s >= 0.92) return "sim-elite";   // basically clones
  if (s >= 0.85) return "sim-hi";      // very strong
  if (s >= 0.78) return "sim-good";    // strong
  if (s >= 0.70) return "sim-mid";     // decent
  if (s >= 0.62) return "sim-ok";      // meh+
  if (s >= 0.55) return "sim-low";     // weak
  return "sim-vlow";                   // nah
}


function formatZ(z) {
  const n = Number(z);
  if (!Number.isFinite(n)) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}σ`;
}

function formatVal(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(3);
}

async function loadAndRenderProfile(player_key, dataset) {
  if (!profileBox) return;

  profileBox.classList.remove("muted");
  profileBox.innerHTML = `<div style="opacity:.8">Loading profile…</div>`;

  const prof = await safeJsonFetch(
    `/api/player_profile?player_key=${encodeURIComponent(player_key)}&dataset=${encodeURIComponent(dataset)}&topk=12`
  );

  // Ensure trait meta for profile traits too
  const traitKeys = (prof.top_traits || []).map(t => t.trait);
  await ensureTraitMeta(traitKeys);

  renderProfileFromProfilePayload(prof);

  // patch tooltips/labels now that TRAIT_META is populated
  patchRenderedTraitElements();
}

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
 * After bulk fetch, update any currently-rendered trait elements
 * so tooltips + labels stop saying "Loading…".
 */
function patchRenderedTraitElements() {
  document.querySelectorAll("[data-trait]").forEach((el) => {
    const key = String(el.dataset.trait || "").trim();
    if (!key) return;

    // label
    const label = labelForTrait(key);
    if (label && el.textContent !== label) el.textContent = label;

    // tooltip
    const tip = tooltipTextForTrait(key);
    if (tip) {
      el.title = tip;
      el.dataset.tooltipLoaded = "1";
    }
  });
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

      // ✅ optional: wipe profile until user hits Go
      if (profileBox) {
        profileBox.classList.remove("muted");
        profileBox.innerHTML = `
          <div style="opacity:.8">Ready when you are. Hit <b>Go</b> to load this player’s profile + comparisons.</div>
        `;
      }
    };

    suggestions.appendChild(div);
  });

  openSuggestions();
}

/**
 * ✅ NEW: render right-hand player profile panel
 * We use payload.source, and (optionally) the first neighbour to show a quick "style snapshot".
 */
function renderProfileFromProfilePayload(prof) {
  if (!profileBox) return;

  const top = prof?.top_traits || [];
  const baseline = prof?.baseline === "role" ? "role peers" : "dataset";

  const listHtml = top.length
    ? top.map(t => {
        const key = t.trait;
        const label = labelForTrait(key);
        const z = Number(t.z);
        const zCls = z >= 0 ? "z-pos" : "z-neg";
        const dir = t.direction_vs_role || (z >= 0 ? "higher" : "lower");

        // tooltip uses our glossary text if available
        const tip = tooltipTextForTrait(key) || "Loading…";

        return `
          <div class="profile-item">
            <div>
              <div class="trait" data-trait="${escapeHtml(key)}" title="${escapeHtml(tip)}">
                ${escapeHtml(label)}
              </div>
              <div class="profile-val" style="margin-top:4px;">
                ${escapeHtml(dir)} vs ${escapeHtml(baseline)}
              </div>
            </div>
            <div class="rhs">
              <div class="profile-z ${zCls}">${escapeHtml(formatZ(z))}</div>
              <div class="profile-val">val: ${escapeHtml(formatVal(t.value))}</div>
            </div>
          </div>
        `;
      }).join("")
    : `<div class="profile-note">No profile traits returned.</div>`;

  profileBox.classList.remove("muted");
  profileBox.innerHTML = `
    <div class="profile-header">
      <div class="profile-name">${escapeHtml(prof.player || "—")}</div>
      <div class="profile-sub">${escapeHtml(prof.dataset || "—")} · ${escapeHtml(prof.player_position || "—")}</div>
      <div class="profile-role"><b>Role:</b> ${escapeHtml(prof.role_name || "—")}</div>
    </div>

    <div class="profile-section-title">Top traits (vs role)</div>
    <div class="profile-note">
      These are the traits where this player differs most from their <b>${escapeHtml(baseline)}</b>.
    </div>

    <div class="profile-list">
      ${listHtml}
    </div>
  `;
}

function renderProfile(payload) {
  // keep this as a lightweight wrapper so existing call sites don’t explode
  const src = payload?.source;
  if (!src || !profileBox) return;

  // We now render via the proper endpoint
  loadAndRenderProfile(src.player_key, src.dataset).catch(err => {
    profileBox.classList.add("muted");
    profileBox.textContent = `Could not load profile: ${err.message}`;
  });
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


    const bucket = simBucket(simVal);
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

      <div class="simbar ${bucket}">
        <div style="width:${simPct.toFixed(1)}%"></div>
      </div>

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
    if (profileBox) {
      profileBox.classList.add("muted");
      profileBox.textContent = "Search and select a player to see their profile.";
    }
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

    // ✅ NEW: render profile (uses trait labels too)
    renderProfile(payload);

    renderResults(payload);

    // ✅ NEW: now we have meta, patch labels + titles that were rendered as Loading…
    patchRenderedTraitElements();

    showStatus(`Found ${payload.results?.length || 0} similar players.`);
  } catch (err) {
    showStatus(err.message, true);
  }
});
