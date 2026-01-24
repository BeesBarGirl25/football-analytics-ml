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
  if (!players.length) {
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

function formatDelta(delta) {
  const d = Number(delta);
  const sign = d >= 0 ? "+" : "";
  return `${sign}${d.toFixed(3)}`;
}

function renderResults(payload) {
  const src = payload.source;
  const recs = payload.results || [];

  results.innerHTML = "";

  if (!recs.length) {
    results.innerHTML = `<div class="card">No recommendations found.</div>`;
    return;
  }

  recs.forEach((r) => {
    const sameRole = (r.shared_roles && r.shared_roles.length > 0);
    const diffs = r.biggest_differences || [];

    const diffsHtml = diffs.map(d => {
      const cls = Number(d.delta) >= 0 ? "delta-pos" : "delta-neg";
      return `
        <div class="item">
          <span>${escapeHtml(d.feature)}</span>
          <span class="${cls}">${escapeHtml(formatDelta(d.delta))}</span>
        </div>
      `;
    }).join("");

    const sharedHtml = sameRole
      ? r.shared_roles.map(nm => `<span class="chip">Same role: ${escapeHtml(nm)}</span>`).join("")
      : `<span class="chip">No role overlap</span>`;

    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-top">
        <div>
          <h3 class="player-name">${escapeHtml(r.player)}</h3>
          <div class="role-name">${escapeHtml(r.role_name || r.role || "—")}</div>
        </div>
        <div class="chip">Similarity: ${Number(r.similarity).toFixed(3)}</div>
      </div>

      <div class="meta">
        <span class="chip">${escapeHtml(r.dataset)}</span>
        <span class="chip">${escapeHtml(r.player_position || "—")}</span>
        ${sameRole ? `<span class="chip">Role match</span>` : ``}
      </div>

      <div class="grid2">
        <div>
          <div class="section-title">Similarities</div>
          <div class="meta">${sharedHtml}</div>
        </div>

        <div>
          <div class="section-title">Biggest differences</div>
          <div class="list">
            ${diffsHtml || `<div class="item"><span>No diffs available</span><span></span></div>`}
          </div>
        </div>
      </div>
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

    // Update selected box based on server’s source (more reliable)
    if (payload.source) showSelected(payload.source);

    renderResults(payload);
    showStatus(`Found ${payload.results?.length || 0} similar players.`);
  } catch (err) {
    showStatus(err.message, true);
  }
});
