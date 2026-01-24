const search = document.getElementById("search");
const suggestions = document.getElementById("suggestions");
const results = document.getElementById("results");
let selected = null;
search.addEventListener("input", async () => {
 const q = search.value;
 if (q.length < 2) return;
 const r = await fetch(`/api/players?q=${encodeURIComponent(q)}`);
 const data = await r.json();
 suggestions.innerHTML = "";
 data.forEach(p => {
 const d = document.createElement("div");
 d.textContent = `${p.player} (${p.dataset})`;
 d.onclick = () => {
 selected = p;
 search.value = p.player;
 suggestions.innerHTML = "";
 };
 suggestions.appendChild(d);
 });
});
document.getElementById("go").addEventListener("click", async () => {
 if (!selected) return;
 const r = await fetch("/api/recommend", {
 method: "POST",
 headers: {"Content-Type": "application/json"},
 body: JSON.stringify({
 player_key: selected.player_key,
 dataset: selected.dataset,
 top_n: 10
 })
 });
 const data = await r.json();
 results.innerHTML = data.results.map(r =>
 `<div><b>${r.player}</b> (${r.similarity.toFixed(3)}) - ${r.role}</div>`
 ).join("");
});
