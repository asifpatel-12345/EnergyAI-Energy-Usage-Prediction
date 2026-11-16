const $ = (id) => document.getElementById(id);

// live slider labels
[["temperature", "tempVal"], ["occupants", "occVal"], ["appliances", "appVal"]].forEach(([inp, lbl]) => {
  $(inp).addEventListener("input", () => { $(lbl).textContent = $(inp).value; });
});

$("predictBtn").addEventListener("click", async () => {
  const btn = $("predictBtn");
  $("error").textContent = "";
  btn.disabled = true;
  btn.textContent = "Calculating...";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        temperature: Number($("temperature").value),
        occupants: Number($("occupants").value),
        appliances: Number($("appliances").value),
        is_weekend: Number($("is_weekend").value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Server error");
    showResults(data);
  } catch (err) {
    $("error").textContent = "Error: " + err.message + " (is the backend running?)";
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict & Optimize";
  }
});

function showResults(d) {
  $("results").classList.remove("hidden");
  $("total").textContent = d.total_kwh;
  $("cost").textContent = d.cost;
  $("saved").textContent = d.saved_kwh + " kWh";
  $("savedPct").textContent = d.saved_percent + "% less";
  $("monthly").textContent = d.monthly_saving;
  $("co2").textContent = d.co2_saved;
  $("tips").innerHTML = d.tips.map((t) => `<li>${t}</li>`).join("");
  $("modelInfo").textContent = `Model: Linear Regression (numpy) | R² = ${d.model.r2} | Avg error = ${d.model.mae} kWh`;
  drawChart(d.forecast, d.optimized);
  $("results").scrollIntoView({ behavior: "smooth" });
}

// Simple canvas line chart (no external library needed, works offline)
function drawChart(a, b) {
  const c = $("chart"), ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = { l: 55, r: 20, t: 20, b: 40 };
  ctx.clearRect(0, 0, W, H);
  const max = Math.ceil(Math.max(...a, ...b) * 1.1 * 10) / 10 || 1;
  const x = (i) => pad.l + (i * (W - pad.l - pad.r)) / 23;
  const y = (v) => H - pad.b - (v / max) * (H - pad.t - pad.b);

  ctx.font = "13px Segoe UI, Arial"; ctx.fillStyle = "#94a3b8"; ctx.strokeStyle = "#334155";
  for (let i = 0; i <= 5; i++) {
    const v = (max * i) / 5, yy = y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke();
    ctx.fillText(v.toFixed(1), 8, yy + 4);
  }
  for (let h = 0; h < 24; h += 3) ctx.fillText(String(h).padStart(2, "0") + ":00", x(h) - 18, H - 14);
  ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("kWh per hour", -35, 0); ctx.restore();

  const line = (arr, color) => {
    ctx.beginPath(); ctx.lineWidth = 3; ctx.strokeStyle = color;
    arr.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
    ctx.fillStyle = color;
    arr.forEach((v, i) => { ctx.beginPath(); ctx.arc(x(i), y(v), 3, 0, Math.PI * 2); ctx.fill(); });
  };
  line(a, "#ef4444");
  line(b, "#22c55e");
}
