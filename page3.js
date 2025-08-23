document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("basketContainer");
  const saveBtn = document.getElementById("saveBtn");

  const stored = localStorage.getItem("basketResult");

  if (!stored) {
    container.innerHTML = `<p class="loading-text">⚠️ No basket found. Please generate one first.</p>`;
    saveBtn.style.display = "none";
    return;
  }

  const parsed = JSON.parse(stored);

  const basket = parsed.basket || parsed.stocks || [];
  const summary = parsed.summary || null;

  renderBasket(basket, summary);

  saveBtn.addEventListener("click", () => {
    alert("✅ Basket saved successfully! (dummy action)");
  });

  function renderBasket(basket, summary) {
    container.innerHTML = "";

    if (!basket || basket.length === 0) {
      container.innerHTML = `<p class="loading-text">⚠️ Basket is empty.</p>`;
      return;
    }

    // 🔹 Basket Summary
    if (summary) {
      const summaryDiv = document.createElement("div");
      summaryDiv.className = "basket-summary glass-card";
      summaryDiv.innerHTML = `
        <h3>📊 Basket Summary</h3>
        <p><b>Theme:</b> ${summary.theme}</p>
        <p><b>Horizon:</b> ${summary.horizon}</p>
        <p><b>Average ROE:</b> ${summary.average_roe}%</p>
        <p><b>Average PE:</b> ${summary.average_pe}</p>
        <p><b>Stocks Returned:</b> ${summary.returned_count}</p>
        <details>
          <summary><b>Applied Filters</b></summary>
          <pre>${JSON.stringify(summary.filters, null, 2)}</pre>
        </details>
      `;
      container.appendChild(summaryDiv);
    }

    // 🔹 Stock cards
    basket.forEach(item => {
      const div = document.createElement("div");
      div.className = "basket-item glass-card";

      div.innerHTML = `
        <div class="basket-left">
          <strong>${item.name || item.symbol}</strong> 
          <span>(${item.symbol})</span>
          <small>${item.sector || ""} • ${item.industry || ""}</small>
        </div>
        <div class="basket-right">
          <p>₹${item.price?.toLocaleString()}</p>
          <small>
            PE: ${item.pe_ratio || "-"} | 
            ROE: ${item.roe || "-"}% | 
            DY: ${item.raw?.dividend_yield || "-"}% | 
            D/E: ${item.debt_equity || "-"}
          </small>
        </div>

        <div class="stock-score">
          <b>Score:</b> ${item.score?.toFixed(2) || "-"} / 100
          <div class="score-bar">
            <div class="score-fill" style="width:${item.score || 0}%"></div>
          </div>
          <details>
            <summary>🔎 Breakdown</summary>
            <pre>${item.score_breakdown?.explain || "No breakdown"}</pre>
          </details>
        </div>

        <div class="stock-reason">
          <p><b>🤖 Why included:</b> ${item.reason || "No reason provided"}</p>
        </div>

        <small class="updated">📅 Updated: ${item.raw?.updated_at || "-"}</small>
      `;
      container.appendChild(div);
    });
  }
});
