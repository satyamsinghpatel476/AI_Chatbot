const SYSTEMS = ["A", "B", "C"];
const SYSTEM_LABELS = {
  A: "System A",
  B: "System B",
  C: "System C",
};
const SYSTEM_COLORS = {
  A: "#2563eb",
  B: "#f97316",
  C: "#16a34a",
};

function escapeHtml(value) {
  return String(value ?? "N/A")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatValue(value, places = 2, suffix = "") {
  if (value === null || value === undefined || value === "N/A") {
    return "N/A";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return escapeHtml(value);
  }
  if (places === 0) {
    return `${Math.round(number)}${suffix}`;
  }
  return `${number.toFixed(places)}${suffix}`;
}

function metricLabel(metric) {
  return metric.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function showError(message) {
  const box = document.getElementById("errorBox");
  box.hidden = false;
  box.textContent = message;
}

async function loadSummary() {
  const response = await fetch("results_summary.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(
      "Could not load results_summary.json. Run python web_page/export_results.py, then refresh this page."
    );
  }
  return response.json();
}

function renderHeader(data) {
  const source = data.source || {};
  document.getElementById("lastUpdated").textContent = [
    `Exported: ${source.loaded_at || data.generated_at || "N/A"}`,
    `File modified: ${source.modified_at || "N/A"}`,
  ].join(" | ");
}

function renderSummaryCards(data) {
  const cards = data.summary_cards || {};
  const cardRows = [
    {
      label: "Total questions",
      value: cards.total_questions ?? 0,
      detail: "Current export",
    },
    {
      label: "Best accuracy system",
      value: cards.best_accuracy_system?.label || "N/A",
      detail: formatValue(cards.best_accuracy_system?.value, 2),
    },
    {
      label: "Fastest system",
      value: cards.fastest_system?.label || "N/A",
      detail:
        cards.fastest_system?.value === null || cards.fastest_system?.value === undefined
          ? "N/A"
          : formatValue(cards.fastest_system.value, 2, " s"),
    },
    {
      label: "Lowest contamination system",
      value: cards.lowest_contamination_system?.label || "N/A",
      detail: formatValue(cards.lowest_contamination_system?.value, 0),
    },
    {
      label: "Best cross-domain robustness system",
      value: cards.best_cross_domain_robustness_system?.label || "N/A",
      detail: formatValue(cards.best_cross_domain_robustness_system?.value, 3),
    },
  ];

  document.getElementById("summaryCards").innerHTML = cardRows
    .map(
      (card) => `
        <article class="summary-card">
          <div class="summary-label">${escapeHtml(card.label)}</div>
          <div class="summary-value">${escapeHtml(card.value)}</div>
          <div class="summary-detail">${escapeHtml(card.detail)}</div>
        </article>
      `
    )
    .join("");
}

function renderComparisonTable(data) {
  const rows = data.comparison_table || [];
  const table = document.getElementById("comparisonTable");
  const header = `
    <thead>
      <tr>
        <th>Metric</th>
        ${SYSTEMS.map((system) => `<th>${SYSTEM_LABELS[system]}</th>`).join("")}
      </tr>
    </thead>
  `;
  const body = rows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.metric)}</td>
          ${SYSTEMS.map((system) => `<td>${formatValue(row[system], row.places ?? 2)}</td>`).join("")}
        </tr>
      `
    )
    .join("");
  table.innerHTML = `${header}<tbody>${body}</tbody>`;
}

function systemValues(data, getter) {
  return SYSTEMS.map((system) => getter(data.system_metrics?.[system] || {}));
}

function createBarChart(canvasId, labels, values, title, yTitle) {
  new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: title,
          data: values,
          backgroundColor: SYSTEMS.map((system) => SYSTEM_COLORS[system]),
          borderColor: SYSTEMS.map((system) => SYSTEM_COLORS[system]),
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: true },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#344054" },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: yTitle },
          grid: { color: "#e8edf5" },
          ticks: { color: "#344054" },
        },
      },
    },
  });
}

function createPieChart(canvasId, noteId, labels, values, title) {
  const total = values.reduce((sum, value) => sum + (Number(value) || 0), 0);
  const note = document.getElementById(noteId);
  if (note && total === 0) {
    note.textContent = "All values are 0 in the current export.";
  }

  new Chart(document.getElementById(canvasId), {
    type: "pie",
    data: {
      labels,
      datasets: [
        {
          label: title,
          data: values,
          backgroundColor: SYSTEMS.map((system) => SYSTEM_COLORS[system]),
          borderColor: "#ffffff",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 14 },
        },
      },
    },
  });
}

function renderCharts(data) {
  const labels = SYSTEMS.map((system) => SYSTEM_LABELS[system]);
  createBarChart(
    "accuracyChart",
    labels,
    systemValues(data, (metrics) => metrics.average_accuracy),
    "Average accuracy",
    "Average accuracy"
  );
  createBarChart(
    "latencyChart",
    labels,
    systemValues(data, (metrics) => metrics.average_latency),
    "Average latency",
    "Seconds"
  );
  createBarChart(
    "contaminationChart",
    labels,
    systemValues(data, (metrics) => metrics.contamination_count),
    "Contamination count",
    "Count"
  );
  createPieChart(
    "hallucinationChart",
    "hallucinationNote",
    labels,
    systemValues(data, (metrics) => metrics.hallucination_count),
    "Hallucination count"
  );
  createPieChart(
    "leakageChart",
    "leakageNote",
    labels,
    systemValues(data, (metrics) => metrics.leakage_count),
    "Leakage count"
  );
  createBarChart(
    "robustnessChart",
    labels,
    systemValues(data, (metrics) => metrics.special_metrics?.cross_domain_robustness),
    "Cross-domain robustness",
    "Average score"
  );
}

function renderSystemCell(systemResult) {
  const metrics = systemResult?.metrics || {};
  const summary = [
    `Acc ${formatValue(metrics.accuracy, 2)}`,
    `Lat ${formatValue(metrics.latency, 2, " s")}`,
  ].join(" | ");

  const metricRows = [
    ["Hallucination", metrics.hallucination, 0],
    ["Leakage", metrics.leakage, 0],
    ["Contamination", metrics.contamination, 0],
    ["False rejection", metrics.false_rejection, 0],
    ["Memory recall", metrics.memory_recall, 3],
    ["Knowledge growth", metrics.knowledge_growth, 3],
    ["Cross-domain robustness", metrics.cross_domain_robustness, 3],
    ["Intent accuracy", metrics.intent_classification_accuracy, 3],
    ["Domain resolution", metrics.domain_resolution_accuracy, 3],
  ]
    .map(
      ([label, value, places]) =>
        `<span><strong>${escapeHtml(label)}:</strong> ${formatValue(value, places)}</span>`
    )
    .join("");

  return `
    <details>
      <summary>${escapeHtml(summary)}</summary>
      <p class="response-text">${escapeHtml(systemResult?.response || "N/A")}</p>
      <div class="mini-metrics">${metricRows}</div>
    </details>
  `;
}

function renderQuestionTable(data) {
  const questions = data.questions || [];
  const table = document.getElementById("questionTable");
  const header = `
    <thead>
      <tr>
        <th>Question</th>
        <th>Type</th>
        ${SYSTEMS.map((system) => `<th>${SYSTEM_LABELS[system]}</th>`).join("")}
      </tr>
    </thead>
  `;
  const body = questions
    .map(
      (question) => `
        <tr>
          <td>${escapeHtml(question.question)}</td>
          <td>${escapeHtml(question.question_type_label || metricLabel(question.question_type || "N/A"))}</td>
          ${SYSTEMS.map((system) => `<td>${renderSystemCell(question.systems?.[system])}</td>`).join("")}
        </tr>
      `
    )
    .join("");
  table.innerHTML = `${header}<tbody>${body}</tbody>`;
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const data = await loadSummary();
    renderHeader(data);
    renderSummaryCards(data);
    renderComparisonTable(data);
    renderCharts(data);
    renderQuestionTable(data);
  } catch (error) {
    showError(error.message);
  }
});
