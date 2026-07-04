const assistantStage = document.getElementById("assistantStage");
const assistantImage = document.getElementById("assistantImage");
const assistantFallback = document.getElementById("assistantFallback");
const assistantBadge = document.getElementById("assistantBadge");
const statusText = document.getElementById("statusText");
const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const systemSelect = document.getElementById("systemSelect");
const voiceSelect = document.getElementById("voiceSelect");
const micButton = document.getElementById("micButton");
const sendButton = document.getElementById("sendButton");
const voiceSupportMessage = document.getElementById("voiceSupportMessage");
const voiceFallbackNote = document.getElementById("voiceFallbackNote");
const uploadButton = document.getElementById("uploadButton");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const autoQuestionSource = document.getElementById("autoQuestionSource");
const autoPdfInput = document.getElementById("autoPdfInput");
const autoPdfSystem = document.getElementById("autoPdfSystem");
const autoPdfCount = document.getElementById("autoPdfCount");
const autoPdfPosition = document.getElementById("autoPdfPosition");
const autoPdfExtractButton = document.getElementById("autoPdfExtractButton");
const loadBenchmarkQuestionsButton = document.getElementById("loadBenchmarkQuestionsButton");
const loadOfficialQuestionsButton = document.getElementById("loadOfficialQuestionsButton");
const autoPdfRunButton = document.getElementById("autoPdfRunButton");
const autoPdfStopButton = document.getElementById("autoPdfStopButton");
const autoPdfClearButton = document.getElementById("autoPdfClearButton");
const autoPdfProgress = document.getElementById("autoPdfProgress");
const autoPdfSummary = document.getElementById("autoPdfSummary");
const autoPdfPreviewList = document.getElementById("autoPdfPreviewList");
const benchmarkCompatibleRunToggle = document.getElementById("benchmarkCompatibleRunToggle");
const keepTemporaryRagToggle = document.getElementById("keepTemporaryRagToggle");
const benchmarkRunWarning = document.getElementById("benchmarkRunWarning");
const researchModeToggle = document.getElementById("researchModeToggle");
const researchModeNote = document.getElementById("researchModeNote");
const dashboardButton = document.getElementById("metricsDashboardBtn");
const clearButton = document.getElementById("clearSessionBtn");
const csvButton = document.getElementById("exportCsvBtn");
const pdfButton = document.getElementById("downloadPdfBtn");
const dashboardModal = document.getElementById("dashboardModal");
const closeDashboardButton = document.getElementById("closeDashboardButton");
const dashboardModeSelect = document.getElementById("dashboardModeSelect");
const dashboardEvaluationBanner = document.getElementById("dashboardEvaluationBanner");
const metricCards = document.getElementById("metricCards");
const noSessionMessage = document.getElementById("noSessionMessage");
const mainComparisonTable = document.getElementById("mainComparisonTable");
const dimensionTable = document.getElementById("dimensionTable");
const specialMetricsTable = document.getElementById("specialMetricsTable");
const categoryAccuracyTable = document.getElementById("categoryAccuracyTable");
const newChatButton = document.getElementById("newChatBtn");
const historyList = document.getElementById("historyList");
const historySearch = document.getElementById("historySearch");
const mobileSidebarButton = document.getElementById("mobileSidebarButton");
const sidebarScrim = document.getElementById("sidebarScrim");
const settingsButton = document.getElementById("settingsBtn");
const researchToolsButton = document.getElementById("researchToolsBtn");
const researchDrawer = document.getElementById("researchDrawer");
const closeResearchToolsButton = document.getElementById("closeResearchToolsButton");
const uploadMenu = document.getElementById("uploadMenu");
const modalForwardButtons = document.querySelectorAll(".modal-forward-button");
const settingsModal = document.getElementById("settingsModal");
const closeSettingsButton = document.getElementById("closeSettingsButton");
const settingsSystemSelect = document.getElementById("settingsSystemSelect");
const settingsVoiceSelect = document.getElementById("settingsVoiceSelect");
const settingsResearchModeToggle = document.getElementById("settingsResearchModeToggle");
const settingsResearchModeNote = document.getElementById("settingsResearchModeNote");
const healthCheckButton = document.getElementById("healthCheckButton");
const healthOutput = document.getElementById("healthOutput");
const mainRagStatus = document.getElementById("mainRagStatus");
const tempRagStatus = document.getElementById("tempRagStatus");
const evaluatorStatus = document.getElementById("evaluatorStatus");

const SYSTEMS = ["A", "B", "C"];
const SYSTEM_LABELS = { A: "System A", B: "System B", C: "System C" };
const SYSTEM_COLORS = {
  A: "rgba(120, 168, 255, 0.78)",
  B: "rgba(242, 181, 107, 0.78)",
  C: "rgba(125, 223, 174, 0.78)",
};
const STANDARD_VOICE_OPTIONS = [
  { id: "american", label: "American English", preferredLangs: ["en-US"] },
  { id: "indian", label: "Indian English", preferredLangs: ["en-IN"] },
  { id: "british", label: "British English", preferredLangs: ["en-GB"] },
];
const DEFAULT_STANDARD_VOICE = "american";
const MAX_CHART_POINTS = 20;

let recognition = null;
let isListening = false;
let currentUtterance = null;
let availableVoices = [];
let charts = {};
let autoPdfQuestions = [];
let autoPdfRunning = false;
let autoPdfStopRequested = false;
let benchmarkStateResetApplied = false;
let dashboardMode = "live";
let historyCounter = 0;
let lastSubmittedQuestion = "";
let autoPdfExtractionInfo = {
  numberedEntriesFound: 0,
  missingNumbers: [],
  duplicateNumbers: [],
  categoryCounts: {},
  extractionMode: null,
  fallbackUsed: false,
  questionSource: "uploaded_pdf",
  sourcePath: null,
  loadedMessage: null,
};
let autoPdfSelectedRange = {
  selected: [],
  startIndex: 0,
  endIndex: 0,
  count: 0,
  total: 0,
};

assistantImage.addEventListener("error", () => {
  assistantImage.style.display = "none";
  assistantFallback.style.display = "grid";
});

function setStatus(text) {
  statusText.textContent = text;
  assistantBadge.textContent = text;
}

function setVoiceSupportMessage(message, warning = false) {
  voiceSupportMessage.textContent = message;
  voiceSupportMessage.classList.toggle("warning", Boolean(message && warning));
}

function openModal(modal) {
  if (!modal) {
    return;
  }
  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
}

function closeModal(modal) {
  if (!modal) {
    return;
  }
  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
}

function researchEvaluationEnabled() {
  return Boolean(researchModeToggle && researchModeToggle.checked);
}

function benchmarkCompatibleRunEnabled() {
  return Boolean(benchmarkCompatibleRunToggle && benchmarkCompatibleRunToggle.checked);
}

function selectedQuestionSource() {
  return autoQuestionSource ? autoQuestionSource.value : "uploaded_pdf";
}

function updateQuestionSourceControls() {
  const source = selectedQuestionSource();
  const uploadedPdf = source === "uploaded_pdf";
  if (autoPdfInput) {
    autoPdfInput.disabled = !uploadedPdf;
  }
  if (autoPdfExtractButton) {
    autoPdfExtractButton.hidden = !uploadedPdf;
  }
  if (loadBenchmarkQuestionsButton) {
    loadBenchmarkQuestionsButton.hidden = source !== "official_benchmark";
  }
  if (loadOfficialQuestionsButton) {
    loadOfficialQuestionsButton.hidden = source !== "official_results";
  }
}

function updateBenchmarkRunWarning() {
  if (benchmarkRunWarning) {
    benchmarkRunWarning.hidden = !benchmarkCompatibleRunEnabled();
  }
}

function updateResearchModeNote() {
  const message = researchEvaluationEnabled()
    ? "Research Mode ON: Benchmark-compatible evaluation enabled"
    : "Research Mode OFF: Demo metrics only";
  const settingsMessage = researchEvaluationEnabled()
    ? "Research Mode ON: Benchmark-compatible evaluation enabled"
    : "Research Mode OFF: Demo metrics only";
  if (researchEvaluationEnabled()) {
    document.body.classList.add("research-mode-on");
  } else {
    document.body.classList.remove("research-mode-on");
  }
  if (researchModeNote) {
    researchModeNote.textContent = message;
  }
  if (settingsResearchModeNote) {
    settingsResearchModeNote.textContent = settingsMessage;
  }
  if (settingsResearchModeToggle) {
    settingsResearchModeToggle.checked = researchEvaluationEnabled();
  }
}

function syncSettingsControlsFromMain() {
  if (settingsSystemSelect) {
    settingsSystemSelect.value = systemSelect.value;
  }
  if (settingsVoiceSelect) {
    settingsVoiceSelect.value = voiceSelect.value;
  }
  if (settingsResearchModeToggle) {
    settingsResearchModeToggle.checked = researchEvaluationEnabled();
  }
  updateResearchModeNote();
}

function openSettingsModal() {
  syncSettingsControlsFromMain();
  openModal(settingsModal);
}

function closeSettingsModal() {
  closeModal(settingsModal);
}

function clearChatWorkspace(message = "Ready", options = {}) {
  stopSpeech();
  if (isListening && recognition) {
    recognition.stop();
  }
  chatLog.innerHTML = `
    <div class="empty-state">
      <div class="empty-orb" aria-hidden="true">AI</div>
      <h2>How can I help?</h2>
      <p>Message Local Multi-Domain Assistant, use voice input, or add temporary file context.</p>
    </div>
  `;
  questionInput.value = "";
  uploadStatus.textContent = "";
  lastSubmittedQuestion = "";
  resetHistoryList();
  resetQuestionInputHeight();
  closeUploadMenu();
  if (!options.keepResearchDrawer) {
    closeResearchDrawer();
  }
  setStatus(message);
}

if (researchModeToggle) {
  researchModeToggle.addEventListener("change", updateResearchModeNote);
  updateResearchModeNote();
}

if (autoQuestionSource) {
  autoQuestionSource.addEventListener("change", () => {
    updateQuestionSourceControls();
    autoPdfProgress.textContent = "";
  });
  updateQuestionSourceControls();
}

if (benchmarkCompatibleRunToggle) {
  benchmarkCompatibleRunToggle.addEventListener("change", () => {
    benchmarkStateResetApplied = false;
    updateBenchmarkRunWarning();
  });
  updateBenchmarkRunWarning();
}

if (systemSelect && settingsSystemSelect) {
  systemSelect.addEventListener("change", () => {
    settingsSystemSelect.value = systemSelect.value;
  });
  settingsSystemSelect.addEventListener("change", () => {
    systemSelect.value = settingsSystemSelect.value;
  });
}

if (voiceSelect && settingsVoiceSelect) {
  voiceSelect.addEventListener("change", () => {
    settingsVoiceSelect.value = voiceSelect.value;
  });
  settingsVoiceSelect.addEventListener("change", () => {
    voiceSelect.value = settingsVoiceSelect.value;
    voiceSelect.dispatchEvent(new Event("change"));
  });
}

if (settingsResearchModeToggle) {
  settingsResearchModeToggle.addEventListener("change", () => {
    researchModeToggle.checked = settingsResearchModeToggle.checked;
    updateResearchModeNote();
  });
}

if (newChatButton) {
  newChatButton.addEventListener("click", () => {
    console.log("New Chat clicked");
    clearChatWorkspace("New chat ready");
    closeMobileSidebar();
  });
}

if (historySearch) {
  historySearch.addEventListener("input", filterHistoryItems);
}

if (mobileSidebarButton) {
  mobileSidebarButton.addEventListener("click", () => {
    document.body.classList.add("sidebar-open");
  });
}

if (sidebarScrim) {
  sidebarScrim.addEventListener("click", closeMobileSidebar);
}

if (settingsButton) {
  settingsButton.addEventListener("click", () => {
    console.log("Settings clicked");
    openSettingsModal();
    closeMobileSidebar();
  });
}

if (researchToolsButton) {
  researchToolsButton.addEventListener("click", () => {
    console.log("Research tools clicked");
    openResearchDrawer();
    closeMobileSidebar();
  });
}

if (closeResearchToolsButton) {
  closeResearchToolsButton.addEventListener("click", closeResearchDrawer);
}

if (closeSettingsButton) {
  closeSettingsButton.addEventListener("click", closeSettingsModal);
}

modalForwardButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.getElementById(button.dataset.forward || "");
    if (target) {
      target.click();
    }
  });
});

function healthLine(label, value) {
  return `${label}: ${displayValue(value)}`;
}

function renderHealthStatus(data) {
  if (mainRagStatus) {
    mainRagStatus.textContent = data.main_rag_available ? "Available" : "Not Available";
  }
  if (tempRagStatus) {
    tempRagStatus.textContent = data.temp_rag_active ? "Active" : "Not Active";
  }
  if (evaluatorStatus) {
    evaluatorStatus.textContent = data.evaluator_bridge_available ? "Available" : "Not Available";
  }
  if (!healthOutput) {
    return;
  }

  const systems = data.systems_available || {};
  healthOutput.textContent = [
    healthLine("System A", systems.A ? "Available" : "Not Available"),
    healthLine("System B", systems.B ? "Available" : "Not Available"),
    healthLine("System C", systems.C ? "Available" : "Not Available"),
    healthLine("System health import probe", data.system_health_probe_imports),
    healthLine("System health note", data.system_health_note || "N/A"),
    healthLine("Main RAG available", data.main_rag_available),
    healthLine("Main RAG source", data.main_rag_source || data.main_rag_path_detected || "N/A"),
    healthLine("Main RAG read only", data.main_rag_read_only),
    healthLine("Main RAG test results", data.main_rag_test_query_result_count),
    healthLine("Main RAG error", data.main_rag_error || "None"),
    healthLine("Temporary RAG active", data.temp_rag_active),
    healthLine("Temp files count", data.temp_files_count),
    healthLine("Evaluator bridge", data.evaluator_bridge_available),
    healthLine("Exact evaluator import", data.evaluator_exact_import_available),
    healthLine("Evaluator bridge method", data.evaluator_bridge_method),
    healthLine("Research eval supported", data.research_evaluation_supported),
    healthLine("Research eval method", data.research_evaluation_method),
    healthLine("Official results available", data.official_results_available),
    healthLine("Official results count", data.official_results_count),
    healthLine("Live session count", data.live_session_count),
    healthLine("Research mode currently enabled", data.research_mode_currently_enabled),
    healthLine("CWD", data.cwd),
    healthLine("Project root", data.project_root),
  ].join("\n");
}

async function runHealthCheck() {
  if (healthOutput) {
    healthOutput.textContent = "Checking system health...";
  }
  const response = await fetch(`/api/health?research_mode_enabled=${researchEvaluationEnabled() ? "true" : "false"}`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Health check failed.");
  }
  renderHealthStatus(data);
  return data;
}

if (healthCheckButton) {
  healthCheckButton.addEventListener("click", async () => {
    try {
      await runHealthCheck();
    } catch (error) {
      if (healthOutput) {
        healthOutput.textContent = `Health check failed: ${error.message}`;
      }
    }
  });
}

document.addEventListener("click", (event) => {
  if (!uploadMenu || !uploadMenu.classList.contains("open")) {
    return;
  }
  if (!event.target.closest(".upload-menu-wrap")) {
    closeUploadMenu();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") {
    return;
  }
  closeUploadMenu();
  closeResearchDrawer();
  closeSettingsModal();
  closeMobileSidebar();
  if (dashboardModal.classList.contains("open")) {
    dashboardModal.classList.remove("open");
    dashboardModal.setAttribute("aria-hidden", "true");
  }
});

function setListening(active) {
  isListening = active;
  assistantStage.classList.toggle("listening", active);
  micButton.classList.toggle("listening", active);
  micButton.textContent = active ? "Stop" : "Mic";
  micButton.dataset.state = active ? "listening" : "ready";
  micButton.setAttribute("aria-label", active ? "Stop voice input" : "Start voice input");
  micButton.title = active ? "Stop voice input" : "Start voice input";
  if (active) {
    setStatus("Listening...");
  } else if (!assistantStage.classList.contains("speaking")) {
    setStatus("Ready");
  }
}

function setSpeaking(active) {
  assistantStage.classList.toggle("speaking", active);
  if (active) {
    assistantStage.classList.remove("listening");
    setStatus("Speaking...");
  } else {
    setStatus(isListening ? "Listening..." : "Ready");
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function displayValue(value, digits = 3) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  const number = Number(value);
  if (Number.isFinite(number)) {
    return Number.isInteger(number) ? String(number) : String(Number(number.toFixed(digits)));
  }
  return String(value);
}

function chartValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function labelFromKey(key) {
  return String(key || "N/A")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatTimestamp(date = new Date()) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function compactText(value, maxLength = 72) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}...` : text;
}

function resetQuestionInputHeight() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 220)}px`;
}

function closeUploadMenu() {
  if (!uploadMenu) {
    return;
  }
  uploadMenu.classList.remove("open");
  uploadButton.setAttribute("aria-expanded", "false");
}

function toggleUploadMenu() {
  if (!uploadMenu) {
    return;
  }
  const willOpen = !uploadMenu.classList.contains("open");
  uploadMenu.classList.toggle("open", willOpen);
  uploadButton.setAttribute("aria-expanded", willOpen ? "true" : "false");
}

function openResearchDrawer() {
  if (!researchDrawer) {
    return;
  }
  researchDrawer.classList.add("open");
  researchDrawer.setAttribute("aria-hidden", "false");
}

function closeResearchDrawer() {
  if (!researchDrawer) {
    return;
  }
  researchDrawer.classList.remove("open");
  researchDrawer.setAttribute("aria-hidden", "true");
}

function closeMobileSidebar() {
  document.body.classList.remove("sidebar-open");
}

function setActiveHistoryItem(item) {
  if (!historyList || !item) {
    return;
  }
  historyList.querySelectorAll(".history-item").forEach((historyItem) => {
    historyItem.classList.toggle("active", historyItem === item);
  });
}

function addHistoryItem(question, targetCard) {
  if (!historyList) {
    return;
  }
  historyCounter += 1;
  const item = document.createElement("button");
  item.className = "history-item active";
  item.type = "button";
  item.dataset.historyId = String(historyCounter);
  item.innerHTML = `
    <span class="history-title">${escapeHtml(compactText(question, 58))}</span>
    <span class="history-meta">${formatTimestamp()}</span>
  `;
  item.addEventListener("click", () => {
    setActiveHistoryItem(item);
    if (targetCard && targetCard.isConnected) {
      targetCard.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    closeMobileSidebar();
  });
  historyList.prepend(item);
  setActiveHistoryItem(item);
}

function resetHistoryList() {
  if (!historyList) {
    return;
  }
  historyCounter = 0;
  historyList.innerHTML = `
    <button class="history-item active" type="button">
      <span class="history-title">Current session</span>
      <span class="history-meta">Ready</span>
    </button>
  `;
}

function filterHistoryItems() {
  if (!historySearch || !historyList) {
    return;
  }
  const query = historySearch.value.trim().toLowerCase();
  historyList.querySelectorAll(".history-item").forEach((item) => {
    const text = item.textContent.toLowerCase();
    item.style.display = !query || text.includes(query) ? "" : "none";
  });
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
}

function tableCells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isTableDivider(line) {
  return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line.trim());
}

function renderMarkdownTable(lines) {
  const headers = tableCells(lines[0]);
  const rows = lines.slice(2).map(tableCells);
  return `
    <table>
      <thead>
        <tr>${headers.map((header) => `<th>${renderInlineMarkdown(header)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderParagraph(lines) {
  if (!lines.length) {
    return "";
  }
  return `<p>${renderInlineMarkdown(lines.join("\n")).replaceAll("\n", "<br>")}</p>`;
}

function renderList(lines, ordered = false) {
  const tag = ordered ? "ol" : "ul";
  const items = lines.map((line) => {
    const item = ordered
      ? line.replace(/^\s*\d+\.\s+/, "")
      : line.replace(/^\s*[-*]\s+/, "");
    return `<li>${renderInlineMarkdown(item)}</li>`;
  });
  return `<${tag}>${items.join("")}</${tag}>`;
}

function renderMarkdown(text) {
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      html.push(renderParagraph(paragraph));
      paragraph = [];
    }
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      continue;
    }

    if (trimmed.startsWith("```")) {
      flushParagraph();
      const codeLines = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      continue;
    }

    if (line.includes("|") && lines[index + 1] && isTableDivider(lines[index + 1])) {
      flushParagraph();
      const tableLines = [line, lines[index + 1]];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        tableLines.push(lines[index]);
        index += 1;
      }
      index -= 1;
      html.push(renderMarkdownTable(tableLines));
      continue;
    }

    if (/^\s*[-*]\s+/.test(line)) {
      flushParagraph();
      const listLines = [line];
      while (lines[index + 1] && /^\s*[-*]\s+/.test(lines[index + 1])) {
        index += 1;
        listLines.push(lines[index]);
      }
      html.push(renderList(listLines));
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      flushParagraph();
      const listLines = [line];
      while (lines[index + 1] && /^\s*\d+\.\s+/.test(lines[index + 1])) {
        index += 1;
        listLines.push(lines[index]);
      }
      html.push(renderList(listLines, true));
      continue;
    }

    paragraph.push(line);
  }

  flushParagraph();
  return html.join("") || `<p>${escapeHtml(text)}</p>`;
}

function removeEmptyState() {
  const empty = chatLog.querySelector(".empty-state");
  if (empty) {
    empty.remove();
  }
}

function scrollChatToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function createQuestionCard(question, label = "") {
  const card = document.createElement("article");
  card.className = "question-card";
  const labelHtml = label ? `<span class="question-label">${escapeHtml(label)}</span>` : "";
  card.innerHTML = `<div class="question-bubble">${labelHtml}<span>${escapeHtml(question)}</span></div><div class="message-avatar user-avatar" aria-hidden="true">You</div>`;
  return card;
}

function addQuestionCard(question, label = "") {
  removeEmptyState();
  const card = createQuestionCard(question, label);
  chatLog.appendChild(card);
  addHistoryItem(question, card);
  scrollChatToBottom();
  return card;
}

function addPendingAutoRunCard(index, total, question) {
  removeEmptyState();
  const card = document.createElement("article");
  card.className = "pending-card auto-run-pending";
  card.setAttribute("aria-live", "polite");
  card.innerHTML = `
    <div class="pending-label">Running question ${index}/${total}...</div>
    <p>${escapeHtml(question)}</p>
  `;
  chatLog.appendChild(card);
  scrollChatToBottom();
  return card;
}

function replacePendingWithQuestion(pendingCard, question, index, total) {
  const questionCard = createQuestionCard(question, `PDF auto question ${index}/${total}`);
  if (pendingCard && pendingCard.isConnected) {
    pendingCard.replaceWith(questionCard);
  } else {
    chatLog.appendChild(questionCard);
  }
  addHistoryItem(question, questionCard);
  scrollChatToBottom();
}

function replacePendingWithError(pendingCard, question, message, index, total) {
  const card = document.createElement("article");
  card.className = "response-card error-card";
  card.innerHTML = `
    <div class="message-avatar assistant-avatar" aria-hidden="true">AI</div>
    <div class="response-content">
      <div class="response-header">
        <div class="response-title">
          <h3>Auto Run Error</h3>
          <span class="system-badge">Error</span>
        </div>
        <div class="response-meta">
          <span class="latency">Question ${index}/${total}</span>
          <span class="timestamp">${formatTimestamp()}</span>
        </div>
      </div>
      <div class="question-fragment">${escapeHtml(question)}</div>
      <div class="answer-text error-text">${renderMarkdown(message)}</div>
    </div>
  `;
  if (pendingCard && pendingCard.isConnected) {
    pendingCard.replaceWith(card);
  } else {
    chatLog.appendChild(card);
  }
  scrollChatToBottom();
}

function ensureStandardVoiceOptions() {
  const currentValue = STANDARD_VOICE_OPTIONS.some((option) => option.id === voiceSelect.value)
    ? voiceSelect.value
    : DEFAULT_STANDARD_VOICE;
  voiceSelect.innerHTML = "";
  STANDARD_VOICE_OPTIONS.forEach((voiceOption) => {
    const option = document.createElement("option");
    option.value = voiceOption.id;
    option.textContent = voiceOption.label;
    voiceSelect.appendChild(option);
  });
  voiceSelect.value = currentValue;
}

function standardVoiceOption(optionId) {
  return STANDARD_VOICE_OPTIONS.find((option) => option.id === optionId) || STANDARD_VOICE_OPTIONS[0];
}

function getVoiceForStandardOption(optionId) {
  const voices = "speechSynthesis" in window ? window.speechSynthesis.getVoices() || [] : [];

  if (optionId === "american") {
    return voices.find((voice) => voice.lang === "en-US" && /Google|Microsoft|Samantha|Zira|Jenny|Aria/i.test(voice.name))
      || voices.find((voice) => voice.lang === "en-US")
      || voices.find((voice) => voice.lang && voice.lang.startsWith("en"))
      || null;
  }

  if (optionId === "indian") {
    return voices.find((voice) => voice.lang === "en-IN")
      || voices.find((voice) => /India|Indian|Heera|Ravi/i.test(voice.name))
      || voices.find((voice) => voice.lang === "en-GB")
      || voices.find((voice) => voice.lang === "en-US")
      || voices.find((voice) => voice.lang && voice.lang.startsWith("en"))
      || null;
  }

  if (optionId === "british") {
    return voices.find((voice) => voice.lang === "en-GB" && /Google|Microsoft|Sonia|Daniel|Serena|British|UK/i.test(voice.name))
      || voices.find((voice) => voice.lang === "en-GB")
      || voices.find((voice) => voice.lang === "en-US")
      || voices.find((voice) => voice.lang && voice.lang.startsWith("en"))
      || null;
  }

  return voices.find((voice) => voice.lang && voice.lang.startsWith("en")) || null;
}

function updateVoiceFallbackMessage() {
  if (!voiceFallbackNote) {
    return;
  }

  const selectedOption = standardVoiceOption(voiceSelect.value || DEFAULT_STANDARD_VOICE);
  const mappedVoice = getVoiceForStandardOption(selectedOption.id);
  const exactVoiceAvailable = Boolean(
    mappedVoice && selectedOption.preferredLangs.some((lang) => mappedVoice.lang === lang),
  );
  voiceFallbackNote.hidden = !mappedVoice || exactVoiceAvailable;
}

function loadVoices() {
  ensureStandardVoiceOptions();
  if (!("speechSynthesis" in window)) {
    voiceSelect.disabled = true;
    return;
  }

  availableVoices = window.speechSynthesis.getVoices() || [];
  voiceSelect.disabled = false;
  updateVoiceFallbackMessage();
}

if ("speechSynthesis" in window) {
  loadVoices();
  window.speechSynthesis.onvoiceschanged = loadVoices;
} else {
  ensureStandardVoiceOptions();
  voiceSelect.disabled = true;
}

voiceSelect.addEventListener("change", updateVoiceFallbackMessage);

function stopSpeech() {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  currentUtterance = null;
  setSpeaking(false);
}

function speakText(text) {
  if (!("speechSynthesis" in window)) {
    setStatus("Text-to-speech is not supported in this browser.");
    return;
  }

  stopSpeech();
  currentUtterance = new SpeechSynthesisUtterance(text);
  currentUtterance.rate = 0.82;
  currentUtterance.pitch = 1.08;
  currentUtterance.volume = 1.0;
  const selectedStandardVoice = voiceSelect.value || DEFAULT_STANDARD_VOICE;
  const selectedOption = standardVoiceOption(selectedStandardVoice);
  const mappedVoice = getVoiceForStandardOption(selectedStandardVoice);
  currentUtterance.lang = (mappedVoice && mappedVoice.lang) || selectedOption.preferredLangs[0] || "en-US";
  if (mappedVoice) {
    currentUtterance.voice = mappedVoice;
  }
  updateVoiceFallbackMessage();
  currentUtterance.onstart = () => setSpeaking(true);
  currentUtterance.onend = () => setSpeaking(false);
  currentUtterance.onerror = () => setSpeaking(false);
  window.speechSynthesis.speak(currentUtterance);
}

async function copyResponseText(text, button) {
  const originalText = button.textContent;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const copyArea = document.createElement("textarea");
      copyArea.value = text;
      copyArea.setAttribute("readonly", "");
      copyArea.style.position = "fixed";
      copyArea.style.opacity = "0";
      document.body.appendChild(copyArea);
      copyArea.select();
      document.execCommand("copy");
      copyArea.remove();
    }
    button.textContent = "Copied";
  } catch (error) {
    button.textContent = "Copy failed";
  } finally {
    window.setTimeout(() => {
      button.textContent = originalText;
    }, 1400);
  }
}

async function regenerateResponse(sourceQuestion, systemKey, button) {
  const question = String(sourceQuestion || "").trim();
  if (!question) {
    setStatus("No question available to regenerate.");
    return;
  }

  const system = SYSTEMS.includes(systemKey) ? systemKey : systemSelect.value;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "Regenerating";
  setStatus("Regenerating...");

  try {
    const data = await sendQuestion(question, system);
    Object.entries(data.answers || {}).forEach(([regeneratedSystemKey, answer]) => {
      addResponseCard(regeneratedSystemKey, answer, question);
    });
    if (dashboardModal.classList.contains("open")) {
      await refreshDashboard();
    }
    setStatus("Ready");
  } catch (error) {
    addResponseCard("Error", {
      response: error.message,
      latency: "N/A",
      metadata: { error: true },
    }, question);
    setStatus("Ready");
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

function addResponseCard(systemKey, answer, sourceQuestion = lastSubmittedQuestion) {
  const card = document.createElement("article");
  card.className = "response-card";
  const normalizedSystem = String(systemKey || "").toUpperCase();
  if (SYSTEMS.includes(normalizedSystem)) {
    card.classList.add(`system-${normalizedSystem.toLowerCase()}`);
  } else {
    card.classList.add("error-card");
  }
  const responseText = answer.response || "";
  const latency = answer.latency ?? "N/A";
  const hasError = answer.metadata && answer.metadata.error;
  const systemLabel = systemKey === "Error" ? "Error" : `System ${escapeHtml(systemKey)}`;

  card.innerHTML = `
    <div class="message-avatar assistant-avatar" aria-hidden="true">AI</div>
    <div class="response-content">
      <div class="response-header">
        <div class="response-title">
          <h3>Assistant</h3>
          <span class="system-badge">${systemLabel}</span>
        </div>
        <div class="response-meta">
          <span class="latency">${escapeHtml(latency)}s</span>
          <span class="timestamp">${formatTimestamp()}</span>
        </div>
      </div>
      <div class="answer-text ${hasError ? "error-text" : ""}">${renderMarkdown(responseText)}</div>
      <div class="card-actions">
        <button class="ghost-button read-button" type="button">Read</button>
        <button class="ghost-button stop-button" type="button">Stop</button>
        <button class="ghost-button copy-button" type="button">Copy</button>
        <button class="ghost-button regenerate-button" type="button">Regenerate</button>
      </div>
    </div>
  `;

  card.querySelector(".read-button").addEventListener("click", () => speakText(responseText));
  card.querySelector(".stop-button").addEventListener("click", stopSpeech);
  card.querySelector(".copy-button").addEventListener("click", (event) => {
    copyResponseText(responseText, event.currentTarget);
  });
  card.querySelector(".regenerate-button").addEventListener("click", (event) => {
    regenerateResponse(sourceQuestion, normalizedSystem, event.currentTarget);
  });
  chatLog.appendChild(card);
  scrollChatToBottom();
}

async function sendQuestion(question, system) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      system,
      research_evaluation_mode: researchEvaluationEnabled(),
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Chat request failed.");
  }
  return data;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    setStatus("Ready");
    questionInput.focus();
    return;
  }

  stopSpeech();
  if (isListening && recognition) {
    recognition.stop();
  }
  lastSubmittedQuestion = question;
  addQuestionCard(question);
  questionInput.value = "";
  resetQuestionInputHeight();
  sendButton.disabled = true;
  setStatus("Thinking...");

  try {
    const data = await sendQuestion(question, systemSelect.value);
    Object.entries(data.answers || {}).forEach(([systemKey, answer]) => {
      addResponseCard(systemKey, answer, question);
    });
    setStatus("Ready");
    if (dashboardModal.classList.contains("open")) {
      await refreshDashboard();
    }
  } catch (error) {
    addResponseCard("Error", {
      response: error.message,
      latency: "N/A",
      metadata: { error: true },
    }, question);
    setStatus("Ready");
  } finally {
    sendButton.disabled = false;
  }
});

questionInput.addEventListener("input", resetQuestionInputHeight);

questionInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) {
    return;
  }
  event.preventDefault();
  if (typeof chatForm.requestSubmit === "function") {
    chatForm.requestSubmit();
  } else {
    sendButton.click();
  }
});

resetQuestionInputHeight();

function speechErrorMessage(errorCode) {
  const messages = {
    "not-allowed": "Microphone permission denied.",
    "no-speech": "No speech detected.",
    "audio-capture": "No microphone found.",
    network: "Speech recognition network error.",
  };
  return messages[errorCode] || `Speech recognition error: ${errorCode || "unknown"}.`;
}

function setupSpeechRecognition() {
  const isFirefox = navigator.userAgent.toLowerCase().includes("firefox");
  if (isFirefox) {
    setVoiceSupportMessage(
      "Voice input works best in Chrome/Edge. Firefox may not support microphone recognition.",
      true,
    );
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micButton.disabled = true;
    setVoiceSupportMessage(
      isFirefox
        ? "Voice input works best in Chrome/Edge. Firefox may not support microphone recognition."
        : "Voice input is not supported in this browser. Use Chrome/Edge or type your question.",
      true,
    );
    return;
  }

  console.log("Speech recognition supported");
  micButton.disabled = false;
  recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.continuous = false;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    console.log("Listening started");
    setVoiceSupportMessage("");
    setListening(true);
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    console.log("Speech result:", transcript);
    questionInput.value = transcript;
    resetQuestionInputHeight();
    questionInput.focus();
  };

  recognition.onerror = (event) => {
    console.log("Speech error:", event.error);
    setVoiceSupportMessage(speechErrorMessage(event.error), true);
    setListening(false);
  };

  recognition.onend = () => {
    setListening(false);
  };

  micButton.addEventListener("click", () => {
    if (!recognition) {
      return;
    }
    if (isListening) {
      recognition.stop();
      setListening(false);
      return;
    }
    try {
      setVoiceSupportMessage("");
      recognition.start();
    } catch (error) {
      console.log("Speech error:", error);
      setVoiceSupportMessage("Could not start voice input. Please try again.", true);
      setListening(false);
    }
  });
}

setupSpeechRecognition();

async function uploadSelectedFile() {
  const file = fileInput.files[0];
  if (!file) {
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  uploadButton.disabled = true;
  uploadStatus.textContent = "Uploading and extracting temporary context...";

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Upload failed.");
    }
    uploadStatus.textContent =
      `${data.filename} uploaded. Stored ${data.temporary_context_characters} temporary context characters.`;
    fileInput.value = "";
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    uploadButton.disabled = false;
  }
}

uploadButton.addEventListener("click", async () => {
  if (fileInput.files[0]) {
    await uploadSelectedFile();
    return;
  }
  toggleUploadMenu();
});

if (uploadMenu) {
  uploadMenu.querySelectorAll("[data-accept]").forEach((button) => {
    button.addEventListener("click", () => {
      fileInput.accept = button.dataset.accept || ".pdf,.txt,.csv,.docx,.json";
      closeUploadMenu();
      fileInput.click();
    });
  });
}

fileInput.addEventListener("change", async () => {
  closeUploadMenu();
  await uploadSelectedFile();
});

function selectQuestionRange(questions, countInput, position) {
  const total = questions.length;
  let count;

  if (!countInput || String(countInput).trim().toLowerCase() === "all") {
    count = total;
  } else {
    count = Number.parseInt(countInput, 10);
    if (!Number.isFinite(count) || count <= 0 || count > total) {
      count = total;
    }
  }

  let startIndex = 0;

  if (count < total) {
    if (position === "middle") {
      startIndex = Math.max(0, Math.floor((total - count) / 2));
    } else if (position === "end") {
      startIndex = Math.max(0, total - count);
    }
  }

  const endIndex = Math.min(total, startIndex + count);
  return {
    selected: questions.slice(startIndex, endIndex),
    startIndex,
    endIndex,
    count,
    total,
  };
}

function autoPdfDisplayText(item) {
  if (item && typeof item === "object") {
    return String(item.raw || item.question || "");
  }
  return String(item || "");
}

function autoPdfCleanQuestion(item) {
  if (item && typeof item === "object") {
    return String(item.question || item.raw || "");
  }
  return String(item || "");
}

function autoPdfQuestionNumber(item) {
  if (item && typeof item === "object" && Number.isInteger(item.number)) {
    return item.number;
  }
  return null;
}

function autoPdfCategory(item) {
  if (item && typeof item === "object" && item.category) {
    return String(item.category);
  }
  return null;
}

function categoryCountsHtml(categoryCounts) {
  const entries = Object.entries(categoryCounts || {});
  if (!entries.length) {
    return "";
  }
  return `
    <span class="category-counts">
      ${entries
        .map(([category, count]) => `<span>${escapeHtml(category)}: ${escapeHtml(count)}</span>`)
        .join("")}
    </span>
  `;
}

function extractionWarningHtml() {
  const missing = autoPdfExtractionInfo.missingNumbers || [];
  if (!missing.length) {
    return "";
  }
  return `
    <span class="pdf-warning">
      Warning: Some numbered questions may not have been extracted.
    </span>
  `;
}

function autoPdfRangeSummary(range) {
  if (!range.total) {
    return "No PDF questions extracted.";
  }
  if (range.count >= range.total) {
    return `Loaded ${range.total} questions. Selected all ${range.total} questions.`;
  }
  return (
    `Loaded ${range.total} questions. ` +
    `Selected questions ${range.startIndex + 1}-${range.endIndex} of ${range.total}.`
  );
}

function refreshAutoPdfSelectionPreview() {
  autoPdfSelectedRange = selectQuestionRange(
    autoPdfQuestions,
    autoPdfCount.value,
    autoPdfPosition.value,
  );
  autoPdfSummary.innerHTML = `
    <span>${escapeHtml(autoPdfRangeSummary(autoPdfSelectedRange))}</span>
    ${autoPdfExtractionInfo.loadedMessage ? `<span>${escapeHtml(autoPdfExtractionInfo.loadedMessage)}</span>` : ""}
    ${autoPdfExtractionInfo.questionSource ? `<span class="source-chip">${escapeHtml(autoPdfExtractionInfo.questionSource)}</span>` : ""}
    ${categoryCountsHtml(autoPdfExtractionInfo.categoryCounts)}
    ${extractionWarningHtml()}
  `;
  autoPdfPreviewList.innerHTML = autoPdfSelectedRange.selected
    .map((question) => `<li>${escapeHtml(autoPdfDisplayText(question))}</li>`)
    .join("");
  return autoPdfSelectedRange;
}

function renderAutoPdfPreview(data) {
  autoPdfQuestions = Array.isArray(data.questions) ? data.questions : [];
  autoPdfExtractionInfo = {
    numberedEntriesFound: Number(data.numbered_entries_found || 0),
    missingNumbers: Array.isArray(data.missing_numbers) ? data.missing_numbers : [],
    duplicateNumbers: Array.isArray(data.duplicate_numbers) ? data.duplicate_numbers : [],
    categoryCounts: data.category_counts && typeof data.category_counts === "object"
      ? data.category_counts
      : {},
    extractionMode: data.extraction_mode || null,
    fallbackUsed: Boolean(data.fallback_used),
    questionSource: data.question_source || "uploaded_pdf",
    sourcePath: data.source_path || null,
    loadedMessage: data.loaded_message || null,
  };
  refreshAutoPdfSelectionPreview();
}

function resetAutoPdfUi(message = "") {
  autoPdfQuestions = [];
  autoPdfSelectedRange = {
    selected: [],
    startIndex: 0,
    endIndex: 0,
    count: 0,
    total: 0,
  };
  autoPdfExtractionInfo = {
    numberedEntriesFound: 0,
    missingNumbers: [],
    duplicateNumbers: [],
    categoryCounts: {},
    extractionMode: null,
    fallbackUsed: false,
    questionSource: "uploaded_pdf",
    sourcePath: null,
    loadedMessage: null,
  };
  autoPdfInput.value = "";
  autoPdfCount.value = "10";
  autoPdfProgress.textContent = message;
  autoPdfSummary.textContent = "No PDF questions extracted.";
  autoPdfPreviewList.innerHTML = "";
}

async function loadResearchQuestionSource(source) {
  const label = source === "official_results" ? "official results.json questions" : "official benchmark questions";
  autoPdfProgress.textContent = `Loading ${label}...`;
  const response = await fetch("/api/load_research_questions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Could not load research questions.");
  }
  renderAutoPdfPreview(data);
  autoPdfProgress.textContent = data.loaded_message || `Loaded ${data.total_questions} questions.`;
  if (source === "official_results" && data.total_questions === 420) {
    autoPdfProgress.textContent = "Loaded 420 official evaluated questions from results.json";
  }
  return data;
}

async function prepareBenchmarkCompatibleRun() {
  if (!benchmarkCompatibleRunEnabled()) {
    benchmarkStateResetApplied = false;
    return null;
  }
  autoPdfProgress.textContent = "Resetting website session for benchmark-compatible live run...";
  const response = await fetch("/api/benchmark_live_reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      keep_temporary_rag: Boolean(keepTemporaryRagToggle && keepTemporaryRagToggle.checked),
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Benchmark-compatible reset failed.");
  }
  benchmarkStateResetApplied = Boolean(data.state_reset_applied);
  clearChatWorkspace("Benchmark-compatible live run ready", { keepResearchDrawer: true });
  if (!(keepTemporaryRagToggle && keepTemporaryRagToggle.checked)) {
    uploadStatus.textContent = "";
  }
  autoPdfProgress.textContent = data.note || "Benchmark-compatible live run reset applied.";
  return data;
}

async function runSingleAutoPdfQuestion(entry, system) {
  const question = autoPdfCleanQuestion(entry);
  const response = await fetch("/api/auto_pdf_run_one", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      system,
      source: "auto_pdf",
      display_question: autoPdfDisplayText(entry),
      pdf_question_number: autoPdfQuestionNumber(entry),
      pdf_category: autoPdfCategory(entry),
      research_evaluation_mode: researchEvaluationEnabled(),
      benchmark_compatible_live_run: benchmarkCompatibleRunEnabled(),
      state_reset_applied: benchmarkStateResetApplied,
      question_source: autoPdfExtractionInfo.questionSource || selectedQuestionSource(),
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Auto PDF question failed.");
  }
  return data;
}

function setAutoPdfRunningState(active) {
  autoPdfRunning = active;
  autoPdfRunButton.disabled = active;
  autoPdfExtractButton.disabled = active;
  autoPdfClearButton.disabled = active;
  if (loadBenchmarkQuestionsButton) {
    loadBenchmarkQuestionsButton.disabled = active;
  }
  if (loadOfficialQuestionsButton) {
    loadOfficialQuestionsButton.disabled = active;
  }
  if (autoQuestionSource) {
    autoQuestionSource.disabled = active;
  }
  if (benchmarkCompatibleRunToggle) {
    benchmarkCompatibleRunToggle.disabled = active;
  }
  if (keepTemporaryRagToggle) {
    keepTemporaryRagToggle.disabled = active;
  }
  autoPdfStopButton.hidden = !active;
  autoPdfStopButton.disabled = !active;
  autoPdfRunButton.textContent = active ? "Running..." : "Auto Run";
  autoPdfStopButton.textContent = "Stop Auto Run";
  autoPdfProgress.classList.toggle("running", active);
}

function waitForPaint() {
  return new Promise((resolve) => {
    window.requestAnimationFrame(() => resolve());
  });
}

async function loadAutoPdfQuestions() {
  try {
    const response = await fetch("/api/auto_pdf_questions");
    const data = await response.json();
    if (response.ok) {
      renderAutoPdfPreview(data);
    }
  } catch (error) {
    console.log("Could not load auto PDF questions:", error);
  }
}

autoPdfCount.addEventListener("input", () => {
  refreshAutoPdfSelectionPreview();
});

autoPdfPosition.addEventListener("change", () => {
  refreshAutoPdfSelectionPreview();
});

autoPdfExtractButton.addEventListener("click", async () => {
  if (selectedQuestionSource() !== "uploaded_pdf") {
    autoPdfProgress.textContent = "Use the source load button for official benchmark questions.";
    return;
  }
  const file = autoPdfInput.files[0];
  if (!file) {
    autoPdfProgress.textContent = "Choose a PDF file first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  autoPdfExtractButton.disabled = true;
  autoPdfProgress.textContent = "Extracting questions from PDF...";

  try {
    const response = await fetch("/api/auto_pdf_extract", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Question extraction failed.");
    }
    renderAutoPdfPreview(data);
    autoPdfInput.value = "";
    autoPdfProgress.textContent = autoPdfExtractionInfo.missingNumbers.length
      ? "Warning: Some numbered questions may not have been extracted."
      : `Extracted ${data.total_questions} questions.`;
  } catch (error) {
    autoPdfProgress.textContent = error.message;
  } finally {
    autoPdfExtractButton.disabled = false;
  }
});

if (loadBenchmarkQuestionsButton) {
  loadBenchmarkQuestionsButton.addEventListener("click", async () => {
    loadBenchmarkQuestionsButton.disabled = true;
    try {
      await loadResearchQuestionSource("official_benchmark");
      autoPdfCount.value = "all";
      refreshAutoPdfSelectionPreview();
    } catch (error) {
      autoPdfProgress.textContent = error.message;
    } finally {
      loadBenchmarkQuestionsButton.disabled = false;
    }
  });
}

if (loadOfficialQuestionsButton) {
  loadOfficialQuestionsButton.addEventListener("click", async () => {
    loadOfficialQuestionsButton.disabled = true;
    try {
      await loadResearchQuestionSource("official_results");
      autoPdfCount.value = "all";
      refreshAutoPdfSelectionPreview();
    } catch (error) {
      autoPdfProgress.textContent = error.message;
    } finally {
      loadOfficialQuestionsButton.disabled = false;
    }
  });
}

autoPdfRunButton.addEventListener("click", async () => {
  if (autoPdfRunning) {
    return;
  }

  let completedCount = 0;
  let processedCount = 0;
  let selected = {
    selected: [],
    startIndex: 0,
    endIndex: 0,
    count: 0,
    total: 0,
  };
  try {
    if (benchmarkCompatibleRunEnabled() && !researchEvaluationEnabled()) {
      researchModeToggle.checked = true;
      updateResearchModeNote();
    }
    if (!autoPdfQuestions.length) {
      if (selectedQuestionSource() === "uploaded_pdf") {
        await loadAutoPdfQuestions();
      } else {
        await loadResearchQuestionSource(selectedQuestionSource());
      }
    }
    if (!autoPdfQuestions.length) {
      throw new Error("Load or extract questions before running.");
    }

    selected = refreshAutoPdfSelectionPreview();
    const selectedQuestions = selected.selected;
    if (!selectedQuestions.length) {
      throw new Error("No questions selected.");
    }
    console.log("PDF Auto Run Selection", {
      total: selected.total,
      count: selected.count,
      position: autoPdfPosition.value,
      startIndex: selected.startIndex,
      endIndex: selected.endIndex,
      selectedQuestions,
    });

    stopSpeech();
    if (isListening && recognition) {
      recognition.stop();
    }

    await prepareBenchmarkCompatibleRun();

    autoPdfStopRequested = false;
    setAutoPdfRunningState(true);
    setStatus("Running PDF questions...");
    await waitForPaint();

    for (const [index, questionEntry] of selectedQuestions.entries()) {
      const visibleIndex = index + 1;
      autoPdfProgress.textContent = researchEvaluationEnabled()
        ? `Running question ${visibleIndex}/${selectedQuestions.length}. Evaluating ${autoPdfSystem.value === "all" ? "System A/B/C" : `System ${autoPdfSystem.value.toUpperCase()}`}...`
        : `Running question ${visibleIndex}/${selectedQuestions.length}...`;
      const displayQuestion = autoPdfDisplayText(questionEntry);
      const pendingCard = addPendingAutoRunCard(visibleIndex, selectedQuestions.length, displayQuestion);
      await waitForPaint();

      try {
        const result = await runSingleAutoPdfQuestion(questionEntry, autoPdfSystem.value);
        replacePendingWithQuestion(
          pendingCard,
          result.display_question || displayQuestion,
          visibleIndex,
          selectedQuestions.length,
        );
        Object.entries(result.answers || {}).forEach(([systemKey, answer]) => {
          addResponseCard(systemKey, answer, autoPdfCleanQuestion(questionEntry));
        });
        completedCount += 1;
        if (dashboardModal.classList.contains("open")) {
          await refreshDashboard();
        }
      } catch (error) {
        replacePendingWithError(
          pendingCard,
          displayQuestion,
          error.message,
          visibleIndex,
          selectedQuestions.length,
        );
      }
      processedCount = visibleIndex;

      await waitForPaint();
      if (autoPdfStopRequested) {
        break;
      }
    }

    const selectedCount = selectedQuestions.length;
    autoPdfProgress.textContent = autoPdfStopRequested
      ? `Stopped after ${processedCount}/${selectedCount} questions (${completedCount} completed).`
      : `Finished ${processedCount}/${selectedCount} questions (${completedCount} completed) ` +
        `(questions ${selected.startIndex + 1}-${selected.endIndex} of ${selected.total}).`;
    setStatus("Ready");
  } catch (error) {
    autoPdfProgress.textContent = error.message;
    addResponseCard("Error", {
      response: error.message,
      latency: "N/A",
      metadata: { error: true },
    }, selected.selected && selected.selected.length ? autoPdfCleanQuestion(selected.selected[0]) : "");
    setStatus("Ready");
  } finally {
    autoPdfStopRequested = false;
    setAutoPdfRunningState(false);
  }
});

autoPdfStopButton.addEventListener("click", () => {
  if (!autoPdfRunning) {
    return;
  }
  autoPdfStopRequested = true;
  autoPdfStopButton.disabled = true;
  autoPdfStopButton.textContent = "Stopping...";
  autoPdfProgress.textContent = "Stopping after the current question finishes...";
});

autoPdfClearButton.addEventListener("click", async () => {
  autoPdfClearButton.disabled = true;
  try {
    await fetch("/api/auto_pdf_clear", { method: "POST" });
    resetAutoPdfUi("Extracted PDF questions cleared.");
  } catch (error) {
    autoPdfProgress.textContent = "Could not clear extracted questions.";
  } finally {
    autoPdfClearButton.disabled = false;
  }
});

loadAutoPdfQuestions();

function metricCard(label, value) {
  return `
    <div class="metric-card">
      <strong>${escapeHtml(displayValue(value))}</strong>
      <span>${escapeHtml(label)}</span>
    </div>
  `;
}

function winnerValue(winner) {
  if (!winner || !winner.label) {
    return "N/A";
  }
  return winner.value === null || winner.value === undefined
    ? winner.label
    : `${winner.label} (${displayValue(winner.value)})`;
}

function renderMetricCards(metrics) {
  const cards = [
    ["Evaluation Mode", metrics.evaluation_mode_label || "N/A"],
    ["Total Questions", metrics.summary_cards.total_questions],
    ["Total Responses", metrics.summary_cards.total_responses],
    ["Main RAG Used", metrics.summary_cards.main_rag_used_count],
    ["Temporary RAG Used", metrics.summary_cards.temporary_rag_used_count],
    ["Best Accuracy", winnerValue(metrics.summary_cards.best_accuracy_system)],
    ["Fastest System", winnerValue(metrics.summary_cards.fastest_system)],
    ["Lowest Contamination", winnerValue(metrics.summary_cards.lowest_contamination_system)],
    ["Best Cross-domain", winnerValue(metrics.summary_cards.best_cross_domain_robustness_system)],
  ];
  metricCards.innerHTML = cards.map(([label, value]) => metricCard(label, value)).join("");
}

function renderTable(container, headers, rows) {
  container.innerHTML = `
    <table>
      <thead>
        <tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>${row.map((cell) => `<td>${escapeHtml(displayValue(cell))}</td>`).join("")}</tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderTables(metrics) {
  renderTable(
    mainComparisonTable,
    ["System", "Count", "Avg Accuracy", "Median Accuracy", "Avg Latency", "Hallucination", "Leakage", "Contamination", "False Rejection"],
    SYSTEMS.map((system) => {
      const item = metrics.systems[system];
      return [
        SYSTEM_LABELS[system],
        item.count,
        item.avg_accuracy,
        item.median_accuracy,
        item.avg_latency,
        item.hallucination_count,
        item.leakage_count,
        item.contamination_count,
        item.false_rejection_count,
      ];
    }),
  );

  renderTable(
    dimensionTable,
    ["System", ...metrics.dimensions.map(labelFromKey)],
    SYSTEMS.map((system) => [
      SYSTEM_LABELS[system],
      ...metrics.dimensions.map((dimension) => metrics.systems[system].dimension_score_averages[dimension]),
    ]),
  );

  renderTable(
    specialMetricsTable,
    [
      "System",
      "Hallucination",
      "Contamination",
      "Leakage",
      "Memory Recall",
      "Knowledge Growth",
      "Cross-domain Robustness",
      "Intent Accuracy",
      "Domain Resolution",
    ],
    SYSTEMS.map((system) => {
      const item = metrics.systems[system];
      const special = metrics.systems[system].special_metrics;
      return [
        SYSTEM_LABELS[system],
        item.hallucination_count,
        item.contamination_count,
        item.leakage_count,
        special.memory_recall,
        special.knowledge_growth,
        special.cross_domain_robustness,
        special.intent_classification_accuracy,
        special.domain_resolution_accuracy,
      ];
    }),
  );

  const categories = metrics.categories.length ? metrics.categories : metrics.all_categories;
  renderTable(
    categoryAccuracyTable,
    ["Category", "System A", "System B", "System C"],
    categories.map((category) => [
      category.label,
      metrics.systems.A.category_accuracy[category.key],
      metrics.systems.B.category_accuracy[category.key],
      metrics.systems.C.category_accuracy[category.key],
    ]),
  );
}

function chartScales() {
  return {
    x: { ticks: { color: "#A9A9B2" }, grid: { color: "rgba(255,255,255,0.06)" } },
    y: { ticks: { color: "#A9A9B2" }, grid: { color: "rgba(255,255,255,0.08)" }, beginAtZero: true },
  };
}

function limitChartData(data) {
  const labels = Array.isArray(data.labels) ? data.labels.slice(-MAX_CHART_POINTS) : [];
  const start = Math.max(0, (data.labels || []).length - labels.length);
  return {
    ...data,
    labels,
    datasets: (data.datasets || []).map((dataset) => ({
      ...dataset,
      data: Array.isArray(dataset.data)
        ? dataset.data.slice(start).slice(-MAX_CHART_POINTS)
        : dataset.data,
    })),
  };
}

function stableChartOptions(options = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    resizeDelay: 150,
    ...options,
    plugins: {
      ...(options.plugins || {}),
      legend: {
        labels: { color: "#ECECF1" },
        ...((options.plugins && options.plugins.legend) || {}),
      },
    },
  };
}

function upsertChart(canvasId, config) {
  if (typeof Chart === "undefined") {
    return;
  }
  const canvas = document.getElementById(canvasId);
  const safeConfig = {
    ...config,
    data: limitChartData(config.data || {}),
    options: stableChartOptions(config.options || {}),
  };
  const existing = charts[canvasId];

  if (existing && existing.config.type === safeConfig.type) {
    existing.data = safeConfig.data;
    existing.options = safeConfig.options;
    existing.update("none");
    return;
  }

  if (existing) {
    existing.destroy();
    delete charts[canvasId];
  }

  const attachedChart = Chart.getChart ? Chart.getChart(canvas) : null;
  if (attachedChart) {
    attachedChart.destroy();
  }

  charts[canvasId] = new Chart(canvas, safeConfig);
}

function barChart(canvasId, label, values, color) {
  upsertChart(canvasId, {
    type: "bar",
    data: {
      labels: SYSTEMS.map((system) => SYSTEM_LABELS[system]),
      datasets: [{ label, data: values.map(chartValue), backgroundColor: color, borderWidth: 1 }],
    },
    options: {
      scales: chartScales(),
    },
  });
}

function renderCharts(metrics) {
  barChart(
    "accuracyChart",
    "Average accuracy",
    SYSTEMS.map((system) => metrics.systems[system].avg_accuracy),
    "rgba(16, 163, 127, 0.82)",
  );
  barChart(
    "latencyChart",
    "Average latency",
    SYSTEMS.map((system) => metrics.systems[system].avg_latency),
    "rgba(120, 168, 255, 0.78)",
  );

  upsertChart("reliabilityChart", {
    type: "bar",
    data: {
      labels: SYSTEMS.map((system) => SYSTEM_LABELS[system]),
      datasets: [
        {
          label: "Hallucination",
          data: SYSTEMS.map((system) => chartValue(metrics.systems[system].hallucination_count)),
          backgroundColor: "rgba(242, 109, 120, 0.78)",
        },
        {
          label: "Leakage",
          data: SYSTEMS.map((system) => chartValue(metrics.systems[system].leakage_count)),
          backgroundColor: "rgba(120, 168, 255, 0.78)",
        },
        {
          label: "Contamination",
          data: SYSTEMS.map((system) => chartValue(metrics.systems[system].contamination_count)),
          backgroundColor: "rgba(242, 181, 107, 0.78)",
        },
      ],
    },
    options: {
      scales: chartScales(),
    },
  });

  upsertChart("dimensionChart", {
    type: "radar",
    data: {
      labels: metrics.dimensions.map(labelFromKey),
      datasets: SYSTEMS.map((system) => ({
        label: SYSTEM_LABELS[system],
        data: metrics.dimensions.map((dimension) => chartValue(metrics.systems[system].dimension_score_averages[dimension])),
        borderColor: SYSTEM_COLORS[system],
        backgroundColor: SYSTEM_COLORS[system].replace(/0\.78\)$/, "0.18)"),
      })),
    },
    options: {
      scales: {
        r: {
          beginAtZero: true,
          ticks: { color: "#A9A9B2", backdropColor: "transparent" },
          grid: { color: "rgba(255,255,255,0.12)" },
          pointLabels: { color: "#ECECF1" },
        },
      },
    },
  });

  barChart(
    "crossDomainChart",
    "Cross-domain robustness",
    SYSTEMS.map((system) => metrics.systems[system].special_metrics.cross_domain_robustness),
    "rgba(16, 163, 127, 0.78)",
  );

  const categories = metrics.categories.length ? metrics.categories : metrics.all_categories;
  upsertChart("categoryChart", {
    type: "bar",
    data: {
      labels: categories.map((category) => category.label),
      datasets: SYSTEMS.map((system) => ({
        label: SYSTEM_LABELS[system],
        data: categories.map((category) => chartValue(metrics.systems[system].category_accuracy[category.key])),
        backgroundColor: SYSTEM_COLORS[system],
      })),
    },
    options: {
      scales: chartScales(),
    },
  });
}

async function refreshDashboard() {
  dashboardMode = dashboardModeSelect ? dashboardModeSelect.value : dashboardMode;
  const response = await fetch(`/api/session_metrics?mode=${encodeURIComponent(dashboardMode)}`);
  const metrics = await response.json();
  if (dashboardEvaluationBanner) {
    dashboardEvaluationBanner.textContent = metrics.evaluation_message
      || `Evaluation mode: ${metrics.evaluation_mode_label || "N/A"}`;
    dashboardEvaluationBanner.classList.toggle(
      "research",
      ["benchmark_live", "official"].includes(String(metrics.evaluation_mode || "")),
    );
  }
  noSessionMessage.style.display = metrics.total_questions || metrics.evaluation_warning ? "block" : "none";
  noSessionMessage.textContent = metrics.total_questions
    ? (metrics.evaluation_warning || metrics.evaluation_message || "")
    : "No session results yet. Ask a question first.";
  renderMetricCards(metrics);
  renderTables(metrics);
  renderCharts(metrics);
}

dashboardButton.addEventListener("click", async () => {
  console.log("Metrics clicked");
  openModal(dashboardModal);
  await refreshDashboard();
});

if (dashboardModeSelect) {
  dashboardModeSelect.addEventListener("change", async () => {
    dashboardMode = dashboardModeSelect.value;
    if (dashboardModal.classList.contains("open")) {
      await refreshDashboard();
    }
  });
}

closeDashboardButton.addEventListener("click", () => {
  closeModal(dashboardModal);
});

dashboardModal.addEventListener("click", (event) => {
  if (event.target === dashboardModal) {
    closeModal(dashboardModal);
  }
});

if (settingsModal) {
  settingsModal.addEventListener("click", (event) => {
    if (event.target === settingsModal) {
      closeSettingsModal();
    }
  });
}

async function clearSession() {
  console.log("Clear session clicked");
  stopSpeech();
  if (isListening && recognition) {
    recognition.stop();
  }
  await fetch("/api/clear_session", { method: "POST" });
  benchmarkStateResetApplied = false;
  clearChatWorkspace("Ready");
  fileInput.value = "";
  fileInput.accept = ".pdf,.txt,.csv,.docx,.json";
  resetAutoPdfUi("");
  if (dashboardModal.classList.contains("open")) {
    await refreshDashboard();
  }
  if (settingsModal && settingsModal.classList.contains("open")) {
    try {
      await runHealthCheck();
    } catch (error) {
      if (healthOutput) {
        healthOutput.textContent = `Health check failed: ${error.message}`;
      }
    }
  }
}

clearButton.addEventListener("click", clearSession);

csvButton.addEventListener("click", () => {
  console.log("CSV export clicked");
  const mode = dashboardModeSelect ? dashboardModeSelect.value : dashboardMode;
  window.location.href = `/api/export_csv?mode=${encodeURIComponent(mode)}`;
});

pdfButton.addEventListener("click", () => {
  console.log("PDF download clicked");
  const mode = dashboardModeSelect ? dashboardModeSelect.value : dashboardMode;
  window.location.href = `/api/download_pdf?mode=${encodeURIComponent(mode)}`;
});

window.addEventListener("beforeunload", stopSpeech);
