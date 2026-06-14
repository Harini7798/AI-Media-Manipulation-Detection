// ─── State ──────────────────────────────────────────────────────────────────

const STORAGE_KEY = "dfHistory";
const SETTINGS_KEY = "dfSettings";

let selectedFile = null;
let history = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
let settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{"theme":"light","animations":true,"compact":false}');

// Chart.js instances (so we can destroy/replace them)
let probChart = null;
let confChart = null;
let histChart = null;
let timelineChart = null;

// ─── Element refs ───────────────────────────────────────────────────────────

const els = {
  // Top bar
  statusText: document.getElementById("statusText"),
  // Dashboard upload
  fileInput: document.getElementById("fileInput"),
  predictBtn: document.getElementById("predictBtn"),
  previewImg: document.getElementById("previewImg"),
  previewVideo: document.getElementById("previewVideo"),
  previewPlaceholder: document.getElementById("previewPlaceholder"),
  timelineChartCard: document.getElementById("timelineChartCard"),
  uploadHint: document.getElementById("uploadHint"),
  resultStatus: document.getElementById("resultStatus"),
  pristineScore: document.getElementById("pristineScore"),
  deepfakeScore: document.getElementById("deepfakeScore"),
  metaLine: document.getElementById("metaLine"),
  uploadPanel: document.querySelector(".uploadPanel"),
  // Sidebar history
  historyList: document.getElementById("historyList"),
  historyEmpty: document.querySelector(".historyEmpty"),
  clearHistory: document.getElementById("clearHistory"),
  readyPct: document.getElementById("readyPct"),
  progressFill: document.getElementById("progressFill"),
  // Nav
  navItems: document.querySelectorAll(".navItem"),
  // Pages
  pages: document.querySelectorAll(".page"),
  // Explain
  explainPanel: document.getElementById("explainPanel"),
  explainSub: document.getElementById("explainSub"),
  // History page
  historyGrid: document.getElementById("historyGrid"),
  historySearchInput: document.getElementById("historySearchInput"),
  historyClearAll: document.getElementById("historyClearAll"),
  statTotal: document.getElementById("statTotal"),
  statPristine: document.getElementById("statPristine"),
  statDeepfake: document.getElementById("statDeepfake"),
  statAvgDeepfake: document.getElementById("statAvgDeepfake"),
  // Settings page
  themePicker: document.getElementById("themePicker"),
  animToggle: document.getElementById("animToggle"),
  compactToggle: document.getElementById("compactToggle"),
  modelStatusDesc: document.getElementById("modelStatusDesc"),
  modelPathDesc: document.getElementById("modelPathDesc"),
  refreshStatusBtn: document.getElementById("refreshStatusBtn"),
  storedCount: document.getElementById("storedCount"),
  clearAllDataBtn: document.getElementById("clearAllDataBtn"),
  exportBtn: document.getElementById("exportBtn"),
};

// ─── Storage ────────────────────────────────────────────────────────────────

function saveHistory() {
  // Cap at 50 entries to keep localStorage manageable
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(0, 50)));
}

function saveSettings() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

// ─── Theme ──────────────────────────────────────────────────────────────────

function applySettings() {
  document.documentElement.setAttribute("data-theme", settings.theme);
  document.documentElement.setAttribute("data-noanim", settings.animations ? "0" : "1");
  document.documentElement.setAttribute("data-compact", settings.compact ? "1" : "0");

  // Sync UI controls
  if (els.animToggle) els.animToggle.checked = settings.animations;
  if (els.compactToggle) els.compactToggle.checked = settings.compact;
  document.querySelectorAll(".themeOption").forEach(function (b) {
    b.classList.toggle("active", b.dataset.theme === settings.theme);
  });
}

// ─── Page routing ───────────────────────────────────────────────────────────

function showPage(pageName) {
  els.pages.forEach(function (p) {
    p.style.display = p.dataset.page === pageName ? "" : "none";
  });
  els.navItems.forEach(function (n) {
    n.classList.toggle("active", n.dataset.page === pageName);
  });
  if (pageName === "history") renderHistoryPage();
  if (pageName === "settings") refreshSettingsPage();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

els.navItems.forEach(function (item) {
  item.addEventListener("click", function (e) {
    e.preventDefault();
    var page = item.dataset.page || "dashboard";
    showPage(page);
    if (item.id === "navUpload") {
      setTimeout(function () { els.fileInput.click(); }, 200);
    }
  });
});

// ─── File type helpers ──────────────────────────────────────────────────────

const VIDEO_EXTS = [".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"];

function isVideoFile(file) {
  if (!file) return false;
  if (file.type && file.type.startsWith("video/")) return true;
  var name = (file.name || "").toLowerCase();
  return VIDEO_EXTS.some(function (e) { return name.endsWith(e); });
}

// ─── Image / video compression for storage ─────────────────────────────────

function compressImage(file, maxDim = 96, quality = 0.55) {
  return new Promise(function (resolve) {
    var img = new Image();
    var url = URL.createObjectURL(file);
    img.onload = function () {
      var scale = Math.min(1, maxDim / Math.max(img.width, img.height));
      var w = Math.round(img.width * scale);
      var h = Math.round(img.height * scale);
      var canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      var ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      var dataUrl = canvas.toDataURL("image/jpeg", quality);
      URL.revokeObjectURL(url);
      resolve(dataUrl);
    };
    img.onerror = function () {
      URL.revokeObjectURL(url);
      resolve(null);
    };
    img.src = url;
  });
}

function captureVideoThumb(file, maxDim = 120, quality = 0.55) {
  // Grab a frame from the middle of the video
  return new Promise(function (resolve) {
    var video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    var url = URL.createObjectURL(file);
    video.src = url;

    var captured = false;
    function done(result) {
      if (captured) return;
      captured = true;
      URL.revokeObjectURL(url);
      resolve(result);
    }

    video.addEventListener("loadedmetadata", function () {
      // Seek to the middle (or 0.5s if very short)
      try {
        video.currentTime = Math.min(video.duration / 2, video.duration - 0.05);
      } catch (e) {
        done(null);
      }
    });

    video.addEventListener("seeked", function () {
      try {
        var w = video.videoWidth;
        var h = video.videoHeight;
        var scale = Math.min(1, maxDim / Math.max(w, h));
        var canvas = document.createElement("canvas");
        canvas.width = Math.round(w * scale);
        canvas.height = Math.round(h * scale);
        var ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        done(canvas.toDataURL("image/jpeg", quality));
      } catch (e) {
        done(null);
      }
    });

    video.addEventListener("error", function () { done(null); });
    setTimeout(function () { done(null); }, 5000); // safety timeout
  });
}

// ─── Sidebar history (compact) ──────────────────────────────────────────────

function renderSidebarHistory() {
  if (!history.length) {
    els.historyEmpty.style.display = "block";
    els.historyList.querySelectorAll(".historyItem").forEach(function (n) { n.remove(); });
    return;
  }
  els.historyEmpty.style.display = "none";
  els.historyList.querySelectorAll(".historyItem").forEach(function (n) { n.remove(); });

  history.slice(0, 8).forEach(function (item) {
    var div = document.createElement("div");
    div.className = "historyItem";

    var top = document.createElement("div");
    top.className = "historyItemTop";

    var labelEl = document.createElement("div");
    labelEl.className = "historyLabel";
    labelEl.textContent = item.label;

    var pillEl = document.createElement("div");
    pillEl.className = "pill";
    pillEl.textContent = item.date;

    top.appendChild(labelEl);
    top.appendChild(pillEl);

    var meta = document.createElement("div");
    meta.className = "historyMeta";
    meta.textContent =
      "Pristine: " + (item.pristine_prob * 100).toFixed(2) +
      "% | Deepfake: " + (item.deepfake_prob * 100).toFixed(2) + "%";

    div.appendChild(top);
    div.appendChild(meta);
    els.historyList.appendChild(div);
  });
}

// ─── History page ───────────────────────────────────────────────────────────

function renderHistoryPage(filter) {
  filter = (filter || "").toLowerCase().trim();
  els.historyGrid.innerHTML = "";

  // Stats
  els.statTotal.textContent = history.length;
  var pCount = history.filter(function (h) { return h.label === "Pristine"; }).length;
  var dCount = history.length - pCount;
  els.statPristine.textContent = pCount;
  els.statDeepfake.textContent = dCount;
  var avg = history.length
    ? history.reduce(function (s, h) { return s + h.deepfake_prob; }, 0) / history.length
    : 0;
  els.statAvgDeepfake.textContent = (avg * 100).toFixed(1) + "%";

  var filtered = history.filter(function (h) {
    return !filter || (h.filename && h.filename.toLowerCase().indexOf(filter) !== -1);
  });

  if (!filtered.length) {
    var empty = document.createElement("div");
    empty.className = "emptyState";
    empty.textContent = filter
      ? "No matches for \"" + filter + "\"."
      : "No images analyzed yet. Go to Dashboard to upload one.";
    els.historyGrid.appendChild(empty);
    return;
  }

  filtered.forEach(function (item) {
    var card = document.createElement("div");
    card.className = "historyCard";

    if (item.thumb) {
      var img = document.createElement("img");
      img.className = "historyCardImg";
      img.src = item.thumb;
      img.alt = item.filename || "thumbnail";
      card.appendChild(img);
    } else {
      var ph = document.createElement("div");
      ph.className = "historyCardImg";
      card.appendChild(ph);
    }

    var label = document.createElement("div");
    label.className = "historyCardLabel " + (item.label === "Pristine" ? "pristine" : "deepfake");
    label.textContent = item.label;
    if (item.is_video) {
      var vbadge = document.createElement("span");
      vbadge.className = "videoBadge";
      vbadge.textContent = "VIDEO";
      label.appendChild(vbadge);
    }
    card.appendChild(label);

    var score = document.createElement("div");
    score.className = "historyCardScore";
    var scoreText = "Deepfake score: " + (item.deepfake_prob * 100).toFixed(2) + "%";
    if (item.is_video && item.frame_count) {
      scoreText += "  ·  " + item.frame_count + " frames";
    }
    score.textContent = scoreText;
    card.appendChild(score);

    var bar = document.createElement("div");
    bar.className = "historyCardBar";
    var fill = document.createElement("div");
    fill.className = "historyCardBarFill";
    fill.style.width = (item.deepfake_prob * 100) + "%";
    bar.appendChild(fill);
    card.appendChild(bar);

    var name = document.createElement("div");
    name.className = "historyCardName";
    name.textContent = (item.filename || "unnamed") + " · " + item.date;
    card.appendChild(name);

    els.historyGrid.appendChild(card);
  });
}

if (els.historySearchInput) {
  els.historySearchInput.addEventListener("input", function () {
    renderHistoryPage(els.historySearchInput.value);
  });
}

if (els.historyClearAll) {
  els.historyClearAll.addEventListener("click", function () {
    if (confirm("Clear all prediction history?")) {
      history = [];
      saveHistory();
      renderHistoryPage();
      renderSidebarHistory();
    }
  });
}

// ─── Settings page ──────────────────────────────────────────────────────────

function refreshSettingsPage() {
  // Stored data size
  var raw = localStorage.getItem(STORAGE_KEY) || "";
  var kb = (raw.length / 1024).toFixed(1);
  els.storedCount.textContent = history.length + " entries · " + kb + " KB";
  loadStatus();
}

document.querySelectorAll(".themeOption").forEach(function (btn) {
  btn.addEventListener("click", function () {
    settings.theme = btn.dataset.theme;
    applySettings();
    saveSettings();
  });
});

if (els.animToggle) {
  els.animToggle.addEventListener("change", function () {
    settings.animations = els.animToggle.checked;
    applySettings();
    saveSettings();
  });
}

if (els.compactToggle) {
  els.compactToggle.addEventListener("change", function () {
    settings.compact = els.compactToggle.checked;
    applySettings();
    saveSettings();
  });
}

if (els.refreshStatusBtn) {
  els.refreshStatusBtn.addEventListener("click", loadStatus);
}

if (els.clearAllDataBtn) {
  els.clearAllDataBtn.addEventListener("click", function () {
    if (confirm("Delete all stored predictions and reset settings?")) {
      history = [];
      saveHistory();
      renderHistoryPage();
      renderSidebarHistory();
      refreshSettingsPage();
    }
  });
}

if (els.exportBtn) {
  els.exportBtn.addEventListener("click", function () {
    var blob = new Blob([JSON.stringify(history, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "deepfake-history-" + Date.now() + ".json";
    a.click();
    URL.revokeObjectURL(url);
  });
}

// ─── Charts ─────────────────────────────────────────────────────────────────

function getThemeColors() {
  var styles = getComputedStyle(document.documentElement);
  return {
    accent: styles.getPropertyValue("--accent").trim() || "#ff5a3d",
    text: styles.getPropertyValue("--text").trim() || "#eef2f7",
    muted: styles.getPropertyValue("--muted").trim() || "rgba(255,255,255,0.7)",
    border: styles.getPropertyValue("--border").trim() || "rgba(255,255,255,0.08)",
  };
}

function renderCharts(pristineProb, deepfakeProb) {
  var c = getThemeColors();
  var animDuration = settings.animations ? 700 : 0;

  if (probChart) probChart.destroy();
  if (confChart) confChart.destroy();
  if (histChart) histChart.destroy();

  // ── 1. Probability split (horizontal bar) ──
  var probCtx = document.getElementById("probChart").getContext("2d");
  probChart = new Chart(probCtx, {
    type: "bar",
    data: {
      labels: ["Pristine", "Deepfake"],
      datasets: [{
        data: [pristineProb * 100, deepfakeProb * 100],
        backgroundColor: ["#41d36b", c.accent],
        borderRadius: 8,
        barThickness: 30,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: animDuration },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (ctx) { return ctx.parsed.x.toFixed(2) + "%"; }
          }
        }
      },
      scales: {
        x: {
          min: 0, max: 100,
          ticks: { color: c.muted, callback: function (v) { return v + "%"; } },
          grid: { color: c.border },
        },
        y: {
          ticks: { color: c.text, font: { weight: "bold" } },
          grid: { display: false },
        },
      },
    },
  });

  // ── 2. Decision strength (doughnut/gauge) ──
  // How far the score is from the 50/50 boundary, scaled to 0-100
  var dominantProb = Math.max(pristineProb, deepfakeProb);
  var strength = Math.round((dominantProb - 0.5) * 200); // 0..100
  var strengthColor = pristineProb >= 0.5 ? "#41d36b" : c.accent;
  var confCtx = document.getElementById("confChart").getContext("2d");
  confChart = new Chart(confCtx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [strength, 100 - strength],
        backgroundColor: [strengthColor, c.border],
        borderWidth: 0,
        circumference: 270,
        rotation: -135,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "70%",
      animation: { duration: animDuration },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      },
    },
    plugins: [{
      id: "centerText",
      afterDraw: function (chart) {
        var ctx = chart.ctx;
        var w = chart.width;
        var h = chart.height;
        ctx.save();
        ctx.font = "900 28px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = strengthColor;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(strength + "%", w / 2, h / 2 - 4);
        ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = c.muted;
        ctx.fillText("certainty", w / 2, h / 2 + 18);
        ctx.restore();
      }
    }],
  });

  // ── 3. Distribution vs history (line/bar of past predictions) ──
  var allScores = history.map(function (h) { return h.deepfake_prob * 100; });
  if (allScores.length === 0) {
    allScores = [deepfakeProb * 100];
  }

  // Build a simple histogram in 10% buckets
  var buckets = new Array(10).fill(0);
  allScores.forEach(function (s) {
    var i = Math.min(9, Math.floor(s / 10));
    buckets[i]++;
  });
  var labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"];
  var thisBucket = Math.min(9, Math.floor((deepfakeProb * 100) / 10));
  var bgColors = buckets.map(function (_, i) {
    return i === thisBucket ? c.accent : "rgba(255,255,255,0.18)";
  });
  // For light theme override
  if (settings.theme === "light") {
    bgColors = buckets.map(function (_, i) {
      return i === thisBucket ? c.accent : "rgba(0,0,0,0.18)";
    });
  }

  var histCtx = document.getElementById("histChart").getContext("2d");
  histChart = new Chart(histCtx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Past predictions",
        data: buckets,
        backgroundColor: bgColors,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: animDuration },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              var hl = ctx.dataIndex === thisBucket ? " (this image)" : "";
              return ctx.parsed.y + " prediction" + (ctx.parsed.y === 1 ? "" : "s") + hl;
            }
          }
        },
      },
      scales: {
        x: {
          ticks: { color: c.muted, font: { size: 10 } },
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          ticks: { color: c.muted, stepSize: 1, precision: 0 },
          grid: { color: c.border },
        },
      },
    },
  });

  // Update explanation text
  var verdict = pristineProb >= 0.5 ? "pristine" : "deepfake";
  var ratioMsg = "The model is " + (strength) + "% certain this image is " + verdict + ".";
  els.explainSub.textContent = ratioMsg + " Higher certainty means the feature signature is far from the decision boundary.";
}

function renderTimelineChart(frameScores, frameTimestamps) {
  if (!els.timelineChartCard) return;
  els.timelineChartCard.style.display = "block";

  var c = getThemeColors();
  var animDuration = settings.animations ? 700 : 0;

  // Convert pristine probability to deepfake probability for clarity
  var deepfakeSeries = frameScores.map(function (s) { return +(((1 - s) * 100).toFixed(2)); });
  var labels = frameTimestamps.map(function (t) { return t.toFixed(2) + "s"; });

  if (timelineChart) timelineChart.destroy();
  var ctx = document.getElementById("timelineChart").getContext("2d");
  timelineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Deepfake probability",
          data: deepfakeSeries,
          borderColor: c.accent,
          backgroundColor: "rgba(255, 90, 61, 0.18)",
          borderWidth: 2.5,
          fill: true,
          tension: 0.35,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: c.accent,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: animDuration },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: function (items) { return "Time: " + items[0].label; },
            label: function (ctx) { return "Deepfake: " + ctx.parsed.y.toFixed(2) + "%"; },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: c.muted, font: { size: 10 }, maxRotation: 0, autoSkipPadding: 12 },
          grid: { display: false },
          title: { display: true, text: "Time (seconds)", color: c.muted, font: { size: 11 } },
        },
        y: {
          min: 0,
          max: 100,
          ticks: {
            color: c.muted,
            callback: function (v) { return v + "%"; },
          },
          grid: { color: c.border },
          title: { display: true, text: "Deepfake probability", color: c.muted, font: { size: 11 } },
        },
      },
    },
  });
}

// ─── Status ─────────────────────────────────────────────────────────────────

async function loadStatus() {
  try {
    var res = await fetch("/api/status");
    var data = await res.json();
    var ready = data.model_loaded ? "Ready" : "Not ready";
    els.statusText.textContent = ready;
    var w = data.model_loaded ? 100 : 0;
    if (els.readyPct) els.readyPct.textContent = w + "%";
    if (els.progressFill) els.progressFill.style.width = w + "%";

    if (els.modelStatusDesc) {
      els.modelStatusDesc.textContent = data.model_loaded
        ? "Loaded and ready for predictions."
        : (data.init_error || "Not loaded");
    }
    if (els.modelPathDesc) {
      els.modelPathDesc.textContent = data.model_path || "—";
    }

    if (!data.model_loaded) {
      els.predictBtn.disabled = true;
      els.predictBtn.title = data.init_error || "Model not loaded";
    }
  } catch (e) {
    els.statusText.textContent = "Offline";
    if (els.readyPct) els.readyPct.textContent = "0%";
    if (els.progressFill) els.progressFill.style.width = "0%";
    if (els.modelStatusDesc) els.modelStatusDesc.textContent = "Server unreachable.";
  }
}

// ─── Upload flow ────────────────────────────────────────────────────────────

function setUploadingState() {
  els.predictBtn.disabled = !selectedFile;
  els.uploadHint.textContent = selectedFile ? selectedFile.name : "Waiting for upload";
}

function resetResults() {
  els.resultStatus.textContent = "\u2014";
  els.pristineScore.textContent = "0.00";
  els.deepfakeScore.textContent = "0.00";
  els.metaLine.textContent = "";
  els.explainPanel.style.display = "none";
  if (els.timelineChartCard) els.timelineChartCard.style.display = "none";
}

function showPreview(file) {
  // Reset both preview elements
  els.previewImg.src = "";
  els.previewImg.style.display = "none";
  if (els.previewVideo) {
    els.previewVideo.pause();
    els.previewVideo.removeAttribute("src");
    els.previewVideo.load();
    els.previewVideo.style.display = "none";
  }
  els.previewPlaceholder.style.display = "none";

  if (!file) {
    els.previewPlaceholder.style.display = "block";
    return;
  }

  var url = URL.createObjectURL(file);
  if (isVideoFile(file) && els.previewVideo) {
    els.previewVideo.src = url;
    els.previewVideo.style.display = "block";
  } else {
    els.previewImg.src = url;
    els.previewImg.style.display = "block";
  }
}

els.fileInput.addEventListener("change", function () {
  selectedFile = els.fileInput.files && els.fileInput.files[0] ? els.fileInput.files[0] : null;
  showPreview(selectedFile);
  if (selectedFile) resetResults();
  setUploadingState();
});

// Drag & drop on preview box
var previewBox = document.querySelector(".previewBox");
if (previewBox) {
  previewBox.addEventListener("dragover", function (e) {
    e.preventDefault();
    previewBox.style.borderColor = "var(--accent)";
  });
  previewBox.addEventListener("dragleave", function () {
    previewBox.style.borderColor = "";
  });
  previewBox.addEventListener("drop", function (e) {
    e.preventDefault();
    previewBox.style.borderColor = "";
    var files = e.dataTransfer.files;
    if (files && files.length > 0) {
      var dt = new DataTransfer();
      dt.items.add(files[0]);
      els.fileInput.files = dt.files;
      els.fileInput.dispatchEvent(new Event("change"));
    }
  });
  previewBox.addEventListener("click", function () { els.fileInput.click(); });
  previewBox.style.cursor = "pointer";
}

els.predictBtn.addEventListener("click", async function () {
  if (!selectedFile) return;

  var isVideo = isVideoFile(selectedFile);
  els.predictBtn.disabled = true;
  els.predictBtn.textContent = isVideo ? "Analyzing video..." : "Detecting...";
  els.resultStatus.textContent = "Working...";
  els.metaLine.textContent = isVideo ? "Extracting frames and scoring each one — this can take a few seconds." : "";
  els.explainPanel.style.display = "none";
  if (els.timelineChartCard) els.timelineChartCard.style.display = "none";

  try {
    var form = new FormData();
    form.append("file", selectedFile);

    var res = await fetch("/api/predict", { method: "POST", body: form });
    var data = await res.json();
    if (!data.ok) throw new Error(data.error || "Prediction failed");

    var label = data.label;
    els.resultStatus.textContent = label;
    els.pristineScore.textContent = (data.pristine_prob * 100).toFixed(2) + "%";
    els.deepfakeScore.textContent = (data.deepfake_prob * 100).toFixed(2) + "%";

    if (data.is_video) {
      els.metaLine.textContent =
        "Analyzed " + data.frame_count + " frames over " +
        data.meta.duration_sec + "s @ " + data.meta.fps + " fps. Score is the average across all frames.";
    } else {
      els.metaLine.textContent = "Analysis complete. Input size: " + data.meta.input_size + "px.";
    }

    // Build a thumbnail (image or video frame)
    var thumb;
    if (isVideo) {
      thumb = await captureVideoThumb(selectedFile);
    } else {
      thumb = await compressImage(selectedFile);
    }

    var now = new Date();
    history.unshift({
      label: label,
      pristine_prob: data.pristine_prob,
      deepfake_prob: data.deepfake_prob,
      filename: selectedFile.name,
      is_video: !!data.is_video,
      frame_count: data.frame_count || null,
      duration_sec: (data.meta && data.meta.duration_sec) || null,
      thumb: thumb,
      timestamp: now.getTime(),
      date: now.toLocaleDateString() + " " + now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    });
    saveHistory();
    renderSidebarHistory();

    // Render the explanation charts
    els.explainPanel.style.display = "block";
    setTimeout(function () {
      renderCharts(data.pristine_prob, data.deepfake_prob);
      if (data.is_video && data.frame_scores) {
        renderTimelineChart(data.frame_scores, data.frame_timestamps);
      }
    }, 50);
  } catch (e) {
    els.resultStatus.textContent = "Error";
    els.metaLine.textContent = String(e.message || e);
    els.pristineScore.textContent = "0.00";
    els.deepfakeScore.textContent = "0.00";
  } finally {
    els.predictBtn.textContent = "Detect";
    setUploadingState();
  }
});

els.clearHistory.addEventListener("click", function () {
  history = [];
  saveHistory();
  renderSidebarHistory();
  if (els.historyGrid) renderHistoryPage();
});

// ─── Init ───────────────────────────────────────────────────────────────────

applySettings();
loadStatus();
setUploadingState();
renderSidebarHistory();
showPage("dashboard");
