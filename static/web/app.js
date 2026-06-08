const state = {
  stream: null,
  running: false,
  pending: false,
  timer: null,
  quality: "fast",
  modelsReady: false,
  latestResult: null,
  history: [],
};

const els = {
  modelStatus: document.getElementById("modelStatus"),
  stage: document.getElementById("stage"),
  video: document.getElementById("cameraVideo"),
  resultImage: document.getElementById("resultImage"),
  stageEmpty: document.getElementById("stageEmpty"),
  faceCount: document.getElementById("faceCount"),
  latency: document.getElementById("latency"),
  activeDetector: document.getElementById("activeDetector"),
  detectorSelect: document.getElementById("detectorSelect"),
  intervalRange: document.getElementById("intervalRange"),
  intervalValue: document.getElementById("intervalValue"),
  startCamera: document.getElementById("startCamera"),
  snapshotCamera: document.getElementById("snapshotCamera"),
  stopCamera: document.getElementById("stopCamera"),
  imageInput: document.getElementById("imageInput"),
  analyzeImage: document.getElementById("analyzeImage"),
  downloadResult: document.getElementById("downloadResult"),
  predictionList: document.getElementById("predictionList"),
  resultState: document.getElementById("resultState"),
  clearHistory: document.getElementById("clearHistory"),
  historyList: document.getElementById("historyList"),
  canvas: document.getElementById("captureCanvas"),
};

function setStatus(stateName, message) {
  els.modelStatus.textContent = message;
  els.modelStatus.classList.remove("ready", "error");
  if (stateName === "ready") {
    els.modelStatus.classList.add("ready");
  }
  if (stateName === "error") {
    els.modelStatus.classList.add("error");
  }
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    const modelState = data.models.state;
    state.modelsReady = modelState === "ready";
    setStatus(modelState, data.models.message);
  } catch (error) {
    state.modelsReady = false;
    setStatus("error", "Server offline");
  }
}

function setResultState(message) {
  els.resultState.textContent = message;
}

function updateMetrics(data) {
  els.faceCount.textContent = String(data.face_count);
  els.latency.textContent = `${data.elapsed_ms} ms`;
  els.activeDetector.textContent = data.detector || els.detectorSelect.value.toUpperCase();
}

function confidenceRow(label, value, text) {
  const percent = Math.max(0, Math.min(100, Math.round(Number(value || 0) * 100)));
  return `
    <div class="confidence-row">
      <strong>${label}</strong>
      <div class="bar-track" aria-hidden="true">
        <div class="bar-fill" style="width: ${percent}%"></div>
      </div>
      <span>${percent}%</span>
    </div>
    <div class="box-line">${text}</div>
  `;
}

function renderPredictions(predictions) {
  els.predictionList.innerHTML = "";
  if (!predictions.length) {
    setResultState("No faces");
    return;
  }

  setResultState("Ready");
  for (const [index, item] of predictions.entries()) {
    const root = document.createElement("article");
    root.className = "prediction-item";
    const crop = item.crop_image
      ? `<img class="face-crop" src="${item.crop_image}" alt="Face ${index + 1} crop">`
      : `<div class="face-crop empty">No crop</div>`;
    root.innerHTML = `
      ${crop}
      <div class="prediction-body">
        <div class="prediction-title">
          <span>Face ${index + 1}</span>
          <span>${item.emotion}</span>
        </div>
        <div class="confidence-list">
          ${confidenceRow("Emotion", item.emotion_confidence, item.emotion)}
          ${confidenceRow("Gender", item.gender_confidence, item.gender)}
          ${confidenceRow("Age", item.age_confidence, item.age)}
        </div>
        <div class="box-line">Box ${item.box.width}x${item.box.height} at ${item.box.x}, ${item.box.y}</div>
      </div>
    `;
    els.predictionList.appendChild(root);
  }
}

function showAnnotatedImage(data, options = {}) {
  const resultData = {
    ...data,
    detector: options.detector || els.detectorSelect.value.toUpperCase(),
    quality: options.quality || state.quality,
    source: options.source || "Live",
  };
  state.latestResult = resultData;
  els.resultImage.src = data.image;
  els.stage.classList.add("has-result");
  els.downloadResult.disabled = false;
  updateMetrics(resultData);
  renderPredictions(resultData.predictions);

  if (options.saveHistory) {
    addHistory(resultData);
  }
}

function scheduleNextFrame() {
  if (!state.running) {
    return;
  }
  const delay = Number(els.intervalRange.value);
  state.timer = window.setTimeout(analyzeCameraFrame, delay);
}

function captureVideoBlob() {
  const videoWidth = els.video.videoWidth || 640;
  const videoHeight = els.video.videoHeight || 480;
  const targetWidth = Math.min(720, videoWidth);
  const targetHeight = Math.max(1, Math.round(videoHeight * (targetWidth / videoWidth)));

  els.canvas.width = targetWidth;
  els.canvas.height = targetHeight;
  const ctx = els.canvas.getContext("2d");
  ctx.drawImage(els.video, 0, 0, targetWidth, targetHeight);

  return new Promise((resolve) => {
    els.canvas.toBlob(resolve, "image/jpeg", 0.76);
  });
}

async function sendImage(blob, source) {
  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  form.append("source", source);
  form.append("detector", els.detectorSelect.value);
  form.append("quality", state.quality);

  const response = await fetch("/api/analyze", {
    method: "POST",
    body: form,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Analyze failed");
  }
  return data;
}

async function analyzeCameraFrame() {
  if (!state.running || state.pending) {
    scheduleNextFrame();
    return;
  }
  if (!state.modelsReady) {
    setResultState("Loading models");
    scheduleNextFrame();
    return;
  }

  state.pending = true;
  try {
    const blob = await captureVideoBlob();
    if (blob) {
      const data = await sendImage(blob, "webcam");
      showAnnotatedImage(data, { saveHistory: false, source: "Live" });
    }
  } catch (error) {
    setResultState(error.message);
  } finally {
    state.pending = false;
    scheduleNextFrame();
  }
}

async function startCamera() {
  if (state.running) {
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 540 },
        facingMode: "user",
      },
      audio: false,
    });
    state.stream = stream;
    els.video.srcObject = stream;
    await els.video.play();
    state.running = true;
    els.stage.classList.add("camera-on");
    els.stage.classList.remove("has-result");
    els.startCamera.disabled = true;
    els.snapshotCamera.disabled = false;
    els.stopCamera.disabled = false;
    setResultState("Live");
    analyzeCameraFrame();
  } catch (error) {
    setResultState(error.message);
  }
}

function stopCamera() {
  state.running = false;
  if (state.timer) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
  if (state.stream) {
    for (const track of state.stream.getTracks()) {
      track.stop();
    }
  }
  state.stream = null;
  els.video.srcObject = null;
  els.stage.classList.remove("camera-on");
  els.startCamera.disabled = false;
  els.snapshotCamera.disabled = true;
  els.stopCamera.disabled = true;
  setResultState("Idle");
}

async function captureSnapshot() {
  if (!state.running || state.pending) {
    return;
  }
  if (!state.modelsReady) {
    setResultState("Loading models");
    return;
  }
  state.pending = true;
  setResultState("Capturing");

  if (state.timer) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }

  try {
    const blob = await captureVideoBlob();
    state.running = false;
    if (state.stream) {
      for (const track of state.stream.getTracks()) {
        track.stop();
      }
    }
    state.stream = null;
    els.video.srcObject = null;
    els.stage.classList.remove("camera-on");
    els.startCamera.disabled = false;
    els.snapshotCamera.disabled = true;
    els.stopCamera.disabled = true;

    const data = await sendImage(blob, "image");
    showAnnotatedImage(data, { saveHistory: true, source: "Snapshot" });
  } catch (error) {
    setResultState(error.message);
  } finally {
    state.pending = false;
  }
}

async function analyzeSelectedImage() {
  const file = els.imageInput.files[0];
  if (!file) {
    setResultState("Select an image");
    return;
  }
  if (!state.modelsReady) {
    setResultState("Loading models");
    return;
  }
  state.pending = true;
  setResultState("Analyzing");
  try {
    const data = await sendImage(file, "image");
    showAnnotatedImage(data, { saveHistory: true, source: file.name || "Image" });
  } catch (error) {
    setResultState(error.message);
  } finally {
    state.pending = false;
  }
}

function downloadCurrentResult() {
  if (!state.latestResult || !state.latestResult.image) {
    return;
  }
  const link = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  link.href = state.latestResult.image;
  link.download = `face-analysis-${stamp}.jpg`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function addHistory(data) {
  const entry = {
    ...data,
    id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
    createdAt: new Date().toLocaleTimeString(),
  };
  state.history.unshift(entry);
  state.history = state.history.slice(0, 10);
  renderHistory();
}

function renderHistory() {
  els.historyList.innerHTML = "";
  if (!state.history.length) {
    els.historyList.innerHTML = `<div class="empty-note">No saved results</div>`;
    return;
  }

  for (const item of state.history) {
    const button = document.createElement("button");
    button.className = "history-item";
    button.type = "button";
    const thumb = item.predictions[0]?.crop_image || item.image;
    const emotion = item.predictions[0]?.emotion || "No face";
    button.innerHTML = `
      <img class="history-thumb" src="${thumb}" alt="">
      <span class="history-meta">
        <span class="history-title">${item.source} · ${item.face_count} face${item.face_count === 1 ? "" : "s"}</span>
        <span class="history-subtitle">${emotion} · ${item.detector} · ${item.createdAt}</span>
      </span>
    `;
    button.addEventListener("click", () => {
      showAnnotatedImage(item, { saveHistory: false, source: item.source, detector: item.detector, quality: item.quality });
      setResultState("History");
    });
    els.historyList.appendChild(button);
  }
}

function bindEvents() {
  els.startCamera.addEventListener("click", startCamera);
  els.snapshotCamera.addEventListener("click", captureSnapshot);
  els.stopCamera.addEventListener("click", stopCamera);
  els.analyzeImage.addEventListener("click", analyzeSelectedImage);
  els.downloadResult.addEventListener("click", downloadCurrentResult);
  els.clearHistory.addEventListener("click", () => {
    state.history = [];
    renderHistory();
  });

  els.detectorSelect.addEventListener("change", () => {
    els.activeDetector.textContent = els.detectorSelect.value.toUpperCase();
  });

  els.intervalRange.addEventListener("input", () => {
    els.intervalValue.textContent = `${els.intervalRange.value} ms`;
  });

  for (const button of document.querySelectorAll(".segment")) {
    button.addEventListener("click", () => {
      for (const item of document.querySelectorAll(".segment")) {
        item.classList.remove("active");
      }
      button.classList.add("active");
      state.quality = button.dataset.quality;
    });
  }
}

bindEvents();
renderHistory();
pollStatus();
window.setInterval(pollStatus, 1500);
