/* ──────────────────────────────────────────────────────────────────
   IncuBrix Music Engine – frontend script
   ─────────────────────────────────────────────────────────────────*/

const API = "";            // Same origin – no base URL needed
const POLL_MS = 1500;      // Job-status polling interval

let pollTimer = null;
let sectionCount = 0;

// ── DOM refs ────────────────────────────────────────────────────────
const form           = document.getElementById("music-form");
const submitBtn      = document.getElementById("submit-btn");
const formError      = document.getElementById("form-error");
const statusCard     = document.getElementById("status-card");
const resultCard     = document.getElementById("result-card");
const statusBadge    = document.getElementById("status-badge");
const jobIdDisplay   = document.getElementById("job-id-display");
const progressBar    = document.getElementById("progress-bar");
const statusMessage  = document.getElementById("status-message");
const mixPlayer      = document.getElementById("mix-player");
const dlMixWav       = document.getElementById("dl-mix-wav");
const dlMixMp3       = document.getElementById("dl-mix-mp3");
const dlManifest     = document.getElementById("dl-manifest");
const stemsGrid      = document.getElementById("stems-grid");
const metaPre        = document.getElementById("meta-pre");
const sectionsContainer = document.getElementById("sections-container");
const addSectionBtn  = document.getElementById("add-section");

// ── Sections builder ─────────────────────────────────────────────────
addSectionBtn.addEventListener("click", addSection);

function addSection() {
  sectionCount++;
  const row = document.createElement("div");
  row.className = "section-row";
  row.dataset.id = sectionCount;
  row.innerHTML = `
    <input type="number" placeholder="Start (s)" min="0" step="0.5"
           class="sec-start" title="Section start time in seconds" />
    <input type="text" placeholder="Mood description"
           class="sec-mood" title="Mood for this section" />
    <button type="button" class="btn-remove" onclick="removeSection(this)">✕</button>`;
  sectionsContainer.appendChild(row);
}

function removeSection(btn) {
  btn.closest(".section-row").remove();
}

// ── Form submission ───────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();

  // Build request body
  const prompt   = document.getElementById("prompt").value.trim();
  const duration = parseFloat(document.getElementById("duration").value);
  const tempo    = document.getElementById("tempo").value.trim();
  const keyVal   = document.getElementById("key").value;
  const seedVal  = document.getElementById("seed").value.trim();
  const sr       = parseInt(document.getElementById("sample-rate").value, 10);
  const lufs     = parseFloat(document.getElementById("target-lufs").value);

  // Client-side validation
  if (!prompt) { showError("Please describe the music mood."); return; }
  if (isNaN(duration) || duration < 5 || duration > 300) {
    showError("Duration must be between 5 and 300 seconds."); return;
  }

  // Collect sections
  const sections = [];
  for (const row of sectionsContainer.querySelectorAll(".section-row")) {
    const start = parseFloat(row.querySelector(".sec-start").value);
    const mood  = row.querySelector(".sec-mood").value.trim();
    if (isNaN(start) || !mood) { showError("Each section needs a start time and a mood."); return; }
    if (start < 0 || start >= duration) {
      showError(`Section start ${start}s must be between 0 and ${duration - 0.01}s.`); return;
    }
    sections.push({ start_s: start, mood });
  }

  const body = {
    prompt,
    duration_s: duration,
    output: { sample_rate: sr, target_lufs: lufs, formats: ["wav", "mp3"] },
  };
  if (tempo)   body.tempo_bpm = parseInt(tempo, 10);
  if (keyVal)  body.key = keyVal;
  if (seedVal) body.seed = parseInt(seedVal, 10);
  if (sections.length) body.sections = sections;

  // Disable form
  submitBtn.disabled = true;
  submitBtn.textContent = "⏳ Submitting…";
  clearResults();

  try {
    const res = await fetch(`${API}/api/v1/music/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      const msg = err?.message || err?.detail?.message || err?.detail || err?.message || `HTTP ${res.status}`;
      throw new Error(msg);
    }

    const job = await res.json();
    startPolling(job.job_id);
  } catch (err) {
    showError(`Submission failed: ${err.message}`);
    submitBtn.disabled = false;
    submitBtn.textContent = "🎼 Generate Music";
  }
});

// ── Polling ───────────────────────────────────────────────────────────
function startPolling(jobId) {
  clearPolling();
  showStatusCard(jobId, "queued", 0, "Job queued – generation starting…");
  pollTimer = setInterval(() => pollJob(jobId), POLL_MS);
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`${API}/api/v1/jobs/${jobId}`);
    if (!res.ok) return;
    const job = await res.json();
    updateStatus(job);
    if (job.status === "succeeded" || job.status === "failed") {
      clearPolling();
      submitBtn.disabled = false;
      submitBtn.textContent = "🎼 Generate Music";
    }
  } catch (_) { /* network hiccup – keep polling */ }
}

function clearPolling() {
  if (pollTimer !== null) { clearInterval(pollTimer); pollTimer = null; }
}

// ── Status updates ────────────────────────────────────────────────────
function showStatusCard(jobId, status, progress, message) {
  statusCard.classList.remove("hidden");
  jobIdDisplay.textContent = `Job: ${jobId}`;
  setBadge(status);
  setProgress(progress);
  statusMessage.textContent = message;
}

function updateStatus(job) {
  statusCard.classList.remove("hidden");
  setBadge(job.status);
  setProgress(job.progress ?? 0);

  const messages = {
    queued:    "Job queued – waiting for a worker thread…",
    running:   `Generating music… ${job.progress ?? 0}%`,
    succeeded: "✅ Generation complete!",
    failed:    `❌ Generation failed: ${job.error?.message || "Unknown error"}`,
  };
  statusMessage.textContent = messages[job.status] || job.status;

  if (job.status === "succeeded" && job.downloads) {
    showResults(job);
  }
}

function setBadge(status) {
  statusBadge.textContent = status;
  statusBadge.className = `badge ${status}`;
}

function setProgress(pct) {
  progressBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

// ── Results ───────────────────────────────────────────────────────────
function showResults(job) {
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

  const links = job.downloads;

  // Mix audio
  if (links.mix_wav || links.mix_mp3) {
    const mixSrc = links.mix_wav || links.mix_mp3;
    mixPlayer.src = mixSrc;
    mixPlayer.load();
  }

  if (links.mix_wav) {
    dlMixWav.href = links.mix_wav;
    dlMixWav.setAttribute("download", "mix.wav");
    dlMixWav.classList.remove("hidden");
  }
  if (links.mix_mp3) {
    dlMixMp3.href = links.mix_mp3;
    dlMixMp3.setAttribute("download", "mix.mp3");
    dlMixMp3.classList.remove("hidden");
  }
  if (links.manifest) {
    dlManifest.href = links.manifest;
    dlManifest.setAttribute("download", "music_manifest.json");
    dlManifest.classList.remove("hidden");
  }

  // Stems
  stemsGrid.innerHTML = "";
  const stemNames = ["drums", "bass", "harmony", "melody"];
  for (const name of stemNames) {
    const key = `stem_${name}`;
    if (!links[key]) continue;
    const card = document.createElement("div");
    card.className = "stem-card";
    card.innerHTML = `
      <div class="stem-title">${stemIcon(name)} ${name}</div>
      <audio controls preload="metadata" src="${links[key]}"></audio>
      <div>
        <a class="btn-dl secondary" href="${links[key]}" download="${name}.wav">⬇ ${name}.wav</a>
      </div>`;
    stemsGrid.appendChild(card);
  }

  // Metadata
  if (job.manifest) {
    const meta = {
      tempo_bpm: job.manifest.tempo_bpm,
      key: job.manifest.key,
      duration_s: job.manifest.delivered_duration_s,
      actual_lufs: job.manifest.actual_lufs,
      peak_dbfs: job.manifest.peak_dbfs,
      seed: job.manifest.seed,
      generation_method: job.manifest.generation_method?.method,
      provider: job.manifest.generation_method?.provider,
      generation_time_s: job.manifest.timings?.generation_time_s,
      rights_status: job.manifest.licensing?.rights_status,
      warnings: job.manifest.warnings,
    };
    metaPre.textContent = JSON.stringify(meta, null, 2);
  }
}

function stemIcon(name) {
  const icons = { drums: "🥁", bass: "🎸", harmony: "🎹", melody: "🎺" };
  return icons[name] || "🎵";
}

// ── Helpers ────────────────────────────────────────────────────────────
function clearResults() {
  resultCard.classList.add("hidden");
  stemsGrid.innerHTML = "";
  metaPre.textContent = "";
  mixPlayer.src = "";
  [dlMixWav, dlMixMp3, dlManifest].forEach(el => el.classList.add("hidden"));
}

function showError(msg) {
  formError.textContent = msg;
  formError.classList.remove("hidden");
}

function hideError() {
  formError.textContent = "";
  formError.classList.add("hidden");
}
