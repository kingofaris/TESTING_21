// app.js — AI Text-to-Video Generator Frontend
// FastAPI + VAX Video · RTX 3050 4GB

const API_BASE = "http://localhost:8000";

let currentJobId = null;
let pollingInterval = null;
let elapsedTimer = null;
let elapsedSeconds = 0;

// ============================================================
// INISIALISASI
// ============================================================
window.onload = () => {
    checkServerStatus();
    loadGallery();
    setupRangeInputs();
    setupCharCounter();
    createToastContainer();
    
    // Check for ongoing job from previous session
    resumeJobIfAny();
};

// ============================================================
// RANGE INPUTS & CHAR COUNTER
// ============================================================
function setupRangeInputs() {
    const framesInput = document.getElementById("num-frames");
    const framesVal = document.getElementById("frames-val");
    const durationEst = document.getElementById("duration-est");

    if (framesInput) {
        framesInput.addEventListener("input", () => {
            const frames = parseInt(framesInput.value);
            framesVal.textContent = frames;
            const duration = (frames / 24).toFixed(1);
            durationEst.textContent = duration;
            
            // Tambahkan peringatan warna jika durasi terlalu panjang untuk 4GB
            if (frames > 80) {
                durationEst.parentElement.style.color = "#f87171"; // Merah
            } else {
                durationEst.parentElement.style.color = ""; // Normal
            }
            
            updateRangeTrack(framesInput);
        });
        // Initialize track fills
        updateRangeTrack(framesInput);
    }

    const stepsInput = document.getElementById("steps");
    const stepsVal = document.getElementById("steps-val");
    if (stepsInput) {
        stepsInput.addEventListener("input", () => {
            stepsVal.textContent = stepsInput.value;
            updateRangeTrack(stepsInput);
        });
        updateRangeTrack(stepsInput);
    }

    const cfgInput = document.getElementById("cfg");
    const cfgVal = document.getElementById("cfg-val");
    if (cfgInput) {
        cfgInput.addEventListener("input", () => {
            cfgVal.textContent = cfgInput.value;
            updateRangeTrack(cfgInput);
        });
        updateRangeTrack(cfgInput);
    }
}

function updateRangeTrack(input) {
    const min = parseFloat(input.min) || 0;
    const max = parseFloat(input.max) || 100;
    const val = parseFloat(input.value) || 0;
    const pct = ((val - min) / (max - min)) * 100;
    input.style.backgroundImage = `linear-gradient(to right, #7c3aed ${pct}%, #1e1e32 ${pct}%)`;
}

function setupCharCounter() {
    const prompt = document.getElementById("prompt");
    const counter = document.getElementById("char-count");
    if (prompt && counter) {
        prompt.addEventListener("input", () => {
            const len = prompt.value.length;
            counter.textContent = len;
            const counterEl = counter.parentElement;
            counterEl.classList.toggle("near-limit", len > 400);
        });
    }
}

// ============================================================
// SERVER STATUS CHECK
// ============================================================
async function checkServerStatus() {
    const badge = document.getElementById("status-badge");
    const statusText = document.getElementById("status-text");
    const gpuBar = document.getElementById("gpu-bar");

    if (!badge || !statusText) return;

    try {
        const res = await fetch(`${API_BASE}/health`, {
            signal: AbortSignal.timeout(5000)
        });
        const data = await res.json();

        if (data.status === "ok") {
            badge.className = "status-badge status-online";
            const icon = badge.querySelector(".status-icon");
            if (icon) icon.textContent = "✅";

            const gpu = data.gpu;
            if (gpu && gpu.gpu_name) {
                statusText.textContent = `Online · ${gpu.gpu_name}`;
                if (document.getElementById("gpu-name")) document.getElementById("gpu-name").textContent = gpu.gpu_name;
                if (document.getElementById("gpu-vram")) document.getElementById("gpu-vram").textContent = `${gpu.vram_free_gb} / ${gpu.vram_total_gb} GB`;
                if (document.getElementById("model-status")) document.getElementById("model-status").textContent = data.model_loaded ? "✅ Di-load" : "⏸ Belum di-load";
                if (document.getElementById("queue-size")) document.getElementById("queue-size").textContent = data.queue_size;
                if (gpuBar) gpuBar.style.display = "flex";

                // Jika server sedang memproses sesuatu dan kita belum polling, sambungkan otomatis
                if (data.active_job && !currentJobId) {
                    currentJobId = data.active_job;
                    localStorage.setItem("active_job_id", currentJobId);
                    resumeJobIfAny();
                }
            } else {
                statusText.textContent = "Server Online (Tidak ada GPU)";
                if (gpuBar) gpuBar.style.display = "none";
            }
        }
    } catch (e) {
        badge.className = "status-badge status-offline";
        const icon = badge.querySelector(".status-icon");
        if (icon) icon.textContent = "❌";
        statusText.textContent = "Server offline — jalankan: cd backend && python main.py";
        if (gpuBar) gpuBar.style.display = "none";
    }

    // Refresh setiap 30 detik
    setTimeout(checkServerStatus, 30000);
}

// ============================================================
// GENERATE VIDEO
// ============================================================
async function generateVideo() {
    const prompt = document.getElementById("prompt") ? document.getElementById("prompt").value.trim() : "";
    if (!prompt) {
        showToast("⚠️ Isi prompt terlebih dahulu!", "error");
        if (document.getElementById("prompt")) document.getElementById("prompt").focus();
        return;
    }

    const resolutionVal = document.getElementById("resolution") ? document.getElementById("resolution").value : "480x256";
    const resolution = resolutionVal.split("x");
    const requestBody = {
        prompt,
        negative_prompt: document.getElementById("negative-prompt") ? document.getElementById("negative-prompt").value : "worst quality, blurry",
        width: parseInt(resolution[0]),
        height: parseInt(resolution[1]),
        num_frames: document.getElementById("num-frames") ? parseInt(document.getElementById("num-frames").value) : 25,
        num_inference_steps: document.getElementById("steps") ? parseInt(document.getElementById("steps").value) : 40,
        guidance_scale: document.getElementById("cfg") ? parseFloat(document.getElementById("cfg").value) : 3.5,
        seed: document.getElementById("seed") ? parseInt(document.getElementById("seed").value) : -1,
    };

    // Update UI
    setGenerateButtonState(false, "⏳ Memproses...");
    showSection("progress");
    if (document.getElementById("progress-text")) document.getElementById("progress-text").textContent = "Mengirim request ke server...";

    try {
        const res = await fetch(`${API_BASE}/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `HTTP ${res.status} - Gagal kirim request`);
        }

        const data = await res.json();
        currentJobId = data.job_id;
        localStorage.setItem("active_job_id", currentJobId); // Save for persistence
        
        if (document.getElementById("job-id-display")) document.getElementById("job-id-display").textContent = currentJobId;
        if (document.getElementById("progress-text")) document.getElementById("progress-text").textContent =
            "Model sedang diproses... (bisa 2–10 menit ⏳)";

        startPolling(currentJobId);
        startElapsedTimer();
        showToast(`✅ Job dimulai: ${currentJobId}`, "info");

    } catch (error) {
        showError(error.message);
        showToast(`❌ ${error.message}`, "error");
    }
}

// ============================================================
// POLLING STATUS
// ============================================================
function startPolling(jobId) {
    pollingInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/status/${jobId}`);
            
            if (!res.ok) {
                if (res.status === 404) {
                    clearPolling();
                    showError("Job tidak ditemukan. Server mungkin direstart.");
                    showToast("❌ Job hilang (404)", "error");
                    return;
                }
            }
            
            const data = await res.json();

            if (data.status === "done") {
                clearPolling();
                showResult(data);
                loadGallery();
                showToast("🎬 Video berhasil digenerate!", "success");
                checkServerStatus(); // refresh GPU info
            } else if (data.status === "failed") {
                clearPolling();
                showError(data.error || "Generate gagal tanpa pesan error");
                showToast("❌ Generate gagal!", "error");
            } else if (data.status === "processing") {
                // Update text based on model state
                const progressText = document.getElementById("progress-text");
                const progressFill = document.getElementById("progress-fill");
                if (progressText) {
                    if (data.model_loaded) {
                        progressText.textContent = "Sedang generate video... (ini proses GPU 🚀)";
                        if (progressFill) progressFill.style.animationDuration = "0.8s";
                    } else {
                        progressText.textContent = "Sedang loading model ke VRAM... (sabar ya ⏳)";
                        if (progressFill) progressFill.style.animationDuration = "2s";
                    }
                }
            }
        } catch (e) {
            console.error("[Polling] Error:", e);
        }
    }, 3000); // cek setiap 3 detik
}

function startElapsedTimer() {
    elapsedSeconds = 0;
    const elapsedEl = document.getElementById("elapsed-time");
    if (!elapsedEl) return;
    
    elapsedTimer = setInterval(() => {
        elapsedSeconds++;
        if (elapsedSeconds < 60) {
            elapsedEl.textContent = `${elapsedSeconds}s berlalu`;
        } else {
            const mins = Math.floor(elapsedSeconds / 60);
            const secs = elapsedSeconds % 60;
            elapsedEl.textContent = `${mins}m ${secs}s berlalu`;
        }
    }, 1000);
}

function clearPolling() {
    if (pollingInterval) { clearInterval(pollingInterval); pollingInterval = null; }
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    localStorage.removeItem("active_job_id");
}

// Tambahkan fungsi untuk melanjutkan job jika halaman di-refresh
async function resumeJobIfAny() {
    const savedJobId = localStorage.getItem("active_job_id");
    if (savedJobId) {
        console.log("[Auto-Resume] Menghubungkan kembali ke Job:", savedJobId);
        showSection("progress");
        currentJobId = savedJobId;
        if (document.getElementById("job-id-display")) document.getElementById("job-id-display").textContent = currentJobId;
        if (document.getElementById("progress-text")) document.getElementById("progress-text").textContent = "Menghubungkan kembali ke proses generate...";
        startPolling(currentJobId);
        
        // Estimasi elapsed time bisa sulit, coba ambil dari server jika ada
        startElapsedTimer(); 
    }
}

// ============================================================
// UI SECTIONS
// ============================================================
function showSection(section) {
    ["progress", "result", "error"].forEach(s => {
        const el = document.getElementById(`${s}-section`);
        if (el) el.style.display = "none";
    });
    if (section) {
        const target = document.getElementById(`${section}-section`);
        if (target) {
            target.style.display = "block";
            target.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    }
}

function showResult(data) {
    showSection("result");
    const videoEl = document.getElementById("result-video");
    const videoUrl = `${API_BASE}${data.video_url}`;

    if (videoEl) {
        videoEl.src = videoUrl;
        videoEl.load();
        videoEl.play().catch(() => {}); // autoplay might be blocked
    }

    if (document.getElementById("result-seed")) document.getElementById("result-seed").textContent = data.seed ?? "—";
    if (document.getElementById("result-duration")) document.getElementById("result-duration").textContent = data.duration_seconds ?? "—";
    if (document.getElementById("result-filename")) document.getElementById("result-filename").textContent = data.filename ?? "—";

    const downloadBtn = document.getElementById("download-btn");
    if (downloadBtn) {
        downloadBtn.href = videoUrl;
        downloadBtn.download = data.filename;
    }

    setGenerateButtonState(true, "🚀 Generate Video");
}

function showError(message) {
    showSection("error");
    if (document.getElementById("error-msg")) document.getElementById("error-msg").textContent = message;
    setGenerateButtonState(true, "🚀 Generate Video");
}

function resetForm() {
    showSection(null);
    clearPolling();
    setGenerateButtonState(true, "🚀 Generate Video");
}

function setGenerateButtonState(enabled, label) {
    const btn = document.getElementById("generate-btn");
    if (btn) {
        btn.disabled = !enabled;
        const span = btn.querySelector("span:last-child") || btn;
        span.textContent = label;
    }
}

// ============================================================
// GALLERY
// ============================================================
async function loadGallery() {
    const refreshBtn = document.getElementById("refresh-btn");
    if (refreshBtn) {
        refreshBtn.style.opacity = "0.5";
        refreshBtn.disabled = true;
    }

    try {
        const res = await fetch(`${API_BASE}/videos`);
        const data = await res.json();
        const gallery = document.getElementById("gallery");

        if (!gallery) return;

        if (!data.videos || data.videos.length === 0) {
            gallery.innerHTML = `
                <div class="gallery-empty">
                    <div class="empty-icon">🎬</div>
                    <p>Belum ada video. Generate video pertamamu!</p>
                </div>`;
            return;
        }

        gallery.innerHTML = data.videos.map(v => {
            const nameParts = v.filename.split("_");
            const dateStr = nameParts[1] || "";
            const displayDate = dateStr
                ? `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`
                : v.filename;

            const promptText = v.prompt ? v.prompt : "Prompt tidak tersedia";
            const seedText = v.seed !== undefined && v.seed !== null ? v.seed : "—";
            const durationText = v.duration_seconds ? `${v.duration_seconds}s` : "—";

            return `
            <div class="gallery-item" id="item-${v.filename}">
                <video src="${API_BASE}${v.url}" controls loop muted preload="metadata"></video>
                
                <div style="padding: 12px; background: rgba(255, 255, 255, 0.03); border-bottom: 1px solid rgba(255, 255, 255, 0.05); font-size: 13px;">
                    <div style="margin-bottom: 8px; color: white; line-height: 1.4;">
                        <strong><i class="fas fa-comment-dots" style="color: #4A4AFF; margin-right: 4px;"></i> Prompt:</strong><br>
                        <span style="opacity: 0.8; word-break: break-word;">${promptText}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; color: rgba(255, 255, 255, 0.6); margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                        <span><i class="fas fa-fingerprint" style="margin-right: 4px;"></i> Seed: ${seedText}</span>
                        <span><i class="fas fa-clock" style="margin-right: 4px;"></i> ${durationText}</span>
                    </div>
                </div>

                <div class="gallery-meta">
                    <span class="gallery-filename" title="${v.filename}">${displayDate}</span>
                    <span class="gallery-size">${v.size_mb} MB</span>
                </div>
                <button class="gallery-delete" onclick="deleteVideo('${v.filename}')">
                    🗑 Hapus
                </button>
            </div>`;
        }).join("");

    } catch (e) {
        console.error("[Gallery] Gagal load:", e);
    } finally {
        if (refreshBtn) {
            refreshBtn.style.opacity = "1";
            refreshBtn.disabled = false;
        }
    }
}

async function deleteVideo(filename) {
    if (!confirm(`Hapus video ${filename}?`)) return;

    try {
        const res = await fetch(`${API_BASE}/videos/${filename}`, { method: "DELETE" });
        if (res.ok) {
            const item = document.getElementById(`item-${filename}`);
            if (item) {
                item.style.opacity = "0";
                item.style.transform = "scale(0.9)";
                item.style.transition = "all 0.3s ease";
                setTimeout(() => { item.remove(); }, 300);
            }
            showToast(`✅ ${filename} dihapus`, "success");
        }
    } catch (e) {
        showToast("❌ Gagal menghapus video", "error");
    }
}

// ============================================================
// MODEL CONTROLS
// ============================================================
async function loadModel() {
    showToast("⏳ Loading model ke memori...", "info");
    try {
        const res = await fetch(`${API_BASE}/model/load`, { method: "POST" });
        const data = await res.json();
        if (data.success) {
            showToast("✅ Model berhasil di-load!", "success");
            checkServerStatus();
        } else {
            showToast("❌ Gagal load model", "error");
        }
    } catch (e) {
        showToast("❌ Server tidak dapat dijangkau", "error");
    }
}

async function unloadModel() {
    showToast("📤 Unloading model...", "info");
    try {
        const res = await fetch(`${API_BASE}/model/unload`, { method: "POST" });
        const data = await res.json();
        if (data.success) {
            showToast("✅ Model berhasil di-unload. VRAM dibebaskan!", "success");
            checkServerStatus();
        }
    } catch (e) {
        showToast("❌ Server tidak dapat dijangkau", "error");
    }
}

// ============================================================
// TOAST NOTIFICATIONS
// ============================================================
function createToastContainer() {
    if (document.getElementById("toast-container")) return;
    const el = document.createElement("div");
    el.className = "toast-container";
    el.id = "toast-container";
    document.body.appendChild(el);
}

function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    if (!container) return;
    
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(20px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
