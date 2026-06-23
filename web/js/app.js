/**
 * Main application logic — form handling, API calls, result rendering.
 */

// ── State ─────────────────────────────────────────────────────
let operators = [];
let hardware = [];
let currentResult = null;

// ── DOM refs ──────────────────────────────────────────────────
const $opSelect = document.getElementById("operator-select");
const $hwSelect = document.getElementById("hardware-select");
const $dtypeSelect = document.getElementById("dtype-select");
const $modeSelect = document.getElementById("mode-select");
const $dimInputs = document.getElementById("dim-inputs");
const $analyzeBtn = document.getElementById("analyze-btn");
const $hwSpec = document.getElementById("hw-spec-content");
const $pipelineConfig = document.getElementById("pipeline-config");
const $asyncCopyToggle = document.getElementById("async-copy-toggle");
const $sparsityToggle = document.getElementById("sparsity-toggle");

const $solTime = document.getElementById("sol-time");
const $solTflops = document.getElementById("sol-tflops");
const $solBottleneck = document.getElementById("sol-bottleneck");
const $bottleneckCard = document.getElementById("bottleneck-card");
const $detailTable = document.getElementById("detail-table").querySelector("tbody");

// ── Initialization ────────────────────────────────────────────
async function init() {
    try {
        [operators, hardware] = await Promise.all([
            API.getOperators(),
            API.getHardware(),
        ]);
    } catch (err) {
        console.error("Failed to load data:", err);
        return;
    }

    // Populate operator select
    $opSelect.innerHTML = operators
        .map(op => `<option value="${op.name}">${op.name} — ${op.description}</option>`)
        .join("");

    // Populate hardware select (sorted by SM version desc from API)
    $hwSelect.innerHTML = hardware
        .map(hw => {
            const smv = hw.sm_version ? ` · SM ${hw.sm_version}` : '';
            return `<option value="${hw.name}">${hw.name.toUpperCase()} — ${hw.architecture || hw.vendor}${smv}</option>`;
        })
        .join("");

    // Default: matmul + a100
    if (operators.find(o => o.name === "matmul")) $opSelect.value = "matmul";
    if (hardware.find(h => h.name === "a100")) $hwSelect.value = "a100";

    updateDimInputs();
    updateHardwareInfo();
}

// ── Dynamic dimension inputs ──────────────────────────────────
function updateDimInputs() {
    const opName = $opSelect.value;
    const op = operators.find(o => o.name === opName);
    if (!op) return;

    $dimInputs.innerHTML = Object.entries(op.param_dims)
        .map(([dim, desc]) => `
            <label for="dim-${dim}">${dim} <span class="dim-desc">${desc}</span></label>
            <input id="dim-${dim}" type="number" value="${getDefaultDim(dim)}" min="1" step="1">
        `)
        .join("");
}

function getDefaultDim(dim) {
    const defaults = {
        M: 4096, N: 4096, K: 4096,
        B: 1, H: 32, S: 4096, D: 128,
        C_in: 3, C_out: 64, H_img: 224, W: 224, K_h: 3, K_w: 3, H_out: 224, W_out: 224,
        ops_per_element: 1,
    };
    return defaults[dim] || 1024;
}

function collectDims() {
    const dims = {};
    const inputs = $dimInputs.querySelectorAll("input");
    inputs.forEach(inp => {
        const dimName = inp.id.replace("dim-", "");
        dims[dimName] = parseInt(inp.value) || 0;
    });
    return dims;
}

// ── Hardware info display ─────────────────────────────────────
async function updateHardwareInfo() {
    const hwName = $hwSelect.value;
    if (!hwName) return;

    try {
        const detail = await API.getHardwareDetail(hwName);
        const cu = detail.compute_unit;

        const peakRows = Object.entries(cu.peak_flops)
            .map(([dt, flops]) => {
                let display;
                if (flops >= 1e12) display = (flops / 1e12).toFixed(0) + " TFLOPS";
                else if (flops >= 1e9) display = (flops / 1e9).toFixed(1) + " GFLOPS";
                else display = flops;
                return `<div class="spec-row"><span class="spec-label">Peak ${dt.toUpperCase()}</span><span class="spec-value">${display}</span></div>`;
            })
            .join("");

        $hwSpec.innerHTML = `
            <p style="margin-bottom:0.5rem;font-family:var(--font-mono);font-size:0.85rem;font-weight:600;color:var(--text-emphasis)">${detail.vendor} ${detail.name.toUpperCase()}</p>
            <div class="spec-row"><span class="spec-label">Compute Units</span><span class="spec-value">${cu.count} ${cu.name}s @ ${cu.clock_mhz} MHz</span></div>
            ${detail.memory_tiers.map(t => `
                <div class="spec-row"><span class="spec-label">${t.name}</span><span class="spec-value">${t.capacity_gb.toFixed(0)} GB, ${t.bandwidth_gb_s.toFixed(0)} GB/s</span></div>
            `).join("")}
            ${peakRows}
        `;
    } catch (err) {
        $hwSpec.innerHTML = `<p style="color:var(--bottleneck)">Error loading specs.</p>`;
    }
}

// ── Pipeline config toggle ─────────────────────────────────────
function togglePipelineConfig() {
    if ($modeSelect.value === "pipeline") {
        $pipelineConfig.classList.remove("hidden");
    } else {
        $pipelineConfig.classList.add("hidden");
    }
    // Toggle solar results visibility
    const $solarResults = document.getElementById("solar-results");
    if ($modeSelect.value === "solar") {
        $solarResults.classList.remove("hidden");
    } else {
        $solarResults.classList.add("hidden");
    }
}

function collectPipelineConfig() {
    if ($modeSelect.value !== "pipeline") return null;
    return {
        async_copy_enabled: $asyncCopyToggle.checked,
        sparsity_2_4_enabled: $sparsityToggle.checked,
    };
}

// ── Analyze ────────────────────────────────────────────────────
let _analysisPending = false;
let _analysisDebounceTimer = null;
let _analysisInFlight = false;

async function runAnalysis() {
    if (_analysisInFlight) {
        // If already running, mark pending and return — the running
        // one will re-trigger when it finishes.
        _analysisPending = true;
        return;
    }

    const opName = $opSelect.value;
    const hwName = $hwSelect.value;
    const dtype = $dtypeSelect.value;
    const mode = $modeSelect.value;
    const dims = collectDims();
    const pipelineConfig = collectPipelineConfig();

    _analysisInFlight = true;
    $analyzeBtn.textContent = "Analyzing…";
    $analyzeBtn.disabled = true;

    try {
        const result = await API.analyze(opName, hwName, dtype, mode, dims, pipelineConfig);
        currentResult = result;
        renderResults(result);
    } catch (err) {
        console.error("Analysis failed:", err);
        alert("Analysis failed: " + err.message);
    } finally {
        $analyzeBtn.textContent = "Rerun Analysis";
        $analyzeBtn.disabled = false;
        _analysisInFlight = false;

        // If a change arrived while we were computing, re-run now
        if (_analysisPending) {
            _analysisPending = false;
            runAnalysis();
        }
    }
}

/** Debounced trigger — fires on every change, but waits 400ms of
 *  inactivity (or immediate for select changes). */
function triggerAnalysis(immediate) {
    if (immediate) {
        if (_analysisDebounceTimer) clearTimeout(_analysisDebounceTimer);
        _analysisPending = false;
        runAnalysis();
    } else {
        if (_analysisDebounceTimer) clearTimeout(_analysisDebounceTimer);
        _analysisDebounceTimer = setTimeout(() => {
            _analysisPending = false;
            runAnalysis();
        }, 400);
    }
}

function renderResults(data) {
    // Metric cards
    $solTime.textContent = data.sol_time_us.toFixed(1);
    $solTflops.textContent = data.sol_tflops.toFixed(1);

    const bottleneck = data.bottleneck;
    $solBottleneck.textContent = bottleneck.replace("_", " ");
    $bottleneckCard.className = "metric-card " + bottleneck;

    // Detail table
    $detailTable.innerHTML = `
        <tr><td>Total FLOPs</td><td>${formatFlops(data.total_flops)}</td></tr>
        <tr><td>Read Bytes</td><td>${formatBytes(data.total_read_bytes)}</td></tr>
        <tr><td>Write Bytes</td><td>${formatBytes(data.total_write_bytes)}</td></tr>
        <tr><td>Memory Read Time</td><td>${data.memory_read_time_us.toFixed(1)} µs</td></tr>
        <tr><td>Compute Time</td><td>${data.compute_time_us.toFixed(1)} µs</td></tr>
        <tr><td>Memory Write Time</td><td>${data.memory_write_time_us.toFixed(1)} µs</td></tr>
        <tr><td><strong>SOL Time</strong></td><td><strong>${data.sol_time_us.toFixed(1)} µs</strong></td></tr>
        <tr><td>Arithmetic Intensity</td><td>${(data.total_flops / (data.total_read_bytes + data.total_write_bytes)).toFixed(2)} FLOP/Byte</td></tr>
    `;

    // Charts — need to reconstruct roofline data
    renderBreakdownChart(data);

    const peakFlops = data.sol_tflops * 1e12;
    const peakBw = data.sol_time_us > 0 && data.memory_read_time_us > 0
        ? data.total_read_bytes / data.memory_read_time_us * 1e6
        : 2e12;

    const totalIo = data.total_read_bytes + data.total_write_bytes;
    const oi = totalIo > 0 ? data.total_flops / totalIo : 1000;
    const achievable = Math.min(peakFlops, oi * peakBw);

    renderRooflineChart({
        operational_intensity: oi,
        peak_flops: peakFlops,
        peak_bandwidth: peakBw,
        achievable_flops: achievable,
    }, data);

    // Solar-specific rendering
    if (data.solar_data) {
        renderSolarResults(data.solar_data);
    }

    // Pipeline-specific rendering
    if (typeof PipelineUI !== "undefined") {
        PipelineUI.render(data);
    }
}

function renderSolarResults(sd) {
    // Show solar results section
    const $solarResults = document.getElementById("solar-results");
    $solarResults.classList.remove("hidden");

    // Performance models table
    const $perfTbody = document.getElementById("solar-perf-table").querySelector("tbody");
    const models = [
        { label: "Unfused", m: sd.unfused },
        { label: "Fused", m: sd.fused },
        { label: "Fused+Prefetched ★", m: sd.fused_prefetched },
    ];
    $perfTbody.innerHTML = models.map(({ label, m }) => `
        <tr>
            <td>${label}</td>
            <td>${m.runtime_ms.toFixed(4)}</td>
            <td>${m.bottleneck}</td>
            <td>${m.arithmetic_intensity.toFixed(1)}</td>
            <td>${(m.memory_bytes / 1e9).toFixed(4)}</td>
        </tr>
    `).join("");

    // Speedup table
    const $speedTbody = document.getElementById("solar-speedup-table").querySelector("tbody");
    $speedTbody.innerHTML = `
        <tr><td>Fused vs Unfused</td><td>${sd.speedup.fused_vs_unfused.toFixed(2)}×</td></tr>
        <tr><td>Fused+Prefetched vs Unfused</td><td>${sd.speedup.fused_prefetched_vs_unfused.toFixed(2)}×</td></tr>
    `;

    // Runtime bar chart
    renderSolarRuntimeChart(sd);

    // Memory pie/doughnut chart
    renderSolarMemoryChart(sd);
}

function renderSolarRuntimeChart(sd) {
    const canvas = document.getElementById("solar-runtime-chart");
    if (!canvas) return;
    // Destroy previous chart instance stored on the element
    if (canvas._chart) canvas._chart.destroy();

    canvas._chart = new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels: ["Unfused", "Fused", "Fused+Prefetched"],
            datasets: [{
                label: "Runtime (ms)",
                data: [sd.unfused.runtime_ms, sd.fused.runtime_ms, sd.fused_prefetched.runtime_ms],
                backgroundColor: [C.memoryBg, C.successBg, C.computeBg],
                borderColor: [C.memory, C.success, C.compute],
                borderWidth: 1.5,
                borderRadius: 4,
                barThickness: 28,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                y: {
                    title: {
                        display: true, text: "Runtime (ms)",
                        color: C.muted,
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    ticks: {
                        color: C.dim,
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    grid: { color: C.gridLight },
                },
                x: {
                    ticks: {
                        color: C.muted,
                        font: { family: "'Inter', sans-serif", size: 11 },
                    },
                    grid: { display: false },
                },
            },
        },
    });
}

function renderSolarMemoryChart(sd) {
    const canvas = document.getElementById("solar-memory-chart");
    if (!canvas) return;
    if (canvas._chart) canvas._chart.destroy();

    const mb = sd.memory_breakdown;
    canvas._chart = new Chart(canvas.getContext("2d"), {
        type: "doughnut",
        data: {
            labels: ["Weights", "Model I/O", "Intermediate"],
            datasets: [{
                data: [mb.weight_bytes, mb.model_io_bytes, mb.intermediate_bytes],
                backgroundColor: [C.computeBg, C.memoryBg, C.successBg],
                borderColor: [C.compute, C.memory, C.success],
                borderWidth: 1.5,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: C.muted,
                        padding: 12,
                        font: { family: "'Inter', sans-serif", size: 10 },
                    },
                },
            },
        },
    });
}

// ── Helpers ────────────────────────────────────────────────────
function formatFlops(n) {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + " TFLOPs";
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GFLOPs";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + " MFLOPs";
    return n + " FLOPs";
}

function formatBytes(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + " MB";
    if (n >= 1e3) return (n / 1e3).toFixed(2) + " KB";
    return n + " B";
}

// ── Tab navigation ─────────────────────────────────────────────
const $tabBtns = document.querySelectorAll(".tab-btn");
const $pages = document.querySelectorAll(".page");

function switchTab(tabName) {
    $tabBtns.forEach(btn => btn.classList.toggle("active", btn.dataset.tab === tabName));
    $pages.forEach(page => page.classList.toggle("active", page.id === `page-${tabName}`));

    // Lazy-init hardware page on first visit
    if (tabName === "hardware" && typeof initHardwarePage === "function" && !window._hwPageInited) {
        window._hwPageInited = true;
        initHardwarePage();
    }

    // Lazy-init overview page on first visit
    if (tabName === "overview" && typeof initOverviewPage === "function" && !window._overviewPageInited) {
        window._overviewPageInited = true;
        initOverviewPage();
    }
}

$tabBtns.forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// ── Event listeners ────────────────────────────────────────────

// Select changes → update UI + immediate re-analysis
$opSelect.addEventListener("change", () => {
    updateDimInputs();
    triggerAnalysis(true);
});
$hwSelect.addEventListener("change", () => {
    updateHardwareInfo();
    triggerAnalysis(true);
});
$dtypeSelect.addEventListener("change", () => triggerAnalysis(true));
$modeSelect.addEventListener("change", () => {
    togglePipelineConfig();
    triggerAnalysis(true);
});

// Pipeline toggles → immediate re-analysis (only relevant in pipeline mode)
$asyncCopyToggle.addEventListener("change", () => triggerAnalysis(true));
$sparsityToggle.addEventListener("change", () => triggerAnalysis(true));

// Manual rerun button
$analyzeBtn.addEventListener("click", () => {
    _analysisPending = false;
    if (_analysisDebounceTimer) clearTimeout(_analysisDebounceTimer);
    runAnalysis();
});

// Dimension inputs → debounced re-analysis (avoid flooding on keystrokes)
$dimInputs.addEventListener("input", (e) => {
    if (e.target.tagName === "INPUT") {
        triggerAnalysis(false);
    }
});

// ── Boot ──────────────────────────────────────────────────────
async function boot() {
    await init();
    togglePipelineConfig();
    // Run initial analysis with defaults
    runAnalysis();
}
boot();
