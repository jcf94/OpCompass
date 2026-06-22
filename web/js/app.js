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

// ── Analyze ────────────────────────────────────────────────────
async function runAnalysis() {
    const opName = $opSelect.value;
    const hwName = $hwSelect.value;
    const dtype = $dtypeSelect.value;
    const mode = $modeSelect.value;
    const dims = collectDims();

    $analyzeBtn.textContent = "Analyzing…";
    $analyzeBtn.disabled = true;

    try {
        const result = await API.analyze(opName, hwName, dtype, mode, dims);
        currentResult = result;
        renderResults(result);
    } catch (err) {
        console.error("Analysis failed:", err);
        alert("Analysis failed: " + err.message);
    } finally {
        $analyzeBtn.textContent = "Run Analysis";
        $analyzeBtn.disabled = false;
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
$opSelect.addEventListener("change", updateDimInputs);
$hwSelect.addEventListener("change", updateHardwareInfo);
$analyzeBtn.addEventListener("click", runAnalysis);

// ── Boot ──────────────────────────────────────────────────────
init();
