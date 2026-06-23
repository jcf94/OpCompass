/**
 * Chart rendering — time breakdown bar chart & roofline model.
 * "Signal Path" light theme palette.
 */

const C = {
    compute:    '#1a56db',
    computeBg:  'rgba(26,86,219,0.08)',
    memory:     '#7c3aed',
    memoryBg:   'rgba(124,58,237,0.08)',
    success:    '#059669',
    successBg:  'rgba(5,150,105,0.08)',
    bottleneck: '#dc2626',
    accent:     '#1a56db',
    grid:       '#e6e4e0',
    gridLight:  '#f0eeea',
    text:       '#1b1b1b',
    muted:      '#6f6f6f',
    dim:        '#9c9c9c',
    surface:    '#ffffff',
    border:     '#e6e4e0',
};

let breakdownChart = null;
let rooflineChart = null;

// ── Chart.js global defaults ──────────────────────────────────────
const CHARTS_AVAILABLE = typeof Chart !== 'undefined';
window.CHARTS_AVAILABLE = CHARTS_AVAILABLE;

if (CHARTS_AVAILABLE) {
    Chart.defaults.color = C.muted;
    Chart.defaults.borderColor = C.grid;
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.plugins.tooltip.backgroundColor = '#1b1b1b';
    Chart.defaults.plugins.tooltip.titleColor = '#fff';
    Chart.defaults.plugins.tooltip.bodyColor = '#e0e0e0';
    Chart.defaults.plugins.tooltip.cornerRadius = 6;
    Chart.defaults.plugins.tooltip.padding = 10;
}

function renderChartUnavailable(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;
    canvas.style.display = 'none';
    let msg = parent.querySelector('.chart-unavailable');
    if (!msg) {
        msg = document.createElement('div');
        msg.className = 'chart-unavailable';
        msg.textContent = 'Chart library unavailable. Metrics tables are still valid.';
        parent.appendChild(msg);
    }
}

function renderBreakdownChart(data) {
    if (!CHARTS_AVAILABLE) {
        renderChartUnavailable('breakdown-chart');
        return;
    }
    const ctx = document.getElementById('breakdown-chart').getContext('2d');
    if (breakdownChart) breakdownChart.destroy();

    const labels = ['Memory Read', 'Compute', 'Memory Write'];
    const values = [
        data.memory_read_time_us,
        data.compute_time_us,
        data.memory_write_time_us,
    ];

    breakdownChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Time (µs)',
                data: values,
                backgroundColor: [C.memoryBg, C.computeBg, C.successBg],
                borderColor: [C.memory, C.compute, C.success],
                borderWidth: 1.5,
                borderRadius: 4,
                barThickness: 24,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
                    bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
                    callbacks: {
                        label(ctx) { return ` ${ctx.parsed.x.toFixed(1)} µs`; },
                    },
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time (µs)',
                        color: C.muted,
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    ticks: {
                        color: C.dim,
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    grid: { color: C.gridLight },
                    border: { color: C.grid },
                },
                y: {
                    ticks: {
                        color: C.muted,
                        font: { family: "'Inter', sans-serif", size: 11 },
                    },
                    grid: { display: false },
                    border: { color: C.grid },
                },
            },
        },
    });
}

function renderRooflineChart(rooflineData, resultData) {
    if (!CHARTS_AVAILABLE) {
        renderChartUnavailable('roofline-chart');
        return;
    }
    const ctx = document.getElementById('roofline-chart').getContext('2d');
    if (rooflineChart) rooflineChart.destroy();

    const oi = rooflineData.operational_intensity;
    const peakFlops = rooflineData.peak_flops / 1e12;
    const peakBw = rooflineData.peak_bandwidth / 1e12;
    const achievable = rooflineData.achievable_flops / 1e12;

    const oiMin = Math.max(0.1, oi * 0.02);
    const oiMax = Math.max(oi * 40, peakFlops / peakBw * 3);
    const ridgePoints = [];
    const nPoints = 120;
    for (let i = 0; i <= nPoints; i++) {
        const x = oiMin * Math.pow(oiMax / oiMin, i / nPoints);
        const y = Math.min(peakFlops, x * peakBw);
        ridgePoints.push({ x, y });
    }

    rooflineChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Roofline ceiling',
                    data: ridgePoints,
                    showLine: true,
                    borderColor: C.dim,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    order: 1,
                },
                {
                    label: 'This operator',
                    data: [{ x: oi, y: achievable }],
                    backgroundColor: C.accent,
                    borderColor: C.accent,
                    borderWidth: 2,
                    pointRadius: 7,
                    pointHoverRadius: 11,
                    pointStyle: 'circle',
                    order: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: C.muted,
                        usePointStyle: true,
                        pointStyleWidth: 8,
                        padding: 16,
                        font: { family: "'Inter', sans-serif", size: 11 },
                    },
                },
                tooltip: {
                    callbacks: {
                        label(ctx) {
                            return ` (${ctx.parsed.x.toFixed(1)} FLOP/B, ${ctx.parsed.y.toFixed(1)} TFLOPS)`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Operational Intensity (FLOP / Byte)',
                        color: C.muted,
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    ticks: {
                        color: C.dim,
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    grid: { color: C.gridLight },
                    border: { color: C.grid },
                },
                y: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Performance (TFLOPS)',
                        color: C.muted,
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    ticks: {
                        color: C.dim,
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    grid: { color: C.gridLight },
                    border: { color: C.grid },
                },
            },
        },
    });
}
