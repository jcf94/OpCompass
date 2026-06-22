/**
 * Chart rendering — time breakdown bar chart & roofline model.
 */
let breakdownChart = null;
let rooflineChart = null;

function renderBreakdownChart(data) {
    const ctx = document.getElementById("breakdown-chart").getContext("2d");
    if (breakdownChart) breakdownChart.destroy();

    const labels = ["Memory Read", "Compute", "Memory Write"];
    const values = [
        data.memory_read_time_us,
        data.compute_time_us,
        data.memory_write_time_us,
    ];
    const colors = ["#f87171", "#818cf8", "#34d399"];

    breakdownChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Time (µs)",
                data: values,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 0,
                borderRadius: 4,
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    title: { display: true, text: "Time (µs)", color: "#94a3b8" },
                    ticks: { color: "#94a3b8" },
                    grid: { color: "#334155" },
                },
                y: {
                    ticks: { color: "#94a3b8" },
                    grid: { color: "#334155" },
                },
            },
        },
    });
}

function renderRooflineChart(rooflineData, resultData) {
    const ctx = document.getElementById("roofline-chart").getContext("2d");
    if (rooflineChart) rooflineChart.destroy();

    const oi = rooflineData.operational_intensity;
    const peakFlops = rooflineData.peak_flops / 1e12;   // TFLOPS
    const peakBw = rooflineData.peak_bandwidth / 1e12;   // TB/s
    const achievable = rooflineData.achievable_flops / 1e12;

    // Build the roofline ridge
    const oiMin = Math.max(0.1, oi * 0.1);
    const oiMax = Math.max(oi * 10, peakFlops / peakBw * 2);
    const ridgePoints = [];
    const nPoints = 100;
    for (let i = 0; i <= nPoints; i++) {
        const x = oiMin * Math.pow(oiMax / oiMin, i / nPoints);
        const y = Math.min(peakFlops, x * peakBw);
        ridgePoints.push({ x, y });
    }

    rooflineChart = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Roofline",
                    data: ridgePoints,
                    showLine: true,
                    borderColor: "#94a3b8",
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    order: 1,
                },
                {
                    label: "This Operator",
                    data: [{ x: oi, y: achievable }],
                    backgroundColor: "#38bdf8",
                    borderColor: "#38bdf8",
                    pointRadius: 8,
                    pointStyle: "circle",
                    order: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: "#94a3b8" } },
            },
            scales: {
                x: {
                    type: "logarithmic",
                    title: { display: true, text: "Operational Intensity (FLOP/Byte)", color: "#94a3b8" },
                    ticks: { color: "#94a3b8" },
                    grid: { color: "#334155" },
                },
                y: {
                    type: "logarithmic",
                    title: { display: true, text: "Performance (TFLOPS)", color: "#94a3b8" },
                    ticks: { color: "#94a3b8" },
                    grid: { color: "#334155" },
                },
            },
        },
    });
}
