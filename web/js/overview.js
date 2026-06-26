/**
 * Overview page — side-by-side hardware comparison table.
 */

// ── DOM refs ──────────────────────────────────────────────────────
const $overviewTable = document.getElementById('overview-table');
const $overviewCount = document.getElementById('overview-count');
const $overviewTableWrapper = document.getElementById('overview-table-wrapper');

// ── Cached data ───────────────────────────────────────────────────
let overviewData = [];

// ── Initialization ────────────────────────────────────────────────
async function initOverviewPage() {
    try {
        overviewData = await API.getHardwareOverview();
        renderOverviewTable();
    } catch (err) {
        console.error('Failed to load overview data:', err);
        if ($overviewTableWrapper) {
            $overviewTableWrapper.innerHTML =
                '<p style="color:var(--bottleneck);padding:var(--space-lg)">Failed to load hardware data.</p>';
        }
    }
}

// Auto-init if overview is the default active page on load
if (document.getElementById('page-overview')?.classList.contains('active')) {
    window._overviewPageInited = true;
    initOverviewPage();
}

// ── Formatting helpers ────────────────────────────────────────────
function fmtFlops(n) {
    if (!n) return '—';
    if (n >= 1e12) return (n / 1e12).toFixed(1) + ' T';
    if (n >= 1e9)  return (n / 1e9).toFixed(1) + ' G';
    return String(n);
}

function fmtGHz(mhz) { return mhz ? (mhz / 1000).toFixed(2) + ' GHz' : '—'; }
function fmtGBps(bw) { return bw ? bw.toFixed(0) + ' GB/s' : '—'; }
function fmtKB(n) { return n ? n + ' KB' : '—'; }
function fmtNum(n) { return n ? n.toLocaleString() : '—'; }

function fmtCombinedFlops(...values) {
    const present = values.filter(v => v);
    if (!present.length) return '—';
    const first = present[0];
    const allSame = present.every(v => v === first);
    if (allSame) return fmtFlops(first);
    return present.map(v => fmtFlops(v)).join(' / ');
}

function peakCudaFlops(hw) {
    const sms = hw.cu_count || 0;
    const coresPerSm = hw.fp32_cores_per_unit || 0;
    const clockHz = (hw.clock_mhz || 0) * 1e6;
    const derived = sms * coresPerSm * 2 * clockHz;
    return derived || (hw.peak_flops || {}).fp32 || 0;
}

/** Build a short memory summary string: "HBM3 80 GB @ 3352 GB/s" */
function memSummary(hw) {
    const tiers = hw.memory_tiers || [];
    if (tiers.length === 0) return '—';
    const t = tiers[0]; // first tier = main memory
    const cap = t.capacity_gb >= 1 ? t.capacity_gb.toFixed(0) + ' GB' : (t.capacity_gb * 1000).toFixed(0) + ' MB';
    return `${t.name} ${cap} @ ${t.bandwidth_gb_s.toFixed(0)} GB/s`;
}

/** L2 summary string */
function l2Summary(hw) {
    const tiers = hw.memory_tiers || [];
    if (tiers.length < 2) return '—';
    const t = tiers[1]; // second tier = L2
    const cap = t.capacity_gb >= 1 ? t.capacity_gb.toFixed(0) + ' GB' : (t.capacity_gb * 1000).toFixed(0) + ' MB';
    return `${cap}`;
}

// ── Rendering ─────────────────────────────────────────────────────
function renderOverviewTable() {
    if (!overviewData.length) return;
    $overviewCount.textContent = `${overviewData.length} targets`;

    // ── Column definitions ───────────────────────────────────────
    const cols = [
        // Group: Identity
        { grp: 'Identity',       key: 'name',        label: 'Name',        fmt: v => `<code>${v}</code>`, cls: 'col-name' },
        { grp: 'Identity',       key: 'architecture', label: 'Architecture', fmt: v => v || '—' },
        { grp: 'Identity',       key: 'sm_version',  label: 'SM Version',  fmt: v => v || '—', cls: 'col-smv' },

        // Group: Chip
        { grp: 'Compute Unit',   key: 'cu_count',    label: 'SMs / CUs',   fmt: v => fmtNum(v) },
        { grp: 'Compute Unit',   key: 'clock_mhz',   label: 'Clock',       fmt: v => fmtGHz(v) },

        // Group: Memory
        { grp: 'Memory',         key: 'memory_tiers', label: 'Main Memory', fmt: (_, hw) => memSummary(hw) },
        { grp: 'Memory',         key: 'memory_tiers', label: 'L2 Cache',    fmt: (_, hw) => l2Summary(hw) },
        { grp: 'Memory',         key: 'hbm_bandwidth_gb_s', label: 'Mem BW', fmt: v => fmtGBps(v) },

        // Group: Peak (Tensor Core dtypes)
        { grp: 'Peak TC',        key: 'peak_flops',  label: 'FP4',   fmt: (_, hw) => fmtFlops((hw.peak_flops || {}).fp4) },
        { grp: 'Peak TC',        key: 'peak_flops',  label: 'FP8',   fmt: (_, hw) => fmtFlops((hw.peak_flops || {}).fp8) },
        { grp: 'Peak TC',        key: 'peak_flops',  label: 'FP16/BF16', fmt: (_, hw) => fmtCombinedFlops((hw.peak_flops || {}).fp16, (hw.peak_flops || {}).bf16) },
        { grp: 'Peak TC',        key: 'peak_flops',  label: 'TF32 TC', fmt: (_, hw) => fmtFlops((hw.peak_flops || {}).tf32) },
        { grp: 'Peak TC',        key: 'peak_flops',  label: 'INT8 TC', fmt: (_, hw) => fmtFlops((hw.peak_flops || {}).int8) },

        // Group: Peak (CUDA core)
        { grp: 'Peak CUDA',      key: 'peak_flops',  label: 'FP16/BF16', fmt: (_, hw) => fmtFlops(peakCudaFlops(hw)) },
        { grp: 'Peak CUDA',      key: 'peak_flops',  label: 'FP32',  fmt: (_, hw) => fmtFlops((hw.peak_flops || {}).fp32) },
        { grp: 'Peak CUDA',      key: 'peak_flops',  label: 'FP64',  fmt: (_, hw) => fmtFlops((hw.peak_flops || {}).fp64) },

        // Group: Per-SM resources
        { grp: 'Per SM',         key: 'tensor_cores_per_unit', label: 'TC/SM', fmt: v => fmtNum(v) },
        { grp: 'Per SM',         key: 'fp32_cores_per_unit',   label: 'FP32/SM', fmt: v => fmtNum(v) },
        { grp: 'Per SM',         key: 'fp64_cores_per_unit',   label: 'FP64/SM', fmt: v => fmtNum(v) },
        { grp: 'Per SM',         key: 'int32_cores_per_unit',  label: 'INT32/SM', fmt: v => fmtNum(v) },
        { grp: 'Per SM',         key: 'register_file_kb',      label: 'Reg File/SM', fmt: v => fmtKB(v) },
        { grp: 'Per SM',         key: 'shared_memory_max_kb',  label: 'Shared Mem/SM', fmt: v => fmtKB(v) },

        // Group: Occupancy
        { grp: 'Occupancy',      key: 'max_threads_per_unit',  label: 'Max Threads/SM', fmt: v => fmtNum(v) },
        { grp: 'Occupancy',      key: 'max_concurrent_warps',  label: 'Max Warps/SM', fmt: v => fmtNum(v) },
        { grp: 'Occupancy',      key: 'max_registers_per_thread', label: 'Max Regs/Thread', fmt: v => fmtNum(v) },

        // Group: Features
        { grp: 'Features',       key: 'can_concurrent_fp32_int32', label: 'FP32+INT32 Concurrent', fmt: v => v ? '✓' : '—' },
        { grp: 'Features',       key: 'tensor_cores_per_unit', label: 'Has Tensor Cores', fmt: v => v > 0 ? '✓' : '—' },
    ];

    // ── Build thead with group header row + column label row ─────
    let groups = [];
    let seen = new Set();
    cols.forEach(c => {
        if (!seen.has(c.grp)) {
            seen.add(c.grp);
            groups.push({ name: c.grp, count: 0 });
        }
        groups[groups.length - 1].count++;
    });

    // Group header row
    let theadHTML = '<tr>';
    theadHTML += '<th class="hdr-grp">#</th>';
    groups.forEach(g => {
        theadHTML += `<th class="hdr-grp" colspan="${g.count}">${g.name}</th>`;
    });
    theadHTML += '</tr>';

    // Column label row
    theadHTML += '<tr>';
    theadHTML += '<th></th>'; // row number
    cols.forEach(c => {
        theadHTML += `<th class="${c.cls || ''}">${c.label}</th>`;
    });
    theadHTML += '</tr>';

    $overviewTable.querySelector('thead').innerHTML = theadHTML;

    // ── Build tbody ──────────────────────────────────────────────
    let tbodyHTML = '';
    overviewData.forEach((hw, idx) => {
        tbodyHTML += '<tr>';
        tbodyHTML += `<td class="row-num">${idx + 1}</td>`;
        cols.forEach(c => {
            const val = c.key === 'name' ? hw.name
                : c.key.includes('_') ? (hw[c.key] ?? '')
                : null;
            let cell;
            if (typeof c.fmt === 'function') {
                // Call formatter with the raw value and the full hw object
                cell = c.fmt(
                    c.key.includes('peak_flops') || c.key.includes('memory_tiers')
                        ? null : hw[c.key],
                    hw
                );
            } else {
                cell = hw[c.key] ?? '—';
            }
            tbodyHTML += `<td>${cell}</td>`;
        });
        tbodyHTML += '</tr>';
    });

    $overviewTable.querySelector('tbody').innerHTML = tbodyHTML;
}
