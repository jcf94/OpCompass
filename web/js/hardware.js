/**
 * Hardware page — specs, memory hierarchy, SM architecture,
 * and pipeline flow diagram (HTML/CSS, no SVG).
 */

// ── DOM refs ──────────────────────────────────────────────────────
const $hwPageSelect = document.getElementById('hw-page-select');
const $hwOverview = document.getElementById('hw-overview-content');
const $hwMemory = document.getElementById('hw-memory-content');
const $smArch = document.getElementById('sm-arch-content');
const $concurrent = document.getElementById('concurrent-content');
const $pipelineFlow = document.getElementById('pipeline-flow-container');
const $pipelineLegend = document.getElementById('pipeline-legend');

// ── Cached detail ─────────────────────────────────────────────────
let hwDetail = null;

function formatHardwareCapacity(bytes) {
    const value = Number(bytes || 0);
    if (!value) return '—';
    if (value >= 1024 ** 4) return `${(value / 1024 ** 4).toFixed(1)} TB`;
    if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(0)} GB`;
    if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(0)} MB`;
    if (value >= 1024) return `${(value / 1024).toFixed(0)} KB`;
    return `${value} B`;
}

// ── Initialization ────────────────────────────────────────────────
async function initHardwarePage() {
    try {
        const hwList = await API.getHardware();
        $hwPageSelect.innerHTML = hwList
            .map(hw => `<option value="${hw.name}">${hw.name.toUpperCase()} — ${hw.vendor}</option>`)
            .join('');
        if (hwList.find(h => h.name === 'a100')) $hwPageSelect.value = 'a100';
        await loadHardwareDetail();
    } catch (err) {
        console.error('Failed to init hardware page:', err);
    }
}

async function loadHardwareDetail() {
    const name = $hwPageSelect.value;
    if (!name) return;
    try {
        hwDetail = await API.getHardwareDetail(name);
        renderHardwarePage();
    } catch (err) {
        console.error('Failed to load hardware detail:', err);
    }
}

function renderHardwarePage() {
    if (!hwDetail) return;
    renderOverview();
    renderMemoryHierarchy();
    renderSMArchitecture();
    renderConcurrent();
    renderPipeline();
}

// ── Overview ──────────────────────────────────────────────────────
function renderOverview() {
    const cu = hwDetail.compute_unit;
    const totalTc = cu.count * cu.tensor_cores_per_unit;
    const totalFp32 = cu.count * cu.fp32_cores_per_unit;
    const totalInt32 = cu.count * cu.int32_cores_per_unit;
    const totalRegKb = cu.count * cu.register_file_kb;

    const peakRows = Object.entries(cu.peak_flops)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([dt, flops]) => {
            let d;
            if (flops >= 1e12) d = (flops / 1e12).toFixed(0) + ' TFLOPS';
            else if (flops >= 1e9) d = (flops / 1e9).toFixed(1) + ' GFLOPS';
            else d = String(flops);
            return `<div class="spec-row"><span class="spec-label">Peak ${dt.toUpperCase()}</span><span class="spec-value">${d}</span></div>`;
        }).join('');

    $hwOverview.innerHTML = `
        <div class="overview-header">
            <span class="chip-badge">${hwDetail.vendor} ${hwDetail.name.toUpperCase()}</span>
            ${hwDetail.sm_version ? `<span class="chip-sm-version">SM ${hwDetail.sm_version} · ${hwDetail.architecture || ''}</span>` : ''}
            <span class="chip-desc">${hwDetail.description}</span>
        </div>
        <div class="spec-row"><span class="spec-label">Compute Units</span><span class="spec-value">${cu.count} ${cu.name}s @ ${cu.clock_mhz} MHz</span></div>
        <div class="spec-row"><span class="spec-label">Total Tensor Cores</span><span class="spec-value">${totalTc}</span></div>
        <div class="spec-row"><span class="spec-label">Total FP32 Cores</span><span class="spec-value">${totalFp32.toLocaleString()}</span></div>
        <div class="spec-row"><span class="spec-label">Total INT32 Cores</span><span class="spec-value">${totalInt32.toLocaleString()}</span></div>
        <div class="spec-row"><span class="spec-label">Register File (total)</span><span class="spec-value">${(totalRegKb / 1024).toFixed(1)} MB (${totalRegKb.toLocaleString()} KB)</span></div>
        ${peakRows}
    `;
}

// ── Memory Hierarchy ──────────────────────────────────────────────
function renderMemoryHierarchy() {
    const tiers = hwDetail.memory_tiers || [];
    const cu = hwDetail.compute_unit;

    const allTiers = [
        ...tiers.map(t => ({
            name: t.name,
            cap: formatHardwareCapacity(t.capacity_bytes),
            bw: t.bandwidth_gb_s >= 1000
                ? `${(t.bandwidth_gb_s / 1000).toFixed(1)} TB/s`
                : `${t.bandwidth_gb_s.toFixed(0)} GB/s`,
            scope: 'Chip-wide',
            cls: t.name.toLowerCase().includes('hbm') ? 'dram' : 'l2',
        })),
        {
            name: 'L1 / Shared Memory',
            cap: `${cu.l1_shared_combined_kb} KB per SM`,
            bw: '~1.5 TB/s (est.)',
            scope: `Per ${cu.name}`,
            cls: 'l1',
        },
        {
            name: 'Register File',
            cap: `${cu.register_file_kb} KB per SM`,
            bw: '~8 TB/s (est.)',
            scope: `Per ${cu.name}`,
            cls: 'regfile',
        },
    ];

    $hwMemory.innerHTML = `
        <div class="memory-stack">
            ${allTiers.map((t, i) => `
                <div class="memory-tier memory-tier-${t.cls}">
                    <div class="tier-name">${t.name}</div>
                    <div class="tier-specs">
                        <span>${t.cap}</span>
                        <span class="tier-bw">${t.bw}</span>
                    </div>
                    <div class="tier-scope">${t.scope}</div>
                    <div class="tier-rank">${i === 0 ? 'Slowest' : i === allTiers.length - 1 ? 'Fastest' : ''}</div>
                </div>
            `).join('')}
        </div>
    `;
}

// ── SM Architecture ───────────────────────────────────────────────
function renderSMArchitecture() {
    const cu = hwDetail.compute_unit;
    const tensorCoreGen = {
        Volta: '1st Gen',
        Turing: '2nd Gen',
        Ampere: '3rd Gen',
        Hopper: '4th Gen',
        Blackwell: '5th Gen',
    }[hwDetail.architecture] || '';
    $smArch.innerHTML = `
        <div class="sm-grid">
            <div class="sm-item">
                <div class="sm-num">${cu.warp_schedulers_per_unit}</div>
                <div class="sm-label">Warp Schedulers</div>
            </div>
            <div class="sm-item">
                <div class="sm-num">${cu.tensor_cores_per_unit}</div>
                <div class="sm-label">Tensor Cores ${tensorCoreGen ? `<span class="sm-sub">${tensorCoreGen}</span>` : ''}</div>
            </div>
            <div class="sm-item">
                <div class="sm-num">${cu.fp32_cores_per_unit}</div>
                <div class="sm-label">FP32 Cores</div>
            </div>
            <div class="sm-item">
                <div class="sm-num">${cu.fp64_cores_per_unit}</div>
                <div class="sm-label">FP64 Cores</div>
            </div>
            <div class="sm-item">
                <div class="sm-num">${cu.int32_cores_per_unit}</div>
                <div class="sm-label">INT32 Cores</div>
            </div>
            <div class="sm-item">
                <div class="sm-num">${cu.ldst_units}</div>
                <div class="sm-label">Load/Store Units</div>
            </div>
        </div>
        <div class="occupancy-info">
            <h3>Occupancy Limits</h3>
            <div class="spec-row"><span>Max Threads / ${cu.name}</span><span>${cu.max_threads_per_unit} (${cu.max_concurrent_warps} warps × ${cu.threads_per_warp})</span></div>
            <div class="spec-row"><span>Max Thread Blocks / ${cu.name}</span><span>${cu.max_thread_blocks_per_unit}</span></div>
            <div class="spec-row"><span>Max Registers / Thread</span><span>${cu.max_registers_per_thread}</span></div>
            <div class="spec-row"><span>Max Registers / Block</span><span>${cu.max_registers_per_block.toLocaleString()}</span></div>
            <div class="spec-row"><span>Register File / ${cu.name}</span><span>${cu.register_file_kb} KB (${cu.register_file_kb * 1024 / 4} × 32-bit)</span></div>
            <div class="spec-row"><span>Shared Memory (max config)</span><span>${cu.shared_memory_max_kb} KB</span></div>
            <div class="spec-row"><span>L1 + Shared Memory (combined)</span><span>${cu.l1_shared_combined_kb} KB</span></div>
        </div>
    `;
}

// ── Concurrent Execution ──────────────────────────────────────────
function renderConcurrent() {
    const cu = hwDetail.compute_unit;
    const stageNames = (cu.pipeline || []).map(s => s.name);
    const hasAsyncCopy = stageNames.some(name => name.includes('async_copy'));
    const hasTensorCores = cu.tensor_cores_per_unit > 0;
    const hasAsyncBarrier = ['Ampere', 'Hopper', 'Blackwell'].includes(hwDetail.architecture);
    const hasWarpReduce = ['Ampere', 'Hopper', 'Blackwell'].includes(hwDetail.architecture);
    const overlaps = [
        {
            icon: '↻',
            title: 'FP32 + INT32 Simultaneous Issue',
            desc: cu.can_concurrent_fp32_int32
                ? `The ${cu.name} has dedicated FP32 (${cu.fp32_cores_per_unit}) and INT32 (${cu.int32_cores_per_unit}) datapaths that issue in the same clock cycle.`
                : 'Not supported on this architecture.',
            active: cu.can_concurrent_fp32_int32,
        },
        {
            icon: '⇅',
            title: 'Async Copy + Compute Overlap',
            desc: hasAsyncCopy
                ? 'The async copy engine moves data from global memory directly into shared memory. Compute proceeds in parallel during transfer.'
                : 'No dedicated async copy stage is modeled for this architecture.',
            active: hasAsyncCopy,
        },
        {
            icon: '∥',
            title: 'Multi-Warp Concurrency',
            desc: `${cu.warp_schedulers_per_unit} warp schedulers each dispatch 1 instruction/clock to different execution units. Up to ${cu.max_concurrent_warps} warps in flight simultaneously.`,
            active: true,
        },
        {
            icon: '⊓',
            title: 'Async Barriers',
            desc: hasAsyncBarrier
                ? 'Hardware-accelerated barrier objects separate arrive from wait, enabling efficient producer-consumer pipelines.'
                : 'Not modeled as a hardware feature for this architecture.',
            active: hasAsyncBarrier,
        },
        {
            icon: 'Σ',
            title: 'Warp-Level Reduction (1-step)',
            desc: hasWarpReduce
                ? 'Hardware-accelerated warp reductions complete in a single step for supported operations.'
                : 'Not modeled as a hardware feature for this architecture.',
            active: hasWarpReduce,
        },
        {
            icon: '⊗',
            title: 'Tensor Core MMA',
            desc: hasTensorCores
                ? `${cu.tensor_cores_per_unit} Tensor Cores per ${cu.name} are modeled for matrix operations.`
                : 'This architecture predates Tensor Cores.',
            active: hasTensorCores,
        },
    ];

    $concurrent.innerHTML = overlaps
        .map(o => `
            <div class="concurrent-item ${o.active ? '' : 'inactive'}">
                <div class="concurrent-icon">${o.icon}</div>
                <div class="concurrent-text">
                    <div class="concurrent-title">${o.title}</div>
                    <div class="concurrent-desc">${o.desc}</div>
                </div>
            </div>
        `).join('');
}

// ═══════════════════════════════════════════════════════════════════
// PIPELINE DIAGRAM — HTML/CSS flow (replaces SVG to fix overlap)
// ═══════════════════════════════════════════════════════════════════

/** Arrow SVG for connecting stages. */
function arrowSvg() {
    return `
    <svg viewBox="0 0 32 16" preserveAspectRatio="xMidYMid meet">
      <line x1="0" y1="8" x2="24" y2="8" />
      <polyline points="18,3 26,8 18,13" />
    </svg>`;
}

/** Dashed arrow SVG for async paths. */
function dashedArrowSvg() {
    return `
    <svg viewBox="0 0 32 16" preserveAspectRatio="xMidYMid meet">
      <line x1="0" y1="8" x2="24" y2="8" stroke-dasharray="4,3" />
      <polyline points="18,3 26,8 18,13" />
    </svg>`;
}

function renderPipeline() {
    const cu = hwDetail.compute_unit;
    const stages = cu.pipeline || [];

    // Build a lookup by name
    const s = {};
    stages.forEach(st => { s[st.name] = st; });

    function stageInfo(stage) {
        const name = stage.name;
        const desc = (stage.description || '').toLowerCase();
        if (name === 'async_copy_load') {
            if (desc.includes('tensor memory accelerator') || desc.includes('tma')) {
                return { type: 'memory', cls: 'tma-load', label: 'Load-TMA', unit: 'B/clk/SM' };
            }
            return { type: 'memory', cls: 'async', label: 'Async Copy', unit: 'B/clk/SM' };
        }
        if (name === 'async_copy_store') {
            const label = desc.includes('dedicated') ? 'Store-TMA' : 'TMA Store';
            return { type: 'memory', cls: 'tma-store', label, unit: 'B/clk/SM' };
        }
        if (name === 'tmem_load') {
            return { type: 'memory', cls: 'tmem', label: 'TMEM', unit: 'B/clk/SM' };
        }
        if (/mma/.test(name)) {
            return { type: 'compute', cls: 'compute', label: 'Tensor Core', unit: 'FMA/clk/SM' };
        }
        if (/alu|fma/.test(name)) {
            return { type: 'compute', cls: 'compute', label: 'CUDA Core', unit: 'FMA/clk/SM' };
        }
        if (/shared/.test(name)) {
            return { type: 'memory', cls: 'shared', label: 'Shared Mem', unit: 'B/clk/SM' };
        }
        if (/read|load|write|store|copy/.test(name)) {
            return { type: 'memory', cls: 'memory', label: 'Memory', unit: 'B/clk/SM' };
        }
        return { type: 'memory', cls: 'memory', label: 'Memory', unit: 'B/clk/SM' };
    }

    function stageType(name) {
        return stageInfo({ name, description: s[name]?.description || '' }).type;
    }

    function stageTypeLabel(info) {
        return info.type === 'compute' ? 'comp' : info.cls;
    }

    function stageBadgeHtml(info) {
        return `<span class="stage-type-badge ${stageTypeLabel(info)}">${info.label}</span>`;
    }

    function stageCardHtml(name) {
        const st = s[name];
        if (!st) return '';
        const info = stageInfo(st);
        const title = st.description ? ` title="${st.description}"` : '';
        return `
        <div class="pipeline-stage ${info.type} ${info.cls}"${title}>
          <div class="stage-name">${st.name}</div>
          ${stageBadgeHtml(info)}
          <div class="stage-latency">${st.latency_cycles} cyc latency</div>
          <div class="stage-throughput">${st.throughput_per_cycle} ${info.unit}</div>
        </div>`;
    }

    function featureChips() {
        const names = new Set(stages.map(st => st.name));
        const chips = [];
        if (names.has('async_copy_load')) {
            const load = s.async_copy_load;
            const desc = (load.description || '').toLowerCase();
            chips.push(desc.includes('tma') ? 'TMA async load' : 'Async copy load');
        }
        if (names.has('async_copy_store')) {
            const store = s.async_copy_store;
            const desc = (store.description || '').toLowerCase();
            chips.push(desc.includes('dedicated') ? 'Dedicated Store-TMA' : 'TMA store');
        }
        if (names.has('tmem_load')) chips.push('TMEM accumulator readout');
        if (stages.some(st => /wgmma|4th-gen tensor/i.test(st.description || ''))) chips.push('WGMMA-style Tensor Core');
        if (stages.some(st => /5th-gen tensor|umma/i.test(st.description || ''))) chips.push('UMMA-style Tensor Core');
        return chips.map(chip => `<span class="pipeline-feature-chip">${chip}</span>`).join('');
    }

    let html = '';

    // ── Modeled hardware stage order ───────────────────────────
    const chips = featureChips();
    if (chips) {
        html += `<div class="pipeline-feature-row">${chips}</div>`;
    }
    html += '<div class="pipeline-flow">';
    stages.forEach((st, i) => {
        html += stageCardHtml(st.name);
        if (i < stages.length - 1) html += `<div class="pipeline-arrow">${arrowSvg()}</div>`;
    });
    html += '</div>'; // end .pipeline-flow

    if (stages.some(st => st.name === 'async_copy_load' || st.name === 'async_copy_store' || st.name === 'tmem_load')) {
        html += `
        <div class="async-callout">
          <div class="async-callout-title">Architecture-specific pipeline features</div>
          <p>Async copy/TMA stages feed shared memory ahead of MMA according to the selected software stage depth. Hopper adds TMA load/store paths; Blackwell adds TMEM accumulator readout and a dedicated Store-TMA stage.</p>
        </div>`;
    }

    // ── Interconnect info ──────────────────────────────────────
    const stageNames = stages.map(s => s.name);
    html += `
    <div style="margin-top:var(--space);padding-top:var(--space);border-top:1px solid var(--border-light);font-size:var(--text-xs);color:var(--text-muted);font-family:var(--font-mono)">
      ${cu.warp_schedulers_per_unit} warp schedulers · ${cu.max_concurrent_warps} warps / ${cu.max_threads_per_unit} threads max &nbsp;|&nbsp;
      ${stageNames.length} modeled pipeline stages
    </div>`;

    $pipelineFlow.innerHTML = html;

    // ── Legend table ───────────────────────────────────────────
    if (stages.length > 0) {
        $pipelineLegend.innerHTML = `
            <h3>Pipeline Stage Details</h3>
            <table class="pipeline-table">
                <thead>
                    <tr>
                        <th>Stage</th>
                        <th>Type</th>
                        <th>Latency (cycles)</th>
                        <th>Throughput / SM / Clock</th>
                        <th>Overlaps With</th>
                    </tr>
                </thead>
                <tbody>
                    ${stages.map(st => {
                        const type = stageType(st.name);
                        const info = stageInfo(st);
                        const unit = type === 'memory' ? 'Bytes' : 'FMA ops';
                        const overlaps = getOverlaps(st.name);
                        return `
                            <tr>
                                <td><code>${st.name}</code></td>
                                <td><span class="stage-badge ${stageTypeLabel(info)}">${info.label}</span></td>
                                <td>${st.latency_cycles}</td>
                                <td>${st.throughput_per_cycle} ${unit}</td>
                                <td class="overlap-cell">${overlaps}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        `;
    }
}

/** Return a human-readable overlap description for a stage. */
function getOverlaps(name) {
    const map = {
        'global_read':     ['mma', 'fma_alu'],
        'async_copy_load': ['mma', 'fma_alu', 'shared_load of ready tiles'],
        'shared_load':     ['mma (other warps)', 'fma_alu (other warps)'],
        'mma':             ['global_read', 'async_copy_load', 'shared_load/store', 'fma_alu (INT32)'],
        'fma_alu':         ['INT32 ops (same clock)', 'global_read', 'async_copy'],
        'tmem_load':       ['late epilogue work', 'Store-TMA drain'],
        'shared_store':    ['mma (other warps)', 'fma_alu (other warps)'],
        'async_copy_store':['late compute / next CTA in persistent kernels'],
        'global_write':    ['mma (other warps)', 'fma_alu (other warps)'],
    };
    return (map[name] || []).join(' · ') || '—';
}

// ── Event listeners ───────────────────────────────────────────────
$hwPageSelect.addEventListener('change', loadHardwareDetail);
