/**
 * Pipeline analysis visualization — Gantt chart, stats, and stage breakdown.
 *
 * The Gantt chart supports:
 * - Iteration range selection (toolbar inputs + presets)
 * - Mouse wheel zoom (centered on cursor)
 * - Drag to pan
 * - "Reset Zoom" to fit the visible iteration range
 *
 * View state lives in `viewState` and is preserved across re-renders from
 * the cached `currentResult` in app.js. No API refetch is needed for
 * view-only changes.
 */

const PipelineUI = {
    // ── Color palette ──────────────────────────────────────────────
    colors: {
        memory: "#7c3aed",      // purple — read/load/write/store stages
        compute: "#1a56db",     // blue — mma/alu/fma stages
        async:  "#059669",      // green — async copy stages
        prologueBg: "#f0fdf4",  // light green
        steadyBg:   "#eff6ff",  // light blue
        epilogueBg: "#fef3c7",  // light amber
        gridLine:   "#e5e7eb",
        axisLabel:  "#6f6f6f",
        phaseLabel: "#1b1b1b",
    },

    // ── View state (preserved across re-renders) ──────────────────
    // iterStart/iterEnd: which K-iterations to render (epilogue always shown)
    // zoomStart/zoomEnd: visible cycle range; null means auto-fit to visible ops
    viewState: {
        iterStart: 0,
        iterEnd: 8,
        zoomStart: 0,
        zoomEnd: null,
        isDragging: false,
        dragStartX: 0,
        dragStartZoomStart: 0,
        dragStartZoomEnd: 0,
    },

    // Cached schedule from last render, used by zoom/pan handlers without
    // requiring app.js to pass it back in.
    _lastSchedule: null,
    _lastContainerWidth: 0,
    _toolbarBound: false,

    // ── Stage type classification ──────────────────────────────────
    stageType(name) {
        const n = name.toLowerCase();
        if (n.includes("mma") || n.includes("alu") || n.includes("fma")) return "compute";
        if (n.includes("async_copy") || n.includes("cp.async")) return "async";
        return "memory";  // read, load, write, store
    },

    stageColor(name) {
        const type = this.stageType(name);
        return this.colors[type];
    },

    // ── Gantt Chart Renderer ───────────────────────────────────────
    renderGanttChart(schedule) {
        const canvas = document.getElementById("gantt-chart");
        if (!canvas || !schedule || !schedule.sub_ops) return;

        const subOps = schedule.sub_ops;
        if (subOps.length === 0) return;

        // Cache for zoom/pan handlers and resize listener
        this._lastSchedule = schedule;

        // Clamp iteration range to the actual K-iteration count
        const numK = schedule.num_k_iterations;
        if (this.viewState.iterStart >= numK) this.viewState.iterStart = 0;
        if (this.viewState.iterEnd > numK) this.viewState.iterEnd = numK;
        if (this.viewState.iterStart >= this.viewState.iterEnd) {
            this.viewState.iterStart = 0;
            this.viewState.iterEnd = Math.min(8, numK);
        }

        // Sync toolbar inputs with current state (in case state changed
        // programmatically, e.g. preset buttons).
        this._syncToolbarInputs();

        // Bind toolbar events once
        if (!this._toolbarBound) {
            this._bindToolbar();
            this._toolbarBound = true;
        }

        // Collect unique pipeline stages (y-axis rows)
        const stageSet = new Set();
        subOps.forEach(op => stageSet.add(op.pipeline_stage));
        const stages = Array.from(stageSet).sort((a, b) => {
            // Sort: memory read stages first, then async, then compute, then write
            const order = { memory: 0, async: 1, compute: 2 };
            return (order[this.stageType(a)] ?? 1) - (order[this.stageType(b)] ?? 1);
        });

        // Filter to visible iterations.
        // The epilogue (iteration === -1) sits at the end of the schedule
        // (cycles ~steadyEnd..total). Only show it when the visible range
        // includes the last K-iteration, otherwise it would float far away
        // from early iterations and create a whitespace gap.
        const includesLastIter = this.viewState.iterEnd >= numK;
        const visibleOps = subOps.filter(op =>
            (op.iteration >= this.viewState.iterStart && op.iteration < this.viewState.iterEnd)
            || (includesLastIter && op.iteration === -1)
        );
        if (visibleOps.length === 0) return;

        // Visible cycle range (for auto-fit when zoom is null)
        const visibleMinCycle = Math.min(...visibleOps.map(op => op.start_cycle));
        const visibleMaxCycle = Math.max(...visibleOps.map(op => op.end_cycle));

        // Resolve zoom range: if null, fit to visible ops
        const zStart = this.viewState.zoomStart ?? visibleMinCycle;
        const zEnd = this.viewState.zoomEnd ?? visibleMaxCycle;
        const zSpan = Math.max(1, zEnd - zStart);

        // Phase boundaries (in cycles, for background shading)
        const prologueEnd = schedule.prologue_cycles;
        const steadyEnd = prologueEnd + schedule.per_iteration_cycles * (schedule.num_k_iterations - 1);

        // Canvas sizing
        const container = document.getElementById("gantt-chart-container");
        const dpr = window.devicePixelRatio || 1;
        const containerWidth = container.clientWidth || 900;
        this._lastContainerWidth = containerWidth;
        const leftMargin = 140;
        const rightMargin = 20;
        const topMargin = 40;
        const bottomMargin = 30;
        const rowHeight = 36;
        const rowGap = 4;
        const chartWidth = containerWidth - leftMargin - rightMargin;
        const chartHeight = stages.length * (rowHeight + rowGap);
        const totalHeight = topMargin + chartHeight + bottomMargin;

        canvas.width = containerWidth * dpr;
        canvas.height = totalHeight * dpr;
        canvas.style.width = containerWidth + "px";
        canvas.style.height = totalHeight + "px";

        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);

        // Scale: cycles -> pixels, mapping [zStart, zEnd] to chart width
        const cycleScale = (cycle) => {
            return leftMargin + ((cycle - zStart) / zSpan) * chartWidth;
        };

        // ── Background ──
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, containerWidth, totalHeight);

        // ── Phase regions (clipped to visible range) ──
        // Only draw a phase band if it intersects [zStart, zEnd].
        const drawPhaseBand = (start, end, color) => {
            const x1 = cycleScale(Math.max(start, zStart));
            const x2 = cycleScale(Math.min(end, zEnd));
            if (x2 <= x1) return;
            ctx.fillStyle = color;
            ctx.fillRect(x1, topMargin, x2 - x1, chartHeight);
        };
        drawPhaseBand(0, prologueEnd, this.colors.prologueBg);
        drawPhaseBand(prologueEnd, steadyEnd, this.colors.steadyBg);
        drawPhaseBand(steadyEnd, visibleMaxCycle + 1, this.colors.epilogueBg);

        // ── Phase labels (only if the phase is visible) ──
        ctx.font = "11px Inter, sans-serif";
        ctx.fillStyle = this.colors.phaseLabel;
        ctx.textAlign = "center";
        const labelY = topMargin - 10;

        if (zStart < prologueEnd) {
            const mid = (Math.max(zStart, 0) + Math.min(zEnd, prologueEnd)) / 2;
            ctx.fillText("Prologue", cycleScale(mid), labelY);
        }
        if (zStart < steadyEnd && zEnd > prologueEnd) {
            const mid = (Math.max(zStart, prologueEnd) + Math.min(zEnd, steadyEnd)) / 2;
            ctx.fillText("Steady State", cycleScale(mid), labelY);
        }
        if (zEnd > steadyEnd) {
            const mid = (Math.max(zStart, steadyEnd) + Math.min(zEnd, visibleMaxCycle)) / 2;
            ctx.fillText("Epilogue", cycleScale(mid), labelY);
        }

        // ── Grid lines ──
        ctx.strokeStyle = this.colors.gridLine;
        ctx.lineWidth = 0.5;

        // Vertical grid: choose a "nice" step that yields ~5-10 lines in view
        const targetLines = 6;
        const rawStep = zSpan / targetLines;
        const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
        const step = (rawStep / magnitude < 2 ? 1 : rawStep / magnitude < 5 ? 2 : 5) * magnitude;

        const gridCycles = new Set([zStart, zEnd]);
        const firstGrid = Math.ceil(zStart / step) * step;
        for (let c = firstGrid; c < zEnd; c += step) gridCycles.add(c);

        for (const cycle of gridCycles) {
            const x = cycleScale(cycle);
            ctx.beginPath();
            ctx.moveTo(x, topMargin);
            ctx.lineTo(x, topMargin + chartHeight);
            ctx.stroke();
        }

        // ── Stage labels (y-axis) ──
        ctx.font = "11px JetBrains Mono, monospace";
        ctx.textAlign = "right";
        ctx.fillStyle = this.colors.axisLabel;

        for (let i = 0; i < stages.length; i++) {
            const y = topMargin + i * (rowHeight + rowGap) + rowHeight / 2 + 4;
            ctx.fillText(stages[i], leftMargin - 8, y);
        }

        // ── Sub-op blocks ──
        for (const op of visibleOps) {
            const stageIdx = stages.indexOf(op.pipeline_stage);
            if (stageIdx === -1) continue;

            const x1 = cycleScale(op.start_cycle);
            const x2 = cycleScale(op.end_cycle);
            // Skip blocks fully outside the view
            if (x2 < leftMargin || x1 > leftMargin + chartWidth) continue;

            const y = topMargin + stageIdx * (rowHeight + rowGap) + 2;
            const w = Math.max(2, x2 - x1);  // minimum 2px width
            const h = rowHeight - 4;

            const color = this.stageColor(op.pipeline_stage);

            // Block fill
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.75;
            ctx.beginPath();
            ctx.roundRect(x1, y, w, h, 3);
            ctx.fill();
            ctx.globalAlpha = 1.0;

            // Block border
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(x1, y, w, h, 3);
            ctx.stroke();

            // Label inside block (if wide enough)
            if (w > 50) {
                ctx.font = "9px JetBrains Mono, monospace";
                ctx.fillStyle = "#ffffff";
                ctx.textAlign = "center";
                ctx.fillText(op.name, x1 + w / 2, y + h / 2 + 3);
            }
        }

        // ── Cycle axis (bottom) ──
        ctx.font = "10px JetBrains Mono, monospace";
        ctx.fillStyle = this.colors.axisLabel;
        ctx.textAlign = "center";

        for (const cycle of gridCycles) {
            const x = cycleScale(cycle);
            const label = cycle >= 1000 ? (cycle / 1000).toFixed(cycle < 10000 ? 1 : 0) + "k" : String(Math.round(cycle));
            ctx.fillText(label, x, topMargin + chartHeight + 15);
        }

        // ── X-axis title ──
        ctx.font = "11px Inter, sans-serif";
        ctx.fillText("Cycles", leftMargin + chartWidth / 2, totalHeight - 5);

        // ── Mouse interaction (zoom + pan) ──
        // Bind once; handlers read live state from this.viewState and
        // this._lastSchedule, so they don't need closure variables.
        this._bindMouseInteraction(canvas);
    },

    // ── Toolbar binding ─────────────────────────────────────────────
    _syncToolbarInputs() {
        const startInput = document.getElementById("iter-start");
        const endInput = document.getElementById("iter-end");
        if (startInput) startInput.value = this.viewState.iterStart;
        if (endInput) endInput.value = this.viewState.iterEnd;
    },

    _bindToolbar() {
        const startInput = document.getElementById("iter-start");
        const endInput = document.getElementById("iter-end");
        const resetBtn = document.getElementById("zoom-reset");

        if (startInput) {
            startInput.addEventListener("change", () => {
                const v = parseInt(startInput.value, 10);
                if (!isNaN(v) && v >= 0 && v < this.viewState.iterEnd) {
                    this.viewState.iterStart = v;
                    this._resetZoom();
                    this._redrawFromCache();
                } else {
                    startInput.value = this.viewState.iterStart;
                }
            });
        }
        if (endInput) {
            endInput.addEventListener("change", () => {
                const v = parseInt(endInput.value, 10);
                const numK = this._lastSchedule ? this._lastSchedule.num_k_iterations : 8;
                if (!isNaN(v) && v > this.viewState.iterStart && v <= numK) {
                    this.viewState.iterEnd = v;
                    this._resetZoom();
                    this._redrawFromCache();
                } else {
                    endInput.value = this.viewState.iterEnd;
                }
            });
        }

        // Preset buttons
        document.querySelectorAll(".btn-mini[data-preset]").forEach(btn => {
            btn.addEventListener("click", () => {
                const preset = btn.getAttribute("data-preset");
                const numK = this._lastSchedule ? this._lastSchedule.num_k_iterations : 8;
                if (preset === "first8") {
                    this.viewState.iterStart = 0;
                    this.viewState.iterEnd = Math.min(8, numK);
                } else if (preset === "last8") {
                    this.viewState.iterEnd = numK;
                    this.viewState.iterStart = Math.max(0, numK - 8);
                } else if (preset === "all") {
                    this.viewState.iterStart = 0;
                    this.viewState.iterEnd = numK;
                }
                this._resetZoom();
                this._syncToolbarInputs();
                this._redrawFromCache();
            });
        });

        if (resetBtn) {
            resetBtn.addEventListener("click", () => {
                this._resetZoom();
                this._redrawFromCache();
            });
        }
    },

    // ── Layout constants (recomputed by handlers, not cached) ──────
    _getLayout() {
        const container = document.getElementById("gantt-chart-container");
        const containerWidth = (container && container.clientWidth) || 900;
        return {
            leftMargin: 140,
            rightMargin: 20,
            containerWidth,
            chartWidth: containerWidth - 140 - 20,
        };
    },

    // ── Visible ops for current viewState ───────────────────────────
    // Epilogue (iteration === -1) is only included when the visible range
    // reaches the last K-iteration — otherwise it would float far away in
    // cycle space and create a whitespace gap. Must match the filter logic
    // used in renderGanttChart.
    _getVisibleOps() {
        if (!this._lastSchedule) return [];
        const numK = this._lastSchedule.num_k_iterations;
        const includesLastIter = this.viewState.iterEnd >= numK;
        return this._lastSchedule.sub_ops.filter(op =>
            (op.iteration >= this.viewState.iterStart && op.iteration < this.viewState.iterEnd)
            || (includesLastIter && op.iteration === -1)
        );
    },

    // ── Mouse interaction (wheel zoom + drag pan), bound once ───────
    _bindMouseInteraction(canvas) {
        if (canvas._opcompassBound) return;
        canvas._opcompassBound = true;

        // Convert mouse x (pixels) to cycle, using current viewState
        const getMouseCycle = (e) => {
            const ops = this._getVisibleOps();
            if (ops.length === 0) return 0;
            const vMin = Math.min(...ops.map(op => op.start_cycle));
            const vMax = Math.max(...ops.map(op => op.end_cycle));
            const zStart = this.viewState.zoomStart ?? vMin;
            const zEnd = this.viewState.zoomEnd ?? vMax;
            const zSpan = Math.max(1, zEnd - zStart);

            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const { leftMargin, chartWidth } = this._getLayout();
            return zStart + ((x - leftMargin) / chartWidth) * zSpan;
        };

        // ── Wheel zoom ──
        canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            if (!this._lastSchedule) return;

            const ops = this._getVisibleOps();
            if (ops.length === 0) return;
            const vMin = Math.min(...ops.map(op => op.start_cycle));
            const vMax = Math.max(...ops.map(op => op.end_cycle));
            let curStart = this.viewState.zoomStart ?? vMin;
            let curEnd = this.viewState.zoomEnd ?? vMax;

            const mouseCycle = getMouseCycle(e);
            const factor = e.deltaY < 0 ? 0.8 : 1.25;  // zoom in / out
            const newSpan = (curEnd - curStart) * factor;
            // Keep mouseCycle at the same relative position
            const ratio = (mouseCycle - curStart) / (curEnd - curStart);
            let newStart = mouseCycle - ratio * newSpan;
            let newEnd = newStart + newSpan;

            // Clamp to visible range
            const minSpan = Math.max(1, (vMax - vMin) * 0.02);  // min 2% zoom
            if (newEnd - newStart < minSpan) {
                const mid = (newStart + newEnd) / 2;
                newStart = mid - minSpan / 2;
                newEnd = mid + minSpan / 2;
            }
            if (newStart < vMin) { newEnd += (vMin - newStart); newStart = vMin; }
            if (newEnd > vMax) { newStart -= (newEnd - vMax); newEnd = vMax; }
            if (newStart < vMin) newStart = vMin;

            this.viewState.zoomStart = newStart;
            this.viewState.zoomEnd = newEnd;
            this._redrawFromCache();
        }, { passive: false });

        // ── Drag pan ──
        canvas.addEventListener("mousedown", (e) => {
            if (!this._lastSchedule) return;
            const ops = this._getVisibleOps();
            if (ops.length === 0) return;
            this.viewState.isDragging = true;
            this.viewState.dragStartX = e.clientX;
            // Capture the baseline range at drag start (resolve nulls)
            const vMin = Math.min(...ops.map(op => op.start_cycle));
            const vMax = Math.max(...ops.map(op => op.end_cycle));
            this.viewState.dragStartZoomStart = this.viewState.zoomStart ?? vMin;
            this.viewState.dragStartZoomEnd = this.viewState.zoomEnd ?? vMax;
            canvas.classList.add("dragging");
        });

        canvas.addEventListener("mousemove", (e) => {
            if (!this.viewState.isDragging || !this._lastSchedule) return;

            const ops = this._getVisibleOps();
            if (ops.length === 0) return;
            const vMin = Math.min(...ops.map(op => op.start_cycle));
            const vMax = Math.max(...ops.map(op => op.end_cycle));

            const startZ = this.viewState.dragStartZoomStart;
            const endZ = this.viewState.dragStartZoomEnd;
            const zSpan = endZ - startZ;

            // Pixel delta → cycle delta (inverted: drag right = pan left)
            const dXPx = e.clientX - this.viewState.dragStartX;
            const { chartWidth } = this._getLayout();
            const dCycle = -(dXPx / chartWidth) * zSpan;

            let newStart = startZ + dCycle;
            let newEnd = endZ + dCycle;

            // Clamp: don't pan beyond visible range
            if (newStart < vMin) { newEnd += (vMin - newStart); newStart = vMin; }
            if (newEnd > vMax) { newStart -= (newEnd - vMax); newEnd = vMax; }
            if (newStart < vMin) newStart = vMin;

            this.viewState.zoomStart = newStart;
            this.viewState.zoomEnd = newEnd;
            this._redrawFromCache();
        });

        const endDrag = () => {
            this.viewState.isDragging = false;
            canvas.classList.remove("dragging");
        };
        canvas.addEventListener("mouseup", endDrag);
        canvas.addEventListener("mouseleave", endDrag);
    },

    _resetZoom() {
        this.viewState.zoomStart = 0;
        this.viewState.zoomEnd = null;
    },

    _redrawFromCache() {
        if (this._lastSchedule) {
            this.renderGanttChart(this._lastSchedule);
        }
    },

    // ── Pipeline Stats ─────────────────────────────────────────────
    formatBytes(bytes) {
        if (bytes == null || Number.isNaN(Number(bytes))) return "—";
        const units = ["B", "KB", "MB", "GB", "TB"];
        let value = Number(bytes);
        let unit = 0;
        while (value >= 1024 && unit < units.length - 1) {
            value /= 1024;
            unit += 1;
        }
        return `${value.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
    },

    renderPipelineStats(schedule, pipelineConfig, memoryBreakdown) {
        const container = document.getElementById("pipeline-stats-content");
        if (!container || !schedule) return;

        const rows = [
            ["K Iterations", schedule.num_k_iterations],
            ["Grid Size", schedule.grid_size],
            ["Wave Count", schedule.wave_count],
            ["Prologue", schedule.prologue_cycles + " cycles"],
            ["Per Iteration", schedule.per_iteration_cycles + " cycles"],
            ["Epilogue", schedule.epilogue_cycles + " cycles"],
            ["Total Block", schedule.total_cycles_per_block + " cycles"],
            ["Bottleneck", schedule.bottleneck_stage],
        ];
        if (memoryBreakdown) {
            rows.push(
                ["Effective HBM Read", this.formatBytes(memoryBreakdown.effective_hbm_read_bytes)],
                ["CTA Logical Read", this.formatBytes(memoryBreakdown.logical_cta_read_bytes)],
                ["Unique Tensor Read", this.formatBytes(memoryBreakdown.unique_tensor_read_bytes)],
                ["L2 Reuse", `${Number(memoryBreakdown.l2_reuse_factor || 1).toFixed(2)}x`],
            );
        }

        container.innerHTML = rows.map(([label, value]) =>
            `<div class="spec-row"><span class="spec-label">${label}</span><span class="spec-value mono">${value}</span></div>`
        ).join("");
    },

    // ── Tiling Info ────────────────────────────────────────────────
    renderTilingInfo(tilingInfo) {
        const container = document.getElementById("tiling-info-content");
        if (!container || !tilingInfo) return;

        // Sync tile input fields with actual values from analysis
        const blockM = document.getElementById("block-m-input");
        const blockN = document.getElementById("block-n-input");
        const blockK = document.getElementById("block-k-input");
        if (blockM) { blockM.value = tilingInfo.block_m; blockM.placeholder = tilingInfo.block_m; }
        if (blockN) { blockN.value = tilingInfo.block_n; blockN.placeholder = tilingInfo.block_n; }
        if (blockK) { blockK.value = tilingInfo.block_k; blockK.placeholder = tilingInfo.block_k; }

        const rows = [
            ["Warps/Block", tilingInfo.num_warps_per_block],
            ["Shared Mem", tilingInfo.shared_memory_per_block != null
                ? (tilingInfo.shared_memory_per_block / 1024).toFixed(1) + " KB"
                : "—"],
        ];

        container.innerHTML = rows.map(([label, value]) =>
            `<div class="spec-row"><span class="spec-label">${label}</span><span class="spec-value mono">${value}</span></div>`
        ).join("");
    },

    // ── Performance Guidance ───────────────────────────────────────
    renderRecommendations(result) {
        let container = document.getElementById("pipeline-recommendations-content");
        if (!container) {
            const pipelineResults = document.getElementById("pipeline-results");
            if (!pipelineResults) return;
            const card = document.createElement("div");
            card.className = "card";
            card.id = "pipeline-recommendations-card";
            card.innerHTML = `
                <h2>Performance Guidance</h2>
                <div id="pipeline-recommendations-content"></div>
            `;
            const ganttCard = document.getElementById("gantt-chart-container")?.closest(".card");
            pipelineResults.insertBefore(card, ganttCard || null);
            container = document.getElementById("pipeline-recommendations-content");
        }
        if (!container || !result.pipeline_schedule || !result.tiling_info) return;

        const schedule = result.pipeline_schedule;
        const tiling = result.tiling_info;
        const memory = result.pipeline_memory_breakdown || {};
        const bottleneck = result.bottleneck || schedule.bottleneck_stage || "";
        const tips = [];

        if (bottleneck.includes("async_copy") || bottleneck.includes("global_read")) {
            tips.push({
                title: "Memory load is limiting throughput",
                text: `HBM guidance is based on effective HBM traffic after L2 reuse. Try increasing Block K from ${tiling.block_k} to improve arithmetic work per global-memory load. If shared memory becomes too high, reduce Block M or Block N first.`,
            });
            if (memory.l2_reuse_factor && memory.l2_reuse_factor > 1.2) {
                tips.push({
                    title: "L2 reuse is reducing HBM pressure",
                    text: `CTA logical reads are about ${Number(memory.l2_reuse_factor).toFixed(2)}x higher than effective HBM reads. Changing Block M/N can alter this reuse pattern.`,
                });
            }
            tips.push({
                title: "Keep async copy enabled",
                text: "Async copy overlaps global-memory movement with compute; disabling it will usually make a memory-bound schedule slower.",
            });
        } else if (bottleneck.includes("shared_load")) {
            tips.push({
                title: "Shared-memory feed is the bottleneck",
                text: `Try smaller Block K or a more balanced M/N tile. Current tile is ${tiling.block_m}x${tiling.block_n}x${tiling.block_k}.`,
            });
        } else if (bottleneck.includes("mma") || bottleneck.includes("fma_alu")) {
            tips.push({
                title: "Compute is limiting throughput",
                text: "Larger Block M/N can improve reuse and occupancy tradeoffs, but if wave count rises or shared memory pressure grows, step back to the previous tile.",
            });
            if (result.pipeline_config && !result.pipeline_config.sparsity_2_4_enabled) {
                tips.push({
                    title: "Check sparse eligibility",
                    text: "If weights are compatible with 2:4 sparsity, enabling it can reduce MMA cycles in the pipeline model.",
                });
            }
        } else if (bottleneck.includes("store") || bottleneck.includes("write")) {
            tips.push({
                title: "Epilogue/writeback is visible",
                text: "Try increasing Block K so more compute is done per output tile write, or reduce Block M/N if the output tile is too large.",
            });
        }

        if (schedule.wave_count > 1) {
            tips.push({
                title: "There are multiple CTA waves",
                text: `Wave count is ${schedule.wave_count}. Smaller Block M/N can increase grid parallelism balance, while larger tiles may reduce wave count but increase per-block time.`,
            });
        }

        if (tiling.shared_memory_per_block > 0) {
            tips.push({
                title: "Shared memory budget",
                text: `Current tile uses ${(tiling.shared_memory_per_block / 1024).toFixed(1)} KB per block. Leave headroom for occupancy when increasing Block K.`,
            });
        }

        if (tips.length === 0) {
            tips.push({
                title: "No single dominant adjustment",
                text: "Try one tile dimension at a time and compare SOL time, bottleneck stage, wave count, and shared memory per block.",
            });
        }

        container.innerHTML = tips.map(t => `
            <div class="recommendation-item">
                <div class="recommendation-title">${t.title}</div>
                <div class="recommendation-text">${t.text}</div>
            </div>
        `).join("");
    },

    // ── Stage Breakdown Table ──────────────────────────────────────
    renderStageBreakdown(stageBreakdown, schedule) {
        const tbody = document.querySelector("#pipeline-stage-table tbody");
        if (!tbody) return;

        const rows = [];
        if (stageBreakdown) {
            for (const [stage, value] of Object.entries(stageBreakdown)) {
                const type = this.stageType(stage);
                const color = this.colors[type];
                // value can be a float (seconds) or an object
                const timeUs = typeof value === "number"
                    ? (value * 1e6).toFixed(2) + " μs"
                    : (value.time_us != null ? value.time_us.toFixed(2) + " μs" : "—");
                rows.push(`<tr>
                    <td><span class="stage-badge" style="background:${color}20;color:${color};border:1px solid ${color}">${stage}</span></td>
                    <td class="mono">${timeUs}</td>
                    <td class="mono">—</td>
                    <td class="mono">—</td>
                </tr>`);
            }
        }
        if (schedule && schedule.sub_ops) {
            // Add sub-op details grouped by recurring/epilogue
            const recurring = schedule.sub_ops.filter(op => op.iteration >= 0);
            const epilogue = schedule.sub_ops.filter(op => op.iteration === -1);
            if (recurring.length > 0 && rows.length > 0) {
                rows.push(`<tr><td colspan="4" style="padding-top:8px"><span class="stage-badge" style="background:#eff6ff;color:#1a56db;border:1px solid #1a56db">Recurring Sub-Ops (per K iteration)</span></td></tr>`);
            }
            const seenNames = new Set();
            for (const op of recurring) {
                if (seenNames.has(op.name)) continue;
                seenNames.add(op.name);
                const color = this.stageColor(op.pipeline_stage);
                rows.push(`<tr>
                    <td><span class="stage-badge" style="background:${color}20;color:${color};border:1px solid ${color}">${op.name}</span></td>
                    <td class="mono">${op.duration_cycles} cycles</td>
                    <td class="mono">${op.work_units != null ? op.work_units : "—"}</td>
                    <td class="mono">${op.iteration}</td>
                </tr>`);
            }
            if (epilogue.length > 0) {
                rows.push(`<tr><td colspan="4" style="padding-top:8px"><span class="stage-badge" style="background:#fef3c7;color:#d97706;border:1px solid #d97706">Epilogue Sub-Ops</span></td></tr>`);
            }
            for (const op of epilogue) {
                const color = this.stageColor(op.pipeline_stage);
                rows.push(`<tr>
                    <td><span class="stage-badge" style="background:${color}20;color:${color};border:1px solid ${color}">${op.name}</span></td>
                    <td class="mono">${op.duration_cycles} cycles</td>
                    <td class="mono">${op.work_units != null ? op.work_units : "—"}</td>
                    <td class="mono">epilogue</td>
                </tr>`);
            }
        }

        tbody.innerHTML = rows.join("");
    },

    // ── Show / Hide pipeline results ───────────────────────────────
    showPipelineResults() {
        document.getElementById("pipeline-results").classList.remove("hidden");
    },

    hidePipelineResults() {
        document.getElementById("pipeline-results").classList.add("hidden");
    },

    // ── Full render ────────────────────────────────────────────────
    render(result) {
        if (!result.pipeline_schedule) {
            this.hidePipelineResults();
            return;
        }

        // Reset view state on fresh analysis (new schedule, new shape).
        // Mouse/toolbar handlers stay bound across renders — they read
        // live state from viewState and _lastSchedule, so no rebind needed.
        const numK = result.pipeline_schedule.num_k_iterations;
        this.viewState.iterStart = 0;
        this.viewState.iterEnd = Math.min(8, numK);
        this._resetZoom();

        // Bind toolbar once (DOM persists across analyses)
        if (!this._toolbarBound) {
            this._bindToolbar();
            this._toolbarBound = true;
        }

        this.showPipelineResults();
        this.renderPipelineStats(result.pipeline_schedule, result.pipeline_config, result.pipeline_memory_breakdown);
        this.renderTilingInfo(result.tiling_info);
        this.renderRecommendations(result);
        this.renderGanttChart(result.pipeline_schedule);
        this.renderStageBreakdown(result.stage_breakdown, result.pipeline_schedule);
    },
};
