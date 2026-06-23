/**
 * Pipeline analysis visualization — Gantt chart, stats, and stage breakdown.
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

        // Collect unique pipeline stages (y-axis rows)
        const stageSet = new Set();
        subOps.forEach(op => stageSet.add(op.pipeline_stage));
        const stages = Array.from(stageSet).sort((a, b) => {
            // Sort: memory read stages first, then compute, then write
            const order = { memory: 0, async: 1, compute: 2 };
            return (order[this.stageType(a)] || 1) - (order[this.stageType(b)] || 1);
        });

        // Compute timeline range
        const maxCycle = Math.max(...subOps.map(op => op.end_cycle));
        const minCycle = Math.min(...subOps.map(op => op.start_cycle));

        // Phase boundaries
        const prologueEnd = schedule.prologue_cycles;
        const steadyEnd = prologueEnd + schedule.per_iteration_cycles * (schedule.num_k_iterations - 1);

        // Canvas sizing
        const container = document.getElementById("gantt-chart-container");
        const dpr = window.devicePixelRatio || 1;
        const containerWidth = container.clientWidth || 900;
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

        // Scale: cycles -> pixels
        const cycleScale = (cycle) => {
            return leftMargin + (cycle / maxCycle) * chartWidth;
        };

        // ── Background ──
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, containerWidth, totalHeight);

        // ── Phase regions ──
        // Prologue
        const prologueX1 = cycleScale(0);
        const prologueX2 = cycleScale(prologueEnd);
        ctx.fillStyle = this.colors.prologueBg;
        ctx.fillRect(prologueX1, topMargin, prologueX2 - prologueX1, chartHeight);

        // Steady state
        ctx.fillStyle = this.colors.steadyBg;
        ctx.fillRect(prologueX2, topMargin, cycleScale(steadyEnd) - prologueX2, chartHeight);

        // Epilogue
        ctx.fillStyle = this.colors.epilogueBg;
        ctx.fillRect(cycleScale(steadyEnd), topMargin, cycleScale(maxCycle) - cycleScale(steadyEnd), chartHeight);

        // ── Phase labels ──
        ctx.font = "11px Inter, sans-serif";
        ctx.fillStyle = this.colors.phaseLabel;
        ctx.textAlign = "center";
        const labelY = topMargin - 10;

        ctx.fillText("Prologue", (prologueX1 + prologueX2) / 2, labelY);
        ctx.fillText("Steady State", (prologueX2 + cycleScale(steadyEnd)) / 2, labelY);
        ctx.fillText("Epilogue", (cycleScale(steadyEnd) + cycleScale(maxCycle)) / 2, labelY);

        // ── Grid lines ──
        ctx.strokeStyle = this.colors.gridLine;
        ctx.lineWidth = 0.5;

        // Vertical grid: every major cycle milestone
        const gridCycles = [0, prologueEnd, steadyEnd, maxCycle];
        // Add intermediate grid lines if chart is wide enough
        if (chartWidth > 400) {
            const step = Math.max(1, Math.pow(10, Math.floor(Math.log10(maxCycle / 4))));
            for (let c = step; c < maxCycle; c += step) {
                gridCycles.push(c);
            }
        }
        gridCycles.sort((a, b) => a - b);

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
        // Only render a limited number of iterations for clarity
        const maxIterations = Math.min(schedule.num_k_iterations, 8);
        const visibleOps = subOps.filter(op =>
            op.iteration >= 0 && op.iteration < maxIterations ||
            op.iteration === -1  // epilogue ops
        );

        for (const op of visibleOps) {
            const stageIdx = stages.indexOf(op.pipeline_stage);
            if (stageIdx === -1) continue;

            const x1 = cycleScale(op.start_cycle);
            const x2 = cycleScale(op.end_cycle);
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
            ctx.fillText(cycle >= 1000 ? (cycle / 1000).toFixed(1) + "k" : String(cycle), x, topMargin + chartHeight + 15);
        }

        // ── X-axis title ──
        ctx.font = "11px Inter, sans-serif";
        ctx.fillText("Cycles", leftMargin + chartWidth / 2, totalHeight - 5);
    },

    // ── Pipeline Stats ─────────────────────────────────────────────
    renderPipelineStats(schedule, pipelineConfig) {
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

        if (pipelineConfig) {
            rows.push(["Async Copy", pipelineConfig.async_copy_enabled ? "ON" : "OFF"]);
            rows.push(["2:4 Sparsity", pipelineConfig.sparsity_2_4_enabled ? "ON" : "OFF"]);
        }

        container.innerHTML = rows.map(([label, value]) =>
            `<div class="spec-row"><span class="spec-label">${label}</span><span class="spec-value mono">${value}</span></div>`
        ).join("");
    },

    // ── Tiling Info ────────────────────────────────────────────────
    renderTilingInfo(tilingInfo) {
        const container = document.getElementById("tiling-info-content");
        if (!container || !tilingInfo) return;

        const rows = [
            ["Block M", tilingInfo.block_m],
            ["Block N", tilingInfo.block_n],
            ["Block K", tilingInfo.block_k],
            ["Warps/Block", tilingInfo.num_warps_per_block],
            ["Shared Mem", tilingInfo.shared_memory_per_block != null
                ? (tilingInfo.shared_memory_per_block / 1024).toFixed(1) + " KB"
                : "—"],
        ];

        container.innerHTML = rows.map(([label, value]) =>
            `<div class="spec-row"><span class="spec-label">${label}</span><span class="spec-value mono">${value}</span></div>`
        ).join("");
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

        this.showPipelineResults();
        this.renderPipelineStats(result.pipeline_schedule, result.pipeline_config);
        this.renderTilingInfo(result.tiling_info);
        this.renderGanttChart(result.pipeline_schedule);
        this.renderStageBreakdown(result.stage_breakdown, result.pipeline_schedule);
    },
};
