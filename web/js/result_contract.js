/** Pure presentation mapping for the v0.2 analysis-result contract. */
const ResultContract = {
    build(result) {
        const requested = result.requested_mode || result.mode || "unknown";
        const executed = result.executed_mode || result.mode || "unknown";
        const fallback = result.fallback || null;
        const evidence = result.evidence || { coverage: "unknown", sources: [] };
        const uncertainty = result.uncertainty || { status: "unknown", reason: "" };
        return {
            requested,
            executed,
            route: requested === executed ? executed : `${requested} → ${executed}`,
            fallback: Boolean(fallback),
            status: fallback ? "Fallback executed" : "Executed as requested",
            message: fallback ? fallback.message : "The requested analysis model ran without fallback.",
            estimate: result.estimate_kind || "unknown",
            support: result.support_level || "unknown",
            model: result.model_id || "unknown",
            build: `${result.implementation_version || "unknown"} @ ${(result.implementation_revision || "unknown").slice(0, 12)}`,
            hardwareSpec: result.hardware_spec_version || "unknown",
            evidence: evidence.coverage || "unknown",
            evidenceSources: (evidence.sources || []).join(" · ") || "No evidence sources declared",
            uncertainty: uncertainty.status || "unknown",
            uncertaintyReason: uncertainty.reason || "No uncertainty statement declared",
        };
    },
};

if (typeof module !== "undefined" && module.exports) module.exports = ResultContract;
