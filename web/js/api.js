/**
 * API client — thin wrapper around fetch for OpCompass backend.
 */
const API = {
    base: "/api",

    async _get(path) {
        const resp = await fetch(`${this.base}${path}`);
        if (!resp.ok) throw new Error(await resp.text());
        return resp.json();
    },

    async _post(path, body) {
        const resp = await fetch(`${this.base}${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!resp.ok) throw new Error(await resp.text());
        return resp.json();
    },

    /** Fetch the list of operators with their param_dims. */
    getOperators() { return this._get("/operators"); },

    /** Fetch the list of hardware targets. */
    getHardware() { return this._get("/hardware"); },

    /** Fetch detailed info for one hardware target. */
    getHardwareDetail(name) { return this._get(`/hardware/${encodeURIComponent(name)}`); },

    /** Fetch full comparison overview of all hardware targets. */
    getHardwareOverview() { return this._get("/hardware/overview"); },

    /** Run a SOL analysis. */
    analyze(operator, hardware, dtype, mode, dims, pipelineConfig) {
        const body = { operator, hardware, dtype, mode, dims };
        if (pipelineConfig) body.pipeline_config = pipelineConfig;
        return this._post("/analyze", body);
    },
};
