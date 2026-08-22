import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ENVIRONMENTS_ROUTE = "/remote/environments";
const PROBE_ROUTE = "/remote/environments/probe";
const BOOTSTRAP_ROUTE = "/remote/environments/bootstrap";
const STYLE_ID = "comfy-remote-environment-styles";

/** Return parsed JSON or raise the server's descriptive error. */
async function requestJson(route, options = {}) {
  const response = await api.fetchApi(route, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

/** Convert a per-second price to the friendlier hourly value used by the form. */
function hourlyCost(host) {
  const perSecond = Number(host?.cost_usd_per_second);
  return Number.isFinite(perSecond) ? String(perSecond * 3600) : "";
}

/** Normalize one form row into the persisted, credential-free API schema. */
function hostFromRow(row, previousHost = {}) {
  const value = (name) => row.querySelector(`[name="${name}"]`)?.value?.trim() ?? "";
  const checked = (name) => Boolean(row.querySelector(`[name="${name}"]`)?.checked);
  const costPerHour = value("cost_per_hour");
  const reserveVramGb = Number(value("reserve_vram_gb") || 0);
  return {
    ...previousHost,
    environment_id: value("environment_id").toLowerCase(),
    display_name: value("display_name"),
    ssh_target: value("ssh_target"),
    enabled: checked("enabled"),
    draining: checked("draining"),
    cost_usd_per_second: costPerHour === "" ? null : Number(costPerHour) / 3600,
    maximum_workers: Number(value("maximum_workers") || 1),
    reserve_vram_bytes: Math.round(reserveVramGb * 1024 ** 3),
    tags: value("tags").split(",").map((tag) => tag.trim()).filter(Boolean),
  };
}

/** Return a compact human-readable capability summary. */
function capabilitySummary(host) {
  const capabilities = host?.capabilities;
  if (!capabilities) {
    return host?.last_error || "Not probed yet";
  }
  const gpus = (capabilities.gpus ?? []).map((gpu) => {
    const vramGb = Number(gpu.total_vram_bytes || 0) / 1024 ** 3;
    return `${gpu.name} (${vramGb.toFixed(1)} GB)`;
  });
  const ramGb = Number(capabilities.total_ram_bytes || 0) / 1024 ** 3;
  return `${gpus.join(", ") || "CPU only"}; ${ramGb.toFixed(1)} GB RAM; Docker ${capabilities.docker_version}`;
}

/** Install the scoped host-manager stylesheet once. */
function installStyles() {
  if (document.getElementById(STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .comfy-remote-overlay { position: fixed; inset: 0; z-index: 100000; display: grid; place-items: center; background: rgba(2,6,23,.72); }
    .comfy-remote-dialog { width: min(1050px, 94vw); max-height: 90vh; overflow: auto; box-sizing: border-box; padding: 22px; border: 1px solid var(--border-color, #475569); border-radius: 12px; background: var(--comfy-menu-bg, #18181b); color: var(--input-text, #f8fafc); box-shadow: 0 24px 80px rgba(0,0,0,.55); font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; }
    .comfy-remote-dialog h2 { margin: 0 0 6px; font-size: 20px; }
    .comfy-remote-dialog .intro { margin: 0 0 18px; color: #cbd5e1; }
    .comfy-remote-host { display: grid; grid-template-columns: 1fr 1.2fr 1.4fr .65fr .65fr .65fr 1fr; gap: 8px; margin: 0 0 10px; padding: 12px; border: 1px solid #475569; border-radius: 8px; }
    .comfy-remote-host label { display: flex; min-width: 0; flex-direction: column; gap: 4px; color: #cbd5e1; font-size: 11px; }
    .comfy-remote-host input { min-width: 0; box-sizing: border-box; padding: 7px; border: 1px solid #64748b; border-radius: 5px; background: #0f172a; color: #f8fafc; }
    .comfy-remote-host .checks { display: flex; align-items: center; gap: 12px; grid-column: 1 / span 3; }
    .comfy-remote-host .checks label { flex-direction: row; align-items: center; }
    .comfy-remote-host .capabilities { grid-column: 4 / -1; align-self: center; color: #94a3b8; }
    .comfy-remote-host .actions { display: flex; gap: 8px; grid-column: 1 / -1; }
    .comfy-remote-dialog button { padding: 7px 11px; border: 1px solid #64748b; border-radius: 6px; background: #334155; color: #f8fafc; cursor: pointer; }
    .comfy-remote-dialog button.primary { border-color: #2563eb; background: #2563eb; }
    .comfy-remote-dialog button.danger { border-color: #991b1b; background: #7f1d1d; }
    .comfy-remote-dialog .footer { display: flex; justify-content: space-between; gap: 8px; margin-top: 16px; }
    .comfy-remote-dialog .footer-group { display: flex; gap: 8px; }
    .comfy-remote-dialog .status { min-height: 20px; margin-top: 10px; color: #93c5fd; }
    @media (max-width: 900px) { .comfy-remote-host { grid-template-columns: 1fr 1fr; } .comfy-remote-host .checks, .comfy-remote-host .capabilities, .comfy-remote-host .actions { grid-column: 1 / -1; } }
  `;
  document.head.appendChild(style);
}

/** Construct one editable SSH host card. */
function createHostRow(host, manager) {
  const row = document.createElement("section");
  row.className = "comfy-remote-host";
  row.__remoteHost = host;
  const field = (label, name, value, type = "text", step = null) => {
    const wrapper = document.createElement("label");
    wrapper.textContent = label;
    const input = document.createElement("input");
    input.name = name;
    input.type = type;
    input.value = value ?? "";
    if (step != null) input.step = step;
    wrapper.appendChild(input);
    row.appendChild(wrapper);
  };
  field("Environment ID", "environment_id", host.environment_id);
  field("Display name", "display_name", host.display_name);
  field("SSH target or alias", "ssh_target", host.ssh_target);
  field("Cost USD/hour", "cost_per_hour", hourlyCost(host), "number", "0.0001");
  field("Max workers", "maximum_workers", host.maximum_workers ?? 1, "number", "1");
  field("Reserve VRAM GB", "reserve_vram_gb", Number(host.reserve_vram_bytes || 0) / 1024 ** 3, "number", "0.1");
  field("Tags (comma separated)", "tags", (host.tags ?? []).join(", "));

  const checks = document.createElement("div");
  checks.className = "checks";
  for (const [label, name, checked] of [
    ["Enabled", "enabled", host.enabled ?? true],
    ["Drain (no new work)", "draining", host.draining ?? false],
  ]) {
    const wrapper = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = name;
    input.checked = checked;
    wrapper.append(input, label);
    checks.appendChild(wrapper);
  }
  row.appendChild(checks);

  const capabilities = document.createElement("div");
  capabilities.className = "capabilities";
  capabilities.textContent = `${host.health ?? "unknown"}: ${capabilitySummary(host)}`;
  row.appendChild(capabilities);

  const actions = document.createElement("div");
  actions.className = "actions";
  const actionButton = (label, callback, className = "") => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = className;
    button.addEventListener("click", callback);
    actions.appendChild(button);
  };
  actionButton("Save and probe", () => manager.saveAndOperate(row, PROBE_ROUTE));
  actionButton("Build / update worker", () => manager.saveAndOperate(row, BOOTSTRAP_ROUTE));
  actionButton("Remove", () => row.remove(), "danger");
  row.appendChild(actions);
  return row;
}

/** Manage the complete SSH host registry in a ComfyUI-local dialog. */
class RemoteEnvironmentManager {
  constructor(config) {
    this.config = config;
    this.overlay = document.createElement("div");
    this.overlay.className = "comfy-remote-overlay";
    this.dialog = document.createElement("div");
    this.dialog.className = "comfy-remote-dialog";
    this.rows = document.createElement("div");
    this.status = document.createElement("div");
    this.status.className = "status";
    this.overlay.appendChild(this.dialog);
  }

  render() {
    const title = document.createElement("h2");
    title.textContent = "Remote execution environments";
    const intro = document.createElement("p");
    intro.className = "intro";
    intro.textContent = "Add SSH destinations that already work non-interactively and have Docker installed. Host-key verification is strict; credentials stay in your SSH agent/config and are never stored here.";
    this.dialog.append(title, intro, this.rows);
    for (const host of this.config.hosts ?? []) {
      this.rows.appendChild(createHostRow(host, this));
    }
    const footer = document.createElement("div");
    footer.className = "footer";
    const left = document.createElement("div");
    left.className = "footer-group";
    const right = document.createElement("div");
    right.className = "footer-group";
    const button = (label, callback, className = "") => {
      const element = document.createElement("button");
      element.type = "button";
      element.textContent = label;
      element.className = className;
      element.addEventListener("click", callback);
      return element;
    };
    left.appendChild(button("Add host", () => {
      this.rows.appendChild(createHostRow({ enabled: true, maximum_workers: 1 }, this));
    }));
    right.append(
      button("Close", () => this.overlay.remove()),
      button("Save", () => this.save(), "primary"),
    );
    footer.append(left, right);
    this.dialog.append(this.status, footer);
    this.overlay.addEventListener("click", (event) => {
      if (event.target === this.overlay) this.overlay.remove();
    });
    document.body.appendChild(this.overlay);
  }

  hosts() {
    return [...this.rows.querySelectorAll(".comfy-remote-host")].map((row) =>
      hostFromRow(row, row.__remoteHost),
    );
  }

  async save() {
    this.status.textContent = "Saving…";
    try {
      this.config = await requestJson(ENVIRONMENTS_ROUTE, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version: 1, hosts: this.hosts() }),
      });
      this.status.textContent = "Saved remote execution environments.";
      return this.config;
    } catch (error) {
      this.status.textContent = `Save failed: ${String(error?.message ?? error)}`;
      throw error;
    }
  }

  async saveAndOperate(row, route) {
    try {
      await this.save();
      const environmentId = row.querySelector('[name="environment_id"]')?.value?.trim();
      this.status.textContent = route === PROBE_ROUTE ? "Probing host…" : "Building worker image…";
      await requestJson(route, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ environment_id: environmentId }),
      });
      this.overlay.remove();
      await openRemoteEnvironmentManager();
    } catch (error) {
      this.status.textContent = `Operation failed: ${String(error?.message ?? error)}`;
    }
  }
}

/** Open the native extension manager for SSH Docker execution environments. */
export async function openRemoteEnvironmentManager() {
  installStyles();
  const config = await requestJson(ENVIRONMENTS_ROUTE);
  new RemoteEnvironmentManager(config).render();
}

app.registerExtension({
  name: "Comfy.RemoteExecution.Environments",
  async setup() {
    app.ui.settings.addSetting({
      id: "Comfy.RemoteExecution.ManageEnvironments",
      name: "Remote Execution: SSH environments",
      type: "button",
      defaultValue: "Manage SSH hosts",
      tooltip: "Configure, probe, and bootstrap Docker workers reached through SSH.",
      onClick: () => void openRemoteEnvironmentManager(),
    });
  },
});
