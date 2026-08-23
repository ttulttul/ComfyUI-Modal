import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const VAST_STATUS_ROUTE = "/remote/vast/status";
const VAST_VERIFY_ROUTE = "/remote/vast/verify";
const VAST_REAP_ROUTE = "/remote/vast/reap";
const VAST_DESTROY_ROUTE = "/remote/vast/destroy";

/** Return JSON or raise the server's credential-safe diagnostic. */
async function requestJson(route, options = {}) {
  const response = await api.fetchApi(route, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

/** Format one epoch deadline for local inspection. */
function deadlineLabel(epoch) {
  const value = Number(epoch);
  return Number.isFinite(value) ? new Date(value * 1000).toLocaleString() : "unknown";
}

/** Build one managed lease row with an exact destructive action. */
function leaseRow(lease, refresh, setStatus) {
  const row = document.createElement("tr");
  const values = [
    lease.instance_id,
    lease.profile_name,
    `${lease.gpu_count}× ${lease.gpu_name}`,
    `$${Number(lease.hourly_cost_usd || 0).toFixed(4)}/h`,
    lease.actual_status,
    lease.active_invocations,
    deadlineLabel(lease.idle_deadline_epoch),
  ];
  for (const value of values) {
    const cell = document.createElement("td");
    cell.textContent = String(value);
    row.appendChild(cell);
  }
  const action = document.createElement("td");
  const destroy = document.createElement("button");
  destroy.type = "button";
  destroy.textContent = "Destroy";
  destroy.disabled = Number(lease.active_invocations || 0) > 0;
  destroy.addEventListener("click", async () => {
    if (typeof window.confirm === "function" && !window.confirm(
      `Permanently destroy managed Vast.ai instance ${lease.instance_id}?`,
    )) return;
    try {
      setStatus(`Destroying instance ${lease.instance_id}…`);
      await requestJson(VAST_DESTROY_ROUTE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance_id: lease.instance_id }),
      });
      await refresh();
    } catch (error) {
      setStatus(`Destroy failed: ${String(error?.message ?? error)}`);
    }
  });
  action.appendChild(destroy);
  row.appendChild(action);
  return row;
}

/** Open credential verification and owned-lease lifecycle controls. */
export async function openVastManager() {
  const overlay = document.createElement("div");
  overlay.className = "comfy-remote-overlay";
  const dialog = document.createElement("div");
  dialog.className = "comfy-remote-dialog";
  const title = document.createElement("h2");
  title.textContent = "Vast.ai managed leases";
  const intro = document.createElement("p");
  intro.className = "intro";
  intro.textContent = "Capacity, price ceilings, and idle retention live in disconnected Vast.ai Lease Configuration workflow nodes. Set VAST_API_KEY and COMFY_MODAL_VAST_IMAGE in the ComfyUI process environment; secrets are never saved in workflows.";
  const table = document.createElement("table");
  table.style.width = "100%";
  const header = document.createElement("thead");
  header.innerHTML = "<tr><th>ID</th><th>Profile</th><th>GPU</th><th>Price</th><th>Status</th><th>Active</th><th>Idle deadline</th><th></th></tr>";
  const body = document.createElement("tbody");
  table.append(header, body);
  const status = document.createElement("div");
  status.className = "status";
  const setStatus = (message) => { status.textContent = message; };
  const refresh = async () => {
    const payload = await requestJson(VAST_STATUS_ROUTE);
    body.replaceChildren();
    for (const lease of payload.leases ?? []) {
      body.appendChild(leaseRow(lease, refresh, setStatus));
    }
    setStatus(`API key: ${payload.configured ? "configured" : "missing"}; worker image: ${payload.image_configured ? "configured" : "missing"}; managed leases: ${(payload.leases ?? []).length}.`);
  };
  const actions = document.createElement("div");
  actions.className = "footer-group";
  const button = (label, callback) => {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = label;
    element.addEventListener("click", callback);
    actions.appendChild(element);
  };
  button("Verify API key", async () => {
    try {
      const result = await requestJson(VAST_VERIFY_ROUTE, { method: "POST" });
      setStatus(`Vast.ai credential verified for account ${result.account?.id ?? "unknown"}.`);
    } catch (error) {
      setStatus(`Verification failed: ${String(error?.message ?? error)}`);
    }
  });
  button("Refresh", () => void refresh());
  button("Destroy expired now", async () => {
    try {
      const result = await requestJson(VAST_REAP_ROUTE, { method: "POST" });
      setStatus(`Destroyed ${(result.destroyed_instance_ids ?? []).length} expired lease(s).`);
      await refresh();
    } catch (error) {
      setStatus(`Cleanup failed: ${String(error?.message ?? error)}`);
    }
  });
  button("Close", () => overlay.remove());
  dialog.append(title, intro, table, status, actions);
  overlay.appendChild(dialog);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) overlay.remove();
  });
  document.body.appendChild(overlay);
  try {
    await refresh();
  } catch (error) {
    setStatus(`Status failed: ${String(error?.message ?? error)}`);
  }
}

app.registerExtension({
  name: "Comfy.RemoteExecution.Vast",
  async setup() {
    app.ui.settings.addSetting({
      id: "Comfy.RemoteExecution.ManageVast",
      name: "Remote Execution: Vast.ai leases",
      type: () => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "Manage Vast.ai leases";
        button.addEventListener("click", () => void openVastManager());
        return button;
      },
      defaultValue: null,
      tooltip: "Verify Vast.ai setup and inspect or destroy node-pack-managed leases.",
    });
  },
});
