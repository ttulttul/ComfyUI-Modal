import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const R2_NODE_ID = "R2StorageBackingConfiguration";
const OAUTH_START_ROUTE = "/remote/storage/r2/oauth/start";
const CREDENTIAL_STATUS_ROUTE = "/remote/storage/r2/status";
const OAUTH_MESSAGE_TYPE = "comfy-modal-r2-oauth";
const pendingNodes = new Set();

/** Return JSON or raise the server's credential-safe diagnostic. */
async function requestJson(route, options = {}) {
  const response = await api.fetchApi(route, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

/** Return one named node widget. */
function widget(node, name) {
  return (node?.widgets ?? []).find((candidate) => candidate.name === name) ?? null;
}

/** Return a persistent, non-secret keyring reference for this workflow node. */
function ensureCredentialId(node) {
  const credentialWidget = widget(node, "credential_id");
  if (!credentialWidget) return "";
  if (!String(credentialWidget.value ?? "").trim()) {
    credentialWidget.value = globalThis.crypto?.randomUUID?.()
      ?? `r2-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    app.graph?.setDirtyCanvas?.(true, true);
  }
  return String(credentialWidget.value).trim();
}

/** Update a serialized node widget and notify the graph UI. */
function setWidgetValue(node, name, value) {
  const target = widget(node, name);
  if (!target) return;
  target.value = value;
  target.callback?.(value, app.canvas, node, [0, 0]);
}

/** Set the Login button's compact connection state. */
function setLoginState(node, label) {
  if (!node?.__r2LoginWidget) return;
  node.__r2LoginWidget.name = label;
  app.graph?.setDirtyCanvas?.(true, true);
}

/** Find an R2 node by its serialized graph ID. */
function r2NodeById(nodeId) {
  return (app.graph?._nodes ?? []).find(
    (node) => String(node.id) === String(nodeId) && node.comfyClass === R2_NODE_ID,
  ) ?? null;
}

/** Refresh credential existence without retrieving any secret material. */
async function refreshCredentialStatus(node) {
  const credentialId = ensureCredentialId(node);
  if (!credentialId) return;
  try {
    const route = `${CREDENTIAL_STATUS_ROUTE}?credential_id=${encodeURIComponent(credentialId)}`;
    const status = await requestJson(route);
    setLoginState(node, status.connected ? "Cloudflare: Connected" : "Login to Cloudflare");
    if (status.connected) {
      setWidgetValue(node, "account_id", status.account_id ?? "");
      setWidgetValue(node, "bucket", status.bucket ?? "");
      setWidgetValue(node, "jurisdiction", status.jurisdiction ?? "default");
    }
  } catch (error) {
    setLoginState(node, `Cloudflare: ${String(error?.message ?? error)}`);
  }
}

/** Start OAuth immediately from a node button, independent of workflow execution. */
async function startCloudflareLogin(node) {
  const popup = window.open(
    "about:blank",
    "comfy-modal-cloudflare-r2",
    "popup,width=720,height=780",
  );
  if (!popup) {
    setLoginState(node, "Cloudflare: popup blocked");
    return;
  }
  pendingNodes.add(String(node.id));
  setLoginState(node, "Cloudflare: Opening login…");
  try {
    const result = await requestJson(OAUTH_START_ROUTE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        node_id: String(node.id),
        credential_id: ensureCredentialId(node),
        account_id: String(widget(node, "account_id")?.value ?? "").trim(),
        bucket: String(widget(node, "bucket")?.value ?? "").trim(),
        jurisdiction: String(widget(node, "jurisdiction")?.value ?? "default"),
        origin: window.location.origin,
      }),
    });
    popup.location.href = result.authorization_url;
  } catch (error) {
    popup.close();
    pendingNodes.delete(String(node.id));
    setLoginState(node, `Cloudflare: ${String(error?.message ?? error)}`);
  }
}

/** Attach the non-serialized Login control to one concrete R2 node. */
function decorateR2Node(node) {
  if (node?.comfyClass !== R2_NODE_ID || node.__r2LoginWidget) return;
  ensureCredentialId(node);
  const loginWidget = node.addWidget(
    "button",
    "Login to Cloudflare",
    null,
    () => void startCloudflareLogin(node),
    { serialize: false },
  );
  loginWidget.serialize = false;
  node.__r2LoginWidget = loginWidget;
  void refreshCredentialStatus(node);
}

/** Apply a credential-free OAuth popup result to its originating node. */
function handleOAuthResult(event) {
  if (event.origin !== window.location.origin || event.data?.type !== OAUTH_MESSAGE_TYPE) return;
  const targetIds = event.data.node_id ? [String(event.data.node_id)] : [...pendingNodes];
  for (const targetId of targetIds) {
    const node = r2NodeById(targetId);
    pendingNodes.delete(targetId);
    if (!node) continue;
    if (!event.data.ok) {
      setLoginState(node, `Cloudflare: ${String(event.data.error || "Login failed")}`);
      continue;
    }
    setWidgetValue(node, "account_id", String(event.data.account_id ?? ""));
    setWidgetValue(node, "bucket", String(event.data.bucket ?? ""));
    setWidgetValue(node, "jurisdiction", String(event.data.jurisdiction ?? "default"));
    setLoginState(node, "Cloudflare: Connected");
  }
  app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "Comfy.RemoteExecution.R2Storage",

  async init() {
    window.addEventListener("message", handleOAuthResult);
  },

  async nodeCreated(node) {
    decorateR2Node(node);
  },

  async loadedGraphNode(node) {
    decorateR2Node(node);
    if (node?.comfyClass === R2_NODE_ID) void refreshCredentialStatus(node);
  },

  async afterConfigureGraph() {
    for (const node of app.graph?._nodes ?? []) decorateR2Node(node);
  },
});
