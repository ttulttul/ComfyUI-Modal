import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SUBROSA_NODE_ID = "SubrosaRemoteConfiguration";
const CREDENTIAL_IMPORT_ROUTE = "/remote/subrosa/credentials";
const CREDENTIAL_STATUS_ROUTE = "/remote/subrosa/status";
const LOGIN_MESSAGE_TYPE = "subrosa-comfyui-login";
const LOGIN_REQUIRED_CODE = "subrosa_login_required";
const AUTHENTICATION_STATUS_EVENT = "subrosa-authentication-status";
const AUTHENTICATE_LABEL = "Click to Authenticate";
const pendingLogins = new Map();
const statusRevisions = new WeakMap();

/** Return JSON or raise one credential-safe local diagnostic. */
async function requestJson(route, options = {}) {
  const response = await api.fetchApi(route, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

/** Return one named serialized node widget. */
function widget(node, name) {
  return (node?.widgets ?? []).find((candidate) => candidate.name === name) ?? null;
}

/** Return the workflow's non-secret keyring reference. */
function credentialId(node) {
  return String(widget(node, "credential_id")?.value ?? "").trim() || String(node?.id ?? "");
}

/** Return the configured relay URL. */
function relayUrl(node) {
  return String(widget(node, "relay_url")?.value ?? "").trim();
}

/** Convert the relay WebSocket URL to the trusted portal origin. */
function portalOrigin(node) {
  const url = new URL(relayUrl(node));
  if (url.protocol === "wss:") url.protocol = "https:";
  else if (url.protocol === "ws:") url.protocol = "http:";
  else throw new Error("relay_url must use ws:// or wss://");
  const hostname = url.hostname.toLowerCase();
  const loopback = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";
  const subrosa = hostname === "subrosa.red" || hostname.endsWith(".subrosa.red");
  if (!loopback && (url.protocol !== "https:" || !subrosa)) {
    throw new Error("Login requires an HTTPS Subrosa service");
  }
  return url.origin;
}

/** Set compact connection state on the non-serialized button. */
function setLoginState(node, label) {
  if (!node?.__subrosaLoginWidget) return;
  node.__subrosaLoginWidget.name = label;
  app.graph?.setDirtyCanvas?.(true, true);
}

/** Invalidate older status requests and return the new node-local revision. */
function nextStatusRevision(node) {
  const revision = (statusRevisions.get(node) ?? 0) + 1;
  statusRevisions.set(node, revision);
  return revision;
}

/** Keep authentication-required state visible until a newer whoami succeeds. */
function requireAuthentication(node) {
  nextStatusRevision(node);
  setLoginState(node, AUTHENTICATE_LABEL);
}

/** Publish a verified connection and clear any older queue-failure decoration. */
function markAuthenticated(node) {
  nextStatusRevision(node);
  setLoginState(node, "Subrosa: Connected");
  window.dispatchEvent(new CustomEvent(AUTHENTICATION_STATUS_EVENT, {
    detail: { node_id: String(node.id), connected: true },
  }));
}

/** Resolve one live Subrosa node by graph ID. */
function subrosaNodeById(nodeId) {
  return (app.graph?._nodes ?? []).find(
    (node) => String(node.id) === String(nodeId) && node.comfyClass === SUBROSA_NODE_ID,
  ) ?? null;
}

/** Ask the local backend whether the saved token still authenticates. */
async function refreshLoginStatus(node) {
  const revision = nextStatusRevision(node);
  const id = credentialId(node);
  if (!id) {
    setLoginState(node, "Subrosa: credential_id required");
    return;
  }
  setLoginState(node, "Subrosa: Checking…");
  try {
    const route = `${CREDENTIAL_STATUS_ROUTE}?credential_id=${encodeURIComponent(id)}&relay_url=${encodeURIComponent(relayUrl(node))}`;
    const status = await requestJson(route);
    if (statusRevisions.get(node) !== revision) return;
    if (status.connected) markAuthenticated(node);
    else requireAuthentication(node);
  } catch (error) {
    if (statusRevisions.get(node) !== revision) return;
    setLoginState(node, `Subrosa: ${String(error?.message ?? error)}`);
  }
}

/** Open the trusted Subrosa portal for a fresh user-authenticated token. */
function startSubrosaLogin(node) {
  let origin;
  try {
    origin = portalOrigin(node);
  } catch (error) {
    setLoginState(node, `Subrosa: ${String(error?.message ?? error)}`);
    return;
  }
  const popup = window.open(
    "about:blank",
    "comfy-modal-subrosa-login",
    "popup,width=720,height=780",
  );
  if (!popup) {
    setLoginState(node, "Subrosa: popup blocked");
    return;
  }
  const state = globalThis.crypto.randomUUID();
  pendingLogins.set(state, { nodeId: String(node.id), origin, popup });
  setLoginState(node, "Subrosa: Sign in in popup…");
  const loginUrl = new URL("/", origin);
  loginUrl.searchParams.set("comfyui_origin", window.location.origin);
  loginUrl.searchParams.set("comfyui_state", state);
  popup.location.href = loginUrl.href;
}

/** Validate and move a portal-minted token into the local OS keyring. */
async function savePortalToken(node, token) {
  return requestJson(CREDENTIAL_IMPORT_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      credential_id: credentialId(node),
      relay_url: relayUrl(node),
      token,
    }),
  });
}

/** Accept only a nonce-bound message from the node's configured relay origin. */
async function handleLoginResult(event) {
  if (event.data?.type !== LOGIN_MESSAGE_TYPE) return;
  const state = String(event.data.state ?? "");
  const pending = pendingLogins.get(state);
  if (!pending || event.origin !== pending.origin || event.source !== pending.popup) return;
  pendingLogins.delete(state);
  const node = subrosaNodeById(pending.nodeId);
  if (!node) return;
  let token = String(event.data.token ?? "");
  try {
    setLoginState(node, "Subrosa: Validating…");
    await savePortalToken(node, token);
    markAuthenticated(node);
  } catch (error) {
    requireAuthentication(node);
  } finally {
    token = "";
  }
}

/** Attach the non-serialized Login control to one concrete Subrosa node. */
function decorateSubrosaNode(node) {
  if (node?.comfyClass !== SUBROSA_NODE_ID || node.__subrosaLoginWidget) return;
  const loginWidget = node.addWidget(
    "button",
    AUTHENTICATE_LABEL,
    null,
    () => startSubrosaLogin(node),
    { serialize: false },
  );
  loginWidget.serialize = false;
  node.__subrosaLoginWidget = loginWidget;
  void refreshLoginStatus(node);
}

/** Apply a queue-time authentication failure to the exact Subrosa node. */
function handleModalStatus(event) {
  const detail = event?.detail ?? event ?? {};
  if (detail.phase !== "error" || detail.error_code !== LOGIN_REQUIRED_CODE) return;
  const node = subrosaNodeById(detail.failed_node_id);
  if (node) requireAuthentication(node);
}

app.registerExtension({
  name: "Comfy.RemoteExecution.SubrosaLogin",

  async init() {
    window.addEventListener("message", (event) => void handleLoginResult(event));
    api.addEventListener("modal_status", handleModalStatus);
  },

  async nodeCreated(node) {
    decorateSubrosaNode(node);
  },

  async loadedGraphNode(node) {
    decorateSubrosaNode(node);
    if (node?.comfyClass === SUBROSA_NODE_ID) void refreshLoginStatus(node);
  },

  async afterConfigureGraph() {
    for (const node of app.graph?._nodes ?? []) decorateSubrosaNode(node);
  },
});
