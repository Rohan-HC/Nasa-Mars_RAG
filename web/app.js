const state = { token: sessionStorage.getItem("nasaRagToken"), documents: [] };
const byId = (id) => document.getElementById(id);

function errorMessage(data, fallback) { return data?.error?.message || data?.detail || fallback; }

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  const response = await fetch(path, { ...options, headers });
  const data = response.status === 204 ? null : await response.json();
  if (!response.ok) {
    if (response.status === 401 && path !== "/auth/token") logout();
    throw new Error(errorMessage(data, `Request failed (${response.status})`));
  }
  return data;
}

function setAuthenticated(authenticated, username = "") {
  byId("login-panel").hidden = authenticated;
  byId("application").hidden = !authenticated;
  byId("session-controls").hidden = !authenticated;
  byId("current-user").textContent = username;
}

function logout() {
  state.token = null;
  state.documents = [];
  sessionStorage.removeItem("nasaRagToken");
  setAuthenticated(false);
}

async function login(event) {
  event.preventDefault();
  const status = byId("login-status");
  status.textContent = "Signing in…";
  try {
    const data = await apiFetch("/auth/token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: byId("username").value, password: byId("password").value })
    });
    state.token = data.access_token;
    sessionStorage.setItem("nasaRagToken", state.token);
    byId("password").value = "";
    await restoreSession();
  } catch (error) { status.textContent = error.message; }
}

function fillSelect(select, documents) {
  select.replaceChildren();
  for (const documentRecord of documents) {
    const option = document.createElement("option");
    option.value = documentRecord.document_id;
    option.textContent = `${documentRecord.name} (${documentRecord.status})`;
    select.append(option);
  }
}

async function loadDocuments() {
  const data = await apiFetch("/documents");
  state.documents = data.documents;
  const ready = state.documents.filter((item) => item.status === "ready");
  fillSelect(byId("replace-document"), state.documents.filter((item) => item.role === "owner"));
  fillSelect(byId("document-a"), ready);
  fillSelect(byId("document-b"), ready);
  if (ready.length > 1) byId("document-b").selectedIndex = 1;
  const list = byId("document-list");
  list.replaceChildren();
  for (const item of state.documents) {
    const row = document.createElement("li");
    row.textContent = `${item.name} — ${item.status} — ${item.role}`;
    list.append(row);
  }
}

function delay(milliseconds) { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }

async function pollJob(jobId, statusElement) {
  for (let attempt = 0; attempt < 600; attempt += 1) {
    const job = await apiFetch(`/jobs/${jobId}`);
    statusElement.textContent = `Ingestion ${job.status}; attempt ${job.attempt_count}.`;
    if (job.status === "succeeded") { await loadDocuments(); return job; }
    if (["failed", "cancelled"].includes(job.status)) throw new Error(job.error_class || job.status);
    await delay(1000);
  }
  throw new Error("Job polling timed out; check the worker process.");
}

async function queueFile(path, file, statusElement) {
  if (!file || !file.name.toLowerCase().endsWith(".pdf")) throw new Error("Select a PDF file.");
  if (file.size > 10 * 1024 * 1024) throw new Error("PDF must be 10 MB or smaller.");
  const body = new FormData();
  body.append("file", file);
  const method = path === "/documents/upload" ? "POST" : "PUT";
  const queued = await apiFetch(path, { method, body });
  statusElement.textContent = `Queued job ${queued.job_id}.`;
  return pollJob(queued.job_id, statusElement);
}

async function upload() {
  const button = byId("upload-button"); const status = byId("upload-status");
  button.disabled = true;
  try { await queueFile("/documents/upload", byId("pdf-upload").files[0], status); byId("pdf-upload").value = ""; }
  catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; }
}

async function replaceDocument() {
  const button = byId("replace-button"); const status = byId("upload-status");
  const documentId = byId("replace-document").value;
  if (!documentId) { status.textContent = "Select an owned document."; return; }
  button.disabled = true;
  try { await queueFile(`/documents/${documentId}/content`, byId("replacement-upload").files[0], status); byId("replacement-upload").value = ""; }
  catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; }
}

function renderSources(list, sources) {
  list.replaceChildren();
  for (const source of sources || []) {
    const row = document.createElement("li");
    row.textContent = `[${source.label}] ${source.title || source.filename} — page ${source.page}: ${source.text || ""}`;
    list.append(row);
  }
}

async function ask() {
  const status = byId("ask-status"); const results = byId("ask-results");
  results.hidden = true;
  try {
    const data = await apiFetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: byId("ask-question").value.trim(), retrieval_mode: byId("retrieval-mode").value }) });
    byId("ask-answer").textContent = data.answer;
    renderSources(byId("ask-sources"), data.retrieved);
    results.hidden = false;
    status.textContent = "Answer generated; verify each claim against its cited passage.";
  } catch (error) { status.textContent = error.message; }
}

async function compare() {
  const status = byId("compare-status"); const results = byId("comparison-results");
  results.hidden = true;
  try {
    const data = await apiFetch("/compare/documents", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document_a: byId("document-a").value, document_b: byId("document-b").value, question: byId("comparison-question").value.trim(), retrieval_mode: byId("retrieval-mode").value }) });
    byId("comparison-answer").textContent = data.answer;
    renderSources(byId("comparison-sources"), data.retrieved);
    byId("validation-status").textContent = data.validation?.passed ? "Citation-label checks passed." : `Validation issues: ${(data.validation?.errors || []).join("; ")}`;
    results.hidden = false;
    status.textContent = "Comparison generated.";
  } catch (error) { status.textContent = error.message; }
}

async function restoreSession() {
  if (!state.token) { setAuthenticated(false); return; }
  try {
    const user = await apiFetch("/auth/me");
    setAuthenticated(true, user.username);
    await loadDocuments();
  } catch (_error) { logout(); }
}

byId("login-form").addEventListener("submit", login);
byId("logout-button").addEventListener("click", logout);
byId("refresh-button").addEventListener("click", loadDocuments);
byId("upload-button").addEventListener("click", upload);
byId("replace-button").addEventListener("click", replaceDocument);
byId("ask-button").addEventListener("click", ask);
byId("compare-button").addEventListener("click", compare);
restoreSession();
