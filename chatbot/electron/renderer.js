let _cachedBaseUrl = null;

async function getBaseUrl() {
  if (_cachedBaseUrl) return _cachedBaseUrl;

  // Preferred: preload exposed API (contextIsolation on)
  if (window?.electronAPI?.getBackendUrl) {
    const url = await window.electronAPI.getBackendUrl();
    if (!url) throw new Error("Backend URL not ready (electronAPI returned null)");
    _cachedBaseUrl = url.replace("localhost", "127.0.0.1");
    return _cachedBaseUrl;
  }

  // Fallback: if you exposed ipcRenderer on window (not recommended)
  if (window?.electron?.ipcRenderer?.invoke) {
    const url = await window.electron.ipcRenderer.invoke("backend:getUrl");
    if (!url) throw new Error("Backend URL not ready (ipc invoke returned null)");
    _cachedBaseUrl = url.replace("localhost", "127.0.0.1");
    return _cachedBaseUrl;
  }

  throw new Error("No Electron bridge found. Ensure preload exposes electronAPI.getBackendUrl()");
}

export async function fetchFromBackend(endpoint, method = "GET", data = null) {
  const baseUrl = await getBaseUrl();

  const options = {
    method,
    headers: { "Content-Type": "application/json" },
  };

  if (data && (method === "POST" || method === "PUT")) {
    options.body = JSON.stringify(data);
  }

  try {
    const response = await fetch(`${baseUrl}${endpoint}`, options);

    // helpful error handling
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`HTTP ${response.status} ${response.statusText} ${text}`);
    }

    // if backend sometimes returns non-json
    const ct = response.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      return { raw: await response.text() };
    }

    return await response.json();
  } catch (error) {
    console.error("Error communicating with backend:", error);
    return { error: "Failed to communicate with backend", detail: String(error) };
  }
}

// Example usage
export async function getChatResponse(message) {
  return await fetchFromBackend("/api/chat", "POST", { message });
}
