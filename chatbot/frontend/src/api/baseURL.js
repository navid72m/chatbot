import { isElectron } from "../utils/environment";

let cached = null;

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitForElectronBackendUrl(timeoutMs = 60000) {
  const start = Date.now();

  // 1) First try IPC polling
  while (Date.now() - start < timeoutMs) {
    try {
      const url = await window?.electronAPI?.getBackendUrl?.();
      if (url) return url;
    } catch {}
    await sleep(300);
  }

  throw new Error("Timed out waiting for backend URL from Electron main process");
}

export async function getBackendURL() {
  if (cached) return cached;

  if (isElectron()) {
    cached = await waitForElectronBackendUrl(60000);
    console.log("[baseURL] Electron backend URL:", cached);
    return cached;
  }

  if (import.meta.env?.DEV) return "http://localhost:8000";
  return "/api";
}
