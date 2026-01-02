// frontend/main.cjs
const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");

const { ensureModelDownloaded } = require("./model_downloader.cjs");
const { startBackend } = require("./electron_backend.cjs");

let mainWindow = null;
let backendProc = null;
let backendBaseUrl = null;

ipcMain.handle("backend:getUrl", async () => backendBaseUrl);

const MODEL = {
  modelId: "mistral-7b-instruct-v0.2.Q4_K_M",
  url: "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf?download=true",
};

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  const indexPath = path.join(__dirname, "dist", "index.html");
  mainWindow.loadFile(indexPath);

  // Forward renderer console to terminal (helps a lot)
  mainWindow.webContents.on("console-message", (_e, level, message, line, sourceId) => {
    console.log(`[renderer:${level}] ${message} (${sourceId}:${line})`);
  });

  mainWindow.webContents.once("did-finish-load", () => {
    if (backendBaseUrl) {
      console.log("[main] sending backendBaseUrl to renderer:", backendBaseUrl);
      mainWindow.webContents.send("backend:ready", backendBaseUrl);
    }
  });

  return mainWindow;
}

async function bootstrapBackend() {
  console.log("[main] Bootstrapping backend…");

  const modelDir = path.join(app.getPath("userData"), "models", MODEL.modelId);
  const modelPath = path.join(modelDir, "model.gguf");

  console.log("[main] Model dir:", modelDir);
  console.log("[main] Model path:", modelPath);

  await ensureModelDownloaded({
    url: MODEL.url,
    destPath: modelPath,
    onProgress: (p) => {
      if (p?.percent != null) console.log(`[main] Model download: ${(p.percent * 100).toFixed(1)}%`);
    },
  });

  console.log("[main] Starting backend process…");
  const started = await startBackend({ modelPath });

  backendProc = started.child;
  backendBaseUrl = started.baseUrl;

  console.log("[main] Backend ready at:", backendBaseUrl);
}

app.whenReady().then(async () => {
  try {
    await bootstrapBackend();
  } catch (err) {
    console.error("[main] Failed to bootstrap backend:", err);
    // still create window so you can show an error UI
  }

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", () => {
  if (backendProc && !backendProc.killed) {
    try { backendProc.kill(); } catch {}
  }
});

process.on("uncaughtException", (err) => console.error("uncaughtException:", err));
process.on("unhandledRejection", (err) => console.error("unhandledRejection:", err));
