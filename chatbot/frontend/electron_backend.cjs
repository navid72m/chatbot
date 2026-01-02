const fs = require("fs");
const path = require("path");
const net = require("net");
const http = require("http");
const { app } = require("electron");
const { spawn } = require("child_process");

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}

function waitForHealth(url, timeoutMs = 60000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(url, (res) => {
        res.resume();
        if (res.statusCode === 200) return resolve(true);
        if (Date.now() - start > timeoutMs) return reject(new Error("Health check timeout"));
        setTimeout(tick, 300);
      });
      req.on("error", () => {
        if (Date.now() - start > timeoutMs) return reject(new Error("Health check timeout"));
        setTimeout(tick, 300);
      });
      req.end?.();
    };
    tick();
  });
}

function getBackendExePath() {
  const exe = process.platform === "win32" ? "chatbot-backend.exe" : "chatbot-backend";

  // packaged
  const prod = path.join(process.resourcesPath, "backend", "chatbot-backend", exe);

  // dev
  const dev = path.join(__dirname, "..", "backend", "dist", "chatbot-backend", exe);

  if (fs.existsSync(prod)) return prod;
  if (fs.existsSync(dev)) return dev;

  throw new Error(`Backend executable not found.\nprod=${prod}\ndev=${dev}`);
}

async function startBackend({ modelPath }) {
  if (!modelPath) throw new Error("startBackend: modelPath is required");

  const backendExe = getBackendExePath();
  const port = await getFreePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  const userData = app.getPath("userData");

  const env = {
    ...process.env,
    LLAMA_CPP_MODEL_PATH: modelPath,
    SENTENCE_TRANSFORMERS_HOME: path.join(userData, "st_cache"),
    HF_HOME: path.join(userData, "hf_cache"),
    TRANSFORMERS_CACHE: path.join(userData, "hf_cache"),
  };

  const child = spawn(
    backendExe,
    ["--host", "127.0.0.1", "--port", String(port), "--model", modelPath],
    { env, stdio: ["ignore", "pipe", "pipe"], windowsHide: true }
  );

  child.stdout.on("data", (d) => console.log("[backend]", d.toString()));
  child.stderr.on("data", (d) => console.error("[backend]", d.toString()));

  await waitForHealth(`${baseUrl}/health`, 60000);
  return { child, baseUrl };
}

module.exports = { startBackend };
