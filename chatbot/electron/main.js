import pkg from 'electron';
const { app, BrowserWindow } = pkg;
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import fs from 'fs';
import http from 'http'; // use built-in http module

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let backendProcess;
let ollamaProcess;

function waitForServerReady(url, timeout = 10000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();

    const check = () => {
      const req = http.get(url, (res) => {
        if (res.statusCode === 200) {
          resolve(true);
        } else {
          retry();
        }
      });

      req.on('error', retry);
      req.setTimeout(1000, retry);

      function retry() {
        if (Date.now() - start > timeout) {
          reject(new Error('Server did not become ready in time'));
        } else {
          setTimeout(check, 500);
        }
      }
    };

    check();
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
    }
  });

  const indexPath = path.join(__dirname, '..', 'dist', 'index.html');
  win.loadFile(indexPath);
  win.webContents.openDevTools();
}

function startBackend() {
  const backendPath = path.join(process.resourcesPath, 'bin', 'backend');
  const pythonPath = path.join(process.resourcesPath, 'backend', 'venv', 'bin', 'python');


  console.log(`Starting backend from: ${backendPath}`);
  backendProcess = spawn(pythonPath, [backendPath], {
    cwd: path.dirname(backendPath),
    stdio: 'pipe',
  });

  backendProcess.stdout.on('data', data => console.log(`Backend: ${data}`));
  backendProcess.stderr.on('data', data => console.error(`Backend Error: ${data}`));
}

function startOllama() {
  const ollamaPath = app.isPackaged
    ? path.join(process.resourcesPath, 'bin', 'ollama')
    : path.join(__dirname, 'bin', 'ollama');

  if (fs.existsSync(ollamaPath)) {
    console.log(`Starting Ollama from: ${ollamaPath}`);
    ollamaProcess = spawn(ollamaPath, ['serve'], {
      cwd: path.dirname(ollamaPath),
      stdio: 'pipe',
    });

    ollamaProcess.stdout.on('data', data => console.log(`Ollama: ${data}`));
    ollamaProcess.stderr.on('data', data => console.error(`Ollama Error: ${data}`));
  } else {
    console.error(`Ollama binary not found at: ${ollamaPath}`);
  }
}

// Handle Electron lifecycle
app.whenReady().then(async () => {
  startBackend();
  startOllama();

  try {
    console.log('⏳ Waiting for FastAPI server to be ready...');
    await waitForServerReady('http://localhost:8000/health');
    console.log('✅ FastAPI is ready. Launching UI...');
    createWindow();
  } catch (err) {
    console.error('❌ FastAPI failed to start in time:', err);
    const failWin = new BrowserWindow({ width: 600, height: 400 });
    failWin.loadURL('data:text/html,<h1>Backend failed to start</h1><p>Try again or check logs.</p>');
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on('will-quit', () => {
  if (backendProcess) backendProcess.kill();
  if (ollamaProcess) ollamaProcess.kill();
});
