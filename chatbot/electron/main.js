import pkg from 'electron';
const { app, BrowserWindow } = pkg;
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import fs from 'fs';
import http from 'http';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let backendProcess;
let ollamaProcess;

function waitForServerReady(url, timeout = 30000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();

    const check = () => {
      const req = http.get(url, res => {
        if (res.statusCode === 200) {
          console.log(`✅ Server ready at ${url}`);
          return resolve(true);
        }
        retry();
      });

      req.on('error', retry);
      req.setTimeout(1000, () => {
        req.destroy();
        retry();
      });

      function retry() {
        if (Date.now() - start > timeout) {
          reject(new Error(`Timeout waiting for ${url}`));
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
    },
  });

  const indexPath = path.join(__dirname, '..', 'dist', 'index.html');
  win.loadFile(indexPath);
  win.webContents.openDevTools();
}

function startBackend() {
  const backendScript = path.join(__dirname, 'bin', 'wrapper.py');
  const pythonExecutable = '/Users/seyednavidmirnourilangeroudi/miniconda3/bin/python'; // Make dynamic if needed

  if (!fs.existsSync(backendScript)) {
    console.error(`❌ Backend script not found at: ${backendScript}`);
    return false;
  }

  console.log(`🚀 Starting backend with: ${pythonExecutable} ${backendScript}`);
  backendProcess = spawn(pythonExecutable, [backendScript], {
    cwd: path.dirname(backendScript),
    stdio: 'pipe',
  });

  backendProcess.stdout.on('data', (data) => console.log(`🟢 Backend: ${data}`));
  backendProcess.stderr.on('data', (data) => console.error(`🔴 Backend Error: ${data}`));

  return true;
}

function startOllama() {
  const ollamaPath = app.isPackaged
    ? path.join(process.resourcesPath, 'bin', 'ollama')
    : path.join(__dirname, 'bin', 'ollama');

  if (!fs.existsSync(ollamaPath)) {
    console.error(`❌ Ollama binary not found at: ${ollamaPath}`);
    return false;
  }

  console.log(`🚀 Starting Ollama from: ${ollamaPath}`);
  ollamaProcess = spawn(ollamaPath, ['serve'], {
    cwd: path.dirname(ollamaPath),
    stdio: 'pipe',
  });

  ollamaProcess.stdout.on('data', data =>
    console.log(`🟢 Ollama: ${data.toString()}`),
  );
  ollamaProcess.stderr.on('data', data =>
    console.error(`🔴 Ollama Error: ${data.toString()}`),
  );

  return true;
}

app.whenReady().then(async () => {
  const ollamaStarted = startOllama();
  if (!ollamaStarted) return;

  try {
    console.log('⏳ Waiting for Ollama to be ready...');
    await waitForServerReady('http://localhost:11434');
    console.log('✅ Ollama is ready');

    const backendStarted = startBackend();
    if (!backendStarted) {
      const errorWin = new BrowserWindow({ width: 600, height: 400 });
      errorWin.loadURL(
        'data:text/html,<h1>Backend binary missing</h1><p>Please check packaging paths.</p>',
      );
      return;
    }

    console.log('⏳ Waiting for FastAPI backend...');
    await waitForServerReady('http://localhost:8000/health');
    console.log('✅ Backend is ready');
    createWindow();
  } catch (err) {
    console.error('❌ Startup error:', err.message);
    const failWin = new BrowserWindow({ width: 600, height: 400 });
    failWin.loadURL(
      `data:text/html,<h1>Startup failed</h1><p>${err.message}</p>`,
    );
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
