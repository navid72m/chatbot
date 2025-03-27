// In your Electron app's main process file
const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

let backendProcess = null;
let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  mainWindow.loadFile('index.html');
}

function startBackend() {
  // Start Docker container for backend
  backendProcess = spawn('bash', ['./launch_app.sh'], {
    detached: false
  });
  
  backendProcess.stdout.on('data', (data) => {
    console.log(`Backend stdout: ${data}`);
  });
  
  backendProcess.stderr.on('data', (data) => {
    console.error(`Backend stderr: ${data}`);
  });
  
  // Wait for backend to be available
  checkBackendAvailability();
}

function checkBackendAvailability() {
  const req = http.request({
    host: 'localhost',
    port: 8000,
    path: '/',
    method: 'GET'
  }, (res) => {
    if (res.statusCode === 200) {
      console.log('Backend is running');
    }
  });
  
  req.on('error', (e) => {
    console.log('Backend not ready yet, retrying in 1 second...');
    setTimeout(checkBackendAvailability, 1000);
  });
  
  req.end();
}

app.whenReady().then(() => {
  startBackend();
  createWindow();
  
  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('will-quit', () => {
  // Cleanup: kill the backend process when Electron app is closing
  if (backendProcess) {
    // On macOS, we need to make sure the Docker container is stopped
    spawn('docker', ['stop', 'chatbot-app'], {
      detached: true
    });
  }
});