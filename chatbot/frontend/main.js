// main.js - Electron main process with quantization support
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');
const { spawn } = require('child_process');
const Store = require('electron-store');

// Initialize config store
const store = new Store();

// Backend API URL
const API_URL = 'http://127.0.0.1:8000';

// Keep a global reference of the window object
let mainWindow;
let backendProcess = null;

// Start Python backend if not already running
async function startBackend() {
  try {
    // Check if the backend is already running
    try {
      await axios.get(`${API_URL}`);
      console.log('Backend is already running');
      return true;
    } catch (error) {
      // If the request fails, the backend is not running
      console.log('Backend is not running, starting it now...');
    }

    // Find Python executable
    // First try python3, then python
    let pythonCommand = 'python3';
    
    try {
      const testProcess = spawn(pythonCommand, ['--version']);
      await new Promise((resolve, reject) => {
        testProcess.on('close', (code) => {
          if (code !== 0) {
            reject();
          } else {
            resolve();
          }
        });
      });
    } catch (error) {
      pythonCommand = 'python';
    }

    // Path to the backend script
    const backendPath = path.join(app.getAppPath(), 'backend', 'app.py');
    
    // Start the backend process
    backendProcess = spawn(pythonCommand, [backendPath], {
      cwd: path.join(app.getAppPath(), 'backend')
    });

    backendProcess.stdout.on('data', (data) => {
      console.log(`Backend stdout: ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
      console.error(`Backend stderr: ${data}`);
    });

    // Wait for the backend to start (simple polling)
    let attempts = 0;
    while (attempts < 30) {
      try {
        await axios.get(`${API_URL}`);
        console.log('Backend started successfully');
        return true;
      } catch (error) {
        attempts++;
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    throw new Error('Failed to start backend after 30 seconds');
  } catch (error) {
    console.error('Error starting backend:', error);
    dialog.showErrorBox(
      'Backend Error',
      'Failed to start backend. Make sure Python and all dependencies are installed.'
    );
    return false;
  }
}

function createWindow() {
  // Get the absolute path to the preload script
  const preloadPath = path.join(app.getAppPath(), 'preload.js');
  
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: preloadPath
    }
  });

  // Log the preload script path for debugging
  console.log('Preload script path:', preloadPath);
  
  // Check if preload script exists
  if (!fs.existsSync(preloadPath)) {
    console.error('Preload script not found at:', preloadPath);
    dialog.showErrorBox(
      'Error',
      'Preload script not found. Please check the file path.'
    );
    app.quit();
    return;
  }

  mainWindow.loadFile('index.html');
  
  // Open DevTools in development
  mainWindow.webContents.openDevTools();

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.on('ready', async () => {
  const backendStarted = await startBackend();
  if (backendStarted) {
    createWindow();
  } else {
    app.quit();
  }
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', function () {
  if (mainWindow === null) createWindow();
});

app.on('will-quit', () => {
  // Kill the backend process if it exists
  if (backendProcess !== null) {
    backendProcess.kill();
  }
});

// IPC handlers for communication with the renderer process
ipcMain.handle('upload-document', async (event, filePath) => {
  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));
    
    const response = await axios.post(`${API_URL}/upload`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error uploading document:', error);
    return {
      success: false,
      error: error.response?.data?.detail || error.message
    };
  }
});

// Updated to include quantization parameter
ipcMain.handle('query-document', async (event, query, model, temperature, quantization) => {
  try {
    const response = await axios.post(`${API_URL}/query`, {
      query,
      model,
      temperature,
      quantization
    });
    
    return response.data;
  } catch (error) {
    console.error('Error querying document:', error);
    return {
      success: false,
      error: error.response?.data?.detail || error.message
    };
  }
});

ipcMain.handle('get-models', async () => {
  try {
    const response = await axios.get(`${API_URL}/models`);
    return response.data;
  } catch (error) {
    console.error('Error fetching models:', error);
    return {
      success: false,
      error: error.response?.data?.detail || error.message
    };
  }
});

ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Documents', extensions: ['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'] }
    ]
  });
  
  if (result.canceled) {
    return { canceled: true };
  }
  
  return { filePath: result.filePaths[0] };
});

ipcMain.handle('open-external-link', async (event, url) => {
  await shell.openExternal(url);
  return true;
});