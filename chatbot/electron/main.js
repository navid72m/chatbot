import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import fs from 'fs';
import http from 'http';
import https from 'https';

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// URLs for our services
const OLLAMA_URL = 'http://127.0.0.1:11434';
const BACKEND_URL = 'http://127.0.0.1:8000';

// Model we want to ensure is downloaded
const REQUIRED_MODEL = 'deepseek-r1';

// Store subprocess references
let ollamaProcess = null;
let backendProcess = null;
let mainWindow = null;
let splashWindow = null;

// Flag to track if the app is ready to show
let servicesReady = false;

// Create a splash screen to show during startup
function createSplashScreen() {
  splashWindow = new BrowserWindow({
    width: 500,
    height: 300,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  // Load a simple HTML file for the splash screen
  splashWindow.loadFile(path.join(__dirname, 'splash.html'));

  // Log status updates that will be shown on the splash screen
  global.splashLog = (message) => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.webContents.send('status-update', message);
    }
    console.log(message);
  };
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false // Allow loading local files
    },
    show: false // Don't show the window until services are ready
  });

  // Set up error event handlers to debug loading issues
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Page failed to load:', errorCode, errorDescription);
    // Display error in window for better debugging
    mainWindow.webContents.executeJavaScript(`
      document.body.innerHTML = '<h1>Error Loading Page</h1><p>Error: ${errorDescription} (${errorCode})</p>';
    `).catch(console.error);
  });

  // Add console log interceptor
  mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
    console.log(`Frontend console (${level}): ${message}`);
  });

  // Load the app once both services are ready
  // Load the app once both services are ready
  try {
    if (process.env.NODE_ENV === 'development') {
      // Development mode - load from dev server
      console.log('Loading from development server...');
      await mainWindow.loadURL('http://localhost:3000');
      mainWindow.webContents.openDevTools();
    } else {
      // Production mode - load from build
      // Check multiple possible paths for the frontend
      const possiblePaths = [
        path.join(__dirname, '../frontend/dist/index.html'),
        path.join(__dirname, '../frontend/build/index.html'),
        path.join(__dirname, '../dist/index.html'),
        path.join(__dirname, 'dist/index.html'),
        path.join(__dirname, '../frontend/public/index.html')
      ];
      
      // Log which paths exist for debugging
      console.log('Checking frontend paths:');
      possiblePaths.forEach(p => {
        console.log(`Path ${p} exists: ${fs.existsSync(p)}`);
      });
      
      // Try to load from any existing path
      let loaded = false;
      for (const p of possiblePaths) {
        if (fs.existsSync(p)) {
          console.log(`Loading from: ${p}`);
          try {
            await mainWindow.loadFile(p);
            loaded = true;
            break;
          } catch (err) {
            console.error(`Error loading ${p}:`, err);
          }
        }
      }
      
      // If no paths worked, use the fallback
      if (!loaded) {
        console.log('No frontend paths found, using fallback HTML');
        mainWindow.loadURL(`data:text/html,
          <html>
            <head>
              <title>Document Chat</title>
              <style>
                body { font-family: system-ui; padding: 2rem; max-width: 800px; margin: 0 auto; }
                h1 { color: #4f46e5; }
                .card { border: 1px solid #ddd; padding: 1rem; margin: 1rem 0; border-radius: 0.5rem; }
                button { background: #4f46e5; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; }
                pre { background: #f1f5f9; padding: 1rem; overflow-x: auto; }
              </style>
            </head>
            <body>
              <h1>Document Chat</h1>
              <div class="card">
                <h2>Frontend Files Not Found</h2>
                <p>The application could not find the frontend files.</p>
                <p>Here are some things you can try:</p>
                <ol>
                  <li>Ensure the frontend is built: Run <code>npm run build</code> in the frontend directory</li>
                  <li>Start the development server: Run <code>npm start</code> in the frontend directory</li>
                  <li>Check the file paths in <code>main.js</code></li>
                </ol>
                <button onclick="window.location.reload()">Reload</button>
              </div>
              <div class="card">
                <h2>Where the app looked for files:</h2>
                <pre>${possiblePaths.map(p => `${p}: ${fs.existsSync(p) ? 'Exists' : 'Not found'}`).join('\n')}</pre>
              </div>
            </body>
          </html>
        `);
      }
    }
  } catch (error) {
    console.error('Error loading frontend:', error);
    mainWindow.loadURL(`data:text/html,
      <html>
        <body>
          <h1>Error Loading Frontend</h1>
          <p>${error.message}</p>
          <p>Please ensure the frontend is built or the development server is running.</p>
          <pre>${error.stack}</pre>
        </body>
      </html>
    `);
  }

  // Setup window events
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Only show window when services are ready
  if (servicesReady) {
    showMainWindow();
  }
}

// Show the main window and close the splash screen
function showMainWindow() {
  if (mainWindow) {
    mainWindow.show();
    
    // Give a small delay to ensure smooth transition
    setTimeout(() => {
      if (splashWindow && !splashWindow.isDestroyed()) {
        splashWindow.close();
        splashWindow = null;
      }
    }, 500);
  }
}

// Utility function to wait with a timeout
function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Make HTTP request using Node.js built-in modules
function makeHttpRequest(url, method = 'GET', data = null) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port,
      path: `${urlObj.pathname}${urlObj.search}`,
      method: method,
      headers: {}
    };

    // Add headers for POST requests with JSON data
    if (method === 'POST' && data) {
      options.headers['Content-Type'] = 'application/json';
      options.headers['Content-Length'] = Buffer.byteLength(JSON.stringify(data));
    }

    const req = http.request(options, (res) => {
      let responseData = '';
      
      res.on('data', (chunk) => {
        responseData += chunk;
      });
      
      res.on('end', () => {
        try {
          // Try to parse as JSON if possible
          if (responseData && responseData.trim().startsWith('{')) {
            resolve(JSON.parse(responseData));
          } else {
            resolve(responseData);
          }
        } catch (e) {
          // Return raw response if JSON parsing fails
          resolve(responseData);
        }
      });
    });
    
    req.on('error', (error) => {
      reject(error);
    });
    
    // Send the data for POST requests
    if (method === 'POST' && data) {
      req.write(JSON.stringify(data));
    }
    
    req.end();
  });
}

// Check if a service is running
async function isServiceRunning(url, maxAttempts = 5) {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      await makeHttpRequest(url);
      return true;
    } catch (error) {
      if (attempt < maxAttempts - 1) {
        await wait(1000); // Wait 1 second before trying again
      }
    }
  }
  return false;
}

// Poll a service until it's ready
async function waitForService(url, serviceName, maxAttempts = 500) {
  global.splashLog(`⏳ Waiting for ${serviceName} to be ready...`);
  
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      await makeHttpRequest(url);
      global.splashLog(`✅ ${serviceName} ready at ${url}`);
      return true;
    } catch (error) {
      if (attempt % 10 === 0 && attempt > 0) {
        global.splashLog(`Still waiting for ${serviceName}... (${attempt}/${maxAttempts})`);
      }
      await wait(1000); // Wait 1 second between attempts
    }
  }
  
  global.splashLog(`❌ Timeout waiting for ${serviceName} to be ready`);
  return false;
}

// Check if the required model is downloaded
async function isModelDownloaded(modelName) {
  try {
    const response = await makeHttpRequest(`${OLLAMA_URL}/api/tags`);
    if (response && response.models) {
      return response.models.some(model => model.name === modelName);
    }
    return false;
  } catch (error) {
    console.error('Error checking if model is downloaded:', error.message);
    return false;
  }
}

// Download the required model
async function downloadModel(modelName) {
  global.splashLog(`⏳ Downloading model ${modelName}...`);
  
  try {
    // Use Ollama's pull API to download the model
    const response = await makeHttpRequest(
      `${OLLAMA_URL}/api/pull`, 
      'POST', 
      { name: modelName, stream: false }
    );

    global.splashLog(`✅ Model ${modelName} downloaded successfully`);
    return true;
  } catch (error) {
    global.splashLog(`❌ Error downloading model ${modelName}: ${error.message}`);
    return false;
  }
}

// Start Ollama with the correct 'serve' command
async function startOllama() {
  const basePath = app.getAppPath();
  console.log('App base path:', basePath);
  
  // Try multiple possible paths relative to the app path
  const possiblePaths = [
    path.join(basePath, 'electron', 'bin', 'ollama'),
    path.join(basePath, 'bin', 'ollama'),
    path.join(basePath, '..', 'bin', 'ollama'),
    // Add system paths as fallbacks
    '/usr/local/bin/ollama',  // macOS/Linux
    'C:\\Program Files\\ollama\\ollama.exe'  // Windows
  ];
  
  // Log all paths we're checking
  console.log('Checking for Ollama at these locations:');
  possiblePaths.forEach(p => console.log(`- ${p}: ${fs.existsSync(p) ? 'Exists' : 'Not found'}`));
  
  // Find the first path that exists
  let ollamaPath = null;
  for (const p of possiblePaths) {
    if (fs.existsSync(p) && fs.statSync(p).isFile()) {
      ollamaPath = p;
      console.log(`Found Ollama at: ${ollamaPath}`);
      break;
    }
  }
  
  if (!ollamaPath) {
    global.splashLog('❌ Ollama binary not found in any expected location');
    return false;
  }
  
  // Wait for Ollama to be ready
  return await waitForService(OLLAMA_URL, 'Ollama');
}

// Start the backend server
async function startBackend() {
  const backendPath = path.join(__dirname, 'bin', 'backend');
  global.splashLog(`🚀 Starting backend from: ${backendPath}`);
  
  // Check if backend is already running
  if (await isServiceRunning(BACKEND_URL)) {
    global.splashLog('✅ Backend is already running');
    return true;
  }
  
  // Start backend process
  backendProcess = spawn(backendPath, [], {
    stdio: ['ignore', 'pipe', 'pipe']
  });
  
  // Handle stdout
  backendProcess.stdout.on('data', (data) => {
    const output = data.toString().trim();
    if (output) console.log(`🟢 Backend: ${output}`);
  });
  
  // Handle stderr
  backendProcess.stderr.on('data', (data) => {
    const output = data.toString().trim();
    if (output) console.log(`🔴 Backend Error: ${output}`);
  });
  
  // Handle process exit
  backendProcess.on('exit', (code) => {
    console.log(`Backend process exited with code ${code}`);
    backendProcess = null;
  });
  
  // Wait for backend to be ready
  return await waitForService(BACKEND_URL, 'Backend');
}

// Initialize all services and then show the app
async function initializeApp() {
  try {
    // Start Ollama first
    global.splashLog('Starting Ollama service...');
    const ollamaReady = await startOllama();
    if (!ollamaReady) {
      throw new Error('Failed to start Ollama');
    }
    
    // Check if the required model is downloaded
    global.splashLog(`Checking if ${REQUIRED_MODEL} is downloaded...`);
    const modelExists = await isModelDownloaded(REQUIRED_MODEL);
    
    // If the model is not downloaded, download it
    if (!modelExists) {
      global.splashLog(`Model ${REQUIRED_MODEL} not found, starting download...`);
      const modelDownloaded = await downloadModel(REQUIRED_MODEL);
      if (!modelDownloaded) {
        global.splashLog(`Warning: Failed to download ${REQUIRED_MODEL}. App will still start, but model may need to be downloaded manually.`);
      }
    } else {
      global.splashLog(`✅ Model ${REQUIRED_MODEL} already exists`);
    }
    
    // Once Ollama is ready, start the backend
    const backendReady = await startBackend();
    if (!backendReady) {
      throw new Error('Failed to start backend');
    }
    
    // Both services are ready
    global.splashLog('✅ All services are running and ready');
    servicesReady = true;
    
    // If the window is already created but hidden, show it now
    if (mainWindow) {
      showMainWindow();
    }
    mainWindow.once('ready-to-show', () => {
      showMainWindow();
    });
    
  } catch (error) {
    global.splashLog(`❌ Error initializing services: ${error.message}`);
    
    // Show an error dialog
    dialog.showErrorBox(
      'Service Initialization Error',
      `Failed to initialize services: ${error.message}\n\nThe app may not function correctly.`
    );
    
    // Still attempt to create window with warning message
    if (mainWindow) {
      showMainWindow();
    }
  }
}

// Create a simple splash screen HTML file if it doesn't exist
function ensureSplashScreenExists() {
  const splashPath = path.join(__dirname, 'splash.html');
  
  // If splash screen HTML doesn't exist, create it
  if (!fs.existsSync(splashPath)) {
    const splashHtml = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <title>Starting Application...</title>
      <style>
        body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          background-color: rgba(30, 30, 30, 0.95);
          color: #fff;
          margin: 0;
          padding: 20px;
          border-radius: 12px;
          height: 100vh;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          overflow: hidden;
        }
        h2 {
          margin-bottom: 20px;
          font-weight: 400;
        }
        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid rgba(255, 255, 255, 0.1);
          border-radius: 50%;
          border-top-color: #3b82f6;
          animation: spin 1s ease-in-out infinite;
          margin-bottom: 30px;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .status-log {
          background-color: rgba(0, 0, 0, 0.3);
          border-radius: 6px;
          padding: 10px;
          width: 90%;
          height: 120px;
          overflow-y: auto;
          font-family: monospace;
          font-size: 12px;
          margin-top: 10px;
        }
        .status-log p {
          margin: 4px 0;
          white-space: pre-wrap;
          word-break: break-word;
        }
      </style>
    </head>
    <body>
      <h2>Starting Application...</h2>
      <div class="spinner"></div>
      <div class="status-log" id="status-log"></div>
      
      <script>
        const { ipcRenderer } = require('electron');
        const logContainer = document.getElementById('status-log');
        
        ipcRenderer.on('status-update', (event, message) => {
          const p = document.createElement('p');
          p.textContent = message;
          logContainer.appendChild(p);
          logContainer.scrollTop = logContainer.scrollHeight;
        });
      </script>
    </body>
    </html>`;
    
    fs.writeFileSync(splashPath, splashHtml);
  }
}

// Add IPC communication for frontend to get backend URL
ipcMain.handle('get-backend-url', async () => {
  return BACKEND_URL;
});

// Electron app events
app.on('ready', async () => {
  // Ensure splash screen HTML exists
  ensureSplashScreenExists();
  
  // Show splash screen first
  createSplashScreen();
  
  // Wait a moment to show splash screen before heavy operations
  await wait(500);
  
  // Create main window (hidden until services are ready)
  await createWindow();
  
  // Initialize services
  await initializeApp();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Handle app exit - clean up subprocesses
app.on('before-quit', () => {
  console.log('Shutting down services...');
  
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
  
  if (ollamaProcess) {
    ollamaProcess.kill();
    ollamaProcess = null;
  }
});