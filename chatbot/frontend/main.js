const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const isDev = require('electron-is-dev');

let mainWindow;
let serverProcess;

// Function to start the FastAPI server
function startFastAPIServer() {
    return new Promise((resolve, reject) => {
        // Path to the Python executable and server script
        const pythonPath = isDev ? 'python' : path.join(process.resourcesPath, 'python', 'python.exe');
        const serverScript = path.join(__dirname, '..', 'backend', 'mcp_server.py');

        // Start the server process
        serverProcess = spawn(pythonPath, [serverScript], {
            stdio: 'pipe',
            shell: true
        });

        // Handle server output
        serverProcess.stdout.on('data', (data) => {
            console.log(`Server: ${data}`);
            // Check if server is ready
            if (data.toString().includes('Application startup complete')) {
                resolve();
            }
        });

        // Handle server errors
        serverProcess.stderr.on('data', (data) => {
            console.error(`Server Error: ${data}`);
        });

        // Handle server process errors
        serverProcess.on('error', (error) => {
            console.error('Failed to start server:', error);
            reject(error);
        });

        // Set a timeout for server startup
        setTimeout(() => {
            reject(new Error('Server startup timeout'));
        }, 30000); // 30 seconds timeout
    });
}

// Function to create the main window
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    // Load the app
    mainWindow.loadURL(
        isDev
            ? 'http://localhost:3000'
            : `file://${path.join(__dirname, '../build/index.html')}`
    );

    // Open DevTools in development
    if (isDev) {
        mainWindow.webContents.openDevTools();
    }

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// Initialize the app
app.whenReady().then(async () => {
    try {
        // Start the FastAPI server first
        console.log('Starting FastAPI server...');
        await startFastAPIServer();
        console.log('FastAPI server started successfully');

        // Then create the window
        createWindow();
    } catch (error) {
        console.error('Failed to start application:', error);
        app.quit();
    }
});

// Quit when all windows are closed
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// Handle app quit
app.on('before-quit', () => {
    // Kill the server process when the app is quitting
    if (serverProcess) {
        serverProcess.kill();
    }
});

// Handle macOS re-activation
app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
}); 