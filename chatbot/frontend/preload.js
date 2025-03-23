// preload.js - Safe exposure of Electron APIs to renderer with quantization support
const { contextBridge, ipcRenderer } = require('electron');

// Expose API functions to renderer process
contextBridge.exposeInMainWorld('api', {
  uploadDocument: (filePath) => ipcRenderer.invoke('upload-document', filePath),
  queryDocument: (query, model, temperature, quantization) => 
    ipcRenderer.invoke('query-document', query, model, temperature, quantization),
  getModels: () => ipcRenderer.invoke('get-models'),
  selectFile: () => ipcRenderer.invoke('select-file'),
  openExternalLink: (url) => ipcRenderer.invoke('open-external-link', url)
});