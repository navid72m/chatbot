// preload.js - Safe exposure of Electron APIs to renderer
const { contextBridge, ipcRenderer } = require('electron');

// Expose API functions to renderer process
contextBridge.exposeInMainWorld('api', {
  uploadDocument: (filePath) => ipcRenderer.invoke('upload-document', filePath),
  queryDocument: (query, model, temperature) => ipcRenderer.invoke('query-document', query, model, temperature),
  getModels: () => ipcRenderer.invoke('get-models'),
  selectFile: () => ipcRenderer.invoke('select-file'),
  openExternalLink: (url) => ipcRenderer.invoke('open-external-link', url)
});