const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  getBackendUrl: () => ipcRenderer.invoke("backend:getUrl"),
  onBackendReady: (cb) => {
    ipcRenderer.removeAllListeners("backend:ready");
    ipcRenderer.on("backend:ready", (_evt, url) => cb(url));
  },
});
