const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // add APIs if needed
});
