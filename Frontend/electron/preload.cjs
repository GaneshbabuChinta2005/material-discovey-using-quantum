const { contextBridge, ipcRenderer } = require('electron');

// Expose safe APIs to the React renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // Example: Listen for a message from main process
  onMessage: (callback) => ipcRenderer.on('message', callback),
  // Example: Send a message to main process
  sendMessage: (data) => ipcRenderer.send('message', data),
});
