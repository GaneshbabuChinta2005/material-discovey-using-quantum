import { app, BrowserWindow } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Check if running in dev environment
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: "Quantum Discovery Platform",
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true,
    },
    show: false, // Don't show until ready-to-show
  });

  // Remove the default Windows menu bar for a cleaner "app" look
  mainWindow.setMenuBarVisibility(false);

  if (isDev) {
    // Load Vite dev server
    mainWindow.loadURL('http://localhost:8080');
    // Open DevTools
    // mainWindow.webContents.openDevTools();
  } else {
    // Load built production files
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  // Show window gracefully when it's ready to avoid white flashing
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });
}

// App lifecycle
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
