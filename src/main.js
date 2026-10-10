/**
 * PixelPet 主进程
 *
 * 窗口参数（参考 clawd 等开源像素桌宠）：
 *   透明、无边框、置顶、跳过任务栏、不抢焦点、不可调整大小
 *
 * 数据投喂：
 *   - 默认扫描项目下的 food/ 目录（递归，代码文件按行数计量）
 *   - 右键菜单可切换饲料目录、直接投喂代码文件、模拟投喂
 *   - 新增行数进入能量池（每秒衰减），能量映射为 NCA 每秒步数并推给渲染端
 */
'use strict';

const path = require('path');
const fs = require('fs');
const { app, BrowserWindow, Menu, dialog, ipcMain, shell, screen } = require('electron');
const { CodeFeeder } = require('./feeder.js');

const GRID = 40;
const CELL = 5;                 // 每个 NCA 像素在屏幕上占 5px
const WIN_SIZE = GRID * CELL;   // 200

const DEFAULT_FOOD_DIR = path.join(__dirname, '..', 'food');

let win = null;
let paused = false;
const feeder = new CodeFeeder(DEFAULT_FOOD_DIR);

function createWindow() {
  const { workArea } = screen.getPrimaryDisplay();
  win = new BrowserWindow({
    width: WIN_SIZE,
    height: WIN_SIZE,
    x: workArea.x + workArea.width - WIN_SIZE - 24,
    y: workArea.y + workArea.height - WIN_SIZE - 24,
    transparent: true,
    frame: false,
    resizable: false,
    movable: true,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    hasShadow: false,
    thickFrame: false,
    focusable: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  // 更高层级置顶；所有工作区可见（macOS）
  win.setAlwaysOnTop(true, 'screen-saver');
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  win.loadFile(path.join(__dirname, 'index.html'));
}

function broadcastState() {
  if (!win || win.isDestroyed()) return;
  win.webContents.send('pet-state', {
    sps: paused ? 0 : feeder.sps,
    energy: Math.round(feeder.energy),
    totalLines: feeder.totalLines,
    paused,
    foodDir: feeder.dir
  });
}

function buildContextMenu() {
  const template = [
    {
      label: `喂 100 行代码（模拟）`,
      click: () => { feeder.feed(100); broadcastState(); }
    },
    {
      label: `喂 1000 行代码（模拟）`,
      click: () => { feeder.feed(1000); broadcastState(); }
    },
    { type: 'separator' },
    {
      label: '投喂代码文件…',
      click: async () => {
        const res = await dialog.showOpenDialog(win, {
          title: '选择要投喂的代码文件',
          properties: ['openFile', 'multiSelections']
        });
        if (!res.canceled) {
          await feeder.feedFiles(res.filePaths);
          broadcastState();
        }
      }
    },
    {
      label: '选择代码目录作为饲料来源…',
      click: async () => {
        const res = await dialog.showOpenDialog(win, {
          title: '选择代码目录',
          properties: ['openDirectory']
        });
        if (!res.canceled && res.filePaths[0]) {
          feeder.setDir(res.filePaths[0]);
          await feeder.scan(); // 目录里现有代码当作第一餐
          broadcastState();
        }
      }
    },
    {
      label: '打开饲料目录',
      click: () => shell.openPath(feeder.dir)
    },
    { type: 'separator' },
    {
      label: paused ? '继续生长' : '暂停（冻结形态）',
      click: () => { paused = !paused; broadcastState(); }
    },
    {
      label: '随机生长（随机新生命）',
      click: () => win.webContents.send('pet-command', 'rehatch')
    },
    {
      label: '图片生长（训练形态）',
      click: () => win.webContents.send('pet-command', 'trained-hatch')
    },
    {
      label: '显示/隐藏状态信息',
      click: () => win.webContents.send('pet-command', 'toggle-hud')
    },
    { type: 'separator' },
    { label: '退出', click: () => app.quit() }
  ];
  return Menu.buildFromTemplate(template);
}

ipcMain.handle('show-context-menu', () => {
  buildContextMenu().popup({ window: win });
});

// ---- JS 拖拽 ----
let dragTimer = null;
let dragStart = null;

ipcMain.handle('drag-start', () => {
  const pos = screen.getCursorScreenPoint();
  const [winX, winY] = win.getPosition();
  dragStart = { mx: pos.x, my: pos.y, wx: winX, wy: winY };
  if (dragTimer) clearInterval(dragTimer);
  dragTimer = setInterval(() => {
    if (!dragStart) return;
    const p = screen.getCursorScreenPoint();
    win.setPosition(dragStart.wx + (p.x - dragStart.mx), dragStart.wy + (p.y - dragStart.my));
  }, 16);
});

ipcMain.handle('drag-end', () => {
  dragStart = null;
  if (dragTimer) { clearInterval(dragTimer); dragTimer = null; }
});

app.whenReady().then(async () => {
  fs.mkdirSync(DEFAULT_FOOD_DIR, { recursive: true });
  createWindow();

  await feeder.scan();
  broadcastState();

  // 每 2 秒扫描代码目录（增量）
  setInterval(() => { feeder.scan().then(broadcastState); }, 2000);
  // 每秒能量衰减
  setInterval(() => { feeder.decay(); broadcastState(); }, 1000);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  win = null;  // 防止定时器在退出时访问已销毁的窗口
});
