/**
 * 预加载脚本：在隔离世界里只暴露最小 IPC 接口。
 */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('petAPI', {
  // 主进程 → 渲染端：投喂状态（sps/能量/行数）
  onPetState(callback) {
    ipcRenderer.on('pet-state', (_event, data) => callback(data));
  },
  // 主进程 → 渲染端：菜单命令（重新孵化 / HUD 切换）
  onPetCommand(callback) {
    ipcRenderer.on('pet-command', (_event, cmd) => callback(cmd));
  },
  // 渲染端 → 主进程：请求弹出右键菜单
  showMenu() {
    ipcRenderer.invoke('show-context-menu');
  },
  // 渲染端 → 主进程：JS 拖拽
  startDrag() {
    ipcRenderer.invoke('drag-start');
  },
  endDrag() {
    ipcRenderer.invoke('drag-end');
  }
});
