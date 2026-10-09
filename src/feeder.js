/**
 * CodeFeeder —— 以代码文件为“饲料”。
 *
 * 当前实现（第一步先做行数）：
 *   - 递归扫描目录内的代码文件，按行数计量
 *   - 新增/增长的行数进入“食物能量池”，能量随时间衰减
 *   - 能量映射为 NCA 每秒步数：
 *       无数据(energy=0) → 0 步（冻结）
 *       少量数据         → 约 1 步/秒
 *       大量数据         → 封顶 30 步/秒
 *
 * 仅依赖 Node 内置模块，可独立在 Node 中测试。
 */
'use strict';

const fs = require('fs');
const path = require('path');

const CODE_EXTENSIONS = new Set([
  '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx',
  '.py', '.java', '.c', '.h', '.cc', '.cpp', '.hpp',
  '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.kts',
  '.scala', '.sh', '.bash', '.ps1', '.lua', '.r', '.m',
  '.vue', '.svelte', '.html', '.htm', '.css', '.scss', '.less',
  '.sql', '.md', '.json', '.yaml', '.yml', '.toml', '.xml'
]);

const IGNORE_DIRS = new Set([
  'node_modules', '.git', '.svn', '.hg', 'dist', 'build', 'out',
  '.vscode', '.idea', 'bin', 'obj', 'target', '.cache', 'coverage'
]);

const MAX_FILE_BYTES = 2 * 1024 * 1024;

function countLines(buf) {
  if (buf.length === 0) return 0;
  let n = 1;
  for (let i = 0; i < buf.length; i++) if (buf[i] === 10) n++;
  return n;
}

/**
 * 食物能量 → NCA 每秒步数。
 * 经验比例：约 1000 行 → 30 步/秒；约 33 行 → 1 步/秒；0 → 冻结。
 */
function energyToSps(energy) {
  if (!(energy > 0)) return 0;
  return Math.min(30, Math.max(1, Math.round(energy * 0.03)));
}

class CodeFeeder {
  constructor(dir) {
    this.dir = dir;
    this.files = new Map(); // path -> { mtimeMs, lines }
    this.energy = 0;
    this.totalLines = 0;
    this.decayFactor = 0.8; // 每秒保留 80%（半衰期约 3 秒）
  }

  setDir(dir) {
    this.dir = dir;
    this.files.clear();
  }

  feed(lines) {
    if (lines > 0) this.energy += lines;
  }

  /** 每秒衰减一次 */
  decay() {
    this.energy *= this.decayFactor;
    if (this.energy < 0.5) this.energy = 0;
  }

  get sps() {
    return energyToSps(this.energy);
  }

  async _walk(dir, out) {
    let entries;
    try {
      entries = await fs.promises.readdir(dir, { withFileTypes: true });
    } catch (e) {
      return;
    }
    for (const ent of entries) {
      const full = path.join(dir, ent.name);
      if (ent.isDirectory()) {
        if (IGNORE_DIRS.has(ent.name) || ent.name.startsWith('.')) continue;
        await this._walk(full, out);
      } else if (ent.isFile()) {
        const ext = path.extname(ent.name).toLowerCase();
        if (!CODE_EXTENSIONS.has(ext)) continue;
        try {
          const st = await fs.promises.stat(full);
          if (st.size > MAX_FILE_BYTES) continue;
          out.push({ full, mtimeMs: st.mtimeMs });
        } catch (e) { /* 忽略不可访问文件 */ }
      }
    }
  }

  /**
   * 扫描目录。返回本次新增的行数（新文件 + 文件增长部分），
   * 并把新增行数直接加入能量池。
   */
  async scan() {
    const found = [];
    if (this.dir) await this._walk(this.dir, found);

    const seen = new Set();
    let added = 0;
    let total = 0;

    for (const f of found) {
      seen.add(f.full);
      const prev = this.files.get(f.full);
      let lines = prev ? prev.lines : 0;
      if (!prev || prev.mtimeMs !== f.mtimeMs) {
        try {
          const buf = await fs.promises.readFile(f.full);
          lines = countLines(buf);
        } catch (e) {
          lines = prev ? prev.lines : 0;
        }
      }
      total += lines;
      this.files.set(f.full, { mtimeMs: f.mtimeMs, lines });
      if (lines > (prev ? prev.lines : 0)) added += lines - (prev ? prev.lines : 0);
    }

    // 已删除文件从映射中移除
    for (const p of Array.from(this.files.keys())) {
      if (!seen.has(p)) this.files.delete(p);
    }

    this.totalLines = total;
    if (added > 0) this.feed(added);
    return added;
  }

  /** 显式投喂若干文件（文件选择对话框用），返回合计行数并加入能量池 */
  async feedFiles(paths) {
    let added = 0;
    for (const p of paths || []) {
      try {
        const ext = path.extname(p).toLowerCase();
        if (!CODE_EXTENSIONS.has(ext)) continue;
        const buf = await fs.promises.readFile(p);
        added += countLines(buf);
      } catch (e) { /* 忽略 */ }
    }
    this.feed(added);
    return added;
  }
}

module.exports = { CodeFeeder, energyToSps, countLines, CODE_EXTENSIONS };
