/**
 * PixelPet 渲染端
 *
 * - Canvas 后台以 NCA 原生分辨率（40×40）维护 ImageData，最近邻放大到透明窗口
 * - "吃数据"驱动生长速度：主进程按代码行数算出每秒步数 sps，
 *   这里用时间累加器精确执行：sps=0 时完全不步进 → 形态冻结
 * - 左键短按 → 破坏 NCA（自我修复需能量）；左键长按/拖动 → 移动窗口
 * - 闲置（sps=0）时画面轻微呼吸闪烁；活跃时有辉光
 */
(function () {
  'use strict';

  const GRID = 40;
  const CELL = 5;
  const VIEW = GRID * CELL;

  const canvas = document.getElementById('stage');
  const ctx = canvas.getContext('2d');

  // 与 NCA 同分辨率的离屏画布：每步只更新这块小 ImageData
  const off = document.createElement('canvas');
  off.width = GRID;
  off.height = GRID;
  const offCtx = off.getContext('2d');
  const imgData = offCtx.createImageData(GRID, GRID);

  const hud = document.getElementById('hud');
  const loading = document.getElementById('loading');

  let ca = null;
  let sps = 0;
  let state = null;
  let hudVisible = false;

  // 随机生长：随机权重 + hatch 筛选 + 领地约束
  function randomHatch() {
    loading.style.display = 'flex';
    requestAnimationFrame(() => requestAnimationFrame(() => {
      ca = NeuralCA.hatch({
        size: GRID,
        channels: 16,
        hidden: 128,
        baseSeed: (Math.random() * 1e9) | 0
      });
      document.title = 'PixelPet [random]';
      loading.style.display = 'none';
      render();
      updateHud();
    }));
  }

  // 图片生长：加载训练好的权重，从种子自动长成目标形状
  function trainedHatch() {
    loading.style.display = 'flex';
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fetch('weights.json')
        .then(r => { if (!r.ok) throw new Error('no weights'); return r.json(); })
        .then(weights => {
          ca = new NeuralCA({ size: GRID, channels: 16, hidden: 128, seed: 42 });
          ca.loadWeights(weights);
          document.title = 'PixelPet [trained]';
          loading.style.display = 'none';
          render();
          updateHud();
        })
        .catch(() => {
          // 没有训练权重，回退到随机
          randomHatch();
        });
    }));
  }

  /**
   * 渲染：离屏 ImageData → 主 canvas（最近邻放大）
   * alpha 参数用于闲置呼吸；辉光通过先画一层半透明放大副本实现
   */
  function render(alpha) {
    alpha = alpha == null ? 1 : alpha;
    ca.toRGBA(imgData.data);
    offCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, VIEW, VIEW);
    // 辉光：半透明放大副本垫在下面
    ctx.globalAlpha = alpha * 0.28;
    ctx.drawImage(off, 0, 0, GRID, GRID, -1, -1, VIEW + 2, VIEW + 2);
    // 主体
    ctx.globalAlpha = alpha;
    ctx.drawImage(off, 0, 0, GRID, GRID, 0, 0, VIEW, VIEW);
    ctx.globalAlpha = 1;
  }

  function updateHud() {
    if (!hudVisible || !ca) return;
    const living = ca.livingCount();
    const foodName = state && state.foodDir ? state.foodDir.split(/[\\/]/).pop() : '-';
    hud.textContent =
      `step   ${ca.stepCount}\n` +
      `living ${living}\n` +
      `speed  ${sps} 步/秒${state && state.paused ? '（已暂停）' : ''}\n` +
      `energy ${state ? state.energy : 0} 行当量\n` +
      `code   ${state ? state.totalLines : 0} 行\n` +
      `food   ${foodName}\n` +
      `左键戳 / 右键菜单`;
  }

  // ---- 数据驱动的步进时钟 ----
  let acc = 0;
  let last = performance.now();
  let lastHudAt = 0;

  function frame(t) {
    const dt = Math.min(0.1, (t - last) / 1000);
    last = t;

    // 训练模式：自动从种子长到目标形状（80 步），长成后冻结
    if (ca && ca.trained && ca.stepCount < 80) {
      acc += 30 * dt;
      if (acc > 6) acc = 6;
      let stepped = 0;
      while (acc >= 1 && stepped < 6) {
        ca.step();
        acc -= 1;
        stepped++;
      }
      if (stepped > 0) render();
    } else if (ca && !ca.trained && sps > 0) {
      // 随机权重模式：数据驱动步进
      acc += sps * dt;
      if (acc > 6) acc = 6;
      let stepped = 0;
      while (acc >= 1 && stepped < 6) {
        ca.step();
        acc -= 1;
        stepped++;
      }
      if (stepped > 0) render();
    } else if (ca) {
      // 闲置呼吸：轻微透明度脉动，暗示"活着但冻结"
      render(PixelInteract.breathAlpha(t));
    }

    if (t - lastHudAt > 200) {
      updateHud();
      lastHudAt = t;
    }
    requestAnimationFrame(frame);
  }

  // ---- 与主进程交互 ----
  window.petAPI.onPetState(data => {
    state = data;
    sps = data.sps;
    updateHud();
  });

  window.petAPI.onPetCommand(cmd => {
    if (cmd === 'rehatch') randomHatch();
    else if (cmd === 'trained-hatch') trainedHatch();
    else if (cmd === 'toggle-hud') {
      hudVisible = !hudVisible;
      hud.style.display = hudVisible ? 'block' : 'none';
      updateHud();
    }
  });

  window.addEventListener('contextmenu', e => {
    e.preventDefault();
    window.petAPI.showMenu();
  });

  // ---- 左键：短按=戳破坏，长按/拖动=移动窗口 ----
  // 阈值判定与呼吸曲线在 interact.js（可单元测试）
  let md = null;                         // mousedown 上下文
  const DAMAGE_RADIUS = PixelInteract.DAMAGE_RADIUS;

  canvas.addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    md = { x: e.screenX, y: e.screenY, t: performance.now(), dragStarted: false };
  });

  window.addEventListener('mousemove', e => {
    if (!md || md.dragStarted) return;
    if (PixelInteract.shouldStartDrag(
          e.screenX - md.x, e.screenY - md.y, performance.now() - md.t)) {
      md.dragStarted = true;
      window.petAPI.startDrag();
    }
  });

  window.addEventListener('mouseup', e => {
    if (e.button !== 0 || !md) return;
    if (md.dragStarted) {
      window.petAPI.endDrag();
    } else {
      // 短按 → 在 NCA 上戳一个洞，周围活格会重新长回（需要能量）
      const rect = canvas.getBoundingClientRect();
      const cx = Math.floor((e.clientX - rect.left) / CELL);
      const cy = Math.floor((e.clientY - rect.top) / CELL);
      if (ca && cx >= 0 && cx < GRID && cy >= 0 && cy < GRID) {
        ca.damage(cx, cy, DAMAGE_RADIUS);
        if (ca.trained) ca.stepCount = 60;  // 触发自动修复（再长 20 步）
        render();
      }
    }
    md = null;
  });

  window.addEventListener('mouseleave', () => {
    if (md && md.dragStarted) window.petAPI.endDrag();
    md = null;
  });

  trainedHatch();
  requestAnimationFrame(frame);
})();
