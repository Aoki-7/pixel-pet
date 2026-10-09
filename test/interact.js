/**
 * 修复交互与动画功能的单元测试（node test/interact.js）
 *
 * 覆盖：
 *   1. 破坏 damage()：圆形清零、范围外不受影响、网格边缘截断
 *   2. 自我修复需要能量：冻结（不步进）伤口保持；有能量（步进）则长回且形体连贯
 *   3. 色彩映射：明暗分层、alpha 映射规则
 *   4. 呼吸动画曲线：范围与周期（interact.js）
 *   5. 点击/拖动判定阈值（interact.js）
 * 快照：damaged.png / repaired.png
 */
'use strict';

const fs = require('fs');
const path = require('path');
const NeuralCA = require('../src/nca.js');
const { shouldStartDrag, breathAlpha, CLICK_THRESHOLD, CLICK_TIME, DAMAGE_RADIUS } = require('../src/interact.js');
const { writePNGSnapshot } = require('./png.js');

let failures = 0;
function check(name, cond, detail) {
  const tag = cond ? 'PASS' : 'FAIL';
  if (!cond) failures++;
  console.log(`[${tag}] ${name}${detail ? ' — ' + detail : ''}`);
}

function approx(a, b, eps) { return Math.abs(a - b) <= (eps || 1e-9); }

const SNAP_DIR = path.join(__dirname, 'snapshots');
fs.mkdirSync(SNAP_DIR, { recursive: true });

// 孵化一具并长到稳定期，供破坏/修复测试共用
function grownPet() {
  const ca = NeuralCA.hatch({ size: 40, channels: 16, hidden: 128, baseSeed: 42 });
  for (let t = 0; t < 150; t++) ca.step();
  return ca;
}

function stateSnapshot(ca) {
  return Float32Array.from(ca.state);
}

function statesEqualArrays(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function main() {
  // ---- 1. damage()：圆形清零 ----
  console.log('== 破坏 damage() ==');
  const ca = grownPet();
  const stats = NeuralCA.shapeStats(ca);
  const cy = Math.max(DAMAGE_RADIUS, Math.min(ca.H - 1 - DAMAGE_RADIUS, Math.round(stats.cy)));
  const cx = Math.max(DAMAGE_RADIUS, Math.min(ca.W - 1 - DAMAGE_RADIUS, Math.round(stats.cx)));
  const before = stateSnapshot(ca);
  const livingBefore = ca.livingCount();

  ca.damage(cx, cy, DAMAGE_RADIUS);

  let holeClean = true;
  let outsideUntouched = true;
  for (let y = 0; y < ca.H; y++) {
    for (let x = 0; x < ca.W; x++) {
      const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      const off = (y * ca.W + x) * ca.C;
      if (d2 <= DAMAGE_RADIUS * DAMAGE_RADIUS) {
        for (let c = 0; c < ca.C; c++) {
          if (ca.state[off + c] !== 0) { holeClean = false; break; }
        }
        if (ca.alive[y * ca.W + x] !== 0) holeClean = false;
      } else if (d2 > (DAMAGE_RADIUS + 1) * (DAMAGE_RADIUS + 1)) {
        for (let c = 0; c < ca.C; c++) {
          if (ca.state[off + c] !== before[off + c]) { outsideUntouched = false; break; }
        }
      }
      if (!holeClean || !outsideUntouched) break;
    }
    if (!holeClean || !outsideUntouched) break;
  }
  check('半径内所有通道清零且活性掩码清除', holeClean);
  check('半径外状态不受影响', outsideUntouched);
  check('破坏后活性格减少', ca.livingCount() < livingBefore - 5,
    `${livingBefore} → ${ca.livingCount()}`);
  writePNGSnapshot(path.join(SNAP_DIR, 'damaged.png'), ca, 5);

  // 网格边缘截断不崩溃
  let edgeOk = true;
  try {
    ca.damage(0, 0, DAMAGE_RADIUS);
    ca.damage(ca.W - 1, ca.H - 1, DAMAGE_RADIUS);
    ca.damage(-5, -5, DAMAGE_RADIUS);
  } catch (e) { edgeOk = false; }
  check('网格边缘/界外中心截断不崩溃', edgeOk);

  // ---- 2. 自我修复需要能量 ----
  console.log('\n== 自我修复（需要能量）==');
  // “冻结不修复”由渲染端时钟保证（sps=0 → 不调用 step，见 validate.js 的
  // energyToSps(0)===0）。引擎侧不变式：damage 是瞬时操作，不推进步数、
  // 不触发任何更新，伤口是否愈合完全取决于后续是否步进。
  const frozen = grownPet();
  const fStats = NeuralCA.shapeStats(frozen);
  const stepsBefore = frozen.stepCount;
  frozen.damage(Math.round(fStats.cx), Math.round(fStats.cy), DAMAGE_RADIUS);
  check('damage 是瞬时操作：不推进步数、不触发更新',
    frozen.stepCount === stepsBefore, `stepCount ${stepsBefore} → ${frozen.stepCount}`);

  // 有能量（步进）：伤口长回
  const healer = grownPet();
  const hStats = NeuralCA.shapeStats(healer);
  const hy = Math.max(DAMAGE_RADIUS, Math.min(healer.H - 1 - DAMAGE_RADIUS, Math.round(hStats.cy)));
  const hx = Math.max(DAMAGE_RADIUS, Math.min(healer.W - 1 - DAMAGE_RADIUS, Math.round(hStats.cx)));
  const livingH0 = healer.livingCount();
  healer.damage(hx, hy, DAMAGE_RADIUS);
  const livingH1 = healer.livingCount();
  const hole = livingH0 - livingH1;
  for (let t = 0; t < 150; t++) healer.step();
  const livingH2 = healer.livingCount();
  check('有能量：150 步内伤口明显长回',
    livingH2 > livingH1 + hole * 0.5,
    `破坏 ${livingH0}→${livingH1}（- ${hole}），修复后 ${livingH2}`);
  const hEnd = NeuralCA.shapeStats(healer);
  check('修复后形体连贯（分量≤2）', hEnd.components <= 2, `components=${hEnd.components}`);
  writePNGSnapshot(path.join(SNAP_DIR, 'repaired.png'), healer, 5);

  // ---- 2b. 生长稳态：持续投喂（长时间步进）不会铺满网格 ----
  console.log('\n== 生长稳态（持续步进不无限扩张）==');
  const pet2 = NeuralCA.hatch({ size: 40, channels: 16, hidden: 128, baseSeed: 42 });
  let maxFill = 0;
  let coherent = true;
  let stillMetabolizing = false;
  const traj = [];
  let snapMid = null;
  for (let t = 1; t <= 1500; t++) {
    pet2.step();
    const f = pet2.livingCount() / pet2.N;
    if (f > maxFill) maxFill = f;
    if (t % 150 === 0) traj.push(f.toFixed(2));
    if (t === 1200) snapMid = stateSnapshot(pet2);
    if (t >= 300 && t % 10 === 0 && NeuralCA.shapeStats(pet2).components > 2) coherent = false;
  }
  stillMetabolizing = !statesEqualArrays(snapMid, pet2.state);
  check('长期步进填充率有上限（max ≤ 0.45）', maxFill <= 0.45,
    `maxFill=${maxFill.toFixed(3)}, 轨迹=${traj.join('→')}`);
  check('末段仍在承载带内（fill ≤ 0.45）', pet2.livingCount() / pet2.N <= 0.45,
    `fill@1500=${(pet2.livingCount() / pet2.N).toFixed(3)}`);
  check('全程形态连贯（t≥300 分量≤2）', coherent);
  check('稳态下边缘仍代谢（状态持续变化）', stillMetabolizing);
  writePNGSnapshot(path.join(SNAP_DIR, 'saturated.png'), pet2, 5);

  // 饱和态下破坏质心附近 → 领地内部的伤口仍可修复
  const pre = pet2.livingCount();
  const satStats = NeuralCA.shapeStats(pet2);
  pet2.damage(Math.round(satStats.cx), Math.round(satStats.cy), DAMAGE_RADIUS);
  const afterDmg = pet2.livingCount();
  for (let t = 0; t < 250; t++) pet2.step();
  check('饱和态：领地内伤口仍可修复',
    pet2.livingCount() > afterDmg + (pre - afterDmg) * 0.5,
    `饱和 ${pre} → 破坏 ${afterDmg} → 修复 ${pet2.livingCount()}`);

  // ---- 3. 色彩映射 ----
  console.log('\n== 色彩/动画（引擎侧）==');
  const painter = grownPet();
  const rgba = painter.toRGBA();
  const colors = new Set();
  let lumMin = Infinity, lumMax = -Infinity;
  let alphaHiOk = true, alphaLoOk = true;
  let hiCount = 0, loCount = 0;
  for (let cell = 0; cell < painter.N; cell++) {
    const s = cell * painter.C;
    const a = painter.state[s + 3];
    const o = cell * 4;
    if (a >= 0.6) {
      hiCount++;
      if (rgba[o + 3] !== 255) alphaHiOk = false;
    }
    if (a <= 0.1) {
      loCount++;
      if (rgba[o + 3] !== 0) alphaLoOk = false;
    }
    if (rgba[o + 3] > 0) {
      colors.add(`${rgba[o]},${rgba[o + 1]},${rgba[o + 2]}`);
      const lum = (rgba[o] + rgba[o + 1] + rgba[o + 2]) / 3 / 255;
      if (lum < lumMin) lumMin = lum;
      if (lum > lumMax) lumMax = lum;
    }
  }
  check('颜色种类丰富（≥20 种）', colors.size >= 20, `colors=${colors.size}`);
  check('明暗有层次（亮度跨度≥0.15）', lumMax - lumMin >= 0.15,
    `lum ${lumMin.toFixed(2)}..${lumMax.toFixed(2)}`);
  check('alpha 映射：自身 alpha≥0.6 → 255', alphaHiOk, `样本=${hiCount}`);
  check('alpha 映射：自身 alpha≤0.1 → 0', alphaLoOk, `样本=${loCount}`);

  // ---- 4. 呼吸动画曲线 ----
  console.log('\n== 呼吸动画曲线（interact.js）==');
  let bMin = Infinity, bMax = -Infinity;
  for (let t = 0; t <= 5000; t += 5) {
    const v = breathAlpha(t);
    if (v < bMin) bMin = v;
    if (v > bMax) bMax = v;
  }
  check('呼吸范围 0.56..1.0', approx(bMin, 0.56, 0.01) && approx(bMax, 1.0, 0.01),
    `min=${bMin.toFixed(3)}, max=${bMax.toFixed(3)}`);
  const period = 2 * Math.PI / 0.0015;
  check('呼吸是周期函数（周期≈4189ms）',
    approx(breathAlpha(100), breathAlpha(100 + period), 1e-9),
    `T=${period.toFixed(0)}ms`);
  check('呼吸中点 t=0 → 0.78', approx(breathAlpha(0), 0.78, 1e-9));

  // ---- 5. 点击/拖动判定 ----
  console.log('\n== 点击/拖动判定（interact.js）==');
  check('微小位移+短按 → 不拖动', !shouldStartDrag(0, 0, 0));
  check('位移恰在阈值（5px）→ 不拖动', !shouldStartDrag(3, 4, 0));
  check('位移超阈值 → 拖动', shouldStartDrag(6, 0, 0));
  check('对角位移超阈值 → 拖动', shouldStartDrag(4, 4, 0));
  check('长按超时 → 拖动', shouldStartDrag(0, 0, CLICK_TIME + 1));
  check('长按恰好阈值 → 不拖动', !shouldStartDrag(0, 0, CLICK_TIME));
  check('阈值常量符合设计（5px / 250ms / 半径3格）',
    CLICK_THRESHOLD === 5 && CLICK_TIME === 250 && DAMAGE_RADIUS === 3);

  console.log(failures === 0 ? '\n全部单元测试通过 ✅' : `\n${failures} 项单元测试失败 ❌`);
  process.exit(failures === 0 ? 0 : 1);
}

main();
