/**
 * 最小原型无头验证（node test/validate.js）
 *
 * 对应文档三条验证标准：
 *   1. 从几个像素点开始：初始只有中心几个点非零，其余透明
 *   2. 形态自行涌现：100 步后出现事先没有“画”过的、有结构的图案，
 *      且 150 步尺度上形体保持（不碎裂、不漂走、不吞没网格）
 *   3. 数据驱动速度：能量→每秒步数映射（0 冻结 / 少量≈1 / 大量=30）
 * 另加：不同随机权重形态不同 + 性能基准 + PNG 快照。
 */
'use strict';

const fs = require('fs');
const path = require('path');
const NeuralCA = require('../src/nca.js');
const { energyToSps, countLines } = require('../src/feeder.js');
const { writePNGSnapshot } = require('./png.js');

let failures = 0;
function check(name, cond, detail) {
  const tag = cond ? 'PASS' : 'FAIL';
  if (!cond) failures++;
  console.log(`[${tag}] ${name}${detail ? ' — ' + detail : ''}`);
}

const SNAP_DIR = path.join(__dirname, 'snapshots');
fs.mkdirSync(SNAP_DIR, { recursive: true });

async function main() {
  console.log('== 标准1：从几个像素点开始 ==');
  const ca = NeuralCA.hatch({ size: 40, channels: 16, hidden: 128, baseSeed: 42 });
  const seedCount0 = ca.livingCount();
  let nonzeroCells = 0;
  for (let cell = 0; cell < ca.N; cell++) {
    for (let c = 0; c < ca.C; c++) {
      if (ca.state[cell * ca.C + c] !== 0) { nonzeroCells++; break; }
    }
  }
  check('初始非零格只有少数几个', nonzeroCells <= 9, `非零格=${nonzeroCells}`);
  check('初始活性格数=5（十字种子）', seedCount0 === 5, `活性格=${seedCount0}`);
  const c0 = ((ca.size - 1) >> 1);
  let centered = true;
  for (let cell = 0; cell < ca.N; cell++) {
    const y = (cell / ca.W) | 0, x = cell % ca.W;
    let nz = false;
    for (let ch = 0; ch < ca.C; ch++) if (ca.state[cell * ca.C + ch] !== 0) { nz = true; break; }
    if (nz && Math.abs(y - c0) > 1 && Math.abs(x - c0) > 1) centered = false;
  }
  check('非零格集中在中心', centered);

  console.log('\n== 标准2：形态自行涌现（100 步）且长期保持（200 步）==');
  writePNGSnapshot(path.join(SNAP_DIR, 'step000.png'), ca, 8);
  const t0 = process.hrtime.bigint();
  let living100 = 0;
  for (let t = 1; t <= 200; t++) {
    ca.step();
    if (t === 50) writePNGSnapshot(path.join(SNAP_DIR, 'step050.png'), ca, 8);
    if (t === 100) {
      living100 = ca.livingCount();
      writePNGSnapshot(path.join(SNAP_DIR, 'step100.png'), ca, 8);
    }
    if (t === 200) writePNGSnapshot(path.join(SNAP_DIR, 'step200.png'), ca, 8);
  }
  const t1 = process.hrtime.bigint();

  // 长期结构指标：对另一具同样经过 hatch 筛选的同卵个体沙盒评估
  const probe = NeuralCA.hatch({ size: 40, channels: 16, hidden: 128, baseSeed: 42 });
  const metrics = NeuralCA.assess(probe, 200);
  check('100 步后活性格明显多于种子', living100 > seedCount0 + 5, `living=${living100}, seed=${seedCount0}`);
  check('图案未吞没整个网格（fill<40%）', living100 / ca.N < 0.4, `fill=${(living100 / ca.N).toFixed(2)}`);
  check('主体为单/双连通结构（非噪点）', metrics.components <= 2, `components@200=${metrics.components}`);
  check('存在形状边界', metrics.boundaryDensity > 0.1, `boundaryDensity=${metrics.boundaryDensity.toFixed(2)}`);
  check('末段仍在运动（非冻结）', metrics.motion > 0.005, `motion=${metrics.motion.toFixed(4)}`);
  check('200 步时已长成（fill≥6%）', metrics.fill >= 0.06, `fill=${metrics.fill.toFixed(2)}`);
  check('全程三个检查点形体保持（fill≤40%、分量≤2、主体占比≥85%、边缘安全≥4）',
    metrics.traces.every(tr =>
      tr.fill <= 0.4 && tr.components <= 2 && tr.largestShare >= 0.85 && tr.edgeInset >= 4),
    metrics.traces.map(tr =>
      `fill=${tr.fill.toFixed(2)},c=${tr.components},ls=${tr.largestShare.toFixed(2)},ei=${tr.edgeInset}`
    ).join(' | '));
  check('质心不游走（漂移≤6格）', metrics.drift <= 6, `drift=${metrics.drift.toFixed(1)}`);

  // 颜色多样性（不是单色块）
  const rgba = ca.toRGBA();
  const colors = new Set();
  for (let i = 0; i < ca.N; i++) {
    if (rgba[i * 4 + 3] > 0) colors.add(`${rgba[i * 4]},${rgba[i * 4 + 1]},${rgba[i * 4 + 2]}`);
  }
  check('颜色种类丰富', colors.size > 20, `colors=${colors.size}`);

  console.log('\n== 随机性：不同权重 → 不同形态 ==');
  const m1 = NeuralCA.assess(NeuralCA.hatch({ size: 40, baseSeed: 7 }), 200);
  const m2 = NeuralCA.assess(NeuralCA.hatch({ size: 40, baseSeed: 999 }), 200);
  check('两组权重 200 步后大小/结构不同',
    Math.abs(m1.living - m2.living) > 3 || m1.components !== m2.components,
    `living ${m1.living} vs ${m2.living}, comp ${m1.components} vs ${m2.components}`);

  console.log('\n== 孵化筛选的稳定性（任意随机基种都能孵出可存活形态）==');
  const trials = [1, 100, 2024, 55555, 88888];
  let viable = 0;
  for (const s of trials) {
    const pet = NeuralCA.hatch({ size: 40, baseSeed: s });
    const m = NeuralCA.assess(pet, 200);
    if (m.traces.every(tr =>
      tr.fill <= 0.4 && tr.components <= 2 &&
      tr.largestShare >= 0.85 && tr.edgeInset >= 4) &&
      m.fill >= 0.06 && m.drift <= 6 && m.motion > 0.005) viable++;
  }
  check('5 个随机基种全部孵出可存活形态', viable === 5, `${viable}/5`);

  console.log('\n== 标准3：数据驱动速度（能量→sps）==');
  check('无数据 → 0 步/秒（冻结）', energyToSps(0) === 0, `sps=${energyToSps(0)}`);
  check('少量数据（约 33 行）→ 1 步/秒', energyToSps(33) === 1, `sps=${energyToSps(33)}`);
  check('100 行 → 3 步/秒', energyToSps(100) === 3, `sps=${energyToSps(100)}`);
  check('1000 行 → 30 步/秒封顶', energyToSps(1000) === 30, `sps=${energyToSps(1000)}`);
  check('更多数据仍封顶 30', energyToSps(99999) === 30, `sps=${energyToSps(99999)}`);

  // 行数计量
  check('行数统计正确', countLines(Buffer.from('a\nb\nc')) === 3 && countLines(Buffer.from('')) === 0);

  console.log('\n== 性能 ==');
  const ms = Number(t1 - t0) / 1e6;
  console.log(`200 步耗时 ${ms.toFixed(1)} ms，平均 ${(ms / 200).toFixed(2)} ms/步`);
  check('单步耗时满足 30 步/秒预算（<33ms）', ms / 200 < 33);

  console.log(`\n快照已写入: ${SNAP_DIR}`);
  console.log(failures === 0 ? '\n全部验证通过 ✅' : `\n${failures} 项验证失败 ❌`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(2); });
