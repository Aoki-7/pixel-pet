/**
 * NeuralCA —— 随机权重神经细胞自动机
 *
 * 架构（参考 Mordvintsev et al., "Growing Neural Cellular Automata"）：
 *   - H×W 网格，每格 C=16 通道：0..2=RGB，3=Alpha，4..15=隐藏状态
 *   - 感知：3×3 邻域上的 identity / SobelX / SobelY 三组卷积，共 3C=48 维
 *   - 更新：MLP 48 → 128(ReLU) → 16，逐格共享权重
 *   - 残差更新 + 每步随机只更新部分格（stochastic update）
 *   - 活性掩码：3×3 maxpool(alpha) > ALIVE_THRESHOLD 才参与/存活
 *
 * 不做训练：权重随机初始化。不同权重从同一颗种子长出完全不同的形态。
 * hatch() 会对随机候选权重做“存活力筛选”——在沙盒里跑 100 步，
 * 选出生长、稳定且有结构的候选，保证“最终形态不可预测但一定能涌现生命”。
 *
 * 该文件同时可在 Node（验证脚本）与浏览器（渲染端）中使用。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NeuralCA = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---- 可复现随机数 ----
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rng) {
    // Box–Muller
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const ALIVE_THRESHOLD = 0.1;
  const STATE_CLAMP = 2.0;
  const DELTA_CLAMP = 1.0;

  // 生长稳态（领地约束）：零状态格（尚未诞生的前沿格）离活体质心超过
  // TERRITORY_RATIO × 网格边长就不再被唤醒。扩张被硬性截断而无需杀伤细胞
  // （无碎裂风险）；领地内部的新生不受影响（破坏修复照常工作）。
  const TERRITORY_RATIO = 0.28;  // 领地半径/网格边长，对应可见填充率上限 ≈ π×0.28² ≈ 0.25

  class NeuralCA {
    constructor(options) {
      options = options || {};
      this.size = options.size || 28;
      this.H = this.W = this.size;
      this.C = options.channels || 16;
      this.HID = options.hidden || 128;
      this.P = this.C * 3; // 感知维度
      this.N = this.H * this.W;
      this.updateProb = options.updateProb != null ? options.updateProb : 0.7;
      this.territoryR2 = Math.pow(this.size * TERRITORY_RATIO, 2);

      this.state = new Float32Array(this.N * this.C);
      this.next = new Float32Array(this.N * this.C);
      this.alive = new Uint8Array(this.N);
      this.percept = new Float32Array(this.P);

      this.w1 = new Float32Array(this.P * this.HID);
      this.b1 = new Float32Array(this.HID);
      this.w2 = new Float32Array(this.HID * this.C);
      this.b2 = new Float32Array(this.C);
      this.hidden = new Float32Array(this.HID);

      this.seed = (options.seed != null ? options.seed : 1234) >>> 0;
      this.rng = mulberry32(this.seed);
      this.stepCount = 0;

      this.initWeights(this.rng, options.weightScale != null ? options.weightScale : 0.13);
      this.reset();
    }

    // ---- 权重：随机初始化（不训练） ----
    initWeights(rng, scale) {
      const std1 = Math.sqrt(2 / this.P);
      const std2 = Math.sqrt(1 / this.HID) * 0.9 * scale;
      for (let i = 0; i < this.w1.length; i++) this.w1[i] = gaussian(rng) * std1;
      for (let i = 0; i < this.b1.length; i++) this.b1[i] = 0;
      for (let i = 0; i < this.w2.length; i++) this.w2[i] = gaussian(rng) * std2;
      for (let i = 0; i < this.b2.length; i++) this.b2[i] = 0;
    }

    // ---- 种子：中心只有“几个”像素非零（十字形 5 格） ----
    reset() {
      this.state.fill(0);
      this.alive.fill(0);
      this.stepCount = 0;
      const c = (this.size - 1) >> 1;
      const seeds = [[c, c], [c - 1, c], [c + 1, c], [c, c - 1], [c, c + 1]];
      for (const [y, x] of seeds) {
        if (y < 0 || y >= this.H || x < 0 || x >= this.W) continue;
        const idx = y * this.W + x;
        const off = idx * this.C;
        this.state[off + 3] = 1.0; // alpha
        for (let ch = 0; ch < this.C; ch++) {
          if (ch === 3) continue;
          // 轻微随机，打破对称
          this.state[off + ch] = (this.rng() - 0.5) * 0.2;
        }
        this.alive[idx] = 1;
      }
    }

    idx(y, x) {
      return (y * this.W + x) * this.C;
    }

    // 3×3 maxpool(alpha) > 阈值 → 活性掩码
    computeAlive(src, out) {
      out.fill(0);
      for (let y = 0; y < this.H; y++) {
        for (let x = 0; x < this.W; x++) {
          let m = -Infinity;
          for (let dy = -1; dy <= 1; dy++) {
            const yy = y + dy;
            if (yy < 0 || yy >= this.H) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const xx = x + dx;
              if (xx < 0 || xx >= this.W) continue;
              const a = src[(yy * this.W + xx) * this.C + 3];
              if (a > m) m = a;
            }
          }
          if (m > ALIVE_THRESHOLD) out[y * this.W + x] = 1;
        }
      }
    }

    // 感知：identity / sobelX / sobelY（同步读取旧状态）
    perceive(src, y, x, p) {
      p.fill(0);
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= this.H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= this.W) continue;
          const off = (yy * this.W + xx) * this.C;
          // identity
          if (dy === 0 && dx === 0) {
            for (let c = 0; c < this.C; c++) p[c] += src[off + c];
          }
          // Sobel X（1/8 归一化）
          const kx = (dx === 0 ? 0 : dx === -1 ? -1 : 1) * (dy === 0 ? 2 : 1) / 8;
          if (kx !== 0) {
            for (let c = 0; c < this.C; c++) p[this.C + c] += kx * src[off + c];
          }
          // Sobel Y
          const ky = (dy === 0 ? 0 : dy === -1 ? -1 : 1) * (dx === 0 ? 2 : 1) / 8;
          if (ky !== 0) {
            for (let c = 0; c < this.C; c++) p[2 * this.C + c] += ky * src[off + c];
          }
        }
      }
    }

    step() {
      this.computeAlive(this.state, this.alive);
      // 生长稳态：领地之外的前沿格在苏醒前就被剔除掩码（限制扩张）。
      // 必须在更新循环前执行——一旦零态格在本步诞生就拦截不及了。
      this.enforceTerritory(this.state, this.alive);
      this.next.set(this.state); // 非更新格保持不变

      const p = this.percept;
      const h = this.hidden;

      for (let y = 0; y < this.H; y++) {
        for (let x = 0; x < this.W; x++) {
          const cell = y * this.W + x;
          if (!this.alive[cell]) continue;
          // 随机更新：这一帧该格“睡着”则跳过
          if (this.rng() > this.updateProb) continue;

          this.perceive(this.state, y, x, p);

          // 48 → 128，ReLU
          for (let j = 0; j < this.HID; j++) {
            let sum = this.b1[j];
            const wrow = j * this.P;
            for (let i = 0; i < this.P; i++) sum += this.w1[wrow + i] * p[i];
            h[j] = sum > 0 ? sum : 0;
          }

          const off = cell * this.C;
          // 128 → 16，残差
          for (let k = 0; k < this.C; k++) {
            let sum = this.b2[k];
            const wrow = k * this.HID;
            for (let j = 0; j < this.HID; j++) sum += this.w2[wrow + j] * h[j];
            if (sum > DELTA_CLAMP) sum = DELTA_CLAMP;
            else if (sum < -DELTA_CLAMP) sum = -DELTA_CLAMP;
            let v = this.state[off + k] + sum;
            if (v > STATE_CLAMP) v = STATE_CLAMP;
            else if (v < -STATE_CLAMP) v = -STATE_CLAMP;
            this.next[off + k] = v;
          }
        }
      }

      // 后置活性：死亡格清零（含全部隐藏通道）
      this.computeAlive(this.next, this.alive);
      for (let cell = 0; cell < this.N; cell++) {
        if (!this.alive[cell]) {
          const off = cell * this.C;
          for (let c = 0; c < this.C; c++) this.next[off + c] = 0;
        }
      }

      // 交换缓冲
      const tmp = this.state;
      this.state = this.next;
      this.next = tmp;
      this.stepCount++;
    }

    livingCount() {
      let n = 0;
      for (let i = 0; i < this.N; i++) if (this.alive[i]) n++;
      return n;
    }

    /**
     * 生长稳态（领地约束）：把离活体质心超过领地半径的“零态前沿格”
     * 从活性掩码中剔除——它们尚未诞生（状态全零），剔除即阻止其苏醒。
     * 实体格不受影响（质心缓慢漂移不会造成死亡螺旋）；
     * 领地内部的零态格（如破坏留下的孔洞）照常新生 → 修复可用。
     * 返回本步被抑制的前沿格数。冻结（不步进）时不发生。
     */
    enforceTerritory(buf, aliveMask) {
      let living = 0, sy = 0, sx = 0;
      for (let i = 0; i < this.N; i++) {
        if (!aliveMask[i]) continue;
        living++;
        sy += (i / this.W) | 0;
        sx += i % this.W;
      }
      if (living === 0) return 0;
      const cy = sy / living, cx = sx / living;
      let suppressed = 0;
      for (let i = 0; i < this.N; i++) {
        if (!aliveMask[i]) continue;
        const off = i * this.C;
        let solid = false;
        for (let c = 0; c < this.C; c++) {
          if (buf[off + c] !== 0) { solid = true; break; }
        }
        if (solid) continue; // 实体格不退出
        const y = (i / this.W) | 0, x = i % this.W;
        const d2 = (y - cy) * (y - cy) + (x - cx) * (x - cx);
        if (d2 > this.territoryR2) {
          aliveMask[i] = 0;
          suppressed++;
        }
      }
      return suppressed;
    }

    /**
     * 破坏：在 (cx,cy) 半径 radius 内清零所有格（含隐藏通道）。
     * NCA 的自我修复能力使周围活格能在后续步进中重新长回。
     * 需要 sps > 0（有能量）才能修复——冻结态下伤口不会愈合。
     */
    damage(cx, cy, radius) {
      const r2 = radius * radius;
      const y0 = Math.max(0, Math.floor(cy - radius));
      const y1 = Math.min(this.H - 1, Math.ceil(cy + radius));
      const x0 = Math.max(0, Math.floor(cx - radius));
      const x1 = Math.min(this.W - 1, Math.ceil(cx + radius));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const dx = x - cx, dy = y - cy;
          if (dx * dx + dy * dy <= r2) {
            const off = (y * this.W + x) * this.C;
            for (let c = 0; c < this.C; c++) this.state[off + c] = 0;
            this.alive[y * this.W + x] = 0;
          }
        }
      }
    }

    /**
     * 把状态网格映射为 RGBA 字节（透明背景）。
     * 双八度色相 + 亮度调制：用 12 个隐藏通道中前 11 个分别贡献
     * 色相粗调/细调和明度，使随机权重下涌现的色彩更丰富、更有层次。
     */
    toRGBA(out) {
      out = out || new Uint8ClampedArray(this.N * 4);
      const TAU = 6.283185307179586;
      for (let cell = 0; cell < this.N; cell++) {
        const s = cell * this.C;
        const o = cell * 4;
        if (!this.alive[cell]) {
          out[o] = out[o + 1] = out[o + 2] = out[o + 3] = 0;
          continue;
        }
        const a = this.state[s + 3];
        const aa = a <= 0.1 ? 0 : a >= 0.6 ? 1 : (a - 0.1) / 0.5;
        // 色相粗调（低频，决定大色块）
        const hue1 = (
          this.state[s + 4] * 1.0 +
          this.state[s + 5] * 0.8 +
          this.state[s + 6] * 1.2
        ) / 3;
        // 色相细调（高频，增加局部色彩变化）
        const hue2 = (
          this.state[s + 7] * 0.9 +
          this.state[s + 8] * 1.1 +
          this.state[s + 9] * 0.7 +
          this.state[s + 10] * 0.8
        ) / 4;
        // 亮度调制（使明暗有层次）
        const lum = (
          this.state[s + 11] * 0.6 +
          this.state[s + 12] * 0.5 +
          this.state[s + 13] * 0.9 +
          this.state[s + 14] * 0.4
        ) / 4;
        const h = 0.28 * hue1 + 0.08 * hue2;
        const l = 0.58 + 0.32 * Math.tanh(lum * 1.5); // 0.26..0.90
        out[o]     = Math.round((0.5 + 0.5 * Math.cos(TAU * (h + 0.00))) * l * 255);
        out[o + 1] = Math.round((0.5 + 0.5 * Math.cos(TAU * (h + 0.33))) * l * 255);
        out[o + 2] = Math.round((0.5 + 0.5 * Math.cos(TAU * (h + 0.67))) * l * 255);
        out[o + 3] = Math.round(aa * 255);
      }
      return out;
    }

    /**
     * 形状统计：四邻接连通分量、边界密度、质心。
     * 分量计数封顶 32（对“是否碎裂”的判断已足够）。
     */
    static shapeStats(ca) {
      const living = ca.livingCount();
      let components = 0;
      let boundary = 0;
      let sy = 0, sx = 0;
      let largest = 0;
      let minY = ca.H, maxY = -1, minX = ca.W, maxX = -1;
      const seen = new Uint8Array(ca.N);
      const stack = [];
      for (let i = 0; i < ca.N; i++) {
        if (!ca.alive[i] || seen[i]) continue;
        components++;
        let compSize = 0;
        stack.length = 0;
        stack.push(i);
        seen[i] = 1;
        while (stack.length) {
          const u = stack.pop();
          compSize++;
          const uy = (u / ca.W) | 0;
          const ux = u % ca.W;
          sy += uy; sx += ux;
          if (uy < minY) minY = uy;
          if (uy > maxY) maxY = uy;
          if (ux < minX) minX = ux;
          if (ux > maxX) maxX = ux;
          const nbr = [u - 1, u + 1, u - ca.W, u + ca.W];
          for (let q = 0; q < 4; q++) {
            const v = nbr[q];
            if (q === 0 && ux === 0) continue;
            if (q === 1 && ux === ca.W - 1) continue;
            if (q === 2 && uy === 0) continue;
            if (q === 3 && uy === ca.H - 1) continue;
            if (ca.alive[v]) {
              if (!seen[v]) { seen[v] = 1; stack.push(v); }
            } else {
              boundary++;
            }
          }
        }
        if (compSize > largest) largest = compSize;
        if (components > 32) break;
      }
      // 形状包围盒到网格边缘的最小距离（防止伸出屏幕）
      const edgeInset = living
        ? Math.min(minY, ca.H - 1 - maxY, minX, ca.W - 1 - maxX)
        : 0;
      return {
        living,
        fill: living / ca.N,
        components,
        largestShare: living ? largest / living : 0,
        edgeInset,
        boundaryDensity: living > 0 ? boundary / living : 0,
        cy: living ? sy / living : 0,
        cx: living ? sx / living : 0
      };
    }

    /**
     * 长期存活力评估：在给定权重上沙盒跑 steps 步（默认 200），
     * 在前 1/3、后 2/3、终点三个检查点记录形状，末段记录运动量。
     * 这样能区分“暂时好看但会碎裂/漂走”与“长期维持形体”的权重。
     */
    static assess(ca, steps) {
      steps = steps || 200;
      const seedCount = ca.livingCount();
      const checkpoints = [Math.round(steps / 3), Math.round(2 * steps / 3), steps];
      const traces = [];
      let cpIdx = 0;

      let prev = Float32Array.from(ca.state);
      let motion = 0;
      let motionSamples = 0;

      for (let t = 1; t <= steps; t++) {
        ca.step();
        if (t > steps - 6) {
          let d = 0;
          for (let i = 0; i < ca.state.length; i++) d += Math.abs(ca.state[i] - prev[i]);
          motion += d / ca.N;
          motionSamples++;
        }
        prev.set(ca.state);
        if (cpIdx < checkpoints.length && t === checkpoints[cpIdx]) {
          traces.push(Object.assign({ t }, NeuralCA.shapeStats(ca)));
          cpIdx++;
        }
      }

      motion = motionSamples ? motion / motionSamples : 0;
      const end = traces[traces.length - 1];
      const first = traces[0];
      const drift = Math.sqrt(
        Math.pow(end.cy - first.cy, 2) + Math.pow(end.cx - first.cx, 2)
      );

      return {
        seedCount, traces, motion, drift,
        // 便于旧调用方读取的末端快照
        living: end.living, fill: end.fill,
        components: end.components, boundaryDensity: end.boundaryDensity
      };
    }

    /**
     * 孵化：随机生成多组权重，长期沙盒评估，选出能持续涌现生命的一组。
     * 通过条件（全程三个检查点都要满足）：
     *   - 形体保持：填充率在 [0.03, 0.6]，连通分量 ≤ 6（不碎裂、不吞没网格）
     *   - 位置稳定：质心漂移 ≤ 8 格（不会游走撞墙）
     *   - 活性适中：末段运动量在 (0.002, 0.35)（不冻结、不剧烈震荡）
     * 返回的实例已重置到种子状态。
     */
    static hatch(options) {
      options = options || {};
      const attempts = options.attempts || 24;
      const steps = options.assessSteps || 200;
      let best = null;
      let bestScore = -Infinity;
      let bestInfo = null;

      for (let a = 0; a < attempts; a++) {
        const seed = (options.baseSeed != null ? options.baseSeed : 1) + a * 7919 + 1;
        const ca = new NeuralCA(Object.assign({}, options, { seed }));
        const info = NeuralCA.assess(ca, steps);

        const endTrace = info.traces[info.traces.length - 1];
        const shapeKept = info.traces.every(tr =>
          tr.fill <= 0.4 &&
          tr.components <= 2 &&
          tr.largestShare >= 0.85 &&
          tr.edgeInset >= 4
        );
        const pass =
          shapeKept &&
          endTrace.fill >= 0.06 &&
          info.drift <= 6 &&
          info.motion > 0.005 && info.motion < 0.3;

        // 打分：偏好全程适度大小、单一主体、形状边界丰富、漂移小
        let sizeScore = 0;
        for (const tr of info.traces) sizeScore += Math.max(0, 1 - Math.abs(tr.fill - 0.2) * 3);
        sizeScore /= info.traces.length;
        const end = info.traces[info.traces.length - 1];
        const score =
          (pass ? 10 : 0) +
          sizeScore * 2 +
          (end.components === 1 ? 2 : 0) +
          end.largestShare +
          Math.min(end.boundaryDensity, 0.8) +
          (info.drift <= 3 ? 1 : 0) -
          (end.living <= info.seedCount ? 5 : 0) -
          (info.traces.some(tr => tr.fill > 0.7) ? 5 : 0);

        if (score > bestScore) {
          bestScore = score;
          best = ca;
          bestInfo = info;
        }
        if (pass && a >= 3) break;
      }

      best.reset();
      best.hatchInfo = bestInfo;
      best.bestScore = bestScore;
      return best;
    }
  }

  NeuralCA.mulberry32 = mulberry32;
  return NeuralCA;
});
