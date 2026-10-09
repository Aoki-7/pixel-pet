/**
 * 交互与动画的纯函数部分——无 DOM 依赖，可被单元测试无头验证。
 *
 * - shouldStartDrag：按下后是否已转入窗口拖动（位移/时长超过阈值）
 * - breathAlpha：闲置（sps=0）时的呼吸透明度曲线
 * - DAMAGE_RADIUS：左键戳破坏的半径（NCA 格数）
 *
 * 浏览器端挂到 window.PixelInteract；Node 端 module.exports。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PixelInteract = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const CLICK_THRESHOLD = 5;   // 像素：位移超过即判定为拖动
  const CLICK_TIME = 250;      // 毫秒：按住超过即判定为拖动
  const DAMAGE_RADIUS = 3;     // NCA 格数：戳破坏半径

  /**
   * 判断当前按下状态是否应转入窗口拖动。
   * @param {number} dx 按下后鼠标位移 x（像素）
   * @param {number} dy 按下后鼠标位移 y（像素）
   * @param {number} elapsedMs 按下经过的毫秒数
   */
  function shouldStartDrag(dx, dy, elapsedMs) {
    return Math.sqrt(dx * dx + dy * dy) > CLICK_THRESHOLD || elapsedMs > CLICK_TIME;
  }

  /**
   * 闲置呼吸透明度：0.56..1.0 正弦脉动，周期约 4.2 秒。
   * @param {number} tMs performance.now() 毫秒
   */
  function breathAlpha(tMs) {
    return 0.78 + 0.22 * Math.sin(tMs * 0.0015);
  }

  return {
    shouldStartDrag,
    breathAlpha,
    CLICK_THRESHOLD,
    CLICK_TIME,
    DAMAGE_RADIUS
  };
});
