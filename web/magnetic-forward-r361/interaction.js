/* View geometry only: no deformation law, solver, or magnetic-model fitting. */
'use strict';
const MagTouchInteraction = (() => {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  function project(p, width, height, yaw, pitch) {
    const scale = Math.min(width / 36, height / 28);
    return [width / 2 + (p[0] * Math.cos(yaw) - p[1] * Math.sin(yaw)) * scale,
      height * .58 + ((p[0] * Math.sin(yaw) + p[1] * Math.cos(yaw)) * Math.sin(pitch) - p[2] * Math.cos(pitch)) * scale];
  }
  function unprojectXY(pixel, width, height, yaw, pitch, physicalZ = 0) {
    if (Math.abs(Math.sin(pitch)) < .1) throw Error('视角过平，无法可靠拾取 XY');
    const scale = Math.min(width / 36, height / 28);
    const x = (pixel[0] - width / 2) / scale;
    const y = ((pixel[1] - height * .58) / scale + physicalZ * Math.cos(pitch)) / Math.sin(pitch);
    return [x * Math.cos(yaw) + y * Math.sin(yaw), -x * Math.sin(yaw) + y * Math.cos(yaw)];
  }
  function advance(q, target, dt = .1, speed = 2) {
    const delta = target.map((v, j) => v - q[j]);
    const gain = Math.min(1, speed * dt / Math.max(Math.hypot(...delta), 1e-12));
    return q.map((v, j) => v + gain * delta[j]);
  }
  return {clamp, project, unprojectXY, advance};
})();
if (typeof module !== 'undefined') module.exports = MagTouchInteraction;
