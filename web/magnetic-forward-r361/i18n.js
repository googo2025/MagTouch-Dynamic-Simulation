/* Presentation-only translation. Scientific data, model state and exports are unchanged. */
(() => {
  'use strict';
  const entries = [
    ['MagTouch · 磁响应正向实验室', 'MagTouch · Forward Magnetic Response Lab'],
    ['最终完整证据：', 'Final evidence:'],
    ['仿真启发真实优化与双语论文', 'Simulation-guided real-data refinement and bilingual paper'],
    ['R461双语网页版 ·', 'R461 bilingual web release ·'],
    ['R430中文旧版', 'Legacy R430 Chinese version'],
    ['逐模型结果', 'Model-by-model results'],
    ['修订论文', 'Revised paper'],
    ['运行与数据', 'Run and data'],
    ['MagTouch 仿真研究平台 ·', 'MagTouch simulation research platform ·'],
    ['论文与最新训练结果', 'Paper and latest training results'],
    ['数据与复现包', 'Data and reproduction package'],
    ['此前论文 GitHub 仓库', 'Previous paper on GitHub'],
    ['磁响应正向实验室', 'Forward Magnetic Response Lab'],
    ['动作与历史 → 五节点 15 维磁响应 · 离线研究版', 'Motion and history → five-node, 15-channel magnetic response · offline research build'],
    ['源器件训练 · 目标器件仅评估', 'Source device for training · target device for evaluation only'],
    ['无硬件连接 · 不改动实时采集程序', 'No hardware connection · live acquisition software is unchanged'],
    ['实验 / 交互模式', 'Experiment / interaction mode'],
    ['对比模型', 'Comparison model'],
    ['联合历史 · 基线模型', 'Joint-history baseline'],
    ['仅当前位置模型', 'Current-position-only model'],
    ['分轴历史 · 新研究模型', 'Axis-separated history · research model'],
    ['自由按压', 'Free interaction'],
    ['播放', 'Play'],
    ['暂停', 'Pause'],
    ['从起点重播', 'Replay from start'],
    ['导出当前序列 CSV', 'Export current sequence as CSV'],
    ['复杂轨迹与稀疏磁场研究 →', 'Complex paths and sparse-field research →'],
    ['导出文本', 'Exported text'],
    ['若浏览器没有弹出下载，请在下方文本框内全选复制，保存为 UTF-8 文本文件。此处提供完整内容，不是抽样预览。', 'If no download appears, select and copy the entire text below and save it as a UTF-8 file. This is the complete export, not a sample.'],
    ['触头动作与固定传感器', 'Indenter motion and fixed sensors'],
    ['旋转视角', 'Rotate view'],
    ['操作触头', 'Move indenter'],
    ['松开鼠标后保持按压', 'Keep pressed after mouse release'],
    ['视角复位', 'Reset view'],
    ['三维图操作方式', '3D interaction mode'],
    ['三维触头交互画布，方向键移动，PageDown 压入，PageUp 抬起，空格切换按压，Escape 撤出', 'Interactive 3D indenter: arrow keys move; PageDown presses; PageUp lifts; Space toggles contact; Escape withdraws'],
    ['网格为名义 Z=0 平面。节点 XY 采用配置，层间距仅示意。箭头表示 ΔB 方向，长度独立缩放；不是磁力线。磁响应由经验模型计算；真实有限元形变在下方独立查看。', 'The grid is the nominal Z=0 plane. Sensor XY positions follow the configuration; layer spacing is schematic. Arrows indicate ΔB direction with independently scaled lengths, not magnetic field lines. An empirical model computes the magnetic response; actual finite-element deformation results can be viewed separately below.'],
    ['回放时刻', 'Playback time'],
    ['模型输出', 'Model output'],
    ['压入 Z / mm', 'Indentation Z / mm'],
    ['动作', 'Motion'],
    ['自由调整目标 XYZ', 'Adjust target XYZ freely'],
    ['圆周运动', 'Circular motion'],
    ['八方向往返', 'Eight-direction reciprocation'],
    ['垂直压入 / 撤出', 'Vertical indentation / withdrawal'],
    ['压深', 'Indentation depth'],
    ['压入并保持', 'Press and hold'],
    ['撤出', 'Withdraw'],
    ['加深 0.1 mm', 'Deeper by 0.1 mm'],
    ['减浅 0.1 mm', 'Shallower by 0.1 mm'],
    ['左键按住并拖动 XY；Shift＋上下拖动或滚轮改变压深。操作自动推进状态，以最高 2 mm/s 跟随目标；不瞬移、不重置材料响应历史。此处不控制实物设备。', 'Hold and drag with the left mouse button to move in XY. Shift-drag vertically or use the wheel to change indentation depth. The state advances automatically and follows the target at up to 2 mm/s; it does not jump or reset material-response history. This interface does not control physical hardware.'],
    ['主评价段 15 通道 RMSE / µT', 'Primary evaluation: 15-channel RMSE / µT'],
    ['主评价段归一化误差', 'Primary evaluation: normalized error'],
    ['预测 ΔB / µT', 'Predicted ΔB / µT'],
    ['实测', 'Measured'],
    ['模型预测', 'Model prediction'],
    ['磁场为 ΔB，单位 µT；不平滑、不做事后时间对齐', 'Field shown as ΔB in µT; no smoothing or post-hoc time alignment'],
    ['虚拟数据有没有帮助？看真实留出实验', 'Does virtual data help? Evidence from held-out real records'],
    ['器件', 'Device'],
    ['真实训练', 'Real-data training'],
    ['虚拟训练', 'Virtual-data training'],
    ['真实＋虚拟', 'Real + virtual'],
    ['表中为 XYZ 平均误差 / mm，整条记录等权、3 个随机种子等权。比较使用相同结构的实验性因果读出器，不是当前 UI 的完整重建算法；不含接触门控。', 'Table values are mean XYZ error in mm, equally weighted by complete record and by three random seeds. The comparison uses experimental causal readouts of the same architecture, not the full reconstruction algorithm used by the live UI; contact gating is excluded.'],
    ['数据划分、有效范围与物理模型接口', 'Data splits, validity range and physical-model interface'],
    ['训练：24 条源器件标准动作＋3 条明确 5 mm 触头的静态标定记录。选参：10 条源器件整条记录。评估：30 条留出记录，包含独立目标器件与完整三维路径。所有目标器件与 waypoint 路径均未用于拟合或选参。', 'Training: 24 standard source-device motions plus three static calibration records with a confirmed 5 mm indenter. Selection: ten complete source-device records. Evaluation: 30 held-out records including an independent target device and full 3D paths. Neither target-device nor waypoint paths were used for fitting or selection.'],
    ['主误差统计：有效五传感器观测、名义压深 0.4–3.0 mm，不按逆解是否输出筛选。实际训练坐标包络约为 XY ±5.5 mm、Z 0.4–2.3 mm；包络外明确标记为外推，包络内也不代表每个位置都有实测覆盖。', 'Primary error statistics use valid five-sensor observations at nominal indentation depths of 0.4–3.0 mm, without selecting frames by inverse-model output. The training-coordinate envelope is approximately XY ±5.5 mm and Z 0.4–2.3 mm. Outside this envelope is explicitly extrapolation; positions inside it are not necessarily measured.'],
    ['旧记录使用原有确认后时标，新记录使用发送时标，均采用日志 EST 动作参考。常量磁场偏置与 XYZ 平移是不同诊断，主曲线不使用任何事后配准。已检查的历史记录属于本次拟合留出，不重新称为盲测。', 'Older records use their existing post-acknowledgement timestamps; newer records use send timestamps. Both use the logged EST motion reference. Constant magnetic-field bias and XYZ translation are separate diagnostics; primary curves have no post-hoc registration. Previously inspected historical records are held out from this fit, not reclassified as blind tests.'],
    ['此模型为实测驱动代理，尚未通过高保真力学/磁场验证。多点同时接触不由两个单点响应相加生成。物理接口可读取逐帧多触点与 15 通道响应，但无独立验证的导入数据只允许查看。', 'This is an empirical surrogate driven by measured data; it has not passed high-fidelity mechanics or magnetics validation. Simultaneous multi-point contact is not generated by summing two single-point responses. The physical interface accepts per-frame multiple contacts and 15-channel responses, but imported data without independent validation are view-only.'],
    ['本地载入物理模型导出 JSON', 'Load physical-model JSON locally'],
    ['文件只在浏览器内解析，不上传', 'Parsed only in your browser; not uploaded'],
    ['有限元形变 · 真实求解结果查看', 'Finite-element deformation · solver-result viewer'],
    ['载入求解器导出的表面网格与逐帧三维位移，可旋转查看形变、未变形轮廓与位移幅值。这里不在浏览器内求解有限元，也不以手工凹陷动画代替形变；当前未附带经验证的成品形变结果。', 'Load a solver-exported surface mesh and frame-by-frame 3D displacements to inspect deformation, the undeformed outline and displacement magnitude. The browser does not solve finite elements or substitute a hand-drawn depression animation. No validated finished-device deformation result is included.'],
    ['等待导入有限元结果', 'Waiting for a finite-element result'],
    ['节点坐标 ＋ 表面三角形 ＋ 逐帧位移', 'Node coordinates + surface triangles + frame-by-frame displacement'],
    ['不会用经验磁场模型反造硅胶形变', 'The empirical magnetic model does not fabricate silicone deformation'],
    ['有限元三维位移结果，可拖动旋转', 'Finite-element 3D displacement result; drag to rotate'],
    ['导入形变 JSON（仅本地读取）', 'Import deformation JSON (local read only)'],
    ['独立结果时间轴，不与上方任意拖动联动。后续接入物理求解器或验证后的形变代理，才能驱动同一动作的实时形变。', 'This result has its own timeline and does not follow arbitrary interaction above. A physical solver or validated deformation surrogate is required for live deformation driven by the same motion.'],
    ['显示位移倍率', 'Displacement display scale'],
    ['1× 真实比例', '1× true scale'],
    ['5× 放大显示（非真实比例）', '5× magnified (not true scale)'],
    ['10× 放大显示（非真实比例）', '10× magnified (not true scale)'],
    ['显示未变形轮廓', 'Show undeformed outline'],
    ['播放形变', 'Play deformation'],
    ['暂停形变', 'Pause deformation'],
    ['形变时刻', 'Deformation time'],
    ['位移幅值 |u|', 'Displacement magnitude |u|'],
    ['形变数据接口与验证边界', 'Deformation data interface and validation limits'],
    ['接口：', 'Interface:'],
    ['。位移单位 mm，逐节点提供完整 ux、uy、uz；颜色使用原始位移幅值，不随显示倍率改变。必须携带求解器、几何、材料与边界条件信息。网页只校验文件结构，不替代网格收敛、接触与实测验证；导入声明不自动升级为“已验证”。', '. Displacements are in mm, with ux, uy and uz at each node. Colors use unscaled displacement magnitude. Solver, geometry, material and boundary-condition provenance is required. Structural file checks do not replace mesh-convergence, contact or experimental validation; an imported claim is not automatically “validated.”'],
    ['尚未把历史近似网格误作当前成品几何；形变文件也不会自动进入虚拟训练集或改变上方 15 通道模型。', 'An older approximate mesh is not represented as current finished-device geometry. Deformation files do not automatically enter the virtual training set or change the 15-channel model above.'],
    ['数据和模型均随页面本地加载。自由模式导出的是带来源标识的研究性虚拟数据，不能当成独立实测证据。', 'Data and models load locally with this page. Free-mode exports are research virtual data with provenance labels, not independent experimental evidence.'],
    ['圆轨迹', 'Circle'], ['八方向', 'Eight directions'], ['定点网格', 'Point grid'],
    ['XY–Z 复合', 'Combined XY–Z'], ['复杂三维路径', 'Complex 3D path'],
    ['交互生成 · 5 mm 单触头', 'Interactive generation · 5 mm single indenter'],
    ['源器件', 'Source device'], ['目标器件', 'Target device'],
    ['当前 50% 混合权重的虚拟增强未优于仅真实数据训练，因此这些虚拟数据未进入正式模型训练。', 'At the current 50% mixing weight, virtual augmentation did not outperform real-data-only training; these virtual samples were not used for the production model.'],
    ['复杂虚拟轨迹与小比例增强已训练并回测，但未获得稳定真实定位收益；仍为研究数据，不替换正式重建模型。', 'Complex virtual paths and low-weight augmentation were trained and replayed, but did not yield consistent gains in real-data localization. They remain research data and do not replace the production reconstruction model.'],
    ['60 条独立生成的复杂单触头动作；真实＋虚拟配比为源侧选出的 2% 虚拟质量。相同实验读出器、3 种子、真实留出记录等权；不含正式系统接触门控。', 'Sixty independently generated complex single-indenter motions; the real + virtual mixture uses 2% virtual sample weight selected on source data. The same experimental readout, three seeds and equal weighting of held-out real records are used; production contact gating is excluded.'],
    ['需要有限 XYZ 与时刻', 'Finite XYZ coordinates and time are required'],
    ['时刻必须严格递增', 'Time must be strictly increasing'],
    ['视角过平，无法可靠拾取 XY', 'View angle is too shallow to pick XY reliably'],
    ['目标 XY', 'Target XY'],
    ['左键拖动 XY · Shift 拖动 / 滚轮调压深', 'Drag XY · Shift-drag / wheel to change depth'],
    ['正在生成', 'Generating'], ['已暂停', 'Paused'],
    ['时间 / s', 'Time / s'], ['自主生成', 'Generated interactively'],
    ['物理模型导入：只读查看，验证声明尚未核实', 'Physical model import: view-only; validation claim unverified'],
    ['此时刻动作坐标尚未确定', 'Motion coordinates are unknown at this time'],
    ['名义抬离阶段：历史状态仍可产生残余磁响应', 'Nominal lift-off: history may still produce residual magnetic response'],
    ['训练坐标包络内；包络并非精度保证', 'Inside the training-coordinate envelope; accuracy is not guaranteed'],
    ['极浅过渡区：未验证，不能作为实际接触真值', 'Very shallow transition: unvalidated as actual contact ground truth'],
    ['训练坐标包络外：外推预测，不作已验证响应', 'Outside the training-coordinate envelope: extrapolation, not a validated response'],
    ['单触头 · 自主生成', 'Single indenter · interactive generation'],
    ['无实测', 'No measured reference'],
    ['只由动作驱动，不读取实测磁场；预测 ΔB 见下表。', 'Driven only by motion, with no measured field input; predicted ΔB is shown below.'],
    ['导入响应', 'Imported response'],
    ['当前序列无独立平移诊断。', 'No independent translation diagnostic is available for this sequence.'],
    ['本次已生成 300 秒。请先导出，再从起点重播开始新序列。', 'This run has reached 300 s. Export it before replaying from the start to begin a new sequence.'],
    ['物理接口 schema 或来源缺失', 'Physical-interface schema or provenance is missing'],
    ['时刻或数组长度无效', 'Invalid timestamps or array lengths'],
    ['需要每帧 15 个有限磁场值', 'Each frame must have 15 finite field values'],
    ['无效触点坐标', 'Invalid contact coordinates'],
    ['不接收把单点相加冒充物理仿真的数据', 'Summed single-contact responses are not accepted as a physical multi-contact simulation'],
    ['需要 MagTouch-FEM-Surface-1，单位 mm', 'MagTouch-FEM-Surface-1 with mm units is required'],
    ['需要 3–20000 个有限三维节点', '3–20,000 finite 3D nodes are required'],
    ['需要表面三角形，最多 50000 个', 'Surface triangles are required, at most 50,000'],
    ['三角形索引无效', 'Invalid triangle indices'],
    ['包含退化三角形', 'Degenerate triangle detected'],
    ['时刻必须有限且严格递增', 'Time values must be finite and strictly increasing'],
    ['位移帧数不匹配', 'Displacement frame count does not match'],
    ['每帧须包含每节点的 ux、uy、uz', 'Every frame needs ux, uy and uz for every node'],
    ['文件超过 40 MB，请减少显示帧或仅导出表面', 'File exceeds 40 MB; reduce displayed frames or export the surface only'],
  ];
  const exact = new Map(entries);
  const hasHan = /[\u3400-\u9fff]/;
  const translations = new WeakMap();
  const attributes = new WeakMap();
  let language = 'en';

  function translate(value) {
    if (exact.has(value)) return exact.get(value);
    let result = value;
    const patterns = [
      [/^S(\d+) 三轴磁场实测与模型预测$/, (_, n) => `S${n} measured and predicted three-axis field`],
      [/^目标 \((.*)\) mm · (正在生成|已暂停)$/, (_, xyz, state) => `Target (${xyz}) mm · ${exact.get(state)}`],
      [/^左键按住移动 · 松开撤出（可保持） · 右键旋转$/, () => 'Hold left button to move · release to withdraw (or keep pressed) · right button to rotate'],
      [/^拖动旋转 · 点击「自由按压」开始操作触头$/, () => 'Drag to rotate · select “Free interaction” to move the indenter'],
      [/^整条记录的 XYZ 常量偏移一阶诊断：\((.*)\) mm；未用于修正曲线。$/, (_, delta) => `First-order constant XYZ offset diagnostic for this record: (${delta}) mm; not applied to the curves.`],
      [/^物理导入 · (\d+) 触点（仅绘制首触点）$/, (_, n) => `Physical import · ${n} contacts (first contact shown)`],
      [/^已生成 (.*) · (\d+) 帧$/, (_, file, n) => `Generated ${file} · ${n} frames`],
      [/^本地物理响应 · (.*)$/, (_, file) => `Local physical response · ${file}`],
      [/^导入位移 · (.*)× 显示 · 不是实时求解$/, (_, gain) => `Imported displacement · ${gain}× display · not a live solve`],
      [/^拖动旋转 · (\d+) 节点 \/ (\d+) 三角形$/, (_, nodes, triangles) => `Drag to rotate · ${nodes} nodes / ${triangles} triangles`],
    ];
    for (const [pattern, replacement] of patterns) if (pattern.test(result)) return result.replace(pattern, replacement);
    const fragments = [
      ['源器件', 'Source device'], ['目标器件', 'Target device'],
      ['来源字段缺失：', 'Missing provenance field: '],
      ['节点×帧数超过 200 万，请仅导出表面或减采样显示帧', 'Node × frame count exceeds 2 million; export only the surface or downsample display frames'],
      ['文件：', 'File: '], ['求解器：', 'Solver: '], ['几何：', 'Geometry: '],
      ['材料：', 'Material: '], ['边界：', 'Boundary conditions: '],
      ['帧 · 来源声明未独立核验 · 不进入训练', 'frames · provenance claim not independently verified · excluded from training'],
      ['复杂三维路径', 'Complex 3D path'],
      ['圆轨迹', 'Circle'], ['八方向', 'Eight directions'], ['定点网格', 'Point grid'],
      ['垂直压入 / 撤出', 'Vertical indentation / withdrawal'], ['XY–Z 复合', 'Combined XY–Z'],
      ['有限元', 'finite element'], ['三维', '3D'], ['位移', 'displacement'],
    ];
    for (const [zh, en] of fragments) result = result.replaceAll(zh, en);
    return result;
  }

  function translateNode(node) {
    if (node.nodeType !== Node.TEXT_NODE) return;
    const current = node.nodeValue;
    if (hasHan.test(current)) translations.set(node, current);
    const original = translations.get(node);
    if (!original) return;
    const trimmed = original.trim();
    const translated = language === 'en' ? translate(trimmed) : trimmed;
    const next = original.replace(trimmed, translated);
    if (current !== next) node.nodeValue = next;
  }
  function translateElement(element) {
    if (element.nodeType !== Node.ELEMENT_NODE || ['SCRIPT', 'STYLE', 'TEXTAREA'].includes(element.tagName)) return;
    for (const key of ['aria-label', 'title']) {
      const value = element.getAttribute(key);
      if (value && hasHan.test(value)) {
        const record = attributes.get(element) || {};
        record[key] = value;
        attributes.set(element, record);
      }
      const original = attributes.get(element)?.[key];
      if (original) element.setAttribute(key, language === 'en' ? translate(original) : original);
    }
    for (const child of element.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) translateNode(child);
      else translateElement(child);
    }
  }
  function setLanguage(next) {
    language = next === 'zh' ? 'zh' : 'en';
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.getElementById('lang-en').setAttribute('aria-pressed', String(language === 'en'));
    document.getElementById('lang-zh').setAttribute('aria-pressed', String(language === 'zh'));
    translateElement(document.documentElement);
    document.dispatchEvent(new CustomEvent('magtouch:language', {detail:{language}}));
  }
  document.getElementById('lang-en').addEventListener('click', () => setLanguage('en'));
  document.getElementById('lang-zh').addEventListener('click', () => setLanguage('zh'));
  const observer = new MutationObserver(changes => {
    for (const change of changes) {
      if (change.type === 'characterData') translateNode(change.target);
      else if (change.type === 'childList') for (const node of change.addedNodes) {
        if (node.nodeType === Node.TEXT_NODE) translateNode(node);
        else translateElement(node);
      }
    }
  });
  observer.observe(document.documentElement, {subtree:true, childList:true, characterData:true});
  const nativeFillText = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function(text, ...rest) {
    return nativeFillText.call(this, language === 'en' ? translate(String(text)) : text, ...rest);
  };
  window.MT_i18n = {get language() { return language; }, translate, setLanguage};
  setLanguage('en');
})();
