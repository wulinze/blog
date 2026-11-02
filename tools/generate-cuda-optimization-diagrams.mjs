import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// One scene description produces both the published SVG and editable shapes.
const output = fileURLToPath(new URL('../source/images/cuda-overlap/', import.meta.url));
const colors = { ink: '#243247', muted: '#667286', paper: '#fffdf6', blue: '#dcecff', green: '#dff3e5', orange: '#ffe4bf', gray: '#f0f0ed', red: '#fbe2df' };
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');

class Drawing {
  constructor(title, height) {
    this.height = height;
    this.elements = [];
    this.svg = [`<title>${escape(title)}</title>`];
    this.box(0, 0, 1100, height, colors.paper, colors.paper);
    this.text(34, 26, title, 30);
  }
  element(type, x, y, width, height, extra = {}) {
    const n = this.elements.length + 1;
    this.elements.push({ id: `shape-${n}`, type, x, y, width, height, angle: 0,
      strokeColor: colors.ink, backgroundColor: 'transparent', fillStyle: 'solid',
      strokeWidth: 1.5, strokeStyle: 'solid', roughness: 1, opacity: 100,
      groupIds: [], frameId: null, roundness: null, seed: n * 7919,
      version: 1, versionNonce: n * 104729, isDeleted: false, boundElements: null,
      updated: 1, link: null, locked: false, ...extra });
  }
  box(x, y, w, h, fill = colors.paper, stroke = colors.ink) {
    this.element('rectangle', x, y, w, h, { backgroundColor: fill, strokeColor: stroke });
    this.svg.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`);
  }
  text(x, y, value, size = 22, color = colors.ink) {
    const width = [...value].reduce((sum, c) => sum + (c.charCodeAt(0) > 255 ? size : size * 0.59), 0);
    this.element('text', x, y, width, size * 1.4, { text: value, originalText: value,
      fontSize: size, fontFamily: 2, textAlign: 'left', verticalAlign: 'top',
      containerId: null, autoResize: true, lineHeight: 1.4, strokeColor: color });
    this.svg.push(`<text x="${x}" y="${y + size}" font-size="${size}" fill="${color}">${escape(value)}</text>`);
  }
  arrow(x1, y1, x2, y2, color = colors.ink) {
    this.element('arrow', x1, y1, Math.abs(x2 - x1), Math.abs(y2 - y1), {
      points: [[0, 0], [x2 - x1, y2 - y1]], startBinding: null, endBinding: null,
      startArrowhead: null, endArrowhead: 'arrow', strokeColor: color });
    this.svg.push(`<path d="M ${x1} ${y1} L ${x2} ${y2}" stroke="${color}" stroke-width="2" fill="none"/>`);
    const a = Math.atan2(y2 - y1, x2 - x1), len = 9;
    this.svg.push(`<path d="M ${x2-len*Math.cos(a-.45)} ${y2-len*Math.sin(a-.45)} L ${x2} ${y2} L ${x2-len*Math.cos(a+.45)} ${y2-len*Math.sin(a+.45)}" stroke="${color}" stroke-width="2" fill="none"/>`);
  }
  save(name) {
    fs.writeFileSync(path.join(output, `${name}.svg`), `<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="${this.height}" viewBox="0 0 1100 ${this.height}" role="img"><g font-family="Xiaolai, 'PingFang SC', 'Microsoft YaHei', sans-serif">${this.svg.join('\n')}</g></svg>\n`);
    fs.writeFileSync(path.join(output, `${name}.excalidraw`), JSON.stringify({ type: 'excalidraw', version: 2,
      source: 'https://excalidraw.com', elements: this.elements,
      appState: { viewBackgroundColor: colors.paper, gridSize: null }, files: {} }, null, 2) + '\n');
  }
}

function coalescing() {
  const d = new Drawing('6.1 · 合并访存：同一条 load，线程的地址是否挨在一起？', 960);
  d.text(34, 78, '向下沿箭头读：线程 → float 下标 → 覆盖的 32-byte sector。', 22, colors.muted);
  d.text(34, 112, '只展开 lane 0–7；每个 float 是 4 byte，起始地址按 32 byte 对齐。', 21, colors.muted);
  d.box(28, 164, 1044, 280, colors.gray);
  d.text(48, 182, '① 跨步访问：lane t 读取 in[32*t]，相邻线程相隔 128 byte', 24);
  for (let t = 0; t < 8; t++) {
    const x = 52 + t * 126;
    d.box(x + 17, 233, 72, 38, colors.red);
    d.text(x + 26, 238, `t${t}`, 20);
    d.arrow(x + 53, 277, x + 53, 313);
    d.text(x + 10, 318, `in[${32*t}]`, 18);
    for (let j = 0; j < 8; j++) d.box(x + j * 12, 357, 12, 28, j === 0 ? colors.orange : colors.paper);
    d.text(x + 6, 395, `sector ${4*t}`, 17);
  }
  d.text(48, 452, '橙色是本次需要的 float；空白格未使用。中间未触及的 sector 已省略。', 21, colors.muted);
  d.box(28, 503, 1044, 255, colors.gray);
  d.text(48, 520, '② 连续访问：lane t 读取 in[t]，8 个相邻线程正好覆盖一个 sector', 24);
  for (let t = 0; t < 8; t++) {
    const x = 100 + t * 112;
    d.box(x, 573, 92, 38, colors.blue);
    d.text(x + 29, 578, `t${t}`, 20);
    d.arrow(x + 46, 615, x + 46, 647);
    d.box(x, 654, 112, 42, colors.green);
    d.text(x + 19, 662, `in[${t}]`, 20);
  }
  d.text(290, 714, 'sector 0：8 × 4 byte = 32 byte，全部被使用', 22);
  d.box(28, 786, 1044, 142, colors.blue);
  d.text(48, 801, '扩展到完整 32-lane warp：两种映射都只需要 128 byte 有效数据', 23);
  d.text(48, 840, '跨步：32 个 sector × 32 B = 1024 B 覆盖范围；连续：4 × 32 B = 128 B。', 22);
  d.text(48, 882, '优化减少请求覆盖的 sector；实际 HBM 流量还取决于缓存等因素（CC 6.0+ 模型）。', 20);
  d.save('14-coalescing-addresses');
}

function tileMatrix(d, x, y, name, selected, fill, targetLabel) {
  d.text(x, y - 43, name, 23);
  for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) {
    const active = selected.includes(`${r}${c}`);
    d.box(x + c * 120, y + r * 70, 120, 70, active ? fill : colors.gray);
    d.text(x + c * 120 + 27, y + r * 70 + 20,
      targetLabel && active ? targetLabel : `${name[0]}${r}${c}`, 24, active ? colors.ink : colors.muted);
  }
}

function tiling() {
  const d = new Drawing('6.2 · tiling：先固定 C 的输出块，再配对 A、B 的输入块', 1280);
  d.text(34, 78, 'A、B、C 都是 4×4 矩阵；每个大格是含 2×2 个元素的 tile。', 23, colors.muted);
  d.text(34, 114, 'Aij / Bij / Cij 是分块编号（从 0 开始），不是单个矩阵元素。', 21, colors.muted);
  d.text(34, 163, '目标：C01（右上块） → 输出行 0、1；输出列 2、3。', 25);
  d.text(34, 211, '① K 的前半段：k = 0、1', 25);
  tileMatrix(d, 70, 303, 'A：行 × K', ['00'], colors.blue);
  tileMatrix(d, 428, 303, 'B：K × 列', ['01'], colors.green);
  tileMatrix(d, 786, 303, 'C：行 × 列', ['01'], colors.orange, 'C01');
  d.text(350, 345, '×', 32); d.arrow(693, 372, 761, 372);
  d.text(70, 457, 'A00：行 0、1；k 0、1', 20);
  d.text(428, 457, 'B01：k 0、1；列 2、3', 20);
  d.text(786, 457, 'C01 = A00 @ B01', 20);
  d.arrow(127, 495, 251, 495, '#386bad'); d.text(70, 513, '下一轮：A 向右换块', 21);
  d.arrow(413, 319, 413, 431, '#27774c');
  d.text(428, 513, '下一轮：B 向下换块', 21);
  d.text(786, 513, 'C 的位置始终不变', 21);
  d.text(34, 576, '② K 的后半段：k = 2、3', 25);
  tileMatrix(d, 70, 668, 'A：行 × K', ['01'], colors.blue);
  tileMatrix(d, 428, 668, 'B：K × 列', ['11'], colors.green);
  tileMatrix(d, 786, 668, 'C：行 × 列', ['01'], colors.orange, 'C01');
  d.text(350, 710, '×', 32); d.arrow(693, 737, 761, 737);
  d.text(70, 822, 'A01：行 0、1；k 2、3', 20);
  d.text(428, 822, 'B11：k 2、3；列 2、3', 20);
  d.text(786, 822, 'C01 += A01 @ B11', 20);
  d.box(28, 875, 1044, 166, colors.orange);
  d.text(48, 890, 'C01 = A00 @ B01 + A01 @ B11     （@ 表示矩阵乘法）', 25);
  d.text(48, 935, '为什么这样取？C 的行决定 A 的行；C 的列决定 B 的列。', 23);
  d.text(48, 978, '同一轮 A 的列范围 = B 的行范围，才能把同一个 k 的两项相乘。', 23);
  d.text(34, 1062, '接到下文的数值例子：整个 C01 分两轮累加', 24);
  d.text(48, 1105, '第 1 轮 [[1, 2], [5, 6]]  +  第 2 轮 [[3, 11], [7, 23]]', 24);
  d.text(48, 1148, '= 最终 [[4, 13], [12, 29]]；其中左下格就是 C[1,2] = 5 + 7 = 12。', 22);
  d.text(34, 1207, '蓝色：本轮 A tile；绿色：本轮 B tile；橙色：累加目标。灰色块本轮不参与。', 21, colors.muted);
  d.save('15-tiling-matrix-blocks');
}

function vectorization() {
  const d = new Drawing('6.3 · 向量化：把单个线程的一次访问从 4 byte 加宽到 16 byte', 1030);
  d.text(34, 78, '取第一轮的线程 t0–t3 作示意；格子是 float，下标表示同一数组中的位置。', 22, colors.muted);
  d.text(34, 114, '箭头表示本线程读取的范围。这里只画 load；copy 的 store 使用相同映射。', 21, colors.muted);
  d.box(28, 165, 1044, 228, colors.gray);
  d.text(48, 181, '① scalar：每个线程这次读 1 个 float，仍然可以合并访存', 24);
  for (let t = 0; t < 4; t++) {
    const x = 65 + 248 * t;
    d.box(x + 58, 233, 100, 38, colors.blue);
    d.text(x + 93, 238, `t${t}`, 21);
    d.arrow(x + 108, 279, x + 108, 312);
    d.box(x + 58, 322, 100, 44, colors.green);
    d.text(x + 79, 330, `in[${t}]`, 20);
  }
  d.box(28, 424, 1044, 264, colors.gray);
  d.text(48, 440, '② float4：每个线程这次读 4 个连续 float，线程间也仍是连续范围', 24);
  const fills = [colors.blue, colors.green, colors.orange, '#eee3fb'];
  for (let t = 0; t < 4; t++) {
    const x = 65 + 248 * t;
    d.box(x + 58, 493, 100, 38, fills[t]);
    d.text(x + 93, 498, `t${t}`, 21);
    d.arrow(x + 108, 539, x + 108, 572);
    for (let j = 0; j < 4; j++) {
      d.box(x + j * 54, 582, 54, 44, fills[t]);
      d.text(x + j * 54 + 17, 590, String(4*t+j), 21);
    }
    d.text(x + 30, 644, `in4[${t}] · 16 byte`, 20);
  }
  d.box(28, 722, 1044, 140, colors.blue);
  d.text(48, 737, '固定工作量看收益：同样复制 16 个 float = 64 byte', 25);
  d.text(48, 779, '标量：16 次线程级元素读取；float4：4 次线程级向量读取。', 23);
  d.text(48, 819, '这是线程级计数，不是 warp 指令数，也不是 HBM 事务数；总数据量不变。', 21);
  d.text(34, 885, '对照代码：in4[t] 覆盖 in[4*t] 到 in[4*t+3]，并非每次跳过 3 个元素。', 22);
  d.text(34, 929, '条件：16-byte 对齐、尾部单独处理，且编译器确实生成宽 load/store。', 22);
  d.text(34, 973, '优化目标是减少访存指令与循环开销；实际指令宽度看 SASS，收益靠实测。', 21, colors.muted);
  d.save('16-vectorized-access');
}

coalescing();
tiling();
vectorization();
