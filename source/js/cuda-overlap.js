(() => {
  'use strict';
  const el = (tag, text, cls) => {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (cls) result.className = cls;
    return result;
  };
  const value = input => Math.max(Number(input.min), Math.min(Number(input.max), Math.trunc(Number(input.value) || Number(input.min))));
  const field = (panel, label, initial, min, max) => {
    const wrap = el('label', label); const input = el('input');
    Object.assign(input, {type: 'number', value: String(initial), min: String(min), max: String(max), step: '1'});
    wrap.append(input); panel.append(wrap); return input;
  };
  const setup = root => {
    root.replaceChildren(el('span', '教学模型 · 不执行 CUDA，不代表实际硬件计时', 'demo-kind'));
    const panel = el('div', undefined, 'demo-controls');
    const result = el('div', '', 'demo-output'); result.setAttribute('role', 'status'); result.setAttribute('aria-live', 'polite');
    root.append(panel, result); return {panel, result};
  };
  const demos = {
    bounds(root) {
      const {panel, result} = setup(root);
      const f = field(panel, '运算量 GFLOP', 20, 1, 200);
      const d = field(panel, '搬运量 GB', 4, 1, 20);
      const render = () => {
        const F = value(f), D = value(d), tc = F / 10, tm = D;
        result.textContent = `假定 P=10 TFLOP/s、B=1 TB/s，可理想重叠\nI = ${F}/${D} = ${(F / D).toFixed(2)} FLOP/byte\n计算下界 ${tc.toFixed(2)} ms；搬运下界 ${tm.toFixed(2)} ms\n总耗时下界 max(${tc.toFixed(2)}, ${tm.toFixed(2)}) = ${Math.max(tc, tm).toFixed(2)} ms\n${tc === tm ? '两条吞吐约束相等。' : tc > tm ? '计算吞吐一侧的约束更紧。' : '内存带宽一侧的约束更紧。'}\n没有输入实际耗时或依赖信息，不能据此诊断 latency-bound。`;
      };
      f.addEventListener('input', render); d.addEventListener('input', render); render();
    },
    warps(root) {
      const {panel, result} = setup(root);
      const w = field(panel, '驻留 warp W', 1, 1, 12);
      const latency = field(panel, '依赖延迟 L', 6, 1, 16);
      const cells = el('div', undefined, 'overlap-slots'); root.append(cells);
      const render = () => {
        const W = value(w), L = value(latency); const ready = Array(W).fill(0);
        let next = 0, issued = 0; cells.replaceChildren();
        for (let t = 0; t < 48; t++) {
          let chosen = -1;
          for (let n = 0; n < W; n++) { const candidate = (next + n) % W; if (ready[candidate] <= t) { chosen = candidate; break; } }
          if (chosen >= 0) { ready[chosen] = t + L; next = (chosen + 1) % W; issued++; }
          const cell = el('span', chosen < 0 ? '—' : `W${chosen}`, chosen < 0 ? 'idle' : 'issued');
          cell.title = `模型时刻 ${t}`; cells.append(cell);
        }
        result.textContent = `一个发出槽，48 个模型时间单位；轮转选择就绪 warp\n已发出 ${issued} 次，空档 ${48 - issued} 次，窗口槽位利用 ${(issued / 48 * 100).toFixed(1)}%\n理想稳态上界 min(1, W/L) = ${(Math.min(1, W / L) * 100).toFixed(1)}%\n有限窗口含起始状态，比例可与稳态上界略有差别。此处无队列竞争、带宽和同步限制。`;
      };
      w.addEventListener('input', render); latency.addEventListener('input', render); render();
    },
    occupancy(root) {
      const {panel, result} = setup(root);
      const wrap = el('label', '每块线程 T'); const t = el('select');
      for (const n of [128, 256, 512]) { const option = el('option', String(n)); option.value = String(n); t.append(option); }
      t.value = '256'; wrap.append(t); panel.append(wrap);
      const r = field(panel, '寄存器/线程 R', 64, 16, 128);
      const stages = field(panel, 'stage 数', 1, 1, 6);
      const render = () => {
        const T = Number(t.value), R = value(r), S = value(stages), shared = S * 16;
        const limits = [Math.floor(65536 / (T * R)), Math.floor(96 / shared), Math.floor(2048 / T), Math.floor(64 / (T / 32)), 16];
        const blocks = Math.min(...limits), warps = blocks * T / 32;
        const names = ['寄存器', 'shared', '线程数', 'warp 数', 'block 数'];
        result.textContent = `虚构 SM：65536 个寄存器、96 KiB shared、2048 线程、64 warp、16 block\n每块 ${T / 32} warp；每块 shared = ${S} × 16 = ${shared} KiB\n各资源允许 block 数：${limits.join(' / ')}\n限制项：${names.filter((_, i) => limits[i] === blocks).join('、')}\n最多 ${blocks} block、${warps} warp，occupancy = ${(warps / 64 * 100).toFixed(1)}%\n忽略资源分配粒度和额外硬件限制；真实配置请用 occupancy API。`;
      };
      t.addEventListener('change', render); r.addEventListener('input', render); stages.addEventListener('input', render); render();
    },
    pipeline(root) {
      const {panel, result} = setup(root);
      const n = field(panel, 'tile 数 N', 4, 1, 8), l = field(panel, '搬运 L', 3, 1, 12), c = field(panel, '计算 C', 5, 1, 12);
      const chart = el('div', undefined, 'overlap-timeline'); root.append(chart);
      const render = () => {
        const N = value(n), L = value(l), C = value(c), serial = N * (L + C);
        const copies = [], computes = []; let copyEnd = 0, computeEnd = 0;
        for (let i = 0; i < N; i++) {
          // Two buffers: copy i must not overwrite the buffer still used by compute i-2.
          const start = Math.max(copyEnd, i >= 2 ? computes[i - 2].end : 0);
          copyEnd = start + L; copies.push({start, end: copyEnd, tile: i});
          const computeStart = Math.max(copyEnd, computeEnd); computeEnd = computeStart + C;
          computes.push({start: computeStart, end: computeEnd, tile: i});
        }
        result.textContent = `单位：模型时间，不是真实 GPU 周期\n串行 N×(L+C) = ${serial}\n双缓冲 L+C+(N−1)×max(L,C) = ${computeEnd}\n图中有独立搬运与计算资源；tile 先搬完再计算，slot 在旧计算结束后才可覆盖。\n实际资源竞争与同步开销会让结果偏离这个理想模型。`;
        chart.replaceChildren();
        const inner = el('div', undefined, 'overlap-timeline-inner'); chart.append(inner);
        const rows = [
          ['串行', Array.from({length: N}, (_, i) => [{start: i * (L + C), end: i * (L + C) + L, tile: i, copy: true}, {start: i * (L + C) + L, end: (i + 1) * (L + C), tile: i, copy: false}]).flat()],
          ['流水搬运', copies.map(x => ({...x, copy: true}))],
          ['流水计算', computes.map(x => ({...x, copy: false}))]
        ];
        for (const [label, entries] of rows) {
          const row = el('div', undefined, 'overlap-time-row'); row.append(el('span', label, 'overlap-time-label'));
          const track = el('div', undefined, 'overlap-time-track'); row.append(track);
          for (const entry of entries) {
            const bar = el('span', `${entry.copy ? '搬' : '算'}${entry.tile}`, entry.copy ? 'copy-bar' : 'compute-bar');
            bar.style.left = `${entry.start / serial * 100}%`; bar.style.width = `${(entry.end - entry.start) / serial * 100}%`;
            bar.title = `tile ${entry.tile}: [${entry.start}, ${entry.end})`; track.append(bar);
          }
          inner.append(row);
        }
        inner.append(el('p', `共同时间轴：0 → ${serial}；蓝色搬运，绿色计算。`));
      };
      for (const input of [n, l, c]) input.addEventListener('input', render); render();
    },
    tma(root) {
      const {panel, result} = setup(root); let step = 0;
      const next = el('button', '下一步'), reset = el('button', '重置'); next.type = reset.type = 'button'; panel.append(next, reset);
      const states = [
        'FREE：slot 可复用。descriptor、barrier、对齐与 proxy 协议已按要求初始化。',
        'FILLING：生产者为本 phase 正确登记参与数和 1024 byte 的关联传输，然后发起 TMA。提交不等于完成。',
        'FILLING：教学假设只有 512/1024 byte 到达。消费者仍不能读取；这里的数字不是可直接轮询的 CUDA API。',
        'READY：关联传输完成且参与者到达条件满足，匹配 phase 的等待已成功。现在才能消费。',
        'READING：消费者使用 slot。即使搬运引擎空闲，也不能覆盖仍在使用的数据。',
        'FREE：所有消费者及相关异步计算已不再读取 slot。通知生产者可为下一 phase 复用。'
      ];
      const render = () => { result.textContent = states[step]; next.disabled = step === states.length - 1; };
      next.addEventListener('click', () => { step = Math.min(step + 1, states.length - 1); render(); });
      reset.addEventListener('click', () => { step = 0; render(); }); render();
    }
  };
  document.querySelectorAll('[data-overlap-demo]').forEach(root => demos[root.dataset.overlapDemo]?.(root));
})();
