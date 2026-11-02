(() => {
  'use strict';
  const integer = (input, min, max) => {
    const parsed = Number(input.value);
    return Math.max(min, Math.min(max, Number.isFinite(parsed) ? Math.trunc(parsed) : min));
  };
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  function controls(root) { const el = node('div', undefined, 'demo-controls'); root.append(el); return el; }
  function number(parent, label, value, min, max) {
    const wrap = node('label', label);
    const input = document.createElement('input');
    Object.assign(input, { type: 'number', value: String(value), min: String(min), max: String(max), step: '1' });
    wrap.append(input); parent.append(wrap); return input;
  }
  function output(root) { const el = node('div', '', 'demo-output'); el.setAttribute('role', 'status'); el.setAttribute('aria-live', 'polite'); root.append(el); return el; }
  const demos = {
    vector(root) {
      const panel = controls(root);
      const factor = number(panel, 'B[i] = i ×', 2, 0, 9);
      const result = output(root);
      const render = () => {
        const k = integer(factor, 0, 9); const a = [0, 1, 2, 3]; const b = a.map(i => i * k);
        result.textContent = `A = [${a.join(', ')}]\nB = [${b.join(', ')}]\nC = [${a.map((i, j) => i + b[j]).join(', ')}]\n每个 C[i] 只依赖同一位置的 A[i]、B[i]。`;
      };
      factor.addEventListener('input', render); render();
    },
    flow(root) {
      const steps = ['CPU 准备 A、B', '申请设备 A、B、C 存储', '把输入拷贝到 GPU', '提交 kernel（此时不保证完成）', '等待 kernel 完成', '拷贝 C 回 CPU 并检查'];
      let current = 0;
      const panel = controls(root); const next = node('button', '下一步'); const reset = node('button', '重置');
      next.type = reset.type = 'button'; panel.append(next, reset);
      const list = node('ol'); root.append(list); const result = output(root);
      const render = () => {
        list.replaceChildren(...steps.map((step, i) => node('li', step, `demo-step ${i === current ? 'active' : i < current ? 'done' : ''}`)));
        next.disabled = current === steps.length - 1;
        const status = ['GPU 还没有本例的数据。', '设备 C 已分配，但不能读取未初始化结果。', 'GPU 输入已就绪；C 尚未计算。', 'CPU 已提交；不能据此断言 GPU 已计算完成。', '等待已返回：设备 C 就绪，CPU 数组仍需拷回。', 'CPU C = [0, 3, 6, 9]，现在可以校验。'];
        result.textContent = status[current];
      };
      next.addEventListener('click', () => { current = Math.min(current + 1, steps.length - 1); render(); });
      reset.addEventListener('click', () => { current = 0; render(); }); render();
    },
    hierarchy(root) {
      const panel = controls(root); const size = number(panel, '每块线程数', 64, 1, 256); const blocks = number(panel, 'block 数', 3, 1, 8);
      const result = output(root);
      const render = () => {
        const t = integer(size, 1, 256); const b = integer(blocks, 1, 8); const warps = Math.ceil(t / 32);
        result.textContent = `${b} 个 block × ${t} 个线程 = ${b * t} 个线程\n每块 ${warps} 个 warp，共 ${b * warps} 个 warp\n每块最后一个 warp 有 ${t % 32 || 32} 个有效线程位置\nwarp 不跨 block；本计算不预测这些 block 实际分配到哪个 SM。`;
      };
      size.addEventListener('input', render); blocks.addEventListener('input', render); render();
    },
    index(root) {
      const panel = controls(root); const length = number(panel, '元素 N', 10, 1, 64); const size = number(panel, '每块线程 T', 4, 1, 32);
      const result = output(root); const grid = node('div', undefined, 'demo-blocks'); root.append(grid);
      const render = () => {
        const n = integer(length, 1, 64); const t = integer(size, 1, 32); const blocks = Math.ceil(n / t);
        result.textContent = `blocks = ceil(${n} / ${t}) = ${blocks}\n启动 ${blocks * t} 个线程；${blocks * t - n} 个线程因 i >= N 不访问数组。`;
        grid.replaceChildren();
        for (let b = 0; b < blocks; b++) {
          const block = node('div', `block ${b}`, 'demo-block'); const cells = node('div', undefined, 'demo-cells');
          for (let lane = 0; lane < t; lane++) {
            const index = b * t + lane;
            cells.append(node('span', `t${lane} → i${index}${index >= n ? ' 越界' : ''}`, `demo-cell${index >= n ? ' invalid' : ''}`));
          }
          block.append(cells); grid.append(block);
        }
      };
      length.addEventListener('input', render); size.addEventListener('input', render); render();
    },
    memory(root) {
      const panel = controls(root);
      const select = (label, options) => {
        const wrap = node('label', label);
        const input = document.createElement('select');
        for (const [value, title] of options) {
          const option = node('option', title); option.value = value; input.append(option);
        }
        wrap.append(input); panel.append(wrap); return input;
      };
      const space = select('访问对象', [['global', 'global：x = A[i]'], ['shared', 'shared：y = tile[t]'], ['register', 'register：复用 x'], ['local', 'local：读取私有存储']]);
      const cache = select('假定缓存状态', [['l1', 'L1 命中'], ['l2', 'L1 未命中，L2 命中'], ['dram', 'L1 / L2 都未命中'], ['bypass', '绕过 L1，L2 命中']]);
      const result = output(root);
      const render = () => {
        const usesCache = space.value === 'global' || space.value === 'local';
        cache.disabled = !usesCache;
        if (space.value === 'register') {
          result.textContent = '值已经在本线程的寄存器中\n寄存器 → 算术执行单元 → 结果寄存器\n本次复用不执行内存 load，不查询 L1 / L2，也不访问 DRAM。\n前提：该值确实保留在寄存器中，未发生 spill。';
          return;
        }
        if (space.value === 'shared') {
          result.textContent = '前提：tile 已被写入，并完成所需的线程间同步\nshared load → 本 SM 为本 block 分配的 shared → 目标寄存器\n不再查询原 global 地址的 L1 / L2 / DRAM。\n填充 shared 的成本发生在此前的显式搬运阶段。';
          return;
        }
        const paths = {
          l1: ['load → L1 命中 → 目标寄存器', '本次所需数据由 L1 提供，无需继续请求 L2 / DRAM。'],
          l2: ['load → L1 未命中 → L2 命中 → 返回目标寄存器', '本次所需数据由 L2 提供，无需读取 DRAM。'],
          dram: ['load → L1 未命中 → L2 未命中 → 内存控制器 → HBM / GDDR', '数据经返回通路送到目标寄存器；缓存填充依具体策略。'],
          bypass: ['load → 绕过 L1 → L2 命中 → 返回目标寄存器', '这是另一种允许的加载策略；L1 不是所有 load 的必经缓存。']
        };
        const [path, explanation] = paths[cache.value];
        result.textContent = `${space.value === 'global' ? 'global：本例 A 位于当前 GPU 的设备分配' : 'local：逻辑上线程私有，物理存储由设备内存支撑'}\n${path}\n${explanation}\n只跟踪一个数据片段；命中状态是教学假设，不是运行测量。`;
      };
      space.addEventListener('change', render); cache.addEventListener('change', render); render();
    },
    shared(root) {
      const panel = controls(root); const base = number(panel, '输入起点', 0, 0, 20); const result = output(root);
      const render = () => {
        const start = integer(base, 0, 20); const input = Array.from({ length: 8 }, (_, i) => start + i);
        const rotated = input.map((_, i) => input[Math.floor(i / 4) * 4 + (i % 4 + 1) % 4]);
        result.textContent = `输入：[${input.join(', ')}]\nblock 0 的 shared：[${input.slice(0, 4).join(', ')}]\nblock 1 的 shared：[${input.slice(4).join(', ')}]\n块内屏障之后，每线程读本块的下一个位置。\n输出：[${rotated.join(', ')}]\n注意：每 4 个数内部循环，并非整个数组一起循环。`;
      };
      base.addEventListener('input', render); render();
    },
    sync(root) {
      let phase = 0;
      const panel = controls(root); const next = node('button', '下一步'); const reset = node('button', '重置'); next.type = reset.type = 'button'; panel.append(next, reset);
      const result = output(root);
      const stages = ['thread 0 已写 shared[0] = 10；thread 1 还没写。\n此时读取 shared[1] 的内容没有得到保证。', 'thread 0 到达块内屏障，等待其他参与线程。\n等待时不能越过屏障先读取 shared[1]。', 'thread 1 写 shared[1] = 20，并到达同一屏障。\n两个参与线程都已到达。', '屏障之后：thread 0 读到 20，thread 1 读到 10。\n这是一次合法执行示意，不是线程固定先后顺序。'];
      const render = () => { result.textContent = stages[phase]; next.disabled = phase === stages.length - 1; };
      next.addEventListener('click', () => { phase = Math.min(phase + 1, stages.length - 1); render(); }); reset.addEventListener('click', () => { phase = 0; render(); }); render();
    },
    address(root) {
      const panel = controls(root); const stride = number(panel, '元素步长', 1, 1, 8); const result = output(root);
      const render = () => {
        const s = integer(stride, 1, 8); const indices = Array.from({ length: 8 }, (_, t) => t * s); const span = (indices.at(-1) + 1) * 4;
        result.textContent = `线程编号：[0, 1, 2, 3, 4, 5, 6, 7]\n元素下标：[${indices.join(', ')}]\n相对字节偏移（float）：[${indices.map(i => i * 4).join(', ')}]\n8 个线程读取 32 字节有效数据，首尾覆盖范围为 ${span} 字节。\n这里计算地址分布，不模拟缓存或统计实际内存事务。`;
      };
      stride.addEventListener('input', render); render();
    }
  };
  document.querySelectorAll('[data-cuda-demo]').forEach(root => {
    const setup = demos[root.dataset.cudaDemo];
    if (!setup) return;
    root.replaceChildren(node('span', '可交互的概念演示 · 在浏览器计算，不执行 CUDA', 'demo-kind'));
    setup(root);
  });
})();
