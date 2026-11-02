(() => {
  'use strict';
  function format(value) {
    if (typeof value === 'number') return Number.isInteger(value) ? String(value) : Number(value.toPrecision(7)).toString();
    if (Array.isArray(value)) return value.map(format).join(', ');
    return String(value);
  }
  function showValues(target, values) {
    target.replaceChildren();
    if (Array.isArray(values)) {
      const grid=document.createElement('div');grid.className='leet-values';
      const matrix=values.some(Array.isArray);
      if(matrix) {
        const table=document.createElement('table');
        const body=document.createElement('tbody');
        for(const row of values) {
          const tr=document.createElement('tr');
          for(const v of (Array.isArray(row)?row:[row])) {const td=document.createElement('td');td.textContent=format(v);tr.append(td);}
          body.append(tr);
        }
        table.append(body);grid.append(table);
      } else values.forEach((v,i)=>{const cell=document.createElement('div');cell.className='leet-cell';const label=document.createElement('small');label.textContent=`[${i}]`;const number=document.createElement('strong');number.textContent=format(v);cell.append(label,number);grid.append(cell);});
      target.append(grid);
    } else {
      const list=document.createElement('dl');
      for(const [key,value] of Object.entries(values)) {const dt=document.createElement('dt');dt.textContent=key;const dd=document.createElement('dd');dd.textContent=format(value);list.append(dt,dd);}
      target.append(list);
    }
  }
  async function mount(root) {
    const output=root.querySelector('[data-output]');
    try {
      const response=await fetch(root.dataset.source);
      if(!response.ok) throw new Error(`HTTP ${response.status}`);
      const cases=await response.json();
      const select=root.querySelector('select');
      cases.forEach((c,i)=>{const option=document.createElement('option');option.value=i;option.textContent=c.label;select.append(option);});
      let position=0;
      const previous=root.querySelector('[data-previous]');const next=root.querySelector('[data-next]');
      function render() {
        const example=cases[Number(select.value)];const state=example.steps[position];
        root.querySelector('[data-progress]').textContent=`步骤 ${position+1} / ${example.steps.length} · ${state.title}`;
        root.querySelector('[data-detail]').textContent=state.detail;
        showValues(output,state.values);
        root.querySelector('[data-check]').textContent=position===example.steps.length-1?`本例校验：${example.check}`:'先预测下一步，再点击推进。模型为已执行校验的 CPU 计算快照，不是 GPU 测速。';
        previous.disabled=position===0;next.disabled=position===example.steps.length-1;
      }
      previous.addEventListener('click',()=>{if(position>0){position--;render();}});
      next.addEventListener('click',()=>{if(position<cases[Number(select.value)].steps.length-1){position++;render();}});
      root.querySelector('[data-reset]').addEventListener('click',()=>{position=0;render();});
      select.addEventListener('change',()=>{position=0;render();});render();root.dataset.ready='true';
    } catch(error) {output.textContent=`演示载入失败：${error.message}。可下载本文 Python demo 运行同一模型。`;root.dataset.error='true';}
  }
  document.querySelectorAll('[data-leetcuda-lab]').forEach(mount);
})();
