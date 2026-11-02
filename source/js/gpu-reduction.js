(() => {
  'use strict';
  const root=document.querySelector('[data-reduction="warp"]');
  if(!root)return;
  let round=0, states=[], mode='xor';
  const nodes=(container,values,kind='')=>{container.replaceChildren(...values.map((value,i)=>{const el=document.createElement('div');el.className='reduce-cell '+(kind==='down'&&i>0?'reduce-muted':'');const label=document.createElement('small');label.textContent='lane '+i;const number=document.createElement('strong');number.textContent=String(value);el.append(label,number);return el;}));};
  function renderWarp(){
    nodes(root.querySelector('[data-lanes]'),states[round],round===3?mode:'');
    const m=round===0?'尚未交换':`刚完成 mask/offset = ${8>>round}`;
    root.querySelector('output').textContent=`第 ${round}/3 轮；${m}。\n${round===3?(mode==='xor'?'所有 lane 都持有全和。':'只有 lane 0 是约定的归约输出；其他 lane 不作为最终结果使用。'):'每个 lane 都从上一轮快照读数据，不按数组次序原地修改。'}`;
    root.querySelector('[data-next]').disabled=round===3;
    root.dataset.result=String(states[round][0]);root.dataset.round=String(round);
  }
  function resetWarp(){
    mode=root.querySelector('select').value;
    const v=root.querySelector('input').value.split(',').map(x=>Number(x.trim()));
    if(v.length!==8||v.some(x=>!Number.isFinite(x))){root.querySelector('output').textContent='请输入 8 个用英文逗号隔开的有限数值。';root.querySelector('[data-next]').disabled=true;return;}
    states=[v];for(const distance of [4,2,1]){const old=states.at(-1);states.push(old.map((x,i)=>x+old[mode==='xor'?(i^distance):(i+distance<8?i+distance:i)]));}
    round=0;renderWarp();
  }
  root.querySelector('[data-next]').addEventListener('click',()=>{round++;renderWarp();});root.querySelector('[data-reset]').addEventListener('click',resetWarp);root.querySelector('select').addEventListener('change',resetWarp);root.querySelector('input').addEventListener('change',resetWarp);resetWarp();
  const hierarchy=document.querySelector('[data-reduction="hierarchy"]');let levels=[],stage=0;
  function renderLevels(){const list=hierarchy.querySelector('[data-levels]');list.replaceChildren();for(let i=0;i<=stage;i++){const row=document.createElement('p');row.textContent=(i===0?'输入':`第 ${i} 次 kernel 的输出`)+': ['+levels[i].join(', ')+']';list.append(row);}
    const finished=stage===levels.length-1;hierarchy.querySelector('[data-next]').disabled=finished;hierarchy.querySelector('output').textContent=finished?`完成：结果 = ${levels.at(-1)[0]??0}。空输入的求和单位元为 0。`:'当前 partial 尚未全部合并；下一次 kernel 按同一 stream 顺序读取已写完的上一层。';hierarchy.dataset.stage=String(stage);hierarchy.dataset.result=finished?String(levels.at(-1)[0]??0):'';
  }
  function resetLevels(){const n=Number(hierarchy.querySelector('input').value),block=Number(hierarchy.querySelector('select').value);if(!Number.isInteger(n)||n<0||n>64){hierarchy.querySelector('output').textContent='教学演示的 N 范围是整数 0…64。';hierarchy.querySelector('[data-next]').disabled=true;return;}
    levels=[Array.from({length:n},(_,i)=>i+1)];while(levels.at(-1).length>1){const old=levels.at(-1),next=[];for(let i=0;i<old.length;i+=block)next.push(old.slice(i,i+block).reduce((a,b)=>a+b,0));levels.push(next);}stage=0;renderLevels();
  }
  hierarchy.querySelector('[data-next]').addEventListener('click',()=>{stage++;renderLevels();});hierarchy.querySelector('[data-reset]').addEventListener('click',resetLevels);hierarchy.querySelector('input').addEventListener('input',resetLevels);hierarchy.querySelector('select').addEventListener('change',resetLevels);resetLevels();
  const fp=document.querySelector('[data-reduction="rounding"]');function rounding(){const exp=Number(fp.querySelector('input').value),big=2**exp;const values=[big,1,-big,1];let serial=0;for(const v of values)serial=Math.fround(serial+v);const left=Math.fround(big+1),right=Math.fround(-big+1),tree=Math.fround(left+right);const paired=Math.fround(Math.fround(big-big)+Math.fround(1+1));
    fp.querySelector('output').textContent=`输入：[2^${exp}, 1, -2^${exp}, 1]，数学精确和 = 2。\n串行 FP32：${serial}；相邻成对树：${tree}（两部分 ${left} 与 ${right}）；先抵消大数再加小数：${paired}。\n每次加法用 Math.fround 模拟一次 binary32 舍入。树更浅不保证每个输入的误差都更小。`;
    fp.dataset.serial=String(serial);fp.dataset.tree=String(tree);fp.dataset.paired=String(paired);
  }
  fp.querySelector('input').addEventListener('input',rounding);rounding();
})();
