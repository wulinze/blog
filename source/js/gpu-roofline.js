(() => {
  'use strict';
  const root = document.querySelector('[data-roofline-lab]');
  if (!root) return;
  const read = name => Number(root.querySelector(`[name="${name}"]`).value);
  const result = root.querySelector('[data-roofline-result]');
  const svg = root.querySelector('svg');
  const format = n => n.toLocaleString('en-US', {maximumSignificantDigits: 6});
  function update() {
    const m=read('m'), n=read('n'), k=read('k'), ib=read('ib'), ob=read('ob');
    const peak=read('peak'), bw=read('bw');
    if (![m,n,k,ib,ob,peak,bw].every(x=>Number.isFinite(x)&&x>0) || ![m,n,k,ib,ob].every(Number.isInteger)) {
      result.textContent='请输入正数；尺寸和每元素字节数必须为整数。'; svg.replaceChildren(); return;
    }
    const op=root.querySelector('[name="operation"]').value;
    let flops,bytes;
    if(op==='add'||op==='axpy'){flops=n*(op==='add'?1:2);bytes=n*(2*ib+ob);}
    else if(op==='gemv'){flops=2*m*k;bytes=ib*(m*k+k)+ob*m;}
    else {flops=2*m*n*k;bytes=ib*(m*k+k*n)+ob*m*n;}
    const ai=flops/bytes,ridge=peak*1000/bw,roof=Math.min(peak,ai*bw/1000);
    const mem=bytes/(bw*1e9)*1e6,compute=flops/(peak*1e12)*1e6;
    result.textContent=`教学模型：F = ${format(flops)} FLOP；Q = ${format(bytes)} byte；AI = ${format(ai)} FLOP/byte。\nridge = ${format(ridge)} FLOP/byte；较低屋顶 = ${format(roof)} TFLOP/s。\n搬运下界 ${format(mem)} μs；计算下界 ${format(compute)} μs；总时间下界 ${format(Math.max(mem,compute))} μs。\n${ai<ridge?'此口径下带宽屋顶更低':'此口径下计算屋顶更低'}，真实瓶颈仍需测量。`;
    root.dataset.ai=String(ai);root.dataset.ridge=String(ridge);root.dataset.roof=String(roof);
    const xmin=Math.min(0.01,ai/10,ridge/100),xmax=Math.max(10000,ai*10,ridge*100);
    const ymin=Math.min(roof/10,peak/10000),ymax=peak*2;
    const x=v=>75+(Math.log10(v)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin))*620;
    const y=v=>280-(Math.log10(v)-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin))*230;
    const nodes=[];
    function el(tag,attrs,text){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));if(text!==undefined)node.textContent=text;nodes.push(node);}
    el('line',{x1:75,y1:280,x2:705,y2:280,stroke:'#98aac4'});el('line',{x1:75,y1:280,x2:75,y2:40,stroke:'#98aac4'});
    for(let p=Math.ceil(Math.log10(xmin));p<=Math.floor(Math.log10(xmax));p++){const xp=x(10**p);el('line',{x1:xp,y1:280,x2:xp,y2:286,stroke:'#98aac4'});el('text',{x:xp,y:304,'text-anchor':'middle',fill:'#c8d6e8','font-size':13},`10^${p}`);}
    for(let p=Math.ceil(Math.log10(ymin));p<=Math.floor(Math.log10(ymax));p++){const yp=y(10**p);el('line',{x1:70,y1:yp,x2:700,y2:yp,stroke:'#314056','stroke-dasharray':'3 5'});el('text',{x:65,y:yp+4,'text-anchor':'end',fill:'#c8d6e8','font-size':13},`10^${p}`);}
    const firstX=Math.max(xmin,ymin*1000/bw);
    el('polyline',{points:`${x(firstX)},${y(firstX*bw/1000)} ${x(ridge)},${y(peak)} ${x(xmax)},${y(peak)}`,fill:'none',stroke:'#79bcff','stroke-width':3});
    el('line',{x1:x(ai),y1:280,x2:x(ai),y2:y(roof),stroke:'#f7c46d','stroke-dasharray':'5 5'});
    el('circle',{cx:x(ai),cy:y(roof),r:6,fill:'#f7c46d'});
    el('text',{x:80,y:24,fill:'#c8d6e8','font-size':15},'性能上界 TFLOP/s（对数轴；黄点是预测上界，不是实测）');
    el('text',{x:390,y:335,'text-anchor':'middle',fill:'#c8d6e8','font-size':15},'算术强度 FLOP/byte（对数轴）');
    svg.replaceChildren(...nodes);
  }
  root.addEventListener('input',update);root.addEventListener('change',update);update();
  const mapping=document.querySelector('[data-mapping-lab]');
  if(mapping){const render=()=>{
    const n=Number(mapping.querySelector('[name="length"]').value), block=Number(mapping.querySelector('[name="tile"]').value), warps=Number(mapping.querySelector('[name="warps"]').value);
    if(!Number.isInteger(n)||n<0){mapping.querySelector('output').textContent='N 必须为非负整数。';return;}
    const grid=Math.ceil(n/block),valid=n===0?0:n-(grid-1)*block;
    mapping.querySelector('output').textContent=`grid = ${grid}；每个 program 有 ${block} 个逻辑位置，配置 ${warps} warp（${warps*32} 线程）。\n${n===0?'空输入：host 直接返回，不发起 kernel。':`最后一个 tile 有 ${valid} 个有效位置、${block-valid} 个越界位置。有效索引 ${(grid-1)*block} … ${n-1}。`}\n这不能直接给出 occupancy：还缺编译后的寄存器、shared、硬件上限和实际任务供应。`;
    mapping.dataset.grid=grid;mapping.dataset.valid=valid;
  };mapping.addEventListener('input',render);mapping.addEventListener('change',render);render();}
})();
