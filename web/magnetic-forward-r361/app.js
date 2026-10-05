/* Deterministic command-driven inference. No sensor measurements in step(). */
'use strict';
function spatial(q,m){const [qx,qy]=q,z=Math.max(0,q[2]),x=qx/6,y=qy/6,d=z/3,g=z/(z+.2);const p=[1,x,y,d,x*x,x*y,y*y,x*d,y*d,d*d];for(const c of m.centers_mm){let r=0;for(let j=0;j<3;j++)r+=((q[j]-c[j])/m.width_mm[j])**2;p.push(Math.exp(-.5*r));}return p.map(v=>v*g);}
class ForwardRuntime{
 constructor(model){this.m=model;this.axis=model.schema==='MagTouch-AxisForward-R363-v1';this.base=this.axis?model.base:model;this.reset();}
 reset(){this.states=Array.from({length:this.m.taus_s.length*(this.axis&&this.m.split?2:1)},()=>Array(110).fill(0));this.previous=null;this.previousQ=null;this.previousPhi=null;}
 step(q,t){
  if(q.length!==3||q.some(v=>!Number.isFinite(v))||!Number.isFinite(t))throw Error('需要有限 XYZ 与时刻');
  if(this.previous!==null&&t<=this.previous)throw Error('时刻必须严格递增');
  const p=spatial(q,this.base),dt=this.previous===null?0:t-this.previous,out=Array(15).fill(0);
  if(this.axis){
   let components=[Array(110).fill(0)];
   if(this.previousQ&&dt<=.5){
    const pp=this.previousPhi,a=spatial([q[0],q[1],this.previousQ[2]],this.base),c=spatial([this.previousQ[0],this.previousQ[1],q[2]],this.base);
    const total=p.map((v,j)=>v-pp[j]),xy=p.map((v,j)=>(a[j]-pp[j]+v-c[j])*.5);
    components=this.m.split?[xy,total.map((v,j)=>v-xy[j])]:[total];
   }else{this.states.forEach(s=>s.fill(0));if(this.m.split)components.push(Array(110).fill(0));}
   for(let i=0;i<110;i++)for(let j=0;j<15;j++)out[j]+=p[i]*this.base.coef[i][j];
   this.states.forEach((s,k)=>{const rho=Math.exp(-dt/this.m.taus_s[k%this.m.taus_s.length]),d=components[Math.floor(k/this.m.taus_s.length)];for(let i=0;i<110;i++){s[i]=rho*(s[i]+d[i]);for(let j=0;j<15;j++)out[j]+=s[i]*this.m.coef[k*110+i][j];}});
  }else{
   const x=p.slice();if(dt>.5)this.states.forEach(s=>s.fill(0));
   this.states.forEach((s,k)=>{const rho=Math.exp(-dt/this.m.taus_s[k]);for(let j=0;j<110;j++){s[j]=rho*s[j]+(1-rho)*p[j];x.push(p[j]-s[j]);}});
   for(let i=0;i<x.length;i++)for(let j=0;j<15;j++)out[j]+=x[i]*this.m.coef[i][j];
  }
  this.previous=t;this.previousQ=q.slice();this.previousPhi=p;return out;
 }
}
if(typeof module!=='undefined')module.exports={ForwardRuntime,spatial};
if(typeof document!=='undefined')(()=>{
const D=window.MAGTOUCH_DATA,$=id=>document.getElementById(id),axis=['#c35762','#178471','#2c6ad1'];
if(D.models.axis){const o=document.createElement('option');o.value='axis';o.textContent='分轴历史 · 新研究模型';$('model').append(o);$('model').value='axis';}
let chosen=null,index=0,playing=false,free=false,history=null,freeQ=[0,0,-1],runtimes={},yaw=-.7,pitch=.65,last=0,drag=null,imported=null,pressDepth=1.4;
const names={circle_scan:'圆轨迹',angle_scan:'八方向',z_path:'垂直压入 / 撤出',grid_points:'定点网格',compound:'XY–Z 复合',waypoint_xyz:'复杂三维路径'};
const option=document.createElement('option');option.value='free';option.textContent='交互生成 · 5 mm 单触头';$('case').append(option);
for(const c of D.cases){const o=document.createElement('option');o.value=c.id;o.textContent=`${c.device==='source'?'源器件':'目标器件'} · ${names[c.mode]} · ${c.id}`;$('case').append(o);}
for(let s=0;s<5;s++){$('charts').insertAdjacentHTML('beforeend',`<section class="panel chart"><h3>S${s+1} · ΔB</h3><canvas id="chart${s}" aria-label="S${s+1} 三轴磁场实测与模型预测"></canvas></section>`);$('values').insertAdjacentHTML('beforeend',`<tr><td>S${s+1}</td><td id="b${s*3}">—</td><td id="b${s*3+1}">—</td><td id="b${s*3+2}">—</td></tr>`);}
for(const [k,v]of Object.entries(D.augmentation)){$('augmentation').insertAdjacentHTML('beforeend',`<tr><td>${k==='source'?'源器件':'目标器件'}</td>${['real','synthetic','mixed'].map(a=>`<td>${v[a].xyz_mean_mm.toFixed(3)}</td>`).join('')}</tr>`);}
$('aug-conclusion').textContent=D.augmentation_conclusion||'当前 50% 混合权重的虚拟增强未优于仅真实数据训练，因此这些虚拟数据未进入正式模型训练。';
if(D.augmentation_scope)$('aug-scope').textContent=D.augmentation_scope;
function canvas(id){const c=$(id),r=c.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);if(c.width!==Math.round(r.width*dpr)||c.height!==Math.round(r.height*dpr)){c.width=Math.round(r.width*dpr);c.height=Math.round(r.height*dpr);}const ctx=c.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,r.width,r.height);return[ctx,r.width,r.height];}
function point3(p,w,h){return MagTouchInteraction.project(p,w,h,yaw,pitch);}
function line3(ctx,a,b,w,h,color='#dce4ed',lw=1){const p=point3(a,w,h),q=point3(b,w,h);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.stroke();}
function current(){const a=free?history:chosen.data,i=Math.min(index,a.t.length-1);return{a,i,q:a.q[i],b:a[$('model').value][i]};}
function insideTrainingBox(q){const bounds=D.models.dynamic.metadata.training_bounds_mm;return q.every((v,j)=>Number.isFinite(v)&&v>=bounds[0][j]-1e-6&&v<=bounds[1][j]+1e-6);}
function drawScene(){const[ctx,w,h]=canvas('scene'),{a,i,q,b}=current();ctx.font='12px Segoe UI,Microsoft YaHei,sans-serif';for(let k=-12;k<=12;k+=3){line3(ctx,[-12,k,0],[12,k,0],w,h);line3(ctx,[k,-12,0],[k,12,0],w,h);}line3(ctx,[-12,-12,0],[13,-12,0],w,h,'#8194a6');line3(ctx,[-12,-12,0],[-12,13,0],w,h,'#8194a6');for(const[p,txt]of[[[13,-12,0],'X / mm'],[[-12,13,0],'Y / mm']]){ctx.fillStyle='#60758b';ctx.fillText(txt,...point3(p,w,h));}let previous=null;for(let n=0;n<=i;n++){const p=a.q[n];if(!p||p.some(x=>x===null)){previous=null;continue;}const v=[p[0],p[1],-p[2]];if(previous)line3(ctx,previous,v,w,h,'#087d8a',1.6);previous=v;}
for(const[s,p]of Object.entries(D.sensors)){const base=[p.x_mm,p.y_mm,-5],uv=point3(base,w,h);ctx.fillStyle='#2c6ad1';ctx.fillRect(uv[0]-4,uv[1]-3,8,6);ctx.fillStyle='#60758b';ctx.fillText(s,uv[0]+7,uv[1]+3);const j=(Number(s.slice(1))-1)*3;if(b&&b.slice(j,j+3).every(Number.isFinite)){const v=b.slice(j,j+3),norm=Math.hypot(...v),factor=3.3/Math.max(norm,160);const end=base.map((x,k)=>x+v[k]*factor);line3(ctx,base,end,w,h,'#db703a',2);const ep=point3(end,w,h);ctx.beginPath();ctx.arc(...ep,2.5,0,Math.PI*2);ctx.fillStyle='#db703a';ctx.fill();}}
if(q&&q.every(Number.isFinite)){const z=-q[2];for(let k=0;k<32;k++){const a=k*2*Math.PI/32,c=(k+1)*2*Math.PI/32;line3(ctx,[q[0]+2.5*Math.cos(a),q[1]+2.5*Math.sin(a),z],[q[0]+2.5*Math.cos(c),q[1]+2.5*Math.sin(c),z],w,h,'#203248',1.4);}line3(ctx,[q[0],q[1],z],[q[0],q[1],z+6],w,h,'#203248',6);const p=point3([q[0],q[1],z+6],w,h);ctx.fillStyle='#203248';ctx.fillText('Ø5 mm',p[0]+8,p[1]);}
if(free){const target=['x','y','z'].map(id=>Number($(id).value));const p=point3([target[0],target[1],0],w,h);ctx.strokeStyle='#d08a20';ctx.setLineDash([4,3]);ctx.beginPath();ctx.arc(...p,8,0,2*Math.PI);ctx.moveTo(p[0]-13,p[1]);ctx.lineTo(p[0]+13,p[1]);ctx.moveTo(p[0],p[1]-13);ctx.lineTo(p[0],p[1]+13);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#966115';ctx.fillText('目标 XY',p[0]+12,p[1]-9);ctx.fillStyle='#087d8a';ctx.fillText('左键拖动 XY · Shift 拖动 / 滚轮调压深',16,25);$('target-readout').textContent='目标 ('+target.map(v=>v.toFixed(2)).join(', ')+') mm · '+(playing?'正在生成':'已暂停');}}
function drawCharts(){const{a,i}=current(),key=$('model').value;for(let s=0;s<5;s++){const[ctx,w,h]=canvas('chart'+s),left=55,right=w-13,top=12,bottom=h-37,t0=a.t[0],t1=Math.max(t0+.1,a.t.at(-1));let lo=Infinity,hi=-Infinity;for(const arr of[a.b,a[key]])if(arr)for(const row of arr)for(let j=s*3;j<s*3+3;j++)if(row&&Number.isFinite(row[j])){lo=Math.min(lo,row[j]);hi=Math.max(hi,row[j]);}if(!Number.isFinite(lo)){lo=-1;hi=1;}const margin=Math.max((hi-lo)*.09,1);lo-=margin;hi+=margin;const xx=t=>left+(t-t0)/(t1-t0)*(right-left),yy=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);ctx.font='11px Segoe UI,Microsoft YaHei';ctx.fillStyle='#60758b';ctx.lineWidth=1;for(let n=0;n<4;n++){const v=lo+(hi-lo)*n/3,y=yy(v);ctx.strokeStyle='#e8edf3';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(v.toFixed(0),left-6,y+4);}ctx.textAlign='left';ctx.fillText('µT',8,15);ctx.fillText(t0.toFixed(0),left,bottom+16);ctx.textAlign='right';ctx.fillText(t1.toFixed(0),right,bottom+16);ctx.textAlign='center';ctx.fillText('时间 / s',(left+right)/2,h-5);ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();for(const observed of[true,false]){const arr=observed?a.b:a[key];if(!arr)continue;ctx.setLineDash(observed?[]:[5,3]);ctx.lineWidth=observed?1.1:1.6;for(let d=0;d<3;d++){ctx.strokeStyle=axis[d];ctx.beginPath();let open=false;for(let n=0;n<arr.length;n++){const value=arr[n]?.[s*3+d];if(!Number.isFinite(value)){open=false;continue;}const p=[xx(a.t[n]),yy(value)];if(open)ctx.lineTo(...p);else ctx.moveTo(...p);open=true;}ctx.stroke();}}ctx.setLineDash([]);ctx.strokeStyle='#203248';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(xx(a.t[i]),top);ctx.lineTo(xx(a.t[i]),bottom);ctx.stroke();ctx.restore();}}
function refresh(){const{a,i,q,b}=current();['qx','qy','qz'].forEach((id,j)=>$(id).textContent=Number.isFinite(q?.[j])?q[j].toFixed(2):'—');$('clock').textContent=a.t[i].toFixed(1)+' s';$('total').textContent=free?'自主生成':a.t.at(-1).toFixed(1)+' s';$('timeline').max=Math.max(0,a.t.length-1);$('timeline').value=i;for(let j=0;j<15;j++)$('b'+j).textContent=Number.isFinite(b?.[j])?b[j].toFixed(1):'—';const bounds=D.models.dynamic.metadata.training_bounds_mm;const supported=q?.every((v,j)=>Number.isFinite(v)&&v>=bounds[0][j]&&v<=bounds[1][j]);$('status').textContent=imported?'物理模型导入：只读查看，验证声明尚未核实':!q?.every(Number.isFinite)?'此时刻动作坐标尚未确定':q[2]<=0?'名义抬离阶段：历史状态仍可产生残余磁响应':supported?'训练坐标包络内；包络并非精度保证':q[2]<.4?'极浅过渡区：未验证，不能作为实际接触真值':'训练坐标包络外：外推预测，不作已验证响应';drawScene();drawCharts();}
function freeInit(){runtimes=Object.fromEntries(Object.entries(D.models).map(([k,m])=>[k,new ForwardRuntime(m)]));freeQ=[0,0,-1];history={t:[0],q:[freeQ.slice()],b:null};for(const[k,rt]of Object.entries(runtimes))history[k]=[rt.step(freeQ,0)];index=0;}
function loadCase(){drag=null;playing=false;$('play').textContent='播放';free=$('case').value==='free';document.body.classList.toggle('free-mode',free);if(!imported||$('case').value!=='physics-import'){imported=null;const oldOption=$('case').querySelector('option[value="physics-import"]');if(oldOption)oldOption.remove();}$('model').disabled=!!imported;$('free-controls').classList.toggle('hidden',!free);$('gesture').value=free?'touch':'camera';updateGesture();if(free){freeInit();$('case-title').textContent='单触头 · 自主生成';$('rmse').textContent='—';$('nrmse').textContent='无实测';$('offset').textContent='只由动作驱动，不读取实测磁场；预测 ΔB 见下表。';}else{if(!imported)chosen=D.cases.find(c=>c.id===$('case').value);index=0;showMetrics();}refresh();}
function showMetrics(){if(free)return;if(imported)$('model').value='dynamic';const r=chosen[$('model').value];$('case-title').textContent=imported?`物理导入 · ${imported.contacts_xyz_mm[0].length} 触点（仅绘制首触点）`:`${chosen.device==='source'?'源器件':'目标器件'} · ${names[chosen.mode]||'导入响应'}`;$('rmse').textContent=r?.rmse_uT?.toFixed(1)||'—';$('nrmse').textContent=r?.nrmse!==undefined?(100*r.nrmse).toFixed(1)+'%':'—';const dx=($('model').value==='axis'?chosen.axis_xyz_offset_diagnostic:chosen.xyz_offset_diagnostic)?.estimated_delta_xyz_mm;$('offset').textContent=dx?'整条记录的 XYZ 常量偏移一阶诊断：('+dx.map(v=>v.toFixed(2)).join(', ')+') mm；未用于修正曲线。':'当前序列无独立平移诊断。';}
function tick(){if(free){const t=history.t.at(-1)+.1;let target=[$('x'),$('y'),$('z')].map(e=>Number(e.value));const p=$('preset').value;if(p==='circle')target=[4*Math.cos(t/2),4*Math.sin(t/2),Number($('z').value)];if(p==='radial'){const k=Math.floor(t/5)%8,u=(t%5)/5,r=4*(1-Math.abs(2*u-1));target=[r*Math.cos(k*Math.PI/4),r*Math.sin(k*Math.PI/4),Number($('z').value)];}if(p==='vertical')target=[Number($('x').value),Number($('y').value),-.6+2.8*(1-Math.abs((t%10)/5-1))];freeQ=MagTouchInteraction.advance(freeQ,target);history.t.push(t);history.q.push(freeQ.slice());for(const[k,rt]of Object.entries(runtimes))history[k].push(rt.step(freeQ,t));index=history.t.length-1;if(index>=3000){playing=false;$('play').textContent='播放';}}else{index++;if(index>=chosen.data.t.length){index=chosen.data.t.length-1;playing=false;$('play').textContent='播放';}}refresh();}
$('case').addEventListener('change',loadCase);$('model').addEventListener('change',()=>{showMetrics();refresh();});$('play').addEventListener('click',()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放';last=0;refresh();});$('reset').addEventListener('click',loadCase);$('timeline').addEventListener('input',()=>{playing=false;$('play').textContent='播放';index=Number($('timeline').value);refresh();});for(const id of['x','y','z'])$(id).addEventListener('input',()=>{$('v'+id).textContent=$(id).value;if(id==='z'&&Number($('z').value)>=0)pressDepth=Number($('z').value);if(free)startFree();refresh();});
function frame(now){if(playing&&(!last||now-last>=100)){last=now;tick();}requestAnimationFrame(frame);}requestAnimationFrame(frame);
function updateGesture(){$('scene').style.cursor=$('gesture').value==='touch'?'crosshair':'grab';$('gesture-hint').textContent=$('gesture').value==='touch'?'左键按住移动 · 松开撤出（可保持） · 右键旋转':'拖动旋转 · 点击「自由按压」开始操作触头';}
function setTarget(id,value){const el=$(id);el.value=MagTouchInteraction.clamp(value,Number(el.min),Number(el.max));$('v'+id).textContent=Number(el.value).toFixed(2);}
function startFree(){if(history.t.length>=3001){playing=false;$('play').textContent='播放';$('error').textContent='本次已生成 300 秒。请先导出，再从起点重播开始新序列。';$('error').classList.remove('hidden');return;}playing=true;$('play').textContent='暂停';}
function enterFree(){if(!free){$('case').value='free';loadCase();setTarget('z',-1);}$('preset').value='manual';$('gesture').value='touch';updateGesture();}
function withdraw(){enterFree();setTarget('z',-1);$('hold-contact').checked=false;startFree();refresh();}
function press(){enterFree();setTarget('z',pressDepth);startFree();refresh();}
function changeDepth(delta){enterFree();const value=MagTouchInteraction.clamp(Number($('z').value)+delta,-1,2.3);setTarget('z',value);if(value>=0)pressDepth=value;startFree();refresh();}
$('enter-free').addEventListener('click',()=>{enterFree();refresh();$('scene').focus();});
$('press-hold').addEventListener('click',()=>{$('hold-contact').checked=true;press();});
$('withdraw').addEventListener('click',withdraw);
$('deeper').addEventListener('click',()=>changeDepth(.1));$('shallower').addEventListener('click',()=>changeDepth(-.1));
$('gesture').addEventListener('change',()=>{if($('gesture').value==='touch')enterFree();updateGesture();refresh();});
$('view-reset').addEventListener('click',()=>{yaw=-.7;pitch=.65;drawScene();});
$('preset').addEventListener('change',()=>{if(free)startFree();});
function pickXY(e){const r=$('scene').getBoundingClientRect();return MagTouchInteraction.unprojectXY([e.clientX-r.left,e.clientY-r.top],r.width,r.height,yaw,pitch);}
$('scene').addEventListener('contextmenu',e=>e.preventDefault());
$('scene').addEventListener('pointerdown',e=>{
 if(drag||![0,2].includes(e.button))return;
 e.preventDefault();$('scene').focus();
 if(e.button===2||$('gesture').value==='camera'){drag={id:e.pointerId,kind:'camera',x:e.clientX,y:e.clientY};}
 else{enterFree();if(e.shiftKey){drag={id:e.pointerId,kind:'depth',y:e.clientY,z:Number($('z').value)};startFree();}else{const xy=pickXY(e),r=$('scene').getBoundingClientRect(),tip=point3([freeQ[0],freeQ[1],-freeQ[2]],r.width,r.height),near=Math.hypot(e.clientX-r.left-tip[0],e.clientY-r.top-tip[1])<38,offset=near?[freeQ[0]-xy[0],freeQ[1]-xy[1]]:[0,0];setTarget('x',xy[0]+offset[0]);setTarget('y',xy[1]+offset[1]);press();drag={id:e.pointerId,kind:'xy',offset};}}
 $('scene').setPointerCapture(e.pointerId);refresh();
});
$('scene').addEventListener('pointermove',e=>{
 if(!drag||e.pointerId!==drag.id)return;
 if(drag.kind==='camera'){yaw+=(e.clientX-drag.x)*.008;pitch=MagTouchInteraction.clamp(pitch+(e.clientY-drag.y)*.006,.2,1.35);drag.x=e.clientX;drag.y=e.clientY;drawScene();return;}
 if(drag.kind==='depth'){setTarget('z',drag.z+(e.clientY-drag.y)*.015);if(Number($('z').value)>=0)pressDepth=Number($('z').value);}else{const xy=pickXY(e);setTarget('x',xy[0]+drag.offset[0]);setTarget('y',xy[1]+drag.offset[1]);}
 refresh();
});
function endDrag(e,cancel=false){if(!drag||e.pointerId!==drag.id)return;const wasTouch=drag.kind!=='camera';drag=null;if($('scene').hasPointerCapture(e.pointerId))$('scene').releasePointerCapture(e.pointerId);if(wasTouch&&(cancel||!$('hold-contact').checked))withdraw();}
$('scene').addEventListener('pointerup',e=>endDrag(e));$('scene').addEventListener('pointercancel',e=>endDrag(e,true));$('scene').addEventListener('lostpointercapture',e=>endDrag(e,true));
$('scene').addEventListener('wheel',e=>{if($('gesture').value!=='touch')return;e.preventDefault();changeDepth(-Math.sign(e.deltaY)*.1);},{passive:false});
$('scene').addEventListener('keydown',e=>{
 if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','PageUp','PageDown',' ','Escape'].includes(e.key))return;e.preventDefault();
 if(e.key==='Escape'){withdraw();return;}if(e.key===' '){if(free&&Number($('z').value)>0)withdraw();else press();return;}
 if(e.key==='PageDown'||e.key==='PageUp'){changeDepth(e.key==='PageDown'?.1:-.1);return;}
 enterFree();const id=['ArrowLeft','ArrowRight'].includes(e.key)?'x':'y';setTarget(id,Number($(id).value)+(['ArrowRight','ArrowUp'].includes(e.key)?.2:-.2));startFree();refresh();
});
$('export').addEventListener('click',()=>{
 const link=document.createElement('a');
 let exportText;
 if(imported){exportText=JSON.stringify(imported,null,2);link.href=URL.createObjectURL(new Blob([exportText],{type:'application/json'}));link.download='magtouch_physics_import_unverified.json';}
 else{
  const{a}=current(),key=$('model').value;
  const rows=[['t_s','x_command_mm','y_command_mm','z_command_mm',...D.channels,'data_origin','model','physics_validated','contact_count','coordinates_known','inside_training_box','real_primary_eval_frame'].join(',')];
  for(let i=0;i<a.t.length;i++)rows.push([a.t[i],...a.q[i],...a[key][i],free?'synthetic_command_rollout':'model_prediction_on_recorded_commands',key,false,1,a.q[i].every(Number.isFinite),insideTrainingBox(a.q[i]),free?false:a.primary[i]].join(','));
  exportText=rows.join('\n');link.href=URL.createObjectURL(new Blob(['\ufeff'+exportText],{type:'text/csv;charset=utf-8'}));link.download='magtouch_'+(free?'synthetic_research':chosen.id)+'.csv';
 }
 $('export-text').value=exportText;$('export-status').textContent='已生成 '+link.download+' · '+(free?history.t.length:chosen.data.t.length)+' 帧';$('export-panel').classList.remove('hidden');$('export-panel').open=true;
 link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
});
$('physics-file').addEventListener('change',async e=>{try{const file=e.target.files[0];if(!file)return;const p=JSON.parse(await file.text());if(p.schema!=='MagTouch-Physics-Forward-1'||!p.provenance||!Array.isArray(p.t_s)||p.t_s.length<2)throw Error('物理接口 schema 或来源缺失');const n=p.t_s.length;if(p.delta_b_uT.length!==n||p.contacts_xyz_mm.length!==n||p.t_s.some((v,i)=>!Number.isFinite(v)||(i&&v<=p.t_s[i-1])))throw Error('时刻或数组长度无效');if(p.delta_b_uT.some(r=>r.length!==15||r.some(v=>!Number.isFinite(v))))throw Error('需要每帧 15 个有限磁场值');if(p.contacts_xyz_mm.some(r=>!Array.isArray(r)||!r.length||r.some(v=>v.length!==3||v.some(x=>!Number.isFinite(x)))))throw Error('无效触点坐标');if(p.provenance.construction==='sum_of_single_contacts')throw Error('不接收把单点相加冒充物理仿真的数据');playing=false;free=false;drag=null;imported=p;index=0;$('play').textContent='播放';document.body.classList.remove('free-mode');$('free-controls').classList.add('hidden');$('model').disabled=true;$('gesture').value='camera';updateGesture();let importOption=$('case').querySelector('option[value="physics-import"]');if(!importOption){importOption=document.createElement('option');importOption.value='physics-import';$('case').append(importOption);}importOption.textContent='本地物理响应 · '+file.name;$('case').value='physics-import';chosen={device:'source',mode:'physics',id:'physics-import',data:{t:p.t_s,q:p.contacts_xyz_mm.map(r=>r[0]),b:null,dynamic:p.delta_b_uT,instantaneous:p.delta_b_uT}};showMetrics();$('case-title').textContent=`物理导入 · ${p.contacts_xyz_mm[0].length} 触点（仅绘制首触点）`;$('error').classList.add('hidden');refresh();}catch(err){$('error').textContent=err.message;$('error').classList.remove('hidden');}});
new ResizeObserver(()=>refresh()).observe($('scene'));$('case').value='r354_source_000006';loadCase();
})();
