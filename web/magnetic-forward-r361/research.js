'use strict';
(()=>{
const R=window.MAGTOUCH_RESEARCH,$=id=>document.getElementById(id),colors=['#bd5660','#158775','#3268bb'];
const names={lissajous_xyz:'异相 XYZ 曲线',variable_depth_loop:'变压深闭合回线',spiral_xyz:'三维螺旋动作',piecewise_xyz:'随机分段 XYZ',stop_reverse:'停驻与反转',press_slide_release:'压入—滑动—撤出'};
let trial=R.virtual[0],frame=0,playing=false,last=0,yaw=-.7,pitch=.65,drag=null;
for(const device of ['source','target']){const f=R.summary.forward[device].forward,old=f.old_history.nrmse,n=f.new_history.nrmse;const row=document.createElement('tr');for(const value of[device==='source'?'源器件（10 条）':'目标器件（20 条）',(old*100).toFixed(2)+'%',(n*100).toFixed(2)+'%',((n/old-1)*100).toFixed(1)+'%']){const td=document.createElement('td');td.textContent=value;row.append(td);}$('forward-results').append(row);}
R.virtual.forEach((a,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' · '+names[a.family]+' · '+a.frames+' 帧';$('virtual-case').append(o);});
R.fieldcases.forEach((a,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i<3?'源器件':'目标器件')+' · '+['垂直压入／撤出','圆轨迹','复杂三维动作'][i%3]+' · '+a.id.replace('r354_','').replace('r355_','');$('field-case').append(o);});
function canvas(id){const c=$(id),r=c.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);if(c.width!==Math.round(r.width*dpr)||c.height!==Math.round(r.height*dpr)){c.width=Math.round(r.width*dpr);c.height=Math.round(r.height*dpr);}const ctx=c.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,r.width,r.height);return[ctx,r.width,r.height];}
function path(){
 const[ctx,w,h]=canvas('virtual-scene'),scale=Math.min(w/21,h/20);
 const p=q=>[w/2+(q[0]*Math.cos(yaw)-q[1]*Math.sin(yaw))*scale,
  h*.5+((q[0]*Math.sin(yaw)+q[1]*Math.cos(yaw))*Math.sin(pitch)-q[2]*Math.cos(pitch))*scale];
 ctx.font='12px Segoe UI,Microsoft YaHei';
 function line(a,b,color='#d7e3ee',lw=1){ctx.beginPath();ctx.moveTo(...p(a));ctx.lineTo(...p(b));ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.stroke();}
 for(let k=-6;k<=6;k+=2){line([-6,k,0],[6,k,0]);line([k,-6,0],[k,6,0]);}
 line([-6,-6,0],[8,-6,0],'#8099ad');line([-6,-6,0],[-6,8,0],'#8099ad');line([-6,-6,0],[-6,-6,-3],'#8099ad');
 ctx.fillStyle='#60758b';
 for(const[q,label]of[[[8,-6,0],'X / mm'],[[-6,8,0],'Y / mm'],[[-6,-6,-3],'压深 Z / mm']])ctx.fillText(label,...p(q));
 // The grey preview is the planned virtual action, NOT an input to inference.
 for(let n=1;n<trial.q.length;n++){
  const a=trial.q[n-1],b=trial.q[n];
  line([a[0],a[1],-a[2]],[b[0],b[1],-b[2]],'#ccd6df',1);
 }
 for(let n=1;n<=frame;n++){
  const a=trial.q[n-1],b=trial.q[n];
  line([a[0],a[1],-a[2]],[b[0],b[1],-b[2]],b[2]>=.4?'#087d8a':'#a5b8c8',1.8);
 }
 const q=trial.q[frame],uv=p([q[0],q[1],-q[2]]);
 ctx.fillStyle='#e39242';ctx.beginPath();ctx.arc(...uv,5,0,2*Math.PI);ctx.fill();
 ctx.fillStyle='#60758b';ctx.fillText('灰：整条虚拟指令 · 青：已播放动作',14,h-18);
 $('virtual-title').textContent=names[trial.family]+' · '+trial.id;
 $('virtual-coordinates').textContent=q.map(v=>v.toFixed(2)).join(' / ')+' mm';
 $('virtual-time').textContent=trial.t[frame].toFixed(1)+' s';$('virtual-time-slider').value=frame;
 $('virtual-fields').replaceChildren();
 for(let n=0;n<5;n++){
  const row=document.createElement('tr');
  for(const v of['S'+(n+1),...trial.b[frame].slice(n*3,n*3+3).map(v=>v.toFixed(1))]){const td=document.createElement('td');td.textContent=v;row.append(td);}
  $('virtual-fields').append(row);
 }
}
function loadVirtual(){playing=false;frame=0;trial=R.virtual[Number($('virtual-case').value)];$('virtual-time-slider').max=trial.t.length-1;$('virtual-play').textContent='播放';$('virtual-export-panel').classList.add('hidden');path();}
$('virtual-case').addEventListener('change',loadVirtual);$('virtual-reset').addEventListener('click',loadVirtual);$('virtual-play').addEventListener('click',()=>{if(frame===trial.t.length-1)frame=0;playing=!playing;last=0;$('virtual-play').textContent=playing?'暂停':'播放';path();});$('virtual-time-slider').addEventListener('input',()=>{playing=false;$('virtual-play').textContent='播放';frame=Number($('virtual-time-slider').value);path();});
function animate(t){if(playing&&(!last||t-last>=100)){last=t;frame++;if(frame>=trial.t.length-1){frame=trial.t.length-1;playing=false;$('virtual-play').textContent='播放';}path();}requestAnimationFrame(animate);}requestAnimationFrame(animate);
$('virtual-scene').addEventListener('pointerdown',e=>{drag={id:e.pointerId,x:e.clientX,y:e.clientY};e.preventDefault();$('virtual-scene').setPointerCapture(e.pointerId);});$('virtual-scene').addEventListener('pointermove',e=>{if(!drag||drag.id!==e.pointerId)return;yaw+=(e.clientX-drag.x)*.008;pitch=MagTouchInteraction.clamp(pitch+(e.clientY-drag.y)*.006,.2,1.35);drag.x=e.clientX;drag.y=e.clientY;path();});for(const event of ['pointerup','pointercancel','lostpointercapture'])$('virtual-scene').addEventListener(event,()=>{drag=null;});
$('virtual-export').addEventListener('click',()=>{const header=['t_s','x_command_mm','y_command_mm','z_command_mm',...Array.from({length:15},(_,j)=>'S'+(1+Math.floor(j/3))+'_deltaB'+['x','y','z'][j%3]+'_uT'),'data_origin','physics_validated','contact_count'];const rows=[header.join(',')];trial.t.forEach((t,i)=>rows.push([t,...trial.q[i],...trial.b[i],'synthetic_axis_history_research',false,1].join(',')));const text=rows.join('\n');$('virtual-export-text').value=text;$('virtual-export-panel').classList.remove('hidden');$('virtual-export-panel').open=true;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+text],{type:'text/csv;charset=utf-8'}));a.download=trial.id+'_research.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
function field(){const a=R.fieldcases[Number($('field-case').value)],node=Number($('field-node').value),method=$('field-method').value,pred=a.predictions[method],score=a.metrics[node][method];$('field-metric').textContent='S'+(node+1)+' · 三轴 RMSE '+score.rmse_uT.toFixed(2)+' µT · NRMSE '+(score.nrmse*100).toFixed(1)+'%（主评价段，未配准）';const[ctx,w,h]=canvas('field-chart'),left=55,right=w-15,band=h/3;ctx.font='11px Segoe UI,Microsoft YaHei';for(let axis=0;axis<3;axis++){const top=axis*band+16,bottom=(axis+1)*band-22,values=[...a.truth.map(v=>v[node*3+axis]),...pred.map(v=>v[node][axis])].filter(Number.isFinite);let lo=Math.min(...values),hi=Math.max(...values);const margin=Math.max((hi-lo)*.08,1);lo-=margin;hi+=margin;const xx=t=>left+(t-a.t[0])/(a.t.at(-1)-a.t[0])*(right-left),yy=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);ctx.fillStyle='#60758b';ctx.textAlign='right';ctx.fillText(hi.toFixed(0),left-7,top+4);ctx.fillText(lo.toFixed(0),left-7,bottom);ctx.textAlign='left';ctx.fillText(['Bx','By','Bz'][axis]+' µT',left,top-4);ctx.strokeStyle='#e0e9f1';ctx.beginPath();ctx.moveTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();for(const observed of [true,false]){ctx.setLineDash(observed?[]:[5,3]);ctx.strokeStyle=colors[axis];ctx.lineWidth=observed?1:1.5;ctx.beginPath();let open=false;for(let i=0;i<a.t.length;i++){const v=observed?a.truth[i][node*3+axis]:pred[i][node][axis];if(!Number.isFinite(v)){open=false;continue;}if(open)ctx.lineTo(xx(a.t[i]),yy(v));else ctx.moveTo(xx(a.t[i]),yy(v));open=true;}ctx.stroke();}ctx.setLineDash([]);ctx.fillStyle='#60758b';ctx.fillText(a.t[0].toFixed(0)+' s',left,bottom+15);ctx.textAlign='right';ctx.fillText(a.t.at(-1).toFixed(0)+' s',right,bottom+15);}}
for(const id of ['field-case','field-node','field-method'])$(id).addEventListener('change',field);
const diagnostic=R.summary.offset_diagnostics;$('offset-summary').textContent=['source','target'].map(k=>{const d=diagnostic[k];return(k==='source'?'源':'目标')+'器件：原始 '+d.raw_rmse_uT.toFixed(2)+' µT → 事后固定 XYZ 偏移诊断 '+d.posthoc_xyz_shift_rmse_uT.toFixed(2)+' µT（'+d.records+' 条；未应用于主结果）';}).join('；');
new ResizeObserver(()=>{path();field();}).observe($('virtual-scene'));loadVirtual();field();
})();
