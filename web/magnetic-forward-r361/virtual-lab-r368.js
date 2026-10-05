(function(){'use strict';
const catalog=window.MAGTOUCH_LAB,$=id=>document.getElementById(id);
const labels={F:'瞬时静态磁图XYZ',M:'＋磁场域慢记忆XY',G:'＋磁证据分支保护XY',static_z:'同一因果XY＋原始静态Z',current_profile:'同一XY＋当前磁场深度剖面',field_branch:'同一XY＋磁证据连续深度',neural_unconstrained:'同一XY＋因果Z神经提议',unified:'同一XY＋磁场证据裁决Z',fast_slow:'同一XY＋有界快慢增量Z'};
for(const [key,label]of Object.entries(labels)){const o=document.createElement('option');o.value=key;o.textContent=label;$('lab-stage').append(o);}$('lab-stage').value='fast_slow';
let record=null,info=null,frame=0,playing=false,last=0,yaw=-.65,pitch=.7,drag=null,request=0;
const finite=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
const stage=()=>$('lab-stage').value;
const filters=['family','depth','speed','dwell','direction','history'];
async function load(){
 const token=++request;playing=false;$('lab-play').textContent='播放';$('lab-play').disabled=true;$('lab-export').disabled=true;
 $('lab-score').textContent='读取预先计算的虚拟实验记录…';$('lab-error').textContent='';
 const f=Object.fromEntries(filters.map(k=>[k,$('lab-'+k).value]));
 const r=catalog.find(v=>v.family===f.family&&v.max_depth_mm===Number(f.depth)&&v.speed_mm_s===Number(f.speed)&&v.dwell_s===Number(f.dwell)&&v.direction===Number(f.direction)&&v.history_strength===Number(f.history));
 try{if(!r)throw Error('找不到此实验条件');const response=await fetch('lab-r368/'+r.id+'.json');if(!response.ok)throw Error('记录加载失败 '+response.status);const data=await response.json();if(token!==request)return;
 record=data;info=r;frame=data.t.length-1;$('lab-time').max=frame;$('lab-time').value=frame;$('lab-play').disabled=false;$('lab-export').disabled=false;
 $('lab-origin').textContent=r.id+' · 数据来源：经验仿真 · 当前角色：机制测试 · '+(r.history_strength===2?'历史放大反事实压力测试，不代表标定材料':r.history_strength===0?'历史关闭控制，不代表正常动态器件':'标称经验历史响应');draw();
 }catch(e){if(token===request){record=null;$('lab-error').textContent=e.message;$('lab-score').textContent='';}}
}
function context(id){const c=$(id),r=c.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);c.width=Math.round(r.width*d);c.height=Math.round(r.height*d);const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return[x,r.width,r.height];}
function draw(){if(!record)return;const p=record.predictions[stage()],base=record.predictions.F,s=info.scores[stage()];
const [c,w,h]=context('lab-scene'),all=[...record.q,...p,...base].filter(finite),bounds=[0,1,2].map(k=>[Math.min(...all.map(v=>v[k])),Math.max(...all.map(v=>v[k]))]);
const center=bounds.map(v=>(v[0]+v[1])/2),extent=Math.max(6,...bounds.map(v=>v[1]-v[0])),scale=Math.min(w,h)/extent*.58;
const project=q=>{const x=q[0]-center[0],y=q[1]-center[1],z=q[2]-center[2];return[w/2+(x*Math.cos(yaw)-y*Math.sin(yaw))*scale,h*.52+((x*Math.sin(yaw)+y*Math.cos(yaw))*Math.sin(pitch)-z*Math.cos(pitch))*scale];};
function line(a,b,col,width=1,dash=[]){c.strokeStyle=col;c.lineWidth=width;c.setLineDash(dash);c.beginPath();c.moveTo(...project(a));c.lineTo(...project(b));c.stroke();c.setLineDash([]);}
for(let k=-6;k<=6;k+=2){line([-6,k,0],[6,k,0],'#dde7ef');line([k,-6,0],[k,6,0],'#dde7ef');}
function trace(q,col,width,dash){for(let i=1;i<=frame;i++)if(finite(q[i-1])&&finite(q[i]))line(q[i-1],q[i],col,width,dash);}
trace(record.q,'#273b50',1.2,[4,3]);if(stage()!=='F')trace(base,'#a1adba',1,[]);trace(p,'#166ab5',1.7,[]);
if(finite(p[frame])){c.beginPath();c.arc(...project(p[frame]),4,0,2*Math.PI);c.fillStyle='#166ab5';c.fill();}
c.font='12px Segoe UI,Microsoft YaHei';c.fillStyle='#526d83';for(const [q,label]of[[[6,0,0],'X / mm'],[[0,6,0],'Y / mm'],[[0,0,3],'Z / mm']]){const v=project(q);c.fillText(label,Math.max(8,Math.min(w-55,v[0])),Math.max(16,Math.min(h-20,v[1])));}
c.fillText('拖动旋转 · 原始坐标 · 无输出时断线',12,h-15);
const [x,tw,th]=context('lab-depth-chart'),arrays=[record.q,record.predictions.static_z,p],values=arrays.flat().filter(finite).map(v=>v[2]);
let lo=Math.min(...values),hi=Math.max(...values);const pad=Math.max(.12,(hi-lo)*.1);lo-=pad;hi+=pad;
const left=55,right=tw-18,top=48,bottom=th-42,px=t=>left+(t-record.t[0])/(record.t.at(-1)-record.t[0])*(right-left),py=z=>bottom-(z-lo)/(hi-lo)*(bottom-top);
x.font='12px Segoe UI,Microsoft YaHei';x.fillStyle='#536e83';x.fillText('Z 压深 / mm',left,22);
for(let i=0;i<5;i++){const z=lo+(hi-lo)*i/4,y=py(z);x.fillText(z.toFixed(1),8,y+4);x.strokeStyle='#dde7ef';x.beginPath();x.moveTo(left,y);x.lineTo(right,y);x.stroke();}
for(let i=0;i<4;i++){const t=record.t.at(-1)*i/3;x.fillText(t.toFixed(1),Math.max(left,Math.min(right-24,px(t)-12)),bottom+23);}x.fillText('时间 / s',right-50,th-7);
for(const [q,col,dash]of[[arrays[0],'#273b50',[4,3]],[arrays[1],'#a1adba',[]],[p,'#166ab5',[]]]){x.strokeStyle=col;x.lineWidth=1.4;x.setLineDash(dash);x.beginPath();let open=false;for(let i=0;i<=frame;i++){if(!finite(q[i])){open=false;continue;}if(open)x.lineTo(px(record.t[i]),py(q[i][2]));else x.moveTo(px(record.t[i]),py(q[i][2]));open=true;}x.stroke();}x.setLineDash([]);
$('lab-score').textContent='此记录：覆盖 '+(100*s.coverage).toFixed(1)+'% · XY '+s.xy_mean_mm.toFixed(3)+' mm · Z '+s.z_mae_mm.toFixed(3)+' mm · Z固定偏移 '+s.bias_xyz_mm[2].toFixed(3)+' mm';
$('lab-time').value=frame;$('lab-time-value').textContent=record.t[frame].toFixed(1)+' s';
$('lab-fields').replaceChildren();for(let k=0;k<5;k++){const tr=document.createElement('tr');for(const v of ['S'+(k+1),...record.b[frame].slice(k*3,k*3+3).map(a=>a.toFixed(2))]){const td=document.createElement('td');td.textContent=v;tr.append(td);}$('lab-fields').append(tr);}
}
filters.forEach(k=>$('lab-'+k).addEventListener('change',load));$('lab-stage').addEventListener('change',draw);
$('lab-time').addEventListener('input',()=>{playing=false;$('lab-play').textContent='播放';frame=Number($('lab-time').value);draw();});
$('lab-play').addEventListener('click',()=>{if(!record)return;if(frame===record.t.length-1)frame=0;playing=!playing;last=0;$('lab-play').textContent=playing?'暂停':'播放';draw();});
function tick(t){if(playing&&record&&(!last||t-last>=100)){last=t;frame=Math.min(frame+1,record.t.length-1);if(frame===record.t.length-1){playing=false;$('lab-play').textContent='播放';}draw();}requestAnimationFrame(tick);}requestAnimationFrame(tick);
const scene=$('lab-scene');scene.addEventListener('pointerdown',e=>{drag={id:e.pointerId,x:e.clientX,y:e.clientY};scene.setPointerCapture(e.pointerId);e.preventDefault();});scene.addEventListener('pointermove',e=>{if(!drag||drag.id!==e.pointerId)return;yaw+=(e.clientX-drag.x)*.008;pitch=Math.max(.15,Math.min(1.35,pitch+(e.clientY-drag.y)*.006));drag.x=e.clientX;drag.y=e.clientY;draw();});for(const name of ['pointerup','pointercancel','lostpointercapture'])scene.addEventListener(name,()=>drag=null);
$('lab-export').addEventListener('click',()=>{if(!record)return;const p=record.predictions[stage()];const fields=Array.from({length:15},(_,i)=>'S'+(Math.floor(i/3)+1)+'_B'+'xyz'[i%3]+'_uT');const lines=[['t_s',...fields,'x_virtual_reference_mm','y_virtual_reference_mm','z_virtual_reference_mm','x_predicted_mm','y_predicted_mm','z_predicted_mm','xyz_valid','data_origin','role','history_strength','variant'].join(',')];record.t.forEach((t,i)=>lines.push([t,...record.b[i],...record.q[i],...(finite(p[i])?p[i]:['','','']),Number(finite(p[i])),'empirical_simulation','mechanism_test',info.history_strength,stage()].join(',')));const csv='\ufeff'+lines.join('\n');const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));link.download=record.id+'_'+stage()+'_comparison.csv';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);});
new ResizeObserver(draw).observe(scene);load();
})();
