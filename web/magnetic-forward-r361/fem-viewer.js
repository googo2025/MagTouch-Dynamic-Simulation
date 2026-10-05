/* Imported FEM displacement visualization, not a solver or a physical validation. */
'use strict';
function validateFEM(p) {
  const fail = message => {throw Error(message);};
  const vector = row => Array.isArray(row) && row.length === 3 && row.every(Number.isFinite);
  if (p?.schema !== 'MagTouch-FEM-Surface-1' || p.units !== 'mm') fail('需要 MagTouch-FEM-Surface-1，单位 mm');
  for (const key of ['solver','geometry_id','material','boundary_conditions'])
    if (typeof p.provenance?.[key] !== 'string' || !p.provenance[key].trim()) fail('来源字段缺失：'+key);
  if (!Array.isArray(p.nodes_mm) || p.nodes_mm.length < 3 || p.nodes_mm.length > 20000 || !p.nodes_mm.every(vector)) fail('需要 3–20000 个有限三维节点');
  const count = p.nodes_mm.length;
  if (!Array.isArray(p.triangles) || !p.triangles.length || p.triangles.length > 50000) fail('需要表面三角形，最多 50000 个');
  for (const tri of p.triangles) {
    if (!Array.isArray(tri) || tri.length !== 3 || new Set(tri).size !== 3 || !tri.every(i => Number.isInteger(i) && i >= 0 && i < count)) fail('三角形索引无效');
    const a=p.nodes_mm[tri[0]], b=p.nodes_mm[tri[1]], c=p.nodes_mm[tri[2]], u=b.map((v,i)=>v-a[i]), v=c.map((x,i)=>x-a[i]);
    if (Math.hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]) < 1e-12) fail('包含退化三角形');
  }
  if (!Array.isArray(p.t_s) || !p.t_s.length || p.t_s.length > 5000 || p.t_s.some((t,i)=>!Number.isFinite(t)||(i>0&&t<=p.t_s[i-1]))) fail('时刻必须有限且严格递增');
  if (count * p.t_s.length > 2000000) fail('节点×帧数超过 200 万，请仅导出表面或减采样显示帧');
  if (!Array.isArray(p.displacement_mm) || p.displacement_mm.length !== p.t_s.length) fail('位移帧数不匹配');
  let max = 0;
  for (const frame of p.displacement_mm) {
    if (!Array.isArray(frame) || frame.length !== count || !frame.every(vector)) fail('每帧须包含每节点的 ux、uy、uz');
    for (const u of frame) max = Math.max(max, Math.hypot(...u));
  }
  const lo=[Infinity,Infinity,Infinity], hi=[-Infinity,-Infinity,-Infinity];
  for (const x of p.nodes_mm) x.forEach((v,j)=>{lo[j]=Math.min(lo[j],v);hi[j]=Math.max(hi[j],v);});
  return {max,lo,hi,center:lo.map((v,j)=>(v+hi[j])/2),span:Math.max(...lo.map((v,j)=>hi[j]-v),1)};
}
function femColor(v) {
  const stops=[[55,124,202],[25,171,167],[246,189,82],[215,85,80]], t=Math.max(0,Math.min(1,v))*3, k=Math.min(2,Math.floor(t));
  return 'rgb('+stops[k].map((x,j)=>Math.round(x+(stops[k+1][j]-x)*(t-k))).join(',')+')';
}
if(typeof module!=='undefined')module.exports={validateFEM,femColor};
if(typeof document!=='undefined')(()=>{
  const $=id=>document.getElementById(id), c=$('fem-scene');
  let data=null, info=null, i=0, yaw=-.7, pitch=.65, drag=null, playing=false, origin=null;
  function draw() {
    const r=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);
    c.width=Math.round(r.width*dpr);c.height=Math.round(r.height*dpr);
    const ctx=c.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
    if(!data)return;
    const gain=Number($('fem-gain').value), scale=Math.min(r.width,r.height)*.62/(info.span+2*info.max*gain), center=info.center;
    const project=p=>{const a=p.map((v,j)=>v-center[j]),x=a[0]*Math.cos(yaw)-a[1]*Math.sin(yaw),y=a[0]*Math.sin(yaw)+a[1]*Math.cos(yaw);return[r.width/2+x*scale,r.height*.55+(y*Math.sin(pitch)-a[2]*Math.cos(pitch))*scale,y*Math.cos(pitch)+a[2]*Math.sin(pitch)];};
    const u=data.displacement_mm[i], deformed=data.nodes_mm.map((p,n)=>p.map((v,j)=>v+gain*u[n][j])), points=deformed.map(project);
    const sorted=data.triangles.map(tri=>({tri,z:tri.reduce((s,n)=>s+points[n][2],0)})).sort((a,b)=>a.z-b.z);
    const path=(tri,pts)=>{ctx.beginPath();tri.forEach((n,j)=>j?ctx.lineTo(pts[n][0],pts[n][1]):ctx.moveTo(pts[n][0],pts[n][1]));ctx.closePath();};
    for(const {tri} of sorted){path(tri,points);const v=tri.reduce((s,n)=>s+Math.hypot(...u[n]),0)/3;ctx.fillStyle=femColor(v/Math.max(info.max,1e-12));ctx.fill();ctx.strokeStyle='rgba(30,54,80,.18)';ctx.lineWidth=.4;ctx.stroke();}
    if($('fem-reference').checked){const ref=data.nodes_mm.map(project);ctx.strokeStyle='rgba(32,50,72,.25)';ctx.lineWidth=.6;for(const tri of data.triangles){path(tri,ref);ctx.stroke();}}
    ctx.font='12px Segoe UI,Microsoft YaHei';ctx.fillStyle='#203248';ctx.fillText('导入位移 · '+gain+'× 显示 · 不是实时求解',14,23);ctx.fillText('拖动旋转 · '+data.nodes_mm.length+' 节点 / '+data.triangles.length+' 三角形',14,r.height-15);
    $('fem-clock').textContent=data.t_s[i].toFixed(3)+' s';$('fem-timeline').value=i;
  }
  $('fem-file').addEventListener('change',async e=>{
    try {
      const file=e.target.files[0];if(!file)return;if(file.size>40*1024*1024)throw Error('文件超过 40 MB，请减少显示帧或仅导出表面');
      const parsed=JSON.parse(await file.text()), checked=validateFEM(parsed);
      // Commit only after complete validation. A failed file must not destroy a loaded sequence.
      data=parsed;info=checked;i=0;playing=false;origin=null;
      $('fem-empty').classList.add('hidden');$('fem-error').textContent='';$('fem-play').disabled=data.t_s.length<2;$('fem-play').textContent='播放形变';
      $('fem-timeline').disabled=false;$('fem-timeline').max=data.t_s.length-1;
      $('fem-max').textContent=info.max.toFixed(4)+' mm';
      $('fem-meta').textContent='文件：'+file.name+'\n求解器：'+data.provenance.solver+'\n几何：'+data.provenance.geometry_id+'\n材料：'+data.provenance.material+'\n边界：'+data.provenance.boundary_conditions+'\n'+data.t_s.length+' 帧 · 来源声明未独立核验 · 不进入训练';
      draw();
    }catch(error){$('fem-error').textContent=error.message;}
  });
  $('fem-gain').addEventListener('change',draw);$('fem-reference').addEventListener('change',draw);
  $('fem-timeline').addEventListener('input',()=>{playing=false;$('fem-play').textContent='播放形变';i=Number($('fem-timeline').value);draw();});
  $('fem-play').addEventListener('click',()=>{if(!data)return;playing=!playing;if(playing&&i===data.t_s.length-1)i=0;origin=null;$('fem-play').textContent=playing?'暂停形变':'播放形变';draw();});
  c.addEventListener('pointerdown',e=>{if(drag)return;drag={id:e.pointerId,x:e.clientX,y:e.clientY};c.setPointerCapture(e.pointerId);});
  c.addEventListener('pointermove',e=>{if(!drag||drag.id!==e.pointerId)return;yaw+=(e.clientX-drag.x)*.008;pitch=Math.max(-1.3,Math.min(1.3,pitch+(e.clientY-drag.y)*.006));drag.x=e.clientX;drag.y=e.clientY;draw();});
  for(const kind of ['pointerup','pointercancel','lostpointercapture'])c.addEventListener(kind,e=>{if(drag?.id===e.pointerId)drag=null;});
  function frame(now){if(playing&&data){if(origin===null)origin=now-data.t_s[i]*1000;const t=(now-origin)/1000;let next=i;while(next+1<data.t_s.length&&data.t_s[next+1]<=t)next++;if(next!==i){i=next;draw();}if(i===data.t_s.length-1){playing=false;$('fem-play').textContent='播放形变';}}requestAnimationFrame(frame);}requestAnimationFrame(frame);
  new ResizeObserver(draw).observe(c);
})();
