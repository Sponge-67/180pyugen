/*
 * 180pyugen Web front end
 * -----------------------
 * This file intentionally contains presentation/session logic only. Camera
 * projection and final stereo rendering are not reimplemented in JavaScript;
 * the browser talks to the Python server so desktop and web results share one
 * authoritative geometry engine.
 *
 * Image workflow:
 *   choose files -> local browser preview -> upload temporary session ->
 *   point matching -> server render -> preview/download.
 *
 * Video workflow:
 *   choose both movies -> upload/probe -> AUTOMATICALLY fetch reference pair ->
 *   A/B -> precomputed server remap -> queued encode -> playback/download.
 *
 * The canvas overlay tools are deliberately diagnostic. Their source-image
 * offsets do not silently change camera calibration; final convergence trim is
 * sent as an explicit render parameter.
 */
'use strict';
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const GOPRO_L='GoPro HERO12 + Max Lens Mod 2.0 L (calibrated)';
const GOPRO_R='GoPro HERO12 + Max Lens Mod 2.0 R (calibrated)';
const EM10_L='EM10mkIII & Meike 6.5mm L';
const EM10_R='EM10mkIII & Meike 6.5mm R';
const KINO_EM10='EM10mkIII & Meike 6.5mm (180Kino)';

const state={
  left:{file:null,bitmap:null,width:0,height:0,transform:null},
  right:{file:null,bitmap:null,width:0,height:0,transform:null},
  sessionId:null,sessionVersion:0,uploadedVersion:-1,uploadPromise:null,
  points:{A:[[466,757],[455,767]],B:[[1286,823],[1271,832]]},active:'A',
  resultURL:null,resultName:'LROut.jpg',maxOutputHeight:4096,alignBitmap:null,alignAuto:[0,0,0],alignScale:1,alignRequestId:0
};
const vstate={
  left:{file:null,bitmap:null,width:0,height:0,transform:null,view:{zoom:1,centerX:.5,centerY:.5}},
  right:{file:null,bitmap:null,width:0,height:0,transform:null,view:{zoom:1,centerX:.5,centerY:.5}},
  sessionId:null,sessionVersion:0,uploadedVersion:-1,uploadPromise:null,
  leftInfo:null,rightInfo:null,
  points:{A:[[466,757],[455,767]],B:[[1286,823],[1271,832]]},active:null,navDrag:null,
  resultURL:null,resultName:'VROut.mp4',previewURL:null,clipURL:null,clipName:'cut_img.zip',alignBitmap:null,alignAuto:[0,0,0],alignScale:1,alignRequestId:0,
  resultFrames:0,resultFps:0,checkedFrame:0
};

function status(s,bad=false){const e=$('#status');e.textContent=s;e.classList.toggle('danger',bad)}
function vstatus(s,bad=false){const e=$('#vStatus');e.textContent=s;e.classList.toggle('danger',bad)}
function progress(v){$('#progressBar').style.width=`${Math.max(0,Math.min(1,Number(v)||0))*100}%`}
function vprogress(v){$('#vProgressBar').style.width=`${Math.max(0,Math.min(1,Number(v)||0))*100}%`}
function setBusy(v){document.body.classList.toggle('busy',!!v);$('#renderBtn').disabled=!!v}
function setVBusy(v){document.body.classList.toggle('busy',!!v);$('#vRenderBtn').disabled=!!v;$('#vClipBtn').disabled=!!v}
async function api(url,opt={}){const r=await fetch(url,opt);let data=null;const ct=r.headers.get('content-type')||'';if(ct.includes('application/json'))data=await r.json();if(!r.ok)throw new Error(data?.detail||data?.error||`${r.status} ${r.statusText}`);return data}

async function health(){
  try{const h=await api('/api/health');const b=$('#serverBadge');b.textContent=`server ready · ${h.queue}`;b.className='serverBadge ok'}
  catch(e){const b=$('#serverBadge');b.textContent='server unavailable';b.className='serverBadge bad';status(e.message,true);vstatus(e.message,true)}
}
async function loadProfiles(){
  const p=await api('/api/profiles');state.maxOutputHeight=p.max_output_height||4096;$('#outHeight').max=state.maxOutputHeight;
  for(const id of ['#leftProfile','#rightProfile','#vLeftProfile','#vRightProfile']){
    const el=$(id);el.innerHTML='';for(const n of p.profiles){const o=document.createElement('option');o.value=n;o.textContent=n;el.appendChild(o)}
  }
  $('#leftProfile').value=p.defaults.left;$('#rightProfile').value=p.defaults.right;
  $('#vLeftProfile').value=p.defaults.left;$('#vRightProfile').value=p.defaults.right;
  await refreshProfileInfo();await refreshVideoProfileInfo();
}

function setMode(mode){
  // Three top-level pages share one document so sessions/results survive while
  // the user reads the mechanics page or switches between still/video work.
  const image=mode==='image', video=mode==='video', info=mode==='info';
  $('#imageMode').classList.toggle('hidden',!image);
  $('#videoMode').classList.toggle('hidden',!video);
  $('#infoMode').classList.toggle('hidden',!info);
  $('#imageTab').classList.toggle('active',image);
  $('#videoTab').classList.toggle('active',video);
  $('#infoTab').classList.toggle('active',info);
  requestAnimationFrame(()=>{if(video)vRedraw();else if(image)redraw()});
}


function applyUiOptions(){
  const theme=$('#uiTheme')?.value||'dark';
  document.body.classList.toggle('theme-dark',theme==='dark');
  localStorage.setItem('180pyugen-theme',theme);
  requestAnimationFrame(()=>{redraw();vRedraw();drawCheckedOutputFrame()});
}
function restoreUiOptions(){
  const theme=localStorage.getItem('180pyugen-theme')||'dark';
  if($('#uiTheme'))$('#uiTheme').value=theme;applyUiOptions();
}

// -------------------------- shared canvas helpers ---------------------------
function fitDraw(canvas,bmp,sideObj,points,side){
  const dpr=devicePixelRatio||1,r=canvas.getBoundingClientRect(),cw=Math.max(1,r.width),ch=Math.max(1,r.height);
  canvas.width=Math.round(cw*dpr);canvas.height=Math.round(ch*dpr);
  const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,cw,ch);c.fillStyle='#111';c.fillRect(0,0,cw,ch);
  if(!bmp){sideObj.transform=null;return}
  const sc=Math.min(cw/bmp.width,ch/bmp.height),dw=bmp.width*sc,dh=bmp.height*sc,dx=(cw-dw)/2,dy=(ch-dh)/2;
  c.drawImage(bmp,dx,dy,dw,dh);sideObj.transform={scale:sc,dx,dy};
  for(const lab of ['A','B']){const p=points[lab][side==='left'?0:1];if(!p)continue;const x=dx+p[0]*sc,y=dy+p[1]*sc;c.strokeStyle=lab==='A'?'#ff3030':'#00cfff';c.lineWidth=2;c.strokeRect(x-5,y-5,10,10);c.fillStyle=c.strokeStyle;c.font='bold 13px system-ui';c.fillText(lab,x+7,y-7)}
}
function eventToSource(e,sideObj){const t=sideObj.transform;if(!t)return null;const r=e.currentTarget.getBoundingClientRect();const x=(e.clientX-r.left-t.dx)/t.scale,y=(e.clientY-r.top-t.dy)/t.scale;if(x<0||y<0||x>=sideObj.width||y>=sideObj.height)return null;return [Math.round(x),Math.round(y)]}
function drawBlend(canvas,baseBmp,overBmp,alpha,points,mode='Negative align',shiftX=0,shiftY=0){
  const dpr=devicePixelRatio||1,r=canvas.getBoundingClientRect(),cw=Math.max(1,r.width),ch=Math.max(1,r.height);
  canvas.width=Math.round(cw*dpr);canvas.height=Math.round(ch*dpr);
  const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,cw,ch);c.fillStyle='#111';c.fillRect(0,0,cw,ch);
  if(!baseBmp||!overBmp)return;
  const bw=baseBmp.width,bh=baseBmp.height,ow=overBmp.width,oh=overBmp.height;
  const sc=Math.min(cw/bw,ch/bh),dw=bw*sc,dh=bh*sc,dx=(cw-dw)/2,dy=(ch-dh)/2,sx=Number(shiftX||0)*sc,sy=Number(shiftY||0)*sc,a=Math.max(0,Math.min(1,Number(alpha)||0));
  const ml=String(mode||'').toLowerCase();
  c.save();
  if(ml.startsWith('negative')){
    c.globalCompositeOperation='lighter';c.globalAlpha=1-a;c.filter='invert(1)';c.drawImage(baseBmp,dx+sx,dy+sy,dw,dh);c.filter='none';c.globalAlpha=a;c.drawImage(overBmp,dx,dy,dw,dh);
  }else if(ml.startsWith('difference')){
    c.globalAlpha=1;c.drawImage(baseBmp,dx+sx,dy+sy,dw,dh);c.globalCompositeOperation='difference';c.drawImage(overBmp,dx,dy,dw,dh);
  }else if(ml.startsWith('red')){
    // Screen-style red/cyan inspection using canvas blend modes.
    c.globalAlpha=1;c.filter='grayscale(1) sepia(1) saturate(8) hue-rotate(310deg)';c.drawImage(overBmp,dx,dy,dw,dh);c.globalCompositeOperation='screen';c.filter='grayscale(1) sepia(1) saturate(8) hue-rotate(125deg)';c.drawImage(baseBmp,dx+sx,dy+sy,dw,dh);
  }else{
    c.globalAlpha=1;c.drawImage(baseBmp,dx+sx,dy+sy,dw,dh);c.globalAlpha=a;c.drawImage(overBmp,dx,dy,dw,dh);
  }
  c.restore();c.globalAlpha=1;c.globalCompositeOperation='source-over';c.filter='none';
  for(const lab of ['A','B']){const lp=points[lab][0],rp=points[lab][1];if(lp){const x=dx+(lp[0]/ow)*dw,y=dy+(lp[1]/oh)*dh;c.strokeStyle='#ff4d6d';c.lineWidth=2;c.beginPath();c.arc(x,y,6,0,Math.PI*2);c.stroke()}if(rp){const x=dx+sx+(rp[0]/bw)*dw,y=dy+sy+(rp[1]/bh)*dh;c.strokeStyle=lab==='A'?'#00ff7f':'#00cfff';c.lineWidth=2;c.strokeRect(x-6,y-6,12,12)}}
}

// ------------------------------- images ------------------------------------
async function decodeLocal(file,side){
  let bmp;try{bmp=await createImageBitmap(file,{imageOrientation:'none'})}catch{bmp=await createImageBitmap(file)}
  if(state[side].bitmap?.close)state[side].bitmap.close();state[side].file=file;state[side].bitmap=bmp;state[side].width=bmp.width;state[side].height=bmp.height;
  $(side==='left'?'#leftName':'#rightName').textContent=`${file.name} — ${bmp.width}×${bmp.height}`;
  if(state.sessionId)deleteSession();state.sessionVersion++;state.uploadedVersion=-1;drawSide(side);await refreshProfileInfo();
  if(state.left.file&&state.right.file)syncSession().catch(e=>status(e.message,true));
}
async function syncSession(){
  if(!state.left.file||!state.right.file)throw new Error('Load both images first');
  if(state.sessionId&&state.uploadedVersion===state.sessionVersion)return state.sessionId;if(state.uploadPromise)return state.uploadPromise;
  state.uploadPromise=(async()=>{status('Uploading images to a temporary render session…');const fd=new FormData();fd.append('left',state.left.file,state.left.file.name);fd.append('right',state.right.file,state.right.file.name);const s=await api('/api/session',{method:'POST',body:fd});state.sessionId=s.session_id;state.uploadedVersion=state.sessionVersion;$('#sessionInfo').textContent=`Temporary session ready · ${s.left_size[0]}×${s.left_size[1]} / ${s.right_size[0]}×${s.right_size[1]}`;status('Images ready. Pick A/B or render with the preset coordinates.');return state.sessionId})().finally(()=>state.uploadPromise=null);return state.uploadPromise;
}
async function deleteSession(){if(!state.sessionId)return;const sid=state.sessionId;state.sessionId=null;state.uploadedVersion=-1;try{await fetch(`/api/session/${sid}`,{method:'DELETE'})}catch{}$('#sessionInfo').textContent='Server inputs deleted. They will re-upload automatically if needed.'}
async function refreshProfileInfo(){
  for(const side of ['left','right']){const s=state[side],out=$(side==='left'?'#leftProfileInfo':'#rightProfileInfo'),sel=$(side==='left'?'#leftProfile':'#rightProfile');if(!sel?.value)continue;if(!s.width){out.textContent='';continue}
    try{const q=new URLSearchParams({name:sel.value,side:side==='left'?'L':'R',width:s.width,height:s.height});const p=await api('/api/profile-info?'+q);out.textContent=`axis (${p.axis[0].toFixed(2)}, ${p.axis[1].toFixed(2)}) · radius ${p.radius.toFixed(2)} · mode ${p.projection_mode} · k ${p.k.toFixed(4)}`+(p.calibration_warning?` · WARNING: ${p.calibration_warning}`:'')}catch(e){out.textContent=e.message}}
}
function drawSide(side){fitDraw($(side==='left'?'#leftCanvas':'#rightCanvas'),state[side].bitmap,state[side],state.points,side)}
function _diagPixels(sbs,mode,alpha,dx,dy){
  const n=sbs.height,w=n,h=n;
  const lc=document.createElement('canvas'),rc=document.createElement('canvas');lc.width=rc.width=w;lc.height=rc.height=h;
  const lctx=lc.getContext('2d'),rctx=rc.getContext('2d');
  lctx.drawImage(sbs,0,0,n,n,0,0,n,n);rctx.drawImage(sbs,n,0,n,n,dx,dy,n,n);
  const L=lctx.getImageData(0,0,w,h),R=rctx.getImageData(0,0,w,h),O=new ImageData(w,h);const a=Math.max(0,Math.min(1,alpha)),m=(mode||'').toLowerCase();
  for(let i=0;i<L.data.length;i+=4){const lr=L.data[i],lg=L.data[i+1],lb=L.data[i+2],rr=R.data[i],rg=R.data[i+1],rb=R.data[i+2];
    if(m.includes('negative')){O.data[i]=lr*a+(255-rr)*(1-a);O.data[i+1]=lg*a+(255-rg)*(1-a);O.data[i+2]=lb*a+(255-rb)*(1-a)}
    else if(m.includes('difference')){O.data[i]=Math.abs(lr-rr);O.data[i+1]=Math.abs(lg-rg);O.data[i+2]=Math.abs(lb-rb)}
    else if(m.includes('cyan')){O.data[i]=lr;O.data[i+1]=rg;O.data[i+2]=rb}
    else if(m.includes('perceptual')||m.includes('fuse')){O.data[i]=(lr+rr)>>1;O.data[i+1]=(lg+rg)>>1;O.data[i+2]=(lb+rb)>>1}
    else{O.data[i]=rr*(1-a)+lr*a;O.data[i+1]=rg*(1-a)+lg*a;O.data[i+2]=rb*(1-a)+lb*a}O.data[i+3]=255;}
  const out=document.createElement('canvas');out.width=w;out.height=h;out.getContext('2d').putImageData(O,0,0);return out;
}
function drawOverlayMessage(canvas,title,detail=''){
  const dpr=devicePixelRatio||1,r=canvas.getBoundingClientRect(),cw=Math.max(1,r.width),ch=Math.max(1,r.height);
  canvas.width=Math.round(cw*dpr);canvas.height=Math.round(ch*dpr);
  const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,cw,ch);c.fillStyle='#090b0f';c.fillRect(0,0,cw,ch);
  c.fillStyle='#c5ced8';c.textAlign='center';c.font='600 15px system-ui';c.fillText(title,cw/2,ch/2-8);
  if(detail){c.fillStyle='#8996a3';c.font='12px system-ui';c.fillText(detail,cw/2,ch/2+16)}
}
function drawProjectedSplit(canvas,bmp,mode,alpha,dx,dy,video=false){
  if(!bmp){drawOverlayMessage(canvas,'Projected alignment preview not loaded','Load reference frames or press Refresh projected view');return}
  const dpr=devicePixelRatio||1,r=canvas.getBoundingClientRect(),cw=Math.max(1,r.width),ch=Math.max(1,r.height);canvas.width=Math.round(cw*dpr);canvas.height=Math.round(ch*dpr);const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,cw,ch);c.fillStyle='#090b0f';c.fillRect(0,0,cw,ch);
  const n=bmp.height;
  if(!n||bmp.width<2*n){drawOverlayMessage(canvas,'Alignment preview decode failed',`Unexpected preview size ${bmp.width||0}×${bmp.height||0}`);return}
  const gap=6,current=_diagPixels(bmp,mode,alpha,0,0),candidate=_diagPixels(bmp,mode,alpha,dx,dy),total=2*n+gap,top=28,sc=Math.max(.01,Math.min((cw-8)/total,(ch-top-6)/n)),dw=n*sc,dh=n*sc,ox=(cw-total*sc)/2,oy=top+(ch-top-dh)/2;
  c.drawImage(current,ox,oy,dw,dh);c.drawImage(candidate,ox+(n+gap)*sc,oy,dw,dh);c.fillStyle='#d8dde6';c.font='600 13px system-ui';c.textAlign='center';c.fillText(video?'CURRENT FUSED VIDEO VIEW':'CURRENT FUSED FINAL VIEW',ox+dw/2,18);c.fillStyle='#7ee7ff';c.fillText('ALIGNED FUSED CANDIDATE',ox+(n+gap)*sc+dw/2,18);c.fillStyle='#59616e';c.fillRect(ox+(n+gap/2)*sc,oy,2,dh);(video?vstate:state).alignScale=sc;
}
async function blobToDrawable(blob){
  // Chromium/Firefox normally take the fast ImageBitmap route.  The HTMLImage
  // fallback prevents a blank black inspector on browsers/GPUs where
  // createImageBitmap() fails to decode a server-produced PNG.
  try{return await createImageBitmap(blob)}catch(_e){
    const url=URL.createObjectURL(blob);
    try{const img=new Image();await new Promise((ok,fail)=>{img.onload=ok;img.onerror=()=>fail(new Error('Could not decode alignment preview image'));img.src=url});return img}
    finally{URL.revokeObjectURL(url)}
  }
}
function drawOverlay(){drawProjectedSplit($('#overlayCanvas'),state.alignBitmap,$('#overlayMode').value,Number($('#overlayAlpha').value||50)/100,Number($('#overlayDx').value||0),Number($('#overlayDy').value||0),false)}
function vDrawOverlay(){drawProjectedSplit($('#vOverlayCanvas'),vstate.alignBitmap,$('#vOverlayMode').value,Number($('#vOverlayAlpha').value||50)/100,Number($('#vOverlayDx').value||0),Number($('#vOverlayDy').value||0),true)}
async function refreshProjectedOverlay(video=false){
  const st=video?vstate:state,canvas=$(video?'#vOverlayCanvas':'#overlayCanvas');
  const requestId=++st.alignRequestId;
  try{
    if(video)vReadCoords();else readCoords();
    drawOverlayMessage(canvas,'Rendering projected alignment…','Using the real final-eye geometry');
    const sid=video?await vSyncSession():await syncSession();
    const payload=video?{session_id:sid,left_profile:$('#vLeftProfile').value,right_profile:$('#vRightProfile').value,points:vstate.points,left_frame:Math.trunc(+$('#vLeftRef').value),right_frame:Math.trunc(+$('#vRightRef').value),roll:+$('#vRoll').value||0,pitch:+$('#vPitch').value||0,yaw:+$('#vYaw').value||0,right_shift_x_deg:+$('#vStereoX').value||0,right_shift_y_deg:+$('#vStereoY').value||0,preview_height:384}:{session_id:sid,left_profile:$('#leftProfile').value,right_profile:$('#rightProfile').value,points:state.points,roll:+$('#roll').value||0,pitch:+$('#pitch').value||0,yaw:+$('#yaw').value||0,right_shift_x_deg:+$('#stereoX').value||0,right_shift_y_deg:+$('#stereoY').value||0,preview_height:384,decoder:'opencv'};
    const url=video?'/api/video/alignment-preview':'/api/alignment-preview';
    (video?vstatus:status)('Rendering projected stereo alignment preview…');
    const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json','cache-control':'no-cache'},cache:'no-store',body:JSON.stringify(payload)});
    if(!r.ok){let d={};try{d=await r.json()}catch{}throw new Error(d.detail||`${r.status} ${r.statusText}`)}
    const blob=await r.blob();
    if(requestId!==st.alignRequestId)return;
    const drawable=await blobToDrawable(blob);
    if(requestId!==st.alignRequestId){if(drawable?.close)drawable.close();return}
    const expected=Number(r.headers.get('X-Preview-Eye')||0);
    if(!drawable.width||!drawable.height||drawable.width<drawable.height*2-2||drawable.width>drawable.height*2+2)throw new Error(`Bad alignment preview dimensions: ${drawable.width}×${drawable.height}`);
    if(expected&&Math.abs(drawable.height-expected)>2)throw new Error(`Alignment preview size mismatch: expected ${expected}px eye, got ${drawable.height}px`);
    if(st.alignBitmap?.close)st.alignBitmap.close();st.alignBitmap=drawable;
    st.alignAuto=[Number(r.headers.get('X-Align-DX')||0),Number(r.headers.get('X-Align-DY')||0),Number(r.headers.get('X-Align-Response')||0)];
    // The left half shows the actual current output.  The right half is meant
    // to be the aligned candidate, so initialize it with the measured residual
    // instead of another unshifted copy of the current stereo view.
    $(video?'#vOverlayDx':'#overlayDx').value=st.alignAuto[0].toFixed(3);
    $(video?'#vOverlayDy':'#overlayDy').value=st.alignAuto[1].toFixed(3);
    video?vDrawOverlay():drawOverlay();
    $(video?'#vOverlayInfo':'#overlayInfo').textContent=`Projected-eye preview ready · candidate ${st.alignAuto[0].toFixed(2)}, ${st.alignAuto[1].toFixed(2)} px · response ${st.alignAuto[2].toFixed(3)}`;
    (video?vstatus:status)('Projected fused stereo-alignment preview ready.');
  }catch(e){
    if(requestId!==st.alignRequestId)return;
    drawOverlayMessage(canvas,'Alignment preview could not be displayed',e.message||String(e));
    (video?vstatus:status)(e.message||String(e),true)
  }
}
function useProjectedAuto(video=false){const st=video?vstate:state;$(video?'#vOverlayDx':'#overlayDx').value=st.alignAuto[0].toFixed(3);$(video?'#vOverlayDy':'#overlayDy').value=st.alignAuto[1].toFixed(3);video?vDrawOverlay():drawOverlay()}
async function applyProjectedCandidate(video=false){const st=video?vstate:state;if(!st.alignBitmap)return;const n=st.alignBitmap.height,dx=Number($(video?'#vOverlayDx':'#overlayDx').value||0),dy=Number($(video?'#vOverlayDy':'#overlayDy').value||0),sx=$(video?'#vStereoX':'#stereoX'),sy=$(video?'#vStereoY':'#stereoY');sx.value=(Number(sx.value||0)+dx*180/n).toFixed(5);sy.value=(Number(sy.value||0)+dy*180/n).toFixed(5);await refreshProjectedOverlay(video)}
function bindProjectedDrag(canvas,video=false){let drag=null;canvas.onpointerdown=e=>{const st=video?vstate:state;if(!st.alignBitmap)return;drag={x:e.clientX,y:e.clientY,dx:Number($(video?'#vOverlayDx':'#overlayDx').value||0),dy:Number($(video?'#vOverlayDy':'#overlayDy').value||0),sc:st.alignScale||1};canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;$(video?'#vOverlayDx':'#overlayDx').value=(drag.dx+(e.clientX-drag.x)/drag.sc).toFixed(3);$(video?'#vOverlayDy':'#overlayDy').value=(drag.dy+(e.clientY-drag.y)/drag.sc).toFixed(3);video?vDrawOverlay():drawOverlay()};canvas.onpointerup=canvas.onpointercancel=()=>drag=null}
function redraw(){drawSide('left');drawSide('right');drawOverlay()}
function syncCoords(){for(const lab of ['A','B'])for(const [si,side] of ['left','right'].entries()){const p=state.points[lab][si];const prefix=lab+(side==='left'?'L':'R');$('#'+prefix+'x').value=p?.[0]??'';$('#'+prefix+'y').value=p?.[1]??''}}
function readCoords(){for(const lab of ['A','B']){const vals=['Lx','Ly','Rx','Ry'].map(k=>Number($('#'+lab+k).value));state.points[lab]=[[vals[0],vals[1]],[vals[2],vals[3]]]}redraw()}
async function pick(e,side){const p=eventToSource(e,state[side]);if(!p)return;const idx=side==='left'?0:1;state.points[state.active][idx]=p;syncCoords();redraw();if(side==='left'&&$('#autoMatch').checked){try{const sid=await syncSession();status(`Matching ${state.active} on right…`);const m=await api('/api/match',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({session_id:sid,x:p[0],y:p[1],decoder:'opencv'})});state.points[state.active][1]=[m.x,m.y];syncCoords();redraw();status(`${state.active}: right (${m.x}, ${m.y}), SQDIFF_NORMED ${Number(m.sqdiff).toPrecision(6)}`)}catch(err){status(err.message,true)}}}
function goproPreset(){state.points={A:[[466,757],[455,767]],B:[[1286,823],[1271,832]]};$('#leftProfile').value=GOPRO_L;$('#rightProfile').value=GOPRO_R;syncCoords();redraw();refreshProfileInfo();status('GoPro calibrated test preset loaded.')}
function em10Preset(){state.points={A:[[1634,1513],[1669,1604]],B:[[3586,1579],[3600,1753]]};$('#leftProfile').value=EM10_L;$('#rightProfile').value=EM10_R;syncCoords();redraw();refreshProfileInfo();status('EM10 native-reference preset loaded.')}
async function render(){
  readCoords();try{setBusy(true);progress(.01);const sid=await syncSession();const fmt=$('#format').value,n=Math.trunc(+$('#outHeight').value);if(n>state.maxOutputHeight)throw new Error(`Server limit is ${state.maxOutputHeight} pixels high`);
    const payload={session_id:sid,left_profile:$('#leftProfile').value,right_profile:$('#rightProfile').value,points:state.points,output_height:n,roll:+$('#roll').value||0,pitch:+$('#pitch').value||0,yaw:+$('#yaw').value||0,right_shift_x_deg:+$('#stereoX').value||0,right_shift_y_deg:+$('#stereoY').value||0,format:fmt,jpeg_quality:Math.trunc(+$('#jpegQuality').value||95),output_name:$('#outputName').value||`LROut.${fmt==='png'?'png':'jpg'}`,decoder:'opencv'};
    status('Queueing render…');const q=await api('/api/render',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});await pollJob(q.job_id,progress,status);await loadImageResult(`/api/jobs/${q.job_id}/result`);setBusy(false);if($('#deleteInputs').checked)deleteSession();
  }catch(e){status(e.message,true);setBusy(false)}
}
async function loadImageResult(url){const r=await fetch(url);if(!r.ok)throw new Error('Could not fetch render result');const blob=await r.blob();if(state.resultURL)URL.revokeObjectURL(state.resultURL);state.resultURL=URL.createObjectURL(blob);state.resultName=$('#outputName').value||'LROut.jpg';const bmp=await createImageBitmap(blob);const c=$('#outputCanvas');c.width=bmp.width;c.height=bmp.height;c.getContext('2d').drawImage(bmp,0,0);$('#downloadBtn').disabled=false;$('#resultMeta').textContent=`${bmp.width} × ${bmp.height} · ${(blob.size/1024/1024).toFixed(2)} MB · ${blob.type}`;progress(1);status('Render complete.')}
function downloadImage(){if(!state.resultURL)return;downloadURL(state.resultURL,state.resultName)}

// -------------------------------- video ------------------------------------
async function vFileChosen(file,side){
  // Selecting the second movie now completes the entire preview setup. Earlier
  // versions merely remembered the filenames and left both reference canvases
  // blank until a separate Upload/Load sequence, which looked like a failure.
  vstate[side].file=file;
  $(side==='left'?'#vLeftName':'#vRightName').textContent=`${file.name} — ${(file.size/1024/1024).toFixed(1)} MB`;
  if(vstate.sessionId)await vDeleteSession();
  vstate.sessionVersion++;vstate.uploadedVersion=-1;
  vstate.leftInfo=vstate.rightInfo=null;vstate[side].bitmap=null;vRedraw();
  if(vstate.left.file&&vstate.right.file){
    try{
      vstatus('Uploading/probing pair and loading reference frames…');
      await vSyncSession();
      await vLoadRefs();
    }catch(e){vstatus(e.message,true)}
  }else{
    vstatus('Choose the other movie; the reference pair will display automatically.');
  }
}
async function vSyncSession(){
  if(!vstate.left.file||!vstate.right.file)throw new Error('Choose both movies first');
  if(vstate.sessionId&&vstate.uploadedVersion===vstate.sessionVersion)return vstate.sessionId;if(vstate.uploadPromise)return vstate.uploadPromise;
  vstate.uploadPromise=(async()=>{vstatus('Uploading and probing the movie pair…');vprogress(.01);const fd=new FormData();fd.append('left',vstate.left.file,vstate.left.file.name);fd.append('right',vstate.right.file,vstate.right.file.name);const s=await api('/api/video/session',{method:'POST',body:fd});vstate.sessionId=s.session_id;vstate.uploadedVersion=vstate.sessionVersion;vstate.leftInfo=s.left_info;vstate.rightInfo=s.right_info;vstate.left.width=s.left_info.width;vstate.left.height=s.left_info.height;vstate.right.width=s.right_info.width;vstate.right.height=s.right_info.height;
    $('#vLeftEnd').value=Math.max(0,s.left_info.frame_count-1);$('#vRightEnd').value=Math.max(0,s.right_info.frame_count-1);$('#vSessionInfo').textContent='Temporary video session ready';
    $('#vProbeInfo').textContent=`Left: ${probeText(s.left_info)} · Right: ${probeText(s.right_info)}`;vprogress(0);vstatus('Movies ready. Set reference frame numbers and load the synchronized frames.');await refreshVideoProfileInfo();return s.session_id})().finally(()=>vstate.uploadPromise=null);return vstate.uploadPromise;
}
function probeText(i){return `${i.width}×${i.height}, ${i.frame_count} frames, ${Number(i.fps).toFixed(3)} fps${i.fourcc?`, ${i.fourcc}`:''}`}
async function vDeleteSession(){if(!vstate.sessionId)return;const sid=vstate.sessionId;vstate.sessionId=null;vstate.uploadedVersion=-1;try{await fetch(`/api/session/${sid}`,{method:'DELETE'})}catch{}$('#vSessionInfo').textContent='Server movies deleted. They will re-upload if needed.'}
async function fetchBitmap(url){const r=await fetch(url);if(!r.ok){let msg=`${r.status} ${r.statusText}`;try{const d=await r.json();msg=d.detail||msg}catch{}throw new Error(msg)}const b=await r.blob();return createImageBitmap(b)}
async function vLoadRefs(){try{const sid=await vSyncSession();const lf=Math.trunc(+$('#vLeftRef').value),rf=Math.trunc(+$('#vRightRef').value);vstatus(`Loading reference frames L${lf} / R${rf}…`);const [lb,rb]=await Promise.all([fetchBitmap(`/api/video/frame/${sid}/L?frame=${lf}`),fetchBitmap(`/api/video/frame/${sid}/R?frame=${rf}`)]);if(vstate.left.bitmap?.close)vstate.left.bitmap.close();if(vstate.right.bitmap?.close)vstate.right.bitmap.close();vstate.left.bitmap=lb;vstate.right.bitmap=rb;vstate.left.width=lb.width;vstate.left.height=lb.height;vstate.right.width=rb.width;vstate.right.height=rb.height;const lw=$('#vLeftCanvas')?.closest('.canvasWrap'),rw=$('#vRightCanvas')?.closest('.canvasWrap');if(lw)lw.style.aspectRatio=`${lb.width}/${lb.height}`;if(rw)rw.style.aspectRatio=`${rb.width}/${rb.height}`;if(vstate.alignBitmap?.close)vstate.alignBitmap.close();vstate.alignBitmap=null;vRedraw();await refreshVideoProfileInfo();vstatus(`Reference frames L${lf} / R${rf} loaded. Preparing projected alignment preview…`);setTimeout(()=>refreshProjectedOverlay(true),0)}catch(e){vstatus(e.message,true)}}
function vStepReferencePair(delta){
  const l=$('#vLeftRef'),r=$('#vRightRef');
  l.value=Math.max(0,Math.trunc(+l.value||0)+delta);r.value=Math.max(0,Math.trunc(+r.value||0)+delta);
  vLoadRefs();
}
async function refreshVideoProfileInfo(){
  for(const side of ['left','right']){const info=side==='left'?vstate.leftInfo:vstate.rightInfo,out=$(side==='left'?'#vLeftProfileInfo':'#vRightProfileInfo'),sel=$(side==='left'?'#vLeftProfile':'#vRightProfile');if(!sel?.value||!info){out.textContent='';continue}try{const q=new URLSearchParams({name:sel.value,side:side==='left'?'L':'R',width:info.width,height:info.height});const p=await api('/api/profile-info?'+q);out.textContent=`axis (${p.axis[0].toFixed(2)}, ${p.axis[1].toFixed(2)}) · radius ${p.radius.toFixed(2)} · mode ${p.projection_mode} · k ${p.k.toFixed(4)}`+(p.calibration_warning?` · WARNING: ${p.calibration_warning}`:'')}catch(e){out.textContent=e.message}}
}
function vClampView(side){
  const s=vstate[side],v=s.view,b=s.bitmap,c=$(side==='left'?'#vLeftCanvas':'#vRightCanvas');if(!b||!c)return;
  const r=c.getBoundingClientRect(),fit=Math.min(r.width/b.width,r.height/b.height),sc=fit*Math.max(1,v.zoom);
  let cx=v.centerX*b.width,cy=v.centerY*b.height;
  if(b.width*sc<=r.width)cx=b.width/2;else{const h=r.width/(2*sc);cx=Math.max(h,Math.min(b.width-h,cx))}
  if(b.height*sc<=r.height)cy=b.height/2;else{const h=r.height/(2*sc);cy=Math.max(h,Math.min(b.height-h,cy))}
  v.centerX=cx/b.width;v.centerY=cy/b.height;
}
function vDrawSide(side){
  const canvas=$(side==='left'?'#vLeftCanvas':'#vRightCanvas'),s=vstate[side],bmp=s.bitmap,dpr=devicePixelRatio||1,r=canvas.getBoundingClientRect(),cw=Math.max(1,r.width),ch=Math.max(1,r.height);
  canvas.width=Math.round(cw*dpr);canvas.height=Math.round(ch*dpr);const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,cw,ch);c.fillStyle='#090b0f';c.fillRect(0,0,cw,ch);if(!bmp){s.transform=null;return}
  vClampView(side);const v=s.view,fit=Math.min(cw/bmp.width,ch/bmp.height),sc=fit*Math.max(1,v.zoom),cx=v.centerX*bmp.width,cy=v.centerY*bmp.height,dw=bmp.width*sc,dh=bmp.height*sc,dx=cw/2-cx*sc,dy=ch/2-cy*sc;
  c.drawImage(bmp,dx,dy,dw,dh);s.transform={scale:sc,dx,dy};const si=side==='left'?0:1;
  for(const lab of ['A','B']){const p=vstate.points[lab][si];if(!p)continue;const x=dx+p[0]*sc,y=dy+p[1]*sc;c.strokeStyle=lab==='A'?'#ff3030':'#00cfff';c.lineWidth=2;c.strokeRect(x-5,y-5,10,10);c.fillStyle=c.strokeStyle;c.font='bold 13px system-ui';c.fillText(lab,x+7,y-7)}
  const frame=Math.trunc(+(side==='left'?$('#vLeftRef').value:$('#vRightRef').value)||0),fmt=p=>p?`${Math.round(p[0])},${Math.round(p[1])}`:'—',A=vstate.points.A[si],B=vstate.points.B[si];
  c.font='600 12px system-ui';const line1=`${side==='left'?'L':'R'} frame ${frame}   zoom ${Math.round(v.zoom*100)}%`,line2=`Reference coordinates  A ${fmt(A)}   B ${fmt(B)}`;
  // Draw metadata directly over the frame with a one-pixel shadow.  No opaque
  // background is used, so synchronization details never hide the footage.
  c.fillStyle='#000';c.fillText(line1,13,29);c.fillStyle='#f3f7fb';c.fillText(line1,12,28);
  c.font='11px system-ui';c.fillStyle='#000';c.fillText(line2,13,46);c.fillStyle='#9eddf1';c.fillText(line2,12,45);
}
function vRedraw(){vDrawSide('left');vDrawSide('right');vDrawOverlay()}
function vSyncCoords(){for(const lab of ['A','B'])for(const [si,side] of ['left','right'].entries()){const p=vstate.points[lab][si];const prefix='v'+lab+(side==='left'?'L':'R');$('#'+prefix+'x').value=p?.[0]??'';$('#'+prefix+'y').value=p?.[1]??''}}
function vReadCoords(){for(const lab of ['A','B']){const vals=['Lx','Ly','Rx','Ry'].map(k=>Number($('#v'+lab+k).value));if(vals.some(x=>!Number.isFinite(x)))throw new Error(`Incomplete ${lab} coordinates`);vstate.points[lab]=[[vals[0],vals[1]],[vals[2],vals[3]]]}vRedraw()}
function vUpdateReferenceCursor(){
  const picking=['A','B'].includes(vstate.active);
  for(const id of ['#vLeftCanvas','#vRightCanvas'])$(id)?.classList.toggle('pointPick',picking);
}
function vSetNavigate(){vstate.active=null;const n=document.querySelector('input[name=vPickPoint][value=navigate]');if(n)n.checked=true;vUpdateReferenceCursor()}
async function vPick(e,side){
  if(!['A','B'].includes(vstate.active))return;const p=eventToSource(e,vstate[side]);if(!p)return;const lab=vstate.active,idx=side==='left'?0:1;vstate.points[lab][idx]=p;vSyncCoords();vRedraw();vSetNavigate();
  if(side==='left'&&$('#vAutoMatch').checked){try{const sid=await vSyncSession(),lf=Math.trunc(+$('#vLeftRef').value),rf=Math.trunc(+$('#vRightRef').value);vstatus(`Matching ${lab} on right reference frame…`);const m=await api('/api/video/match',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({session_id:sid,left_frame:lf,right_frame:rf,x:p[0],y:p[1]})});vstate.points[lab][1]=[m.x,m.y];vSyncCoords();vRedraw();vstatus(`${lab}: right (${m.x}, ${m.y}), SQDIFF ${Number(m.sqdiff).toPrecision(5)} · Navigate mode restored.`)}catch(err){vstatus(err.message,true)}}
}
function vFitReferenceViews(){for(const side of ['left','right']){vstate[side].view={zoom:1,centerX:.5,centerY:.5}}vRedraw()}
function vOneToOneViews(){for(const side of ['left','right']){const s=vstate[side],c=$(side==='left'?'#vLeftCanvas':'#vRightCanvas');if(!s.bitmap)continue;const r=c.getBoundingClientRect(),fit=Math.min(r.width/s.bitmap.width,r.height/s.bitmap.height);s.view.zoom=Math.max(1,1/Math.max(fit,1e-9))}vRedraw()}
function vReferenceWheel(e,side){e.preventDefault();const v=vstate[side].view;v.zoom=Math.max(1,Math.min(32,v.zoom*(e.deltaY<0?1.2:1/1.2)));vDrawSide(side)}
function vReferencePointerDown(e,side){
  if(['A','B'].includes(vstate.active)){vPick(e,side);return}
  const s=vstate[side];if(!s.bitmap)return;vstate.navDrag={side,x:e.clientX,y:e.clientY,cx:s.view.centerX,cy:s.view.centerY,scale:s.transform?.scale||1};e.currentTarget.setPointerCapture?.(e.pointerId)
}
function vReferencePointerMove(e,side){const d=vstate.navDrag,s=vstate[side];if(!d||d.side!==side||!s.bitmap)return;s.view.centerX=d.cx-(e.clientX-d.x)/(d.scale*s.bitmap.width);s.view.centerY=d.cy-(e.clientY-d.y)/(d.scale*s.bitmap.height);vClampView(side);vDrawSide(side)}
function vReferencePointerUp(){vstate.navDrag=null}
function vGoproPreset(){$('#vLeftProfile').value=GOPRO_L;$('#vRightProfile').value=GOPRO_R;vSyncCoords();vRedraw();refreshVideoProfileInfo();vstatus('GoPro calibrated L/R profiles selected. Existing A/B preserved. A/B are capture-specific: choose them from this video reference pair; the still-test coordinates are never copied into video.')}
function vEm10Preset(){$('#vLeftProfile').value=KINO_EM10;$('#vRightProfile').value=KINO_EM10;refreshVideoProfileInfo();vstatus('180Kino EM10 camera/lens profiles selected. Choose A/B from your synchronized reference frames.')}

function vTutorialPreset(){
  $('#vLeftRef').value=180;$('#vRightRef').value=186;
  $('#vLeftStart').value=180;$('#vLeftEnd').value=480;
  $('#vRightStart').value=186;$('#vRightEnd').value=486;
  $('#vLeftProfile').value='DJI Action2';$('#vRightProfile').value='DJI Action2';
  $('#vOutputWidth').value='8192';$('#vCodec').value='hevc';$('#vFpsMode').value='kino';$('#vLengthPolicy').value='strict';$('#vOutputName').value='VROut.mp4';
  vstate.points={A:[[1152,1272],[1202,1293]],B:[[3180,1492],[3227,1522]]};
  vSyncCoords();vRedraw();refreshVideoProfileInfo();
  vstatus('180Kino tutorial preset loaded: L180/R186, DJI Action2, tutorial A/B and conversion ranges.');
  if(vstate.left.bitmap&&vstate.right.bitmap)setTimeout(()=>refreshProjectedOverlay(true),0);
}
function vUseSyncLength(){if(!vstate.leftInfo||!vstate.rightInfo){vstatus('Upload / inspect the pair first.',true);return}const ls=Math.max(0,Math.trunc(+$('#vLeftStart').value)),rs=Math.max(0,Math.trunc(+$('#vRightStart').value));const n=Math.min(vstate.leftInfo.frame_count-ls,vstate.rightInfo.frame_count-rs);if(n<=0){vstatus('Start frame is outside a movie.',true);return}$('#vLeftEnd').value=ls+n-1;$('#vRightEnd').value=rs+n-1;$('#vLengthPolicy').value='trim';vstatus(`Trimmed to shorter remaining length: ${n} frame pairs. No frames are added.`)}
function vUseExtendedLength(){if(!vstate.leftInfo||!vstate.rightInfo){vstatus('Upload / inspect the pair first.',true);return}const ls=Math.max(0,Math.trunc(+$('#vLeftStart').value)),rs=Math.max(0,Math.trunc(+$('#vRightStart').value));if(ls>=vstate.leftInfo.frame_count||rs>=vstate.rightInfo.frame_count){vstatus('Start frame is outside a movie.',true);return}$('#vLeftEnd').value=vstate.leftInfo.frame_count-1;$('#vRightEnd').value=vstate.rightInfo.frame_count-1;$('#vLengthPolicy').value='repeat_last';const lc=vstate.leftInfo.frame_count-ls,rc=vstate.rightInfo.frame_count-rs;vstatus(`Using all remaining frames: left ${lc}, right ${rc}. Shorter side will repeat its last frame for ${Math.abs(lc-rc)} frame(s).`)}
function clipRangePayload(){const ls=Math.trunc(+$('#vLeftStart').value),le=Math.trunc(+$('#vLeftEnd').value),rs=Math.trunc(+$('#vRightStart').value),re=Math.trunc(+$('#vRightEnd').value);if([ls,le,rs,re].some(x=>!Number.isInteger(x)||x<0))throw new Error('Frame ranges must be non-negative integers');if(le<ls||re<rs)throw new Error('End frame must be >= start frame');return {left_start:ls,left_end:le,right_start:rs,right_end:re}}
function videoRangePayload(){const r=clipRangePayload(),lc=r.left_end-r.left_start+1,rc=r.right_end-r.right_start+1,policy=$('#vLengthPolicy').value||'strict';if(policy==='strict'&&lc!==rc)throw new Error(`Selected ranges differ (left ${lc}, right ${rc}). Choose Trim or an Extend mode.`);return {...r,length_policy:policy}}
async function vRender(){
  try{setVBusy(true);vprogress(.01);vReadCoords();const sid=await vSyncSession();const range=videoRangePayload();const payload={session_id:sid,left_profile:$('#vLeftProfile').value,right_profile:$('#vRightProfile').value,points:vstate.points,...range,output_width:Math.trunc(+$('#vOutputWidth').value),roll:+$('#vRoll').value||0,pitch:+$('#vPitch').value||0,yaw:+$('#vYaw').value||0,right_shift_x_deg:+$('#vStereoX').value||0,right_shift_y_deg:+$('#vStereoY').value||0,codec:$('#vCodec').value,fps_mode:$('#vFpsMode').value,sampling:$('#vSampling').value,output_name:$('#vOutputName').value||'VROut.mp4'};vstatus('Queueing video conversion…');const q=await api('/api/video/render',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});const j=await pollJob(q.job_id,vprogress,vstatus);await loadVideoResult(j.result_url,`/api/jobs/${q.job_id}/preview`,j.result);setVBusy(false);if($('#vDeleteInputs').checked)vDeleteSession();
  }catch(e){vstatus(e.message,true);setVBusy(false)}
}
async function vClip(){
  try{setVBusy(true);vprogress(.01);const sid=await vSyncSession();const range=clipRangePayload();const payload={session_id:sid,...range,jpeg_quality:95};vstatus('Queueing JPEG clipping…');const q=await api('/api/video/clip',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});const j=await pollJob(q.job_id,vprogress,vstatus);const r=await fetch(j.result_url);if(!r.ok)throw new Error('Could not fetch JPEG ZIP');const blob=await r.blob();if(vstate.clipURL)URL.revokeObjectURL(vstate.clipURL);vstate.clipURL=URL.createObjectURL(blob);vstate.clipName=j.result?.filename||'cut_img.zip';$('#vClipDownload').disabled=false;vprogress(1);vstatus(`JPEG clipping complete: ${j.result?.frames??''} files.`);setVBusy(false)
  }catch(e){vstatus(e.message,true);setVBusy(false)}
}
async function loadVideoResult(resultURL,previewURL,meta){
  // Keep the full-resolution output on the server until Download is requested.
  // The player/checker uses the H.264 proxy to avoid 8K HEVC browser stalls.
  vstate.resultURL=resultURL;vstate.previewURL=previewURL;vstate.resultName=meta?.filename||$('#vOutputName').value||'VROut.mp4';
  vstate.resultFrames=Math.max(0,Math.trunc(meta?.frames||0));vstate.resultFps=Number(meta?.fps||0);vstate.checkedFrame=0;
  const video=$('#vResultVideo');video.pause();video.src=previewURL;video.load();$('#vDownloadBtn').disabled=false;
  const fullMB=(Number(meta?.result_bytes||0)/1024/1024).toFixed(2),previewMB=(Number(meta?.preview_bytes||0)/1024/1024).toFixed(2);
  $('#vResultMeta').textContent=`${meta?.width??'?'} × ${meta?.height??'?'} · ${meta?.frames??'?'} frames · ${Number(meta?.fps||0).toFixed(3)} fps (${meta?.fps_mode||'source'}) · length ${meta?.length_policy||'strict'} [L ${meta?.left_frames??'?'} / R ${meta?.right_frames??'?'}] · codec ${meta?.codec||'?'} · full ${fullMB} MB · preview ${previewMB} MB · audio: no`;
  setupOutputFrameChecker();vprogress(1);
  if(meta?.preview_error)vstatus(`Render complete, but optimized preview failed: ${meta.preview_error}`,true);else vstatus('Video conversion complete. Player and frame inspector are ready.');
}
function setupOutputFrameChecker(){const max=Math.max(0,vstate.resultFrames-1),slider=$('#vCheckSlider'),input=$('#vCheckFrame');slider.max=max;slider.value=0;slider.disabled=vstate.resultFrames<1;input.max=max;input.value=0;input.disabled=vstate.resultFrames<1;$('#vCheckPrev').disabled=vstate.resultFrames<1;$('#vCheckNext').disabled=vstate.resultFrames<1;$('#vResumePlayback').disabled=vstate.resultFrames<1;$('#vCheckInfo').textContent=vstate.resultFrames?`Frame 0 / ${max}`:'No frame metadata available.';$('#vCheckCanvas').classList.add('hidden')}
function updateOutputFrameUi(frame){if(!vstate.resultFrames)return;frame=Math.max(0,Math.min(vstate.resultFrames-1,Math.trunc(frame)));vstate.checkedFrame=frame;$('#vCheckSlider').value=frame;$('#vCheckFrame').value=frame;$('#vCheckInfo').textContent=`Frame ${frame} / ${vstate.resultFrames-1}`+(vstate.resultFps>0?` · ${(frame/vstate.resultFps).toFixed(3)} s`:'')}
function seekCheckedOutputFrame(frame){if(!vstate.previewURL||!vstate.resultFrames)return;frame=Math.max(0,Math.min(vstate.resultFrames-1,Math.trunc(frame)));updateOutputFrameUi(frame);const video=$('#vResultVideo');video.pause();video.currentTime=vstate.resultFps>0?Math.min(video.duration||Infinity,(frame+.01)/vstate.resultFps):0}
function drawCheckedOutputFrame(){const video=$('#vResultVideo'),canvas=$('#vCheckCanvas');if(!video||!canvas||video.readyState<2)return;const vw=video.videoWidth||1,vh=video.videoHeight||1;canvas.width=vw;canvas.height=vh;canvas.getContext('2d').drawImage(video,0,0,vw,vh);canvas.classList.remove('hidden')}
function resumeOutputPlayback(){const video=$('#vResultVideo');$('#vCheckCanvas').classList.add('hidden');video.play().catch(()=>{})}
function syncPlaybackFrameUi(){const video=$('#vResultVideo');if(!vstate.resultFrames||!vstate.resultFps||video.seeking)return;updateOutputFrameUi(Math.min(vstate.resultFrames-1,Math.floor(video.currentTime*vstate.resultFps)))}
function downloadVideo(){if(vstate.resultURL)downloadURL(vstate.resultURL,vstate.resultName)}
function downloadClip(){if(vstate.clipURL)downloadURL(vstate.clipURL,vstate.clipName)}

async function pollJob(id,setProgress,setStatus){for(;;){const j=await api(`/api/jobs/${id}`);setProgress(j.progress||0);setStatus(`${j.stage||j.status} · ${Math.round((j.progress||0)*100)}%`);if(j.status==='finished')return j;if(j.status==='failed')throw new Error(j.error||'Job failed');await new Promise(r=>setTimeout(r,500))}}
function downloadURL(url,name){const a=document.createElement('a');a.href=url;a.download=name;a.click()}

function bind(){
  $('#imageTab').onclick=()=>setMode('image');$('#videoTab').onclick=()=>setMode('video');$('#infoTab').onclick=()=>setMode('info');
  $('#uiTheme').onchange=applyUiOptions;
  $('#openLeft').onclick=()=>$('#leftFile').click();$('#openRight').onclick=()=>$('#rightFile').click();$('#leftFile').onchange=e=>e.target.files[0]&&decodeLocal(e.target.files[0],'left');$('#rightFile').onchange=e=>e.target.files[0]&&decodeLocal(e.target.files[0],'right');
  $('#leftCanvas').onclick=e=>pick(e,'left');$('#rightCanvas').onclick=e=>pick(e,'right');$$('input[name=pickPoint]').forEach(x=>x.onchange=()=>{if(x.checked)state.active=x.value});$$('.coord').forEach(x=>x.onchange=readCoords);$('#goproPreset').onclick=goproPreset;$('#em10Preset').onclick=em10Preset;$('#clearPoints').onclick=()=>{state.points={A:[null,null],B:[null,null]};syncCoords();redraw()};$('#leftProfile').onchange=refreshProfileInfo;$('#rightProfile').onchange=refreshProfileInfo;$('#renderBtn').onclick=render;$('#downloadBtn').onclick=downloadImage;
  $('#format').onchange=()=>{$('#jpegBox').style.display=$('#format').value==='jpeg'?'flex':'none';const n=$('#outputName');if($('#format').value==='png'&&/\.jpe?g$/i.test(n.value))n.value=n.value.replace(/\.jpe?g$/i,'.png');if($('#format').value==='jpeg'&&/\.png$/i.test(n.value))n.value=n.value.replace(/\.png$/i,'.jpg')};
  $('#overlayAlpha').oninput=drawOverlay;$('#overlayMode').onchange=drawOverlay;$('#overlayDx').oninput=drawOverlay;$('#overlayDy').oninput=drawOverlay;$('#overlayRefresh').onclick=()=>refreshProjectedOverlay(false);$('#overlayAuto').onclick=()=>useProjectedAuto(false);$('#overlayApply').onclick=()=>applyProjectedCandidate(false);$('#overlayReset').onclick=()=>{$('#overlayDx').value=0;$('#overlayDy').value=0;drawOverlay()};

  $('#vOpenLeft').onclick=()=>$('#vLeftFile').click();$('#vOpenRight').onclick=()=>$('#vRightFile').click();$('#vLeftFile').onchange=e=>e.target.files[0]&&vFileChosen(e.target.files[0],'left');$('#vRightFile').onchange=e=>e.target.files[0]&&vFileChosen(e.target.files[0],'right');$('#vUploadBtn').onclick=async()=>{try{await vSyncSession();await vLoadRefs()}catch(e){vstatus(e.message,true)}};$('#vLoadRefs').onclick=vLoadRefs;$('#vPrevPair').onclick=()=>vStepReferencePair(-1);$('#vNextPair').onclick=()=>vStepReferencePair(1);$('#vSyncFull').onclick=vUseSyncLength;$('#vExtendFull').onclick=vUseExtendedLength;$('#vFitRefs').onclick=vFitReferenceViews;$('#vOneToOne').onclick=vOneToOneViews;$('#vClipBtn').onclick=vClip;$('#vClipDownload').onclick=downloadClip;
  for(const [side,sel] of [['left','#vLeftCanvas'],['right','#vRightCanvas']]){const c=$(sel);c.onpointerdown=e=>vReferencePointerDown(e,side);c.onpointermove=e=>vReferencePointerMove(e,side);c.onpointerup=c.onpointercancel=vReferencePointerUp;c.onwheel=e=>vReferenceWheel(e,side)}$$('input[name=vPickPoint]').forEach(x=>x.onchange=()=>{if(x.checked){vstate.active=(x.value==='navigate'?null:x.value);vUpdateReferenceCursor()}});vUpdateReferenceCursor();$$('.vcoord').forEach(x=>x.onchange=()=>{try{vReadCoords()}catch(e){vstatus(e.message,true)}});$('#vClearPoints').onclick=()=>{vstate.points={A:[null,null],B:[null,null]};vSyncCoords();vRedraw()};$('#vLeftProfile').onchange=refreshVideoProfileInfo;$('#vRightProfile').onchange=refreshVideoProfileInfo;$('#vGoproPreset').onclick=vGoproPreset;$('#vEm10Preset').onclick=vEm10Preset;$('#vTutorialPreset').onclick=vTutorialPreset;$('#vRenderBtn').onclick=vRender;$('#vDownloadBtn').onclick=downloadVideo;$('#vOverlayAlpha').oninput=vDrawOverlay;$('#vOverlayMode').onchange=vDrawOverlay;$('#vOverlayDx').oninput=vDrawOverlay;$('#vOverlayDy').oninput=vDrawOverlay;$('#vOverlayRefresh').onclick=()=>refreshProjectedOverlay(true);$('#vOverlayAuto').onclick=()=>useProjectedAuto(true);$('#vOverlayApply').onclick=()=>applyProjectedCandidate(true);$('#vOverlayReset').onclick=()=>{$('#vOverlayDx').value=0;$('#vOverlayDy').value=0;vDrawOverlay()};
  bindProjectedDrag($('#overlayCanvas'),false);bindProjectedDrag($('#vOverlayCanvas'),true);
  $('#vCheckPrev').onclick=()=>seekCheckedOutputFrame(vstate.checkedFrame-1);$('#vCheckNext').onclick=()=>seekCheckedOutputFrame(vstate.checkedFrame+1);$('#vCheckSlider').oninput=e=>seekCheckedOutputFrame(e.target.value);$('#vCheckFrame').onchange=e=>seekCheckedOutputFrame(e.target.value);$('#vResumePlayback').onclick=resumeOutputPlayback;$('#vResultVideo').addEventListener('seeked',drawCheckedOutputFrame);$('#vResultVideo').addEventListener('play',()=>$('#vCheckCanvas').classList.add('hidden'));$('#vResultVideo').addEventListener('timeupdate',syncPlaybackFrameUi);
  window.addEventListener('resize',()=>requestAnimationFrame(()=>{redraw();vRedraw()}));
  window.addEventListener('beforeunload',()=>{if(state.sessionId)fetch(`/api/session/${state.sessionId}`,{method:'DELETE',keepalive:true}).catch(()=>{});if(vstate.sessionId)fetch(`/api/session/${vstate.sessionId}`,{method:'DELETE',keepalive:true}).catch(()=>{})});
}

(async()=>{bind();restoreUiOptions();syncCoords();vSyncCoords();await health();await loadProfiles();requestAnimationFrame(()=>{redraw();vRedraw()})})().catch(e=>{status(e.message,true);vstatus(e.message,true)});
