'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="studio-token"]').content;
const keys = ['rotation_cpm','motion_depth','room_size','crossover_hz'];
const defaults = {rotation_cpm:5.78,motion_depth:1,room_size:0.18,crossover_hz:150,panning_preset:'binaural_8d'};
let project=null, projects=[], jobs=[], peaks=[], mode='original', preview=null, saveTimer=null, saveChain=Promise.resolve(), revision=0, switching=0;
const audio=$('audio');
function notice(message){$('notice').textContent=message; $('notice').hidden=!message;}
function fail(error){notice(error.message || String(error));}
async function api(path, options={}){
  const headers={...options.headers};
  if(options.method && options.method!=='GET') headers['X-Studio-Token']=token;
  if(options.body && !(options.body instanceof FormData)) {headers['Content-Type']='application/json';options.body=JSON.stringify(options.body);}
  const response=await fetch('/api'+path,{...options,headers});
  if(!response.ok){let data;try{data=await response.json();}catch{data={detail:response.statusText};}throw Error(typeof data.detail==='string'?data.detail: 'Some settings are invalid. Check their allowed ranges.');}
  return response.json();
}
const seconds=n=>`${Math.floor((n||0)/60)}:${String(Math.floor((n||0)%60)).padStart(2,'0')}`;
function settings(){const s={panning_preset:$('preset').value};for(const k of keys){s[k]=Number($(k+'-number').value);if(!$(k+'-number').checkValidity()||!Number.isFinite(s[k]))throw Error('Check the value for '+k.replaceAll('_',' '));}return s;}
function showSettings(s){for(const k of keys)$(k).value=$(k+'-number').value=s[k];$('preset').value=s.panning_preset;}
function matching(job){return project && job.project_id===project.id && JSON.stringify(job.settings)===JSON.stringify(project.settings);}
function sameSettings(a,b){return Object.keys(defaults).every(k=>a[k]===b[k]);}
async function save(){
  clearTimeout(saveTimer);
  if(!project)return;
  const id=project.id, value=settings(), rev=revision;
  const request=saveChain.catch(()=>{}).then(()=>api('/projects/'+id+'/settings',{method:'PATCH',body:value}));
  saveChain=request;
  const updated=await request;
  if(project?.id===id && revision===rev){project.settings=updated.settings;$('save-state').textContent='Settings saved locally';}
}
function invalidate(){revision++;if(mode==='processed')switchMode('original').catch(fail);preview=null;$('processed').disabled=true;$('preview-note').textContent='Settings changed. Create a new preview to hear them.';$('save-state').textContent='Saving…';clearTimeout(saveTimer);saveTimer=setTimeout(()=>save().catch(fail),450);}
for(const key of keys){
  $(key).addEventListener('input',()=>{$(key+'-number').value=$(key).value;invalidate();});
  $(key+'-number').addEventListener('change',()=>{if($(key+'-number').checkValidity())$(key).value=$(key+'-number').value;invalidate();});
}
$('preset').addEventListener('change',invalidate);
$('reset').onclick=()=>{showSettings(defaults);invalidate();};
$('preview-start').onchange=()=>{if(mode==='processed')switchMode('original').catch(fail);preview=null;$('processed').disabled=true;$('preview-note').textContent='Preview region changed. Create a new preview.';};
function draw(){const canvas=$('waveform');const rect=canvas.getBoundingClientRect();const scale=devicePixelRatio||1;canvas.width=Math.max(1,rect.width*scale);canvas.height=150*scale;const ctx=canvas.getContext('2d');ctx.scale(scale,scale);const w=rect.width,h=150;ctx.clearRect(0,0,w,h);if(!peaks.length)return;const progress=project?(mode==='processed'?(Number(preview?.options.start_seconds||0)+audio.currentTime):audio.currentTime)/project.asset.duration:0;const count=Math.min(peaks.length,Math.floor(w/3));for(let i=0;i<count;i++){const p=peaks[Math.floor(i*peaks.length/count)];const amp=Math.max(.015,Math.max(Math.abs(p[0]),Math.abs(p[1])));const height=Math.min(125,amp*125);ctx.fillStyle=i/count<progress?'#b7f36b':'#7d779b';ctx.fillRect(i*w/count,75-height/2,Math.max(1,w/count-1),height);}}
window.addEventListener('resize',draw);
async function loadProjects(){projects=await api('/projects');$('projects').replaceChildren();for(const p of projects){const b=document.createElement('button');b.textContent='♫  '+p.title;b.title=p.title;b.className=project?.id===p.id?'active':'';b.onclick=()=>selectProject(p.id).catch(fail);$('projects').append(b);}if(!projects.length){const p=document.createElement('p');p.className='muted';p.textContent='Your tracks will appear here.';$('projects').append(p);}}
async function selectProject(id){
  const generation=++switching;
  if(project)await save();
  if(generation!==switching)return;
  audio.pause();const selected=await api('/projects/'+id);
  if(generation!==switching)return;
  project=selected;
  preview=null;mode='original';revision++;
  showSettings(project.settings);$('track-title').textContent=project.title;
  $('track-meta').textContent=seconds(project.asset.duration)+' · '+project.asset.sample_rate.toLocaleString()+' Hz · '+project.asset.channels+' channels';
  $('source-spec').textContent='SOURCE PRESERVED';$('import').textContent='＋ Import another';
  $('wave-empty').hidden=true;$('play').disabled=$('preview').disabled=$('render').disabled=$('seek').disabled=false;
  $('processed').disabled=true;$('original').classList.add('selected');$('processed').classList.remove('selected');
  $('save-state').textContent='Settings saved locally';$('preview-start').value=0;$('preview-start').max=Math.max(0,Math.floor(project.asset.duration-.01));
  $('preview-note').textContent='Audition a short excerpt before rendering the full track.';
  audio.src='/api/assets/'+project.asset.id;audio.load();
  const wave=await api('/projects/'+id+'/waveform');if(generation!==switching)return;peaks=wave;draw();
  localStorage.setItem('8d-project',id);await loadProjects();await refreshJobs();
}
async function importFile(file){if(!file)return;if(file.size>200*1024*1024)throw Error('Choose a file of 200 MB or less.');notice('Importing and preparing your audio…');$('import').disabled=true;try{const form=new FormData();form.append('file',file);const p=await api('/projects',{method:'POST',body:form});await selectProject(p.id);notice('');}finally{$('import').disabled=false;$('file').value='';}}
for(const id of ['import','new-project'])$(id).onclick=()=>$('file').click();
$('file').onchange=()=>importFile($('file').files[0]).catch(fail);
$('dropzone').ondragover=e=>{e.preventDefault();$('dropzone').classList.add('drag');};
$('dropzone').ondragleave=()=>$('dropzone').classList.remove('drag');
$('dropzone').ondrop=e=>{e.preventDefault();$('dropzone').classList.remove('drag');importFile(e.dataTransfer.files[0]).catch(fail);};
async function switchMode(next){
  if(!project||next===mode)return;
  if(next==='processed'&&!preview)return;
  const wasPlaying=!audio.paused, start=Number(preview?.options.start_seconds||0);
  const absolute=mode==='processed'?audio.currentTime+start:audio.currentTime;
  const desired=next==='processed'?Math.max(0,Math.min(absolute-start,(preview.report?.duration||15)-.01)):absolute;
  audio.pause();mode=next;audio.src=next==='processed'?'/api/jobs/'+preview.id+'/download':'/api/assets/'+project.asset.id;
  $('original').classList.toggle('selected',next==='original');$('processed').classList.toggle('selected',next==='processed');
  audio.load();await new Promise((resolve,reject)=>{const done=()=>{cleanup();resolve();},bad=()=>{cleanup();reject(Error('Your browser could not play this audio. Download the WAV to audition it.'));};const cleanup=()=>{audio.removeEventListener('loadedmetadata',done);audio.removeEventListener('error',bad);};audio.addEventListener('loadedmetadata',done,{once:true});audio.addEventListener('error',bad,{once:true});});
  audio.currentTime=Math.min(desired,Math.max(0,audio.duration-.01));if(wasPlaying)await audio.play();draw();
}
$('original').onclick=()=>switchMode('original').catch(fail);$('processed').onclick=()=>switchMode('processed').catch(fail);
$('play').onclick=()=>{if(audio.paused)audio.play().catch(()=>notice('Playback could not start. Try downloading the WAV.'));else audio.pause();};
audio.onplay=()=>$('play').textContent='Ⅱ';audio.onpause=()=>$('play').textContent='▶';
audio.ontimeupdate=()=>{$('clock').textContent=seconds(audio.currentTime)+' / '+seconds(audio.duration);$('seek').max=Number.isFinite(audio.duration)?audio.duration:1;$('seek').value=audio.currentTime;draw();};
audio.onloadedmetadata=()=>audio.ontimeupdate();audio.onerror=()=>notice('Browser playback is unavailable for this file. You can still render and download it.');
$('seek').oninput=()=>audio.currentTime=Number($('seek').value);
async function submit(kind){if(!project)return;const selected=project;const format=kind==='preview'?'wav':$('format').value;const start=Number($('preview-start').value);await save();if(!Number.isFinite(start)||start<0||start>=selected.asset.duration)throw Error('Choose a preview start inside the track.');await api('/projects/'+selected.id+'/renders',{method:'POST',body:{kind,format,start_seconds:start,duration_seconds:15}});notice(kind==='preview'?'Preview queued. It will appear below when ready.':'Full-track render queued. Your settings are saved with this render.');await refreshJobs();}
$('preview').onclick=()=>submit('preview').catch(fail);$('render').onclick=()=>submit('final').catch(fail);
$('resume').onclick=()=>api('/queue/resume',{method:'POST'}).then(refreshJobs).catch(fail);
function element(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
async function refreshJobs(){
  const [all,health]=await Promise.all([api('/jobs'),api('/health')]);jobs=all;
  $('connection').textContent=health.status==='ready'?'● Local engine ready':'Storage unavailable — check free disk space';$('resume').hidden=!health.queue_paused;
  const list=jobs.filter(j=>!project||j.project_id===project.id).reverse();$('job-count').textContent=list.length+' renders';
  const compatible=list.find(j=>j.kind==='preview'&&j.status==='completed'&&project&&sameSettings(j.settings,settings())&&j.options.start_seconds===Number($('preview-start').value));
  if(compatible){preview=compatible;$('processed').disabled=false;$('preview-note').textContent='Preview ready. Switch to 8D preview to compare. Excerpts may differ slightly from full-track mastering.';}
  $('jobs').replaceChildren();
  if(!list.length){$('jobs').append(element('p','muted','No renders for this project yet.'));return;}
  for(const j of list){const row=element('div','job'),head=element('div','job-head');head.append(element('span','',j.kind==='preview'?'15-second preview':'Full track · '+j.options.format.toUpperCase()),element('span','muted',j.cancel_requested&&j.status==='running'?'Stopping…':j.status));row.append(head);if(['queued','running'].includes(j.status)){const bar=element('progress');bar.max=1;bar.value=j.progress;row.append(bar);}
    if(j.error)row.append(element('small','',j.error));
    if(j.report){const r=j.report;row.append(element('small','',`${r.sample_rate.toLocaleString()} Hz · ${seconds(r.duration)} · peak ${r.sample_peak.toFixed(3)} · correlation ${r.correlation.toFixed(2)}`));}
    const actions=element('div','job-actions');
    if(j.status==='completed'){const a=element('a','',j.kind==='preview'?'Download preview':'Download '+j.options.format.toUpperCase());a.href='/api/jobs/'+j.id+'/download';a.download='';actions.append(a);}
    if(['queued','running','failed','interrupted','cancelled'].includes(j.status)){const action=['queued','running'].includes(j.status)?'cancel':'retry';const b=element('button','secondary',action==='cancel'?'Cancel':'Retry');b.disabled=!!j.cancel_requested&&j.status==='running';b.onclick=()=>api('/jobs/'+j.id+'/'+action,{method:'POST'}).then(refreshJobs).catch(fail);actions.append(b);}
    row.append(actions);$('jobs').append(row);
  }
}
async function boot(){const health=await api('/health');$('connection').textContent='● Local engine ready';$('format').querySelector('[value="mp3"]').disabled=!health.ffmpeg;await loadProjects();const id=localStorage.getItem('8d-project');if(projects.length)await selectProject(projects.some(p=>p.id===id)?id:projects[0].id);await refreshJobs();}
boot().catch(fail);
setInterval(()=>{if(!document.hidden)refreshJobs().catch(()=>{$('connection').textContent='Engine disconnected';});},1500);
