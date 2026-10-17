const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
function harness(){
 const elements=new Map();
 const element=()=>({value:'0',textContent:'',children:[],hidden:false,disabled:false,duration:20,currentTime:0,
   classList:{add(){},remove(){},toggle(){}},addEventListener(){},pause(){},load(){},
   replaceChildren(...children){this.children=children;},append(...children){this.children.push(...children);},checkValidity(){return true;}});
 const get=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);};
 const context=vm.createContext({document:{getElementById:get,querySelector:()=>({content:'token'}),createElement:element},window:{addEventListener(){}},localStorage:{setItem(){}},setTimeout,clearTimeout,console});
 const source=fs.readFileSync('studio/static/studio.js','utf8').split('boot().catch(fail);')[0];
 vm.runInContext(source,context);
 return {context,get,run:code=>vm.runInContext(code,context)};
}
test('late project response cannot replace the latest selection',async()=>{
 const h=harness();
 h.run(`pending={}; api=path=>path.endsWith('/waveform')?Promise.resolve([]):new Promise(resolve=>pending[path]=resolve); loadProjects=async()=>{};refreshJobs=async()=>{};draw=()=>{};`);
 const first=h.run("selectProject('first')");const second=h.run("selectProject('second')");
 h.run("pending['/projects/second']({id:'second',title:'Second',settings:defaults,asset:{id:'two',duration:20,sample_rate:8000,channels:2}})");await second;
 h.run("pending['/projects/first']({id:'first',title:'First',settings:defaults,asset:{id:'one',duration:20,sample_rate:8000,channels:2}})");await first;
 assert.equal(h.run('project.id'),'second');
});
test('empty project clears previous render actions',async()=>{
 const h=harness();h.get('jobs').children=['previous output'];
 h.run("api=async path=>path==='/jobs'?[]:{queue_paused:false};");
 await h.run('refreshJobs()');
 assert.equal(h.get('jobs').children.includes('previous output'),false);
});
test('render submission keeps the track selected when clicked',async()=>{
 const h=harness();
 h.run("project={id:'first',asset:{duration:20}}; captured=[]; save=()=>new Promise(resolve=>finishSave=resolve);api=async(path,options)=>{captured.push(path);return {};};refreshJobs=async()=>{};");
 const submitted=h.run("submit('final')");
 h.run("project={id:'second',asset:{duration:20}};finishSave();");
 await submitted;
 assert.equal(h.run('captured[0]'),'/projects/first/renders');
});
test('changing settings retains preview offset during return to original',()=>{
 const h=harness();
 h.run("mode='processed';preview={options:{start_seconds:60}};observed=null;switchMode=async()=>{observed=preview?.options.start_seconds;};save=async()=>{};invalidate();clearTimeout(saveTimer);");
 assert.equal(h.run('observed'),60);
});
