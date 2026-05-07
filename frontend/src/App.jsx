import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

/* ── Design constants ───────────────────────────────────────────────────────── */
const NODE_COLORS = {
  paper:   "#e8a838", author:  "#5b8def", method:  "#ef5b5b",
  dataset: "#4ecdc4", metric:  "#a78bfa", concept: "#f97316",
  entity:  "#94a3b8", note:    "#34d399",
};
const NODE_SIZES = {
  paper:34, author:22, method:24, dataset:22, metric:18, concept:17, entity:16, note:28,
};
const ENTITY_META = {
  authors:      { icon:"👤", color:"#5b8def", label:"Authors" },
  methods:      { icon:"⚙️", color:"#ef5b5b", label:"Methods" },
  datasets:     { icon:"🗃️", color:"#4ecdc4", label:"Datasets" },
  key_concepts: { icon:"💡", color:"#f97316", label:"Key Concepts" },
};
const DEFAULT_VISIBLE = new Set(["paper","author","method","dataset"]);

/* ── SVG Icons ──────────────────────────────────────────────────────────────── */
const Icon = {
  upload: <svg className="pt-nav-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M10 13V7m0 0L7 10m3-3 3 3"/><path d="M4 16a4 4 0 0 1 0-8h.5A5.5 5.5 0 0 1 15.5 8H16a3 3 0 0 1 0 6H4z" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  graph:  <svg className="pt-nav-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="5" cy="5" r="2"/><circle cx="15" cy="5" r="2"/><circle cx="10" cy="15" r="2"/><path d="M7 5h6M6 7l3 6m5-6-3 6"/></svg>,
  ask:    <svg className="pt-nav-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M3 6a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H9l-4 3v-3H5a2 2 0 0 1-2-2V6z" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  library:<svg className="pt-nav-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M4 4h3v12H4zM8.5 4h3v12h-3zM13 4h3v12h-3z" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  trash:  <svg style={{width:13,height:13}} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7"><path d="M8 9v5m4-5v5M3 5h14l-1 11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2L3 5zm3-2h8" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  send:   <svg style={{width:15,height:15}} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M4 10l12-6-6 12-2-4-4-2z" strokeLinecap="round" strokeLinejoin="round"/></svg>,
};

/* ── Inline markdown ────────────────────────────────────────────────────────── */
function renderInline(text) {
  const parts = []; let s = String(text), k = 0;
  while (s.length) {
    const hits = [
      { re:/\*\*(.+?)\*\*/, t:"b" },
      { re:/\*(.+?)\*/,     t:"i" },
      { re:/`([^`]+)`/,     t:"c" },
    ].map(({re,t})=>{ const m=s.match(re); return m?{m,t,i:m.index}:null; })
     .filter(Boolean).sort((a,b)=>a.i-b.i);
    if (!hits.length) { parts.push(<span key={k++}>{s}</span>); break; }
    const {m,t,i} = hits[0];
    if (i>0) parts.push(<span key={k++}>{s.slice(0,i)}</span>);
    if (t==="b") parts.push(<strong key={k++} style={{color:"#f1f5f9",fontWeight:700}}>{m[1]}</strong>);
    if (t==="i") parts.push(<em key={k++} style={{color:"#c8d8f0"}}>{m[1]}</em>);
    if (t==="c") parts.push(<code key={k++} className="pt-md-code">{m[1]}</code>);
    s = s.slice(i+m[0].length);
  }
  return parts;
}

function MarkdownResponse({ text }) {
  if (!text) return null;
  const lines = text.split("\n");
  const out = [];
  let listItems=[], listOl=false, inCode=false, codeLines=[], codeLang="";

  const flushList = key => {
    if (!listItems.length) return;
    const Tag = listOl ? "ol" : "ul";
    out.push(
      <Tag key={`l${key}`} className={`pt-md-${listOl?"ol":"ul"}`}>
        {listItems.map((it,j)=><li key={j} className="pt-md-li">{renderInline(it)}</li>)}
      </Tag>
    );
    listItems=[];
  };

  lines.forEach((line,i) => {
    if (line.startsWith("```")) {
      if (inCode) {
        out.push(
          <pre key={`c${i}`} className="pt-md-pre">
            {codeLang && <div style={{color:"#334155",fontSize:10,marginBottom:8}}>{codeLang}</div>}
            <code>{codeLines.join("\n")}</code>
          </pre>
        );
        codeLines=[]; inCode=false; codeLang="";
      } else { flushList(i); inCode=true; codeLang=line.slice(3).trim(); }
      return;
    }
    if (inCode) { codeLines.push(line); return; }
    if (line.startsWith("### "))      { flushList(i); out.push(<div key={i} className="pt-md-h3">{renderInline(line.slice(4))}</div>); }
    else if (line.startsWith("## ")) { flushList(i); out.push(<div key={i} className="pt-md-h2">{renderInline(line.slice(3))}</div>); }
    else if (line.startsWith("# "))  { flushList(i); out.push(<div key={i} className="pt-md-h1">{renderInline(line.slice(2))}</div>); }
    else if (/^[-*_]{3,}$/.test(line.trim())) { flushList(i); out.push(<hr key={i} className="pt-md-hr"/>); }
    else if (line.startsWith("> "))  { flushList(i); out.push(<div key={i} className="pt-md-blockquote">{renderInline(line.slice(2))}</div>); }
    else if (/^[-*•]\s+/.test(line)) { if(listOl){flushList(i);} listOl=false; listItems.push(line.replace(/^[-*•]\s+/,"")); }
    else if (/^\d+\.\s+/.test(line)) { if(!listOl){flushList(i);} listOl=true; listItems.push(line.replace(/^\d+\.\s+/,"")); }
    else if (!line.trim())           { flushList(i); if(out.length) out.push(<div key={`sp${i}`} style={{height:5}}/>); }
    else { flushList(i); out.push(<div key={i} className="pt-md-p">{renderInline(line)}</div>); }
  });
  flushList("end");
  return <div>{out}</div>;
}

/* ── Extraction card ────────────────────────────────────────────────────────── */
function ExtractionCard({ result }) {
  if (!result) return null;
  const { title, pages, chunks, entities_found={}, entities_sample={} } = result;
  const rows = [
    { key:"authors",      count:entities_found.authors  ||0, sample:entities_sample.authors      ||[] },
    { key:"methods",      count:entities_found.methods  ||0, sample:entities_sample.methods      ||[] },
    { key:"datasets",     count:entities_found.datasets ||0, sample:entities_sample.datasets     ||[] },
    { key:"key_concepts", count:entities_found.concepts ||0, sample:entities_sample.key_concepts ||[] },
  ];
  return (
    <div className="pt-extract-card">
      <div className="pt-extract-header">
        <div className="pt-extract-status">✓ Indexed successfully</div>
        <div className="pt-extract-row1">
          <div className="pt-extract-title">{title}</div>
          <div className="pt-extract-counters">
            {pages && (
              <div className="pt-extract-count">
                <div className="pt-extract-count-val">{pages}</div>
                <div className="pt-extract-count-label">pages</div>
              </div>
            )}
            <div className="pt-extract-count">
              <div className="pt-extract-count-val">{chunks}</div>
              <div className="pt-extract-count-label">chunks</div>
            </div>
          </div>
        </div>
      </div>
      <div className="pt-extract-body">
        {rows.map(({key,count,sample}) => {
          if (!count && !sample.length) return null;
          const m = ENTITY_META[key];
          return (
            <div key={key} className="pt-extract-entity-row">
              <div className="pt-extract-entity-label">
                <span className="pt-entity-icon">{m.icon}</span>
                <span className="pt-entity-name" style={{color:m.color}}>{m.label}</span>
                <span className="pt-entity-count" style={{background:m.color+"18",color:m.color,border:`1px solid ${m.color}33`}}>{count}</span>
              </div>
              <div className="pt-chips-row">
                {sample.map((item,j)=>(
                  <span key={j} className="pt-entity-chip">
                    {typeof item==="object" ? item.name||item.value : item}
                  </span>
                ))}
                {count>sample.length && <span style={{fontSize:11,color:"var(--text-4)",padding:"3px 0"}}>+{count-sample.length} more</span>}
              </div>
            </div>
          );
        })}
        {(entities_found.relationships||0)>0 && (
          <div className="pt-extract-footer">
            🔗 <strong style={{color:"var(--green)"}}>{entities_found.relationships}</strong> relationships extracted into knowledge graph
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Force Graph ────────────────────────────────────────────────────────────── */
function ForceGraph({ graphData }) {
  const containerRef = useRef(null);
  const canvasRef    = useRef(null);
  const nodesRef     = useRef([]);
  const edgesRef     = useRef([]);
  const dragRef      = useRef(null);
  const hoveredRef   = useRef(null);
  const offsetRef    = useRef({x:0,y:0});
  const animRef      = useRef(null);
  const [tooltip, setTooltip]  = useState(null);
  const [dims, setDims]        = useState({width:800,height:520});
  const [visibleTypes, setVis] = useState(DEFAULT_VISIBLE);

  useEffect(()=>{
    if (!containerRef.current) return;
    const obs = new ResizeObserver(entries => {
      const w = Math.floor(entries[0].contentRect.width);
      if (w>0) setDims({width:w, height:Math.max(480,Math.floor(w*0.56))});
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  },[]);

  const {width,height} = dims;

  useEffect(()=>{
    if (!graphData?.nodes.length) return;
    if (animRef.current) cancelAnimationFrame(animRef.current);

    const visIds = new Set(graphData.nodes.filter(n=>visibleTypes.has(n.type)).map(n=>n.id));
    const fNodes = graphData.nodes.filter(n=>visIds.has(n.id));
    const fEdges = graphData.edges.filter(e=>visIds.has(e.source)&&visIds.has(e.target));
    const papers = fNodes.filter(n=>n.type==="paper");
    const others = fNodes.filter(n=>n.type!=="paper");

    const nodes = [
      ...papers.map((n,i)=>{
        const a=(i/Math.max(papers.length,1))*Math.PI*2-Math.PI/2, r=Math.min(130,papers.length*50);
        return {...n, x:width/2+Math.cos(a)*r, y:height/2+Math.sin(a)*r, vx:0, vy:0};
      }),
      ...others.map(n=>({...n, x:width/2+(Math.random()-.5)*width*.75, y:height/2+(Math.random()-.5)*height*.75, vx:0, vy:0})),
    ].map(n=>({...n, radius:NODE_SIZES[n.type]||18, color:NODE_COLORS[n.type]||"#94a3b8"}));

    const nodeMap={}; nodes.forEach(n=>nodeMap[n.id]=n);
    const edges = fEdges
      .filter(e=>nodeMap[e.source]&&nodeMap[e.target])
      .map(e=>({...e, sourceNode:nodeMap[e.source], targetNode:nodeMap[e.target]}));

    nodesRef.current=nodes; edgesRef.current=edges;
    let iter=0; const MAX=450;

    function arrow(ctx,x1,y1,x2,y2,r) {
      const dx=x2-x1, dy=y2-y1, len=Math.sqrt(dx*dx+dy*dy)||1, ux=dx/len, uy=dy/len;
      const tx=x2-ux*(r+2), ty=y2-uy*(r+2), al=7, aw=3.5;
      ctx.beginPath(); ctx.moveTo(tx,ty);
      ctx.lineTo(tx-ux*al+uy*aw, ty-uy*al-ux*aw);
      ctx.lineTo(tx-ux*al-uy*aw, ty-uy*al+ux*aw);
      ctx.closePath(); ctx.fill();
    }

    function pill(ctx,x,y,text,color) {
      ctx.font="11px 'Inter',system-ui,sans-serif";
      const tw=ctx.measureText(text).width, pw=tw+10, ph=17;
      ctx.fillStyle="rgba(7,11,20,0.88)";
      ctx.beginPath(); ctx.roundRect(x-pw/2,y-ph/2,pw,ph,4); ctx.fill();
      ctx.fillStyle=color; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(text,x,y);
    }

    function draw() {
      const canvas=canvasRef.current; if(!canvas)return;
      const ctx=canvas.getContext("2d");
      ctx.clearRect(0,0,width,height);
      ctx.fillStyle="rgba(26,40,68,0.2)";
      for(let gx=28;gx<width;gx+=28) for(let gy=28;gy<height;gy+=28){ctx.beginPath();ctx.arc(gx,gy,1,0,Math.PI*2);ctx.fill();}

      const hov=hoveredRef.current;
      for (const e of edgesRef.current) {
        const hi=hov&&(e.sourceNode===hov||e.targetNode===hov);
        ctx.strokeStyle=hi?"rgba(232,168,56,0.65)":"rgba(26,40,68,0.75)";
        ctx.lineWidth=hi?1.5:1;
        ctx.beginPath(); ctx.moveTo(e.sourceNode.x,e.sourceNode.y); ctx.lineTo(e.targetNode.x,e.targetNode.y); ctx.stroke();
        ctx.fillStyle=hi?"rgba(232,168,56,0.7)":"rgba(26,40,68,0.85)";
        arrow(ctx,e.sourceNode.x,e.sourceNode.y,e.targetNode.x,e.targetNode.y,e.targetNode.radius);
        if (hi&&e.relation) pill(ctx,(e.sourceNode.x+e.targetNode.x)/2,(e.sourceNode.y+e.targetNode.y)/2,e.relation,"#64748b");
      }
      for (const n of nodesRef.current) {
        const isPaper=n.type==="paper", isHov=n===hov, glowR=n.radius+(isHov?11:isPaper?7:4);
        const g=ctx.createRadialGradient(n.x,n.y,n.radius*.4,n.x,n.y,glowR);
        g.addColorStop(0,n.color+(isHov?"60":isPaper?"40":"22")); g.addColorStop(1,"transparent");
        ctx.beginPath(); ctx.arc(n.x,n.y,glowR,0,Math.PI*2); ctx.fillStyle=g; ctx.fill();
        const bg=ctx.createRadialGradient(n.x-n.radius*.22,n.y-n.radius*.22,0,n.x,n.y,n.radius);
        bg.addColorStop(0,n.color+"ff"); bg.addColorStop(1,n.color+"99");
        ctx.beginPath(); ctx.arc(n.x,n.y,n.radius,0,Math.PI*2); ctx.fillStyle=bg; ctx.fill();
        ctx.strokeStyle=isHov?n.color:n.color+"77"; ctx.lineWidth=isHov?2.5:isPaper?2:1.5; ctx.stroke();
        ctx.fillStyle="rgba(255,255,255,0.92)"; ctx.font=`bold ${Math.floor(n.radius*.68)}px 'JetBrains Mono',monospace`;
        ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(n.type[0].toUpperCase(),n.x,n.y);
        if (isPaper||n.type==="author"||isHov) {
          const lbl=(n.label?.length>22)?n.label.slice(0,22)+"…":(n.label||n.id);
          pill(ctx,n.x,n.y+n.radius+12,lbl,isPaper?"#e8a838":isHov?"#f0f4ff":"#8fa3c8");
        }
      }
    }

    function tick() {
      const ns=nodesRef.current, es=edgesRef.current;
      for(let i=0;i<ns.length;i++){
        ns[i]._fx=0; ns[i]._fy=0;
        for(let j=0;j<ns.length;j++){
          if(i===j)continue;
          const dx=ns[i].x-ns[j].x, dy=ns[i].y-ns[j].y, d2=dx*dx+dy*dy+.01, d=Math.sqrt(d2);
          const f=4500/d2; ns[i]._fx+=(dx/d)*f; ns[i]._fy+=(dy/d)*f;
        }
      }
      for(const e of es){
        const dx=e.targetNode.x-e.sourceNode.x, dy=e.targetNode.y-e.sourceNode.y;
        const d=Math.sqrt(dx*dx+dy*dy)||.1, f=(d-140)*.007, fx=(dx/d)*f, fy=(dy/d)*f;
        e.sourceNode._fx+=fx; e.sourceNode._fy+=fy; e.targetNode._fx-=fx; e.targetNode._fy-=fy;
      }
      const cap=Math.max(.3,(1-iter/MAX)*14);
      for(const n of ns){
        if(n===dragRef.current)continue;
        const f=Math.sqrt(n._fx*n._fx+n._fy*n._fy)||1, d=Math.min(f,cap);
        n.x+=(n._fx/f)*d; n.y+=(n._fy/f)*d;
        n.x+=(width/2-n.x)*.004; n.y+=(height/2-n.y)*.004;
        const p=n.radius+24; n.x=Math.max(p,Math.min(width-p,n.x)); n.y=Math.max(p,Math.min(height-p,n.y));
      }
      draw(); iter++;
      if(iter<MAX||dragRef.current) animRef.current=requestAnimationFrame(tick);
    }
    tick();
    return ()=>{ if(animRef.current) cancelAnimationFrame(animRef.current); };
  },[graphData,visibleTypes,width,height]);

  const toCanvas = e => {
    const r=canvasRef.current.getBoundingClientRect();
    return {x:(e.clientX-r.left)*(width/r.width), y:(e.clientY-r.top)*(height/r.height)};
  };
  const hit = p => nodesRef.current.find(n=>(p.x-n.x)**2+(p.y-n.y)**2<n.radius**2)||null;
  const onDown = e => { const n=hit(toCanvas(e)); if(n){dragRef.current=n; const p=toCanvas(e); offsetRef.current={x:p.x-n.x,y:p.y-n.y};} };
  const onMove = e => {
    const p=toCanvas(e);
    if(dragRef.current){dragRef.current.x=p.x-offsetRef.current.x; dragRef.current.y=p.y-offsetRef.current.y; dragRef.current.vx=dragRef.current.vy=0;}
    const h=hit(p); hoveredRef.current=h;
    setTooltip(h?{x:e.clientX,y:e.clientY,text:`${h.type}: ${h.label||h.id}`}:null);
  };
  const onUp = () => { dragRef.current=null; };

  const typeCounts={};
  graphData?.nodes.forEach(n=>{ typeCounts[n.type]=(typeCounts[n.type]||0)+1; });

  return (
    <div className="pt-graph-page">
      <div className="pt-graph-filters">
        {Object.entries(NODE_COLORS).map(([type,color])=>{
          const count=typeCounts[type]||0; if(!count)return null;
          const on=visibleTypes.has(type);
          return (
            <button key={type} className="pt-type-toggle"
              style={{color:on?color:"var(--text-4)",background:on?color+"14":"transparent",borderColor:on?color+"55":"var(--border-2)"}}
              onClick={()=>setVis(p=>{const n=new Set(p);n.has(type)?n.delete(type):n.add(type);return n;})}>
              <span className="pt-type-dot" style={{background:on?color:"var(--border-3)"}}/>
              {type}
              <span style={{opacity:.5,fontFamily:"var(--font-mono)",fontSize:10}}>{count}</span>
            </button>
          );
        })}
      </div>

      <div ref={containerRef} style={{position:"relative",borderRadius:14,overflow:"hidden",background:"var(--bg-base)",border:"1px solid var(--border-1)",boxShadow:"var(--shadow-m)"}}>
        <canvas ref={canvasRef} width={width} height={height}
          onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
          style={{display:"block",width:"100%",height:"auto",cursor:"crosshair"}}/>
        {tooltip && (
          <div style={{position:"fixed",left:tooltip.x+14,top:tooltip.y-10,background:"var(--bg-card)",color:"var(--text-1)",padding:"7px 12px",borderRadius:8,fontSize:12,pointerEvents:"none",border:"1px solid var(--border-2)",zIndex:999,boxShadow:"var(--shadow-m)",maxWidth:260}}>
            {tooltip.text}
          </div>
        )}
      </div>
      <div style={{fontSize:11,color:"var(--text-4)",textAlign:"center",marginTop:4}}>
        Drag nodes · Hover to reveal labels and edge relations · {nodesRef.current.length} nodes shown
      </div>
    </div>
  );
}

/* ── Main App ───────────────────────────────────────────────────────────────── */
export default function PaperTrail() {
  const [activeTab,    setTab]          = useState("upload");
  const [papers,       setPapers]       = useState([]);
  const [graphData,    setGraph]        = useState({nodes:[],edges:[]});
  const [stats,        setStats]        = useState(null);
  const [uploading,    setUploading]    = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError,  setUploadError]  = useState(null);
  const [question,     setQuestion]     = useState("");
  const [chatHistory,  setChat]         = useState([]);
  const [querying,     setQuerying]     = useState(false);
  const [noteTitle,    setNoteTitle]    = useState("");
  const [noteContent,  setNoteContent]  = useState("");
  const [noteStatus,   setNoteStatus]   = useState(null);
  const [globalError,  setGlobalError]  = useState(null);
  const [isDragging,   setIsDragging]   = useState(false);
  const chatEndRef   = useRef(null);
  const fileInputRef = useRef(null);

  const fetchData = useCallback(async()=>{
    try {
      const [pR,gR,sR] = await Promise.all([fetch(`${API_BASE}/papers`),fetch(`${API_BASE}/graph`),fetch(`${API_BASE}/stats`)]);
      if(pR.ok) setPapers((await pR.json()).papers);
      if(gR.ok) setGraph(await gR.json());
      if(sR.ok) setStats(await sR.json());
      setGlobalError(null);
    } catch { setGlobalError("Cannot connect to backend. Is it running on port 8000?"); }
  },[]);

  useEffect(()=>{ fetchData(); },[fetchData]);
  useEffect(()=>{ chatEndRef.current?.scrollIntoView({behavior:"smooth"}); },[chatHistory]);

  const processFile = async file => {
    if(!file) return;
    if(!file.name.toLowerCase().endsWith(".pdf")){ setUploadError("Only PDF files are supported."); return; }
    setUploading(true); setUploadResult(null); setUploadError(null);
    const fd=new FormData(); fd.append("file",file);
    try {
      const res=await fetch(`${API_BASE}/upload`,{method:"POST",body:fd});
      const data=await res.json();
      if(!res.ok) setUploadError(res.status===429?"Rate limited — try again in a moment.":(data.detail||"Upload failed."));
      else { setUploadResult(data); fetchData(); }
    } catch(e){ setUploadError("Upload failed: "+e.message); }
    setUploading(false);
    if(fileInputRef.current) fileInputRef.current.value="";
  };

  const onFileChange = e => processFile(e.target.files?.[0]);
  const onDragOver   = e => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave  = ()=> setIsDragging(false);
  const onDrop       = e => { e.preventDefault(); setIsDragging(false); processFile(e.dataTransfer.files?.[0]); };

  const handleNote = async()=>{
    if(!noteTitle.trim()||!noteContent.trim()) return;
    setNoteStatus(null);
    try {
      const res=await fetch(`${API_BASE}/note`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:noteTitle,content:noteContent})});
      if(!res.ok){ const d=await res.json().catch(()=>{}); setNoteStatus({ok:false,msg:d?.detail||"Failed."}); return; }
      setNoteTitle(""); setNoteContent("");
      setNoteStatus({ok:true,msg:"Note added to your library!"}); fetchData();
      setTimeout(()=>setNoteStatus(null),4000);
    } catch(e){ setNoteStatus({ok:false,msg:"Error: "+e.message}); }
  };

  const handleQuery = async()=>{
    if(!question.trim()||querying) return;
    const q=question;
    setChat(p=>[...p,{role:"user",text:q}]); setQuestion(""); setQuerying(true);
    try {
      const res=await fetch(`${API_BASE}/query`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
      const data=await res.json();
      if(!res.ok) setChat(p=>[...p,{role:"assistant",text:res.status===429?"Rate limited — wait a moment and retry.":(data.detail||"Query failed."),isError:true}]);
      else setChat(p=>[...p,{role:"assistant",text:data.answer,sources:data.sources||[],confidence:data.confidence,followUp:data.follow_up_questions||[]}]);
    } catch { setChat(p=>[...p,{role:"assistant",text:"Error: Could not reach the backend.",isError:true}]); }
    setQuerying(false);
  };

  const handleReset = async()=>{
    if(!window.confirm("Reset everything? All papers and knowledge graph will be cleared.")) return;
    try {
      const res=await fetch(`${API_BASE}/reset`,{method:"DELETE"});
      if(!res.ok){ setGlobalError("Reset failed."); return; }
      setChat([]); setUploadResult(null); setUploadError(null); fetchData();
    } catch { setGlobalError("Reset failed."); }
  };

  const NAV = [
    {id:"upload",  label:"Upload",          icon:Icon.upload},
    {id:"graph",   label:"Knowledge Graph", icon:Icon.graph},
    {id:"ask",     label:"Ask Library",     icon:Icon.ask},
    {id:"library", label:"Library",         icon:Icon.library},
  ];

  return (
    <div className="pt-app">
      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <aside className="pt-sidebar">
        <div className="pt-sidebar-logo">
          <div className="pt-logo-icon">P</div>
          <div className="pt-logo-text">
            <span className="pt-logo-name">PaperTrail</span>
            <span className="pt-logo-sub">Research Memory Agent</span>
          </div>
        </div>

        <nav className="pt-nav">
          {NAV.map(n=>(
            <button key={n.id} className={`pt-nav-item${activeTab===n.id?" active":""}`} onClick={()=>setTab(n.id)}>
              {n.icon}{n.label}
            </button>
          ))}
        </nav>

        <div className="pt-sidebar-stats">
          {stats && (
            <div className="pt-stats-grid">
              <div className="pt-stat-tile">
                <div className="pt-stat-val">{stats.papers}</div>
                <div className="pt-stat-label">Papers</div>
              </div>
              <div className="pt-stat-tile">
                <div className="pt-stat-val">{stats.graph_nodes}</div>
                <div className="pt-stat-label">Nodes</div>
              </div>
              <div className="pt-stat-tile">
                <div className="pt-stat-val">{stats.graph_edges}</div>
                <div className="pt-stat-label">Edges</div>
              </div>
              <div className="pt-stat-tile">
                <div className="pt-stat-val">{stats.vector_chunks}</div>
                <div className="pt-stat-label">Chunks</div>
              </div>
            </div>
          )}
          <button className="pt-reset-btn" onClick={handleReset}>
            {Icon.trash} Reset all data
          </button>
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────────────────────────── */}
      <div className="pt-main">
        {globalError && <div className="pt-global-error">{globalError}</div>}

        {/* Upload */}
        {activeTab==="upload" && (
          <div className="pt-page">
            <div className="pt-page-title">Upload a Paper</div>

            <div className="pt-card">
              <div className="pt-card-label">PDF File</div>
              <label className={`pt-upload-zone${isDragging?" drag":""}`}
                onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
                <input ref={fileInputRef} type="file" accept=".pdf" onChange={onFileChange} style={{display:"none"}}/>
                <div className="pt-upload-icon">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#e8a838" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                </div>
                <div className="pt-upload-title">
                  {uploading ? "Processing your paper…" : isDragging ? "Release to upload" : "Drop a PDF here, or click to browse"}
                </div>
                <div className="pt-upload-sub">
                  Extracts text · identifies entities · builds knowledge graph connections
                </div>
              </label>

              {uploading && (
                <div className="pt-alert warning">
                  <div className="pt-dot-pulse"><span/><span/><span/></div>
                  <span>Running entity recognition — this takes 15–30 seconds…</span>
                </div>
              )}
              {uploadError && !uploading && <div className="pt-alert error">✗ {uploadError}</div>}
              {uploadResult && !uploading && !uploadError && <ExtractionCard result={uploadResult}/>}
            </div>

            <div className="pt-card">
              <div className="pt-card-label">Add a Note</div>
              <input className="pt-input" style={{marginBottom:10}} placeholder="Note title…"
                value={noteTitle} onChange={e=>setNoteTitle(e.target.value)}/>
              <textarea className="pt-textarea" placeholder="Paste insights, key concepts, or text fragments to index…"
                value={noteContent} onChange={e=>setNoteContent(e.target.value)}/>
              <div style={{display:"flex",alignItems:"center",gap:12,marginTop:12}}>
                <button className="pt-btn pt-btn-primary" onClick={handleNote}
                  disabled={!noteTitle.trim()||!noteContent.trim()}>
                  Add Note
                </button>
                {noteStatus && (
                  <span style={{fontSize:12.5,color:noteStatus.ok?"var(--green)":"var(--red)"}}>
                    {noteStatus.ok?"✓":"✗"} {noteStatus.msg}
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Graph */}
        {activeTab==="graph" && (
          graphData.nodes.length>0
            ? <ForceGraph graphData={graphData}/>
            : <div className="pt-page">
                <div className="pt-page-title">Knowledge Graph</div>
                <div className="pt-card">
                  <div className="pt-empty">
                    <div className="pt-empty-icon">🕸️</div>
                    <div className="pt-empty-title">No graph yet</div>
                    <div className="pt-empty-sub">Upload a PDF to build the knowledge graph.</div>
                  </div>
                </div>
              </div>
        )}

        {/* Ask */}
        {activeTab==="ask" && (
          <div className="pt-chat-wrap">
            <div className="pt-chat-page-header">
              <div className="pt-page-title" style={{marginBottom:4}}>Ask Your Library</div>
              <div style={{fontSize:12,color:"var(--text-4)",marginBottom:0}}>
                Queries run across the knowledge graph and vector store simultaneously
              </div>
            </div>

            <div className="pt-chat-messages">
              {chatHistory.length===0 && (
                <div className="pt-empty" style={{paddingTop:60}}>
                  <div className="pt-empty-icon">🔍</div>
                  <div className="pt-empty-title">Ask anything about your papers</div>
                  <div className="pt-empty-sub">
                    "Which papers use attention mechanisms?"<br/>
                    "Compare methods across all papers"<br/>
                    "What datasets appear most frequently?"
                  </div>
                </div>
              )}

              {chatHistory.map((msg,i)=>(
                <div key={i}>
                  <div className={`pt-chat-row${msg.role==="user"?" user":""}`}>
                    <div className={`pt-chat-avatar${msg.role==="user"?" user-av":" ai"}`}>
                      {msg.role==="user" ? "U" : "P"}
                    </div>
                    <div className={`pt-chat-bubble${msg.role==="user"?" user":" ai"}`}>
                      {msg.role==="user"
                        ? <div style={{whiteSpace:"pre-wrap"}}>{msg.text}</div>
                        : <MarkdownResponse text={msg.text}/>
                      }

                      {msg.sources?.length>0 && (
                        <div className="pt-chat-sources">
                          <div className="pt-sources-label">Sources</div>
                          {msg.sources.map((s,si)=>(
                            <div key={si} className="pt-source-item">
                              <span className="pt-source-paper">{s.paper_title}</span>
                              {s.relevant_detail && <span className="pt-source-detail"> — {s.relevant_detail}</span>}
                            </div>
                          ))}
                        </div>
                      )}

                      {msg.confidence!=null && !msg.isError && (
                        <div className="pt-confidence-bar-wrap">
                          <div className="pt-confidence-track">
                            <div className="pt-confidence-fill" style={{
                              width:`${Math.round(msg.confidence*100)}%`,
                              background:msg.confidence>.7?"var(--green)":msg.confidence>.4?"var(--accent)":"var(--red)"
                            }}/>
                          </div>
                          <span className="pt-confidence-label">{Math.round(msg.confidence*100)}% confidence</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {msg.followUp?.length>0 && (
                    <div className="pt-followup-chips">
                      {msg.followUp.map((fq,fi)=>(
                        <button key={fi} className="pt-chip" onClick={()=>setQuestion(fq)}>{fq}</button>
                      ))}
                    </div>
                  )}
                </div>
              ))}

              {querying && (
                <div className="pt-chat-row">
                  <div className="pt-chat-avatar ai">P</div>
                  <div className="pt-chat-bubble ai">
                    <div className="pt-chat-typing">
                      <div className="pt-dot-pulse"><span/><span/><span/></div>
                      Searching graph + vector store…
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef}/>
            </div>

            <div className="pt-chat-input-bar">
              <input className="pt-input" style={{flex:1}}
                placeholder="Ask a question about your papers…"
                value={question}
                onChange={e=>setQuestion(e.target.value)}
                onKeyDown={e=>e.key==="Enter"&&!e.shiftKey&&handleQuery()}/>
              <button className="pt-btn pt-btn-primary" onClick={handleQuery} disabled={querying}
                style={{opacity:querying?.4:1,cursor:querying?"not-allowed":"pointer"}}>
                {Icon.send} Send
              </button>
            </div>
          </div>
        )}

        {/* Library */}
        {activeTab==="library" && (
          <div className="pt-page">
            <div className="pt-page-title">
              Library{" "}
              <span style={{color:"var(--text-4)",fontSize:14,fontWeight:500}}>({papers.length})</span>
            </div>
            <div className="pt-card">
              {papers.length===0
                ? <div className="pt-empty">
                    <div className="pt-empty-icon">📚</div>
                    <div className="pt-empty-title">No papers yet</div>
                    <div className="pt-empty-sub">Upload PDFs from the Upload tab to start building your library.</div>
                  </div>
                : papers.map(p=>(
                  <div key={p.id} className="pt-paper-row">
                    <div className="pt-paper-icon">{p.type==="note"?"📝":"📄"}</div>
                    <div style={{flex:1,minWidth:0}}>
                      <div className="pt-paper-title">{p.title}</div>
                      <div className="pt-paper-meta">
                        <span className="pt-badge" style={{background:(NODE_COLORS[p.type]||"#64748b")+"18",color:NODE_COLORS[p.type]||"#64748b"}}>{p.type}</span>
                        <span className="pt-paper-meta-dot"/>
                        {p.pages && <><span>{p.pages} pages</span><span className="pt-paper-meta-dot"/></>}
                        <span style={{fontFamily:"var(--font-mono)",fontSize:11}}>{p.chunks} chunks</span>
                        <span className="pt-paper-meta-dot"/>
                        <span>{new Date(p.uploaded_at).toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}</span>
                      </div>
                    </div>
                  </div>
                ))
              }
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
