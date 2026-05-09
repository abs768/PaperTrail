import { useState, useEffect, useRef, useCallback } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";

cytoscape.use(fcose);

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

/* ── Design constants ───────────────────────────────────────────────────────── */
const NODE_COLORS = {
  paper:   "#e8a838", author:  "#5b8def", method:  "#ef5b5b",
  dataset: "#4ecdc4", metric:  "#a78bfa", concept: "#f97316",
  entity:  "#94a3b8", note:    "#34d399",
};
const NODE_SIZES = {
  paper:50, author:22, method:26, dataset:24, metric:20, concept:18, entity:16, note:38,
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

/* ── Cytoscape Graph ────────────────────────────────────────────────────────── */
function CytoscapeGraph({ graphData }) {
  const containerRef = useRef(null);
  const cyRef        = useRef(null);
  const [tooltip, setTooltip]  = useState(null);
  const [visibleTypes, setVis] = useState(DEFAULT_VISIBLE);

  useEffect(() => {
    if (!containerRef.current || !graphData?.nodes?.length) return;

    const visibleNodeIds = new Set(graphData.nodes.filter(n => visibleTypes.has(n.type)).map(n => n.id));
    const elements = [
      ...graphData.nodes
        .filter(n => visibleNodeIds.has(n.id))
        .map(n => ({
          data: {
            id: n.id,
            label: n.label || n.id,
            type: n.type,
            color: NODE_COLORS[n.type] || "#94a3b8",
            size: NODE_SIZES[n.type] || 18,
          },
        })),
      ...graphData.edges
        .filter(e => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target))
        .map((e, i) => ({
          data: {
            id: `e${i}`,
            source: e.source,
            target: e.target,
            label: e.relation || "",
          },
        })),
    ];

    if (cyRef.current) cyRef.current.destroy();

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      wheelSensitivity: 0.25,
      minZoom: 0.2,
      maxZoom: 3,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            "border-color": "data(color)",
            "border-width": 1.5,
            "border-opacity": 0.85,
            "background-opacity": 0.95,
            label: ele => {
              const t = ele.data("type");
              const lbl = ele.data("label") || "";
              if (t === "paper" || t === "author" || t === "note") {
                return lbl.length > 26 ? lbl.slice(0, 26) + "…" : lbl;
              }
              return lbl.length > 18 ? lbl.slice(0, 18) + "…" : lbl;
            },
            color: "#dbe4f5",
            "font-size": 11,
            "font-family": "Inter, system-ui, sans-serif",
            "text-valign": "bottom",
            "text-halign": "center",
            "text-margin-y": 6,
            "text-background-color": "rgba(7,11,20,0.85)",
            "text-background-opacity": 1,
            "text-background-padding": 3,
            "text-background-shape": "round-rectangle",
            "text-border-color": "rgba(26,40,68,0.6)",
            "text-border-opacity": 1,
            "text-border-width": 0.5,
            width: "data(size)",
            height: "data(size)",
            "overlay-opacity": 0,
            "transition-property": "opacity, border-width, border-color",
            "transition-duration": "180ms",
          },
        },
        {
          selector: "node[type='paper'], node[type='note']",
          style: {
            shape: "round-rectangle",
            "border-width": 3,
            "border-opacity": 1,
            "font-weight": 700,
            "font-size": 13,
            color: "#f1d28a",
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.2,
            // Edge color follows the source node's type so each paper "radiates"
            // its color outward into its entity satellites. Low alpha keeps the
            // graph readable when zoomed out.
            "line-color": ele => {
              const c = NODE_COLORS[ele.source().data("type")] || "#475569";
              return c + "66"; // ~40% alpha
            },
            "target-arrow-color": ele => {
              const c = NODE_COLORS[ele.source().data("type")] || "#475569";
              return c + "99"; // ~60% alpha
            },
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.9,
            "curve-style": "bezier",
            "control-point-step-size": 25,
            opacity: 1,
            "transition-property": "opacity, width, line-color",
            "transition-duration": "180ms",
          },
        },
        {
          selector: "node:selected",
          style: {
            "border-width": 4,
            "border-color": "#ffd166",
            "border-opacity": 1,
          },
        },
        {
          selector: "node.faded, edge.faded",
          style: {
            opacity: 0.12,
          },
        },
        {
          selector: "node.focused",
          style: {
            "border-width": 4,
            "border-color": "#ffd166",
          },
        },
        {
          selector: "edge.highlight",
          style: {
            "line-color": "rgba(255, 209, 102, 0.95)",
            "target-arrow-color": "rgba(255, 209, 102, 0.95)",
            width: 2.2,
            opacity: 1,
            label: "data(label)",
            "font-size": 10,
            "text-background-color": "rgba(7,11,20,0.95)",
            "text-background-opacity": 1,
            "text-background-padding": 3,
            "text-rotation": "autorotate",
            color: "#fde7a7",
          },
        },
      ],
      layout: {
        name: "fcose",
        animate: true,
        animationDuration: 700,
        animationEasing: "ease-out",
        randomize: true,
        idealEdgeLength: 95,
        nodeRepulsion: 8000,
        nodeSeparation: 90,
        gravity: 0.3,
        gravityRangeCompound: 1.2,
        padding: 40,
      },
    });

    cy.on("mouseover", "node", evt => {
      const n = evt.target;
      const neighborhood = n.closedNeighborhood();
      // Fade everything outside the hovered node's neighborhood so the focus is
      // unmistakable. Highlight the connecting edges with the relation labels.
      cy.elements().difference(neighborhood).addClass("faded");
      n.connectedEdges().addClass("highlight");
      n.addClass("focused");
      const pos = n.renderedPosition();
      const r = containerRef.current.getBoundingClientRect();
      setTooltip({
        x: r.left + pos.x,
        y: r.top + pos.y - 30,
        text: `${n.data("type")}: ${n.data("label")}`,
      });
    });
    cy.on("mouseout", "node", evt => {
      cy.elements().removeClass("faded");
      evt.target.removeClass("focused");
      evt.target.connectedEdges().removeClass("highlight");
      setTooltip(null);
    });

    cyRef.current = cy;
    return () => { cy.destroy(); cyRef.current = null; };
  }, [graphData, visibleTypes]);

  const typeCounts = {};
  graphData?.nodes.forEach(n => { typeCounts[n.type] = (typeCounts[n.type] || 0) + 1; });
  const visibleCount = graphData?.nodes.filter(n => visibleTypes.has(n.type)).length || 0;

  const fit = () => cyRef.current?.fit(undefined, 30);
  const recenter = () => cyRef.current?.center();

  return (
    <div className="pt-graph-page">
      <div className="pt-graph-filters">
        {Object.entries(NODE_COLORS).map(([type, color]) => {
          const count = typeCounts[type] || 0;
          if (!count) return null;
          const on = visibleTypes.has(type);
          return (
            <button key={type} className="pt-type-toggle"
              style={{ color: on ? color : "var(--text-4)", background: on ? color + "14" : "transparent", borderColor: on ? color + "55" : "var(--border-2)" }}
              onClick={() => setVis(p => { const n = new Set(p); n.has(type) ? n.delete(type) : n.add(type); return n; })}>
              <span className="pt-type-dot" style={{ background: on ? color : "var(--border-3)" }} />
              {type}
              <span style={{ opacity: 0.5, fontFamily: "var(--font-mono)", fontSize: 10 }}>{count}</span>
            </button>
          );
        })}
        <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          <button className="pt-type-toggle" onClick={fit} title="Fit graph to view">⤢ Fit</button>
          <button className="pt-type-toggle" onClick={recenter} title="Recenter">◎ Center</button>
        </div>
      </div>

      <div style={{
        position: "relative",
        borderRadius: 14,
        overflow: "hidden",
        // Subtle radial gradient on the canvas backdrop — gives depth so the
        // graph nodes feel like they're floating in a dark gallery rather than
        // sitting on a flat panel.
        background: "radial-gradient(ellipse 70% 60% at 50% 45%, rgba(232,168,56,0.06) 0%, rgba(91,141,239,0.04) 35%, var(--bg-base) 75%)",
        backgroundColor: "var(--bg-base)",
        border: "1px solid var(--border-1)",
        boxShadow: "var(--shadow-m), inset 0 0 80px rgba(0,0,0,0.35)",
      }}>
        <div ref={containerRef} style={{ width: "100%", height: "max(520px, 60vh)" }} />
        {tooltip && (
          <div style={{ position: "fixed", left: tooltip.x + 14, top: tooltip.y - 10, background: "var(--bg-card)", color: "var(--text-1)", padding: "7px 12px", borderRadius: 8, fontSize: 12, pointerEvents: "none", border: "1px solid var(--border-2)", zIndex: 999, boxShadow: "var(--shadow-m)", maxWidth: 260 }}>
            {tooltip.text}
          </div>
        )}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-4)", textAlign: "center", marginTop: 4 }}>
        Drag nodes · Scroll to zoom · Hover for tooltips and edge labels · {visibleCount} nodes shown
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
  const [pdfUrl,       setPdfUrl]       = useState("");
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

  const handleUrlUpload = async () => {
    const u = pdfUrl.trim();
    if (!u || uploading) return;
    setUploading(true); setUploadResult(null); setUploadError(null);
    try {
      const res = await fetch(`${API_BASE}/upload-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: u }),
      });
      const data = await res.json();
      if (!res.ok) setUploadError(res.status === 429 ? "Rate limited — try again in a moment." : (data.detail || "URL upload failed."));
      else { setUploadResult(data); setPdfUrl(""); fetchData(); }
    } catch (e) { setUploadError("URL upload failed: " + e.message); }
    setUploading(false);
  };

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
      else setChat(p=>[...p,{role:"assistant",text:data.answer,sources:data.sources||[],confidence:data.confidence,followUp:data.follow_up_questions||[],unsupported:data.unsupported_claims||[]}]);
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

              <div style={{display:"flex",alignItems:"center",gap:8,margin:"14px 0 10px"}}>
                <div style={{flex:1,height:1,background:"var(--border-2)"}}/>
                <span style={{fontSize:11,color:"var(--text-4)",letterSpacing:1.5,textTransform:"uppercase"}}>or paste a link</span>
                <div style={{flex:1,height:1,background:"var(--border-2)"}}/>
              </div>
              <div style={{display:"flex",gap:8}}>
                <input className="pt-input" style={{flex:1}}
                  placeholder="arXiv URL or direct PDF link…  e.g. https://arxiv.org/abs/1706.03762"
                  value={pdfUrl}
                  onChange={e=>setPdfUrl(e.target.value)}
                  onKeyDown={e=>e.key==="Enter" && handleUrlUpload()}/>
                <button className="pt-btn pt-btn-primary"
                  onClick={handleUrlUpload}
                  disabled={!pdfUrl.trim()||uploading}>
                  Fetch & Index
                </button>
              </div>

              {uploading && (
                <div className="pt-alert warning" style={{marginTop:12}}>
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
            ? <CytoscapeGraph graphData={graphData}/>
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
                              <div>
                                <span className="pt-source-paper">{s.paper_title}</span>
                                {s.page != null && (
                                  <span style={{
                                    marginLeft:6,padding:"1px 6px",borderRadius:4,
                                    background:"rgba(232,168,56,0.12)",border:"1px solid rgba(232,168,56,0.3)",
                                    color:"#e8a838",fontFamily:"var(--font-mono)",fontSize:10
                                  }}>p.{s.page}</span>
                                )}
                                {s.verified && (
                                  <span title="Quote verified against source chunk" style={{
                                    marginLeft:6,padding:"1px 6px",borderRadius:4,
                                    background:"rgba(60,180,120,0.12)",border:"1px solid rgba(60,180,120,0.3)",
                                    color:"#3cb478",fontFamily:"var(--font-mono)",fontSize:10
                                  }}>✓ verified</span>
                                )}
                                {s.relevant_detail && <span className="pt-source-detail"> — {s.relevant_detail}</span>}
                              </div>
                              {s.quote && (
                                <div style={{
                                  marginTop:4,paddingLeft:10,
                                  borderLeft:"2px solid rgba(232,168,56,0.4)",
                                  fontStyle:"italic",color:"var(--muted)",fontSize:12
                                }}>
                                  "{s.quote}"
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {msg.unsupported?.length>0 && (
                        <div style={{
                          marginTop:8,padding:"8px 10px",borderRadius:6,
                          background:"rgba(220,80,80,0.08)",border:"1px solid rgba(220,80,80,0.25)"
                        }}>
                          <div style={{fontSize:11,color:"#dc5050",fontWeight:600,marginBottom:4}}>
                            ⚠ Claims not supported by retrieved passages
                          </div>
                          <ul style={{margin:0,paddingLeft:16,fontSize:12,color:"var(--muted)"}}>
                            {msg.unsupported.map((c,i)=><li key={i} style={{marginTop:2}}>{c}</li>)}
                          </ul>
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
