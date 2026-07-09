import { useState, useEffect, useRef, useCallback } from "react";

import { API_BASE } from "./constants";
import Sidebar from "./components/Sidebar";
import UploadPanel from "./components/UploadPanel";
import CytoscapeGraph from "./components/CytoscapeGraph";
import ChatPanel from "./components/ChatPanel";
import LibraryPanel from "./components/LibraryPanel";

/* ── Main App: owns all server state and handlers, renders per-tab panels ───── */
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
  const [queryStage,   setQueryStage]   = useState(null);
  const [noteTitle,    setNoteTitle]    = useState("");
  const [noteContent,  setNoteContent]  = useState("");
  const [noteStatus,   setNoteStatus]   = useState(null);
  const [globalError,  setGlobalError]  = useState(null);
  const [isDragging,   setIsDragging]   = useState(false);
  const [pdfUrl,       setPdfUrl]       = useState("");
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
    setChat(p=>[...p,{role:"user",text:q}]); setQuestion(""); setQuerying(true); setQueryStage(null);
    try {
      // /query/stream is SSE: progress events while the pipeline runs, then one result event.
      const res=await fetch(`${API_BASE}/query/stream`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
      if(!res.ok||!res.body) throw new Error(res.status===429?"Rate limited — wait a moment and retry.":`Query failed (${res.status}).`);
      const reader=res.body.getReader();
      const decoder=new TextDecoder();
      let buf="", result=null, errMsg=null;
      for(;;){
        const {done,value}=await reader.read();
        if(done) break;
        buf+=decoder.decode(value,{stream:true});
        const frames=buf.split("\n\n"); buf=frames.pop();
        for(const frame of frames){
          const line=frame.split("\n").find(l=>l.startsWith("data: "));
          if(!line) continue;
          let evt; try { evt=JSON.parse(line.slice(6)); } catch { continue; }
          if(evt.event==="progress") setQueryStage(evt.stage);
          else if(evt.event==="result") result=evt.data;
          else if(evt.event==="error") errMsg=evt.message;
        }
      }
      if(errMsg) throw new Error(errMsg);
      if(!result) throw new Error("Stream ended without a result.");
      setChat(p=>[...p,{role:"assistant",text:result.answer,sources:result.sources||[],confidence:result.confidence,followUp:result.follow_up_questions||[],unsupported:result.unsupported_claims||[]}]);
    } catch(e){ setChat(p=>[...p,{role:"assistant",text:e?.message||"Error: Could not reach the backend.",isError:true}]); }
    setQuerying(false); setQueryStage(null);
  };

  const handleDeletePaper = async p => {
    if(!window.confirm(`Delete "${p.title}"? Its chunks and graph entities will be removed.`)) return;
    try {
      const res=await fetch(`${API_BASE}/papers/${encodeURIComponent(p.id)}`,{method:"DELETE"});
      if(!res.ok){ const d=await res.json().catch(()=>{}); setGlobalError(d?.detail||"Delete failed."); return; }
      setGlobalError(null); fetchData();
    } catch { setGlobalError("Delete failed: could not reach the backend."); }
  };

  const handleReset = async()=>{
    if(!window.confirm("Reset everything? All papers and knowledge graph will be cleared.")) return;
    try {
      const res=await fetch(`${API_BASE}/reset`,{method:"DELETE"});
      if(!res.ok){ setGlobalError("Reset failed."); return; }
      setChat([]); setUploadResult(null); setUploadError(null); fetchData();
    } catch { setGlobalError("Reset failed."); }
  };

  return (
    <div className="pt-app">
      <Sidebar activeTab={activeTab} onNavigate={setTab} stats={stats} onReset={handleReset}/>

      <div className="pt-main">
        {globalError && <div className="pt-global-error">{globalError}</div>}

        {activeTab==="upload" && (
          <UploadPanel
            uploading={uploading} uploadResult={uploadResult} uploadError={uploadError} isDragging={isDragging}
            fileInputRef={fileInputRef} onFileChange={onFileChange}
            onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
            pdfUrl={pdfUrl} setPdfUrl={setPdfUrl} onUrlUpload={handleUrlUpload}
            noteTitle={noteTitle} setNoteTitle={setNoteTitle}
            noteContent={noteContent} setNoteContent={setNoteContent}
            noteStatus={noteStatus} onAddNote={handleNote}
          />
        )}

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

        {activeTab==="ask" && (
          <ChatPanel
            chatHistory={chatHistory} question={question} setQuestion={setQuestion}
            querying={querying} queryStage={queryStage} onSend={handleQuery}
          />
        )}

        {activeTab==="library" && (
          <LibraryPanel papers={papers} onDelete={handleDeletePaper}/>
        )}
      </div>
    </div>
  );
}
