import ExtractionCard from "./ExtractionCard";

/* ── Upload tab: PDF drop zone, URL ingest, and note form ───────────────────── */
export default function UploadPanel({
  uploading, uploadResult, uploadError, isDragging,
  fileInputRef, onFileChange, onDragOver, onDragLeave, onDrop,
  pdfUrl, setPdfUrl, onUrlUpload,
  noteTitle, setNoteTitle, noteContent, setNoteContent, noteStatus, onAddNote,
}) {
  return (
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
            onKeyDown={e=>e.key==="Enter" && onUrlUpload()}/>
          <button className="pt-btn pt-btn-primary"
            onClick={onUrlUpload}
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
          <button className="pt-btn pt-btn-primary" onClick={onAddNote}
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
  );
}
