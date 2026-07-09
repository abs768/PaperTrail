import { NODE_COLORS } from "../constants";
import { Icon } from "./Icons";

/* ── Library tab: indexed papers and notes ──────────────────────────────────── */
export default function LibraryPanel({ papers, onDelete }) {
  return (
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
              <button className="pt-paper-delete" title={`Delete ${p.type}`}
                onClick={()=>onDelete(p)}>
                {Icon.trash}
              </button>
            </div>
          ))
        }
      </div>
    </div>
  );
}
