import { useState } from "react";

import { API_BASE, NODE_COLORS, ENTITY_META } from "../constants";
import { Icon } from "./Icons";

/* ── Detail drawer: extracted entities for one paper ────────────────────────── */
function PaperDetails({ details }) {
  if (!details) return <div className="pt-paper-details">Loading details…</div>;
  if (details.error) return <div className="pt-paper-details">✗ {details.error}</div>;
  const entities = details.entities || {};
  const rows = Object.entries(ENTITY_META)
    .map(([key, meta]) => ({ meta, items: entities[key] || [] }))
    .filter(r => r.items.length);
  const metrics = entities.metrics || [];
  if (!rows.length && !metrics.length) {
    return <div className="pt-paper-details">No entities were extracted for this item.</div>;
  }
  return (
    <div className="pt-paper-details">
      {rows.map(({meta, items}) => (
        <div key={meta.label} className="pt-extract-entity-row">
          <div className="pt-extract-entity-label">
            <span className="pt-entity-icon">{meta.icon}</span>
            <span className="pt-entity-name" style={{color:meta.color}}>{meta.label}</span>
            <span className="pt-entity-count" style={{background:meta.color+"18",color:meta.color,border:`1px solid ${meta.color}33`}}>{items.length}</span>
          </div>
          <div className="pt-chips-row">
            {items.map((item,j)=><span key={j} className="pt-entity-chip">{item}</span>)}
          </div>
        </div>
      ))}
      {metrics.length>0 && (
        <div className="pt-extract-entity-row">
          <div className="pt-extract-entity-label">
            <span className="pt-entity-icon">📈</span>
            <span className="pt-entity-name" style={{color:"#a78bfa"}}>Metrics</span>
            <span className="pt-entity-count" style={{background:"#a78bfa18",color:"#a78bfa",border:"1px solid #a78bfa33"}}>{metrics.length}</span>
          </div>
          <div className="pt-chips-row">
            {metrics.map((m,j)=>(
              <span key={j} className="pt-entity-chip">
                {m.name}{m.value ? `: ${m.value}` : ""}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Library tab: indexed papers and notes ──────────────────────────────────── */
export default function LibraryPanel({ papers, onDelete }) {
  const [expandedId, setExpanded] = useState(null);
  const [detailsById, setDetails] = useState({});

  const toggleDetails = async p => {
    if (expandedId === p.id) { setExpanded(null); return; }
    setExpanded(p.id);
    if (detailsById[p.id]) return;  // cached
    try {
      const res = await fetch(`${API_BASE}/papers/${encodeURIComponent(p.id)}`);
      const data = res.ok ? await res.json() : { error: "Could not load details." };
      setDetails(prev => ({ ...prev, [p.id]: data }));
    } catch {
      setDetails(prev => ({ ...prev, [p.id]: { error: "Could not reach the backend." } }));
    }
  };

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
            <div key={p.id}>
              <div className="pt-paper-row" style={{cursor:"pointer"}} onClick={()=>toggleDetails(p)}
                title={expandedId===p.id ? "Hide extracted entities" : "Show extracted entities"}>
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
                <span style={{alignSelf:"center",color:"var(--text-4)",fontSize:10,marginRight:2}}>
                  {expandedId===p.id ? "▲" : "▼"}
                </span>
                <button className="pt-paper-delete" title={`Delete ${p.type}`}
                  onClick={e=>{ e.stopPropagation(); onDelete(p); }}>
                  {Icon.trash}
                </button>
              </div>
              {expandedId===p.id && <PaperDetails details={detailsById[p.id]}/>}
            </div>
          ))
        }
      </div>
    </div>
  );
}
