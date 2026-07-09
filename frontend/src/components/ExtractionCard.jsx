import { ENTITY_META } from "../constants";

/* ── Extraction card ────────────────────────────────────────────────────────── */
export default function ExtractionCard({ result }) {
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
