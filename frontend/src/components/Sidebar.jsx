import { Icon } from "./Icons";

const NAV = [
  {id:"upload",  label:"Upload",          icon:Icon.upload},
  {id:"graph",   label:"Knowledge Graph", icon:Icon.graph},
  {id:"ask",     label:"Ask Library",     icon:Icon.ask},
  {id:"library", label:"Library",         icon:Icon.library},
];

/* ── Sidebar: navigation, stats, reset ──────────────────────────────────────── */
export default function Sidebar({ activeTab, onNavigate, stats, onReset }) {
  return (
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
          <button key={n.id} className={`pt-nav-item${activeTab===n.id?" active":""}`} onClick={()=>onNavigate(n.id)}>
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
        <button className="pt-reset-btn" onClick={onReset}>
          {Icon.trash} Reset all data
        </button>
      </div>
    </aside>
  );
}
