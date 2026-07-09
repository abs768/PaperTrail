import { useState, useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";

import { NODE_COLORS, NODE_SIZES, DEFAULT_VISIBLE } from "../constants";

cytoscape.use(fcose);

/* ── Cytoscape Graph ────────────────────────────────────────────────────────── */
export default function CytoscapeGraph({ graphData }) {
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
