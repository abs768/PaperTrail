export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

/* ── Design constants ───────────────────────────────────────────────────────── */
export const NODE_COLORS = {
  paper:   "#e8a838", author:  "#5b8def", method:  "#ef5b5b",
  dataset: "#4ecdc4", metric:  "#a78bfa", concept: "#f97316",
  entity:  "#94a3b8", note:    "#34d399",
};
export const NODE_SIZES = {
  paper:50, author:22, method:26, dataset:24, metric:20, concept:18, entity:16, note:38,
};
export const ENTITY_META = {
  authors:      { icon:"👤", color:"#5b8def", label:"Authors" },
  methods:      { icon:"⚙️", color:"#ef5b5b", label:"Methods" },
  datasets:     { icon:"🗃️", color:"#4ecdc4", label:"Datasets" },
  key_concepts: { icon:"💡", color:"#f97316", label:"Key Concepts" },
};
export const DEFAULT_VISIBLE = new Set(["paper","author","method","dataset"]);
export const QUERY_STAGE_LABELS = {
  classifying:   "Classifying your question…",
  retrieving:    "Searching graph + vector store…",
  generating:    "Generating cited answer…",
  verifying:     "Verifying citations against sources…",
  fact_checking: "Fact-checking claims…",
};
