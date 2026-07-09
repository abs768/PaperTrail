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
    if (!hits.length) { parts.push(<span key={k}>{s}</span>); break; }
    const {m,t,i} = hits[0];
    if (i>0) parts.push(<span key={k++}>{s.slice(0,i)}</span>);
    if (t==="b") parts.push(<strong key={k++} style={{color:"#f1f5f9",fontWeight:700}}>{m[1]}</strong>);
    if (t==="i") parts.push(<em key={k++} style={{color:"#c8d8f0"}}>{m[1]}</em>);
    if (t==="c") parts.push(<code key={k++} className="pt-md-code">{m[1]}</code>);
    s = s.slice(i+m[0].length);
  }
  return parts;
}

export default function MarkdownResponse({ text }) {
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
