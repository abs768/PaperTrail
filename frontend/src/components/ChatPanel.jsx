import { useEffect, useRef } from "react";

import { QUERY_STAGE_LABELS } from "../constants";
import { Icon } from "./Icons";
import MarkdownResponse from "./Markdown";

/* ── Ask tab: chat over the library ─────────────────────────────────────────── */
export default function ChatPanel({
  chatHistory, question, setQuestion, querying, queryStage, onSend,
}) {
  const chatEndRef = useRef(null);
  useEffect(()=>{ chatEndRef.current?.scrollIntoView({behavior:"smooth"}); },[chatHistory]);

  const sendDisabled = querying || !question.trim();

  return (
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
                {QUERY_STAGE_LABELS[queryStage] || "Searching graph + vector store…"}
              </div>
            </div>
          </div>
        )}
        <div ref={chatEndRef}/>
      </div>

      <div className="pt-chat-input-bar">
        <textarea className="pt-input pt-chat-textarea" rows={1}
          placeholder="Ask a question about your papers…  (Shift+Enter for a new line)"
          value={question}
          onChange={e=>{
            setQuestion(e.target.value);
            e.target.style.height="auto";
            e.target.style.height=Math.min(e.target.scrollHeight,140)+"px";
          }}
          onKeyDown={e=>{ if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); onSend(); } }}/>
        <button className="pt-btn pt-btn-primary" onClick={onSend}
          disabled={sendDisabled}
          style={{opacity:sendDisabled ? 0.4 : 1,cursor:sendDisabled ? "not-allowed" : "pointer"}}>
          {Icon.send} Send
        </button>
      </div>
    </div>
  );
}
