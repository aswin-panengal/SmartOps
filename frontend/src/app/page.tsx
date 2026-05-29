"use client";

import ReactMarkdown from "react-markdown";
import { useState, useRef, useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Engine = "csv" | "pdf" | null;

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  engine?: Engine;
  sources?: string[];
  chunks_used?: number;
  error?: boolean;
  timestamp: Date;
}

interface UploadedFile {
  name: string;
  type: "csv" | "pdf";
  file: File;
}

export default function SmartOpsPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);

  // NETWORK LOCK: Prevents uploading the same file multiple times
  const [isFileIngested, setIsFileIngested] = useState(false);

  const [sessionId, setSessionId] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setSessionId(`session-${Date.now()}`);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileUpload = (file: File) => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (ext !== "csv" && ext !== "pdf") {
      alert("Only CSV and PDF files are supported.");
      return;
    }
    setUploadedFile({ name: file.name, type: ext as "csv" | "pdf", file });
    setIsFileIngested(false); // Reset network lock for the new file
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("question", userMessage.content);
      formData.append("session_id", sessionId);

      // OPTIMIZATION: Only attach the heavy file payload if it hasn't been sent yet
      if (uploadedFile && !isFileIngested) {
        formData.append("file", uploadedFile.file);
      }

      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        body: formData,
      });

      // If the request succeeds, lock the file state so it isn't sent on the next turn
      if (res.ok) {
        setIsFileIngested(true);
      }

      const data = await res.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer || data.error || "No response received.",
        engine: data.engine_used as Engine,
        sources: data.sources,
        chunks_used: data.chunks_used,
        error: data.status === "error",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "Failed to connect to SmartOps backend. Make sure your server is running.",
          error: true,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setUploadedFile(null);
    setIsFileIngested(false); // Reset lock on clear
  };

  const formatTime = (date: Date) =>
    date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <div style={styles.root}>
      <style dangerouslySetInnerHTML={{ __html: globalStyles }} />

      {/* Sidebar */}
      <aside style={{ ...styles.sidebar, ...(sidebarOpen ? {} : styles.sidebarClosed) }}>
        <div style={styles.sidebarHeader}>
          <div style={styles.logo}>
            {/* SAFELY ENCODED GLYPH */}
            <span style={styles.logoIcon}>{"\u2318"}</span>
            <span style={styles.logoText}>SmartOps</span>
          </div>
          <button style={styles.iconBtn} onClick={() => setSidebarOpen(false)}>
            ←
          </button>
        </div>

        <div style={styles.sidebarSection}>
          <p style={styles.sectionLabel}>UPLOAD FILE</p>
          <div
            style={{ ...styles.dropzone, ...(dragOver ? styles.dropzoneActive : {}) }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.pdf"
              style={{ display: "none" }}
              onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
            />
            {uploadedFile ? (
              <div style={styles.uploadedFile}>
                <span style={{
                  ...styles.fileTypeBadge,
                  background: uploadedFile.type === "csv" ? "rgba(16, 185, 129, 0.1)" : "rgba(59, 130, 246, 0.1)",
                  color: uploadedFile.type === "csv" ? "#34D399" : "#60A5FA",
                  border: uploadedFile.type === "csv" ? "1px solid rgba(16, 185, 129, 0.2)" : "1px solid rgba(59, 130, 246, 0.2)",
                }}>
                  {uploadedFile.type.toUpperCase()}
                </span>
                <span style={styles.fileName}>{uploadedFile.name}</span>
                <button
                  style={styles.removeBtn}
                  onClick={(e) => { e.stopPropagation(); setUploadedFile(null); setIsFileIngested(false); }}
                >
                  ✕
                </button>
              </div>
            ) : (
              <div style={styles.dropzoneEmpty}>
                <span style={styles.dropzoneIcon}>↑</span>
                <p style={styles.dropzoneText}>Drop CSV or PDF here</p>
                <p style={styles.dropzoneHint}>or click to browse</p>
              </div>
            )}
          </div>
        </div>

        <div style={styles.sidebarSection}>
          <p style={styles.sectionLabel}>ENGINES</p>
          <div style={styles.engineCard}>
            <div style={styles.engineDot} />
            <div>
              <p style={styles.engineName}>Analytical Engine</p>
              <p style={styles.engineDesc}>CSV · tabular data · pandas</p>
            </div>
          </div>
          <div style={styles.engineCard}>
            <div style={{ ...styles.engineDot, background: "#3B82F6", boxShadow: "0 0 8px rgba(59,130,246,0.4)" }} />
            <div>
              <p style={styles.engineName}>Semantic Engine</p>
              <p style={styles.engineDesc}>PDF · RAG · Qdrant</p>
            </div>
          </div>
        </div>

        <div style={styles.sidebarSection}>
          <p style={styles.sectionLabel}>SESSION</p>
          <p style={styles.sessionId}>{sessionId ? `${sessionId.slice(0, 20)}...` : "Initializing..."}</p>
          <p style={styles.messageCount}>{messages.length} messages</p>
        </div>

        <div style={styles.sidebarFooter}>
          <button style={styles.clearBtn} onClick={clearChat}>
            Clear conversation
          </button>
        </div>
      </aside>

      {/* Main */}
      <main style={styles.main}>
        {/* Topbar */}
        <header style={styles.topbar}>
          {!sidebarOpen && (
            <button style={styles.iconBtn} onClick={() => setSidebarOpen(true)}>
              ☰
            </button>
          )}
          <div style={styles.topbarTitle}>
            {uploadedFile ? (
              <span style={styles.topbarFile}>
                <span style={{
                  ...styles.fileTypeBadge,
                  fontSize: "10px",
                  padding: "2px 6px",
                  background: uploadedFile.type === "csv" ? "rgba(16, 185, 129, 0.1)" : "rgba(59, 130, 246, 0.1)",
                  color: uploadedFile.type === "csv" ? "#34D399" : "#60A5FA",
                  border: uploadedFile.type === "csv" ? "1px solid rgba(16, 185, 129, 0.2)" : "1px solid rgba(59, 130, 246, 0.2)",
                }}>
                  {uploadedFile.type.toUpperCase()}
                </span>
                {uploadedFile.name}
              </span>
            ) : (
              <span style={styles.topbarHint}>Interactive Data Workspace</span>
            )}
          </div>
          <div style={styles.statusDot} title="Backend connected" />
        </header>

        {/* Messages */}
        <div style={styles.messages}>
          {messages.length === 0 ? (
            <div style={styles.emptyState}>
              <div style={styles.emptyIcon}>{"\u2318"}</div>
              <h2 style={styles.emptyTitle}>SmartOps Intelligence</h2>
              <p style={styles.emptySubtitle}>
                Ask questions about your CSV data or PDF documents.<br />
                The system automatically routes to the right engine.
              </p>
              <div style={styles.suggestions}>
                {[
                  "What is the total sales by region?",
                  "Summarize the HR policy document",
                  "How many rows are in this dataset?",
                  "What is the remote work policy?",
                ].map((s) => (
                  <button
                    key={s}
                    style={styles.suggestionBtn}
                    onClick={() => setInput(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                style={{
                  ...styles.messageRow,
                  justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                {msg.role === "assistant" && (
                  <div style={styles.avatar}>{"\u2318"}</div>
                )}
                <div style={{
                  ...styles.bubble,
                  ...(msg.role === "user" ? styles.userBubble : styles.aiBubble),
                  ...(msg.error ? styles.errorBubble : {}),
                }}>

                  {msg.role === "user" ? (
                    <p style={styles.bubbleText}>{msg.content}</p>
                  ) : (
                    <div className="markdown-wrapper" style={styles.bubbleText}>
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  )}

                  {msg.engine && (
                    <div style={styles.bubbleMeta}>
                      <span style={{
                        ...styles.engineBadge,
                        background: msg.engine === "csv" ? "rgba(16, 185, 129, 0.1)" : "rgba(59, 130, 246, 0.1)",
                        color: msg.engine === "csv" ? "#34D399" : "#60A5FA",
                        border: msg.engine === "csv" ? "1px solid rgba(16, 185, 129, 0.2)" : "1px solid rgba(59, 130, 246, 0.2)",
                      }}>
                        {msg.engine === "csv" ? "Analytical" : "Semantic"} Engine
                      </span>
                      {msg.chunks_used && (
                        <span style={styles.chunksBadge}>
                          {msg.chunks_used} chunks
                        </span>
                      )}
                    </div>
                  )}

                  {msg.sources && msg.sources.length > 0 && (
                    <div style={styles.sources}>
                      <p style={styles.sourcesLabel}>Sources</p>
                      {msg.sources.map((src, i) => (
                        <span key={i} style={styles.sourceTag}>{src}</span>
                      ))}
                    </div>
                  )}

                  <span style={styles.timestamp}>{formatTime(msg.timestamp)}</span>
                </div>
              </div>
            ))
          )}

          {loading && (
            <div style={styles.messageRow}>
              <div style={styles.avatar}>{"\u2318"}</div>
              <div style={{ ...styles.bubble, ...styles.aiBubble }}>
                <div style={styles.typingDots}>
                  <span style={styles.dot} className="dot1" />
                  <span style={styles.dot} className="dot2" />
                  <span style={styles.dot} className="dot3" />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div style={styles.inputArea}>
          <div style={styles.inputWrapper}>
            <textarea
              ref={textareaRef}
              style={styles.textarea}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your data or documents..."
              rows={1}
            />
            <button
              style={{
                ...styles.sendBtn,
                ...(loading || !input.trim() ? styles.sendBtnDisabled : {}),
              }}
              onClick={sendMessage}
              disabled={loading || !input.trim()}
            >
              ↑
            </button>
          </div>
          <p style={styles.inputHint}>
            Press Enter to send · Shift+Enter for new line ·{" "}
            {uploadedFile
              ? `File attached: ${uploadedFile.name}`
              : "No file attached — querying stored documents"}
          </p>
        </div>
      </main>
    </div>
  );
}


const styles: Record<string, React.CSSProperties> = {
  root: {
    display: "flex",
    height: "100vh",
    background: "#0A0A0B",
    color: "#F3F4F6",
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    overflow: "hidden",
  },
  sidebar: {
    width: "280px",
    minWidth: "280px",
    background: "#0A0A0B",
    borderRight: "1px solid #1F2937",
    display: "flex",
    flexDirection: "column",
    transition: "width 0.2s ease, min-width 0.2s ease",
    overflow: "hidden",
  },
  sidebarClosed: {
    width: "0px",
    minWidth: "0px",
  },
  sidebarHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "20px 20px 16px",
    borderBottom: "1px solid #1F2937",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  logoIcon: {
    fontSize: "18px",
    color: "#FFFFFF",
    background: "#2563EB",
    padding: "4px",
    borderRadius: "6px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    boxShadow: "0 2px 10px rgba(37, 99, 235, 0.3)",
  },
  logoText: {
    fontSize: "16px",
    fontWeight: 600,
    letterSpacing: "-0.01em",
    color: "#F9FAFB",
  },
  iconBtn: {
    background: "transparent",
    border: "1px solid #374151",
    color: "#9CA3AF",
    width: "28px",
    height: "28px",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "12px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 0.2s",
  },
  sidebarSection: {
    padding: "20px",
    borderBottom: "1px solid #1F2937",
  },
  sectionLabel: {
    fontSize: "10px",
    fontWeight: 600,
    letterSpacing: "0.1em",
    color: "#6B7280",
    marginBottom: "12px",
    margin: "0 0 12px 0",
  },
  dropzone: {
    border: "1px dashed #374151",
    background: "rgba(255,255,255,0.01)",
    borderRadius: "12px",
    padding: "20px",
    cursor: "pointer",
    transition: "all 0.2s",
    minHeight: "90px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  dropzoneActive: {
    borderColor: "#3B82F6",
    background: "rgba(59, 130, 246, 0.05)",
  },
  dropzoneEmpty: {
    textAlign: "center",
  },
  dropzoneIcon: {
    fontSize: "20px",
    color: "#6B7280",
    display: "block",
    marginBottom: "6px",
  },
  dropzoneText: {
    fontSize: "13px",
    fontWeight: 500,
    color: "#D1D5DB",
    margin: "0 0 2px 0",
  },
  dropzoneHint: {
    fontSize: "11px",
    color: "#6B7280",
    margin: 0,
  },
  uploadedFile: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    width: "100%",
  },
  fileTypeBadge: {
    fontSize: "10px",
    fontWeight: 600,
    letterSpacing: "0.05em",
    padding: "4px 8px",
    borderRadius: "6px",
    flexShrink: 0,
  },
  fileName: {
    fontSize: "12px",
    fontWeight: 500,
    color: "#D1D5DB",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    flex: 1,
  },
  removeBtn: {
    background: "transparent",
    border: "none",
    color: "#6B7280",
    cursor: "pointer",
    fontSize: "12px",
    padding: "4px",
    flexShrink: 0,
  },
  engineCard: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "10px 0",
  },
  engineDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#10B981",
    flexShrink: 0,
    boxShadow: "0 0 8px rgba(16, 185, 129, 0.4)",
  },
  engineName: {
    fontSize: "13px",
    fontWeight: 500,
    color: "#E5E7EB",
    margin: "0 0 2px 0",
  },
  engineDesc: {
    fontSize: "11px",
    color: "#6B7280",
    margin: 0,
  },
  sessionId: {
    fontSize: "11px",
    color: "#9CA3AF",
    fontFamily: "'Fira Code', monospace",
    margin: "0 0 6px 0",
    background: "#111827",
    padding: "4px 8px",
    borderRadius: "4px",
    border: "1px solid #1F2937",
    display: "inline-block",
  },
  messageCount: {
    fontSize: "12px",
    color: "#6B7280",
    margin: 0,
  },
  sidebarFooter: {
    marginTop: "auto",
    padding: "20px",
  },
  clearBtn: {
    width: "100%",
    padding: "10px",
    background: "#111827",
    border: "1px solid #1F2937",
    borderRadius: "8px",
    color: "#9CA3AF",
    fontSize: "13px",
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.2s",
  },
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    background: "#0A0A0B",
    position: "relative",
  },
  topbar: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "16px 24px",
    borderBottom: "1px solid #1F2937",
    background: "rgba(10, 10, 11, 0.8)",
    backdropFilter: "blur(12px)",
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    zIndex: 10,
  },
  topbarTitle: {
    flex: 1,
    fontSize: "14px",
    fontWeight: 500,
    color: "#9CA3AF",
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  topbarFile: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    color: "#E5E7EB",
  },
  topbarHint: {
    color: "#6B7280",
  },
  statusDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#10B981",
    boxShadow: "0 0 10px rgba(16, 185, 129, 0.5)",
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    padding: "80px 24px 120px",
    display: "flex",
    flexDirection: "column",
    gap: "24px",
  },
  emptyState: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    textAlign: "center",
    padding: "40px 20px",
  },
  emptyIcon: {
    fontSize: "48px",
    color: "#374151",
    marginBottom: "20px",
  },
  emptyTitle: {
    fontSize: "24px",
    fontWeight: 600,
    color: "#F9FAFB",
    margin: "0 0 12px 0",
    letterSpacing: "-0.01em",
  },
  emptySubtitle: {
    fontSize: "14px",
    color: "#9CA3AF",
    lineHeight: 1.6,
    margin: "0 0 40px 0",
    maxWidth: "420px",
  },
  suggestions: {
    display: "flex",
    flexWrap: "wrap",
    gap: "10px",
    justifyContent: "center",
    maxWidth: "600px",
  },
  suggestionBtn: {
    background: "#111827",
    border: "1px solid #1F2937",
    borderRadius: "8px",
    color: "#D1D5DB",
    fontSize: "13px",
    fontWeight: 500,
    padding: "10px 16px",
    cursor: "pointer",
    transition: "all 0.2s",
  },
  messageRow: {
    display: "flex",
    alignItems: "flex-end",
    gap: "12px",
  },
  avatar: {
    width: "32px",
    height: "32px",
    borderRadius: "8px",
    background: "#111827",
    border: "1px solid #1F2937",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "16px",
    color: "#3B82F6",
    flexShrink: 0,
    boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
  },
  bubble: {
    maxWidth: "75%",
    padding: "16px 20px",
    position: "relative",
    fontSize: "14px",
    lineHeight: 1.6,
  },
  userBubble: {
    background: "#2563EB",
    color: "#FFFFFF",
    borderRadius: "20px 20px 4px 20px",
    boxShadow: "0 4px 15px rgba(37, 99, 235, 0.2)",
  },
  aiBubble: {
    background: "#111827",
    border: "1px solid #1F2937",
    color: "#E5E7EB",
    borderRadius: "20px 20px 20px 4px",
    boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
  },
  errorBubble: {
    border: "1px solid rgba(239, 68, 68, 0.3)",
    background: "rgba(239, 68, 68, 0.05)",
    color: "#FCA5A5",
  },
  bubbleText: {
    margin: 0,
    whiteSpace: "pre-wrap",
  },
  bubbleMeta: {
    display: "flex",
    gap: "8px",
    marginTop: "16px",
    paddingTop: "12px",
    borderTop: "1px solid rgba(255,255,255,0.05)",
    flexWrap: "wrap",
  },
  engineBadge: {
    fontSize: "10px",
    fontWeight: 600,
    letterSpacing: "0.05em",
    padding: "4px 8px",
    borderRadius: "6px",
    textTransform: "uppercase",
  },
  chunksBadge: {
    fontSize: "11px",
    fontWeight: 500,
    color: "#9CA3AF",
    padding: "4px 8px",
    background: "rgba(255,255,255,0.05)",
    borderRadius: "6px",
    border: "1px solid rgba(255,255,255,0.05)",
  },
  sources: {
    marginTop: "12px",
    paddingTop: "12px",
    borderTop: "1px solid rgba(255,255,255,0.05)",
  },
  sourcesLabel: {
    fontSize: "10px",
    fontWeight: 600,
    letterSpacing: "0.05em",
    color: "#6B7280",
    margin: "0 0 8px 0",
    textTransform: "uppercase",
  },
  sourceTag: {
    fontSize: "11px",
    fontWeight: 500,
    color: "#60A5FA",
    background: "rgba(59, 130, 246, 0.1)",
    border: "1px solid rgba(59, 130, 246, 0.2)",
    padding: "4px 10px",
    borderRadius: "6px",
    marginRight: "6px",
    display: "inline-block",
    marginBottom: "4px",
  },
  timestamp: {
    fontSize: "11px",
    color: "rgba(255,255,255,0.3)",
    display: "block",
    marginTop: "12px",
    textAlign: "right",
  },
  typingDots: {
    display: "flex",
    gap: "4px",
    alignItems: "center",
    height: "24px",
  },
  dot: {
    width: "6px",
    height: "6px",
    borderRadius: "50%",
    background: "#3B82F6",
    opacity: 0.4,
    animation: "pulse 1.2s ease-in-out infinite",
  },
  inputArea: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    padding: "24px",
    background: "linear-gradient(to top, #0A0A0B 70%, transparent)",
  },
  inputWrapper: {
    display: "flex",
    gap: "12px",
    alignItems: "flex-end",
    maxWidth: "800px",
    margin: "0 auto",
    position: "relative",
  },
  textarea: {
    flex: 1,
    background: "#111827",
    border: "1px solid #374151",
    borderRadius: "24px",
    color: "#F9FAFB",
    fontSize: "14px",
    padding: "16px 56px 16px 20px",
    resize: "none",
    outline: "none",
    fontFamily: "inherit",
    lineHeight: 1.5,
    minHeight: "54px",
    maxHeight: "150px",
    overflowY: "auto",
    boxShadow: "0 4px 20px rgba(0,0,0,0.2)",
    transition: "border-color 0.2s",
  },
  sendBtn: {
    position: "absolute",
    right: "6px",
    bottom: "6px",
    width: "42px",
    height: "42px",
    background: "#2563EB",
    border: "none",
    borderRadius: "20px",
    color: "#FFFFFF",
    fontSize: "18px",
    fontWeight: 600,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 0.2s",
    boxShadow: "0 2px 10px rgba(37, 99, 235, 0.3)",
  },
  sendBtnDisabled: {
    opacity: 0.5,
    background: "#374151",
    boxShadow: "none",
    cursor: "not-allowed",
  },
  inputHint: {
    fontSize: "11px",
    color: "#6B7280",
    margin: "12px 0 0 0",
    textAlign: "center",
    fontWeight: 500,
  },
};

const globalStyles = `
  * { box-sizing: border-box; }
  body { margin: 0; padding: 0; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #4B5563; }
  .dot1 { animation-delay: 0s; }
  .dot2 { animation-delay: 0.2s; }
  .dot3 { animation-delay: 0.4s; }
  @keyframes pulse {
    0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1.1); box-shadow: 0 0 8px rgba(59,130,246,0.5); }
  }
  textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.2), 0 4px 20px rgba(0,0,0,0.2) !important;
  }
  button:hover:not(:disabled) { transform: translateY(-1px); }
  .suggestionBtn:hover { border-color: #374151 !important; background: #1F2937 !important; }
  .clearBtn:hover { background: #1F2937 !important; color: #E5E7EB !important; border-color: #374151 !important; }

  /* Markdown Styles */
  .markdown-wrapper p { margin: 0 0 12px 0; }
  .markdown-wrapper p:last-child { margin: 0; }
  .markdown-wrapper ul, .markdown-wrapper ol { margin: 0 0 12px 0; padding-left: 20px; }
  .markdown-wrapper li { margin-bottom: 6px; }
  .markdown-wrapper strong { color: #F9FAFB; font-weight: 600; }
  .markdown-wrapper code { background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; font-family: 'Fira Code', monospace; font-size: 12px; }
`;