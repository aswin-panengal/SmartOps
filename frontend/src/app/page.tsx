"use client";

import ReactMarkdown from "react-markdown";
import { useState, useRef, useEffect, useCallback } from "react";

// Enforce strict runtime verification for production endpoints
const API_BASE = (() => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }

  // If compiling on client side and environment key is dropped, warn clearly
  if (typeof window !== "undefined") {
    console.warn("⚠️ Next.js Warning: NEXT_PUBLIC_API_URL variable is missing. Routing to localhost fallback.");
  }

  return "http://localhost:8000";
})();

type Engine = "csv" | "pdf" | "clarify" | null;
type ServerState = "waking" | "ready" | "error";

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

interface ChatSession {
  id: string;
  displayName: string;
  uploadedFile: UploadedFile | null;
  isFileIngested: boolean;
  messages: Message[];
  loading: boolean;
}

// Generate a single stable initial ID to share across both initialization states on cold boot
const initialId = `session-${Date.now()}`;

export default function SmartOpsPage() {
  // Fixes the duplicate session bug by ensuring the array is never born empty
  const [sessions, setSessions] = useState<ChatSession[]>([
    {
      id: initialId,
      displayName: "✨ New Interactive Context",
      uploadedFile: null,
      isFileIngested: false,
      messages: [],
      loading: false,
    },
  ]);
  const [activeSessionId, setActiveSessionId] = useState<string>(initialId);
  const [input, setInput] = useState("");
  const [serverState, setServerState] = useState<ServerState>("waking");
  const [dragOver, setDragOver] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Derive all active state parameters cleanly based on the selected workspace pointer
  const activeSession = sessions.find((s) => s.id === activeSessionId) || null;
  const uploadedFile = activeSession?.uploadedFile || null;
  const isFileIngested = activeSession?.isFileIngested || false;
  const messages = activeSession?.messages || [];

  // Utility helper to handle granular message state shifts inside a focused array context
  const setMessages = (updateFn: (prev: Message[]) => Message[]) => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeSessionId ? { ...s, messages: updateFn(s.messages) } : s
      )
    );
  };

  const setIsFileIngested = (val: boolean) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === activeSessionId ? { ...s, isFileIngested: val } : s))
    );
  };

  // Metadata-driven workspace title display rules
  const getSessionDisplayName = (file: UploadedFile | null, firstQuery?: string) => {
    if (file) {
      const cleanName = file.name.replace(/\.[^/.]+$/, "");
      return `${file.type === "csv" ? "📊" : "📄"} ${cleanName}`;
    }
    if (firstQuery) {
      return `💬 ${firstQuery.trim().substring(0, 18)}${firstQuery.trim().length > 18 ? "..." : ""}`;
    }
    return "✨ New Interactive Context";
  };

  const createNewSession = useCallback((file: UploadedFile | null = null) => {
    const newId = `session-${Date.now()}`;
    const newSession: ChatSession = {
      id: newId,
      displayName: getSessionDisplayName(file),
      uploadedFile: file,
      isFileIngested: false,
      messages: [],
      loading: false,
    };
    setSessions((prev) => [...prev, newSession]);
    setActiveSessionId(newId);
  }, []);

  const wakeUpServer = useCallback(async () => {
    setServerState("waking");
    try {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 90000);

      const res = await fetch(`${API_BASE}/health`, {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      });

      window.clearTimeout(timeout);
      setServerState(res.ok ? "ready" : "error");
    } catch {
      setServerState("error");
    }
  }, []);

  useEffect(() => {
    wakeUpServer();
  }, [wakeUpServer]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileUpload = (file: File) => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (ext !== "csv" && ext !== "pdf") {
      alert("Only CSV and PDF files are supported.");
      return;
    }
    const fileObj: UploadedFile = { name: file.name, type: ext as "csv" | "pdf", file };

    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeSessionId
          ? {
            ...s,
            uploadedFile: fileObj,
            isFileIngested: false,
            displayName: getSessionDisplayName(fileObj),
          }
          : s
      )
    );
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  };

  const sendMessage = async () => {
    // Reads performance-critical loading flag from isolated workspace item map
    if (!input.trim() || activeSession?.loading || serverState !== "ready") return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    const isFirstQuery = messages.length === 0;

    // Locks down the typing loading status strictly within the current session block array
    setSessions((prev) =>
      prev.map((s) =>
        s.id === activeSessionId
          ? { ...s, messages: [...s.messages, userMessage], loading: true }
          : s
      )
    );

    setInput("");

    try {
      const formData = new FormData();
      formData.append("question", userMessage.content);
      formData.append("session_id", activeSessionId);

      if (uploadedFile && !isFileIngested) {
        formData.append("file", uploadedFile.file);
      }

      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        body: formData,
      });

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

      // Clear loading state and deliver payload purely to the operational source workspace context
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? {
              ...s,
              messages: [...s.messages, assistantMessage],
              loading: false,
              displayName: !s.uploadedFile && isFirstQuery
                ? getSessionDisplayName(null, userMessage.content)
                : s.displayName,
            }
            : s
        )
      );
    } catch {
      // Clear loading state on fail conditions cleanly within the workspace pointer boundary
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? {
              ...s,
              messages: [
                ...s.messages,
                {
                  id: (Date.now() + 1).toString(),
                  role: "assistant",
                  content: "Failed to connect to SmartOps backend. Make sure your server is running.",
                  error: true,
                  timestamp: new Date(),
                },
              ],
              loading: false,
            }
            : s
        )
      );
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const deleteSession = async (idToDelete: string, e: React.MouseEvent) => {
    e.stopPropagation();

    try {
      await fetch(`${API_BASE}/api/session/${idToDelete}`, {
        method: "DELETE",
      });
    } catch (err) {
      console.error("Failed to sync session termination state to backend cache:", err);
    }

    const remaining = sessions.filter((s) => s.id !== idToDelete);
    setSessions(remaining);

    if (activeSessionId === idToDelete) {
      if (remaining.length > 0) {
        setActiveSessionId(remaining[remaining.length - 1].id);
      } else {
        const fallbackId = `session-${Date.now()}`;
        setSessions([
          {
            id: fallbackId,
            displayName: "✨ New Interactive Context",
            uploadedFile: null,
            isFileIngested: false,
            messages: [],
            loading: false,
          },
        ]);
        setActiveSessionId(fallbackId);
      }
    }
  };

  const clearChat = async () => {
    if (activeSessionId) {
      try {
        await fetch(`${API_BASE}/api/session/${activeSessionId}`, {
          method: "DELETE",
        });
      } catch (err) {
        console.error("Failed to wipe active backend session memory cache:", err);
      }

      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId
            ? {
              ...s,
              uploadedFile: null,
              isFileIngested: false,
              messages: [],
              loading: false,
              displayName: "✨ New Interactive Context",
            }
            : s
        )
      );
    }
  };

  const formatTime = (date: Date) =>
    date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <div style={styles.root}>
      <style dangerouslySetInnerHTML={{ __html: globalStyles }} />

      {/* Sidebar Layout */}
      <aside style={{ ...styles.sidebar, ...(sidebarOpen ? {} : styles.sidebarClosed) }}>
        <div style={styles.sidebarHeader}>
          <div style={styles.logo}>
            <span style={styles.logoIcon}>{"\u2318"}</span>
            <span style={styles.logoText}>SmartOps</span>
          </div>
          <button style={styles.iconBtn} onClick={() => setSidebarOpen(false)}>
            ←
          </button>
        </div>

        {/* Upload Slot */}
        <div style={styles.sidebarSection}>
          <p style={styles.sectionLabel}>UPLOAD CONTEXT FILE</p>
          <div
            style={{ ...styles.dropzone, ...(dragOver ? styles.dropzoneActive : {}) }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
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
                <span
                  style={{
                    ...styles.fileTypeBadge,
                    background:
                      uploadedFile.type === "csv"
                        ? "rgba(16, 185, 129, 0.1)"
                        : "rgba(59, 130, 246, 0.1)",
                    color: uploadedFile.type === "csv" ? "#34D399" : "#60A5FA",
                    border:
                      uploadedFile.type === "csv"
                        ? "1px solid rgba(16, 185, 129, 0.2)"
                        : "1px solid rgba(59, 130, 246, 0.2)",
                  }}
                >
                  {uploadedFile.type.toUpperCase()}
                </span>
                <span style={styles.fileName}>{uploadedFile.name}</span>
                <button
                  style={styles.removeBtn}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSessions((prev) =>
                      prev.map((s) =>
                        s.id === activeSessionId
                          ? {
                            ...s,
                            uploadedFile: null,
                            isFileIngested: false,
                            displayName: "✨ New Interactive Context",
                          }
                          : s
                      )
                    );
                  }}
                >
                  ✕
                </button>
              </div>
            ) : (
              <div style={styles.dropzoneEmpty}>
                <span style={styles.dropzoneIcon}>↑</span>
                <p style={styles.dropzoneText}>Drop CSV or PDF context</p>
                <p style={styles.dropzoneHint}>or click to select</p>
              </div>
            )}
          </div>
        </div>

        {/* Dynamic Parallel Active Workspaces Nav */}
        <div style={{ ...styles.sidebarSection, flex: 1, overflowY: "auto", borderBottom: "none" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "14px" }}>
            <p style={{ ...styles.sectionLabel, margin: 0 }}>ACTIVE WORKSPACES</p>
            <button
              style={{
                background: "#1F2937",
                border: "none",
                color: "#F3F4F6",
                width: "22px",
                height: "22px",
                borderRadius: "6px",
                cursor: "pointer",
                fontSize: "14px",
                fontWeight: "bold",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 0.2s",
              }}
              onClick={() => createNewSession(null)}
              title="Spawn a clean parallel workspace context"
              className="plus-session-btn"
            >
              +
            </button>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {sessions.map((session) => {
              const isActive = session.id === activeSessionId;
              return (
                <div
                  key={session.id}
                  onClick={() => setActiveSessionId(session.id)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "10px 12px",
                    borderRadius: "8px",
                    cursor: "pointer",
                    background: isActive ? "#111827" : "transparent",
                    border: isActive ? "1px solid #374151" : "1px solid transparent",
                    transition: "all 0.15s ease",
                  }}
                  className="session-row-item"
                >
                  <span
                    style={{
                      fontSize: "13px",
                      fontWeight: isActive ? 600 : 500,
                      color: isActive ? "#F9FAFB" : "#9CA3AF",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      flex: 1,
                    }}
                  >
                    {session.displayName}
                  </span>
                  <button
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "#4B5563",
                      cursor: "pointer",
                      fontSize: "11px",
                      marginLeft: "8px",
                      padding: "2px",
                    }}
                    onClick={(e) => deleteSession(session.id, e)}
                    className="session-close-btn"
                  >
                    ✕
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        <div style={styles.sidebarFooter}>
          <button style={styles.clearBtn} onClick={clearChat}>
            Reset current session
          </button>
        </div>
      </aside>

      {/* Main Framework Block */}
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
                <span
                  style={{
                    ...styles.fileTypeBadge,
                    fontSize: "10px",
                    padding: "2px 6px",
                    background:
                      uploadedFile.type === "csv"
                        ? "rgba(16, 185, 129, 0.1)"
                        : "rgba(59, 130, 246, 0.1)",
                    color: uploadedFile.type === "csv" ? "#34D399" : "#60A5FA",
                    border:
                      uploadedFile.type === "csv"
                        ? "1px solid rgba(16, 185, 129, 0.2)"
                        : "1px solid rgba(59, 130, 246, 0.2)",
                  }}
                >
                  {uploadedFile.type.toUpperCase()}
                </span>
                {uploadedFile.name}
              </span>
            ) : (
              <span style={styles.topbarHint}>Interactive Sandbox Workspace</span>
            )}
          </div>
          <div style={styles.serverStatus}>
            <span
              style={{
                ...styles.statusDot,
                ...(serverState === "waking" ? styles.statusDotWaking : {}),
                ...(serverState === "error" ? styles.statusDotError : {}),
              }}
            />
            <span style={styles.statusText}>
              {serverState === "ready"
                ? "Ready"
                : serverState === "waking"
                  ? "Starting backend"
                  : "Connection issue"}
            </span>
          </div>
        </header>

        {/* Messaging Logs Node */}
        <div style={styles.messages}>
          {messages.length === 0 ? (
            <div style={styles.emptyState}>
              <div style={styles.emptyIcon}>{"\u2318"}</div>
              <h2 style={styles.emptyTitle}>SmartOps Intelligence</h2>
              {serverState !== "ready" && (
                <div
                  style={{
                    ...styles.wakeNotice,
                    ...(serverState === "error" ? styles.wakeNoticeError : {}),
                  }}
                >
                  <div style={styles.wakeNoticeHeader}>
                    <span
                      style={{
                        ...styles.statusDot,
                        ...(serverState === "waking" ? styles.statusDotWaking : {}),
                        ...(serverState === "error" ? styles.statusDotError : {}),
                      }}
                    />
                    <span>
                      {serverState === "waking"
                        ? "Backend is starting on Render"
                        : "Backend did not respond"}
                    </span>
                  </div>
                  <p style={styles.wakeNoticeText}>
                    {serverState === "waking"
                      ? "First visit can take about a minute while the server wakes up."
                      : "The server may still be waking. Try again in a moment."}
                  </p>
                  {serverState === "error" && (
                    <button style={styles.retryBtn} onClick={wakeUpServer}>
                      Retry
                    </button>
                  )}
                </div>
              )}
              <p style={styles.emptySubtitle}>
                Ask questions about your CSV data or PDF documents.
                <br />
                The system automatically paths to the optimized isolated execution sandbox.
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
                {msg.role === "assistant" && <div style={styles.avatar}>{"\u2318"}</div>}
                <div
                  style={{
                    ...styles.bubble,
                    ...(msg.role === "user" ? styles.userBubble : styles.aiBubble),
                    ...(msg.error ? styles.errorBubble : {}),
                  }}
                >
                  {msg.role === "user" ? (
                    <p style={styles.bubbleText}>{msg.content}</p>
                  ) : (
                    <div className="markdown-wrapper" style={styles.bubbleText}>
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  )}

                  {msg.engine && msg.engine !== "clarify" && (
                    <div style={styles.bubbleMeta}>
                      <span
                        style={{
                          ...styles.engineBadge,
                          background:
                            msg.engine === "csv"
                              ? "rgba(16, 185, 129, 0.1)"
                              : "rgba(59, 130, 246, 0.1)",
                          color: msg.engine === "csv" ? "#34D399" : "#60A5FA",
                          border:
                            msg.engine === "csv"
                              ? "1px solid rgba(16, 185, 129, 0.2)"
                              : "1px solid rgba(59, 130, 246, 0.2)",
                        }}
                      >
                        {msg.engine === "csv" ? "Analytical" : "Semantic"} Engine
                      </span>
                      {msg.chunks_used && (
                        <span style={styles.chunksBadge}>{msg.chunks_used} chunks</span>
                      )}
                    </div>
                  )}

                  {msg.sources && msg.sources.length > 0 && (
                    <div style={styles.sources}>
                      <p style={styles.sourcesLabel}>Sources</p>
                      {msg.sources.map((src, i) => (
                        <span key={i} style={styles.sourceTag}>
                          {src}
                        </span>
                      ))}
                    </div>
                  )}

                  <span style={styles.timestamp}>{formatTime(msg.timestamp)}</span>
                </div>
              </div>
            ))
          )}

          {/* Typing Dots Animation Bubble - Scoped strictly to the active target item state */}
          {activeSession?.loading && (
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

        {/* Input Text Box Frame */}
        <div style={styles.inputArea}>
          <div style={styles.inputWrapper}>
            <textarea
              ref={textareaRef}
              style={styles.textarea}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your data or documents..."
              disabled={serverState !== "ready"}
              rows={1}
            />
            <button
              style={{
                ...styles.sendBtn,
                ...(activeSession?.loading || !input.trim() || serverState !== "ready"
                  ? styles.sendBtnDisabled
                  : {}),
              }}
              onClick={sendMessage}
              disabled={activeSession?.loading || !input.trim() || serverState !== "ready"}
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
    fontFamily:
      "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
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
  serverStatus: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "6px 10px",
    background: "#111827",
    border: "1px solid #1F2937",
    borderRadius: "8px",
    color: "#9CA3AF",
    fontSize: "12px",
    fontWeight: 500,
  },
  statusDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#10B981",
    boxShadow: "0 0 10px rgba(16, 185, 129, 0.5)",
  },
  statusDotWaking: {
    background: "#F59E0B",
    boxShadow: "0 0 10px rgba(245, 158, 11, 0.45)",
    animation: "pulse 1.2s ease-in-out infinite",
  },
  statusDotError: {
    background: "#EF4444",
    boxShadow: "0 0 10px rgba(239, 68, 68, 0.45)",
  },
  statusText: {
    whiteSpace: "nowrap",
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
  wakeNotice: {
    width: "min(420px, 100%)",
    margin: "0 0 28px 0",
    padding: "14px 16px",
    background: "#111827",
    border: "1px solid #1F2937",
    borderRadius: "8px",
    textAlign: "left",
  },
  wakeNoticeError: {
    border: "1px solid rgba(239, 68, 68, 0.35)",
    background: "rgba(239, 68, 68, 0.06)",
  },
  wakeNoticeHeader: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    color: "#E5E7EB",
    fontSize: "13px",
    fontWeight: 600,
    marginBottom: "6px",
  },
  wakeNoticeText: {
    color: "#9CA3AF",
    fontSize: "12px",
    lineHeight: 1.5,
    margin: "0",
  },
  retryBtn: {
    marginTop: "12px",
    padding: "8px 12px",
    background: "#1F2937",
    border: "1px solid #374151",
    borderRadius: "6px",
    color: "#E5E7EB",
    fontSize: "12px",
    fontWeight: 600,
    cursor: "pointer",
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
  .plus-session-btn:hover { background: #374151 !important; color: #FFFFFF !important; }
  .suggestionBtn:hover { border-color: #374151 !important; background: #1F2937 !important; }
  .clearBtn:hover { background: #1F2937 !important; color: #E5E7EB !important; border-color: #374151 !important; }
  
  /* Workspace List Hover Transitions */
  .session-row-item:hover {
    background: rgba(31, 41, 55, 0.4) !important;
  }
  .session-row-item:hover .session-close-btn {
    color: #9CA3AF !important;
  }
  .session-close-btn:hover {
    color: #EF4444 !important;
  }

  /* Markdown Styles */
  .markdown-wrapper p { margin: 0 0 12px 0; }
  .markdown-wrapper p:last-child { margin: 0; }
  .markdown-wrapper ul, .markdown-wrapper ol { margin: 0 0 12px 0; padding-left: 20px; }
  .markdown-wrapper li { margin-bottom: 6px; }
  .markdown-wrapper strong { color: #F9FAFB; font-weight: 600; }
  .markdown-wrapper code { background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; font-family: 'Fira Code', monospace; font-size: 12px; }
`;