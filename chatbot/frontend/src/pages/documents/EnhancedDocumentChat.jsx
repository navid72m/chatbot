import React, { useState, useRef, useEffect } from "react";
import "../../styles/pages/MCPDocumentChat.css";
import "./EnhancedDocumentChat.css";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";
import { useNavigate } from "react-router-dom";
import PDFContextViewer from "./PDFContextViewer";
import contextTrackingService from "./ContextTrackingService";

/**
 * Robust backend URL resolver for Electron + Dev.
 * - In Electron: asks preload API for dynamic port (ipcMain handler).
 * - In browser dev: falls back to localhost:8000 (or VITE_BACKEND_URL if set).
 */
async function resolveBackendURL({ timeoutMs = 60000, intervalMs = 300 } = {}) {
  const start = Date.now();

  // If you want to override in dev:
  const envUrl =
    (import.meta?.env && import.meta.env.VITE_BACKEND_URL) ||
    (typeof process !== "undefined" ? process.env?.VITE_BACKEND_URL : null);

  while (Date.now() - start < timeoutMs) {
    try {
      // Electron preload API
      if (window?.electronAPI?.getBackendUrl) {
        const url = await window.electronAPI.getBackendUrl();
        if (url && typeof url === "string" && url.startsWith("http")) return url;
      }
    } catch (e) {
      // ignore and retry
    }

    // Browser dev fallback
    if (import.meta?.env?.DEV) {
      return envUrl || "http://127.0.0.1:8000";
    }

    await new Promise((r) => setTimeout(r, intervalMs));
  }

  // Final fallback (production browser scenario)
  if (envUrl) return envUrl;

  throw new Error("Timed out waiting for backend URL (Electron IPC).");
}

/** Better axios error logging */
function logAxiosError(prefix, err) {
  const msg = err?.message || String(err);
  const code = err?.code;
  const status = err?.response?.status;
  const data = err?.response?.data;
  const url = err?.config?.url;
  console.error(prefix, {
    message: msg,
    code,
    status,
    url,
    data,
  });
}

const EnhancedDocumentChat = ({ onError = () => {} }) => {
  const navigate = useNavigate();

  // ===== Backend URL / Connection =====
  const [backendURL, setBackendURL] = useState(null);
  const [connected, setConnected] = useState(false);
  const [serverInfo, setServerInfo] = useState(null);

  // ===== Upload state =====
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);

  // ===== Chat state =====
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const [showSuggestions, setShowSuggestions] = useState(true);

  // ===== Evaluation state =====
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState(null);

  // ===== PDF Viewer state =====
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState(null);

  // ===== Options state =====
  const [advancedOptions, setAdvancedOptions] = useState({
    use_advanced_rag: true,
    use_llama_index: true,
    model: "mixtral-8x7b-instruct-v0.1.Q4_K_M",
    temperature: 0.3,
    context_window: 5,
    quantization: "4bit",
    use_prf: true,
    use_variants: true,
    prf_iterations: 1,
    fusion_method: "rrf",
    rerank: true,
  });

  const [availableModels, setAvailableModels] = useState([
    "mixtral-8x7b-instruct-v0.1.Q4_K_M",
  ]);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [ragOptions, setRagOptions] = useState([
    { value: "default", label: "Default RAG" },
    { value: "enhanced", label: "Enhanced RAG with Query Rewriting" },
    { value: "llama_index", label: "LlamaIndex RAG (Legacy)" },
  ]);

  const [queryRewritingOptions, setQueryRewritingOptions] = useState({
    techniques: [],
    fusion_methods: [],
    parameters: {},
  });

  const [queryStats, setQueryStats] = useState(null);
  const [showStats, setShowStats] = useState(false);

  const [documentContent, setDocumentContent] = useState(null);

  // ===== Resolve backendURL once on mount =====
  useEffect(() => {
    (async () => {
      try {
        const url = await resolveBackendURL();
        console.log("[renderer] Resolved backend URL:", url);
        setBackendURL(url);
      } catch (e) {
        console.error("[renderer] Failed to resolve backend URL:", e);
        setConnected(false);
      }
    })();
  }, []);

  // ===== API helpers (use resolved backendURL only) =====
  const apiGet = async (path, config = {}) => {
    if (!backendURL) throw new Error("backendURL not resolved");
    const full = `${backendURL}${path}`;
    return axios.get(full, config);
  };

  const apiPost = async (path, data, config = {}) => {
    if (!backendURL) throw new Error("backendURL not resolved");
    const full = `${backendURL}${path}`;
    return axios.post(full, data, config);
  };

  // ===== Connect check when backendURL becomes available =====
  useEffect(() => {
    if (!backendURL) return;
    checkServerConnection();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendURL]);

  const checkServerConnection = async () => {
    try {
      console.log("[renderer] Checking server connection to:", backendURL);

      // Use /health for a clean signal
      const health = await apiGet("/health", { timeout: 5000 });
      console.log("[renderer] /health:", health.data);

      const response = await apiGet("/", { timeout: 5000 });
      setServerInfo(response.data);
      setConnected(true);

      // Kick off additional loads
      fetchAvailableModels();
      fetchRagOptions();
      fetchQueryRewritingOptions();
    } catch (error) {
      logAxiosError("[renderer] Server connection error", error);
      setConnected(false);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await apiGet("/models", { timeout: 10000 });
      if (response.data && response.data.models) setAvailableModels(response.data.models);
    } catch (error) {
      logAxiosError("[renderer] Error fetching models", error);
    }
  };

  const fetchRagOptions = async () => {
    try {
      const response = await apiGet("/rag-options", { timeout: 10000 });
      if (response.data && response.data.options) setRagOptions(response.data.options);
    } catch (error) {
      logAxiosError("[renderer] Error fetching RAG options", error);
    }
  };

  const fetchQueryRewritingOptions = async () => {
    try {
      const response = await apiGet("/query-rewriting-options", { timeout: 10000 });
      if (response.data) setQueryRewritingOptions(response.data);
    } catch (error) {
      logAxiosError("[renderer] Error fetching query rewriting options", error);
    }
  };

  const fetchQueryStats = async () => {
    if (!uploadResult) return;
    try {
      const response = await apiGet(
        `/query-rewriting-stats?document=${encodeURIComponent(uploadResult.filename)}`,
        { timeout: 15000 }
      );
      if (response.data && response.data.stats) setQueryStats(response.data.stats);
    } catch (error) {
      logAxiosError("[renderer] Error fetching query stats", error);
    }
  };

  const clearCaches = async () => {
    try {
      await apiPost("/clear-caches", {}, { timeout: 30000 });
      setMessages((prev) => [
        ...prev,
        {
          id: uuidv4(),
          role: "system",
          content:
            "Caches cleared successfully. This may improve performance for new queries.",
          timestamp: new Date().toISOString(),
        },
      ]);
    } catch (error) {
      logAxiosError("[renderer] Error clearing caches", error);
    }
  };

  // ===== Upload handlers =====
  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      setUploadError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setUploadError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadError("Please select a file first");
      return;
    }
    if (!backendURL) {
      setUploadError("Backend URL not ready yet");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("use_advanced_rag", advancedOptions.use_advanced_rag);
    formData.append("use_llama_index", advancedOptions.use_llama_index);

    setUploading(true);
    setUploadProgress(0);

    try {
      const response = await apiPost("/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        },
        timeout: 300000,
      });

      if (!response.data || !response.data.filename) {
        throw new Error("Invalid upload response from server");
      }

      const documentInfo = {
        filename: response.data.filename,
        chunks: response.data.chunks || 0,
        preview: response.data.preview || "",
      };

      setUploadResult(documentInfo);

      if (fileInputRef.current) fileInputRef.current.value = "";
      setFile(null);
      setMessages([]);

      try {
        await apiPost("/set_document", { document: documentInfo.filename }, { timeout: 10000 });
      } catch (err) {
        logAxiosError("[renderer] Error setting document", err);
      }

      try {
        const res = await apiGet(
          `/suggestions?document=${encodeURIComponent(documentInfo.filename)}`,
          { timeout: 30000 }
        );
        setSuggestedQuestions(res.data.questions || []);
      } catch (err) {
        logAxiosError("[renderer] Error fetching suggestions", err);
      }

      const fileExtension = documentInfo.filename.toLowerCase().split(".").pop();
      const isImage = ["jpg", "jpeg", "png", "gif", "bmp", "webp"].includes(fileExtension);
      const isPdf = fileExtension === "pdf";

      if (isImage || isPdf) {
        try {
          const contentResponse = await apiGet(
            `/document-text/${encodeURIComponent(documentInfo.filename)}`,
            { timeout: 30000 }
          );
          setDocumentContent(contentResponse.data);
        } catch (err) {
          logAxiosError("[renderer] Error fetching document content", err);
        }
      }

      setShowPdfViewer(isPdf || isImage);

      setMessages([
        {
          id: uuidv4(),
          role: "system",
          content: `Document "${documentInfo.filename}" uploaded successfully. You can now ask questions about it.`,
          timestamp: new Date().toISOString(),
        },
      ]);

      setTimeout(() => {
        const textArea = document.querySelector("textarea");
        if (textArea) textArea.focus();
      }, 300);
    } catch (error) {
      logAxiosError("[renderer] Upload error", error);
      setUploadError(error.response?.data?.detail || "Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  // ===== Chat helpers =====
  const handleInputChange = (e) => setInput(e.target.value);

  const formatResponseForDisplay = (text, sources, queryRewritingInfo) => {
    if (!text) return "No response received.";
    let formattedText = text.replace(/^\s*-\s+/gm, "• ");

    if (queryRewritingInfo?.all_queries?.length > 1) {
      const rewritingDetails = [];

      if (queryRewritingInfo.techniques_used) {
        const techniques = Object.entries(queryRewritingInfo.techniques_used)
          .filter(([, value]) => value)
          .map(([key, value]) => {
            switch (key) {
              case "prf":
                return "Pseudo Relevance Feedback";
              case "variants":
                return "Query Variants";
              case "reranking":
                return "Cross-Encoder Reranking";
              case "fusion":
                return `Result Fusion (${value})`;
              default:
                return key;
            }
          });

        if (techniques.length > 0) rewritingDetails.push(`🔍 Enhanced with: ${techniques.join(", ")}`);
      }

      rewritingDetails.push(`📝 Generated ${queryRewritingInfo.all_queries.length} query variants`);

      if (queryRewritingInfo.query_time_ms) {
        rewritingDetails.push(`⚡ Query time: ${Math.round(queryRewritingInfo.query_time_ms)}ms`);
      }

      if (rewritingDetails.length > 0) {
        formattedText = `${formattedText}\n\n${rewritingDetails.join("\n")}`;
      }
    }

    if (sources && sources.length > 0) {
      const uniqueSources = [...new Set(sources)];
      if (uniqueSources.length === 1 && uniqueSources[0] === uploadResult?.filename) {
        return formattedText;
      }
      const sourcesList = uniqueSources.map((s) => `• ${s}`).join("\n");
      return `${formattedText}\n\nSources:\n${sourcesList}`;
    }

    return formattedText;
  };

  const handleOptionChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue =
      type === "checkbox" ? checked : type === "number" ? parseFloat(value) : value;

    setAdvancedOptions((prev) => ({ ...prev, [name]: newValue }));
  };

  const togglePdfViewer = () => setShowPdfViewer((v) => !v);

  const handleSelectMessage = (message) => {
    if (message.role === "assistant" && message.id !== selectedMessage?.id) {
      setSelectedMessage(message);
    }
  };

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => scrollToBottom(), [messages]);

  // ===== Query submit =====
  const handleSubmit = async () => {
    if (!input.trim() || loading || !uploadResult || !connected) return;

    const currentInput = input.trim();
    setInput("");
    setLoading(true);

    const userMessageId = uuidv4();
    const userMessage = {
      id: userMessageId,
      role: "user",
      content: currentInput,
      timestamp: new Date().toISOString(),
    };

    const thinkingMessage = {
      id: uuidv4(),
      role: "assistant",
      content:
        advancedOptions.use_prf || advancedOptions.use_variants
          ? "🔍 Analyzing query and generating variants..."
          : "Thinking...",
      isThinking: true,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage, thinkingMessage]);

    try {
      // ensure backend knows document
      try {
        await apiPost("/set_document", { document: uploadResult.filename }, { timeout: 10000 });
      } catch {}

      const response = await apiPost(
        "/query-with-context",
        {
          query: currentInput,
          document: uploadResult.filename,
          model: advancedOptions.model,
          temperature: advancedOptions.temperature,
          context_window: advancedOptions.context_window,
          quantization: advancedOptions.quantization,
          use_advanced_rag: advancedOptions.use_advanced_rag,
          use_llama_index: advancedOptions.use_llama_index,
          use_prf: advancedOptions.use_prf,
          use_variants: advancedOptions.use_variants,
          prf_iterations: advancedOptions.prf_iterations,
          fusion_method: advancedOptions.fusion_method,
          rerank: advancedOptions.rerank,
        },
        { timeout: 300000 }
      );

      setMessages((prev) => prev.filter((m) => m.id !== thinkingMessage.id));

      if (!response.data) throw new Error("Invalid response from server");

      let contextInfo = null;
      if (response.data.context_tracking) {
        contextInfo = contextTrackingService.processResponseContext(response.data, thinkingMessage.id);
      }

      const assistantMessageId = uuidv4();
      const formattedResponse = formatResponseForDisplay(
        response.data.response,
        response.data.sources,
        response.data.query_rewriting_info
      );

      const assistantMessage = {
        id: assistantMessageId,
        role: "assistant",
        content: formattedResponse,
        rawSources: response.data.sources || [],
        system: response.data.system || "default",
        queryRewritingInfo: response.data.query_rewriting_info,
        enhancementStats: response.data.enhancement_stats,
        timestamp: new Date().toISOString(),
        contextRanges: contextInfo?.contextRanges || [],
        contextText: contextInfo?.contextText || "",
        responseId: contextInfo?.responseId || assistantMessageId,
        replyTo: userMessageId,
      };

      setMessages((prev) => [...prev.filter((m) => !m.isThinking), assistantMessage]);
      setSelectedMessage(assistantMessage);
    } catch (error) {
      logAxiosError("[renderer] Query error", error);

      setMessages((prev) => prev.filter((m) => m.id !== thinkingMessage.id));

      setMessages((prev) => [
        ...prev.filter((m) => !m.isThinking),
        {
          id: uuidv4(),
          role: "system",
          content: "I couldn't process your question. Check logs for details.",
          isError: true,
          timestamp: new Date().toISOString(),
        },
      ]);

      onError(error.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="document-chat-page">
      <header className="page-header">
        <h1>Enhanced Document Chat with Context Visualization</h1>
        <p>Upload documents, chat, and see exactly what content was used to generate answers</p>
        {backendURL && (
          <p style={{ opacity: 0.7, fontSize: 12 }}>
            Backend: <code>{backendURL}</code>
          </p>
        )}
      </header>

      {/* Connection Status */}
      <div className={`connection-status ${connected ? "connected" : "disconnected"}`}>
        <span className="status-indicator"></span>
        <span>{connected ? "Connected" : "Disconnected"}</span>
      </div>

      <div className="enhanced-chat-layout">
        {/* Left sidebar */}
        <div className="upload-sidebar">
          <div className="upload-card">
            <div className="card-header">
              <h2>Upload Document</h2>
            </div>

            <div
              className={`dropzone ${file ? "active" : ""} ${uploading ? "uploading" : ""}`}
              onClick={() => !uploading && fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: "none" }}
                accept=".pdf,.doc,.docx,.txt,.csv,.xlsx,.jpg,.jpeg,.png"
              />
              <div className="dropzone-content">
                <div className="upload-icon">
                  <span>📄</span>
                </div>
                <p className="upload-text">{file ? file.name : "Drag & drop or click to upload"}</p>
                <p className="file-types">
                  {file ? `${(file.size / 1024).toFixed(2)} KB` : "PDF, DOCX, TXT, CSV, XLSX, JPG/PNG"}
                </p>
              </div>
            </div>

            {uploading && (
              <div className="progress-container">
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${uploadProgress}%` }}></div>
                </div>
                <div className="progress-text">Uploading: {uploadProgress}%</div>
              </div>
            )}

            {uploadError && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                <p>{uploadError}</p>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={!file || uploading || !connected}
              className={`upload-button ${(!file || uploading || !connected) ? "disabled" : ""}`}
            >
              {uploading ? "Uploading..." : "Upload Document"}
            </button>
          </div>
        </div>

        {/* Middle chat panel */}
        <div className="chat-container">
          <div className="chat-header">
            <div className="chat-title">
              <h2>Document Chat</h2>
              <p>{uploadResult ? `Ask questions about: ${uploadResult.filename}` : "Upload a document to start chatting"}</p>
            </div>
          </div>

          <div className="messages-container">
            {!connected ? (
              <div className="empty-state">
                <div className="empty-icon disconnected">🔄</div>
                <h3>Connecting to Server...</h3>
                <p>Backend URL: {backendURL ? backendURL : "resolving..."}</p>
              </div>
            ) : !uploadResult ? (
              <div className="empty-state">
                <div className="empty-icon">📄</div>
                <h3>No Document Uploaded</h3>
                <p>Upload a document using the panel on the left to start asking questions.</p>
              </div>
            ) : messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">💬</div>
                <h3>Start the Conversation</h3>
                <p>Your document is ready. Type a question below to start chatting.</p>
              </div>
            ) : (
              <div className="messages-list">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`message ${message.role} ${message.isThinking ? "thinking" : ""} ${
                      message.isError ? "error" : ""
                    } ${message.contextRanges?.length ? "has-context" : ""} ${
                      selectedMessage?.id === message.id ? "selected" : ""
                    }`}
                    onClick={() => handleSelectMessage(message)}
                  >
                    <div className="message-content">{message.content}</div>
                    <div className="message-footer">
                      <span className="message-time">
                        {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </span>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input area */}
          <div className="input-container">
            <textarea
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder={
                connected && uploadResult
                  ? "Type your question..."
                  : connected
                  ? "Upload a document to start chatting"
                  : "Connecting to server..."
              }
              disabled={!connected || !uploadResult || loading}
              className={`chat-input ${(!connected || !uploadResult) ? "disabled" : ""}`}
              rows={1}
            />
            <button
              onClick={handleSubmit}
              disabled={!connected || !uploadResult || !input.trim() || loading}
              className={`send-button ${(!connected || !uploadResult || !input.trim() || loading) ? "disabled" : ""}`}
            >
              {loading ? "Thinking..." : "Send"}
            </button>
          </div>
        </div>

        {/* Right PDF viewer panel */}
        {showPdfViewer && uploadResult && (
          <div className="pdf-viewer-panel">
            <div className="pdf-viewer-header">
              <h3>Document Viewer</h3>
              <button onClick={() => setShowPdfViewer(false)} className="close-button">✕</button>
            </div>

            {uploadResult.filename.toLowerCase().endsWith(".pdf") ? (
              <PDFContextViewer
                documentName={uploadResult.filename}
                currentQuery={selectedMessage?.replyTo ? messages.find((m) => m.id === selectedMessage.replyTo)?.content : null}
                currentAnswer={selectedMessage?.content}
                currentMessage={selectedMessage}
              />
            ) : documentContent?.is_image ? (
              <div className="image-viewer">
                <div className="image-container">
                  {/* IMPORTANT: use backendURL, NOT async getBackendURL() */}
                  <img
                    src={`${backendURL}/document/${encodeURIComponent(uploadResult.filename)}`}
                    alt={uploadResult.filename}
                    className="document-image"
                  />
                </div>
                {documentContent.extracted_text && (
                  <div className="image-text-content">
                    <h4>Extracted Text</h4>
                    <div className="text-content">{documentContent.extracted_text}</div>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedDocumentChat;
