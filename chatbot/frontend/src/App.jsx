import React, { useState, useEffect, useCallback } from "react";
import { HashRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import axios from "axios";

import ModelDownloadPage from "./components/ModelDownloadPage";
import RAGEvaluationDashboard from "./pages/documents/RAGEvaluationDashboard";
import EnhancedDocumentChat from "./pages/documents/EnhancedDocumentChat";

// IMPORTANT: use the same helper your chat uses
import { getBackendURL } from "@/api/baseURL";

const NotFound = () => (
  <div className="page-container">
    <h1>404</h1>
    <p>Page not found</p>
  </div>
);

const StreamingIndicator = () => (
  <div className="global-streaming-indicator">
    <div className="pulse-dot"></div>
    <span>AI is responding...</span>
  </div>
);

function App() {
  const [isStreaming, setIsStreaming] = useState(false);

  // Backend URL resolved once (dynamic port in Electron)
  const [backendURL, setBackendURL] = useState(null);
  const [backendInitError, setBackendInitError] = useState(null);

  const [serverStatus, setServerStatus] = useState({
    initialized: false,
    streamingSupported: true,
    modelManagementSupported: false,
  });

  const [modelState, setModelState] = useState({
    needsDownload: false,
    selectedModel: "llama2",
    downloadCompleted: false,
    availableModels: [],
  });

  // ----------------------------
  // Resolve backend URL once
  // ----------------------------
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const url = await getBackendURL(); // <-- dynamic in Electron
        if (cancelled) return;

        setBackendURL(url);

        // Set axios global baseURL so any component that uses axios without a base URL works
        axios.defaults.baseURL = url;
        axios.defaults.headers.common["Accept"] = "application/json";

      } catch (err) {
        console.error("Failed to resolve backend URL:", err);
        if (!cancelled) setBackendInitError(err?.message || String(err));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // ----------------------------
  // Check server capabilities once backendURL is known
  // ----------------------------
  const checkModelStatus = useCallback(async () => {
    try {
      const response = await axios.get("/models"); // uses axios.defaults.baseURL
      const availableModels = response.data?.models || [];
      const downloadedModels = response.data?.downloaded_models || [];

      setModelState((prev) => ({
        ...prev,
        availableModels,
      }));

      if (downloadedModels.length === 0) {
        setModelState((prev) => ({
          ...prev,
          needsDownload: true,
          selectedModel: availableModels[0] || "llama2",
        }));
      } else {
        setModelState((prev) => ({
          ...prev,
          needsDownload: false,
          selectedModel: downloadedModels[0],
        }));
      }
    } catch (error) {
      console.error("Error checking model status:", error);
    }
  }, []);

  useEffect(() => {
    if (!backendURL) return;

    let cancelled = false;

    (async () => {
      try {
        const response = await axios.get("/"); // uses axios.defaults.baseURL

        // Your backend returns features as an array of strings, not [{id, enabled}]
        // Example: ["Universal Entity Extraction", ...]
        // So we can't detect streaming/model management from your current backend response.
        // We'll keep safe defaults.
        const features = response.data?.features || [];

        // If you later implement structured features, this logic will still work:
        const streamingSupported = Array.isArray(features)
          ? true
          : true;

        const modelManagementSupported = false; // your backend doesn't expose downloaded_models manager currently

        if (cancelled) return;

        setServerStatus({
          initialized: true,
          streamingSupported,
          modelManagementSupported,
        });

        if (modelManagementSupported) {
          await checkModelStatus();
        }
      } catch (error) {
        console.warn("Could not check server capabilities:", error);
        if (!cancelled) {
          setServerStatus({
            initialized: true,
            streamingSupported: true,
            modelManagementSupported: false,
          });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [backendURL, checkModelStatus]);

  // ----------------------------
  // Handlers
  // ----------------------------
  const handleStreamingStateChange = (isActive) => setIsStreaming(isActive);

  const handleChatError = (error) => {
    console.error("Chat error:", error);
  };

  const handleModelDownloadComplete = (model) => {
    console.log(`Model ${model} download complete`);
    setModelState((prev) => ({
      ...prev,
      needsDownload: false,
      selectedModel: model,
      downloadCompleted: true,
    }));
  };

  // ----------------------------
  // Render guards
  // ----------------------------
  if (backendInitError) {
    return (
      <div className="page-container">
        <h2>Backend connection failed</h2>
        <p>{backendInitError}</p>
        <p style={{ maxWidth: 720 }}>
          This usually means the Electron main process didn’t expose the backend URL
          to the renderer, or the backend didn’t start.
        </p>
      </div>
    );
  }

  if (!backendURL) {
    return (
      <div className="page-container">
        <h2>Starting…</h2>
        <p>Resolving backend URL…</p>
      </div>
    );
  }

  // Conditionally render model download page OR chat
  const contentElement =
    serverStatus.modelManagementSupported && modelState.needsDownload ? (
      <ModelDownloadPage
        onDownloadComplete={handleModelDownloadComplete}
        selectedModel={modelState.selectedModel}
      />
    ) : (
      <EnhancedDocumentChat
        backendURL={backendURL} // <-- give it to the page (recommended)
        onStreamingStateChange={handleStreamingStateChange}
        onError={handleChatError}
        initialStreamingEnabled={serverStatus.streamingSupported}
        selectedModel={modelState.selectedModel}
      />
    );

  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/" element={<Navigate to="/documents/upload" replace />} />
          <Route path="/documents/upload" element={contentElement} />
          <Route
            path="/models/download"
            element={
              <ModelDownloadPage
                onDownloadComplete={handleModelDownloadComplete}
                selectedModel={modelState.selectedModel}
              />
            }
          />
          <Route path="/evaluation" element={<RAGEvaluationDashboard />} />
          <Route path="*" element={<NotFound />} />
        </Routes>

        {isStreaming && <StreamingIndicator />}
      </div>

      <style jsx="true">{`
        .app {
          height: 100vh;
          position: relative;
        }
        .global-streaming-indicator {
          position: fixed;
          bottom: 20px;
          right: 20px;
          background-color: rgba(59, 130, 246, 0.9);
          color: white;
          padding: 8px 16px;
          border-radius: 20px;
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
          z-index: 1000;
          animation: fadeIn 0.3s ease-in-out;
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .pulse-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background-color: white;
          animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
          0% {
            transform: scale(0.8);
            opacity: 0.5;
          }
          50% {
            transform: scale(1.2);
            opacity: 1;
          }
          100% {
            transform: scale(0.8);
            opacity: 0.5;
          }
        }
        .page-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100vh;
          text-align: center;
          padding: 24px;
        }
      `}</style>
    </Router>
  );
}

export default App;
