import React, { useState, useEffect } from 'react';
import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';

// Import components with the correct paths
import MCPDocumentChat from './pages/documents/MCPDocumentChat';
import ModelDownloadPage from './components/ModelDownloadPage'; // Updated import path

// Simple placeholder component for not found page
const NotFound = () => <div className="page-container"><h1>404</h1><p>Page not found</p></div>;

// Optional global loading indicator for streaming state
const StreamingIndicator = () => (
  <div className="global-streaming-indicator">
    <div className="pulse-dot"></div>
    <span>AI is responding...</span>
  </div>
);

function App() {
  // Global state for streaming status
  const [isStreaming, setIsStreaming] = useState(false);
  const [serverStatus, setServerStatus] = useState({
    initialized: false,
    streamingSupported: true,
    modelManagementSupported: false
  });
  
  // Model related state
  const [modelState, setModelState] = useState({
    needsDownload: false,
    selectedModel: "llama2",
    downloadCompleted: false,
    availableModels: []
  });

  // Function to check server capabilities on app load
  useEffect(() => {
    const checkServerCapabilities = async () => {
      try {
        // You can adjust this URL based on your API configuration
        const baseUrl = 'http://localhost:8000';
        const response = await axios.get(`${baseUrl}/`);
        
        // Check if streaming is supported
        const features = response.data.features || [];
        const streamingSupported = features.some(
          f => f.id === 'streaming' && f.enabled
        );
        
        // Check if model management is supported
        const modelManagementSupported = features.some(
          f => f.id === 'model_management' && f.enabled
        );

        setServerStatus({
          initialized: true,
          streamingSupported,
          modelManagementSupported
        });
        
        // Check if we need to download a model
        if (modelManagementSupported) {
          checkModelStatus();
        }
      } catch (error) {
        console.warn('Could not check server capabilities:', error);
        // Default to assuming streaming is supported if we can't check
        setServerStatus({
          initialized: true,
          streamingSupported: true,
          modelManagementSupported: false
        });
      }
    };

    checkServerCapabilities();
  }, []);
  
  // Check if we need to download a model
  const checkModelStatus = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/models`);
      const availableModels = response.data.models || [];
      const downloadedModels = response.data.downloaded_models || [];
      
      // Set the available models
      setModelState(prev => ({
        ...prev,
        availableModels: availableModels
      }));
      
      // If no models are downloaded, show the download page
      if (downloadedModels.length === 0) {
        setModelState(prev => ({
          ...prev,
          needsDownload: true,
          selectedModel: availableModels[0] || "llama2"
        }));
      } else {
        // Set the selected model to the first downloaded model
        setModelState(prev => ({
          ...prev,
          needsDownload: false,
          selectedModel: downloadedModels[0]
        }));
      }
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  };

  // Handler for streaming state changes
  const handleStreamingStateChange = (isActive) => {
    setIsStreaming(isActive);
  };

  // Handler for errors from the chat component
  const handleChatError = (error) => {
    console.error('Chat error:', error);
  };
  
  // Handler for when model download completes
  const handleModelDownloadComplete = (model) => {
    console.log(`Model ${model} download complete`);
    setModelState(prev => ({
      ...prev,
      needsDownload: false,
      selectedModel: model,
      downloadCompleted: true
    }));
  };

  // Conditionally render the model download page or chat interface
  const renderContent = () => {
    // If the server supports model management and we need to download a model
    if (serverStatus.modelManagementSupported && modelState.needsDownload) {
      return (
        <ModelDownloadPage 
          onDownloadComplete={handleModelDownloadComplete}
          selectedModel={modelState.selectedModel}
          autoDownload={false}
        />
      );
    }
    
    // Otherwise, show the chat interface
    return (
      <MCPDocumentChat 
        onStreamingStateChange={handleStreamingStateChange}
        onError={handleChatError}
        initialStreamingEnabled={serverStatus.streamingSupported}
        selectedModel={modelState.selectedModel}
      />
    );
  };

  return (
    <Router>
      <div className="app">
        <Routes>
          {/* Redirect root path to the document upload page */}
          <Route path="/" element={<Navigate to="/documents/upload" replace />} />
          
          {/* Document chat with conditional model download */}
          <Route path="/documents/upload" element={renderContent()} />
          
          {/* Explicit route for model download page */}
          <Route path="/models/download" element={
            <ModelDownloadPage 
              onDownloadComplete={handleModelDownloadComplete}
              selectedModel={modelState.selectedModel}
            />
          } />
          
          <Route path="*" element={<NotFound />} />
        </Routes>
        
        {/* Global streaming indicator */}
        {isStreaming && <StreamingIndicator />}
      </div>

      {/* Global styles */}
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
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .pulse-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background-color: white;
          animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
          0% { transform: scale(0.8); opacity: 0.5; }
          50% { transform: scale(1.2); opacity: 1; }
          100% { transform: scale(0.8); opacity: 0.5; }
        }
        
        .page-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100vh;
          text-align: center;
        }
      `}</style>
    </Router>
  );
}

export default App;