import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

// Helper function to get backend URL
const getBackendURL = () => {
  return process.env.NODE_ENV === 'production' 
    ? 'http://localhost:8000' 
    : 'http://localhost:8000';
};

const ModelDownloadPage = ({ 
  onDownloadComplete = () => {}, 
  selectedModel = "llama2", 
  autoDownload = true 
}) => {
  // Download state
  const [downloadStatus, setDownloadStatus] = useState({
    isDownloading: false,
    progress: 0,
    status: 'idle', // 'idle', 'downloading', 'completed', 'error'
    error: null,
    modelInfo: null
  });

  // Available models
  const [availableModels, setAvailableModels] = useState([
    { id: "llama2", name: "Llama 2 7B", size: "3.8 GB" },
    { id: "mistral", name: "Mistral 7B", size: "4.1 GB" },
    { id: "neural-chat", name: "Neural Chat 7B", size: "4.1 GB" }
  ]);
  
  // Current model selection
  const [modelSelection, setModelSelection] = useState(selectedModel);

  // Fetch available models on component mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  // Auto-start download if specified
  useEffect(() => {
    if (autoDownload && downloadStatus.status === 'idle') {
      handleStartDownload();
    }
  }, [autoDownload, downloadStatus.status]);

  // Poll download progress
  useEffect(() => {
    let pollInterval;
    
    if (downloadStatus.isDownloading) {
      pollInterval = setInterval(() => {
        checkDownloadProgress();
      }, 1000);
    }
    
    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [downloadStatus.isDownloading]);

  // Fetch available models
  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${getBackendURL()}/models`);
      if (response.data && response.data.models) {
        // Check for models that are already downloaded vs. available
        const models = response.data.models.map(model => ({
          id: model,
          name: modelNameFormatter(model),
          size: estimateModelSize(model),
          downloaded: response.data.downloaded_models?.includes(model) || false
        }));
        
        setAvailableModels(models);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  // Start model download
  const handleStartDownload = useCallback(async () => {
    try {
      setDownloadStatus({
        isDownloading: true,
        progress: 0,
        status: 'downloading',
        error: null,
        modelInfo: availableModels.find(m => m.id === modelSelection)
      });
      
      // Initiate download
      await axios.post(`${getBackendURL()}/download-model`, { 
        model: modelSelection 
      });
      
      // The download happens asynchronously on the server,
      // so we'll poll for progress
    } catch (error) {
      setDownloadStatus(prev => ({
        ...prev,
        isDownloading: false,
        status: 'error',
        error: error.response?.data?.detail || 'Failed to start download'
      }));
    }
  }, [modelSelection, availableModels]);

  // Check download progress
  const checkDownloadProgress = async () => {
    try {
      const response = await axios.get(`${getBackendURL()}/download-status`, {
        params: { model: modelSelection }
      });
      
      const { progress, status } = response.data;
      
      setDownloadStatus(prev => ({
        ...prev,
        progress: progress,
        status: status
      }));
      
      // If download completed
      if (status === 'completed') {
        setDownloadStatus(prev => ({
          ...prev,
          isDownloading: false,
          progress: 100
        }));
        
        // Wait a moment to show the 100% progress state
        setTimeout(() => {
          onDownloadComplete(modelSelection);
        }, 1000);
      }
      
      // If download failed
      if (status === 'error') {
        setDownloadStatus(prev => ({
          ...prev,
          isDownloading: false,
          error: response.data.error || 'Download failed'
        }));
      }
    } catch (error) {
      console.error('Error checking download status:', error);
    }
  };

  // For demo purposes: Simulate download progress
  // Remove this in production and use the real API
  const simulateDownloadProgress = useCallback(() => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 5;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        
        setDownloadStatus({
          isDownloading: false,
          progress: 100,
          status: 'completed',
          error: null,
          modelInfo: availableModels.find(m => m.id === modelSelection)
        });
        
        // Notify parent that download is complete
        setTimeout(() => {
          onDownloadComplete(modelSelection);
        }, 1000);
        
        return;
      }
      
      setDownloadStatus(prev => ({
        ...prev,
        progress: progress,
      }));
    }, 600);
    
    return () => clearInterval(interval);
  }, [modelSelection, availableModels, onDownloadComplete]);

  // Handle model change
  const handleModelChange = (e) => {
    setModelSelection(e.target.value);
  };

  // For demo: Format model names nicely
  const modelNameFormatter = (modelId) => {
    // Find in our available models
    const knownModel = availableModels.find(m => m.id === modelId);
    if (knownModel) return knownModel.name;
    
    // Otherwise, format the ID
    return modelId
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // For demo: Estimate model size based on ID
  const estimateModelSize = (modelId) => {
    const knownModel = availableModels.find(m => m.id === modelId);
    if (knownModel?.size) return knownModel.size;
    
    // If we don't know, make a guess based on the name
    if (modelId.includes('7b') || modelId.includes('llama2')) return "4.0 GB";
    if (modelId.includes('13b')) return "7.8 GB";
    if (modelId.includes('70b')) return "35 GB";
    return "5-8 GB";
  };

  // Start simulated download for demo
  const handleDemoDownload = () => {
    setDownloadStatus({
      isDownloading: true,
      progress: 0,
      status: 'downloading',
      error: null,
      modelInfo: availableModels.find(m => m.id === modelSelection)
    });
    
    // Start the simulation
    simulateDownloadProgress();
  };

  // Calculate estimated time remaining
  const getEstimatedTime = (progress) => {
    if (progress <= 0 || progress >= 100) return '';
    
    // This is a very simplified calculation
    const modelInfo = downloadStatus.modelInfo;
    if (!modelInfo) return '';
    
    // Extract size in GB from string like "3.4 GB"
    const sizeMatch = modelInfo.size.match(/(\d+(\.\d+)?)/);
    if (!sizeMatch) return '';
    
    const sizeInGB = parseFloat(sizeMatch[1]);
    const remainingPercent = 100 - progress;
    const averageSpeedMbps = 10; // Assume 10 Mbps download speed
    
    // Convert GB to Mb, calculate time in seconds
    const remainingSizeMb = (sizeInGB * remainingPercent / 100) * 1024;
    const remainingTimeSeconds = remainingSizeMb / averageSpeedMbps;
    
    // Format time
    if (remainingTimeSeconds < 60) {
      return 'less than a minute remaining';
    } else if (remainingTimeSeconds < 3600) {
      return `about ${Math.ceil(remainingTimeSeconds / 60)} minutes remaining`;
    } else {
      const hours = Math.floor(remainingTimeSeconds / 3600);
      const minutes = Math.ceil((remainingTimeSeconds % 3600) / 60);
      return `about ${hours} hour${hours > 1 ? 's' : ''} ${minutes > 0 ? `and ${minutes} minute${minutes > 1 ? 's' : ''}` : ''} remaining`;
    }
  };

  return (
    <div className="model-download-page">
      <div className="download-card">
        <div className="download-header">
          <h2>Model Download Required</h2>
          <p>
            To use the document chat, we need to download a language model. 
            This is a one-time process and the model will be stored locally.
          </p>
        </div>
        
        <div className="model-selector">
          <label htmlFor="model-select">Select a model:</label>
          <select 
            id="model-select" 
            value={modelSelection}
            onChange={handleModelChange}
            disabled={downloadStatus.isDownloading}
          >
            {availableModels.map(model => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.size})
                {model.downloaded ? ' - Already Downloaded' : ''}
              </option>
            ))}
          </select>
        </div>
        
        {downloadStatus.status === 'idle' && (
          <div className="model-info">
            <h3>{modelNameFormatter(modelSelection)}</h3>
            <p>Size: {estimateModelSize(modelSelection)}</p>
            <p>
              This model will be downloaded and stored locally. You can change the model later in settings.
            </p>
            <button 
              className="download-button"
              onClick={handleDemoDownload} // Use handleStartDownload in production
            >
              Download Model
            </button>
          </div>
        )}
        
        {downloadStatus.status === 'downloading' && (
          <div className="download-progress">
            <div className="progress-header">
              <h3>Downloading {downloadStatus.modelInfo?.name || modelNameFormatter(modelSelection)}</h3>
              <p className="status-text">
                {downloadStatus.progress.toFixed(1)}% complete
              </p>
            </div>
            
            <div className="progress-bar-container">
              <div 
                className="progress-bar" 
                style={{ width: `${downloadStatus.progress}%` }}
              ></div>
            </div>
            
            <div className="progress-details">
              <p className="time-remaining">
                {getEstimatedTime(downloadStatus.progress)}
              </p>
              <p className="size-info">
                {(parseFloat(downloadStatus.modelInfo?.size) * downloadStatus.progress / 100).toFixed(1)} GB 
                of {downloadStatus.modelInfo?.size}
              </p>
            </div>
            
            <p className="download-note">
              The download may take some time depending on your internet connection.
              Please keep this window open until the download is complete.
            </p>
          </div>
        )}
        
        {downloadStatus.status === 'error' && (
          <div className="download-error">
            <div className="error-icon">❌</div>
            <h3>Download Failed</h3>
            <p>{downloadStatus.error || 'An error occurred while downloading the model.'}</p>
            <button 
              className="retry-button"
              onClick={handleDemoDownload} // Use handleStartDownload in production
            >
              Retry Download
            </button>
          </div>
        )}
        
        {downloadStatus.status === 'completed' && (
          <div className="download-complete">
            <div className="success-icon">✓</div>
            <h3>Download Complete</h3>
            <p>The model has been successfully downloaded and is ready to use.</p>
            <button 
              className="continue-button"
              onClick={() => onDownloadComplete(modelSelection)}
            >
              Continue
            </button>
          </div>
        )}
      </div>
      
      <style jsx>{`
        .model-download-page {
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 100vh;
          padding: 20px;
          background-color: #f8fafc;
        }
        
        .download-card {
          background-color: white;
          width: 100%;
          max-width: 600px;
          border-radius: 10px;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          padding: 30px;
        }
        
        .download-header {
          text-align: center;
          margin-bottom: 24px;
        }
        
        .download-header h2 {
          font-size: 24px;
          margin-bottom: 8px;
          color: #1e293b;
        }
        
        .download-header p {
          color: #64748b;
          font-size: 16px;
        }
        
        .model-selector {
          margin-bottom: 24px;
        }
        
        .model-selector label {
          display: block;
          margin-bottom: 8px;
          font-weight: 500;
          color: #334155;
        }
        
        .model-selector select {
          width: 100%;
          padding: 10px;
          border-radius: 6px;
          border: 1px solid #cbd5e1;
          font-size: 16px;
          color: #1e293b;
        }
        
        .model-info {
          background-color: #f8fafc;
          padding: 16px;
          border-radius: 8px;
          margin-bottom: 24px;
        }
        
        .model-info h3 {
          font-size: 18px;
          margin-bottom: 8px;
          color: #334155;
        }
        
        .model-info p {
          color: #64748b;
          margin-bottom: 16px;
        }
        
        .download-button, .retry-button, .continue-button {
          width: 100%;
          padding: 12px;
          border: none;
          border-radius: 6px;
          font-size: 16px;
          font-weight: 500;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        
        .download-button {
          background-color: #4f46e5;
          color: white;
        }
        
        .download-button:hover {
          background-color: #4338ca;
        }
        
        .retry-button {
          background-color: #ef4444;
          color: white;
        }
        
        .retry-button:hover {
          background-color: #dc2626;
        }
        
        .continue-button {
          background-color: #10b981;
          color: white;
        }
        
        .continue-button:hover {
          background-color: #059669;
        }
        
        .download-progress {
          margin-top: 16px;
        }
        
        .progress-header {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          margin-bottom: 12px;
        }
        
        .progress-header h3 {
          font-size: 18px;
          color: #334155;
          margin: 0;
        }
        
        .status-text {
          font-size: 16px;
          font-weight: 500;
          color: #3b82f6;
          margin: 0;
        }
        
        .progress-bar-container {
          height: 10px;
          background-color: #e2e8f0;
          border-radius: 5px;
          overflow: hidden;
          margin-bottom: 12px;
        }
        
        .progress-bar {
          height: 100%;
          background-color: #3b82f6;
          transition: width 0.5s ease;
        }
        
        .progress-details {
          display: flex;
          justify-content: space-between;
          margin-bottom: 16px;
          font-size: 14px;
          color: #64748b;
        }
        
        .time-remaining, .size-info {
          margin: 0;
        }
        
        .download-note {
          font-size: 14px;
          color: #94a3b8;
          text-align: center;
          margin-top: 24px;
        }
        
        .download-error, .download-complete {
          text-align: center;
          padding: 20px;
        }
        
        .error-icon, .success-icon {
          font-size: 48px;
          margin-bottom: 16px;
        }
        
        .error-icon {
          color: #ef4444;
        }
        
        .success-icon {
          color: #10b981;
          font-weight: bold;
        }
      `}</style>
    </div>
  );
};

export default ModelDownloadPage;