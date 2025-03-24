import React, { useState, useEffect, useRef } from 'react';
import { AdvancedRAGClient, formatAdvancedResponse } from './frontend_integration';

// Create CSS style element
const createStyleElement = () => {
  // Import styles from frontend_integration.js
  const styles = `
.advanced-rag-settings {
  background-color: #f8f9fa;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 20px;
}

.settings-group {
  display: flex;
  align-items: center;
  margin: 10px 0;
}

.switch {
  position: relative;
  display: inline-block;
  width: 48px;
  height: 24px;
  margin-right: 10px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: .4s;
  border-radius: 24px;
}

.slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: .4s;
  border-radius: 50%;
}

input:checked + .slider {
  background-color: #2196F3;
}

input:focus + .slider {
  box-shadow: 0 0 1px #2196F3;
}

input:checked + .slider:before {
  transform: translateX(24px);
}

.save-config-btn {
  margin-top: 16px;
  background-color: #4CAF50;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
}

.save-config-btn:hover {
  background-color: #45a049;
}

.advanced-response-container {
  margin-top: 20px;
}

.answer-section {
  margin-bottom: 16px;
}

.answer {
  padding: 12px;
  border-radius: 6px;
  background-color: #f0f7ff;
  border-left: 4px solid #2196F3;
}

.confidence-badge {
  display: inline-block;
  margin-left: 8px;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.8em;
  font-weight: bold;
}

.confidence-high .confidence-badge {
  background-color: #4CAF50;
  color: white;
}

.confidence-medium .confidence-badge {
  background-color: #FFC107;
  color: black;
}

.confidence-low .confidence-badge {
  background-color: #F44336;
  color: white;
}

details {
  margin: 10px 0;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 8px;
}

summary {
  cursor: pointer;
  padding: 8px;
  font-weight: bold;
}

.reasoning-content, .verification-content {
  padding: 12px;
  background-color: #f9f9f9;
  border-radius: 4px;
  margin-top: 8px;
}

.supported-claim {
  color: #2E7D32;
}

.unsupported-claim {
  color: #C62828;
}

.sources-section {
  margin-top: 16px;
  border-top: 1px solid #eee;
  padding-top: 16px;
}

.file-upload {
  margin-bottom: 20px;
  padding: 20px;
  border: 2px dashed #ccc;
  border-radius: 8px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.3s;
}

.file-upload:hover {
  border-color: #2196F3;
}

.file-upload input {
  display: none;
}

.file-list {
  margin: 20px 0;
}

.file-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  margin: 4px 0;
  background-color: #f5f5f5;
  border-radius: 4px;
}

.upload-status {
  font-size: 0.9em;
  margin-left: 8px;
}

.query-input {
  margin: 20px 0;
}

.query-input textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-family: inherit;
  font-size: 1em;
  height: 100px;
  resize: vertical;
}

.submit-button {
  background-color: #2196F3;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s;
}

.submit-button:hover {
  background-color: #0b7dda;
}

.submit-button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.loading {
  margin: 20px 0;
  text-align: center;
  color: #666;
}

.loader {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid rgba(0, 0, 0, 0.1);
  border-radius: 50%;
  border-top-color: #2196F3;
  animation: spin 1s ease infinite;
  margin-right: 10px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.knowledge-graph-view {
  margin-top: 20px;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 16px;
}

.graph-controls {
  margin-bottom: 16px;
}

.graph-container {
  height: 400px;
  border: 1px solid #eee;
  border-radius: 4px;
}

.error {
  background-color: #ffebee;
  color: #c62828;
  padding: 12px;
  border-radius: 4px;
  margin: 20px 0;
}

.model-selector {
  margin: 12px 0;
}

.model-selector select {
  padding: 8px;
  border-radius: 4px;
  border: 1px solid #ddd;
  margin-left: 8px;
}

.quantization-selector {
  margin: 12px 0;
}

.quantization-selector select {
  padding: 8px;
  border-radius: 4px;
  border: 1px solid #ddd;
  margin-left: 8px;
}

.temp-slider {
  margin: 12px 0;
  display: flex;
  align-items: center;
}

.temp-slider label {
  margin-right: 10px;
}

.temp-slider input {
  width: 200px;
}

.temp-value {
  margin-left: 10px;
  min-width: 40px;
}

.advanced-settings-toggle {
  margin: 12px 0;
  cursor: pointer;
  color: #2196F3;
  display: inline-block;
}

.advanced-settings-toggle:hover {
  text-decoration: underline;
}

.advanced-settings-panel {
  background-color: #f5f5f5;
  border-radius: 4px;
  padding: 12px;
  margin-top: 8px;
}

.upload-results {
  margin: 20px 0;
  padding: 12px;
  background-color: #e8f5e9;
  border-radius: 4px;
}

.success {
  color: #2e7d32;
}

.failure {
  color: #c62828;
}

.message-history {
  margin: 20px 0;
  border: 1px solid #ddd;
  border-radius: 4px;
  max-height: 400px;
  overflow-y: auto;
}

.message-item {
  padding: 12px;
  border-bottom: 1px solid #eee;
}

.message-item:last-child {
  border-bottom: none;
}

.user-message {
  background-color: #e3f2fd;
}

.system-message {
  background-color: #f5f5f5;
}

.clear-history {
  margin-top: 8px;
  background-color: #f44336;
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9em;
}

.clear-history:hover {
  background-color: #d32f2f;
}
  `;

  const styleElement = document.createElement('style');
  styleElement.textContent = styles;
  return styleElement;
};

// Settings component
const AdvancedRAGSettings = ({ client, onUpdate }) => {
  const [config, setConfig] = useState(client.config);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleToggle = (key) => {
    const newConfig = { ...config, [key]: !config[key] };
    setConfig(newConfig);
    client.updateConfig(newConfig);
    onUpdate(newConfig);
  };

  const handleTemperatureChange = (e) => {
    const temp = parseFloat(e.target.value);
    const newConfig = { ...config, temperature: temp };
    setConfig(newConfig);
    client.updateConfig(newConfig);
    onUpdate(newConfig);
  };

  const handleModelChange = (e) => {
    const model = e.target.value;
    const newConfig = { ...config, model };
    setConfig(newConfig);
    client.updateConfig(newConfig);
    onUpdate(newConfig);
  };

  const handleQuantizationChange = (e) => {
    const quantization = e.target.value;
    const newConfig = { ...config, quantization };
    setConfig(newConfig);
    client.updateConfig(newConfig);
    onUpdate(newConfig);
  };

  const saveConfiguration = async () => {
    try {
      await client.saveConfiguration();
      alert('Configuration saved successfully!');
    } catch (error) {
      alert('Error saving configuration');
    }
  };

  return (
    <div className="advanced-rag-settings">
      <h3>Advanced RAG Settings</h3>
      
      {/* Main toggle */}
      <div className="settings-group">
        <label className="switch">
          <input
            type="checkbox"
            checked={config.use_advanced_rag}
            onChange={() => handleToggle('use_advanced_rag')}
          />
          <span className="slider"></span>
        </label>
        <span>Enable Advanced RAG</span>
      </div>
      
      {/* Model selector */}
      <div className="model-selector">
        <label>
          Model:
          <select value={config.model} onChange={handleModelChange}>
            {client.availableModels.map((model, index) => (
              <option key={index} value={model}>{model}</option>
            ))}
          </select>
        </label>
      </div>
      
      {/* Basic temperature control */}
      <div className="temp-slider">
        <label>Temperature:</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={config.temperature}
          onChange={handleTemperatureChange}
        />
        <span className="temp-value">{config.temperature}</span>
      </div>
      
      {/* Quantization selector */}
      <div className="quantization-selector">
        <label>
          Quantization:
          <select value={config.quantization} onChange={handleQuantizationChange}>
            {client.quantizationOptions.map((option, index) => (
              <option key={index} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
      </div>
      
      {/* Advanced settings toggle */}
      <div 
        className="advanced-settings-toggle"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        {showAdvanced ? '▼ Hide Advanced Settings' : '► Show Advanced Settings'}
      </div>
      
      {/* Feature toggles (only shown if advanced settings are visible) */}
      {showAdvanced && (
        <div className="advanced-settings-panel">
          {/* Chain of Thought */}
          <div className="settings-group">
            <label className="switch">
              <input
                type="checkbox"
                checked={config.use_cot}
                onChange={() => handleToggle('use_cot')}
              />
              <span className="slider"></span>
            </label>
            <span>Chain of Thought Reasoning</span>
          </div>
          
          {/* Knowledge Graph */}
          <div className="settings-group">
            <label className="switch">
              <input
                type="checkbox"
                checked={config.use_kg}
                onChange={() => handleToggle('use_kg')}
              />
              <span className="slider"></span>
            </label>
            <span>Knowledge Graph</span>
          </div>
          
          {/* Answer Verification */}
          <div className="settings-group">
            <label className="switch">
              <input
                type="checkbox"
                checked={config.verify_answers}
                onChange={() => handleToggle('verify_answers')}
              />
              <span className="slider"></span>
            </label>
            <span>Answer Verification</span>
          </div>
          
          {/* Multi-hop Reasoning */}
          <div className="settings-group">
            <label className="switch">
              <input
                type="checkbox"
                checked={config.use_multihop}
                onChange={() => handleToggle('use_multihop')}
              />
              <span className="slider"></span>
            </label>
            <span>Multi-hop Reasoning</span>
          </div>
          
          {/* Save Configuration Button */}
          <button
            className="save-config-btn"
            onClick={saveConfiguration}
          >
            Save Configuration
          </button>
        </div>
      )}
    </div>
  );
};

// File Upload Component
const FileUploader = ({ client, onUpload }) => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);
  
  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFiles([...files, ...Array.from(e.target.files)]);
    }
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    
    if (e.dataTransfer.files.length > 0) {
      setFiles([...files, ...Array.from(e.dataTransfer.files)]);
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
  };
  
  const uploadFiles = async () => {
    if (files.length === 0) return;
    
    setUploading(true);
    const results = [];
    
    for (const file of files) {
      try {
        const result = await client.uploadDocument(file);
        results.push({ file: file.name, success: true, result });
      } catch (error) {
        results.push({ file: file.name, success: false, error });
      }
    }
    
    setUploading(false);
    setFiles([]);
    onUpload(results);
  };
  
  return (
    <div>
      <div 
        className="file-upload" 
        onClick={() => fileInputRef.current.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          ref={fileInputRef}
          id="file-input"
          type="file"
          multiple
          onChange={handleFileChange}
        />
        <p>Click to select documents or drag and drop here</p>
      </div>
      
      {files.length > 0 && (
        <div className="file-list">
          <h4>Selected Files:</h4>
          {files.map((file, index) => (
            <div key={index} className="file-item">
              <span>{file.name}</span>
              <button onClick={() => setFiles(files.filter((_, i) => i !== index))}>Remove</button>
            </div>
          ))}
          
          <button 
            className="submit-button" 
            onClick={uploadFiles} 
            disabled={uploading}
          >
            {uploading ? 'Uploading...' : 'Upload Documents'}
          </button>
        </div>
      )}
    </div>
  );
};

// Response Component
const AdvancedResponse = ({ response }) => {
  if (!response) return null;
  
  if (!response.formatted) {
    // Basic response
    return (
      <div className="response-container">
        <div className="answer">{response.answer}</div>
        {response.sources && response.sources.length > 0 && (
          <div className="sources">
            <h4>Sources:</h4>
            <ul>
              {response.sources.map((source, index) => (
                <li key={index}>{source}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }
  
  // Advanced response with reasoning
  return (
    <div className="advanced-response-container">
      {/* Answer section */}
      <div className="answer-section">
        <h3>Answer</h3>
        <div className={`answer confidence-${response.confidence.toLowerCase()}`}>
          {response.answer}
          <div className="confidence-badge">{response.confidence}</div>
        </div>
      </div>
      
      {/* Reasoning section (collapsible) */}
      <details className="reasoning-section">
        <summary>View Reasoning Process</summary>
        <div className="reasoning-content">{response.reasoning}</div>
      </details>
      
      {/* Verification section (if available) */}
      {response.verification && (
        <details className="verification-section">
          <summary>View Verification Details</summary>
          <div className="verification-content">
            <h4>Supported Claims:</h4>
            <ul>
              {(response.verification.supported_claims || []).map((claim, index) => (
                <li key={index} className="supported-claim">{claim}</li>
              ))}
            </ul>
            
            <h4>Unsupported Claims:</h4>
            {response.verification.unsupported_claims && 
             response.verification.unsupported_claims.length > 0 ? (
              <ul>
                {response.verification.unsupported_claims.map((claim, index) => (
                  <li key={index} className="unsupported-claim">{claim}</li>
                ))}
              </ul>
            ) : (
              <p>No unsupported claims found.</p>
            )}
            
            <h4>Explanation:</h4>
            <p>{response.verification.explanation || 'No explanation provided.'}</p>
          </div>
        </details>
      )}
      
      {/* Sources */}
      <div className="sources-section">
        <h4>Sources:</h4>
        <ul>
          {(response.sources || []).map((source, index) => (
            <li key={index}>{source}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

// Message History Component
const MessageHistory = ({ messages, onClearHistory }) => {
  const messagesEndRef = useRef(null);
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  
  if (messages.length === 0) return null;
  
  return (
    <div>
      <div className="message-history">
        {messages.map((msg, index) => (
          <div 
            key={index} 
            className={`message-item ${msg.role === 'user' ? 'user-message' : 'system-message'}`}
          >
            <strong>{msg.role === 'user' ? 'You' : 'Assistant'}:</strong>
            <div>{msg.content}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <button className="clear-history" onClick={onClearHistory}>
        Clear Chat History
      </button>
    </div>
  );
};

// Knowledge Graph Visualization Component
const KnowledgeGraphView = ({ client, entities }) => {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [maxHops, setMaxHops] = useState(2);
  
  const fetchGraphData = async () => {
    if (!entities || entities.length === 0) return;
    
    setLoading(true);
    try {
      const data = await client.getEntityGraph(entities, maxHops);
      setGraphData(data);
    } catch (error) {
      console.error('Error fetching graph data:', error);
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    if (entities && entities.length > 0) {
      fetchGraphData();
    }
  }, [entities]); // eslint-disable-line react-hooks/exhaustive-deps
  
  // This is a placeholder for graph visualization
  // In a real implementation, you would use a library like react-force-graph, 
  // vis-network, or cytoscape.js to render the graph
  const renderGraph = () => {
    if (!graphData) return <p>No graph data available</p>;
    
    return (
      <div className="graph-placeholder">
        <p>Graph visualization would go here with {graphData.nodes.length} nodes and {graphData.edges.length} edges</p>
        <p>Use a library like react-force-graph, vis-network, or cytoscape.js to implement the actual visualization</p>
        
        <div>
          <h4>Nodes:</h4>
          <ul>
            {graphData.nodes.slice(0, 5).map((node, index) => (
              <li key={index}>{node.id} ({node.type})</li>
            ))}
            {graphData.nodes.length > 5 && <li>... and {graphData.nodes.length - 5} more</li>}
          </ul>
          
          <h4>Edges:</h4>
          <ul>
            {graphData.edges.slice(0, 5).map((edge, index) => (
              <li key={index}>{edge.source} → {edge.target} ({edge.relationship})</li>
            ))}
            {graphData.edges.length > 5 && <li>... and {graphData.edges.length - 5} more</li>}
          </ul>
        </div>
      </div>
    );
  };
  
  return (
    <div className="knowledge-graph-view">
      <h3>Knowledge Graph</h3>
      
      <div className="graph-controls">
        <label>
          Max Hops:
          <select 
            value={maxHops} 
            onChange={(e) => setMaxHops(Number(e.target.value))}
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </label>
        
        <button 
          onClick={fetchGraphData} 
          disabled={loading || !entities || entities.length === 0}
        >
          Refresh Graph
        </button>
      </div>
      
      <div className="graph-container">
        {loading ? (
          <div className="loading">
            <div className="loader"></div>
            <span>Loading graph data...</span>
          </div>
        ) : renderGraph()}
      </div>
    </div>
  );
};

// Main Component
const AdvancedRAGComponent = ({ apiBaseUrl = 'http://localhost:8000' }) => {
  const [client, setClient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [response, setResponse] = useState(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [uploadResults, setUploadResults] = useState([]);
  const [entities, setEntities] = useState([]);
  
  // Initialize the client
  useEffect(() => {
    const init = async () => {
      try {
        const newClient = new AdvancedRAGClient(apiBaseUrl);
        await newClient.initialize();
        setClient(newClient);
        setLoading(false);
      } catch (error) {
        console.error('Error initializing client:', error);
        setError('Failed to connect to the backend. Please check if the server is running.');
        setLoading(false);
      }
    };
    
    init();
    
    // Add styles to document
    const styleElement = createStyleElement();
    document.head.appendChild(styleElement);
    
    return () => {
      if (styleElement && document.head.contains(styleElement)) {
        document.head.removeChild(styleElement);
      }
    };
  }, [apiBaseUrl]);
  
  const handleConfigUpdate = (newConfig) => {
    console.log('Configuration updated:', newConfig);
  };
  
  const handleUploadComplete = (results) => {
    setUploadResults(results);
    console.log('Upload results:', results);
  };
  
  const handleQuerySubmit = async () => {
    if (!query.trim()) return;
    
    // Add user message to chat
    const userMessage = { role: 'user', content: query };
    setMessages([...messages, userMessage]);
    
    setQueryLoading(true);
    try {
      // Extract entities for knowledge graph visualization
      const entitiesResponse = await client.getEntities(query);
      setEntities(entitiesResponse.entities || []);
      
      // Query the document
      const queryResponse = await client.queryDocument(query);
      const formattedResponse = formatAdvancedResponse(queryResponse);
      setResponse(formattedResponse);
      
      // Add system response to chat
      const systemMessage = { 
        role: 'system', 
        content: formattedResponse.answer 
      };
      setMessages(prev => [...prev, systemMessage]);
      
      // Clear the query input
      setQuery('');
    } catch (error) {
      console.error('Error submitting query:', error);
      setResponse({
        answer: 'An error occurred while processing your query. Please try again.',
        formatted: false
      });
      
      // Add error message to chat
      const errorMessage = { 
        role: 'system', 
        content: 'An error occurred while processing your query. Please try again.' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setQueryLoading(false);
    }
  };
  
  const handleClearHistory = () => {
    setMessages([]);
    setResponse(null);
    setEntities([]);
  };
  
  const handleKeyDown = (e) => {
    // Submit on Enter (but not with Shift+Enter)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuerySubmit();
    }
  };
  
  if (loading) {
    return (
      <div className="loading">
        <div className="loader"></div>
        <span>Loading Advanced RAG system...</span>
      </div>
    );
  }
  
  if (error) {
    return <div className="error">{error}</div>;
  }
  
  return (
    <div className="advanced-rag-container">
      <h2>Advanced RAG Document Chat</h2>
      
      {/* Settings */}
      <AdvancedRAGSettings client={client} onUpdate={handleConfigUpdate} />
      
      {/* File Upload */}
      <h3>Upload Documents</h3>
      <FileUploader client={client} onUpload={handleUploadComplete} />
      
      {/* Upload Results */}
      {uploadResults.length > 0 && (
        <div className="upload-results">
          <h4>Upload Results:</h4>
          <ul>
            {uploadResults.map((result, index) => (
              <li key={index} className={result.success ? 'success' : 'failure'}>
                {result.file}: {result.success ? 'Success' : 'Failed'}
                {result.success && result.result.chunks && 
                  <span className="upload-status">
                    ({result.result.chunks} chunks processed)
                  </span>
                }
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {/* Message History */}
      <MessageHistory 
        messages={messages} 
        onClearHistory={handleClearHistory} 
      />
      
      {/* Query Input */}
      <div className="query-input">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter your question here... (Press Enter to submit)"
        />
        
        <button 
          className="submit-button" 
          onClick={handleQuerySubmit} 
          disabled={queryLoading || !query.trim()}
        >
          {queryLoading ? 'Thinking...' : 'Submit Question'}
        </button>
      </div>
      
      {/* Response */}
      {queryLoading ? (
        <div className="loading">
          <div className="loader"></div>
          <span>Processing your question...</span>
        </div>
      ) : response && (
        <div>
          <h3>Response Details</h3>
          <AdvancedResponse response={response} />
        </div>
      )}
      
      {/* Knowledge Graph View */}
      {client && client.config.use_kg && entities && entities.length > 0 && (
        <KnowledgeGraphView client={client} entities={entities} />
      )}
    </div>
  );
};

export default AdvancedRAGComponent;