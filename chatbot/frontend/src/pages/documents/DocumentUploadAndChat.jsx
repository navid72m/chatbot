import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid'; // Make sure to add this dependency

const DocumentUploadAndChat = () => {
  // Client ID & WebSocket state
  const [clientId] = useState(() => uuidv4());
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [serverInfo, setServerInfo] = useState(null);
  
  // Upload state
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);
  
  // Chat state
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Advanced options state
  const [advancedOptions, setAdvancedOptions] = useState({
    use_advanced_rag: true,
    use_cot: true,
    use_kg: true,
    verify_answers: true,
    use_multihop: true,
    model: "mistral",
    temperature: 0.7,
    context_window: 10,
    quantization: "4bit"
  });
  const [showOptions, setShowOptions] = useState(false);
  const [availableModels, setAvailableModels] = useState(["mistral"]);
  
  // Connect to WebSocket on component mount
  useEffect(() => {
    connectWebSocket();
    
    // Cleanup on unmount
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, []);
  
  // Connect to WebSocket
  const connectWebSocket = () => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setSocket(ws);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      
      // Try to reconnect after a delay
      setTimeout(() => {
        if (!connected) {
          connectWebSocket();
        }
      }, 3000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
  };
  
  // Handle WebSocket messages
  const handleWebSocketMessage = (data) => {
    const messageType = data.type;
    
    switch (messageType) {
      case 'server_info':
        setServerInfo(data);
        fetchAvailableModels();
        break;
        
      case 'query_response':
        // Remove any thinking messages
        setMessages(prev => prev.filter(msg => !msg.isThinking));
        
        // Add assistant response to the conversation
        const assistantMessage = {
          id: data.message_id,
          role: 'assistant',
          content: data.response,
          sources: data.sources || [],
          reasoning: data.reasoning || null,
          confidence: data.confidence || null,
          retrieval_time: data.retrieval_time || null,
          verification: data.verification || null,
          document: data.document,
          timestamp: data.timestamp || new Date().toISOString()
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        setLoading(false);
        break;
        
      case 'set_document_response':
        console.log('Document set:', data.document);
        break;
        
      case 'list_models_response':
        setAvailableModels(data.models || []);
        break;
        
      case 'error':
        console.error('Server error:', data.error);
        
        // Remove any thinking messages
        setMessages(prev => prev.filter(msg => !msg.isThinking));
        
        // Add error message
        const errorMessage = {
          id: data.message_id,
          role: 'system',
          content: data.error || 'An error occurred while processing your request.',
          isError: true,
          timestamp: data.timestamp || new Date().toISOString()
        };
        
        setMessages(prev => [...prev, errorMessage]);
        setLoading(false);
        break;
        
      default:
        console.log('Unhandled message type:', messageType, data);
    }
  };
  
  // Fetch available models
  const fetchAvailableModels = () => {
    if (socket && connected) {
      const message = {
        type: 'list_models',
        message_id: uuidv4(),
        client_id: clientId,
        timestamp: new Date().toISOString()
      };
      
      socket.send(JSON.stringify(message));
    }
  };
  
  // Send WebSocket message
  const sendMessage = (type, data = {}) => {
    if (!socket || !connected) {
      console.error('WebSocket not connected');
      return null;
    }
    
    const message = {
      type,
      message_id: uuidv4(),
      client_id: clientId,
      timestamp: new Date().toISOString(),
      ...data
    };
    
    socket.send(JSON.stringify(message));
    return message.message_id;
  };
  
  // Handle file selection
  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      setUploadError(null);
    }
  };
  
  // Handle file drop
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setUploadError(null);
    }
  };
  
  // Prevent default drag behavior
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  // Upload file to server (still using REST API)
  const handleUpload = async () => {
    if (!file) {
      setUploadError('Please select a file first');
      return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    setUploading(true);
    setUploadProgress(0);
    
    try {
      // Upload file to server using REST API
      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        }
      });
      
      console.log('Upload response:', response.data);
      setUploadResult(response.data);
      
      // Set the document as current via WebSocket
      if (socket && connected && response.data.filename) {
        sendMessage('set_document', { document: response.data.filename });
      }
      
      // Reset file input and messages for new document
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      setFile(null);
      setMessages([]);
      
    } catch (error) {
      console.error('Upload error:', error);
      setUploadError(error.response?.data?.message || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };
  
  // Handle input change for chat
  const handleInputChange = (e) => {
    setInput(e.target.value);
  };
  
  // Handle advanced option changes
  const handleAdvancedOptionChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : 
                    type === 'number' ? parseFloat(value) : value;
    
    setAdvancedOptions(prev => ({
      ...prev,
      [name]: newValue
    }));
    
    // Send configuration update to server
    if (socket && connected) {
      sendMessage('configure', { 
        config: { 
          [name]: newValue 
        } 
      });
    }
  };
  
  // Handle enter key in chat input
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };
  
  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  // React to message changes to scroll
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // Submit query to server via WebSocket
  const handleSubmit = () => {
    if (!input.trim() || loading || !uploadResult || !connected) return;
    
    // Add user message to the conversation
    const userMessage = {
      id: uuidv4(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    
    // Add a "thinking" message
    const thinkingMessage = {
      id: uuidv4() + '-thinking',
      role: 'system',
      content: 'Processing your query... (this may take a while)',
      isThinking: true,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, thinkingMessage]);
    
    // Send query via WebSocket
    sendMessage('query', {
      query: userMessage.content,
      ...advancedOptions
    });
  };
  
  // Connection status indicator
  const ConnectionStatus = () => (
    <div style={{ 
      position: 'fixed', 
      top: '10px', 
      right: '10px',
      padding: '8px 12px',
      borderRadius: '4px',
      backgroundColor: connected ? '#10b981' : '#ef4444',
      color: 'white',
      fontSize: '12px',
      fontWeight: '500',
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)'
    }}>
      <div style={{ 
        width: '8px', 
        height: '8px', 
        borderRadius: '50%', 
        backgroundColor: 'white'
      }}></div>
      {connected ? 'Connected to MCP' : 'Disconnected'}
    </div>
  );
  
  return (
    <div style={{ padding: '24px', maxWidth: '1000px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '24px' }}>Document Chat</h1>
      <ConnectionStatus />
      
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '24px' }}>
        {/* Left sidebar */}
        <div>
          {/* Document upload card */}
          <div style={{ 
            backgroundColor: 'white',
            borderRadius: '8px',
            padding: '20px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            marginBottom: '24px'
          }}>
            <h2 style={{ fontSize: '18px', marginBottom: '16px' }}>Upload Document</h2>
            
            <div 
              style={{ 
                border: '2px dashed #cbd5e1',
                borderRadius: '8px',
                padding: '24px',
                textAlign: 'center',
                backgroundColor: '#f8fafc',
                marginBottom: '16px',
                cursor: 'pointer'
              }}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
                accept=".pdf,.doc,.docx,.txt,.csv,.xlsx"
              />
              <div style={{ fontSize: '36px', marginBottom: '12px' }}>📄</div>
              <p style={{ marginBottom: '8px' }}>
                {file ? file.name : 'Drag & drop or click to browse'}
              </p>
              {file && (
                <p style={{ fontSize: '12px', color: '#64748b' }}>
                  {(file.size / 1024).toFixed(2)} KB
                </p>
              )}
            </div>
            
            {/* Progress bar */}
            {uploading && (
              <div style={{ marginBottom: '16px' }}>
                <div style={{ 
                  height: '6px',
                  backgroundColor: '#e2e8f0',
                  borderRadius: '3px',
                  overflow: 'hidden',
                  marginBottom: '8px'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${uploadProgress}%`,
                    backgroundColor: '#4f46e5',
                    transition: 'width 0.3s'
                  }}></div>
                </div>
                <p style={{ fontSize: '12px', color: '#64748b', textAlign: 'center' }}>
                  Uploading: {uploadProgress}%
                </p>
              </div>
            )}
            
            {/* Error message */}
            {uploadError && (
              <div style={{
                backgroundColor: '#fee2e2',
                color: '#ef4444',
                padding: '10px',
                borderRadius: '4px',
                marginBottom: '16px',
                fontSize: '14px'
              }}>
                {uploadError}
              </div>
            )}
            
            {/* Upload button */}
            <button
              onClick={handleUpload}
              disabled={!file || uploading || !connected}
              style={{
                backgroundColor: !file || uploading || !connected ? '#94a3b8' : '#4f46e5',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '10px 0',
                width: '100%',
                fontSize: '14px',
                fontWeight: '500',
                cursor: !file || uploading || !connected ? 'not-allowed' : 'pointer',
              }}
            >
              {uploading ? 'Uploading...' : 'Upload Document'}
            </button>
          </div>
          
          {/* Document info card */}
          {uploadResult && (
            <div style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              padding: '20px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              borderLeft: '4px solid #10b981'
            }}>
              <h2 style={{ fontSize: '18px', color: '#10b981', marginBottom: '16px' }}>Document Details</h2>
              
              <div style={{
                backgroundColor: '#f8fafc',
                padding: '16px',
                borderRadius: '4px',
                fontSize: '14px'
              }}>
                <p style={{ marginBottom: '8px' }}><strong>Filename:</strong> {uploadResult.filename || 'Document'}</p>
                <p style={{ marginBottom: '8px' }}><strong>Chunks:</strong> {uploadResult.chunks || 'Unknown'}</p>
                {uploadResult.preview && (
                  <details style={{ marginTop: '12px' }}>
                    <summary style={{ cursor: 'pointer', color: '#4f46e5' }}>Document Preview</summary>
                    <div style={{ 
                      marginTop: '8px',
                      padding: '8px',
                      backgroundColor: 'rgba(0, 0, 0, 0.05)',
                      borderRadius: '4px',
                      fontSize: '12px',
                      maxHeight: '200px',
                      overflowY: 'auto'
                    }}>
                      {uploadResult.preview}
                    </div>
                  </details>
                )}
              </div>
              
              {/* Server features */}
              {serverInfo && (
                <div style={{ marginTop: '16px' }}>
                  <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Server Features</h3>
                  <div style={{
                    backgroundColor: '#f8fafc',
                    padding: '12px',
                    borderRadius: '4px',
                    fontSize: '13px'
                  }}>
                    {serverInfo.features.map((feature, idx) => (
                      <div key={feature.id} style={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        padding: '4px 0',
                        borderBottom: idx < serverInfo.features.length - 1 ? '1px solid #e2e8f0' : 'none'
                      }}>
                        <div style={{
                          width: '10px',
                          height: '10px',
                          borderRadius: '50%',
                          backgroundColor: feature.enabled ? '#10b981' : '#94a3b8',
                          marginRight: '8px'
                        }}></div>
                        <span>{feature.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Advanced options toggle */}
              <button
                onClick={() => setShowOptions(!showOptions)}
                style={{
                  backgroundColor: showOptions ? '#4f46e5' : 'transparent',
                  color: showOptions ? 'white' : '#64748b',
                  border: '1px solid ' + (showOptions ? '#4f46e5' : '#cbd5e1'),
                  borderRadius: '4px',
                  padding: '8px 0',
                  marginTop: '16px',
                  width: '100%',
                  fontSize: '14px',
                  cursor: 'pointer'
                }}
              >
                {showOptions ? 'Hide Advanced Options' : 'Show Advanced Options'}
              </button>
              
              {/* Advanced options */}
              {showOptions && (
                <div style={{
                  marginTop: '16px',
                  backgroundColor: '#f8fafc',
                  padding: '16px',
                  borderRadius: '4px',
                  fontSize: '14px'
                }}>
                  <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Query Options</h3>
                  
                  <div style={{ marginBottom: '16px' }}>
                    <label style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                      <input
                        type="checkbox"
                        name="use_advanced_rag"
                        checked={advancedOptions.use_advanced_rag}
                        onChange={handleAdvancedOptionChange}
                        style={{ marginRight: '8px' }}
                      />
                      Use Advanced RAG
                    </label>
                    
                    <label style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                      <input
                        type="checkbox"
                        name="use_cot"
                        checked={advancedOptions.use_cot}
                        onChange={handleAdvancedOptionChange}
                        style={{ marginRight: '8px' }}
                      />
                      Chain of Thought
                    </label>
                    
                    <label style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                      <input
                        type="checkbox"
                        name="use_kg"
                        checked={advancedOptions.use_kg}
                        onChange={handleAdvancedOptionChange}
                        style={{ marginRight: '8px' }}
                      />
                      Knowledge Graph
                    </label>
                    
                    <label style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                      <input
                        type="checkbox"
                        name="verify_answers"
                        checked={advancedOptions.verify_answers}
                        onChange={handleAdvancedOptionChange}
                        style={{ marginRight: '8px' }}
                      />
                      Verify Answers
                    </label>
                    
                    <label style={{ display: 'flex', alignItems: 'center' }}>
                      <input
                        type="checkbox"
                        name="use_multihop"
                        checked={advancedOptions.use_multihop}
                        onChange={handleAdvancedOptionChange}
                        style={{ marginRight: '8px' }}
                      />
                      Multi-hop Reasoning
                    </label>
                  </div>
                  
                  <div style={{ marginBottom: '12px' }}>
                    <label style={{ display: 'block', marginBottom: '4px' }}>Model:</label>
                    <select
                      name="model"
                      value={advancedOptions.model}
                      onChange={handleAdvancedOptionChange}
                      style={{
                        width: '100%',
                        padding: '6px',
                        borderRadius: '4px',
                        border: '1px solid #cbd5e1'
                      }}
                    >
                      {availableModels.map(model => (
                        <option key={model} value={model}>{model}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div style={{ marginBottom: '12px' }}>
                    <label style={{ display: 'block', marginBottom: '4px' }}>
                      Temperature: {advancedOptions.temperature}
                    </label>
                    <input
                      type="range"
                      name="temperature"
                      min="0"
                      max="1"
                      step="0.1"
                      value={advancedOptions.temperature}
                      onChange={handleAdvancedOptionChange}
                      style={{ width: '100%' }}
                    />
                  </div>
                  
                  <div>
                    <label style={{ display: 'block', marginBottom: '4px' }}>Context Window:</label>
                    <select
                      name="context_window"
                      value={advancedOptions.context_window}
                      onChange={handleAdvancedOptionChange}
                      style={{
                        width: '100%',
                        padding: '6px',
                        borderRadius: '4px',
                        border: '1px solid #cbd5e1'
                      }}
                    >
                      <option value="5">5 chunks</option>
                      <option value="10">10 chunks</option>
                      <option value="15">15 chunks</option>
                      <option value="20">20 chunks</option>
                    </select>
                  </div>
                  
                  <div style={{ marginTop: '12px' }}>
                    <label style={{ display: 'block', marginBottom: '4px' }}>Quantization:</label>
                    <select
                      name="quantization"
                      value={advancedOptions.quantization}
                      onChange={handleAdvancedOptionChange}
                      style={{
                        width: '100%',
                        padding: '6px',
                        borderRadius: '4px',
                        border: '1px solid #cbd5e1'
                      }}
                    >
                      <option value="None">None (Full Precision)</option>
                      <option value="8bit">8-bit Quantization</option>
                      <option value="4bit">4-bit Quantization</option>
                      <option value="1bit">1-bit Quantization</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Chat interface */}
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          height: '600px',
          backgroundColor: 'white',
          borderRadius: '8px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          overflow: 'hidden'
        }}>
          {/* Chat header */}
          <div style={{ 
            padding: '16px', 
            borderBottom: '1px solid #e2e8f0',
            backgroundColor: '#f8fafc'
          }}>
            <h2 style={{ margin: 0, fontSize: '18px' }}>Document Chat</h2>
            <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: '#64748b' }}>
              {uploadResult 
                ? 'Ask questions about your uploaded document' 
                : 'Upload a document to start chatting'}
            </p>
          </div>
          
          {/* Messages container */}
          <div style={{ 
            flex: 1, 
            overflowY: 'auto',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px',
            backgroundColor: '#f8fafc'
          }}>
            {!connected ? (
              <div style={{ 
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: '#94a3b8',
                textAlign: 'center',
                padding: '0 32px'
              }}>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔄</div>
                <h3 style={{ marginBottom: '8px', color: '#64748b' }}>Connecting to Server...</h3>
                <p>Please wait while we establish connection to the server.</p>
              </div>
            ) : !uploadResult ? (
              <div style={{ 
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: '#94a3b8',
                textAlign: 'center',
                padding: '0 32px'
              }}>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>📄</div>
                <h3 style={{ marginBottom: '8px', color: '#64748b' }}>No Document Uploaded</h3>
                <p>Upload a document using the panel on the left to start asking questions about it.</p>
              </div>
            ) : messages.length === 0 ? (
              <div style={{ 
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                color: '#94a3b8',
                textAlign: 'center',
                padding: '0 32px'
              }}>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>💬</div>
                <h3 style={{ marginBottom: '8px', color: '#64748b' }}>Start the Conversation</h3>
                <p>Your document is ready. Type a question below to start chatting about its contents.</p>
              </div>
            ) : (
              messages.map(message => (
                <div 
                  key={message.id} 
                  style={{
                    maxWidth: message.role === 'user' ? '80%' : '90%',
                    padding: '12px 16px',
                    borderRadius: '8px',
                    alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
                    backgroundColor: message.role === 'user' ? '#4f46e5' : 
                                    message.role === 'system' ? (message.isThinking ? '#f97316' : '#f87171') : 
                                    '#ffffff',
                    color: message.role === 'user' ? 'white' : 
                          message.role === 'system' ? 'white' : '#1e293b',
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)'
                  }}
                >
                  <div style={{ fontSize: '14px', whiteSpace: 'pre-wrap' }}>
                    {message.content}
                  </div>
                  
                  {/* Display sources if available */}
                  {message.sources && message.sources.length > 0 && (
                    <div style={{ 
                      marginTop: '8px',
                      fontSize: '12px',
                      padding: '8px',
                      backgroundColor: 'rgba(0, 0, 0, 0.05)',
                      borderRadius: '4px'
                    }}>
                      <strong>Sources:</strong>
                      <ul style={{ 
                        margin: '4px 0 0 0', 
                        paddingLeft: '16px',
                        color: '#475569'
                      }}>
                        {message.sources.map((source, idx) => (
                          <li key={idx}>{source}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {/* Display reasoning if available */}
                  {message.reasoning && (
                    <details style={{ marginTop: '8px', fontSize: '12px' }}>
                      <summary style={{ cursor: 'pointer', color: '#4f46e5' }}>View reasoning</summary>
                      <div style={{ 
                        marginTop: '8px',
                        padding: '8px',
                        backgroundColor: 'rgba(0, 0, 0, 0.05)',
                        borderRadius: '4px',
                        whiteSpace: 'pre-wrap'
                      }}>
                        {message.reasoning}
                      </div>
                    </details>
                  )}
                  
                  {/* Display verification if available */}
                  {message.verification && (
                    <details style={{ marginTop: '8px', fontSize: '12px' }}>
                      <summary style={{ cursor: 'pointer', color: '#4f46e5' }}>View verification</summary>
                      <div style={{ 
                        marginTop: '8px',
                        padding: '8px',
                        backgroundColor: 'rgba(0, 0, 0, 0.05)',
                        borderRadius: '4px'
                      }}>
                        <p><strong>Verified:</strong> {message.verification.is_verified ? 'Yes' : 'No'}</p>
                        {message.verification.unsupported_claims && message.verification.unsupported_claims.length > 0 && (
                          <>
                            <p><strong>Unsupported Claims:</strong></p>
                            <ul style={{ paddingLeft: '16px' }}>
                              {message.verification.unsupported_claims.map((claim, idx) => (
                                <li key={idx}>{claim}</li>
                              ))}
                            </ul>
                          </>
                        )}
                      </div>
                    </details>
                  )}
                  
                  {/* Display metadata */}
                  <div style={{ 
                    fontSize: '11px', 
                    marginTop: '4px',
                    opacity: 0.7,
                    textAlign: message.role === 'user' ? 'right' : 'left',
                    display: 'flex',
                    justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    {message.retrieval_time && (
                      <span>Retrieval: {message.retrieval_time.toFixed(2)}s</span>
                    )}
                    {message.confidence && (
                      <span>Confidence: {Math.round(message.confidence * 100)}%</span>
                    )}
                    <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              ))
            )}
            
            {/* Scrolling anchor */}
            <div ref={messagesEndRef} />
          </div>
          
          {/* Input area */}
          <div style={{ 
            padding: '16px',
            borderTop: '1px solid #e2e8f0',
            backgroundColor: '#ffffff',
            display: 'flex',
            gap: '8px'
          }}>
            <textarea
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder={connected && uploadResult ? "Type your question..." : connected ? "Upload a document to start chatting" : "Connecting to server..."}
              disabled={!connected || !uploadResult || loading}
              style={{
                flex: 1,
                border: '1px solid #cbd5e1',
                borderRadius: '8px',
                padding: '12px',
                resize: 'none',
                fontSize: '14px',
                minHeight: '24px',
                maxHeight: '120px',
                fontFamily: 'inherit',
                backgroundColor: connected && uploadResult ? '#ffffff' : '#f1f5f9'
              }}
              rows={1}
            />
            <button
              onClick={handleSubmit}
              disabled={!connected || !uploadResult || !input.trim() || loading}
              style={{
                backgroundColor: !connected || !uploadResult || !input.trim() || loading ? '#94a3b8' : '#4f46e5',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                padding: '0 16px',
                fontSize: '14px',
                cursor: !connected || !uploadResult || !input.trim() || loading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>
      
      {/* CSS for animations */}
      <style jsx="true">{`
        .typing-indicator {
          display: flex;
          align-items: center;
          gap: 4px;
        }
        
        .typing-indicator span {
          width: 8px;
          height: 8px;
          background-color: #ffffff;
          border-radius: 50%;
          display: inline-block;
          animation: bounce 1.4s infinite ease-in-out both;
        }
        
        .typing-indicator span:nth-child(1) {
          animation-delay: -0.32s;
        }
        
        .typing-indicator span:nth-child(2) {
          animation-delay: -0.16s;
        }
        
        @keyframes bounce {
          0%, 80%, 100% { 
            transform: scale(0);
          } 40% { 
            transform: scale(1.0);
          }
        }
      `}</style>
    </div>
  );
};

export default DocumentUploadAndChat;