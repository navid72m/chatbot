import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { getBackendURL } from '@/api/baseURL'; // adjust based on actual path

// MCPDocumentChat component
const MCPDocumentChat = ({ 
  onStreamingStateChange = () => {}, 
  onError = () => {},
  initialStreamingEnabled = true
}) => {
  // Server state
  const [connected, setConnected] = useState(false);
  const [serverInfo, setServerInfo] = useState(null);
  const [streamingSupported, setStreamingSupported] = useState(initialStreamingEnabled);
  
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
  const [currentStream, setCurrentStream] = useState(null);
  const messagesEndRef = useRef(null);
  const eventSourceRef = useRef(null);
  
  // Advanced options state with debounce mechanism
  const [advancedOptions, setAdvancedOptions] = useState({
    use_advanced_rag: true,
    use_cot: true,
    use_kg: true,
    verify_answers: true,
    use_multihop: true,
    model: "deepseek-r1",
    temperature: 0.7,
    context_window: 10,
    quantization: "4bit"
  });
  const [showOptions, setShowOptions] = useState(false);
  const [availableModels, setAvailableModels] = useState(["deepseek-r1"]);
  const configChangeTimeoutRef = useRef(null);

  
  
  // Check server connection on component mount
  useEffect(() => {
    checkServerConnection();
    
    // Cleanup function to close EventSource on unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);
  
  // Check server connection
  const checkServerConnection = async () => {
    try {
      console.log('Checking server connection...');
      const backendURL = getBackendURL();
      console.log('Backend URL:', backendURL);
      const response = await axios.get(`${backendURL}/`);
      console.log('Server response:', response.data);
      setServerInfo(response.data);
      setConnected(true);
      
      // Check if streaming is supported
      if (response.data.features) {
        const streamingFeature = response.data.features.find(f => f.id === 'streaming');
        const isStreamingSupported = streamingFeature ? streamingFeature.enabled : initialStreamingEnabled;
        console.log('Streaming supported:', isStreamingSupported);
        setStreamingSupported(isStreamingSupported);
      }
      
      fetchAvailableModels();
    } catch (error) {
      console.error('Server connection error:', error);
      setConnected(false);
    }
  };
  
  // Fetch available models
  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${getBackendURL()}/models`);
      if (response.data && response.data.models) {
        setAvailableModels(response.data.models);
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
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
  
  // Upload file to server
// Upload file to server - fixed version with improved chat transition
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
    console.log('Starting upload to:', `${getBackendURL()}/upload`);
    const response = await axios.post(`${getBackendURL()}/upload`, formData, {
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
    
    // IMPORTANT: Ensure we have a valid upload result
    if (!response.data || !response.data.filename) {
      throw new Error('Invalid upload response from server');
    }
    
    // Store upload result with explicit object structure
    const documentInfo = {
      filename: response.data.filename,
      chunks: response.data.chunks || 0,
      preview: response.data.preview || '',
      features: response.data.features || {}
    };
    
    // Set upload result in state - this is critical for chat functionality
    setUploadResult(documentInfo);
    
    // Reset file input for next upload
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setFile(null);
    
    // Clear any existing messages
    setMessages([]);
    
    // Tell backend which document to use for chat - critical step!
    try {
      await axios.post(`${getBackendURL()}/set_document`, { 
        document: documentInfo.filename 
      });
      console.log('Document set in backend:', documentInfo.filename);
    } catch (setDocErr) {
      console.warn('Error setting document:', setDocErr);
      // Continue anyway - not all backends require this
    }
    
    // Add a welcome/confirmation message to initialize chat
    setMessages([
      {
        id: uuidv4(),
        role: 'system',
        content: `Document "${documentInfo.filename}" uploaded successfully. You can now ask questions about it.`,
        timestamp: new Date().toISOString()
      }
    ]);
    
    // Force-focus the input box to encourage asking a question
    setTimeout(() => {
      const textArea = document.querySelector('textarea');
      if (textArea) {
        textArea.focus();
      }
    }, 300);
    
    // Ensure the chat interface is visible by scrolling to it
    setTimeout(scrollToBottom, 100);
    
  } catch (error) {
    console.error('Upload error:', error);
    setUploadError(error.response?.data?.detail || 'Upload failed. Please try again.');
  } finally {
    setUploading(false);
  }
};
  
  // Handle input change for chat
  const handleInputChange = (e) => {
    setInput(e.target.value);
  };
  
  // Handle advanced option changes with debouncing
  const handleAdvancedOptionChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : 
                   type === 'number' ? parseFloat(value) : value;
    
    setAdvancedOptions(prev => ({
      ...prev,
      [name]: newValue
    }));
    
    // Debounce configuration changes to avoid too many requests
    if (configChangeTimeoutRef.current) {
      clearTimeout(configChangeTimeoutRef.current);
    }
    
    configChangeTimeoutRef.current = setTimeout(() => {
      // Send configuration update to server
      updateConfiguration({ [name]: newValue });
      configChangeTimeoutRef.current = null;
    }, 500);
  };
  
  // Update configuration on server
  const updateConfiguration = async (config) => {
    try {
      await axios.post(`${getBackendURL()}/configure`, { config });
    } catch (error) {
      console.error('Error updating configuration:', error);
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

  // Simple alternative streaming implementation that fallbacks to polling if needed
  const handleStreamFallback = useCallback(async (userMessageId, currentInput) => {
    console.log('Using fallback streaming method');
    
    // Create initial assistant message
    const assistantMessageId = uuidv4();
    const initialAssistantMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      timestamp: new Date().toISOString()
    };
    
    // Add initial assistant message
    setMessages(prev => [...prev.filter(msg => !msg.isThinking), initialAssistantMessage]);
    setCurrentStream(assistantMessageId);
    
    // Notify parent that streaming has started
    onStreamingStateChange(true);
    
    try {
      console.log('Sending query to backend with document:', uploadResult.filename);
      
      // Start with POST request to initialize the response
      const response = await axios.post(`${getBackendURL()}/query-sync`, {
        query: currentInput,
        document: uploadResult.filename,
        ...advancedOptions
      });
      
      console.log('Got response from backend:', response.data);
      
      // Use polling to simulate streaming
      let fullResponse = '';
      const wordCount = currentInput.split(' ').length;
      const estimatedTokenCount = Math.max(20, wordCount * 3); // Rough estimate
      const tokenDelay = 50; // ms between tokens
      
      if (response.data && response.data.response) {
        // Get the full response
        fullResponse = response.data.response;
        
        // Simulate streaming by revealing one token at a time
        let accumulatedText = '';
        const chunks = fullResponse.split(' ');
        
        for (let i = 0; i < chunks.length; i++) {
          accumulatedText += (i > 0 ? ' ' : '') + chunks[i];
          
          // Update message with new token
          setMessages(prevMessages => {
            return prevMessages.map(msg => {
              if (msg.id === assistantMessageId) {
                return { ...msg, content: accumulatedText };
              }
              return msg;
            });
          });
          
          // Wait for next token
          if (i < chunks.length - 1) {
            await new Promise(resolve => setTimeout(resolve, tokenDelay));
          }
        }
      }
      
      // Final update with complete response
      setMessages(prevMessages => {
        return prevMessages.map(msg => {
          if (msg.id === assistantMessageId) {
            return { 
              ...msg, 
              content: fullResponse || 'Sorry, I couldn\'t generate a response.',
              isStreaming: false,
              sources: response.data.sources || []
            };
          }
          return msg;
        });
      });
      
      setCurrentStream(null);
      setLoading(false);
      onStreamingStateChange(false);
      
    } catch (error) {
      console.error('Error in fallback streaming:', error);
      
      // Add error message
      setMessages(prev => {
        // Remove the assistant message
        const filteredMessages = prev.filter(msg => msg.id !== assistantMessageId);
        
        // Add error message
        const errorMessage = {
          id: uuidv4(),
          role: 'system',
          content: 'Failed to process your request. Please try again.',
          isError: true,
          timestamp: new Date().toISOString()
        };
        
        return [...filteredMessages, errorMessage];
      });
      
      setLoading(false);
      setCurrentStream(null);
      onStreamingStateChange(false);
      onError('Fallback streaming error: ' + error.message);
    }
  }, [advancedOptions, getBackendURL, onError, onStreamingStateChange, uploadResult]);

  // Submit query to server
// Submit query to server - fixed version with better error handling
const handleSubmit = useCallback(async () => {
  // Add detailed logging to help troubleshoot issues
  console.log('Submit attempt with state:', {
    inputEmpty: !input.trim(),
    loading,
    uploadResult: uploadResult ? 'Yes ('+uploadResult.filename+')' : 'No',
    connected,
    messagesCount: messages.length
  });

  if (!input.trim() || loading || !uploadResult || !connected) {
    console.log('Submit blocked due to:', {
      emptyInput: !input.trim(),
      loading,
      noUploadResult: !uploadResult,
      notConnected: !connected
    });
    return;
  }
  
  try {
    // Store current input and clear it
    const currentInput = input.trim();
    console.log('Submitting query:', currentInput);
    console.log('For document:', uploadResult.filename);
    
    // Clear input immediately for better UX
    setInput('');
    
    // Cancel any existing stream
    if (eventSourceRef.current) {
      console.log('Closing existing EventSource');
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setCurrentStream(null);
    }
    
    // Add user message to the conversation
    const userMessageId = uuidv4();
    const userMessage = {
      id: userMessageId,
      role: 'user',
      content: currentInput,
      timestamp: new Date().toISOString()
    };
    
    // Update messages state with the user message
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    
    // Create a temporary thinking message
    const thinkingMessageId = uuidv4();
    const thinkingMessage = {
      id: thinkingMessageId,
      role: 'assistant',
      content: 'Thinking...',
      isThinking: true,
      isStreaming: true,
      timestamp: new Date().toISOString()
    };
    
    // Add thinking message
    setMessages(prev => [...prev, thinkingMessage]);
    
    try {
      // Make sure document is set on backend
      await axios.post(`${getBackendURL()}/set_document`, { 
        document: uploadResult.filename 
      }).catch(e => console.warn('Set document warning:', e));
      
      // Use sync endpoint for better reliability
      console.log('Sending query to backend...');
      const response = await axios.post(`${getBackendURL()}/query-sync`, {
        query: currentInput,
        document: uploadResult.filename,
        ...advancedOptions
      });
      
      console.log('Response received:', response.data);
      
      // Remove thinking message
      setMessages(prev => prev.filter(msg => msg.id !== thinkingMessageId));
      
      if (!response.data || !response.data.response) {
        throw new Error('Invalid response from server');
      }
      
      // Add assistant response message
      const assistantMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: response.data.response,
        sources: response.data.sources || [],
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev.filter(msg => !msg.isThinking), assistantMessage]);
      scrollToBottom();
      
    } catch (error) {
      console.error('Query error:', error);
      
      // Remove thinking message
      setMessages(prev => prev.filter(msg => msg.id !== thinkingMessageId));
      
      // Add error message
      const errorMessage = {
        id: uuidv4(),
        role: 'system',
        content: `Error: ${error.message || 'Failed to get response from server'}. Please try again.`,
        isError: true,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev.filter(msg => !msg.isThinking), errorMessage]);
    } finally {
      setLoading(false);
    }
  } catch (error) {
    console.error('Unexpected error during submit:', error);
    setLoading(false);
  }
}, [input, loading, uploadResult, connected, advancedOptions, getBackendURL, scrollToBottom]);
  
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
        backgroundColor: 'white',
        boxShadow: connected ? '0 0 5px #10b981' : 'none',
        animation: connected ? 'none' : 'pulse 1.5s infinite'
      }}></div>
      {connected ? 'Connected to Server' : 'Disconnected'}
    </div>
  );
  
  // Streaming indicator component
  const StreamingIndicator = () => (
    <div className="typing-indicator">
      <span></span><span></span><span></span>
    </div>
  );
  
  // Debugging component - toggle with key press
  const [showDebug, setShowDebug] = useState(false);
  useEffect(() => {
    const toggleDebug = (e) => {
      if (e.ctrlKey && e.key === 'd') {
        setShowDebug(prev => !prev);
      }
    };
    window.addEventListener('keydown', toggleDebug);
    return () => window.removeEventListener('keydown', toggleDebug);
  }, []);
  
  return (
    <div style={{ padding: '24px', maxWidth: '1000px', margin: '0 auto' }}>
      <h1 style={{ marginBottom: '24px' }}>MCP Document Chat</h1>
      <ConnectionStatus />
      
      {/* Debug info panel - hidden by default, toggle with Ctrl+D */}
      {showDebug && (
        <div style={{
          position: 'fixed',
          bottom: 10,
          right: 10,
          width: 300,
          maxHeight: 400,
          overflowY: 'auto',
          backgroundColor: 'rgba(0,0,0,0.8)',
          color: 'white',
          padding: 10,
          fontFamily: 'monospace',
          fontSize: 12,
          zIndex: 1000,
          borderRadius: 4
        }}>
          <div>Connected: {connected ? 'Yes' : 'No'}</div>
          <div>Upload Result: {uploadResult ? 'Yes' : 'No'}</div>
          <div>Doc: {uploadResult?.filename || 'None'}</div>
          <div>Messages: {messages.length}</div>
          <div>Loading: {loading ? 'Yes' : 'No'}</div>
          <div>Streaming: {currentStream ? 'Yes' : 'No'}</div>
          <button onClick={() => console.log({messages, uploadResult})}>Log State</button>
        </div>
      )}
      
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
                  
                  <div style={{ marginBottom: '12px' }}>
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
                ? `Ask questions about: ${uploadResult.filename}` 
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
              // Map through and render messages
              messages.map(message => (
                <div 
                  key={message.id} 
                  style={{
                    maxWidth: message.role === 'user' ? '80%' : '90%',
                    padding: '12px 16px',
                    borderRadius: '8px',
                    alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
                    backgroundColor: message.role === 'user' ? '#4f46e5' : 
                                    message.role === 'system' ? (message.isThinking ? '#f97316' : '#f8fafc') : 
                                    '#ffffff',
                    color: message.role === 'user' ? 'white' : 
                          message.role === 'system' ? (message.isError ? '#ef4444' : '#1e293b') : '#1e293b',
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                    borderLeft: message.isStreaming ? '3px solid #3b82f6' : 
                               message.role === 'system' ? '3px solid #64748b' : 'none',
                    transition: 'border-left-color 0.3s ease',
                    border: message.role === 'system' ? '1px solid #e2e8f0' : 'none'
                  }}
                >
                  <div 
                    style={{ 
                      fontSize: '14px', 
                      whiteSpace: 'pre-wrap',
                    }}
                    className={message.isStreaming ? 'streaming-message' : ''}
                  >
                    {message.content}
                    {message.isStreaming && (
                      <StreamingIndicator />
                    )}
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
                      <span>Retrieval: {typeof message.retrieval_time === 'number' ? message.retrieval_time.toFixed(2) : message.retrieval_time}s</span>
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
        @keyframes pulse {
          0% {
            opacity: 0.6;
            transform: scale(0.8);
          }
          50% {
            opacity: 1;
            transform: scale(1);
          }
          100% {
            opacity: 0.6;
            transform: scale(0.8);
          }
        }
        
        .typing-indicator {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          margin-left: 8px;
          height: 16px;
          vertical-align: middle;
        }
        
        .typing-indicator span {
          width: 6px;
          height: 6px;
          background-color: #3b82f6;
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
        
        /* Add a subtle blinking cursor animation for streaming messages */
        @keyframes blink-cursor {
          0%, 100% { border-left-color: transparent; }
          50% { border-left-color: #3b82f6; }
        }
        
        .streaming-message {
          animation: blink-cursor 1s step-end infinite;
        }
      `}</style>
    </div>
  );
};

export default MCPDocumentChat;