import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { getBackendURL } from '@/api/baseURL'; // adjust based on actual path
import { useNavigate } from 'react-router-dom';

// MCPDocumentChat component with RAG evaluation integration
const MCPDocumentChat = ({ 
  onError = () => {}
}) => {
  const navigate = useNavigate();
  
  // Server state
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
  
  // Evaluation state
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState(null);
  
  // Simplified options with better defaults for performance
  const [advancedOptions, setAdvancedOptions] = useState({
    use_advanced_rag: true,  // Simplified to basic RAG
    use_llama_index: true,   // Added option for LlamaIndex
    model: "mixtral-8x7b-instruct-v0.1.Q4_K_M",  // Smaller model by default
    temperature: 0.3,         // Lower temperature for more factual responses
    context_window: 5,        // Fewer chunks for speed
    quantization: "4bit"      // 4-bit quantization for better performance
  });
  const [showOptions, setShowOptions] = useState(false);
  const [availableModels, setAvailableModels] = useState(["mixtral-8x7b-instruct-v0.1.Q4_K_M"]);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [ragOptions, setRagOptions] = useState([
    { value: "default", label: "Default RAG" },
    { value: "llama_index", label: "LlamaIndex RAG (Advanced)" }
  ]);

  // Check server connection on component mount
  useEffect(() => {
    checkServerConnection();
  }, []);
  
  // Check server connection
  const checkServerConnection = async () => {
    try {
      console.log('Checking server connection...');
      const backendURL = getBackendURL();
      const response = await axios.get(`${backendURL}/`);
      setServerInfo(response.data);
      setConnected(true);
      fetchAvailableModels();
      fetchRagOptions();
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
  
  // Fetch RAG options
  const fetchRagOptions = async () => {
    try {
      const response = await axios.get(`${getBackendURL()}/rag-options`);
      if (response.data && response.data.options) {
        setRagOptions(response.data.options);
      }
    } catch (error) {
      console.error('Error fetching RAG options:', error);
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
  
  // Upload file to server - simplified version
  const handleUpload = async () => {
    if (!file) {
      setUploadError('Please select a file first');
      return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_advanced_rag', advancedOptions.use_advanced_rag);
    formData.append('use_llama_index', advancedOptions.use_llama_index);
    
    setUploading(true);
    setUploadProgress(0);
    
    try {
      // Upload file to server
      const response = await axios.post(`${getBackendURL()}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        },
        timeout: 30000000 // 30 second timeout
      });
      
      // Ensure we have a valid upload result
      if (!response.data || !response.data.filename) {
        throw new Error('Invalid upload response from server');
      }
      
      // Store upload result
      const documentInfo = {
        filename: response.data.filename,
        chunks: response.data.chunks || 0,
        preview: response.data.preview || ''
      };
      
      setUploadResult(documentInfo);
      
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      setFile(null);
      
      // Clear messages
      setMessages([]);
      
      // Set document in backend
      try {
        await axios.post(`${getBackendURL()}/set_document`, { 
          document: documentInfo.filename 
        });
      } catch (err) {
        console.warn('Error setting document:', err);
      }
      try {
        const res = await axios.get(`${getBackendURL()}/suggestions?document=${documentInfo.filename}`);
        setSuggestedQuestions(res.data.questions || []);
      } catch (err) {
        console.warn("Couldn't fetch suggested questions:", err);
      }
      
      
      // Add welcome message
      setMessages([
        {
          id: uuidv4(),
          role: 'system',
          content: `Document "${documentInfo.filename}" uploaded successfully. You can now ask questions about it.`,
          timestamp: new Date().toISOString()
        }
      ]);
      
      // Focus input
      setTimeout(() => {
        const textArea = document.querySelector('textarea');
        if (textArea) {
          textArea.focus();
        }
      }, 300);
      
    } catch (error) {
      console.error('Upload error:', error);
      setUploadError(error.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };
  
  // Handle input change
  const handleInputChange = (e) => {
    setInput(e.target.value);
  };
  
  // Format response for display
  const formatResponseForDisplay = (text, sources) => {
    if (!text) return "No response received.";
    
    // Clean up bullet points for consistency
    let formattedText = text.replace(/^\s*-\s+/gm, "• ");
    
    // Add sources if available
    if (sources && sources.length > 0) {
      const uniqueSources = [...new Set(sources)];
      
      // Don't add sources if it's just the current document
      if (uniqueSources.length === 1 && uniqueSources[0] === uploadResult?.filename) {
        return formattedText;
      }
      
      const sourcesList = uniqueSources.map(source => `• ${source}`).join('\n');
      return `${formattedText}\n\n${sourcesList.length > 0 ? 'Sources:\n' + sourcesList : ''}`;
    }
    
    return formattedText;
  };
  
  // Handle option changes
  const handleOptionChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : 
                   type === 'number' ? parseFloat(value) : value;
    
    setAdvancedOptions(prev => ({
      ...prev,
      [name]: newValue
    }));
  };
  
  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  // React to message changes to scroll
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle RAG evaluation
  const handleEvaluateRAG = async () => {
    if (!uploadResult) {
      setEvaluationError('Please upload a document first');
      return;
    }
    
    setEvaluating(true);
    setEvaluationError(null);
    
    try {
      // Add system message
      setMessages(prev => [
        ...prev,
        {
          id: uuidv4(),
          role: 'system',
          content: `Starting RAG evaluation for "${uploadResult.filename}"...`,
          timestamp: new Date().toISOString()
        }
      ]);
      
      // Instead of checking for existing datasets, directly create one
      try {
        // Create a new evaluation dataset using your /evaluate/basic endpoint
        const response = await axios.post(`${getBackendURL()}/evaluate/basic`, {
          document: uploadResult.filename,
          query: "What is the main topic of this document?" // Sample query for evaluation
        });
        
        if (response.data && response.data.evaluation) {
          // Add success message with results
          setMessages(prev => [
            ...prev,
            {
              id: uuidv4(),
              role: 'system',
              content: `Evaluation complete! Original RAG and LlamaIndex RAG have been compared on a sample query.`,
              timestamp: new Date().toISOString()
            }
          ]);
          
          // You can optionally add specific result details
          const evalResult = response.data.evaluation;
          setMessages(prev => [
            ...prev,
            {
              id: uuidv4(),
              role: 'system',
              content: `Results:\n\nOriginal RAG: "${evalResult.original.response.slice(0, 150)}..."\n\nLlamaIndex RAG: "${evalResult.llama_index.response.slice(0, 150)}..."`,
              timestamp: new Date().toISOString()
            }
          ]);
        }
      } catch (err) {
        console.error('Error running basic evaluation:', err);
        throw new Error('Failed to evaluate RAG systems');
      }
    } catch (err) {
      console.error('Error initiating evaluation:', err);
      setEvaluationError('Failed to start evaluation. Check console for details.');
      
      // Add error message
      setMessages(prev => [
        ...prev,
        {
          id: uuidv4(),
          role: 'system',
          content: `Error running evaluation: ${err.response?.data?.detail || err.message || 'Unknown error'}`,
          isError: true,
          timestamp: new Date().toISOString()
        }
      ]);
    } finally {
      setEvaluating(false);
    }
  };

  // Simplified submit handler
  const handleSubmit = async () => {
    if (!input.trim() || loading || !uploadResult || !connected) {
      return;
    }
    
    // Store current input and clear it
    const currentInput = input.trim();
    setInput('');
    setLoading(true);
    
    // Add user message
    const userMessage = {
      id: uuidv4(),
      role: 'user',
      content: currentInput,
      timestamp: new Date().toISOString()
    };
    
    // Add thinking message
    const thinkingMessage = {
      id: uuidv4(),
      role: 'assistant',
      content: 'Thinking...',
      isThinking: true,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage, thinkingMessage]);
    
    try {
      // Try to set document
      try {
        await axios.post(`${getBackendURL()}/set_document`, { 
          document: uploadResult.filename 
        });
      } catch (err) {
        // Ignore errors
      }
      
      // Send query - simplified options
      const response = await axios.post(`${getBackendURL()}/query-sync`, {
        query: currentInput,
        document: uploadResult.filename,
        model: advancedOptions.model,
        temperature: advancedOptions.temperature,
        context_window: advancedOptions.context_window,
        quantization: advancedOptions.quantization,
        use_advanced_rag: advancedOptions.use_advanced_rag,
        use_llama_index: advancedOptions.use_llama_index
      }, {
        timeout: 30000000 // 30 second timeout
      });
      
      // Remove thinking message
      setMessages(prev => prev.filter(msg => msg.id !== thinkingMessage.id));
      
      if (!response.data) {
        throw new Error('Invalid response from server');
      }
      
      // Format and add assistant response
      const formattedResponse = formatResponseForDisplay(
        response.data.response, 
        response.data.sources
      );
      
      const assistantMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: formattedResponse,
        rawSources: response.data.sources || [],
        system: response.data.system || 'default',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev.filter(msg => !msg.isThinking), assistantMessage]);
      
    } catch (error) {
      console.error('Query error:', error);
      
      // Remove thinking message
      setMessages(prev => prev.filter(msg => msg.id !== thinkingMessage.id));
      
      // Add simple error message
      const errorMessage = {
        id: uuidv4(),
        role: 'system',
        content: "I couldn't process your question. Please try again with a simpler query.",
        isError: true,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev.filter(msg => !msg.isThinking), errorMessage]);
      onError(error.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle enter key in chat input
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
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
        backgroundColor: 'white',
        boxShadow: connected ? '0 0 5px #10b981' : 'none',
        animation: connected ? 'none' : 'pulse 1.5s infinite'
      }}></div>
      {connected ? 'Connected' : 'Disconnected'}
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
            
            {/* Model Settings */}
            <div style={{ marginBottom: '16px' }}>
              <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Model Settings</h3>
              <div style={{
                backgroundColor: '#f8fafc',
                padding: '12px',
                borderRadius: '4px',
                fontSize: '13px'
              }}>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', marginBottom: '4px' }}>Model:</label>
                  <select
                    name="model"
                    value={advancedOptions.model}
                    onChange={handleOptionChange}
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
                    onChange={handleOptionChange}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                    Lower = more factual, Higher = more creative
                  </div>
                </div>

                <div style={{ marginBottom: '12px' }}>
                  <label style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '8px',
                    cursor: 'pointer'
                  }}>
                    <input
                      type="checkbox"
                      name="use_llama_index"
                      checked={advancedOptions.use_llama_index}
                      onChange={handleOptionChange}
                    />
                    Use LlamaIndex RAG
                  </label>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                    Enable advanced LlamaIndex RAG for better retrieval
                  </div>
                </div>

                <div style={{ marginBottom: '12px' }}>
                  <label style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '8px',
                    cursor: 'pointer'
                  }}>
                    <input
                      type="checkbox"
                      name="use_advanced_rag"
                      checked={advancedOptions.use_advanced_rag}
                      onChange={handleOptionChange}
                    />
                    Use Advanced RAG
                  </label>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                    Enable advanced retrieval for better context understanding
                  </div>
                </div>
              </div>
            </div>

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
              borderLeft: '4px solid #10b981',
              marginBottom: '24px'
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
            </div>
          )}

          {/* RAG Evaluation Button */}
          {uploadResult && (
            <div style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              padding: '20px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              borderLeft: '4px solid #8b5cf6',
              marginBottom: '24px'
            }}>
              <h2 style={{ fontSize: '18px', color: '#8b5cf6', marginBottom: '16px' }}>RAG Evaluation</h2>
              
              <div style={{
                backgroundColor: '#f8fafc',
                padding: '16px',
                borderRadius: '4px',
                fontSize: '14px',
                marginBottom: '16px'
              }}>
                <p>Evaluate and compare the performance of different RAG systems on this document.</p>
              </div>
              
              {evaluationError && (
                <div style={{
                  backgroundColor: '#fee2e2',
                  color: '#ef4444',
                  padding: '10px',
                  borderRadius: '4px',
                  marginBottom: '16px',
                  fontSize: '14px'
                }}>
                  {evaluationError}
                </div>
              )}
              
              <button
                onClick={handleEvaluateRAG}
                disabled={evaluating || !connected || !uploadResult}
                style={{
                  backgroundColor: evaluating || !connected || !uploadResult ? '#94a3b8' : '#8b5cf6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '10px 0',
                  width: '100%',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: evaluating || !connected || !uploadResult ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
              >
                {evaluating ? (
                  <>
                    <div style={{ 
                      width: '16px', 
                      height: '16px', 
                      borderRadius: '50%', 
                      border: '2px solid rgba(255,255,255,0.3)',
                      borderTopColor: 'white',
                      animation: 'spin 1s linear infinite' 
                    }}></div>
                    Evaluating...
                  </>
                ) : (
                  <>📊 Evaluate RAG Systems</>
                )}
              </button>
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
                    borderLeft: message.isThinking ? '3px solid #f97316' : 
                               message.role === 'system' ? '3px solid #64748b' : 
                               message.system === 'llama_index' ? '3px solid #8b5cf6' : 'none',
                    border: message.role === 'system' ? '1px solid #e2e8f0' : 'none'
                  }}
                >
                  <div 
                    style={{ 
                      fontSize: '14px', 
                      whiteSpace: 'pre-wrap',
                    }}
                  >
                    {message.content}
                  </div>
                  
                  {/* Display metadata */}
                  <div style={{ 
                    fontSize: '11px', 
                    marginTop: '4px',
                    opacity: 0.7,
                    textAlign: message.role === 'user' ? 'right' : 'left',
                    display: 'flex',
                    justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                    alignItems: 'center',
                    gap: '6px'
                  }}>
                    <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
                    {message.system === 'llama_index' && (
                      <span style={{ 
                        backgroundColor: '#8b5cf6',
                        color: 'white',
                        fontSize: '9px',
                        padding: '2px 4px',
                        borderRadius: '4px'
                      }}>LlamaIndex</span>
                    )}
                  </div>
                </div>
              ))
            )}
            
            {/* Scrolling anchor */}
            <div ref={messagesEndRef} />
          </div>
          
          {/* Suggested questions */}
          {suggestedQuestions.length > 0 && (
            <div style={{ 
              padding: '12px 16px',
              borderTop: '1px solid #e2e8f0',
              backgroundColor: '#f8fafc',
              display: 'flex',
              flexWrap: 'wrap',
              gap: '8px'
            }}>
              {suggestedQuestions.map((question, idx) => (
                <button 
                  key={idx}
                  onClick={() => setInput(question)}
                  style={{
                    fontSize: '12px',
                    padding: '6px 12px',
                    borderRadius: '16px',
                    border: '1px solid #cbd5e1',
                    backgroundColor: '#f1f5f9',
                    cursor: 'pointer',
                    transition: 'background 0.2s ease'
                  }}
                  onMouseEnter={e => e.currentTarget.style.backgroundColor = '#e2e8f0'}
                  onMouseLeave={e => e.currentTarget.style.backgroundColor = '#f1f5f9'}
                >
                  {question}
                </button>
              ))}
            </div>
          )}
          
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
              {loading ? 'Thinking...' : 'Send'}
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
        
        @keyframes spin {
          0% {
            transform: rotate(0deg);
          }
          100% {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
};

export default MCPDocumentChat;