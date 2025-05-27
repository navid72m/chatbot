import React, { useState, useRef, useEffect, useCallback } from 'react';
import '../../styles/pages/MCPDocumentChat.css';
import './EnhancedDocumentChat.css';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { getBackendURL } from '@/api/baseURL';
import { useNavigate } from 'react-router-dom';
import PDFContextViewer from './PDFContextViewer';
import contextTrackingService from './ContextTrackingService';

// Enhanced Document Chat component with PDF context viewer
const EnhancedDocumentChat = ({ 
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
  const [showSuggestions, setShowSuggestions] = useState(true);
  
  // Evaluation state
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState(null);
  
  // PDF Viewer state
  const [showPdfViewer, setShowPdfViewer] = useState(true);
  const [selectedMessage, setSelectedMessage] = useState(null);
  
  // Enhanced options state with query rewriting
  const [advancedOptions, setAdvancedOptions] = useState({
    use_advanced_rag: true,
    use_llama_index: true,
    model: "mixtral-8x7b-instruct-v0.1.Q4_K_M",
    temperature: 0.3,
    context_window: 5,
    quantization: "4bit",
    // New query rewriting options
    use_prf: true,
    use_variants: true,
    prf_iterations: 1,
    fusion_method: "rrf",
    rerank: true
  });
  
  const [showOptions, setShowOptions] = useState(false);
  const [showQueryRewritingOptions, setShowQueryRewritingOptions] = useState(false);
  const [availableModels, setAvailableModels] = useState(["mixtral-8x7b-instruct-v0.1.Q4_K_M"]);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [ragOptions, setRagOptions] = useState([
    { value: "default", label: "Default RAG" },
    { value: "enhanced", label: "Enhanced RAG with Query Rewriting" },
    { value: "llama_index", label: "LlamaIndex RAG (Legacy)" }
  ]);
  
  // Query rewriting configuration state
  const [queryRewritingOptions, setQueryRewritingOptions] = useState({
    techniques: [],
    fusion_methods: [],
    parameters: {}
  });
  
  // Query rewriting stats
  const [queryStats, setQueryStats] = useState(null);
  const [showStats, setShowStats] = useState(false);

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
      fetchQueryRewritingOptions();
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
  
  // Fetch query rewriting options
  const fetchQueryRewritingOptions = async () => {
    try {
      const response = await axios.get(`${getBackendURL()}/query-rewriting-options`);
      if (response.data) {
        setQueryRewritingOptions(response.data);
      }
    } catch (error) {
      console.error('Error fetching query rewriting options:', error);
    }
  };
  
  // Fetch query rewriting stats
  const fetchQueryStats = async () => {
    if (!uploadResult) return;
    
    try {
      const response = await axios.get(`${getBackendURL()}/query-rewriting-stats?document=${uploadResult.filename}`);
      if (response.data && response.data.stats) {
        setQueryStats(response.data.stats);
      }
    } catch (error) {
      console.error('Error fetching query stats:', error);
    }
  };
  
  // Clear caches
  const clearCaches = async () => {
    try {
      await axios.post(`${getBackendURL()}/clear-caches`);
      // Show success message
      setMessages(prev => [
        ...prev,
        {
          id: uuidv4(),
          role: 'system',
          content: 'Caches cleared successfully. This may improve performance for new queries.',
          timestamp: new Date().toISOString()
        }
      ]);
    } catch (error) {
      console.error('Error clearing caches:', error);
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
        timeout: 300000 // 5 minute timeout
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
      
      // Show document viewer for PDFs
      if (documentInfo.filename.toLowerCase().endsWith('.pdf')) {
        setShowPdfViewer(true);
      } else {
        setShowPdfViewer(false);
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
  
  // Format response for display with enhanced information
  const formatResponseForDisplay = (text, sources, queryRewritingInfo) => {
    if (!text) return "No response received.";
    
    // Clean up bullet points for consistency
    let formattedText = text.replace(/^\s*-\s+/gm, "• ");
    
    // Add query rewriting information if available
    if (queryRewritingInfo && queryRewritingInfo.all_queries && queryRewritingInfo.all_queries.length > 1) {
      const rewritingDetails = [];
      
      if (queryRewritingInfo.techniques_used) {
        const techniques = Object.entries(queryRewritingInfo.techniques_used)
          .filter(([key, value]) => value)
          .map(([key, value]) => {
            switch(key) {
              case 'prf': return 'Pseudo Relevance Feedback';
              case 'variants': return 'Query Variants';
              case 'reranking': return 'Cross-Encoder Reranking';
              case 'fusion': return `Result Fusion (${value})`;
              default: return key;
            }
          });
        
        if (techniques.length > 0) {
          rewritingDetails.push(`🔍 Enhanced with: ${techniques.join(', ')}`);
        }
      }
      
      if (queryRewritingInfo.all_queries.length > 1) {
        rewritingDetails.push(`📝 Generated ${queryRewritingInfo.all_queries.length} query variants`);
      }
      
      if (queryRewritingInfo.query_time_ms) {
        rewritingDetails.push(`⚡ Query time: ${Math.round(queryRewritingInfo.query_time_ms)}ms`);
      }
      
      if (rewritingDetails.length > 0) {
        formattedText = `${formattedText}\n\n${rewritingDetails.join('\n')}`;
      }
    }
    
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
  
  // Toggle PDF viewer
  const togglePdfViewer = () => {
    setShowPdfViewer(!showPdfViewer);
  };
  
  // Select a message to show its context in the PDF viewer
  const handleSelectMessage = (message) => {
    if (message.role === 'assistant' && message.id !== selectedMessage?.id) {
      setSelectedMessage(message);
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

  // Handle enhanced RAG evaluation
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
          content: `Starting enhanced RAG evaluation for "${uploadResult.filename}" with query rewriting techniques...`,
          timestamp: new Date().toISOString()
        }
      ]);
      
      // Use the enhanced evaluation endpoint
      const response = await axios.post(`${getBackendURL()}/evaluate/enhanced`, {
        document: uploadResult.filename,
        query: "What are the main topics and key information in this document?",
        use_prf: advancedOptions.use_prf,
        use_variants: advancedOptions.use_variants,
        prf_iterations: advancedOptions.prf_iterations
      });
      
      if (response.data && response.data.evaluation) {
        const evaluation = response.data.evaluation;
        const summary = response.data.summary;
        
        // Add detailed results
        const resultsMessage = `
📊 **Enhanced RAG Evaluation Results**

**Query Expansion**: ${summary.query_expansion ? '✅ Applied' : '❌ Not applied'}
**Techniques Used**: ${summary.techniques_applied.join(', ') || 'None'}

**Performance Comparison**:
• Enhanced with rewriting: ${evaluation.enhanced_with_rewriting.query_time_ms}ms
• Enhanced without rewriting: ${evaluation.enhanced_without_rewriting.query_time_ms}ms
• Original system: Basic retrieval

**Response Quality**:
The enhanced system with query rewriting generated more comprehensive responses by analyzing ${evaluation.enhanced_with_rewriting.query_rewriting.all_queries.length} query variants.

🎯 **Recommendation**: ${summary.query_expansion ? 'Query rewriting is providing enhanced results for your documents.' : 'Consider enabling query rewriting features for better results.'}
        `.trim();
        
        setMessages(prev => [
          ...prev,
          {
            id: uuidv4(),
            role: 'system',
            content: resultsMessage,
            timestamp: new Date().toISOString()
          }
        ]);
        
        // Update stats
        fetchQueryStats();
      }
    } catch (err) {
      console.error('Error running enhanced evaluation:', err);
      setEvaluationError('Failed to run enhanced evaluation. Check console for details.');
      
      // Add error message
      setMessages(prev => [
        ...prev,
        {
          id: uuidv4(),
          role: 'system',
          content: `Error running enhanced evaluation: ${err.response?.data?.detail || err.message || 'Unknown error'}`,
          isError: true,
          timestamp: new Date().toISOString()
        }
      ]);
    } finally {
      setEvaluating(false);
    }
  };

  // Handle submit with enhanced query rewriting and context tracking
  const handleSubmit = async () => {
    if (!input.trim() || loading || !uploadResult || !connected) {
      return;
    }
    
    // Store current input and clear it
    const currentInput = input.trim();
    setInput('');
    setLoading(true);
    
    // Add user message
    const userMessageId = uuidv4();
    const userMessage = {
      id: userMessageId,
      role: 'user',
      content: currentInput,
      timestamp: new Date().toISOString()
    };
    
    // Add enhanced thinking message
    const thinkingMessage = {
      id: uuidv4(),
      role: 'assistant',
      content: advancedOptions.use_prf || advancedOptions.use_variants ? 
        '🔍 Analyzing query and generating variants...' : 'Thinking...',
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
      
      // Send query with enhanced options and context tracking
      const response = await axios.post(`${getBackendURL()}/query-with-context`, {
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
        rerank: advancedOptions.rerank
      }, {
        timeout: 300000 // 5 minute timeout
      });
      
      // Remove thinking message
      setMessages(prev => prev.filter(msg => msg.id !== thinkingMessage.id));
      
      if (!response.data) {
        throw new Error('Invalid response from server');
      }
      
      // Process context tracking information
      let contextInfo = null;
      if (response.data.context_tracking) {
        contextInfo = contextTrackingService.processResponseContext(
          response.data, 
          thinkingMessage.id
        );
      }
      
      // Format and add assistant response with enhanced information and context
      const assistantMessageId = uuidv4();
      const formattedResponse = formatResponseForDisplay(
        response.data.response, 
        response.data.sources,
        response.data.query_rewriting_info
      );
      
      const assistantMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: formattedResponse,
        rawSources: response.data.sources || [],
        system: response.data.system || 'default',
        queryRewritingInfo: response.data.query_rewriting_info,
        enhancementStats: response.data.enhancement_stats,
        timestamp: new Date().toISOString(),
        
        // Context tracking information
        contextRanges: contextInfo?.contextRanges || [],
        contextText: contextInfo?.contextText || "",
        responseId: contextInfo?.responseId || assistantMessageId,
        
        // Reference to the user message that triggered this response
        replyTo: userMessageId
      };
      
      setMessages(prev => [...prev.filter(msg => !msg.isThinking), assistantMessage]);
      
      // Set the latest message as selected for PDF viewing
      setSelectedMessage(assistantMessage);
      
    } catch (error) {
      console.error('Query error:', error);
      
      // Remove thinking message
      setMessages(prev => prev.filter(msg => msg.id !== thinkingMessage.id));
      
      // Add error message
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

  return (
    <div className="document-chat-page">
      <header className="page-header">
        <h1>Enhanced Document Chat with Context Visualization</h1>
        <p>Upload documents, chat, and see exactly what content was used to generate answers</p>
      </header>
      
      {/* Connection Status */}
      <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
        <span className="status-indicator"></span>
        <span>{connected ? 'Connected' : 'Disconnected'}</span>
        {connected && serverInfo?.features && (
          <div className="server-features">
            <span className="feature-badge">🔍 Enhanced RAG</span>
            <span className="feature-badge">🔄 Query Rewriting</span>
            <span className="feature-badge">📄 Context Tracking</span>
          </div>
        )}
      </div>
      
      <div className="enhanced-chat-layout">
        {/* Left sidebar */}
        <div className="upload-sidebar">
          {/* Document upload card */}
          <div className="upload-card">
            <div className="card-header">
              <h2>Upload Document</h2>
            </div>
            
            <div 
              className={`dropzone ${file ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
              onClick={() => !uploading && fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
                accept=".pdf,.doc,.docx,.txt,.csv,.xlsx,.jpg,.jpeg,.png"
              />
              
              <div className="dropzone-content">
                <div className="upload-icon">
                  <span>📄</span>
                </div>
                <p className="upload-text">
                  {file ? file.name : 'Drag & drop or click to upload'}
                </p>
                <p className="file-types">
                  {file 
                    ? `${(file.size / 1024).toFixed(2)} KB` 
                    : 'PDF, DOCX, TXT, CSV, XLSX (Max 20MB)'}
                </p>
              </div>
            </div>
            
            {/* Progress bar */}
            {uploading && (
              <div className="progress-container">
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <div className="progress-text">
                  Uploading: {uploadProgress}%
                </div>
              </div>
            )}
            
            {/* Error message */}
            {uploadError && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                <p>{uploadError}</p>
              </div>
            )}
            
            {/* Upload button */}
            <button
              onClick={handleUpload}
              disabled={!file || uploading || !connected}
              className={`upload-button ${(!file || uploading || !connected) ? 'disabled' : ''}`}
            >
              {uploading ? 'Uploading...' : 'Upload Document'}
            </button>
          </div>
          
          {/* Document info card */}
          {uploadResult && (
            <div className="document-card">
              <div className="card-header success">
                <h2>Document Details</h2>
              </div>
              
              <div className="document-details">
                <div className="info-item">
                  <span className="info-label">Filename:</span>
                  <span className="info-value">{uploadResult.filename}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Chunks:</span>
                  <span className="info-value">{uploadResult.chunks}</span>
                </div>
                
                {/* Toggle PDF viewer button (only for PDF files) */}
                {uploadResult.filename.toLowerCase().endsWith('.pdf') && (
                  <div className="pdf-viewer-toggle">
                    <button 
                      onClick={togglePdfViewer}
                      className={`toggle-button ${showPdfViewer ? 'active' : ''}`}
                    >
                      {showPdfViewer ? 'Hide PDF Viewer' : 'Show PDF Viewer'}
                    </button>
                    <p className="toggle-description">
                      {showPdfViewer 
                        ? 'PDF viewer is showing context used to generate answers' 
                        : 'Enable PDF viewer to see highlighted context'}
                    </p>
                  </div>
                )}
                
                {/* Enhancement status */}
                <div className="enhancement-status">
                  <h4>Active Enhancements:</h4>
                  <div className="enhancement-badges">
                    {advancedOptions.use_prf && <span className="enhancement-badge prf">PRF</span>}
                    {advancedOptions.use_variants && <span className="enhancement-badge variants">Variants</span>}
                    {advancedOptions.rerank && <span className="enhancement-badge rerank">Reranking</span>}
                    {advancedOptions.fusion_method === 'rrf' && <span className="enhancement-badge fusion">RRF</span>}
                    <span className="enhancement-badge context">Context Tracking</span>
                  </div>
                </div>
                
                {uploadResult.preview && (
                  <details className="document-preview">
                    <summary>Document Preview</summary>
                    <div className="preview-content">
                      {uploadResult.preview}
                    </div>
                  </details>
                )}
              </div>
            </div>
          )}

          {/* Options panel - condensed for space */}
          {connected && (
            <div className="options-card">
              <div className="card-header">
                <h2>RAG Options</h2>
              </div>
              
              <div className="options-content">
                {/* Show minimal options for space */}
                <div className="option-group">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      name="use_prf"
                      checked={advancedOptions.use_prf}
                      onChange={handleOptionChange}
                    />
                    <span>Pseudo Relevance Feedback</span>
                  </label>
                </div>
                
                <div className="option-group">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      name="use_variants"
                      checked={advancedOptions.use_variants}
                      onChange={handleOptionChange}
                    />
                    <span>Query Variants</span>
                  </label>
                </div>
                
                <div className="option-group">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      name="rerank"
                      checked={advancedOptions.rerank}
                      onChange={handleOptionChange}
                    />
                    <span>Cross-Encoder Reranking</span>
                  </label>
                </div>
                
                <div className="option-group">
                  <label>Temperature:</label>
                  <input
                    type="range"
                    name="temperature"
                    min="0"
                    max="1"
                    step="0.1"
                    value={advancedOptions.temperature}
                    onChange={handleOptionChange}
                  />
                  <span>{advancedOptions.temperature}</span>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Middle chat panel */}
        <div className="chat-container">
          <div className="chat-header">
            <div className="chat-title">
              <h2>Document Chat</h2>
              <p>
                {uploadResult 
                  ? `Ask questions about: ${uploadResult.filename}` 
                  : 'Upload a document to start chatting'}
              </p>
            </div>
            
            {/* Enhancement indicators */}
            {uploadResult && (
              <div className="enhancement-indicators">
                {advancedOptions.use_prf && (
                  <div className="enhancement-indicator prf" title="Pseudo Relevance Feedback enabled">
                    🔍 PRF
                  </div>
                )}
                {advancedOptions.use_variants && (
                  <div className="enhancement-indicator variants" title="Query variants enabled">
                    📝 Variants
                  </div>
                )}
                {advancedOptions.rerank && (
                  <div className="enhancement-indicator rerank" title="Cross-encoder reranking enabled">
                    🎯 Reranking
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className="messages-container">
            {!connected ? (
              <div className="empty-state">
                <div className="empty-icon disconnected">🔄</div>
                <h3>Connecting to Server...</h3>
                <p>Please wait while we establish connection to the enhanced RAG server.</p>
              </div>
            ) : !uploadResult ? (
              <div className="empty-state">
                <div className="empty-icon">📄</div>
                <h3>No Document Uploaded</h3>
                <p>Upload a document using the panel on the left to start asking questions.</p>
                
                <div className="feature-preview">
                  <h4>New Feature: Context Visualization</h4>
                  <p>For PDF documents, you can now see exactly what parts of the document were used to generate answers!</p>
                  <div className="feature-highlights">
                    <div className="feature-highlight">
                      <span className="highlight-icon">🔍</span>
                      <span>View highlighted context</span>
                    </div>
                    <div className="feature-highlight">
                      <span className="highlight-icon">📄</span>
                      <span>Navigate document sections</span>
                    </div>
                    <div className="feature-highlight">
                      <span className="highlight-icon">💬</span>
                      <span>Verify answer sources</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">💬</div>
                <h3>Start the Conversation</h3>
                <p>Your document is ready. Type a question below to start chatting.</p>
                
                {uploadResult.filename.toLowerCase().endsWith('.pdf') && (
                  <div className="feature-callout">
                    <h4>Context Visualization Enabled</h4>
                    <p>When you ask questions, you'll see exactly which parts of your PDF were used to generate the answer.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="messages-list">
                {messages.map(message => (
                  <div 
                    key={message.id} 
                    className={`message ${message.role} ${message.isThinking ? 'thinking' : ''} ${message.isError ? 'error' : ''} ${message.contextRanges && message.contextRanges.length > 0 ? 'has-context' : ''} ${selectedMessage?.id === message.id ? 'selected' : ''}`}
                    onClick={() => handleSelectMessage(message)}
                  >
                    <div className="message-content">
                      {message.content}
                    </div>
                    
                    {/* Context indicator */}
                    {message.role === 'assistant' && message.contextRanges && message.contextRanges.length > 0 && (
                      <div className="context-indicator" title="This message has context visualization available">
                        <span className="context-icon">📌</span>
                        <span className="context-label">
                          {message.contextRanges.length} context {message.contextRanges.length === 1 ? 'region' : 'regions'} available
                        </span>
                      </div>
                    )}
                    
                    <div className="message-footer">
                      <span className="message-time">
                        {new Date(message.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                      </span>
                      {message.queryRewritingInfo && message.queryRewritingInfo.query_time_ms && (
                        <span className="performance-badge">
                          ⚡ {Math.round(message.queryRewritingInfo.query_time_ms)}ms
                        </span>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
          
          {/* Enhanced suggested questions */}
          {suggestedQuestions.length > 0 && showSuggestions && (
            <div className="suggested-questions-container">
              <div className="suggested-questions-header">
                <span className="suggested-questions-title">
                  💡 Suggested Questions
                </span>
                <button 
                  className="suggested-questions-close" 
                  onClick={() => setShowSuggestions(false)}
                  aria-label="Close suggested questions"
                >
                  ✕
                </button>
              </div>
              <div className="suggested-questions">
                {suggestedQuestions.map((question, idx) => (
                  <button 
                    key={idx}
                    onClick={() => setInput(question)}
                    className="suggested-question"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {/* Input area */}
          <div className="input-container">
            <textarea
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder={connected && uploadResult ? 
                "Type your question..." : 
                connected ? "Upload a document to start chatting" : 
                "Connecting to server..."}
              disabled={!connected || !uploadResult || loading}
              className={`chat-input ${(!connected || !uploadResult) ? 'disabled' : ''}`}
              rows={1}
            />
            
            <button
              onClick={handleSubmit}
              disabled={!connected || !uploadResult || !input.trim() || loading}
              className={`send-button ${(!connected || !uploadResult || !input.trim() || loading) ? 'disabled' : ''}`}
            >
              {loading ? 'Thinking...' : 'Send'}
            </button>
          </div>
        </div>
        
        {/* Right PDF viewer panel - only show for PDFs */}
        {showPdfViewer && uploadResult && uploadResult.filename.toLowerCase().endsWith('.pdf') && (
          <div className="pdf-viewer-panel">
            <PDFContextViewer 
              documentName={uploadResult.filename}
              currentQuery={selectedMessage?.replyTo ? messages.find(m => m.id === selectedMessage.replyTo)?.content : null}
              currentAnswer={selectedMessage?.content}
              currentMessage={selectedMessage}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedDocumentChat;