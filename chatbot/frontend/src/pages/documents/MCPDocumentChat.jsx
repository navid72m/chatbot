import React, { useState, useRef, useEffect, useCallback } from 'react';
import '../../styles/pages/MCPDocumentChat.css';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { getBackendURL } from '@/api/baseURL';
import { useNavigate } from 'react-router-dom';

// Enhanced MCPDocumentChat component with query rewriting features and image text display
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
  const [showSuggestions, setShowSuggestions] = useState(true);
  
  // Evaluation state
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState(null);
  
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
  
  // Image text extraction state
  const [documentText, setDocumentText] = useState(null);
  const [loadingDocumentText, setLoadingDocumentText] = useState(false);
  const [showFullText, setShowFullText] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  
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

  // Copy text to clipboard with success feedback
  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        setCopySuccess(true);
        setTimeout(() => setCopySuccess(false), 2000);
      } catch (fallbackErr) {
        console.error('Fallback copy failed:', fallbackErr);
      }
      document.body.removeChild(textArea);
    }
  };

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
  
  // Fetch document text content (especially for images)
  const fetchDocumentText = async (filename) => {
    if (!filename) return;
    
    setLoadingDocumentText(true);
    try {
      const response = await axios.get(`${getBackendURL()}/document-text/${filename}`);
      if (response.data) {
        setDocumentText(response.data);
      }
    } catch (error) {
      console.error('Error fetching document text:', error);
      setDocumentText({
        error: error.response?.data?.detail || 'Failed to extract text',
        document: filename,
        is_image: false
      });
    } finally {
      setLoadingDocumentText(false);
    }
  };
  
  // Reprocess image with different OCR method
  const reprocessImage = async (method = 'auto') => {
    if (!uploadResult || !documentText?.is_image) return;
    
    setReprocessing(true);
    try {
      const response = await axios.post(`${getBackendURL()}/reprocess-image`, {
        document: uploadResult.filename,
        method: method
      });
      
      if (response.data) {
        // Update the document text with new extraction
        setDocumentText(prev => ({
          ...prev,
          extracted_text: response.data.extracted_text,
          text_length: response.data.text_length,
          extraction_method: `Reprocessed with ${response.data.reprocessing_method}`,
          reprocessing_results: response.data.all_results
        }));
        
        // Add success message to chat
        setMessages(prev => [
          ...prev,
          {
            id: uuidv4(),
            role: 'system',
            content: `Image reprocessed successfully with ${method} method. Extracted ${response.data.text_length} characters.`,
            timestamp: new Date().toISOString()
          }
        ]);
      }
    } catch (error) {
      console.error('Error reprocessing image:', error);
      setMessages(prev => [
        ...prev,
        {
          id: uuidv4(),
          role: 'system',
          content: `Failed to reprocess image: ${error.response?.data?.detail || error.message}`,
          isError: true,
          timestamp: new Date().toISOString()
        }
      ]);
    } finally {
      setReprocessing(false);
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
      
      // Fetch initial stats
      fetchQueryStats();
      
      // Fetch document text content
      fetchDocumentText(documentInfo.filename);
      
      // Add welcome message with enhancement info
      const enhancementFeatures = [];
      if (advancedOptions.use_prf) enhancementFeatures.push('Pseudo Relevance Feedback');
      if (advancedOptions.use_variants) enhancementFeatures.push('Query Variants');
      if (advancedOptions.rerank) enhancementFeatures.push('Cross-Encoder Reranking');
      
      const welcomeMessage = `Document "${documentInfo.filename}" uploaded successfully with enhanced RAG capabilities.${enhancementFeatures.length > 0 ? `\n\nActive enhancements: ${enhancementFeatures.join(', ')}` : ''}\n\nYou can now ask questions about it.`;
      
      setMessages([
        {
          id: uuidv4(),
          role: 'system',
          content: welcomeMessage,
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

  // Handle submit with enhanced query rewriting
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
      
      // Send query with enhanced options
      const response = await axios.post(`${getBackendURL()}/query-sync`, {
        query: currentInput,
        document: uploadResult.filename,
        model: advancedOptions.model,
        temperature: advancedOptions.temperature,
        context_window: advancedOptions.context_window,
        quantization: advancedOptions.quantization,
        use_advanced_rag: advancedOptions.use_advanced_rag,
        use_llama_index: advancedOptions.use_llama_index,
        // Enhanced query rewriting parameters
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
      
      // Format and add assistant response with enhanced information
      const formattedResponse = formatResponseForDisplay(
        response.data.response, 
        response.data.sources,
        response.data.query_rewriting_info
      );
      
      const assistantMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: formattedResponse,
        rawSources: response.data.sources || [],
        system: response.data.system || 'default',
        queryRewritingInfo: response.data.query_rewriting_info,
        enhancementStats: response.data.enhancement_stats,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev.filter(msg => !msg.isThinking), assistantMessage]);
      
      // Update stats after query
      fetchQueryStats();
      
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
    <div className="document-chat-container">
      <div className="document-chat-header">
        <h1>Enhanced Document Chat</h1>
        <p>Upload documents and chat with advanced query rewriting capabilities</p>
      </div>
      
      {/* Connection Status */}
      <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
        <span className="status-indicator"></span>
        <span>{connected ? 'Connected' : 'Disconnected'}</span>
        {connected && serverInfo?.features && (
          <div className="server-features">
            <span className="feature-badge">🔍 Enhanced RAG</span>
            <span className="feature-badge">🔄 Query Rewriting</span>
            <span className="feature-badge">🎯 PRF</span>
          </div>
        )}
      </div>
      
      <div className="document-chat-grid">
        {/* Left sidebar */}
        <div className="upload-sidebar">
          {/* Document upload card */}
          <div className="card">
            <div className="card-header">
              <h2>Upload Document</h2>
            </div>
            
            {/* Model Settings */}
            <div className="settings-panel">
              <div 
                className="settings-header"
                onClick={() => setShowOptions(!showOptions)}
              >
                <span className="settings-title">Model Settings</span>
                <span className={`toggle-icon ${showOptions ? 'open' : ''}`}>▼</span>
              </div>
              
              {showOptions && (
                <div className="settings-content">
                  <div className="setting-group">
                    <label className="setting-label">Model:</label>
                    <select
                      name="model"
                      value={advancedOptions.model}
                      onChange={handleOptionChange}
                      className="select-input"
                    >
                      {availableModels.map(model => (
                        <option key={model} value={model}>{model}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div className="range-container">
                    <div className="range-header">
                      <label className="setting-label">Temperature:</label>
                      <span className="range-value">{advancedOptions.temperature}</span>
                    </div>
                    <input
                      type="range"
                      name="temperature"
                      min="0"
                      max="1"
                      step="0.1"
                      value={advancedOptions.temperature}
                      onChange={handleOptionChange}
                      className="range-input"
                    />
                    <div className="range-labels">
                      <span>Precise</span>
                      <span>Creative</span>
                    </div>
                  </div>

                  <div className="checkbox-group">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        name="use_llama_index"
                        checked={advancedOptions.use_llama_index}
                        onChange={handleOptionChange}
                        className="checkbox-input"
                      />
                      <div className="checkbox-text">
                        <span>Use Enhanced RAG</span>
                        <span className="checkbox-description">Advanced retrieval with query rewriting</span>
                      </div>
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Query Rewriting Settings */}
            <div className="settings-panel">
              <div 
                className="settings-header"
                onClick={() => setShowQueryRewritingOptions(!showQueryRewritingOptions)}
              >
                <span className="settings-title">🔍 Query Rewriting</span>
                <span className={`toggle-icon ${showQueryRewritingOptions ? 'open' : ''}`}>▼</span>
              </div>
              
              {showQueryRewritingOptions && (
                <div className="settings-content">
                  <div className="checkbox-group">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        name="use_prf"
                        checked={advancedOptions.use_prf}
                        onChange={handleOptionChange}
                        className="checkbox-input"
                      />
                      <div className="checkbox-text">
                        <span>Pseudo Relevance Feedback</span>
                        <span className="checkbox-description">Expand queries using document terms</span>
                      </div>
                    </label>
                  </div>

                  <div className="checkbox-group">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        name="use_variants"
                        checked={advancedOptions.use_variants}
                        onChange={handleOptionChange}
                        className="checkbox-input"
                      />
                      <div className="checkbox-text">
                        <span>Query Variants</span>
                        <span className="checkbox-description">Generate multiple query reformulations</span>
                      </div>
                    </label>
                  </div>

                  <div className="checkbox-group">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        name="rerank"
                        checked={advancedOptions.rerank}
                        onChange={handleOptionChange}
                        className="checkbox-input"
                      />
                      <div className="checkbox-text">
                        <span>Cross-Encoder Reranking</span>
                        <span className="checkbox-description">Advanced relevance scoring</span>
                      </div>
                    </label>
                  </div>

                  <div className="setting-group">
                    <label className="setting-label">PRF Iterations:</label>
                    <select
                      name="prf_iterations"
                      value={advancedOptions.prf_iterations}
                      onChange={handleOptionChange}
                      className="select-input"
                      disabled={!advancedOptions.use_prf}
                    >
                      <option value={1}>1 iteration</option>
                      <option value={2}>2 iterations</option>
                      <option value={3}>3 iterations</option>
                    </select>
                  </div>

                  <div className="setting-group">
                    <label className="setting-label">Fusion Method:</label>
                    <select
                      name="fusion_method"
                      value={advancedOptions.fusion_method}
                      onChange={handleOptionChange}
                      className="select-input"
                    >
                      <option value="rrf">Reciprocal Rank Fusion</option>
                      <option value="score">Score-based Fusion</option>
                    </select>
                  </div>
                </div>
              )}
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
                    : 'PDF, DOCX, TXT, CSV, XLSX, Images (Max 20MB)'}
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
              className={`button full-width ${(!file || uploading || !connected) ? 'disabled' : ''}`}
            >
              {uploading ? 'Uploading...' : 'Upload Document'}
            </button>
          </div>
          
          {/* Document info card */}
          {uploadResult && (
            <div className="card card-success">
              <div className="card-header success">
                <h2>Document Details</h2>
              </div>
              
              <div className="card-body">
                <div className="info-item">
                  <span className="info-label">Filename:</span>
                  <span className="info-value">{uploadResult.filename}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Chunks:</span>
                  <span className="info-value">{uploadResult.chunks}</span>
                </div>
                
                {/* Enhancement status */}
                <div className="enhancement-status">
                  <h4>Active Enhancements:</h4>
                  <div className="enhancement-badges">
                    {advancedOptions.use_prf && <span className="enhancement-badge prf">PRF</span>}
                    {advancedOptions.use_variants && <span className="enhancement-badge variants">Variants</span>}
                    {advancedOptions.rerank && <span className="enhancement-badge rerank">Reranking</span>}
                    {advancedOptions.fusion_method === 'rrf' && <span className="enhancement-badge fusion">RRF</span>}
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
        </div>
        
        {/* Chat interface */}
        <div className="chat-container">
          <div className="chat-header">
            <div className="chat-title">
              <h2>Enhanced Document Chat</h2>
              <p className="chat-subtitle">
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
                <h3 className="empty-title">Connecting to Server...</h3>
                <p className="empty-text">Please wait while we establish connection to the enhanced RAG server.</p>
              </div>
            ) : !uploadResult ? (
              <div className="empty-state">
                <div className="empty-icon">📄</div>
                <h3 className="empty-title">No Document Uploaded</h3>
                <p className="empty-text">Upload a document using the panel on the left to start asking questions with enhanced RAG capabilities.</p>
                
                <div className="enhancement-preview">
                  <h4>Available Enhancements:</h4>
                  <ul>
                    <li>🔍 <strong>Pseudo Relevance Feedback</strong> - Expands queries using document terms</li>
                    <li>📝 <strong>Query Variants</strong> - Generates multiple query reformulations</li>
                    <li>🎯 <strong>Cross-Encoder Reranking</strong> - Advanced relevance scoring</li>
                    <li>🔄 <strong>Result Fusion</strong> - Intelligently combines multiple query results</li>
                  </ul>
                </div>
              </div>
            ) : messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">💬</div>
                <h3 className="empty-title">Start the Enhanced Conversation</h3>
                <p className="empty-text">Your document is ready with enhanced RAG capabilities. Type a question below to experience improved retrieval quality.</p>
                
                {/* Show active enhancements */}
                <div className="active-enhancements">
                  <h4>Active Enhancements:</h4>
                  <div className="enhancement-list">
                    {advancedOptions.use_prf && (
                      <div className="enhancement-item">
                        <span className="enhancement-icon">🔍</span>
                        <span>Pseudo Relevance Feedback ({advancedOptions.prf_iterations} iteration{advancedOptions.prf_iterations > 1 ? 's' : ''})</span>
                      </div>
                    )}
                    {advancedOptions.use_variants && (
                      <div className="enhancement-item">
                        <span className="enhancement-icon">📝</span>
                        <span>Query Variants Generation</span>
                      </div>
                    )}
                    {advancedOptions.rerank && (
                      <div className="enhancement-item">
                        <span className="enhancement-icon">🎯</span>
                        <span>Cross-Encoder Reranking</span>
                      </div>
                    )}
                    <div className="enhancement-item">
                      <span className="enhancement-icon">🔄</span>
                      <span>{advancedOptions.fusion_method.toUpperCase()} Result Fusion</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="messages-list">
                {messages.map(message => (
                  <div 
                    key={message.id} 
                    className={`message ${message.role} ${message.isThinking ? 'thinking' : ''} ${message.isError ? 'error' : ''} ${message.system === 'enhanced_llama_index' ? 'enhanced' : ''}`}
                  >
                    <div className="message-content">
                      {message.content}
                    </div>
                    
                    {/* Enhanced message details */}
                    {message.queryRewritingInfo && message.queryRewritingInfo.all_queries && message.queryRewritingInfo.all_queries.length > 1 && (
                      <details className="query-details">
                        <summary>Query Enhancement Details</summary>
                        <div className="query-expansion-info">
                          <div className="expansion-item">
                            <strong>Original Query:</strong>
                            <span>{message.queryRewritingInfo.original_query}</span>
                          </div>
                          <div className="expansion-item">
                            <strong>Generated Queries ({message.queryRewritingInfo.all_queries.length}):</strong>
                            <ul>
                              {message.queryRewritingInfo.all_queries.map((query, idx) => (
                                <li key={idx} className={idx === 0 ? 'original' : 'variant'}>
                                  {query} {idx === 0 ? '(original)' : `(variant ${idx})`}
                                </li>
                              ))}
                            </ul>
                          </div>
                          {message.queryRewritingInfo.techniques_used && (
                            <div className="expansion-item">
                              <strong>Applied Techniques:</strong>
                              <div className="technique-badges">
                                {Object.entries(message.queryRewritingInfo.techniques_used)
                                  .filter(([key, value]) => value)
                                  .map(([key, value]) => (
                                    <span key={key} className={`technique-badge ${key}`}>
                                      {key === 'prf' ? 'PRF' : 
                                       key === 'variants' ? 'Variants' : 
                                       key === 'reranking' ? 'Reranking' : 
                                       key === 'fusion' ? `Fusion (${value})` : key}
                                    </span>
                                  ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </details>
                    )}
                    
                    <div className="message-footer">
                      <span className="message-time">
                        {new Date(message.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                      </span>
                      {message.system === 'enhanced_llama_index' && (
                        <span className="model-badge enhanced">Enhanced RAG</span>
                      )}
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
            <div className="suggested-questions-container enhanced">
              <div className="suggested-questions-header">
                <span className="suggested-questions-title">
                  💡 AI-Generated Questions
                  <span className="enhancement-note">(Will use active enhancements)</span>
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
                    className="suggested-question enhanced"
                  >
                    <span className="question-text">{question}</span>
                    <div className="enhancement-preview-badges">
                      {advancedOptions.use_prf && <span className="mini-badge">PRF</span>}
                      {advancedOptions.use_variants && <span className="mini-badge">Variants</span>}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {/* Enhanced input area */}
          <div className="input-container enhanced">
            <div className="input-wrapper">
              <textarea
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={connected && uploadResult ? 
                  "Type your question... (Enhanced RAG will improve your results)" : 
                  connected ? "Upload a document to start chatting" : 
                  "Connecting to server..."}
                disabled={!connected || !uploadResult || loading}
                className={`chat-input ${(!connected || !uploadResult) ? 'disabled' : ''}`}
                rows={1}
              />
              
              {/* Enhancement status in input */}
              {uploadResult && (advancedOptions.use_prf || advancedOptions.use_variants || advancedOptions.rerank) && (
                <div className="input-enhancements">
                  <div className="enhancement-tooltip">
                    🔍 Enhanced retrieval active
                    <div className="tooltip-content">
                      <div>Your query will be enhanced using:</div>
                      {advancedOptions.use_prf && <div>• Pseudo Relevance Feedback</div>}
                      {advancedOptions.use_variants && <div>• Query Variants</div>}
                      {advancedOptions.rerank && <div>• Cross-Encoder Reranking</div>}
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <button
              onClick={handleSubmit}
              disabled={!connected || !uploadResult || !input.trim() || loading}
              className={`button send-button enhanced ${(!connected || !uploadResult || !input.trim() || loading) ? 'disabled' : ''}`}
            >
              {loading ? (
                <span className="loading-text">
                  {advancedOptions.use_prf || advancedOptions.use_variants ? 'Enhancing...' : 'Thinking...'}
                </span>
              ) : (
                <span className="send-text">
                  Send
                  {(advancedOptions.use_prf || advancedOptions.use_variants) && (
                    <span className="enhancement-indicator">🔍</span>
                  )}
                </span>
              )}
            </button>
          </div>
        </div>

        {/* Right side image panel */}
        <div >
          <div >
            <h2>Document Preview</h2>
          </div>
          
          {uploadResult && documentText?.is_image ? (
            <div className="image-content">
              <div className="image-container">
                <p>Image</p>
                <img 
                  src={`${getBackendURL()}/document/${uploadResult.filename}`}
                  alt={`Preview of ${uploadResult.filename}`}
                  className="document-image"
                />
              </div>
              
              <div className="image-controls">
                <button 
                  onClick={() => window.open(`${getBackendURL()}/document/${uploadResult.filename}`, '_blank')}
                  className="view-full-btn"
                >
                  🔍 View Full Size
                </button>
                
                <div className="image-info">
                  {documentText.image_info && (
                    <>
                      <div className="info-row">
                        <span className="info-label">Dimensions:</span>
                        <span className="info-value">
                          {documentText.image_info.width}×{documentText.image_info.height}px
                        </span>
                      </div>
                      <div className="info-row">
                        <span className="info-label">Format:</span>
                        <span className="info-value">{documentText.image_info.format}</span>
                      </div>
                      <div className="info-row">
                        <span className="info-label">Size:</span>
                        <span className="info-value">
                          {(documentText.image_info.size_bytes / 1024).toFixed(1)} KB
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="no-image">
              <div className="empty-icon">🖼️</div>
              <p>Upload an image to see preview</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MCPDocumentChat;