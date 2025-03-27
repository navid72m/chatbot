import React, { useState, useRef } from 'react';
import axios from 'axios';

// DocumentChat component embedded within the same file for simplicity
const DocumentChat = ({ documentId }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [advancedOptions, setAdvancedOptions] = useState({
    use_advanced_rag: false,
    use_cot: false,
    use_kg: true,
    verify_answers: true,
    use_multihop: true,
    model: "llama3",
    temperature: 0.7,
    context_window: 10,
    quantization: "Q4_0"
  });
  const [showOptions, setShowOptions] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Load initial messages if any exist
  React.useEffect(() => {
    if (documentId) {
      fetchConversationHistory();
    }
  }, [documentId]);
  
  // Auto-scroll to bottom of messages
  React.useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const fetchConversationHistory = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`http://localhost:8000/api/documents/${documentId}/conversation`);
      if (response.data && Array.isArray(response.data)) {
        setMessages(response.data);
      }
    } catch (error) {
      console.error('Error fetching conversation history:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  const handleInputChange = (e) => {
    setInput(e.target.value);
  };
  
  const handleAdvancedOptionChange = (e) => {
    const { name, value, type, checked } = e.target;
    setAdvancedOptions(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };
  
  const handleSubmit = async () => {
    if (!input.trim() || loading) return;
    
    // Add user message to the conversation
    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    
    // Add a "thinking" message
    const thinkingMessage = {
      id: Date.now().toString() + '-thinking',
      role: 'system',
      content: 'Processing your query... (this may take a while)',
      isThinking: true,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, thinkingMessage]);
    
    try {
      // Send the query to the server with advanced options
      const response = await axios.post(`http://localhost:8000/api/documents/${documentId}/query`, {
        query: userMessage.content,
        ...advancedOptions
      });
      
      // Remove the thinking message
      setMessages(prev => prev.filter(msg => !msg.isThinking));
      
      // Add assistant response to the conversation
      const assistantMessage = {
        id: Date.now().toString() + '-assistant',
        role: 'assistant',
        content: response.data.response,
        sources: response.data.sources,
        reasoning: response.data.reasoning,
        confidence: response.data.confidence,
        retrieval_time: response.data.retrieval_time,
        verification: response.data.verification,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error querying document:', error);
      
      // Remove the thinking message
      setMessages(prev => prev.filter(msg => !msg.isThinking));
      
      // Add error message
      const errorMessage = {
        id: Date.now().toString() + '-error',
        role: 'system',
        content: 'Sorry, there was an error processing your query. Please try again.',
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };
  
  return (
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
        backgroundColor: '#f8fafc',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px' }}>Document Chat</h2>
          <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: '#64748b' }}>
            Ask questions about your uploaded document
          </p>
        </div>
        <button 
          onClick={() => setShowOptions(!showOptions)}
          style={{
            backgroundColor: showOptions ? '#4f46e5' : 'transparent',
            color: showOptions ? 'white' : '#64748b',
            border: '1px solid ' + (showOptions ? '#4f46e5' : '#cbd5e1'),
            borderRadius: '4px',
            padding: '6px 12px',
            fontSize: '12px',
            cursor: 'pointer'
          }}
        >
          {showOptions ? 'Hide Options' : 'Show Options'}
        </button>
      </div>
      
      {/* Advanced options */}
      {showOptions && (
        <div style={{
          padding: '16px',
          borderBottom: '1px solid #e2e8f0',
          backgroundColor: '#f1f5f9',
          fontSize: '14px'
        }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: '16px' }}>Advanced Query Options</h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div>
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
                Use Chain-of-Thought
              </label>
              
              <label style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                <input
                  type="checkbox"
                  name="use_kg"
                  checked={advancedOptions.use_kg}
                  onChange={handleAdvancedOptionChange}
                  style={{ marginRight: '8px' }}
                />
                Use Knowledge Graph
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
              
              <label style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                <input
                  type="checkbox"
                  name="use_multihop"
                  checked={advancedOptions.use_multihop}
                  onChange={handleAdvancedOptionChange}
                  style={{ marginRight: '8px' }}
                />
                Use Multi-hop Reasoning
              </label>
            </div>
            
            <div>
              <div style={{ marginBottom: '8px' }}>
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
                  <option value="llama3">Llama 3</option>
                  <option value="mistral">Mistral</option>
                  <option value="mixtral">Mixtral</option>
                </select>
              </div>
              
              <div style={{ marginBottom: '8px' }}>
                <label style={{ display: 'block', marginBottom: '4px' }}>Temperature: {advancedOptions.temperature}</label>
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
              
              <div style={{ marginBottom: '8px' }}>
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
            </div>
          </div>
        </div>
      )}
      
      {/* Messages container */}
      <div style={{ 
        flex: 1, 
        overflowY: 'auto',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {messages.length === 0 ? (
          <div style={{ 
            textAlign: 'center', 
            color: '#94a3b8',
            margin: 'auto'
          }}>
            <p>No messages yet. Start the conversation by asking a question.</p>
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
                                message.role === 'system' ? '#f87171' : '#f1f5f9',
                color: message.role === 'user' ? 'white' : 
                       message.role === 'system' ? 'white' : '#1e293b',
                opacity: message.isThinking ? 0.7 : 1,
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
                  <span>Retrieval: {message.retrieval_time}s</span>
                )}
                {message.confidence && (
                  <span>Confidence: {Math.round(message.confidence * 100)}%</span>
                )}
                <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
              </div>
            </div>
          ))
        )}
        {loading && !messages.some(m => m.isThinking) && (
          <div style={{ 
            alignSelf: 'flex-start',
            padding: '12px 16px',
            borderRadius: '8px',
            backgroundColor: '#f1f5f9',
            color: '#64748b'
          }}>
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input container */}
      <div style={{ 
        padding: '16px',
        borderTop: '1px solid #e2e8f0',
        backgroundColor: '#f8fafc',
        display: 'flex',
        gap: '8px'
      }}>
        <textarea
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Type your question..."
          style={{
            flex: 1,
            border: '1px solid #cbd5e1',
            borderRadius: '8px',
            padding: '12px',
            resize: 'none',
            fontSize: '14px',
            minHeight: '24px',
            maxHeight: '120px',
            fontFamily: 'inherit'
          }}
          rows={1}
          disabled={loading}
        />
        <button
          onClick={handleSubmit}
          disabled={!input.trim() || loading}
          style={{
            backgroundColor: !input.trim() || loading ? '#94a3b8' : '#4f46e5',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            padding: '0 16px',
            fontSize: '14px',
            cursor: !input.trim() || loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
};

// DocumentUpload component would remain the same as before