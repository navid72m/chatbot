import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const DocumentChat = ({ documentId }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Load initial messages if any exist
  useEffect(() => {
    if (documentId) {
      fetchConversationHistory();
    }
  }, [documentId]);
  
  // Auto-scroll to bottom of messages
  useEffect(() => {
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
    
    try {
      // Send the query to the server
      const response = await axios.post(`http://localhost:8000/api/documents/${documentId}/query`, {
        query: userMessage.content
      });
      
      // Add assistant response to the conversation
      const assistantMessage = {
        id: response.data.id || Date.now().toString() + '-assistant',
        role: 'assistant',
        content: response.data.response || response.data.content,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error querying document:', error);
      
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
        backgroundColor: '#f8fafc'
      }}>
        <h2 style={{ margin: 0, fontSize: '18px' }}>Document Chat</h2>
        <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: '#64748b' }}>
          Ask questions about your uploaded document
        </p>
      </div>
      
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
                maxWidth: '80%',
                padding: '12px 16px',
                borderRadius: '8px',
                alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
                backgroundColor: message.role === 'user' ? '#4f46e5' : 
                                message.role === 'system' ? '#f87171' : '#f1f5f9',
                color: message.role === 'user' ? 'white' : 
                       message.role === 'system' ? 'white' : '#1e293b',
              }}
            >
              <div style={{ fontSize: '14px', whiteSpace: 'pre-wrap' }}>
                {message.content}
              </div>
              <div style={{ 
                fontSize: '11px', 
                marginTop: '4px',
                opacity: 0.7,
                textAlign: message.role === 'user' ? 'right' : 'left'
              }}>
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))
        )}
        {loading && (
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

export default DocumentChat;