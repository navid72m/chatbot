// src/components/ServerStatusComponent.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ServerStatusComponent = ({ onStatusChange }) => {
  const [status, setStatus] = useState('checking');
  const [serverInfo, setServerInfo] = useState(null);
  const [error, setError] = useState(null);

  // Function to check server connection manually
  const checkConnection = async (url) => {
    try {
      const response = await axios.get(`${url}/health`, { timeout: 2000 });
      if (response.status === 200) {
        setStatus('connected');
        setError(null);
        if (onStatusChange) onStatusChange(true);
        return true;
      }
    } catch (err) {
      console.error('Server connection error:', err);
      setStatus('disconnected');
      setError(err.message);
      if (onStatusChange) onStatusChange(false);
      return false;
    }
    return false;
  };

  // Handle server restart request
  const handleRestartServer = () => {
    setStatus('restarting');
    // Check if we're in Electron
    if (window.api) {
      window.api.send('restart-server');
    } else {
      // If not in Electron, just retry the connection check
      setTimeout(() => checkConnection(serverInfo?.url || 'http://localhost:8000'), 1000);
    }
  };

  useEffect(() => {
    // Handle messages from Electron main process if we're in Electron
    if (window.api) {
      // Listen for server status updates
      window.api.receive('server-status', (data) => {
        console.log('Received server status from main process:', data);
        setServerInfo(data);
        
        if (data.running) {
          setStatus('connected');
          setError(null);
          if (onStatusChange) onStatusChange(true);
        } else {
          setStatus('disconnected');
          setError(data.error || 'Server is not running');
          if (onStatusChange) onStatusChange(false);
        }
      });
      
      return () => {
        // Cleanup listeners if component unmounts
      };
    } else {
      // If not in Electron, check connection directly
      checkConnection('http://localhost:8000');
    }
  }, [onStatusChange]);

  return (
    <div 
      style={{ 
        position: 'fixed',
        top: '10px',
        right: '10px',
        padding: '8px 12px',
        borderRadius: '4px',
        backgroundColor: status === 'connected' ? '#10b981' : 
                         status === 'checking' || status === 'restarting' ? '#f59e0b' : 
                         '#ef4444',
        color: 'white',
        fontSize: '12px',
        fontWeight: '500',
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        zIndex: 1000,
        boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)'
      }}
    >
      {/* Status indicator dot */}
      <div 
        style={{ 
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor: 'white',
          opacity: 0.8,
          animation: (status === 'checking' || status === 'restarting') ? 'pulse 1.5s infinite' : 'none'
        }}
      />
      
      {/* Status text */}
      <span>
        {status === 'connected' ? 'Connected to Server' :
         status === 'checking' ? 'Checking Connection...' :
         status === 'restarting' ? 'Restarting Server...' :
         'Server Disconnected'}
      </span>
      
      {/* Restart button */}
      {status === 'disconnected' && (
        <button
          onClick={handleRestartServer}
          style={{
            marginLeft: '8px',
            background: 'rgba(255, 255, 255, 0.2)',
            border: 'none',
            borderRadius: '3px',
            padding: '2px 6px',
            color: 'white',
            fontSize: '10px',
            cursor: 'pointer'
          }}
        >
          Restart
        </button>
      )}
      
      {/* Animation styles */}
      <style jsx>{`
        @keyframes pulse {
          0% { opacity: 0.4; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1); }
          100% { opacity: 0.4; transform: scale(0.8); }
        }
      `}</style>
    </div>
  );
};

export default ServerStatusComponent;