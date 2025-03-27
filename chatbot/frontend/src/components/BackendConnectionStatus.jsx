// frontend/src/components/BackendConnectionStatus.jsx
import React, { useState, useEffect } from 'react';
import { checkBackendAvailability } from '@/api/baseURL';

/**
 * Component to display backend connection status and provide a way to retry connection
 */
const BackendConnectionStatus = ({ onStatusChange }) => {
  const [connected, setConnected] = useState(false);
  const [checking, setChecking] = useState(true);
  const [retryCount, setRetryCount] = useState(0);

  // Check backend status when component mounts or retry is requested
  useEffect(() => {
    let isMounted = true;
    const checkConnection = async () => {
      setChecking(true);
      
      try {
        // Try to connect to backend
        const isAvailable = await checkBackendAvailability();
        
        if (isMounted) {
          setConnected(isAvailable);
          if (onStatusChange) {
            onStatusChange(isAvailable);
          }
        }
      } catch (error) {
        console.error('Error checking backend:', error);
        if (isMounted) {
          setConnected(false);
          if (onStatusChange) {
            onStatusChange(false);
          }
        }
      } finally {
        if (isMounted) {
          setChecking(false);
        }
      }
    };

    checkConnection();

    // Listen for backend status updates from Electron main process
    if (window.api) {
      const removeListener = window.api.receive('backend-status', (status) => {
        console.log('Received backend status from main process:', status);
        if (isMounted) {
          setConnected(status.started);
          if (onStatusChange) {
            onStatusChange(status.started);
          }
        }
      });
      
      // Cleanup listener
      return () => {
        isMounted = false;
        if (removeListener) removeListener();
      };
    }
    
    return () => {
      isMounted = false;
    };
  }, [retryCount, onStatusChange]);

  // Function to retry connection
  const handleRetry = () => {
    console.log('Retrying backend connection...');
    
    // If we're in Electron, we can try to restart the backend
    if (window.api) {
      window.api.send('restart-backend');
    }
    
    // Increment retry counter to trigger re-check
    setRetryCount(prev => prev + 1);
  };

  return (
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
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
      zIndex: 1000
    }}>
      <div style={{ 
        width: '8px', 
        height: '8px', 
        borderRadius: '50%', 
        backgroundColor: 'white',
        boxShadow: connected ? '0 0 5px #10b981' : 'none',
        animation: connected ? 'none' : checking ? 'pulse 1.5s infinite' : 'none'
      }}></div>
      
      <span>
        {checking 
          ? 'Checking Connection...'
          : connected 
            ? 'Connected to Server' 
            : 'Disconnected'}
      </span>
      
      {!connected && !checking && (
        <button 
          onClick={handleRetry}
          style={{
            marginLeft: '6px',
            backgroundColor: 'rgba(255, 255, 255, 0.2)',
            border: 'none',
            borderRadius: '4px',
            padding: '2px 6px',
            fontSize: '10px',
            color: 'white',
            cursor: 'pointer'
          }}
        >
          Retry
        </button>
      )}
      
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
      `}</style>
    </div>
  );
};

export default React.memo(BackendConnectionStatus);