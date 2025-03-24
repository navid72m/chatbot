// src/context/NotificationContext.jsx

import React, { createContext, useState, useCallback, useEffect } from 'react';
import PropTypes from 'prop-types';
import { createPortal } from 'react-dom';
import { FaCheck, FaInfo, FaExclamationTriangle, FaTimes, FaExclamationCircle } from 'react-icons/fa';

// Notification types
export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info',
};

// Default auto-close durations for each type
const AUTO_CLOSE_DURATIONS = {
  [NOTIFICATION_TYPES.SUCCESS]: 5000, // 5 seconds
  [NOTIFICATION_TYPES.ERROR]: 0, // Don't auto-close errors
  [NOTIFICATION_TYPES.WARNING]: 8000, // 8 seconds
  [NOTIFICATION_TYPES.INFO]: 5000, // 5 seconds
};

// Icons for each notification type
const NOTIFICATION_ICONS = {
  [NOTIFICATION_TYPES.SUCCESS]: <FaCheck />,
  [NOTIFICATION_TYPES.ERROR]: <FaExclamationCircle />,
  [NOTIFICATION_TYPES.WARNING]: <FaExclamationTriangle />,
  [NOTIFICATION_TYPES.INFO]: <FaInfo />,
};

// Create context
export const NotificationContext = createContext(null);

// Generate unique ID for each notification
const generateId = () => `notification-${Date.now()}-${Math.floor(Math.random() * 1000)}`;

// Notification Provider
export const NotificationProvider = ({ children, position = 'top-right', maxNotifications = 5 }) => {
  const [notifications, setNotifications] = useState([]);
  
  // Remove notification by ID
  const removeNotification = useCallback((id) => {
    setNotifications((prevNotifications) =>
      prevNotifications.filter((notification) => notification.id !== id)
    );
  }, []);
  
  // Auto-close notifications
  useEffect(() => {
    notifications.forEach((notification) => {
      if (notification.autoClose && !notification.timer) {
        const timer = setTimeout(() => {
          removeNotification(notification.id);
        }, notification.duration);
        
        // Add timer to notification
        setNotifications((prevNotifications) =>
          prevNotifications.map((n) =>
            n.id === notification.id ? { ...n, timer } : n
          )
        );
      }
    });
    
    // Clean up timers
    return () => {
      notifications.forEach((notification) => {
        if (notification.timer) {
          clearTimeout(notification.timer);
        }
      });
    };
  }, [notifications, removeNotification]);
  
  // Add notification
  const addNotification = useCallback(
    (type, message, options = {}) => {
      const id = generateId();
      
      // Determine auto-close duration
      const autoClose = options.autoClose !== undefined
        ? options.autoClose
        : AUTO_CLOSE_DURATIONS[type] > 0;
      
      const duration = options.duration || AUTO_CLOSE_DURATIONS[type] || 5000;
      
      // Create notification object
      const notification = {
        id,
        type,
        message,
        icon: options.icon || NOTIFICATION_ICONS[type],
        autoClose,
        duration,
        timer: null,
        title: options.title,
        onClose: options.onClose,
      };
      
      // Add notification to state
      setNotifications((prevNotifications) => {
        // Limit number of notifications
        const filteredNotifications = [...prevNotifications]
          .slice(0, maxNotifications - 1);
        
        return [notification, ...filteredNotifications];
      });
      
      return id;
    },
    [maxNotifications]
  );
  
  // Helper methods for each type
  const showSuccess = useCallback(
    (message, options = {}) => addNotification(NOTIFICATION_TYPES.SUCCESS, message, options),
    [addNotification]
  );
  
  const showError = useCallback(
    (message, options = {}) => addNotification(NOTIFICATION_TYPES.ERROR, message, options),
    [addNotification]
  );
  
  const showWarning = useCallback(
    (message, options = {}) => addNotification(NOTIFICATION_TYPES.WARNING, message, options),
    [addNotification]
  );
  
  const showInfo = useCallback(
    (message, options = {}) => addNotification(NOTIFICATION_TYPES.INFO, message, options),
    [addNotification]
  );
  
  // Clear all notifications
  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);
  
  // Handle notification close
  const handleClose = useCallback(
    (notification) => {
      // Call onClose callback if provided
      if (notification.onClose) {
        notification.onClose();
      }
      
      // Clear timer if exists
      if (notification.timer) {
        clearTimeout(notification.timer);
      }
      
      // Remove notification
      removeNotification(notification.id);
    },
    [removeNotification]
  );
  
  // Context value
  const contextValue = {
    notifications,
    addNotification,
    removeNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    clearAll,
  };
  
  // Position classes
  const getPositionClasses = () => {
    switch (position) {
      case 'top-right':
        return 'notification-container--top-right';
      case 'top-left':
        return 'notification-container--top-left';
      case 'bottom-right':
        return 'notification-container--bottom-right';
      case 'bottom-left':
        return 'notification-container--bottom-left';
      case 'top-center':
        return 'notification-container--top-center';
      case 'bottom-center':
        return 'notification-container--bottom-center';
      default:
        return 'notification-container--top-right';
    }
  };
  
  return (
    <NotificationContext.Provider value={contextValue}>
      {children}
      
      {createPortal(
        <div className={`notification-container ${getPositionClasses()}`}>
          {notifications.map((notification) => (
            <div
              key={notification.id}
              className={`notification notification--${notification.type}`}
            >
              <div className="notification__icon">{notification.icon}</div>
              
              <div className="notification__content">
                {notification.title && (
                  <div className="notification__title">{notification.title}</div>
                )}
                <div className="notification__message">{notification.message}</div>
              </div>
              
              <button
                className="notification__close"
                onClick={() => handleClose(notification)}
                aria-label="Close notification"
              >
                <FaTimes />
              </button>
              
              {notification.autoClose && (
                <div
                  className="notification__progress-bar"
                  style={{
                    animationDuration: `${notification.duration}ms`,
                  }}
                />
              )}
            </div>
          ))}
        </div>,
        document.body
      )}
    </NotificationContext.Provider>
  );
};

NotificationProvider.propTypes = {
  children: PropTypes.node.isRequired,
  position: PropTypes.oneOf([
    'top-right',
    'top-left',
    'bottom-right',
    'bottom-left',
    'top-center',
    'bottom-center',
  ]),
  maxNotifications: PropTypes.number,
};

// Custom hook for using notifications
export const useNotification = () => {
  const context = React.useContext(NotificationContext);
  
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  
  return context;
};