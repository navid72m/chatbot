// src/context/AuthContext.jsx

import React, { createContext, useState, useEffect, useCallback, useMemo } from 'react';
import authApi from '../api/auth';

// Create context
export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Load user on initial render if token exists
  useEffect(() => {
    const loadUser = async () => {
      try {
        // Check if token exists
        if (authApi.isAuthenticated()) {
          setLoading(true);
          const userData = await authApi.getCurrentUser();
          setUser(userData);
        }
      } catch (err) {
        console.error('Failed to load user:', err);
        setError('Failed to authenticate user.');
        // Clear tokens if authentication fails
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
      } finally {
        setLoading(false);
      }
    };
    
    loadUser();
  }, []);
  
  // Login function
  const login = useCallback(async (email, password) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await authApi.login(email, password);
      setUser(response.user);
      
      return response;
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to login.');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Logout function
  const logout = useCallback(async () => {
    try {
      setLoading(true);
      await authApi.logout();
      setUser(null);
    } catch (err) {
      console.error('Logout error:', err);
      // Still clear user data even if API call fails
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Register function
  const register = useCallback(async (userData) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await authApi.register(userData);
      
      // Auto login if backend returns tokens
      if (response.access_token) {
        setUser(response.user);
      }
      
      return response;
    } catch (err) {
      setError(err.response?.data?.message || 'Registration failed.');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Update profile function
  const updateProfile = useCallback(async (profileData) => {
    try {
      setLoading(true);
      setError(null);
      
      const updatedUser = await authApi.updateProfile(profileData);
      setUser(updatedUser);
      
      return updatedUser;
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to update profile.');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);
  
  // Check permission function
  const hasPermission = useCallback((permission) => {
    if (!user || !user.permissions) return false;
    
    // Check if user has specific permission
    return user.permissions.includes(permission);
  }, [user]);
  
  // Check role function
  const hasRole = useCallback((role) => {
    if (!user || !user.roles) return false;
    
    // Check if user has specific role
    return user.roles.includes(role);
  }, [user]);
  
  // Memoize context value to prevent unnecessary renders
  const contextValue = useMemo(() => ({
    user,
    loading,
    error,
    isAuthenticated: !!user,
    login,
    logout,
    register,
    updateProfile,
    hasPermission,
    hasRole,
  }), [user, loading, error, login, logout, register, updateProfile, hasPermission, hasRole]);
  
  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook for using auth context
export const useAuth = () => {
  const context = React.useContext(AuthContext);
  
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  return context;
};