// src/api/auth.js

import api from './client';

const authApi = {
  /**
   * User login
   * @param {string} email - User email
   * @param {string} password - User password
   * @returns {Promise} Auth tokens and user data
   */
  login: async (email, password) => {
    try {
      const response = await api.post('/auth/login', { email, password });
      
      // Store tokens in localStorage
      localStorage.setItem('auth_token', response.access_token);
      localStorage.setItem('refresh_token', response.refresh_token);
      
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * User logout
   * @returns {Promise} Success status
   */
  logout: async () => {
    try {
      // Send logout request to invalidate token on server
      const response = await api.post('/auth/logout');
      
      // Remove tokens from localStorage
      localStorage.removeItem('auth_token');
      localStorage.removeItem('refresh_token');
      
      return response;
    } catch (error) {
      // Even if the request fails, remove tokens
      localStorage.removeItem('auth_token');
      localStorage.removeItem('refresh_token');
      throw error;
    }
  },
  
  /**
   * User registration
   * @param {Object} userData - User registration data
   * @returns {Promise} Registration result
   */
  register: async (userData) => {
    try {
      return await api.post('/auth/register', userData);
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Request password reset
   * @param {string} email - User email
   * @returns {Promise} Success status
   */
  forgotPassword: async (email) => {
    try {
      return await api.post('/auth/forgot-password', { email });
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Reset password with token
   * @param {string} token - Reset token from email
   * @param {string} password - New password
   * @returns {Promise} Success status
   */
  resetPassword: async (token, password) => {
    try {
      return await api.post('/auth/reset-password', { token, password });
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Get current user profile
   * @returns {Promise} User profile data
   */
  getCurrentUser: async () => {
    try {
      return await api.get('/auth/me');
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Update user profile
   * @param {Object} profileData - Profile data to update
   * @returns {Promise} Updated profile
   */
  updateProfile: async (profileData) => {
    try {
      return await api.put('/auth/profile', profileData);
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Change password
   * @param {string} currentPassword - Current password
   * @param {string} newPassword - New password
   * @returns {Promise} Success status
   */
  changePassword: async (currentPassword, newPassword) => {
    try {
      return await api.post('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Verify email with token
   * @param {string} token - Email verification token
   * @returns {Promise} Success status
   */
  verifyEmail: async (token) => {
    try {
      return await api.post('/auth/verify-email', { token });
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Refresh access token
   * @param {string} refreshToken - Refresh token
   * @returns {Promise} New auth tokens
   */
  refreshToken: async (refreshToken) => {
    try {
      const response = await api.post('/auth/refresh', { refresh_token: refreshToken });
      
      // Store new tokens in localStorage
      localStorage.setItem('auth_token', response.access_token);
      localStorage.setItem('refresh_token', response.refresh_token);
      
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Check if user is authenticated
   * @returns {boolean} Authentication status
   */
  isAuthenticated: () => {
    const token = localStorage.getItem('auth_token');
    return !!token;
  },
};

export default authApi;