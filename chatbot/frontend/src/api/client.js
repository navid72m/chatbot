// src/api/client.js

import axios from 'axios';

// Get the API base URL from environment variables
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Request interceptor for adding auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling common errors
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    
    // Handle 401 - Unauthorized errors (token expired)
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      // Try to refresh the token
      const refreshToken = localStorage.getItem('refresh_token');
      if (refreshToken) {
        try {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken
          });
          
          // Store the new tokens
          const { access_token, refresh_token } = response.data;
          localStorage.setItem('auth_token', access_token);
          localStorage.setItem('refresh_token', refresh_token);
          
          // Retry the original request with new token
          originalRequest.headers['Authorization'] = `Bearer ${access_token}`;
          return apiClient(originalRequest);
        } catch (refreshError) {
          // If refresh fails, redirect to login
          localStorage.removeItem('auth_token');
          localStorage.removeItem('refresh_token');
          window.location.href = '/login';
          return Promise.reject(refreshError);
        }
      } else {
        // No refresh token, redirect to login
        window.location.href = '/login';
      }
    }
    
    // Handle 403 - Forbidden errors (permissions)
    if (error.response?.status === 403) {
      // Dispatch global permission denied notification
      console.error('Permission denied');
      // You could dispatch to a global state: store.dispatch(showNotification('Permission denied'));
    }
    
    // Handle 500 - Server errors
    if (error.response?.status >= 500) {
      console.error('Server error', error.response?.data);
      // You could dispatch to a global state: store.dispatch(showNotification('Server error occurred'));
    }
    
    return Promise.reject(error);
  }
);

// Helper methods for common API operations
const api = {
  // GET request
  get: async (url, params = {}) => {
    try {
      const response = await apiClient.get(url, { params });
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  // POST request
  post: async (url, data = {}) => {
    try {
      const response = await apiClient.post(url, data);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  // PUT request
  put: async (url, data = {}) => {
    try {
      const response = await apiClient.put(url, data);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  // PATCH request
  patch: async (url, data = {}) => {
    try {
      const response = await apiClient.patch(url, data);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  // DELETE request
  delete: async (url) => {
    try {
      const response = await apiClient.delete(url);
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  // Upload file(s)
  upload: async (url, files, additionalData = {}, onProgress = null) => {
    try {
      const formData = new FormData();
      
      // Add files to form data
      if (Array.isArray(files)) {
        files.forEach((file, index) => {
          formData.append(`file${index}`, file);
        });
      } else {
        formData.append('file', files);
      }
      
      // Add additional data to form data
      Object.keys(additionalData).forEach(key => {
        formData.append(key, additionalData[key]);
      });
      
      const response = await apiClient.post(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: onProgress
          ? (progressEvent) => {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              onProgress(percentCompleted);
            }
          : undefined,
      });
      
      return response.data;
    } catch (error) {
      throw error;
    }
  },
  
  // Download file
  download: async (url, filename, params = {}) => {
    try {
      const response = await apiClient.get(url, {
        params,
        responseType: 'blob',
      });
      
      // Create download link and trigger download
      const downloadUrl = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      return true;
    } catch (error) {
      throw error;
    }
  },
};

export default api;