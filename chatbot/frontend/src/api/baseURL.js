// frontend/src/api/baseURL.js
import { isElectron } from '../utils/environment';

/**
 * Returns the appropriate backend URL based on the environment
 * In Electron, we use localhost since the backend is spawned locally
 * In browser dev mode, we might use a different URL or proxy
 */
export const getBackendURL = () => {
  // Always use localhost in Electron
  if (isElectron()) {
    return 'http://localhost:8000';
  }
  
  // For browser development
  if (process.env.NODE_ENV === 'development') {
    return 'http://localhost:8000';  // Development backend URL
  }
  
  // For production browser deployment (if applicable)
  return '/api';  // Typically you'd use a relative path in production
};

/**
 * Checks if the backend is available by making a simple request
 * @returns {Promise<boolean>} True if backend is available
 */
export const checkBackendAvailability = async () => {
  try {
    const response = await fetch(`${getBackendURL()}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
      // Small timeout to quickly detect if backend is available
      signal: AbortSignal.timeout(2000)
    });
    return response.ok;
  } catch (error) {
    console.error('Backend availability check failed:', error);
    return false;
  }
};