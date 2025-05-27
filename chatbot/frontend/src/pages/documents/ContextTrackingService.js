// ContextTrackingService.js
import axios from 'axios';
import { getBackendURL } from '@/api/baseURL';

/**
 * Service for tracking context and source locations in documents
 */
class ContextTrackingService {
  constructor() {
    this.contextByResponseId = new Map();
  }

  /**
   * Process and store context information from a response
   * @param {Object} response - The API response with context tracking info
   * @param {string} messageId - The message ID to associate with the context
   * @returns {Object} - Enhanced message with context information
   */
  processResponseContext(response, messageId) {
    if (!response || !messageId) return null;
    
    let contextInfo = null;
    
    // Extract context tracking information
    if (response.context_tracking) {
      contextInfo = {
        responseId: response.context_tracking.response_id || messageId,
        contextRanges: response.context_tracking.context_ranges || [],
        contextText: response.context_tracking.context_text || "",
      };
      
      // Store in our local cache
      this.contextByResponseId.set(messageId, contextInfo);
    }
    
    return contextInfo;
  }

  /**
   * Get context information for a message
   * @param {string} messageId - The message ID
   * @returns {Object|null} - Context information or null
   */
  getContextForMessage(messageId) {
    return this.contextByResponseId.get(messageId) || null;
  }

  /**
   * Fetch context information from the server
   * @param {string} documentName - Document name
   * @param {string} queryId - Query message ID
   * @param {string} responseId - Response message ID
   * @returns {Promise<Object>} - Context information
   */
  async fetchContextInfo(documentName, queryId, responseId) {
    try {
      // First check our local cache
      const cachedContext = this.getContextForMessage(responseId);
      if (cachedContext) {
        return cachedContext;
      }
      
      // Fetch from server if not in cache
      const response = await axios.get(`${getBackendURL()}/context-info`, {
        params: {
          document: documentName,
          query_id: queryId,
          response_id: responseId
        }
      });
      
      if (response.data) {
        const contextInfo = {
          responseId: responseId,
          contextRanges: response.data.context_ranges || [],
          contextText: response.data.context_text || "",
        };
        
        // Store in our local cache
        this.contextByResponseId.set(responseId, contextInfo);
        
        return contextInfo;
      }
      
      return null;
    } catch (error) {
      console.error('Error fetching context info:', error);
      return null;
    }
  }

  /**
   * Clear context for a specific message
   * @param {string} messageId - The message ID
   */
  clearContext(messageId) {
    if (this.contextByResponseId.has(messageId)) {
      this.contextByResponseId.delete(messageId);
    }
  }

  /**
   * Clear all stored context information
   */
  clearAllContext() {
    this.contextByResponseId.clear();
  }
}

// Create and export a singleton instance
const contextTrackingService = new ContextTrackingService();
export default contextTrackingService;