// documentContextAPI.js
import axios from 'axios';
import { getBackendURL } from '@/api/baseURL';

/**
 * Fetch document content by name
 * @param {string} documentName - Name of the document to fetch
 * @returns {Promise<Blob>} - Document content as a blob
 */
export const fetchDocument = async (documentName) => {
  try {
    const response = await axios.get(`${getBackendURL()}/document/${documentName}`, {
      responseType: 'blob',
      timeout: 30000
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching document:', error);
    throw error;
  }
};

/**
 * Fetch context information for a specific answer
 * @param {string} documentName - Name of the document
 * @param {string} queryId - ID of the query message
 * @param {string} responseId - ID of the response message
 * @returns {Promise<Object>} - Context information
 */
export const fetchContextInfo = async (documentName, queryId, responseId) => {
  try {
    const response = await axios.get(`${getBackendURL()}/context-info`, {
      params: {
        document: documentName,
        query_id: queryId,
        response_id: responseId
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching context info:', error);
    throw error;
  }
};

/**
 * Extract PDF text from a specific page range
 * @param {string} documentName - Name of the PDF document
 * @param {number} pageNumber - Page number to extract from
 * @param {Object} coords - Coordinates of the text to extract (top, left, width, height)
 * @returns {Promise<Object>} - Extracted text and metadata
 */
export const extractPageText = async (documentName, pageNumber, coords) => {
  try {
    const response = await axios.post(`${getBackendURL()}/extract-text`, {
      document: documentName,
      page: pageNumber,
      coords: coords
    });
    
    return response.data;
  } catch (error) {
    console.error('Error extracting page text:', error);
    throw error;
  }
};

/**
 * Get all document highlights for a specific response
 * @param {string} documentName - Name of the document
 * @param {string} responseId - ID of the response message
 * @returns {Promise<Array>} - Array of highlights
 */
export const getDocumentHighlights = async (documentName, responseId) => {
  try {
    const response = await axios.get(`${getBackendURL()}/document-highlights`, {
      params: {
        document: documentName,
        response_id: responseId
      }
    });
    
    return response.data.highlights || [];
  } catch (error) {
    console.error('Error getting document highlights:', error);
    throw error;
  }
};

/**
 * Enhanced query with context tracking
 * @param {Object} queryParams - Query parameters
 * @returns {Promise<Object>} - Query response with context information
 */
export const queryWithContextTracking = async (queryParams) => {
  try {
    const response = await axios.post(`${getBackendURL()}/query-with-context`, queryParams, {
      timeout: 300000 // 5 minute timeout
    });
    
    return response.data;
  } catch (error) {
    console.error('Error in query with context tracking:', error);
    throw error;
  }
};