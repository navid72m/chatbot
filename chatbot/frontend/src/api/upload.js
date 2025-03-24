import api from './client';

const uploadApi = {
  /**
   * Upload a single file
   * @param {File} file - The file to upload
   * @param {Object} additionalData - Any additional data to send with the file
   * @param {Function} onProgress - Optional callback for upload progress
   * @returns {Promise} Upload result
   */
  uploadFile: async (file, additionalData = {}, onProgress = null) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // Add any additional data to the form
      Object.keys(additionalData).forEach(key => {
        formData.append(key, additionalData[key]);
      });
      
      const response = await api.post('/documents/upload', formData, {
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
      
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Upload multiple files
   * @param {Array<File>} files - Array of files to upload
   * @param {Object} additionalData - Any additional data to send with the files
   * @param {Function} onProgress - Optional callback for upload progress
   * @returns {Promise} Upload result
   */
  uploadMultipleFiles: async (files, additionalData = {}, onProgress = null) => {
    try {
      const formData = new FormData();
      
      // Add files to form data
      files.forEach((file, index) => {
        formData.append(`files[${index}]`, file);
      });
      
      // Add any additional data to the form
      Object.keys(additionalData).forEach(key => {
        formData.append(key, additionalData[key]);
      });
      
      const response = await api.post('/documents/upload-multiple', formData, {
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
      
      return response;
    } catch (error) {
      throw error;
    }
  }
};

export default uploadApi;