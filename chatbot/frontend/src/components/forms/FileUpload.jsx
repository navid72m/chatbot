import React, { useState, useRef } from 'react';
import PropTypes from 'prop-types';
import uploadApi from '../../api/upload';

const FileUpload = ({
  multiple = false,
  accept = '*/*',
  maxSize = 10 * 1024 * 1024, // 10MB default
  onUploadSuccess,
  onUploadError,
  additionalData = {},
  className = '',
}) => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  
  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    
    // Check file size
    const oversizedFiles = selectedFiles.filter(file => file.size > maxSize);
    if (oversizedFiles.length > 0) {
      setError(`File(s) too large. Maximum size is ${maxSize / (1024 * 1024)}MB.`);
      return;
    }
    
    setFiles(selectedFiles);
    setError(null);
  };
  
  // Handle file upload
  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select a file to upload.');
      return;
    }
    
    try {
      setUploading(true);
      setUploadProgress(0);
      setError(null);
      
      let response;
      
      // Upload progress callback
      const onProgress = (progress) => {
        setUploadProgress(progress);
      };
      
      if (multiple) {
        response = await uploadApi.uploadMultipleFiles(files, additionalData, onProgress);
      } else {
        response = await uploadApi.uploadFile(files[0], additionalData, onProgress);
      }
      
      // Call success callback with response
      if (onUploadSuccess) {
        onUploadSuccess(response);
      }
      
      // Reset form
      setFiles([]);
      setUploadProgress(0);
      
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.message || 'Failed to upload file(s). Please try again.');
      
      if (onUploadError) {
        onUploadError(err);
      }
    } finally {
      setUploading(false);
    }
  };
  
  // Handle drag and drop
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      
      // If not multiple, take only the first file
      const filesToProcess = multiple ? droppedFiles : [droppedFiles[0]];
      
      // Check file size
      const oversizedFiles = filesToProcess.filter(file => file.size > maxSize);
      if (oversizedFiles.length > 0) {
        setError(`File(s) too large. Maximum size is ${maxSize / (1024 * 1024)}MB.`);
        return;
      }
      
      setFiles(filesToProcess);
      setError(null);
    }
  };
  
  return (
    <div className={`file-upload ${className}`}>
      <div 
        className={`upload-dropzone ${files.length > 0 ? 'has-files' : ''}`}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <input
          type="file"
          ref={fileInputRef}
          className="file-input"
          onChange={handleFileChange}
          accept={accept}
          multiple={multiple}
          disabled={uploading}
        />
        
        {files.length === 0 ? (
          <div className="upload-placeholder">
            <div className="upload-icon">📁</div>
            <p>Drag & drop files here or click to browse</p>
            <span className="upload-hint">
              {multiple ? 'You can upload multiple files' : 'You can upload a single file'}
            </span>
          </div>
        ) : (
          <div className="selected-files">
            <h4>Selected Files:</h4>
            <ul className="file-list">
              {files.map((file, index) => (
                <li key={index} className="file-item">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">({(file.size / 1024).toFixed(2)} KB)</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      
      {error && (
        <div className="upload-error">
          {error}
        </div>
      )}
      
      {uploading && (
        <div className="upload-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
          <span className="progress-text">{uploadProgress}%</span>
        </div>
      )}
      
      <div className="upload-actions">
        <button 
          className="upload-button" 
          onClick={handleUpload}
          disabled={files.length === 0 || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
        
        <button 
          className="cancel-button" 
          onClick={() => {
            setFiles([]);
            setError(null);
            if (fileInputRef.current) {
              fileInputRef.current.value = '';
            }
          }}
          disabled={files.length === 0 || uploading}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

FileUpload.propTypes = {
  multiple: PropTypes.bool,
  accept: PropTypes.string,
  maxSize: PropTypes.number,
  onUploadSuccess: PropTypes.func,
  onUploadError: PropTypes.func,
  additionalData: PropTypes.object,
  className: PropTypes.string,
};

export default FileUpload;