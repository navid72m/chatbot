import React, { useState, useEffect, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import axios from 'axios';

import { getBackendURL } from '@/api/baseURL';
import './PDFContextViewer.css';
// import { pdfjs } from 'react-pdf';
// Set the worker source for react-pdf
// import { pdfjs } from 'react-pdf';
// Vite-compatible workaround
// import { pdfjs } from 'react-pdf';
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.js';




const PDFContextViewer = ({ 
  documentName, 
  currentQuery, 
  currentAnswer,
  currentMessage 
}) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [scale, setScale] = useState(1.2);
  const [pdfFile, setPdfFile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [contextRanges, setContextRanges] = useState([]);
  const [highlightedText, setHighlightedText] = useState('');
  const [activeHighlightIndex, setActiveHighlightIndex] = useState(0);
  const [viewerMode, setViewerMode] = useState('context'); // 'context' or 'full'
  
  const documentRef = useRef(null);
  const pageRefs = useRef({});
  const highlightRefs = useRef({});

  // Function to fetch the PDF file from the backend
  const fetchPDF = async () => {
    if (!documentName) {
      setError('No document selected');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // For PDF files
      if (documentName.toLowerCase().endsWith('.pdf')) {
        const response = await axios.get(`${getBackendURL()}/document/${documentName}`, {
          responseType: 'blob',
          timeout: 30000
        });
        
        const pdfBlob = new Blob([response.data], { type: 'application/pdf' });
        const pdfUrl = URL.createObjectURL(pdfBlob);
        setPdfFile(pdfUrl);
      } else {
        setError('Only PDF documents are supported for preview');
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching document:', err);
      setError(`Error loading document: ${err.message}`);
      setLoading(false);
    }
  };

  // Function to fetch context information from the backend
  const fetchContextInfo = async () => {
    if (!currentMessage) return;
    
    try {
      // Either use the document context directly from the message or fetch it
      if (currentMessage.contextRanges) {
        setContextRanges(currentMessage.contextRanges);
        setHighlightedText(currentMessage.contextText || '');
      } else if (currentMessage.queryRewritingInfo) {
        const response = await axios.get(`${getBackendURL()}/context-info`, {
          params: {
            document: documentName,
            query_id: currentMessage.id,
            response_id: currentMessage.id
          }
        });
        
        if (response.data && response.data.context_ranges) {
          setContextRanges(response.data.context_ranges);
          setHighlightedText(response.data.context_text || '');
        }
      }
    } catch (err) {
      console.error('Error fetching context info:', err);
      setContextRanges([]);
    }
  };

  // Load PDF when document name changes
  useEffect(() => {
    if (documentName) {
      fetchPDF();
    }
  }, [documentName]);

  // Fetch context information when message changes
  useEffect(() => {
    if (currentMessage && documentName) {
      fetchContextInfo();
    }
  }, [currentMessage, documentName]);

  // Handle document load success
  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setPageNumber(1);
    setLoading(false);
  };

  // Handle document load error
  const onDocumentLoadError = (error) => {
    console.error('Error loading PDF document:', error);
    setError(`Error loading PDF: ${error.message}`);
    setLoading(false);
  };

  // Zoom functions
  const zoomIn = () => setScale(prevScale => Math.min(prevScale + 0.2, 3));
  const zoomOut = () => setScale(prevScale => Math.max(prevScale - 0.2, 0.6));
  const resetZoom = () => setScale(1.2);

  // Page navigation
  const goToPrevPage = () => {
    if (pageNumber > 1) {
      setPageNumber(pageNumber - 1);
    }
  };

  const goToNextPage = () => {
    if (pageNumber < numPages) {
      setPageNumber(pageNumber + 1);
    }
  };

  // Set reference to page element
  const setPageRef = (pageNumber, ref) => {
    pageRefs.current[pageNumber] = ref;
  };

  // Navigate to specific context highlight
  const navigateToHighlight = (index) => {
    if (!contextRanges || contextRanges.length === 0) return;
    
    const highlight = contextRanges[index];
    if (!highlight) return;
    
    // Set page number
    setPageNumber(highlight.page);
    setActiveHighlightIndex(index);
    
    // Scroll to highlight after page renders
    setTimeout(() => {
      const highlightElement = highlightRefs.current[`highlight-${index}`];
      if (highlightElement) {
        highlightElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 300);
  };

  // Navigate to next/previous highlight
  const nextHighlight = () => {
    if (activeHighlightIndex < contextRanges.length - 1) {
      navigateToHighlight(activeHighlightIndex + 1);
    }
  };

  const prevHighlight = () => {
    if (activeHighlightIndex > 0) {
      navigateToHighlight(activeHighlightIndex - 1);
    }
  };

  // Set highlight element reference
  const setHighlightRef = (id, ref) => {
    highlightRefs.current[id] = ref;
  };

  // Custom page renderer with highlights
  const renderPage = ({ pageNumber }) => {
    // Filter highlights for current page
    const pageHighlights = contextRanges.filter(range => range.page === pageNumber);
    
    return (
      <div 
        className="pdf-page-container" 
        ref={(ref) => setPageRef(pageNumber, ref)}
      >
        <Page 
          pageNumber={pageNumber} 
          scale={scale} 
          renderTextLayer={true}
          renderAnnotationLayer={true}
        />
        
        {/* Render highlight overlays */}
        <div className="highlight-container">
          {pageHighlights.map((highlight, index) => {
            const globalIndex = contextRanges.findIndex(r => 
              r.page === highlight.page && 
              r.top === highlight.top && 
              r.left === highlight.left
            );
            
            return (
              <div
                key={`highlight-${globalIndex}`}
                ref={(ref) => setHighlightRef(`highlight-${globalIndex}`, ref)}
                className={`highlight-overlay ${globalIndex === activeHighlightIndex ? 'active' : ''}`}
                style={{
                  top: `${highlight.top * scale}px`,
                  left: `${highlight.left * scale}px`,
                  width: `${highlight.width * scale}px`,
                  height: `${highlight.height * scale}px`
                }}
                onClick={() => setActiveHighlightIndex(globalIndex)}
              >
                <span className="highlight-number">{globalIndex + 1}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Toggle viewer mode between context view and full document view
  const toggleViewerMode = () => {
    setViewerMode(viewerMode === 'context' ? 'full' : 'context');
  };

  return (
    <div className="pdf-viewer-container">
      <div className="pdf-viewer-header">
        <h3>Document Context Viewer</h3>
        <div className="viewer-mode-toggle">
          <button 
            className={`mode-button ${viewerMode === 'context' ? 'active' : ''}`}
            onClick={() => setViewerMode('context')}
          >
            Context View
          </button>
          <button 
            className={`mode-button ${viewerMode === 'full' ? 'active' : ''}`}
            onClick={() => setViewerMode('full')}
          >
            Full Document
          </button>
        </div>
      </div>
      
      {/* PDF Context Summary */}
      {viewerMode === 'context' && contextRanges.length > 0 && (
        <div className="context-summary">
          <h4>Answer Context</h4>
          <p className="context-description">
            The answer was generated using the following content from {documentName}:
          </p>
          
          <div className="context-text">
            {highlightedText && (
              <div className="context-text-container">
                <p>{highlightedText}</p>
              </div>
            )}
            
            <div className="context-navigation">
              <button onClick={prevHighlight} disabled={activeHighlightIndex === 0}>
                Previous Context
              </button>
              <span className="context-count">
                {activeHighlightIndex + 1} of {contextRanges.length}
              </span>
              <button onClick={nextHighlight} disabled={activeHighlightIndex === contextRanges.length - 1}>
                Next Context
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Main PDF Viewer */}
      <div className="pdf-document-container">
        {loading && (
          <div className="pdf-loading">
            <div className="loader"></div>
            <p>Loading document...</p>
          </div>
        )}
        
        {error && (
          <div className="pdf-error">
            <p>{error}</p>
          </div>
        )}
        
        {!loading && !error && pdfFile && (
          <>
            <div className="pdf-controls">
              <div className="page-navigation">
                <button onClick={goToPrevPage} disabled={pageNumber <= 1}>
                  Previous
                </button>
                <span className="page-info">
                  Page {pageNumber} of {numPages}
                </span>
                <button onClick={goToNextPage} disabled={pageNumber >= numPages}>
                  Next
                </button>
              </div>
              
              <div className="zoom-controls">
                <button onClick={zoomOut}>-</button>
                <button onClick={resetZoom}>Reset</button>
                <button onClick={zoomIn}>+</button>
              </div>
              
              {contextRanges.length > 0 && viewerMode === 'full' && (
                <div className="highlight-navigation">
                  <button 
                    className="highlight-button"
                    onClick={() => navigateToHighlight(activeHighlightIndex)}
                  >
                    Jump to Context #{activeHighlightIndex + 1}
                  </button>
                </div>
              )}
            </div>
            
            <div className="pdf-viewer" ref={documentRef}>
              <Document
                file={pdfFile}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
                loading={<div className="pdf-loading"><div className="loader"></div></div>}
              >
                {renderPage({ pageNumber })}
              </Document>
            </div>
          </>
        )}
        
        {!loading && !error && !pdfFile && documentName && !documentName.toLowerCase().endsWith('.pdf') && (
          <div className="non-pdf-message">
            <p>The current document ({documentName}) is not a PDF file.</p>
            <p>PDF preview is only available for PDF documents.</p>
          </div>
        )}
        
        {!loading && !error && !pdfFile && !documentName && (
          <div className="no-document-message">
            <p>No document selected.</p>
            <p>Please upload a PDF document to view it here.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PDFContextViewer;