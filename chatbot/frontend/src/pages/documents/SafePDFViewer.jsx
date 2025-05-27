// SafePDFViewer.jsx - Updated to avoid HEAD request issues
import React, { useState, useRef, useEffect } from 'react';

const SafePDFViewer = ({ 
  pdfUrl, 
  highlightedChunks = [], 
  currentPage = 1, 
  onPageChange,
  onTextSelection,
  className = '' 
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [scale, setScale] = useState(1.0);
  const [pdfLoaded, setPdfLoaded] = useState(false);
  const iframeRef = useRef(null);

  // Check if PDF URL is valid using the exists endpoint
  useEffect(() => {
    if (pdfUrl) {
      setLoading(true);
      setError(null);
      
      // Extract filename from URL for existence check
      const filename = pdfUrl.split('/').pop();
      const baseUrl = pdfUrl.substring(0, pdfUrl.lastIndexOf('/'));
      
      // Use the /exists endpoint instead of HEAD request
      fetch(`${baseUrl}/${filename}/exists`)
        .then(response => response.json())
        .then(data => {
          if (data.exists) {
            console.log('PDF file exists, loading viewer...');
            setLoading(false);
            setPdfLoaded(true);
          } else {
            throw new Error(data.error || 'PDF file not found');
          }
        })
        .catch(err => {
          console.error('PDF existence check failed, trying direct load:', err);
          // Fallback: try to load directly without checking
          setLoading(false);
          setPdfLoaded(true);
        });
    }
  }, [pdfUrl]);

  // Handle text selection from iframe (if possible)
  const handleTextSelection = () => {
    try {
      const selection = window.getSelection();
      if (selection.toString().trim() && onTextSelection) {
        onTextSelection({
          text: selection.toString(),
          page: currentPage
        });
      }
    } catch (error) {
      console.log('Text selection not available in PDF iframe');
    }
  };

  // Zoom controls
  const zoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 2.0));
  };
  
  const zoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.5));
  };
  
  const resetZoom = () => {
    setScale(1.0);
  };

  // Get current page highlights
  const getCurrentPageHighlights = () => {
    return highlightedChunks.filter(chunk => 
      chunk.page_number === currentPage
    );
  };

  // Handle iframe load success
  const handleIframeLoad = () => {
    console.log('PDF iframe loaded successfully');
    setLoading(false);
    setError(null);
  };

  // Handle iframe load error
  const handleIframeError = () => {
    console.error('PDF iframe failed to load');
    setError('Failed to display PDF. The file may be corrupted or the server may be unavailable.');
  };

  if (loading) {
    return (
      <div className={`pdf-viewer-container ${className}`}>
        <div className="pdf-loading">
          <div className="loading-spinner"></div>
          <p>Loading PDF document...</p>
          <p className="loading-details">
            📄 {pdfUrl ? pdfUrl.split('/').pop() : 'Unknown file'}
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`pdf-viewer-container ${className}`}>
        <div className="pdf-error">
          <h3>❌ PDF Loading Error</h3>
          <p>{error}</p>
          <div className="error-details">
            <p><strong>PDF URL:</strong> {pdfUrl}</p>
            <p><strong>Possible solutions:</strong></p>
            <ul>
              <li>✅ Check if the backend server is running on port 8000</li>
              <li>✅ Verify the file was uploaded successfully</li>
              <li>✅ Make sure the /files/ endpoint supports HEAD requests</li>
              <li>✅ Try refreshing the page</li>
            </ul>
          </div>
          <div className="error-actions">
            <button onClick={() => {
              setError(null);
              setLoading(true);
              setPdfLoaded(true);
            }}>
              🔄 Retry Loading
            </button>
            <button onClick={() => window.open(pdfUrl, '_blank')}>
              🔗 Open PDF in New Tab
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!pdfUrl) {
    return (
      <div className={`pdf-viewer-container ${className}`}>
        <div className="pdf-error">
          <p>❌ No PDF URL provided</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`pdf-viewer-container ${className}`}>
      {/* PDF Controls */}
      <div className="pdf-controls">
        <div className="pdf-info">
          <span className="pdf-filename">
            📄 {pdfUrl.split('/').pop()?.replace(/%20/g, ' ')}
          </span>
        </div>

        <div className="zoom-controls">
          <button onClick={zoomOut} className="control-btn" title="Zoom out">
            🔍-
          </button>
          <span className="zoom-info">{Math.round(scale * 100)}%</span>
          <button onClick={zoomIn} className="control-btn" title="Zoom in">
            🔍+
          </button>
          <button onClick={resetZoom} className="control-btn" title="Reset zoom">
            Reset
          </button>
        </div>

        <div className="pdf-actions">
          <button 
            onClick={() => window.open(pdfUrl, '_blank')} 
            className="control-btn"
            title="Open in new tab"
          >
            🔗 Open
          </button>
        </div>

        {highlightedChunks.length > 0 && (
          <div className="highlight-info">
            <span className="highlight-count">
              🔍 {highlightedChunks.length} relevant sections
            </span>
          </div>
        )}
      </div>

      {/* PDF Document Display */}
      <div className="pdf-document-container">
        <div 
          className="pdf-iframe-wrapper"
          style={{ 
            transform: `scale(${scale})`,
            transformOrigin: 'top left',
            width: `${100 / scale}%`,
            height: `${100 / scale}%`
          }}
        >
          {pdfLoaded && (
            <iframe
              ref={iframeRef}
              src={`${pdfUrl}#toolbar=1&navpanes=1&scrollbar=1&zoom=page-fit`}
              width="100%"
              height="100%"
              style={{ 
                border: 'none',
                borderRadius: '8px',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)'
              }}
              title="PDF Document"
              onLoad={handleIframeLoad}
              onError={handleIframeError}
              onMouseUp={handleTextSelection}
              sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
            />
          )}
        </div>

        {/* Loading overlay for iframe */}
        {loading && (
          <div className="pdf-iframe-loading">
            <div className="loading-spinner"></div>
            <p>Rendering PDF...</p>
          </div>
        )}

        {/* Highlight Overlay */}
        {getCurrentPageHighlights().length > 0 && (
          <div className="pdf-highlights-overlay">
            <div className="highlight-indicator">
              <span>🔍 {getCurrentPageHighlights().length} sections highlighted</span>
            </div>
          </div>
        )}
      </div>

      {/* Quick Navigation to Highlights */}
      {highlightedChunks.length > 0 && (
        <div className="highlight-navigation">
          <h4>📍 Relevant Sections Found:</h4>
          <div className="highlight-nav-grid">
            {highlightedChunks.slice(0, 6).map((chunk, index) => (
              <div
                key={index}
                className="highlight-nav-item"
                title={`Relevance: ${((chunk.relevance_score || 0.5) * 100).toFixed(1)}%`}
              >
                <div className="chunk-header">
                  <span className="page-number">Page {chunk.page_number || '?'}</span>
                  <span className="relevance-score">
                    {((chunk.relevance_score || 0.5) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="chunk-preview">
                  {chunk.text?.substring(0, 120)}...
                </div>
              </div>
            ))}
          </div>
          {highlightedChunks.length > 6 && (
            <p className="more-chunks">
              ... and {highlightedChunks.length - 6} more relevant sections
            </p>
          )}
        </div>
      )}

      {/* Help Text */}
      <div className="pdf-help">
        <p>
          💡 <strong>Tips:</strong> 
          Use the PDF's built-in controls for navigation. 
          {highlightedChunks.length > 0 && ' Yellow highlights show relevant content for your query.'}
        </p>
      </div>
    </div>
  );
};

// Error boundary wrapper
class PDFErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('PDF Viewer Error Boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="pdf-error-boundary">
          <h3>🚨 PDF Viewer Crashed</h3>
          <p>The PDF viewer encountered an unexpected error.</p>
          <details>
            <summary>Technical Details</summary>
            <pre>{this.state.error?.toString()}</pre>
          </details>
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            🔄 Reset Viewer
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Main export with error boundary
const SafePDFViewerWithBoundary = (props) => (
  <PDFErrorBoundary>
    <SafePDFViewer {...props} />
  </PDFErrorBoundary>
);

export default SafePDFViewerWithBoundary;