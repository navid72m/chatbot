import React from 'react';
import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// import Dashboard from './pages/dashboard/Dashboard';
// import DocumentUpload from './pages/documents/DocumentUpload';
// import DocumentUploadAndChat from './pages/documents/DocumentUploadandChat';
import MCPDocumentChat from './pages/documents/MCPDocumentChat';
// Simple placeholder component for not found page
const NotFound = () => <div className="page-container"><h1>404</h1><p>Page not found</p></div>;

function App() {
  return (
    <Router>
      <div className="app">
        <Routes>
          {/* Redirect root path to the document upload page */}
          <Route path="/" element={<Navigate to="/documents/upload" replace />} />
          {/* <Route path="/dashboard" element={<Dashboard />} /> */}
          {/* <Route path="/documents/upload" element={<DocumentUploadAndChat />} /> */}
          <Route path="/documents/upload" element={<MCPDocumentChat />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;