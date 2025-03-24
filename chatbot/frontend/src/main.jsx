import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Import base styling
import './styles/variables.css';
import './styles/base.css';
import './styles/utilities.css';

// Import component styles
import './styles/components/file-upload.css';

// Import page styles
import './styles/pages/document-upload.css';


ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);