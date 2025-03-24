import React, { useState } from 'react';
import { AdvancedRAGClient, formatAdvancedResponse } from './frontend_integration.js';
import './styles.css';

const AdvancedRAGComponent = () => {
  const [client] = useState(() => new AdvancedRAGClient());
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState('');

  const handleQuery = async () => {
    setLoading(true);
    try {
      const raw = await client.queryDocument(query);
      const formatted = formatAdvancedResponse(raw);
      setResponse(formatted);
    } catch (err) {
      console.error('Error querying document:', err);
      setResponse({ answer: 'An error occurred.', formatted: false });
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setFileName(file.name);
    try {
      await client.uploadDocument(file);
      alert(`✅ Uploaded: ${file.name}`);
    } catch (err) {
      console.error('Error uploading file:', err);
      alert('❌ Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="rag-container" style={{ padding: '20px' }}>
      <h2>Advanced RAG Chat</h2>

      <div style={{ marginBottom: '20px' }}>
        <label style={{ fontWeight: 'bold' }}>Upload a document:</label><br />
        <input type="file" onChange={handleUpload} disabled={uploading} />
        {fileName && <p style={{ marginTop: '5px' }}>Last uploaded: {fileName}</p>}
      </div>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        rows={4}
        placeholder="Type your question..."
        style={{ width: '100%', padding: '10px', fontSize: '1rem' }}
      />

      <button onClick={handleQuery} disabled={loading || uploading} style={{ marginTop: '10px' }}>
        {loading ? 'Loading...' : 'Submit Query'}
      </button>

      {response && (
        <div className="response-box">
          <h3>Answer:</h3>
          <p>{response.answer}</p>

          {response.reasoning && (
            <>
              <h4>Reasoning:</h4>
              <pre>{response.reasoning}</pre>
            </>
          )}

          {response.sources?.length > 0 && (
            <>
              <h4>Sources:</h4>
              <ul>
                {response.sources.map((src, idx) => (
                  <li key={idx}>{src}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default AdvancedRAGComponent;
