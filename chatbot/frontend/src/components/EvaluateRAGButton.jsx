import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const EvaluateRAGButton = ({ currentDocument, apiUrl = 'http://localhost:8000' }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleEvaluate = async () => {
    if (!currentDocument) {
      setError('Please select a document first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // First check if a dataset exists
      const checkResponse = await axios.get(`${apiUrl}/evaluation/datasets`);
      const datasets = checkResponse.data.datasets || [];
      const datasetExists = datasets.some(d => d.document_name === currentDocument);

      if (!datasetExists) {
        // Create a new evaluation dataset
        await axios.post(`${apiUrl}/evaluate/create-dataset`, {
          document: currentDocument,
          num_questions: 10 // Default to 10 questions
        });
      }

      // Run the comparison
      await axios.post(`${apiUrl}/evaluate/compare`, {
        document: currentDocument
      });

      // Navigate to the evaluation dashboard
      navigate('/evaluation');
    } catch (err) {
      console.error('Error initiating evaluation:', err);
      setError('Failed to start evaluation. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4">
      <button
        onClick={handleEvaluate}
        disabled={loading || !currentDocument}
        className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-md shadow-sm disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
      >
        {loading ? (
          <>
            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Evaluating...
          </>
        ) : (
          <>Evaluate RAG Systems</>
        )}
      </button>
      
      {error && (
        <div className="mt-2 text-sm text-red-600">
          {error}
        </div>
      )}
    </div>
  );
};

export default EvaluateRAGButton;