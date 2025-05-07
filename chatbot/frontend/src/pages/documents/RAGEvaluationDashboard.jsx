import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';

const RAGEvaluationDashboard = () => {
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [apiUrl] = useState('http://localhost:8000'); // Adjust based on your backend URL

  // Fetch available documents on component mount
  useEffect(() => {
    const fetchDocuments = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`${apiUrl}/evaluation/datasets`);
        if (response.data.datasets && response.data.datasets.length > 0) {
          setDocuments(response.data.datasets);
          // Auto-select the first document
          setSelectedDocument(response.data.datasets[0].document_name);
        } else {
          setError('No evaluation datasets found. Please create a dataset first.');
        }
      } catch (err) {
        console.error('Failed to fetch evaluation datasets:', err);
        setError('Failed to fetch evaluation datasets. Check the console for details.');
      } finally {
        setLoading(false);
      }
    };

    fetchDocuments();
  }, [apiUrl]);

  // Fetch comparison data when a document is selected
  useEffect(() => {
    if (!selectedDocument) return;

    const fetchComparisonData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await axios.get(`${apiUrl}/visualization/comparison?document=${selectedDocument}`);
        
        if (response.data.success) {
          setComparisonData(response.data.visualization_data);
        } else {
          setError(response.data.error || 'Failed to fetch comparison data');
        }
      } catch (err) {
        console.error('Error loading comparison data:', err);
        setError('Error loading comparison data. Check the console for details.');
      } finally {
        setLoading(false);
      }
    };

    fetchComparisonData();
  }, [selectedDocument, apiUrl]);

  // Handle document selection change
  const handleDocumentChange = (e) => {
    setSelectedDocument(e.target.value);
  };

  // Handle running a new evaluation
  const handleRunEvaluation = async () => {
    if (!selectedDocument) {
      setError('Please select a document first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Trigger the comparison endpoint with force_new=true
      const response = await axios.post(`${apiUrl}/evaluate/compare`, {
        document: selectedDocument,
        force_new: true
      });

      if (response.data.success) {
        setComparisonData(response.data.comparison);
        alert('Evaluation completed successfully!');
      } else {
        setError(response.data.error || 'Failed to run evaluation');
      }
    } catch (err) {
      console.error('Error running evaluation:', err);
      setError('Error running evaluation. Check the console for details.');
    } finally {
      setLoading(false);
    }
  };

  // Render metrics comparison chart
  const renderMetricsChart = () => {
    if (!comparisonData || !comparisonData.metrics) return null;

    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Metrics Comparison</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={comparisonData.metrics}
            margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(value) => value.toFixed(2)} />
            <Legend />
            <Bar dataKey="original" name="Original RAG" fill="#8884d8" />
            <Bar dataKey="llama_index" name="LlamaIndex RAG" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Render question types comparison
  const renderQuestionTypesChart = () => {
    if (!comparisonData || !comparisonData.question_types) return null;

    // Transform data for radar chart
    const radarData = comparisonData.question_types.map(item => ({
      type: item.type,
      original: item.original_generation,
      llamaIndex: item.llama_index_generation,
    }));

    return (
      <div className="bg-white p-6 rounded-lg shadow-md mt-6">
        <h3 className="text-lg font-semibold mb-4">Performance by Question Type</h3>
        <ResponsiveContainer width="100%" height={400}>
          <RadarChart outerRadius={150} data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="type" />
            <PolarRadiusAxis domain={[0, 1]} />
            <Radar
              name="Original RAG"
              dataKey="original"
              stroke="#8884d8"
              fill="#8884d8"
              fillOpacity={0.5}
            />
            <Radar
              name="LlamaIndex RAG"
              dataKey="llamaIndex"
              stroke="#82ca9d"
              fill="#82ca9d"
              fillOpacity={0.5}
            />
            <Legend />
            <Tooltip formatter={(value) => value.toFixed(2)} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Render individual question comparison
  const renderQuestionComparison = () => {
    if (!comparisonData || !comparisonData.questions) return null;

    return (
      <div className="bg-white p-6 rounded-lg shadow-md mt-6">
        <h3 className="text-lg font-semibold mb-4">Individual Question Performance</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={comparisonData.questions}
            margin={{ top: 20, right: 30, left: 20, bottom: 100 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="question" angle={-45} textAnchor="end" height={100} interval={0} />
            <YAxis domain={[0, 1]} />
            <Tooltip 
              formatter={(value) => value.toFixed(2)} 
              labelFormatter={(label) => `Question: ${label}`}
            />
            <Legend />
            <Bar dataKey="original_similarity" name="Original RAG" fill="#8884d8" />
            <Bar dataKey="llama_index_similarity" name="LlamaIndex RAG" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Render recommendations
  const renderRecommendations = () => {
    if (!comparisonData || !comparisonData.recommendations) return null;

    return (
      <div className="bg-white p-6 rounded-lg shadow-md mt-6">
        <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
        <ul className="list-disc pl-5 space-y-2">
          {comparisonData.recommendations.map((rec, index) => (
            <li key={index} className="text-gray-700">{rec}</li>
          ))}
        </ul>
      </div>
    );
  };

  // Determine if LlamaIndex is better overall
  const getLlamaIndexPerformance = () => {
    if (!comparisonData || !comparisonData.metrics) return null;
    
    const overallDifference = 
      (comparisonData.metrics.reduce((acc, curr) => acc + curr.difference, 0)) / 
      comparisonData.metrics.length;
    
    if (overallDifference > 0.05) {
      return {
        better: true,
        message: `LlamaIndex RAG outperforms the original system by ${(overallDifference * 100).toFixed(1)}% overall.`
      };
    } else if (overallDifference < -0.05) {
      return {
        better: false,
        message: `Original RAG outperforms LlamaIndex by ${(Math.abs(overallDifference) * 100).toFixed(1)}% overall.`
      };
    } else {
      return {
        better: null,
        message: "Both RAG systems perform similarly overall."
      };
    }
  };

  // Render overview dashboard
  const renderOverview = () => {
    if (!comparisonData) return null;
    
    const llamaIndexPerformance = getLlamaIndexPerformance();
    
    return (
      <div className="space-y-6">
        {llamaIndexPerformance && (
          <div className={`p-4 rounded-lg ${
            llamaIndexPerformance.better === true ? 'bg-green-100' : 
            llamaIndexPerformance.better === false ? 'bg-red-100' : 'bg-blue-100'
          }`}>
            <p className="font-medium">{llamaIndexPerformance.message}</p>
          </div>
        )}
        
        {renderMetricsChart()}
        {renderQuestionTypesChart()}
        {renderRecommendations()}
      </div>
    );
  };

  // Render the page content with dashboard header and controls
  return (
    <div className="container mx-auto py-6 px-4 max-w-7xl">
      <div className="mb-6 flex justify-between items-center">
        <h1 className="text-2xl font-bold">RAG Evaluation Dashboard</h1>
        <button 
          onClick={handleRunEvaluation}
          disabled={loading || !selectedDocument}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {loading ? 'Running...' : 'Run New Evaluation'}
        </button>
      </div>
      
      {/* Document selector */}
      <div className="mb-6">
        <label htmlFor="documentSelect" className="block text-sm font-medium text-gray-700 mb-2">
          Select Document:
        </label>
        <select
          id="documentSelect"
          className="block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          value={selectedDocument || ''}
          onChange={handleDocumentChange}
          disabled={loading}
        >
          <option value="">Select a document...</option>
          {documents.map((doc) => (
            <option key={doc.document_name} value={doc.document_name}>
              {doc.display_name} ({doc.question_count} questions)
            </option>
          ))}
        </select>
      </div>
      
      {/* Loading state */}
      {loading && (
        <div className="flex justify-center items-center h-64">
          <div className="text-lg font-medium">Loading evaluation data...</div>
        </div>
      )}
      
      {/* Error state */}
      {error && !loading && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}
      
      {/* Tabs */}
      {!loading && !error && (
        <div className="mb-6 border-b border-gray-200">
          <nav className="flex space-x-8" aria-label="Tabs">
            <button
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'overview' 
                  ? 'border-indigo-500 text-indigo-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('overview')}
            >
              Overview
            </button>
            <button
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'questionTypes' 
                  ? 'border-indigo-500 text-indigo-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('questionTypes')}
            >
              Question Types
            </button>
            <button
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'questions' 
                  ? 'border-indigo-500 text-indigo-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('questions')}
            >
              Individual Questions
            </button>
          </nav>
        </div>
      )}
      
      {/* Tab content */}
      {!loading && !error && selectedDocument && comparisonData ? (
        <div>
          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'questionTypes' && renderQuestionTypesChart()}
          {activeTab === 'questions' && renderQuestionComparison()}
        </div>
      ) : !loading && !error ? (
        <div className="bg-blue-50 p-4 rounded-md">
          <p>Please select a document to view evaluation results.</p>
        </div>
      ) : null}
    </div>
  );
};

export default RAGEvaluationDashboard;