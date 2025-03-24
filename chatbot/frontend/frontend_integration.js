// frontend_integration.js - Integration of Advanced RAG features into the frontend
import axios from 'axios';

// Base API URL
const API_BASE_URL = 'http://localhost:8000';

// Default configuration for advanced RAG features
const defaultAdvancedRAGConfig = {
  use_advanced_rag: true,
  use_cot: true,
  use_kg: true,
  verify_answers: true,
  use_multihop: true,
  model: "mistral",
  temperature: 0.7,
  quantization: "4bit"
};

/**
 * Class for handling Advanced RAG features in the frontend
 */
export class AdvancedRAGClient {
  constructor(baseUrl = API_BASE_URL, config = defaultAdvancedRAGConfig) {
    this.baseUrl = baseUrl;
    this.config = { ...defaultAdvancedRAGConfig, ...config };
    this.availableModels = [];
    this.availableFeatures = [];
    this.quantizationOptions = [];
  }

  /**
   * Initialize the client by fetching available models and features
   */
  async initialize() {
    try {
      // Fetch available models
      const modelsResponse = await axios.get(`${this.baseUrl}/models`);
      this.availableModels = modelsResponse.data.models || [];

      // Fetch available features
      const featuresResponse = await axios.get(`${this.baseUrl}/advanced-rag/features`);
      this.availableFeatures = featuresResponse.data.features || [];

      // Fetch quantization options
      const quantizationResponse = await axios.get(`${this.baseUrl}/quantization-options`);
      this.quantizationOptions = quantizationResponse.data.options || [];

      return {
        models: this.availableModels,
        features: this.availableFeatures,
        quantizationOptions: this.quantizationOptions
      };
    } catch (error) {
      console.error('Error initializing Advanced RAG client:', error);
      throw error;
    }
  }

  /**
   * Update the client configuration
   * @param {Object} newConfig - New configuration options
   */
  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
    return this.config;
  }

  /**
   * Upload a document to the backend
   * @param {File} file - The file to upload
   * @returns {Promise} - Upload response
   */
  async uploadDocument(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${this.baseUrl}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error uploading document:', error);
      throw error;
    }
  }

  /**
   * Query the Advanced RAG system
   * @param {string} query - The user's query
   * @returns {Promise} - Query response
   */
  async queryDocument(query) {
    try {
      const response = await axios.post(`${this.baseUrl}/query`, {
        query,
        ...this.config
      });

      return response.data;
    } catch (error) {
      console.error('Error querying document:', error);
      throw error;
    }
  }

  /**
   * Configure the backend Advanced RAG system
   * @returns {Promise} - Configuration response
   */
  async saveConfiguration() {
    try {
      const response = await axios.post(`${this.baseUrl}/config/rag`, {
        use_advanced_rag: this.config.use_advanced_rag,
        use_cot: this.config.use_cot,
        use_kg: this.config.use_kg,
        verify_answers: this.config.verify_answers,
        use_multihop: this.config.use_multihop,
        max_hops: this.config.max_hops || 2,
        max_kg_results: this.config.max_kg_results || 3,
        max_vector_results: this.config.max_vector_results || 5,
        vector_weight: this.config.vector_weight || 0.7,
        kg_weight: this.config.kg_weight || 0.3
      });

      return response.data;
    } catch (error) {
      console.error('Error saving configuration:', error);
      throw error;
    }
  }

  /**
   * Get entities from a query
   * @param {string} query - The query to extract entities from
   * @returns {Promise} - Entities response
   */
  async getEntities(query) {
    try {
      const response = await axios.get(`${this.baseUrl}/knowledge-graph/entities`, {
        params: { query }
      });

      return response.data;
    } catch (error) {
      console.error('Error getting entities:', error);
      throw error;
    }
  }

  /**
   * Get a subgraph for visualization
   * @param {Array} entities - List of entity names
   * @param {number} maxHops - Maximum path length
   * @returns {Promise} - Graph data response
   */
  async getEntityGraph(entities, maxHops = 2) {
    try {
      const response = await axios.post(`${this.baseUrl}/knowledge-graph/entity-graph`, {
        entities,
        max_hops: maxHops
      });

      return response.data;
    } catch (error) {
      console.error('Error getting entity graph:', error);
      throw error;
    }
  }

  /**
   * Find path between two entities
   * @param {string} entity1 - First entity name
   * @param {string} entity2 - Second entity name
   * @param {number} maxHops - Maximum path length
   * @returns {Promise} - Path response
   */
  async findPath(entity1, entity2, maxHops = 3) {
    try {
      const response = await axios.get(`${this.baseUrl}/knowledge-graph/path`, {
        params: {
          entity1,
          entity2,
          max_hops: maxHops
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error finding path:', error);
      throw error;
    }
  }

  /**
   * Debug a query to see how it would be processed
   * @param {string} query - The query to analyze
   * @returns {Promise} - Analysis response
   */
  async analyzeQuery(query) {
    try {
      const formData = new FormData();
      formData.append('query', query);

      const response = await axios.post(`${this.baseUrl}/debug/analyze-query`, formData);

      return response.data;
    } catch (error) {
      console.error('Error analyzing query:', error);
      throw error;
    }
  }
}

/**
 * Helper function to format response with reasoning steps
 * @param {Object} response - Raw response from the API
 * @returns {Object} - Formatted response
 */
export const formatAdvancedResponse = (response) => {
  if (!response) return { answer: 'No response received', formatted: false };
  
  // Basic response
  if (!response.reasoning) {
    return {
      answer: response.response || 'No answer provided',
      sources: response.sources || [],
      formatted: false
    };
  }
  
  // Advanced response with reasoning
  return {
    answer: response.response,
    reasoning: response.reasoning,
    sources: response.sources || [],
    confidence: response.confidence || 'MEDIUM',
    verification: response.verification || null,
    formatted: true
  };
};

/**
 * Component factory for Advanced RAG settings UI
 * @param {Function} h - createElement function from your UI framework
 * @param {Object} client - AdvancedRAGClient instance
 * @param {Function} onUpdate - Callback for config updates
 * @returns {VNode} - Settings component
 */
export const createAdvancedRAGSettings = (h, client, onUpdate) => {
  const features = client.availableFeatures;
  const config = client.config;
  
  return h('div', { class: 'advanced-rag-settings' }, [
    h('h3', {}, 'Advanced RAG Settings'),
    
    // Main toggle
    h('div', { class: 'settings-group' }, [
      h('label', { class: 'switch' }, [
        h('input', {
          type: 'checkbox',
          checked: config.use_advanced_rag,
          onChange: (e) => {
            client.updateConfig({ use_advanced_rag: e.target.checked });
            onUpdate(client.config);
          }
        }),
        h('span', { class: 'slider' }),
      ]),
      h('span', {}, 'Enable Advanced RAG')
    ]),
    
    // Feature toggles (only shown if advanced RAG is enabled)
    config.use_advanced_rag && h('div', { class: 'feature-toggles' }, [
      // Chain of Thought
      h('div', { class: 'settings-group' }, [
        h('label', { class: 'switch' }, [
          h('input', {
            type: 'checkbox',
            checked: config.use_cot,
            onChange: (e) => {
              client.updateConfig({ use_cot: e.target.checked });
              onUpdate(client.config);
            }
          }),
          h('span', { class: 'slider' }),
        ]),
        h('span', {}, 'Chain of Thought Reasoning')
      ]),
      
      // Knowledge Graph
      h('div', { class: 'settings-group' }, [
        h('label', { class: 'switch' }, [
          h('input', {
            type: 'checkbox',
            checked: config.use_kg,
            onChange: (e) => {
              client.updateConfig({ use_kg: e.target.checked });
              onUpdate(client.config);
            }
          }),
          h('span', { class: 'slider' }),
        ]),
        h('span', {}, 'Knowledge Graph')
      ]),
      
      // Answer Verification
      h('div', { class: 'settings-group' }, [
        h('label', { class: 'switch' }, [
          h('input', {
            type: 'checkbox',
            checked: config.verify_answers,
            onChange: (e) => {
              client.updateConfig({ verify_answers: e.target.checked });
              onUpdate(client.config);
            }
          }),
          h('span', { class: 'slider' }),
        ]),
        h('span', {}, 'Answer Verification')
      ]),
      
      // Multi-hop Reasoning
      h('div', { class: 'settings-group' }, [
        h('label', { class: 'switch' }, [
          h('input', {
            type: 'checkbox',
            checked: config.use_multihop,
            onChange: (e) => {
              client.updateConfig({ use_multihop: e.target.checked });
              onUpdate(client.config);
            }
          }),
          h('span', { class: 'slider' }),
        ]),
        h('span', {}, 'Multi-hop Reasoning')
      ]),
      
      // Save Configuration Button
      h('button', {
        class: 'save-config-btn',
        onClick: async () => {
          try {
            await client.saveConfiguration();
            alert('Configuration saved successfully!');
          } catch (error) {
            alert('Error saving configuration');
          }
        }
      }, 'Save Configuration')
    ])
  ]);
};

/**
 * Component factory for rendering advanced response with reasoning
 * @param {Function} h - createElement function from your UI framework
 * @param {Object} response - Formatted response from the API
 * @returns {VNode} - Response component
 */
export const createAdvancedResponseView = (h, response) => {
  if (!response.formatted) {
    // Basic response
    return h('div', { class: 'response-container' }, [
      h('div', { class: 'answer' }, response.answer),
      response.sources && response.sources.length > 0 && h('div', { class: 'sources' }, [
        h('h4', {}, 'Sources:'),
        h('ul', {}, response.sources.map(source => 
          h('li', {}, source)
        ))
      ])
    ]);
  }
  
  // Advanced response with reasoning
  return h('div', { class: 'advanced-response-container' }, [
    // Answer section
    h('div', { class: 'answer-section' }, [
      h('h3', {}, 'Answer'),
      h('div', { class: `answer confidence-${response.confidence.toLowerCase()}` }, [
        response.answer,
        h('div', { class: 'confidence-badge' }, response.confidence)
      ])
    ]),
    
    // Reasoning section (collapsible)
    h('details', { class: 'reasoning-section' }, [
      h('summary', {}, 'View Reasoning Process'),
      h('div', { class: 'reasoning-content' }, response.reasoning)
    ]),
    
    // Verification section (if available)
    response.verification && h('details', { class: 'verification-section' }, [
      h('summary', {}, 'View Verification Details'),
      h('div', { class: 'verification-content' }, [
        h('h4', {}, 'Supported Claims:'),
        h('ul', {}, (response.verification.supported_claims || []).map(claim => 
          h('li', { class: 'supported-claim' }, claim)
        )),
        
        h('h4', {}, 'Unsupported Claims:'),
        response.verification.unsupported_claims && response.verification.unsupported_claims.length > 0 
          ? h('ul', {}, response.verification.unsupported_claims.map(claim => 
              h('li', { class: 'unsupported-claim' }, claim)
            ))
          : h('p', {}, 'No unsupported claims found.'),
        
        h('h4', {}, 'Explanation:'),
        h('p', {}, response.verification.explanation || 'No explanation provided.')
      ])
    ]),
    
    // Sources
    h('div', { class: 'sources-section' }, [
      h('h4', {}, 'Sources:'),
      h('ul', {}, (response.sources || []).map(source => 
        h('li', {}, source)
      ))
    ])
  ]);
};

/**
 * CSS styles for Advanced RAG UI components
 */
export const advancedRAGStyles = `
.advanced-rag-settings {
  background-color: #f8f9fa;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 20px;
}

.settings-group {
  display: flex;
  align-items: center;
  margin: 10px 0;
}

.switch {
  position: relative;
  display: inline-block;
  width: 48px;
  height: 24px;
  margin-right: 10px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: .4s;
  border-radius: 24px;
}

.slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: .4s;
  border-radius: 50%;
}

input:checked + .slider {
  background-color: #2196F3;
}

input:focus + .slider {
  box-shadow: 0 0 1px #2196F3;
}

input:checked + .slider:before {
  transform: translateX(24px);
}

.save-config-btn {
  margin-top: 16px;
  background-color: #4CAF50;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
}

.save-config-btn:hover {
  background-color: #45a049;
}

.advanced-response-container {
  margin-top: 20px;
}

.answer-section {
  margin-bottom: 16px;
}

.answer {
  padding: 12px;
  border-radius: 6px;
  background-color: #f0f7ff;
  border-left: 4px solid #2196F3;
}

.confidence-badge {
  display: inline-block;
  margin-left: 8px;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.8em;
  font-weight: bold;
}

.confidence-high .confidence-badge {
  background-color: #4CAF50;
  color: white;
}

.confidence-medium .confidence-badge {
  background-color: #FFC107;
  color: black;
}

.confidence-low .confidence-badge {
  background-color: #F44336;
  color: white;
}

details {
  margin: 10px 0;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 8px;
}

summary {
  cursor: pointer;
  padding: 8px;
  font-weight: bold;
}

.reasoning-content, .verification-content {
  padding: 12px;
  background-color: #f9f9f9;
  border-radius: 4px;
  margin-top: 8px;
}

.supported-claim {
  color: #2E7D32;
}

.unsupported-claim {
  color: #C62828;
}

.sources-section {
  margin-top: 16px;
  border-top: 1px solid #eee;
  padding-top: 16px;
}
`;