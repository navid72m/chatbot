const axios = require('axios');

const API_BASE_URL = 'http://localhost:8000';

const defaultAdvancedRAGConfig = {
  use_advanced_rag: true,
  use_cot: true,
  use_kg: true,
  verify_answers: true,
  use_multihop: true,
  model: "mistral",
  temperature: 0.7,
  quantization: "4bit",
};

/**
 * Client for use in the Electron main process (Node.js)
 */
class AdvancedRAGClient {
  constructor(baseUrl = API_BASE_URL, config = {}) {
    this.baseUrl = baseUrl;
    this.config = { ...defaultAdvancedRAGConfig, ...config };
    this.client = axios.create({ baseURL: this.baseUrl });
  }

  async queryDocument(query) {
    const response = await this.client.post('/query', {
      query,
      ...this.config,
    });
    return response.data;
  }

  async saveConfiguration() {
    const response = await this.client.post('/config/rag', this.config);
    return response.data;
  }

  updateConfig(newConfig) {
    this.config = { ...this.config, ...newConfig };
    return this.config;
  }
}

/**
 * Format responses with or without reasoning
 */
const formatAdvancedResponse = (response) => {
  if (!response) return { answer: 'No response received', formatted: false };

  if (!response.reasoning) {
    return {
      answer: response.response || 'No answer provided',
      sources: response.sources || [],
      formatted: false,
    };
  }

  return {
    answer: response.response,
    reasoning: response.reasoning,
    sources: response.sources || [],
    confidence: response.confidence || 'MEDIUM',
    verification: response.verification || null,
    formatted: true,
  };
};

module.exports = {
  AdvancedRAGClient,
  formatAdvancedResponse,
};
