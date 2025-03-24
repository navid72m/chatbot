import axios from 'axios';

// Default base URL for your backend API
const API_BASE_URL = 'http://localhost:8000';

// Default config values
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
 * Client for communicating with the RAG backend
 */
export class AdvancedRAGClient {
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
  
    async uploadDocument(file) {
      const formData = new FormData();
      formData.append('file', file);
  
      const response = await this.client.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log(response.data);
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
 * Helper to format reasoning-style responses from the RAG system
 */
export const formatAdvancedResponse = (response) => {
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
