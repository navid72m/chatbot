// In your renderer.js or equivalent
async function fetchFromBackend(endpoint, method = 'GET', data = null) {
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json'
      }
    };
    
    if (data && (method === 'POST' || method === 'PUT')) {
      options.body = JSON.stringify(data);
    }
    
    try {
      const response = await fetch(`http://localhost:8000${endpoint}`, options);
      return await response.json();
    } catch (error) {
      console.error('Error communicating with backend:', error);
      return { error: 'Failed to communicate with backend' };
    }
  }
  
  // Example usage
  async function getChatResponse(message) {
    const response = await fetchFromBackend('/api/chat', 'POST', { message });
    return response;
  }