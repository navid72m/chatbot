// renderer.js - Frontend logic with quantization support

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded');
    
    const uploadButton = document.getElementById('upload-button');
    console.log('Upload button element:', uploadButton);
    
    const modelSelect = document.getElementById('model-select');
    const quantizationSelect = document.getElementById('quantization-select');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const documentsList = document.getElementById('documents');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const statusMessage = document.getElementById('status-message');
    
    // Track uploaded documents
    const uploadedDocuments = new Set();
    
    // Check if Ollama is installed and running
    const checkOllama = async () => {
      console.log('Checking Ollama status...');
      try {
        const models = await window.api.getModels();
        
        if (models && models.models) {
          // Populate the model selector
          modelSelect.innerHTML = '';
          models.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
          });
          
          // Enable chat if there are documents
          if (uploadedDocuments.size > 0) {
            chatInput.disabled = false;
            sendButton.disabled = false;
          }
          
          statusMessage.textContent = 'Connected to Ollama';
          return true;
        }
      } catch (error) {
        console.log('Error checking Ollama:', error);
        statusMessage.textContent = 'Ollama not detected';
        addMessage('Ollama not detected. Please make sure Ollama is installed and running.', 'system');
        return false;
      }
    };
    
    // Initialize the application
    async function init() {
      console.log('Initializing application...');
      // Check Ollama status
      await checkOllama();
      
      // Update documents list
      updateDocumentsList();
      
      // Set up periodic Ollama status check
      setInterval(checkOllama, 30000);
      
      // Show info about quantization
      addMessage('This version supports 4-bit quantization, which reduces memory usage and improves inference speed with minimal quality loss. You can change the quantization level in the sidebar.', 'system');
    }
    
    // Initialize temperature display
    temperatureSlider.addEventListener('input', () => {
      temperatureValue.textContent = temperatureSlider.value;
    });
    
    // Handle document upload
    console.log('Setting up upload button event listener');
    uploadButton.addEventListener('click', async () => {
      console.log('Upload button clicked!');
      try {
        console.log('Calling selectFile API...');
        const result = await window.api.selectFile();
        console.log('selectFile result:', result);
        
        if (result.canceled) {
          console.log('File selection canceled');
          return;
        }
        
        statusMessage.textContent = 'Processing document...';
        addMessage(`Processing document: ${result.filePath.split('/').pop()}`, 'system');
        
        console.log('Uploading document:', result.filePath);
        const response = await window.api.uploadDocument(result.filePath);
        console.log('Upload response:', response);
        
        if (response.success) {
          // Add to document list
          uploadedDocuments.add(response.filename);
          updateDocumentsList();
          
          // Enable chat
          chatInput.disabled = false;
          sendButton.disabled = false;
          
          statusMessage.textContent = 'Document processed';
          addMessage(`Document processed: ${response.filename}`, 'system');
          addMessage(`Preview: ${response.preview}`, 'system');
        } else {
          statusMessage.textContent = 'Error processing document';
          addMessage(`Error: ${response.error || 'Failed to process document'}`, 'system');
        }
      } catch (error) {
        console.error('Error in upload handler:', error);
        statusMessage.textContent = 'Error';
        addMessage(`Error: ${error.message || 'Unknown error occurred'}`, 'system');
      }
    });
    
    // Handle chat interaction
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendMessage();
    });
    
    async function sendMessage() {
      const query = chatInput.value.trim();
      if (!query) return;
      
      // Add user message to chat
      addMessage(query, 'user');
      
      // Clear input and disable until response received
      chatInput.value = '';
      chatInput.disabled = true;
      sendButton.disabled = true;
      statusMessage.textContent = 'Generating response...';
      
      try {
        const model = modelSelect.value;
        const temperature = parseFloat(temperatureSlider.value);
        const quantization = quantizationSelect.value;
        
        // Updated to include quantization parameter
        const response = await window.api.queryDocument(query, model, temperature, quantization);
        
        if (response.response) {
          // Add bot message
          const messageElement = document.createElement('div');
          messageElement.classList.add('message', 'bot');
          
          // For simple text-only responses
          messageElement.textContent = response.response;
          
          // Add sources if available
          if (response.sources && response.sources.length > 0) {
            const sourcesElement = document.createElement('div');
            sourcesElement.classList.add('sources');
            sourcesElement.textContent = `Sources: ${response.sources.join(', ')}`;
            messageElement.appendChild(sourcesElement);
          }
          
          chatMessages.appendChild(messageElement);
          chatMessages.scrollTop = chatMessages.scrollHeight;
        } else if (response.error) {
          addMessage(`Error: ${response.error}`, 'system');
        }
      } catch (error) {
        console.error('Error querying document:', error);
        addMessage(`Error: ${error.message || 'Failed to generate response'}`, 'system');
      } finally {
        // Re-enable input
        chatInput.disabled = false;
        sendButton.disabled = false;
        statusMessage.textContent = 'Ready';
        chatInput.focus();
      }
    }
    
    function updateDocumentsList() {
      documentsList.innerHTML = '';
      
      if (uploadedDocuments.size === 0) {
        const emptyItem = document.createElement('li');
        emptyItem.textContent = 'No documents uploaded';
        emptyItem.style.opacity = '0.5';
        documentsList.appendChild(emptyItem);
        return;
      }
      
      uploadedDocuments.forEach(doc => {
        const item = document.createElement('li');
        item.textContent = doc;
        documentsList.appendChild(item);
      });
    }
    
    function addMessage(text, sender) {
      const messageElement = document.createElement('div');
      messageElement.classList.add('message', sender);
      messageElement.textContent = text;
      
      chatMessages.appendChild(messageElement);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Start the app
    init();
  });