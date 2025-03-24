# llm_interface.py - Enhanced Interface for Ollama LLM with quantization support
import requests
import json
from typing import List, Dict, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API URL - ensure this is correct
OLLAMA_API = "http://localhost:11434"

def query_ollama(
    query: str, 
    context: str, 
    model: str = "mistral", 
    temperature: float = 0.7,
    quantization: str = "4bit"
) -> str:
    """Query Ollama with the given prompt, supporting quantization settings"""
    
    # Get available models first
    available_models = []
    try:
        available_models = list_ollama_models()
        logger.info(f"Available models: {available_models}")
    except Exception as e:
        logger.warning(f"Could not get available models: {e}")
    
    # Format the model name based on quantization
    model_name = model
    if quantization and quantization.lower() != "none":
        if quantization == "4bit":
            model_name = f"{model}:q4_0"
        elif quantization == "8bit":
            model_name = f"{model}:q8_0"
        elif quantization == "1bit":
            model_name = f"{model}:q1_1"
    
    # Check if model exists, if we were able to get the list
    if available_models and model_name not in available_models:
        # Try base model as fallback
        if model in available_models:
            logger.info(f"Model {model_name} not found, falling back to {model}")
            model_name = model
        else:
            logger.info(f"Attempting to use model {model_name} anyway")
    
    logger.info(f"Using model: {model_name} (quantization: {quantization})")
    
    # Format the system message with context
    system_message = f"""
Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question.
""".strip()

    # Chat request data
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "temperature": temperature,
        "stream": False
    }
    
    try:
        # Log the API request
        logger.info(f"Sending request to: {OLLAMA_API}/api/chat")
        logger.info(f"Request data: {json.dumps(data, indent=2)}")
        
        # Make the API request
        response = requests.post(f"{OLLAMA_API}/api/chat", json=data)
        
        # Check for errors
        if response.status_code == 404:
            # Try the generate endpoint as fallback
            logger.info("Chat endpoint returned 404, trying generate endpoint instead")
            generate_data = {
                "model": model_name,
                "prompt": f"{system_message}\n\nQuestion: {query}",
                "temperature": temperature,
                "stream": False
            }
            response = requests.post(f"{OLLAMA_API}/api/generate", json=generate_data)
        
        response.raise_for_status()
        
        # Parse the response based on endpoint used
        result = response.json()
        
        if "message" in result:
            # Chat API response
            response_text = result["message"]["content"].strip()
        elif "response" in result:
            # Generate API response
            response_text = result["response"].strip()
        else:
            logger.error(f"Unexpected response format: {result}")
            response_text = "Error: Unexpected response format from Ollama"
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response length: {len(response_text)}")
        
        return response_text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Ollama: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response content: {e.response.text}")
        
        # Specific handling for model not found
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
            error_text = e.response.text
            if "model not found" in error_text.lower():
                # Try to pull the model
                try:
                    logger.info(f"Model not found, attempting to pull {model}")
                    pull_response = requests.post(f"{OLLAMA_API}/api/pull", json={"name": model})
                    if pull_response.status_code == 200:
                        return f"Model {model} not found. I've started downloading it. Please try again in a moment."
                except Exception as pull_error:
                    logger.error(f"Error pulling model: {pull_error}")
        
        raise Exception(f"Error querying Ollama: {str(e)}")
    
    
def list_ollama_models() -> List[str]:
    """List available models from Ollama"""
    try:
        logger.info(f"Fetching models from: {OLLAMA_API}/api/tags")
        response = requests.get(f"{OLLAMA_API}/api/tags")
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Found {len(result.get('models', []))} models")
        
        # Process the models to show base models
        base_models = set()
        for model in result.get("models", []):
            name = model.get("name", "")
            # Strip quantization suffixes for grouping
            base_name = name.split(":")[0]
            base_models.add(base_name)
        
        return list(base_models)
    except Exception as e:
        logger.error(f"Error listing Ollama models: {str(e)}")
        # Return default model if we can't list models
        return ["mistral"]

# Test the API connection if this file is run directly
if __name__ == "__main__":
    print("Testing Ollama API connection...")
    try:
        models = list_ollama_models()
        print(f"Available models: {models}")
        
        if models:
            test_model = models[0]
            print(f"Testing query with model: {test_model}")
            response = query_ollama(
                query="What is the capital of France?",
                context="France is a country in Europe. Its capital is Paris.",
                model=test_model,
                temperature=0.7,
                quantization="4bit"
            )
            print(f"Response: {response}")
    except Exception as e:
        print(f"Error testing Ollama API: {e}")