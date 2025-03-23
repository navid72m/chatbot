# llm_interface.py - Enhanced Interface for Ollama LLM with quantization support
import requests
import json
from typing import List, Dict, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API URL - ensure this is correct
OLLAMA_API = "http://localhost:11434/api"

def query_ollama(
    query: str, 
    context: str, 
    model: str = "mistral", 
    temperature: float = 0.7,
    quantization: str = "4bit"  # New parameter for quantization
) -> str:
    """
    Query Ollama with the given prompt, supporting quantization settings
    
    Args:
        query: The question to answer
        context: The context information from retrieved documents
        model: The model name to use
        temperature: The sampling temperature
        quantization: Quantization level (1bit, 4bit, 8bit, or None)
    
    Returns:
        The model's response text
    """
    # Format the model name based on quantization
    model_name = model
    if quantization and quantization.lower() != "none":
        if quantization == "4bit":
            # For 4-bit quantization, using the proper format
            if not "q4_0" in model_name and not "q4_K_M" in model_name:
                model_name = f"{model}:q4_0"
        elif quantization == "8bit":
            # For 8-bit quantization
            if not "q8_0" in model_name:
                model_name = f"{model}:q8_0"
        elif quantization == "1bit":
            # For 1-bit quantization
            if not "q1_1" in model_name:
                model_name = f"{model}:q1_1" 
    
    logger.info(f"Using model: {model_name} (quantization: {quantization})")
    
    prompt = f"""
Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question: {query}
""".strip()

    # Basic request data
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        # Log the API request
        logger.info(f"Sending request to: {OLLAMA_API}/generate")
        logger.info(f"Request data: {json.dumps(data, indent=2)}")
        
        # Make the API request
        response = requests.post(f"{OLLAMA_API}/generate", json=data)
        
        # Check for errors
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Log the response summary (not the full content for privacy)
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response length: {len(result.get('response', ''))}")
        
        return result["response"].strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Ollama: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response content: {e.response.text}")
        raise Exception(f"Error querying Ollama: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise Exception(f"Error querying Ollama: {str(e)}")

def list_ollama_models() -> List[str]:
    """List available models from Ollama"""
    try:
        logger.info(f"Fetching models from: {OLLAMA_API}/tags")
        response = requests.get(f"{OLLAMA_API}/tags")
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