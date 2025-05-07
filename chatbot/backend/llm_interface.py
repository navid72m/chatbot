import os
import json
import logging
import requests
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default Ollama API URL
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")

def list_ollama_models() -> List[str]:
    """List all available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name") for model in models]
        else:
            logger.error(f"Failed to list models: {response.status_code} - {response.text}")
            return ["mistral"]  # Default fallback
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return ["mistral"]  # Default fallback

def stream_ollama_response(query, context, model="mistral", temperature=0.7, quantization=None, timeout=600):
    """
    Send a query to Ollama API with enhanced prompt engineering.
    
    Args:
        query (str): The user's query
        context (str): Context information to help answer the query
        model (str): Which Ollama model to use
        temperature (float): Temperature for response generation
        quantization (str): Quantization level (ignored but included for compatibility)
        timeout (int): Timeout in seconds for the API call
        
    Returns:
        str: The generated response text
    """
    try:
        logger.info(f"Querying model {model} with temperature {temperature}")
        
        # Enhanced prompt with clear instructions
        prompt = f"""
You are a precise document assistant. Your task is to answer questions based ONLY on the provided context.

CONTEXT:
{context}

INSTRUCTION:
- Answer the question based only on the context above
- If the answer isn't in the context, say "I don't see information about that in the document"
- Be concise and direct - focus only on what's asked
- Do not invent or assume information not present in the context
- Include only relevant details that directly answer the question

QUESTION:
{query}

ANSWER:
"""
        
        # Basic payload - keeping it simple
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        
        # Make the API call
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=payload,
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json().get("response", "")
            return result.strip()
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"Error: The language model returned an error. Please try again."
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout when calling Ollama API after {timeout} seconds")
        return "The model took too long to respond. Please try a simpler query or a different model."
    except Exception as e:
        logger.error(f"Error in stream_ollama_response: {str(e)}")
        return f"An error occurred when processing your question. Please try again."