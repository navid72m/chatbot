import os
import json
import logging
import asyncio
import httpx
import requests
from typing import List, Dict, Any, AsyncGenerator, Optional, Union

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
            return ["deepseek-r1"]  # Default fallback
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return ["deepseek-r1"]  # Default fallback

def stream_ollama_response(
    query: str,
    context: str,
    model: str = "deepseek-r1",
    temperature: float = 0.7,
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Stream response from Ollama API.
    
    Args:
        query: The user query
        context: The context from retrieved documents
        model: Model to use (e.g., "deepseek-r1", "llama2", etc.)
        temperature: Sampling temperature (0.0 to 1.0)
        stream: Whether to stream the response
        
    Returns:
        If stream=False: Complete response as a string
        If stream=True: AsyncGenerator yielding response tokens
    """
    if not stream:
        return _get_ollama_response_sync(query, context, model, temperature)
    else:
        return _stream_ollama_response_async(query, context, model, temperature)

def _get_ollama_response_sync(
    query: str,
    context: str,
    model: str = "deepseek-r1",
    temperature: float = 0.7
) -> str:
    """Get a complete response from Ollama API (non-streaming)."""
    prompt = _build_prompt(query, context)
    
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            },
            timeout=60,  # Increased timeout for longer generations
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return "Sorry, I couldn't process your query due to a server error."
    except Exception as e:
        logger.error(f"Error calling Ollama API: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

async def _stream_ollama_response_async(
    query: str,
    context: str,
    model: str = "deepseek-r1",
    temperature: float = 0.7
) -> AsyncGenerator[str, None]:
    """Stream response from Ollama API token by token."""
    prompt = _build_prompt(query, context)
    
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST",
                f"{OLLAMA_API_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"Ollama API streaming error: {response.status_code} - {error_text.decode('utf-8')}")
                    yield "Sorry, I couldn't process your query due to a server error."
                    return
                
                # Process the streaming response
                async for chunk in response.aiter_text():
                    # Ollama sends each chunk as a complete JSON object
                    try:
                        chunk_data = json.loads(chunk)
                        token = chunk_data.get("response", "")
                        # Only yield non-empty tokens
                        if token:
                            yield token
                        
                        # Break if done
                        if chunk_data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from chunk: {chunk}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing stream chunk: {str(e)}")
                        continue
                
        except Exception as e:
            logger.error(f"Error streaming from Ollama API: {str(e)}")
            yield f"Sorry, I encountered an error: {str(e)}"

def _build_prompt(query: str, context: str) -> str:
    """Build a formatted prompt with user query and document context."""
    prompt = f"""
You are a helpful AI assistant tasked with answering questions about documents.
Below is a relevant section from a document:

---
{context}
---

Answer the following question based on the document above:
{query}

Your response should be accurate, concise, and directly address the question.
If the document does not contain information to answer the question, say so rather than speculating.
"""
    return prompt.strip()

# Advanced response generation function for future use with more complex reasoning
async def generate_advanced_response(
    query: str,
    context: str,
    model: str = "deepseek-r1",
    temperature: float = 0.7,
    use_cot: bool = True,
    verify_answers: bool = True,
) -> Dict[str, Any]:
    """
    Generate a response with optional chain-of-thought reasoning and answer verification.
    This is a placeholder for future implementation.
    """
    # Base response using standard approach
    response_text = stream_ollama_response(query, context, model, temperature)
    
    result = {
        "response": response_text,
        "reasoning": None,
        "verification": None,
        "confidence": 0.85,  # Placeholder confidence
    }
    
    # Future: Implement chain-of-thought reasoning
    if use_cot:
        # This would be implemented to show reasoning steps
        result["reasoning"] = "Reasoning process will be implemented in a future update."
    
    # Future: Implement answer verification
    if verify_answers:
        # This would be implemented to verify the response against source
        result["verification"] = {
            "is_verified": True,
            "unsupported_claims": []
        }
    
    return result