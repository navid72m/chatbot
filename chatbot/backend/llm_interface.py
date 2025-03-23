# llm_interface.py - Interface for Ollama LLM
import requests
import json
from typing import List, Dict, Optional

OLLAMA_API = "http://localhost:11434/api"

def query_ollama(query: str, context: str, model: str = "mistral", temperature: float = 0.7) -> str:
    """Query Ollama with the given prompt"""
    prompt = f"""
Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question: {query}
""".strip()

    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }

    try:
        response = requests.post(f"{OLLAMA_API}/generate", json=data)
        response.raise_for_status()
        result = response.json()
        return result["response"].strip()
    except Exception as e:
        raise Exception(f"Error querying Ollama: {str(e)}")

def list_ollama_models() -> List[str]:
    """List available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_API}/tags")
        response.raise_for_status()
        result = response.json()
        return [model["name"] for model in result["models"]]
    except Exception as e:
        print(f"Error listing Ollama models: {str(e)}")
        # Return default model if we can't list models
        return ["mistral"] 