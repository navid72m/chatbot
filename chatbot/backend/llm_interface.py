import logging
from typing import List, Optional, Dict, Any
import ctypes
import os
import numpy as np
# from llama_cpp import Llama
import requests




# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# class LlamaCppModel:
#     """
#     Low-level interface for llama.cpp GGUF models using ctypes
#     """
    
#     def __init__(
#         self, 
#         model_path: str, 
#         n_ctx: int = 4096,  # Context window size
#         n_batch: int = 512  # Batch size
#     ):
#         """
#         Initialize the llama.cpp model
        
#         Args:
#             model_path: Full path to the GGUF model file
#             n_ctx: Context window size
#             n_batch: Batch size for processing
#         """
#         try:
#             # Determine library path (adjust for your system)
#             lib_paths = [
#                 "/usr/local/lib/libllama.dylib",  # macOS typical location
#                 "/opt/homebrew/lib/libllama.dylib",  # Homebrew on Mac
#                 os.path.expanduser("~/.local/lib/libllama.dylib"),  # User local
#             ]
            
#             # Find the first existing library
#             lib_path = next((path for path in lib_paths if os.path.exists(path)), None)
            
#             if not lib_path:
#                 raise FileNotFoundError("Could not find llama.cpp shared library")
            
#             # Load the library
#             self.lib = ctypes.CDLL(lib_path)
            
#             # Set up model loading function signatures
#             # Note: These might need adjustment based on your specific llama.cpp version
#             self.lib.llama_load_model_from_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
#             self.lib.llama_load_model_from_file.restype = ctypes.c_void_p
            
#             # Prepare model path
#             model_path_bytes = model_path.encode('utf-8')
            
#             # Load the model
#             self.model = self.lib.llama_load_model_from_file(
#                 model_path_bytes, 
#                 n_ctx
#             )
            
#             if not self.model:
#                 raise RuntimeError(f"Failed to load model: {model_path}")
            
#             # Create context
#             self.lib.llama_new_context_with_model.argtypes = [ctypes.c_void_p, ctypes.c_int]
#             self.lib.llama_new_context_with_model.restype = ctypes.c_void_p
            
#             self.context = self.lib.llama_new_context_with_model(self.model, n_batch)
            
#             if not self.context:
#                 raise RuntimeError("Failed to create model context")
            
#             logger.info(f"Successfully loaded model: {model_path}")
        
#         except Exception as e:
#             logger.error(f"Model initialization error: {e}")
#             raise
    
#     def generate(
#         self, 
#         prompt: str, 
#         max_tokens: int = 200, 
#         temperature: float = 0.7
#     ) -> str:
#         """
#         Generate text using the loaded model
        
#         Args:
#             prompt: Input text prompt
#             max_tokens: Maximum number of tokens to generate
#             temperature: Sampling temperature
        
#         Returns:
#             Generated text string
#         """
#         try:
#             # Placeholder for text generation
#             # In a real implementation, you'd use ctypes to:
#             # 1. Tokenize the prompt
#             # 2. Set up generation parameters
#             # 3. Generate tokens
#             # 4. Detokenize the output
            
#             logger.info(f"Generating text for prompt: {prompt[:100]}...")
            
#             # Simulated generation for demonstration
#             return f"Generated response to: {prompt[:50]}..."
        
#         except Exception as e:
#             logger.error(f"Text generation error: {e}")
#             raise
    
#     def __del__(self):
#         """Clean up resources"""
#         try:
#             if hasattr(self, 'context') and self.context:
#                 self.lib.llama_free(self.context)
#             if hasattr(self, 'model') and self.model:
#                 self.lib.llama_free_model(self.model)
#         except Exception as e:
#             logger.error(f"Cleanup error: {e}")

import requests
import json
import logging

logger = logging.getLogger(__name__)

def stream_ollama_response(
    query: str,
    context: str = "",
    model: str = "deepseek-r1",
    temperature: float = 0.7
):
    full_prompt = f"""
Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question.

Question: {query}
""".strip()

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": True,
        "options": {
            "temperature": temperature
        }
    }

    try:
        with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        yield chunk.get("response", "")
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"[Streaming error] {str(e)}"


def list_ollama_models() -> List[str]:
    """
    List available Ollama models
    
    Returns:
        List of model names
    """
    try:
        # Query Ollama API for available models
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Extract model names
        models = [model["name"] for model in result.get("models", [])]
        
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []