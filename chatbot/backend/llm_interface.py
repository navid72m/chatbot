import logging
from typing import List, Optional, Dict, Any
import ctypes
import os
import numpy as np
from llama_cpp import Llama




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

def query_ollama(
    query: str, 
    context: str = "", 
    model: str = "mistral", 
    temperature: float = 0.7,
    quantization: str = "4bit"
) -> str:
    """
    Unified query interface for language models
    
    Args:
        query: User's query
        context: Additional context information
        model: Model name
        temperature: Sampling temperature
        quantization: Quantization level (not used for GGUF, kept for API compatibility)
    
    Returns:
        Model-generated response
    """
    try:
        # Determine model path (update with your actual path)
        model_path = os.path.expanduser("~/startup/chatbot/external/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        
        # Ensure model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Prepare full prompt with context
        full_prompt = f"""
Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question.

Question: {query}
""".strip()
        
        # Initialize model
        # llm = LlamaCppModel(model_path=model_path)
        
        # # Generate response
        # response = llm.generate(
        #     prompt=full_prompt, 
        #     temperature=temperature
        # )
        llm = Llama(
        model_path="/Users/seyednavidmirnourilangeroudi/startup/chatbot/external/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=4096,
        n_threads=4  # Adjust based on CPU
        )

        response = llm(
            prompt=full_prompt,
            max_tokens=400,
            temperature=0.7,
        )

# print(response["choices"][0]["text"])
        
        return response
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        return f"Error generating response: {str(e)}"

def list_ollama_models() -> List[str]:
    """
    List available models
    
    Returns:
        List of model names
    """
    # In a real implementation, scan the Ollama models directory
    return [
        "mistral",
        "llama3",
        "phi"
    ]