import os
import logging
from typing import List
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






# Load llama.cpp model once during startup
# MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
# MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
# MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/mamba-790m-hf.Q4_K_M.gguf")
MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/gemma-3-4b-it-q4_0.gguf")
CTX_SIZE = int(os.environ.get("LLAMA_CTX_SIZE", 7500))
N_THREADS = int(os.environ.get("LLAMA_THREADS", os.cpu_count() ))
N_GPU_LAYERS = int(os.environ.get("LLAMA_GPU_LAYERS", -1))  # -1 means use all available GPU layers

try:
    # Add n_gpu_layers parameter for Apple Silicon
    llm = Llama(
        model_path=MODEL_PATH, 
        n_ctx=CTX_SIZE, 
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS  # Use Metal on Apple Silicon
    )
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load LLaMA model: {str(e)}")
    llm = None

# In llama_cpp_interface.py
from improved_prompt_template import improved_llama_cpp_response

# Find your stream_llama_cpp_response function and replace or modify it
def stream_llama_cpp_response(query: str, context: str, model: str = None, temperature: float = 0.3, **kwargs) -> str:
    """
    Generate a response using llama.cpp LLM.
    
    Args:
        query (str): The user's question
        context (str): Retrieved context chunks
        model (str): Ignored (for API compatibility)
        temperature (float): Sampling temperature

    Returns:
        str: Generated answer
    """
    # Use improved prompt template
    return improved_llama_cpp_response( llm,query, context, model, temperature)

def list_llama_cpp_models() -> List[str]:
    """Stub function for compatibility with Ollama-style list."""
    return [os.path.basename(MODEL_PATH)] if llm else []

def clean_response(response: str) -> str:
    """Clean up verbose reasoning from responses."""
    import re
    
    # Remove numbered thinking steps
    response = re.sub(r'\d+\.\s+.*?(?=\d+\.|OUTPUT:|RESPONSE:|ANSWER:|$)', '', response, flags=re.DOTALL)
    
    # Remove headers
    for header in ["RESPONSE:", "OUTPUT:", "ANSWER:", "CONCLUSION:"]:
        if header in response:
            parts = response.split(header)
            if len(parts) > 1:
                response = parts[1].strip()
    
    # Remove "Based on the context" phrases
    response = re.sub(r'Based on (?:the |)context,?\s*', '', response)
    
    return response.strip()
def improved_llama_cpp_responses(query: str, context: str, model: str = None, temperature: float = 0.7, **kwargs) -> str:
    """
    Generate a response using llama.cpp LLM with an improved prompt.
    """
    # prompt = f"""
    # You are an intelligent document assistant that answers questions based on the provided context. Your goal is to be helpful, accurate, and thorough.

    # CONTEXT:
    # {context}

    # INSTRUCTIONS:
    # 1. First, carefully analyze what information the question is asking for.
    # 2. Then, identify any relevant information in the context that could help answer the question.
    # 3. If the information is explicitly present, provide a direct and specific answer.
    # 4. If the information requires connecting multiple pieces, synthesize them into a coherent answer.
    # 5. If some aspects are missing but you can make reasonable inferences from the context, do so while indicating what is inferred.
    # 6. If the question cannot be answered based on the context, clearly state that the information is not available.

    # QUESTION:
    # {query}

    # THOUGHT PROCESS:
    # Let me think through what information I need to answer this question and what's available in the context...

    # ANSWER:
    # """
    prompt = f"""
You are a precise document assistant. Your task is to extract answers from the provided context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Extract information directly from the context
- Look carefully through ALL parts of the context before saying information isn't present
- If the exact answer is in the context, provide it directly
- If pieces of the answer are in different parts of the context, combine them
- Only say "The document doesn't mention [topic]" if you've thoroughly checked and the information is truly absent
- Be concise but complete
- Do NOT make up information

ANSWER:
"""
    
    try:
        logger.info(f"llama.cpp prompt: {prompt}")
        if not llm:
            logger.error("llm not initialized")
        response = llm(prompt=prompt, max_tokens=512, temperature=temperature)
        
        # Extract generated text
        generated_text = response["choices"][0]["text"] if isinstance(response, dict) and "choices" in response else response
        
        # Remove "THOUGHT PROCESS" if it appears in the output
        if "THOUGHT PROCESS:" in generated_text:
            generated_text = generated_text.split("ANSWER:")[1].strip()
        
        logger.info(f"llama.cpp response generated successfully")
        return clean_response(generated_text)
    except Exception as e:
        logger.error(f"llama.cpp generation error: {str(e)}")
        return "Error generating response."
# def stream_llama_cpp_response(query: str, context: str, model: str = None, temperature: float = 0.7, **kwargs) -> str:
#     """
#     Generate a response using llama.cpp LLM.

#     Args:
#         query (str): The user's question
#         context (str): Retrieved context chunks
#         model (str): Ignored (for API compatibility)
#         temperature (float): Sampling temperature

#     Returns:
#         str: Generated answer
#     """
#     if not llm:
#         return "Model not available."

#     prompt = f"""
# You are a precise document assistant. Your task is to answer questions based ONLY on the provided context.

# CONTEXT:
# {context}

# INSTRUCTION:
# - Answer the question based only on the context above
# - If the answer isn't in the context, say "I don't see information about that in the document"
# - Be concise and direct - focus only on what's asked
# - Do not hallucinate or make up information
# - Do not invent or assume information not present in the context
# - Include only relevant details that directly answer the question

# QUESTION:
# {query}

# ANSWER:
# """
#     try:
#         logger.info(f"llama.cpp prompt: {prompt}")
#         if not llm:
#             logger.error("llm not initialized")
#         response = llm(prompt=prompt, max_tokens=512, temperature=temperature)
        
#         # Extract generated text
#         generated_text = response["choices"][0]["text"] if isinstance(response, dict) and "choices" in response else response
        
#         logger.info(f"llama.cpp response generated successfully")
#         return generated_text
#     except Exception as e:
#         logger.error(f"llama.cpp generation error: {str(e)}")
#         return "Error generating response."
