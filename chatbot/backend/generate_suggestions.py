# generate_suggestions.py
from bitnet_wrapper import BitNetCppInterface
bitnet_cpp = BitNetCppInterface()
import os
from llama_cpp import Llama
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/gemma-3-4b-it-q4_0.gguf")
# MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/mamba-790m-hf.Q4_K_M.gguf")
CTX_SIZE = int(os.environ.get("LLAMA_CTX_SIZE", 2048))
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
def generate_suggested_questions(doc_text: str, model="mistral", temperature=0.7, quantization="4bit") -> list[str]:
    from llama_cpp_interface import stream_llama_cpp_response

    prompt = (
        "Given the following document, generate 3 to 5 interesting questions a user might ask about it.\n\n"
        "DOCUMENT:\n"
        f"{doc_text[:2000]}\n\n"
        "QUESTIONS:"
    )
    # raw_response = stream_llama_cpp_response(prompt, "", model=model, temperature=temperature, quantization=quantization)
    # raw_response = bitnet_cpp.generate(prompt, 128, 2, 2048, 0.7)
    # raw_response = stream_llama_cpp_response(prompt, doc_text, model="mamba", temperature=temperature)
    raw_response = llm(prompt, max_tokens=2048, temperature=temperature)
    logger.info(f"raw_response: {raw_response}")
    raw_response = raw_response['choices'][0]['text']
    questions = [line.strip("-• ") for line in raw_response.split("\n") if "?" in line]
    return questions[:5]
