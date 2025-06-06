import sys
import os
sys.path.append(os.path.dirname(__file__))
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
import logging
import shutil
from pydantic import BaseModel
# from knowledge_graph import KnowledgeGraph
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from generate_suggestions import generate_suggested_questions
from enhanced_rag import EnhancedRAG
from llama_cpp import Llama
# Import simplified modules
from document_processor_patched import process_and_index_file , query_index, query_index_with_context
# from vector_store import VectorStore, build_focused_context
from llama_cpp_interface import stream_llama_cpp_response, list_llama_cpp_models
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global components
# vector_store = VectorStore()

# knowledge_graph = KnowledgeGraph()
suggested_questions_by_doc = {}

# Storage directory for documents
UPLOAD_DIR = os.path.expanduser("~/Library/Application Support/Document Chat/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    model: str = "mistral"
    temperature: float = 0.7
    context_window: int = 5
    quantization: str = "4bit"
    use_advanced_rag: bool = False
    current_document: Optional[str] = None

# Create FastAPI application
app = FastAPI(title="Document Chat Backend")
# enhanced_rag = add_to_app_integration(app)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import the evaluator
# from rag_evaluator import RAGEvaluator
# from integrate_evaluation_framework import integrate_evaluation_framework

# # Add this after initializing your app and vector_store
# integrate_evaluation_framework(
#     app=app,
#     vector_store=vector_store,
#     document_storage_path=os.path.expanduser("~/Library/Application Support/Document Chat")
# )
# MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
# MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
# MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/mamba-790m-hf.Q4_K_M.gguf")
# MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/  ggml-model-i2_s.gguf"
# )
# MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/llama-3.1-70b-versatile.Q4_K_M.gguf")
MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/gemma-3-4b-it-q4_0.gguf")
CTX_SIZE = int(os.environ.get("LLAMA_CTX_SIZE", 2048))
N_THREADS = int(os.environ.get("LLAMA_THREADS", os.cpu_count() ))
N_GPU_LAYERS = int(os.environ.get("LLAMA_GPU_LAYERS", -1))  # -1 means use all available GPU layers

try:
    # Add n_gpu_layers parameter for Apple Silicon
    llm = Llama(
        model_path=MODEL_PATH, 
        n_ctx=CTX_SIZE, 
        n_threads=N_THREADS,
        n_predict=128
        # n_gpu_layers=N_GPU_LAYERS  # Use Metal on Apple Silicon
    )
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load LLaMA model: {str(e)}")
    llm = None

from enhanced_rag import EnhancedRAG
# enhanced_rag = EnhancedRAG(vector_store, llm)
# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def root():
    return {
        "message": "Document Chat Backend API",
        "features": ["Vector Search", "Document Q&A"]
    }
@app.get("/kg")
async def kg_view():
    return FileResponse("static/kg.html")
from llama_index_integration import LlamaIndexRAG
llama_index_rag = LlamaIndexRAG(MODEL_PATH)
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    use_advanced_rag: bool = Form(False),
    use_llama_index: bool = Form(True)  # New parameter
):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process with original system
        chunks, suggested_questions = process_and_index_file(file_path)
        
        # Also process with LlamaIndex if requested
        # if use_llama_index:
        try:
            logger.info("Processing document with LlamaIndex")
            llama_chunks = llama_index_rag.process_document(file_path)
            logger.info(f"LlamaIndex processed {len(llama_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error in LlamaIndex processing: {str(e)}")
    
        suggested_questions_by_doc[file.filename] = suggested_questions

        return {
            "success": True,
            "filename": file.filename,
            "chunks": len(chunks),
            "preview": "Document uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/set_document")
async def set_document(document: dict):
    try:
        document_name = document.get('document')
        if not document_name:
            raise HTTPException(status_code=400, detail="No document name provided")
        app.state.current_document = document_name
        return {"success": True, "document": document_name}
    except Exception as e:
        logger.error(f"Error setting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting document: {str(e)}")

@app.get("/suggestions")
async def get_suggested_questions(document: Optional[str] = None):
    try:
        if not document:
            document = getattr(app.state, "current_document", None)
        if not document:
            return {"success": False, "error": "No document selected"}
        return {"document": document, "questions": suggested_questions_by_doc.get(document, [])}
    except Exception as e:
        logger.error(f"Error fetching suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# @app.post("/query-sync")
# async def query_sync(request: dict):
#     try:
#         query = request.get("query")
#         document = request.get("document")
#         if not query:
#             raise HTTPException(status_code=400, detail="No query provided")
#         if not document:
#             document = getattr(app.state, "current_document", None)
#             if not document:
#                 return {"response": "Please upload a document first before querying.", "error": "No document selected"}
#         model = request.get("model", "deepseek-r1")
#         temperature = request.get("temperature", 0.7)
#         context_window = request.get("context_window", 5)
       
#         relevant_chunks = vector_store.hybrid_search(query=query, k=context_window, filter={"source": document})
#         # relevant_chunks = enhanced_rag.hybrid_search(query=query, k=context_window, filter={"source": document})
#         if not relevant_chunks:
#             return {"response": "I don't have enough information to answer this question based on the document.", "document": document}
#         context = build_focused_context(query, relevant_chunks)
#         if request.get("use_advanced_rag"):
#             kg_context = knowledge_graph.get_kg_context(query)
#             context = kg_context + "\n\n" + context

#         logger.info(f"context: {context}")
#         response = stream_llama_cpp_response(query=query, context=context, model=model, temperature=temperature)
#         # response = enhanced_rag.query(query, document)
#         relevant_chunks = response["chunks_retrieved"]
#         return {"response": response["response"], "sources": [], "document": document}
#     except Exception as e:
#         logger.error(f"Error processing query-sync: {str(e)}")
#         return {"response": f"Error processing your query: {str(e)}. Please try again.", "error": str(e), "success": False}
@app.post("/query-sync")
async def query_sync(request: dict):
    try:
        query = request.get("query")
        document = request.get("document")
        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        if not document:
            document = getattr(app.state, "current_document", None)
            if not document:
                return {"response": "Please upload a document first before querying.", "error": "No document selected"}

        temperature = request.get("temperature", 0.3)
        context_window = request.get("context_window", 5)
        use_llama_index = True
        
        # Use LlamaIndex if requested
        if use_llama_index:
            logger.info(f"Using LlamaIndex for query: {query}")
            try:
                # Query using LlamaIndex
                result = llama_index_rag.query(
                    query_text=query,
                    document_name=document,
                    top_k=context_window
                )
                
                # If LlamaIndex has LLM configured, it returns full response
                if result["response"] and len(result["response"]) > 10:
                    return {
                        "response": result["response"],
                        "sources": result["sources"],
                        "document": document
                    }
                
                # Otherwise, we need to use our local LLM with retrieved context
                chunks_text = "\n\n".join([f"Document: {chunk}" for chunk, _ in result["chunks_retrieved"]])
                if not chunks_text:
                    return {
                        "response": "I don't have enough information to answer this question based on the document.",
                        "document": document
                    }
                
                # Use local model with retrieved context
                response_text = stream_llama_cpp_response(
                    query=query, 
                    context=chunks_text, 
                    model="mamba", 
                    temperature=temperature
                )
                
                return {
                    "response": response_text["response"] if isinstance(response_text, dict) else response_text,
                    "sources": result["sources"],
                    "document": document
                }
                
            except Exception as e:
                logger.error(f"LlamaIndex query failed: {str(e)}, falling back to default method")
                # Continue with original approach as fallback
        
        # Original implementation as fallback
        relevant_chunks = query_index(query, context_window)
        if not relevant_chunks:
            return {"response": "I don't have enough information to answer this question based on the document.", "document": document}

        context = query_index_with_context(query, context_window)
        response_text = stream_llama_cpp_response(query=query, context=context, model="mamba", temperature=temperature)

        return {
            "response": response_text["response"] if isinstance(response_text, dict) else response_text,
            "sources": [],
            "document": document
        }

    except Exception as e:
        logger.error(f"Error processing query-sync: {str(e)}")
        return {
            "response": f"Error processing your query: {str(e)}. Please try again.",
            "error": str(e),
            "success": False
        }


@app.post("/configure")
async def configure(config: dict):
    try:
        config_data = config.get("config", {})
        if not config_data:
            raise HTTPException(status_code=400, detail="No configuration provided")
        logger.info(f"Received configuration update: {config_data}")
        return {"success": True, "message": "Configuration updated", "config": config_data}
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        current_document = request.current_document
        if not current_document:
            current_document = getattr(app.state, "current_document", None)
            if not current_document:
                return {"response": "Please upload a document first before querying.", "error": "No document selected"}
        relevant_chunks = vector_store.search(query=request.query, k=request.context_window, filter={"source": current_document})
        if not relevant_chunks:
            return {"response": "I don't have enough information to answer this question based on the document.", "document": current_document}
        context = build_focused_context(request.query, relevant_chunks)
        if request.use_advanced_rag:
            kg_context = knowledge_graph.get_kg_context(request.query)
            context = kg_context + "\n\n" + context

        response = stream_llama_cpp_response(query=request.query, context=context, model=request.model, temperature=request.temperature, quantization=request.quantization)
        return {"response": response, "sources": [chunk.metadata.get("source", "Unknown") for chunk in relevant_chunks], "document": current_document}
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

@app.get("/models")
async def get_models():
    try:
        models = list_llama_cpp_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.get("/quantization-options")
async def get_quantization_options():
    return {
        "options": [
            {"value": "None", "label": "None (Full Precision)"},
            {"value": "8bit", "label": "8-bit Quantization"},
            {"value": "4bit", "label": "4-bit Quantization (Recommended)"},
            {"value": "1bit", "label": "1-bit Quantization (Fastest, Lower Quality)"}
        ]
    }
@app.get("/kg/neighbors")
async def get_kg_neighbors(entity: str):
    try:
        neighbors = knowledge_graph.get_neighbors(entity)
        return {"entity": entity, "neighbors": neighbors}
    except Exception as e:
        logger.error(f"Error fetching neighbors for {entity}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"KG error: {str(e)}")

@app.get("/kg/triples")
async def get_all_kg_triples():
    try:
        triples = knowledge_graph.get_all_triples()
        return {"triples": triples}
    except Exception as e:
        logger.error(f"Error fetching triples: {str(e)}")
        raise HTTPException(status_code=500, detail=f"KG error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Use fixed port 8000
    port = 8000
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port, workers=1, reload=False)