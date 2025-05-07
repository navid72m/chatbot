import sys
import os
import json
sys.path.append(os.path.dirname(__file__))
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
import logging
import shutil
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from generate_suggestions import generate_suggested_questions
from llama_cpp import Llama

# Import the original modules
from document_processor_patched import process_and_index_file, query_index, query_index_with_context

# Import new LlamaIndex integration (fixed version)
from llama_index_integration_fixed import LlamaIndexRAG

# Import LLM interface
from llama_cpp_interface import stream_llama_cpp_response, list_llama_cpp_models

os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Storage directory for documents
UPLOAD_DIR = os.path.expanduser("~/Library/Application Support/Document Chat/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Storage directory for evaluation results
EVAL_DIR = os.path.expanduser("~/Library/Application Support/Document Chat/evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    model: str = "mistral"
    temperature: float = 0.7
    context_window: int = 5
    quantization: str = "4bit"
    use_advanced_rag: bool = False
    use_llama_index: bool = True  # New parameter to toggle LlamaIndex
    current_document: Optional[str] = None

# Create FastAPI application
app = FastAPI(title="Document Chat Backend")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model paths
MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/mamba-790m-hf.Q4_K_M.gguf")
CTX_SIZE = int(os.environ.get("LLAMA_CTX_SIZE", 2048))
N_THREADS = int(os.environ.get("LLAMA_THREADS", os.cpu_count()))
N_GPU_LAYERS = int(os.environ.get("LLAMA_GPU_LAYERS", -1))

# Global variables
suggested_questions_by_doc = {}

try:
    # Add n_gpu_layers parameter for Apple Silicon
    llm = Llama(
        model_path=MODEL_PATH, 
        n_ctx=CTX_SIZE, 
        n_threads=N_THREADS,
        n_predict=128
    )
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load LLaMA model: {str(e)}")
    llm = None

# Initialize LlamaIndex RAG system
llama_index_rag = LlamaIndexRAG(MODEL_PATH)

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
        "features": ["Vector Search", "Document Q&A", "LlamaIndex RAG"]
    }

@app.get("/kg")
async def kg_view():
    return FileResponse("static/kg.html")

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
        if use_llama_index:
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
        use_llama_index = request.get("use_llama_index", True)
        
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
                
                # If LlamaIndex returns a response, use it
                if result["response"] and len(result["response"]) > 10:
                    return {
                        "response": result["response"],
                        "sources": result["sources"],
                        "document": document,
                        "system": "llama_index"
                    }
                
                # Otherwise, use our local LLM with retrieved context
                chunks_text = "\n\n".join([f"Document: {chunk}" for chunk, _ in result["chunks_retrieved"]])
                if not chunks_text:
                    return {
                        "response": "I don't have enough information to answer this question based on the document.",
                        "document": document,
                        "system": "llama_index"
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
                    "document": document,
                    "system": "llama_index"
                }
                
            except Exception as e:
                logger.error(f"LlamaIndex query failed: {str(e)}, falling back to default method")
                # Continue with original approach as fallback
        
        # Original implementation as fallback
        relevant_chunks = query_index(query, context_window)
        if not relevant_chunks:
            return {"response": "I don't have enough information to answer this question based on the document.", "document": document, "system": "original"}

        context = query_index_with_context(query, context_window)
        response_text = stream_llama_cpp_response(query=query, context=context, model="mamba", temperature=temperature)

        return {
            "response": response_text["response"] if isinstance(response_text, dict) else response_text,
            "sources": [],
            "document": document,
            "system": "original"
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

@app.get("/rag-options")
async def get_rag_options():
    return {
        "options": [
            {"value": "default", "label": "Default RAG"},
            {"value": "llama_index", "label": "LlamaIndex RAG (Advanced)"}
        ]
    }

@app.get("/documents")
async def get_documents():
    try:
        # Get LlamaIndex documents
        llama_index_docs = llama_index_rag.get_document_list()
        
        # Get all uploaded documents
        all_docs = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
        
        return {
            "all_documents": all_docs,
            "indexed_documents": llama_index_docs
        }
    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")

# Simple evaluation endpoint
@app.post("/evaluate/basic")
async def basic_evaluation(request: dict):
    try:
        document_name = request.get("document")
        if not document_name:
            document_name = getattr(app.state, "current_document", None)
            if not document_name:
                raise HTTPException(status_code=400, detail="No document specified")
                
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        
        # Run query with both systems
        # 1. Original system
        original_context = query_index_with_context(query, 5)
        original_response = stream_llama_cpp_response(
            query=query, 
            context=original_context, 
            model="mamba", 
            temperature=0.3
        )
        
        # 2. LlamaIndex system
        llama_index_result = llama_index_rag.query(
            query_text=query,
            document_name=document_name,
            top_k=5
        )
        
        llama_index_response = llama_index_result["response"]
        if not llama_index_response or len(llama_index_response) < 10:
            chunks_text = "\n\n".join([f"Document: {chunk}" for chunk, _ in llama_index_result["chunks_retrieved"]])
            llama_index_response = stream_llama_cpp_response(
                query=query, 
                context=chunks_text, 
                model="mamba", 
                temperature=0.3
            )
            if isinstance(llama_index_response, dict):
                llama_index_response = llama_index_response["response"]
        
        # Save results to evaluation directory
        eval_result = {
            "document": document_name,
            "query": query,
            "timestamp": str(datetime.datetime.now()),
            "original": {
                "response": original_response["response"] if isinstance(original_response, dict) else original_response,
                "context": original_context
            },
            "llama_index": {
                "response": llama_index_response,
                "chunks": [(chunk, float(score)) for chunk, score in llama_index_result["chunks_retrieved"]]
            }
        }
        
        # Return evaluation result
        return {
            "success": True,
            "document": document_name,
            "evaluation": eval_result
        }
        
    except Exception as e:
        logger.error(f"Error in basic evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in evaluation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import argparse
    import datetime
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Document Chat Backend with LlamaIndex")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the LLM model")
    
    args = parser.parse_args()
    
    # Start the web server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1, reload=False)