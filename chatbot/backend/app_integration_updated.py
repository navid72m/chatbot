import os
# CRITICAL: Set these BEFORE importing ML libraries
os.environ["NNPACK_DISABLE"] = "1"
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
# import os
import json
import time
import logging
import shutil
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from llama_cpp import Llama

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import enhanced document processor with smart retrieval
from updated_document_processor import (
    process_and_index_file, 
    query_index, 
    query_index_with_context,
    get_smart_processing_stats,
    smart_rag_pipeline
)

# Import query rewriting components (if available)
try:
    from enhanced_llama_index_integration import EnhancedLlamaIndexRAG
    QUERY_REWRITING_AVAILABLE = True
    logging.info("Query rewriting components loaded successfully")
except ImportError as e:
    logging.warning(f"Enhanced query rewriting not available: {e}")
    QUERY_REWRITING_AVAILABLE = False

# Import other components
from generate_suggestions import generate_suggested_questions
from llama_cpp_interface import stream_llama_cpp_response, list_llama_cpp_models

# Environment setup
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Storage directories
UPLOAD_DIR = os.path.expanduser("~/Library/Application Support/Document Chat/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

EVAL_DIR = os.path.expanduser("~/Library/Application Support/Document Chat/evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    model: str = "mistral"
    temperature: float = 0.7
    context_window: int = 5
    quantization: str = "4bit"
    use_advanced_rag: bool = False
    use_llama_index: bool = True
    current_document: Optional[str] = None
    # Smart retrieval parameters
    use_smart_retrieval: bool = True
    # Query rewriting parameters
    use_prf: bool = True
    use_variants: bool = True
    prf_iterations: int = 1
    fusion_method: str = "rrf"
    rerank: bool = True

class SmartProcessingConfig(BaseModel):
    use_smart_processing: bool = True
    extract_entities: bool = True
    create_smart_chunks: bool = True
    enable_smart_search: bool = True
    auto_detect_document_type: bool = True

class QueryRewritingConfig(BaseModel):
    use_prf: bool = True
    use_variants: bool = True
    prf_iterations: int = 1
    fusion_method: str = "rrf"
    rerank: bool = True
    prf_top_k: int = 3
    expansion_terms: int = 5
    variant_count: int = 3

# Create FastAPI application
app = FastAPI(title="Smart Document Chat Backend with Universal Retrieval + Query Rewriting")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model configuration
MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/mamba-790m-hf.Q4_K_M.gguf")
CTX_SIZE = int(os.environ.get("LLAMA_CTX_SIZE", 2048))
N_THREADS = int(os.environ.get("LLAMA_THREADS", os.cpu_count()))
N_GPU_LAYERS = int(os.environ.get("LLAMA_GPU_LAYERS", -1))

# Global variables
suggested_questions_by_doc = {}
document_processing_results = {}

# Initialize LLM
try:
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

# Initialize enhanced RAG with query rewriting
enhanced_rag_with_rewriting = None
if QUERY_REWRITING_AVAILABLE:
    try:
        enhanced_rag_with_rewriting = EnhancedLlamaIndexRAG(MODEL_PATH)
        logger.info("Enhanced RAG with query rewriting initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize enhanced RAG with query rewriting: {e}")

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
    features = [
        "Universal Entity Extraction",
        "Smart Document Type Detection", 
        "Intelligent Chunking",
        "Adaptive Query Processing",
        "Multi-Strategy Retrieval"
    ]
    
    if QUERY_REWRITING_AVAILABLE:
        features.extend([
            "Pseudo Relevance Feedback (PRF)",
            "Query Variants Generation", 
            "Reciprocal Rank Fusion (RRF)",
            "Cross-Encoder Reranking",
            "Multi-Query Processing"
        ])
    
    return {
        "message": "Smart Document Chat Backend with Universal Retrieval + Query Rewriting",
        "features": features,
        "supported_documents": ["PDF", "Text", "Images", "Resumes", "Reports", "Contracts", "Manuals"],
        "query_rewriting": QUERY_REWRITING_AVAILABLE,
        "version": "2.0"
    }

@app.get("/kg")
async def kg_view():
    return FileResponse("static/kg.html")

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    use_advanced_rag: bool = Form(False),
    use_llama_index: bool = Form(True),
    use_smart_processing: bool = Form(True)
):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing {file.filename} with smart processing: {use_smart_processing}")
        
        # Process with enhanced document processor
        processing_result = process_and_index_file(file_path, use_smart_processing)
        
        # Also process with query rewriting system if available
        rewriting_result = None
        if QUERY_REWRITING_AVAILABLE and enhanced_rag_with_rewriting and use_llama_index:
            try:
                logger.info("Also processing with query rewriting system")
                rewriting_result = enhanced_rag_with_rewriting.process_document_robust(file_path)
                logger.info("Query rewriting system processing complete")
            except Exception as e:
                logger.error(f"Query rewriting processing failed: {str(e)}")
        
        # Handle different return formats
        if len(processing_result) == 3:
            chunks, suggested_questions, smart_result = processing_result
            document_processing_results[file.filename] = smart_result
            
            response_data = {
                "success": True,
                "filename": file.filename,
                "chunks": len(chunks),
                "preview": f"Smart processing complete. Document type: {smart_result.get('document_type', 'general')}",
                "smart_processing": True,
                "document_type": smart_result.get("document_type", "general"),
                "entities_found": smart_result.get("entities_found", 0),
                "key_entities": [
                    {
                        "text": entity.text,
                        "type": entity.label,
                        "confidence": entity.confidence
                    } for entity in smart_result.get("key_entities", [])[:5]
                ],
                "query_rewriting_ready": rewriting_result is not None
            }
        else:
            chunks, suggested_questions = processing_result
            response_data = {
                "success": True,
                "filename": file.filename,
                "chunks": len(chunks),
                "preview": "Traditional processing complete",
                "smart_processing": False,
                "query_rewriting_ready": rewriting_result is not None
            }
        
        suggested_questions_by_doc[file.filename] = suggested_questions
        return response_data
        
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

        # Get parameters
        temperature = request.get("temperature", 0.3)
        context_window = request.get("context_window", 5)
        use_smart_retrieval = request.get("use_smart_retrieval", True)
        
        # Query rewriting parameters
        use_prf = request.get("use_prf", True)
        use_variants = request.get("use_variants", True)
        prf_iterations = request.get("prf_iterations", 1)
        fusion_method = request.get("fusion_method", "rrf")
        rerank = request.get("rerank", True)
        
        logger.info(f"Processing query with smart retrieval: {use_smart_retrieval}, PRF: {use_prf}, variants: {use_variants}")
        
        # Try to set document context
        try:
            await set_document({"document": document})
        except:
            pass
        
        # Strategy 1: Enhanced RAG with Query Rewriting (highest priority)
        if QUERY_REWRITING_AVAILABLE and enhanced_rag_with_rewriting and (use_prf or use_variants):
            try:
                logger.info("Using enhanced RAG with query rewriting")
                
                enhanced_result = enhanced_rag_with_rewriting.query_robust(
                    query_text=query,
                    document_name=document,
                    top_k=context_window,
                    use_robust_retrieval=True,
                    use_prf=use_prf,
                    use_variants=use_variants,
                    prf_iterations=prf_iterations,
                    fusion_method=fusion_method,
                    rerank=rerank,
                    temperature=temperature
                )
                
                return {
                    "response": enhanced_result["response"],
                    "sources": enhanced_result["sources"],
                    "document": document,
                    "system": enhanced_result.get("system", "enhanced_with_query_rewriting"),
                    "query_rewriting_info": {
                        "prf_applied": use_prf,
                        "variants_generated": use_variants,
                        "fusion_method": fusion_method,
                        "reranking_applied": rerank,
                        "iterations": prf_iterations,
                        "all_queries": enhanced_result.get("query_rewriting", {}).get("all_queries", [query])
                    },
                    "entity_matches": enhanced_result.get("entity_matches", []),
                    "enhancement_info": enhanced_result.get("enhancement_info", {}),
                    "enhancement_applied": True
                }
                
            except Exception as e:
                logger.error(f"Enhanced RAG with query rewriting failed: {str(e)}")
        
        # Strategy 2: Smart RAG without query rewriting
        if use_smart_retrieval and smart_rag_pipeline:
            try:
                logger.info("Using smart RAG without query rewriting")
                
                smart_result = smart_rag_pipeline.query_smart(query, document, context_window)
                
                response_text = stream_llama_cpp_response(
                    query=query,
                    context=smart_result["context"],
                    model="mamba",
                    temperature=temperature
                )
                
                return {
                    "response": response_text["response"] if isinstance(response_text, dict) else response_text,
                    "sources": [document],
                    "document": document,
                    "system": "smart_rag",
                    "smart_retrieval_info": {
                        "query_analysis": smart_result["query_analysis"],
                        "entity_info": smart_result["entity_info"],
                        "retrieval_strategy": smart_result["retrieval_strategy"],
                        "chunks_searched": smart_result["total_chunks_searched"]
                    },
                    "query_rewriting_info": {
                        "available": QUERY_REWRITING_AVAILABLE,
                        "applied": False,
                        "reason": "Smart retrieval used instead"
                    },
                    "enhancement_applied": True
                }
                
            except Exception as e:
                logger.error(f"Smart retrieval failed: {str(e)}")
        
        # Strategy 3: Enhanced traditional retrieval
        try:
            logger.info("Using enhanced traditional retrieval")
            
            context = query_index_with_context(query, context_window, use_smart_retrieval=False)
            
            if not context.strip():
                return {
                    "response": "I don't have enough information to answer this question based on the document.",
                    "document": document,
                    "system": "traditional_fallback"
                }
            
            response_text = stream_llama_cpp_response(
                query=query,
                context=context,
                model="mamba",
                temperature=temperature
            )
            
            return {
                "response": response_text["response"] if isinstance(response_text, dict) else response_text,
                "sources": [document],
                "document": document,
                "system": "enhanced_traditional",
                "query_rewriting_info": {
                    "available": QUERY_REWRITING_AVAILABLE,
                    "applied": False,
                    "reason": "Fallback to traditional retrieval"
                },
                "enhancement_applied": False
            }
            
        except Exception as e:
            logger.error(f"Enhanced traditional retrieval failed: {str(e)}")
            
            # Final fallback
            return {
                "response": "I encountered an error processing your query. Please try again.",
                "document": document,
                "system": "error_fallback",
                "error": str(e)
            }

    except Exception as e:
        logger.error(f"Error processing query-sync: {str(e)}")
        return {
            "response": f"Error processing your query: {str(e)}. Please try again.",
            "error": str(e),
            "success": False
        }

@app.post("/configure-query-rewriting")
async def configure_query_rewriting(config: QueryRewritingConfig):
    try:
        logger.info(f"Updating query rewriting configuration: {config.dict()}")
        app.state.query_rewriting_config = config.dict()
        return {
            "success": True, 
            "message": "Query rewriting configuration updated",
            "config": config.dict(),
            "available": QUERY_REWRITING_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Error updating query rewriting configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@app.get("/query-rewriting-config")
async def get_query_rewriting_config():
    try:
        default_config = QueryRewritingConfig().dict()
        current_config = getattr(app.state, "query_rewriting_config", default_config)
        return {
            "config": current_config,
            "available": QUERY_REWRITING_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Error fetching query rewriting configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/query-rewriting-options")
async def get_query_rewriting_options():
    return {
        "available": QUERY_REWRITING_AVAILABLE,
        "techniques": [
            {"value": "prf", "label": "Pseudo Relevance Feedback (PRF)", 
             "description": "Expand queries using terms from relevant documents"},
            {"value": "variants", "label": "Query Variants", 
             "description": "Generate multiple query reformulations"},
            {"value": "reranking", "label": "Cross-Encoder Reranking", 
             "description": "Rerank results using advanced models"},
            {"value": "fusion", "label": "Result Fusion", 
             "description": "Combine results from multiple queries"}
        ],
        "fusion_methods": [
            {"value": "rrf", "label": "Reciprocal Rank Fusion", 
             "description": "Combine rankings using reciprocal rank scores"},
            {"value": "score", "label": "Score-based Fusion", 
             "description": "Average similarity scores across queries"}
        ],
        "parameters": {
            "prf_iterations": {"min": 1, "max": 3, "default": 1, 
                             "description": "Number of PRF feedback iterations"},
            "prf_top_k": {"min": 1, "max": 10, "default": 3, 
                         "description": "Number of top documents for feedback"},
            "expansion_terms": {"min": 1, "max": 10, "default": 5, 
                              "description": "Number of terms to add in expansion"},
            "variant_count": {"min": 1, "max": 5, "default": 3, 
                            "description": "Number of query variants to generate"}
        }
    }

@app.get("/models")
async def get_models():
    try:
        models = list_llama_cpp_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.get("/rag-options")
async def get_rag_options():
    options = [
        {"value": "smart", "label": "Smart Universal RAG (Recommended)", 
         "description": "Intelligent retrieval for all document types"},
        {"value": "enhanced_traditional", "label": "Enhanced Traditional RAG", 
         "description": "Improved traditional approach"},
        {"value": "basic", "label": "Basic RAG", 
         "description": "Simple semantic similarity search"}
    ]
    
    if QUERY_REWRITING_AVAILABLE:
        options.insert(0, {
            "value": "enhanced_with_rewriting", 
            "label": "Enhanced RAG with Query Rewriting (Best)", 
            "description": "Advanced retrieval with PRF, variants, and fusion"
        })
    
    return {"options": options}

@app.get("/documents")
async def get_documents():
    try:
        all_docs = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
        smart_processed_docs = list(document_processing_results.keys())
        
        detailed_docs = []
        for doc in all_docs:
            doc_info = {
                "name": doc,
                "smart_processed": doc in smart_processed_docs,
                "query_rewriting_ready": QUERY_REWRITING_AVAILABLE
            }
            
            if doc in document_processing_results:
                smart_result = document_processing_results[doc]
                doc_info.update({
                    "document_type": smart_result.get("document_type", "general"),
                    "entities_found": smart_result.get("entities_found", 0),
                    "chunks_created": smart_result.get("chunks_created", 0)
                })
            
            detailed_docs.append(doc_info)
        
        return {
            "all_documents": all_docs,
            "smart_processed_documents": smart_processed_docs,
            "detailed_info": detailed_docs,
            "smart_rag_enabled": True,
            "query_rewriting_enabled": QUERY_REWRITING_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")

@app.get("/processing-stats")
async def get_processing_stats():
    try:
        stats = get_smart_processing_stats()
        return {"stats": stats, "query_rewriting_available": QUERY_REWRITING_AVAILABLE}
    except Exception as e:
        logger.error(f"Error fetching processing stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/evaluate/enhanced")
async def enhanced_evaluation(request: dict):
    try:
        document_name = request.get("document")
        if not document_name:
            document_name = getattr(app.state, "current_document", None)
            if not document_name:
                raise HTTPException(status_code=400, detail="No document specified")
                
        query = request.get("query", "What is this document about and who is mentioned?")
        use_prf = request.get("use_prf", True)
        use_variants = request.get("use_variants", True)
        
        evaluation_results = {}
        
        # Test enhanced RAG with query rewriting
        if QUERY_REWRITING_AVAILABLE and enhanced_rag_with_rewriting:
            start_time = time.time()
            try:
                result = enhanced_rag_with_rewriting.query_robust(
                    query_text=query,
                    document_name=document_name,
                    top_k=5,
                    use_robust_retrieval=True,
                    use_prf=use_prf,
                    use_variants=use_variants,
                    temperature=0.3
                )
                
                evaluation_results["enhanced_with_query_rewriting"] = {
                    "response": result["response"],
                    "time_ms": (time.time() - start_time) * 1000,
                    "success": True,
                    "entity_matches": result.get("entity_matches", [])
                }
            except Exception as e:
                evaluation_results["enhanced_with_query_rewriting"] = {
                    "response": f"Failed: {str(e)}",
                    "time_ms": (time.time() - start_time) * 1000,
                    "success": False,
                    "error": str(e)
                }
        
        # Test smart RAG
        start_time = time.time()
        try:
            if smart_rag_pipeline:
                smart_result = smart_rag_pipeline.query_smart(query, document_name, 5)
                smart_response = stream_llama_cpp_response(
                    query=query, context=smart_result["context"], model="mamba", temperature=0.3
                )
                
                evaluation_results["smart_rag"] = {
                    "response": smart_response["response"] if isinstance(smart_response, dict) else smart_response,
                    "time_ms": (time.time() - start_time) * 1000,
                    "success": True,
                    "entity_info": smart_result.get("entity_info", {})
                }
            else:
                evaluation_results["smart_rag"] = {
                    "response": "Smart RAG not available",
                    "time_ms": 0,
                    "success": False
                }
        except Exception as e:
            evaluation_results["smart_rag"] = {
                "response": f"Failed: {str(e)}",
                "time_ms": (time.time() - start_time) * 1000,
                "success": False
            }
        
        # Test traditional RAG
        start_time = time.time()
        try:
            context = query_index_with_context(query, 5, use_smart_retrieval=False)
            response = stream_llama_cpp_response(query=query, context=context, model="mamba", temperature=0.3)
            
            evaluation_results["traditional_rag"] = {
                "response": response["response"] if isinstance(response, dict) else response,
                "time_ms": (time.time() - start_time) * 1000,
                "success": True
            }
        except Exception as e:
            evaluation_results["traditional_rag"] = {
                "response": f"Failed: {str(e)}",
                "time_ms": (time.time() - start_time) * 1000,
                "success": False
            }
        
        return {
            "success": True,
            "document": document_name,
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_results": evaluation_results,
            "query_rewriting_available": QUERY_REWRITING_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Error in enhanced evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in evaluation: {str(e)}")

@app.post("/clear-caches")
async def clear_caches():
    try:
        if smart_rag_pipeline and hasattr(smart_rag_pipeline, 'vector_store'):
            if hasattr(smart_rag_pipeline.vector_store, 'retrieval_cache'):
                smart_rag_pipeline.vector_store.retrieval_cache.clear()
        
        if QUERY_REWRITING_AVAILABLE and enhanced_rag_with_rewriting:
            try:
                enhanced_rag_with_rewriting.clear_caches()
            except:
                pass
        
        return {"success": True, "message": "All caches cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing caches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing caches: {str(e)}")

import sys
import os
import json
import logging
import io
import shutil
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

# Add these route handlers to your existing FastAPI app in app_integration_updated.py

@app.get("/document/{document_name}")
async def get_document(document_name: str):
    """
    Serve the document file
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, document_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {document_name} not found")
        
        return FileResponse(
            path=file_path, 
            filename=document_name,
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

@app.get("/document-highlights")
async def get_document_highlights(
    document: str,
    response_id: str
):
    """
    Get all highlights for a document associated with a specific response
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, document)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {document} not found")
        
        # In a real implementation, fetch stored highlights from your database or tracking system
        # For now, we'll return mock data for demonstration
        
        # Check if we can access the tracked context
        highlights = []
        if hasattr(smart_rag_pipeline, 'vector_store') and hasattr(smart_rag_pipeline.vector_store, 'context_tracking'):
            tracked_context = smart_rag_pipeline.vector_store.context_tracking.get(response_id)
            if tracked_context:
                highlights = tracked_context.get('ranges', [])
        
        # If no highlights found, generate some mock data
        if not highlights:
            # Mock data - in a real implementation, these would be accurate coordinates
            # based on the actual context used for retrieval
            highlights = [
                {
                    "page": 1,
                    "top": 150,
                    "left": 72,
                    "width": 450,
                    "height": 80,
                    "text": "The document was processed with smart processing enabled.",
                    "relevance_score": 0.92
                },
                {
                    "page": 1,
                    "top": 350,
                    "left": 72,
                    "width": 450,
                    "height": 60,
                    "text": "Enhanced RAG with query rewriting was applied to improve retrieval quality.",
                    "relevance_score": 0.87
                },
                {
                    "page": 2,
                    "top": 200,
                    "left": 72,
                    "width": 450,
                    "height": 100,
                    "text": "The system combines PRF, query variants, and cross-encoder reranking for optimal results.",
                    "relevance_score": 0.95
                }
            ]
        
        return {
            "document": document,
            "response_id": response_id,
            "highlights": highlights
        }
    except Exception as e:
        logger.error(f"Error getting document highlights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document highlights: {str(e)}")

@app.post("/query-with-context")
async def query_with_context_tracking(request: dict):
    """
    Enhanced query endpoint that tracks context and source locations
    """
    try:
        query = request.get("query")
        document = request.get("document")
        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        if not document:
            document = getattr(app.state, "current_document", None)
            if not document:
                return {"response": "Please upload a document first before querying.", "error": "No document selected"}

        # Get parameters
        temperature = request.get("temperature", 0.3)
        context_window = request.get("context_window", 5)
        use_smart_retrieval = request.get("use_smart_retrieval", True)
        
        # Query rewriting parameters
        use_prf = request.get("use_prf", True)
        use_variants = request.get("use_variants", True)
        prf_iterations = request.get("prf_iterations", 1)
        fusion_method = request.get("fusion_method", "rrf")
        rerank = request.get("rerank", True)
        
        # Generate a unique ID for context tracking
        import uuid
        response_id = str(uuid.uuid4())
        
        # Run the regular query with additional context tracking
        result = await query_sync(request)
        
        # In a real implementation, you would extract the context locations during retrieval
        # For now, we'll add mock data for demonstration
        
        # For PDF documents, include page numbers and coordinates for visualization
        if document.lower().endswith('.pdf'):
            # Create mock context ranges
            # In a real implementation, these would be the actual locations in the PDF
            # where the text that was used for retrieval is located
            import random
            
            context_ranges = [
                {
                    "page": 1,
                    "top": 150,
                    "left": 72,
                    "width": 450,
                    "height": 80,
                    "text": "The document was processed with smart processing enabled.",
                    "relevance_score": 0.92
                },
                {
                    "page": 1,
                    "top": 350,
                    "left": 72,
                    "width": 450,
                    "height": 60,
                    "text": "Enhanced RAG with query rewriting was applied to improve retrieval quality.",
                    "relevance_score": 0.87
                },
                {
                    "page": 2,
                    "top": 200,
                    "left": 72,
                    "width": 450,
                    "height": 100,
                    "text": "The system combines PRF, query variants, and cross-encoder reranking for optimal results.",
                    "relevance_score": 0.95
                }
            ]
            
            context_text = "\n\n".join([r["text"] for r in context_ranges])
            
            # Add context information to the result
            result["context_tracking"] = {
                "response_id": response_id,
                "context_ranges": context_ranges,
                "context_text": context_text
            }
            
            # In a real implementation, store this information in your tracking system
            # For example:
            if hasattr(smart_rag_pipeline, 'vector_store'):
                if not hasattr(smart_rag_pipeline.vector_store, 'context_tracking'):
                    smart_rag_pipeline.vector_store.context_tracking = {}
                
                smart_rag_pipeline.vector_store.context_tracking[response_id] = {
                    "ranges": context_ranges,
                    "text": context_text,
                    "query": query,
                    "timestamp": str(datetime.datetime.now())
                }
        
        return result
    except Exception as e:
        logger.error(f"Error in query with context tracking: {str(e)}")
        return {
            "response": f"Error processing your query: {str(e)}. Please try again.",
            "error": str(e),
            "success": False
        } 
@app.get("/context-info")
async def get_context_info(
    document: str, 
    query_id: str, 
    response_id: str
):
    """
    Get context information for a specific query-response pair
    """
    try:
        # Check if we have the document
        file_path = os.path.join(UPLOAD_DIR, document)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {document} not found")
        
        # This is where you would look up the actual context used for this response
        # For now, we'll return mock data for PDF context visualization
        
        # For PDF documents, we need to include page numbers and coordinates
        if document.lower().endswith('.pdf'):
            # Mock data for demonstration - in a real implementation, retrieve from your retrieval system
            context_ranges = []
            context_text = ""
            
            # Check if we have enhanced context tracking in our vector store
            if hasattr(smart_rag_pipeline, 'vector_store') and hasattr(smart_rag_pipeline.vector_store, 'context_tracking'):
                # Try to get tracked context for this specific response
                tracked_context = smart_rag_pipeline.vector_store.context_tracking.get(response_id)
                
                if tracked_context:
                    context_ranges = tracked_context.get('ranges', [])
                    context_text = tracked_context.get('text', "")
                else:
                    # Fall back to generating some mock data based on the document
                    # In a real implementation, you should extract actual context locations
                    
                    # Let's create some mock context ranges for demonstration
                    import random
                    context_ranges = [
                        {
                            "page": 1,
                            "top": 150,
                            "left": 72,
                            "width": 450,
                            "height": 80,
                            "text": "The document was processed with smart processing enabled."
                        },
                        {
                            "page": 1,
                            "top": 350,
                            "left": 72,
                            "width": 450,
                            "height": 60,
                            "text": "Enhanced RAG with query rewriting was applied to improve retrieval quality."
                        },
                        {
                            "page": 2,
                            "top": 200,
                            "left": 72,
                            "width": 450,
                            "height": 100,
                            "text": "The system combines PRF, query variants, and cross-encoder reranking for optimal results."
                        }
                    ]
                    
                    context_text = "\n\n".join([r["text"] for r in context_ranges])
            
            return {
                "document": document,
                "query_id": query_id,
                "response_id": response_id,
                "context_ranges": context_ranges,
                "context_text": context_text
            }
        else:
            # For non-PDF documents, we can just return text ranges
            return {
                "document": document,
                "query_id": query_id,
                "response_id": response_id,
                "context_ranges": [],
                "context_text": "Context visualization is currently only supported for PDF documents."
            }
    except Exception as e:
        logger.error(f"Error getting context info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting context info: {str(e)}")

@app.post("/extract-text")
async def extract_page_text(request: dict):
    """
    Extract text from a specific page region of a PDF document
    """
    try:
        document_name = request.get("document")
        page_number = request.get("page")
        coords = request.get("coords")
        
        if not document_name or not page_number or not coords:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        file_path = os.path.join(UPLOAD_DIR, document_name)
        if not os.path.exists(file_path) or not document_name.lower().endswith('.pdf'):
            raise HTTPException(status_code=404, detail=f"PDF document {document_name} not found")
        
        # In a real implementation, you would use a PDF parsing library 
        # to extract text from the specified coordinates
        # For now, we'll return mock data
        
        return {
            "text": "Extracted text would appear here in a real implementation.",
            "page": page_number,
            "coords": coords
        }
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")
    

# Add this to your app_integration_updated.py file

@app.get("/document-text/{document_name}")
async def get_document_text(document_name: str):
    """
    Get the extracted text content from a document, especially useful for images
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, document_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {document_name} not found")
        
        # Get file extension
        file_extension = os.path.splitext(document_name)[1].lower()
        
        # Check if it's an image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
        is_image = file_extension in image_extensions
        
        if is_image:
            # Extract text using our OCR function
            from updated_document_processor import extract_text_from_image
            extracted_text = extract_text_from_image(file_path)
            
            # Get image metadata
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    image_info = {
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "mode": img.mode,
                        "size_bytes": os.path.getsize(file_path),
                        "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                    }
            except Exception as e:
                image_info = {
                    "error": f"Could not read image metadata: {e}",
                    "size_bytes": os.path.getsize(file_path)
                }
            
            return {
                "document": document_name,
                "is_image": True,
                "extracted_text": extracted_text,
                "image_info": image_info,
                "text_length": len(extracted_text),
                "extraction_method": "OCR (Tesseract/Apple Vision)",
                "image_url": f"/document/{document_name}"  # For thumbnail display
            }
        
        elif file_extension == '.pdf':
            # Extract text from PDF
            from updated_document_processor import extract_pdf_text
            extracted_text = extract_pdf_text(file_path)
            
            return {
                "document": document_name,
                "is_image": False,
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "extraction_method": "PDF text extraction"
            }
        
        elif file_extension in ['.txt', '.md', '.csv', '.json', '.xml', '.html']:
            # Extract text from text files
            from updated_document_processor import extract_text_file
            extracted_text = extract_text_file(file_path)
            
            return {
                "document": document_name,
                "is_image": False,
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "extraction_method": "Direct text reading"
            }
        
        else:
            return {
                "document": document_name,
                "is_image": False,
                "extracted_text": "[Unsupported file type for text extraction]",
                "text_length": 0,
                "extraction_method": "Not supported"
            }
    
    except Exception as e:
        logger.error(f"Error extracting document text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting document text: {str(e)}")

@app.get("/document-thumbnail/{document_name}")
async def get_document_thumbnail(document_name: str, size: int = 200):
    """
    Get a thumbnail of an image document
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, document_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {document_name} not found")
        
        # Check if it's an image
        file_extension = os.path.splitext(document_name)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
        
        if file_extension not in image_extensions:
            raise HTTPException(status_code=400, detail="File is not an image")
        
        try:
            from PIL import Image
            import io
            
            # Open and resize image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (for JPEG output)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Calculate thumbnail size maintaining aspect ratio
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                
                # Save to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                img_byte_arr.seek(0)
                
                return StreamingResponse(
                    io.BytesIO(img_byte_arr.read()),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"}
                )
        
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating thumbnail: {e}")
            
    except Exception as e:
        logger.error(f"Error processing thumbnail request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing thumbnail request: {str(e)}")

@app.get("/document-preview/{document_name}")
async def get_document_preview(document_name: str, max_length: int = 1000):
    """
    Get a preview of the document text (first max_length characters)
    """
    try:
        # Get full text
        full_text_response = await get_document_text(document_name)
        
        if "error" in full_text_response:
            return full_text_response
        
        extracted_text = full_text_response.get("extracted_text", "")
        
        # Create preview
        if len(extracted_text) <= max_length:
            preview_text = extracted_text
            is_truncated = False
        else:
            # Find a good breaking point (end of sentence or word)
            truncate_point = max_length
            
            # Try to break at sentence end
            last_sentence = extracted_text[:max_length].rfind('.')
            if last_sentence > max_length * 0.7:  # If we found a sentence end in the last 30%
                truncate_point = last_sentence + 1
            else:
                # Try to break at word boundary
                last_space = extracted_text[:max_length].rfind(' ')
                if last_space > max_length * 0.8:  # If we found a space in the last 20%
                    truncate_point = last_space
            
            preview_text = extracted_text[:truncate_point].strip()
            is_truncated = True
        
        return {
            "document": document_name,
            "is_image": full_text_response.get("is_image", False),
            "preview_text": preview_text,
            "is_truncated": is_truncated,
            "full_length": len(extracted_text),
            "preview_length": len(preview_text),
            "extraction_method": full_text_response.get("extraction_method", "Unknown"),
            "image_info": full_text_response.get("image_info"),
            "image_url": full_text_response.get("image_url")
        }
        
    except Exception as e:
        logger.error(f"Error getting document preview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document preview: {str(e)}")


@app.get("/document-preview/{document_name}")
async def get_document_preview(document_name: str, max_length: int = 1000):
    """
    Get a preview of the document text (first max_length characters)
    """
    try:
        # Get full text
        full_text_response = await get_document_text(document_name)
        
        if "error" in full_text_response:
            return full_text_response
        
        extracted_text = full_text_response.get("extracted_text", "")
        
        # Create preview
        if len(extracted_text) <= max_length:
            preview_text = extracted_text
            is_truncated = False
        else:
            # Find a good breaking point (end of sentence or word)
            truncate_point = max_length
            
            # Try to break at sentence end
            last_sentence = extracted_text[:max_length].rfind('.')
            if last_sentence > max_length * 0.7:  # If we found a sentence end in the last 30%
                truncate_point = last_sentence + 1
            else:
                # Try to break at word boundary
                last_space = extracted_text[:max_length].rfind(' ')
                if last_space > max_length * 0.8:  # If we found a space in the last 20%
                    truncate_point = last_space
            
            preview_text = extracted_text[:truncate_point].strip()
            is_truncated = True
        
        return {
            "document": document_name,
            "is_image": full_text_response.get("is_image", False),
            "preview_text": preview_text,
            "is_truncated": is_truncated,
            "full_length": len(extracted_text),
            "preview_length": len(preview_text),
            "extraction_method": full_text_response.get("extraction_method", "Unknown"),
            "image_info": full_text_response.get("image_info")
        }
        
    except Exception as e:
        logger.error(f"Error getting document preview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document preview: {str(e)}")

@app.post("/reprocess-image")
async def reprocess_image_ocr(request: dict):
    """
    Reprocess an image with different OCR settings
    """
    try:
        document_name = request.get("document")
        ocr_method = request.get("method", "auto")  # auto, tesseract, apple_vision
        
        if not document_name:
            raise HTTPException(status_code=400, detail="Document name required")
        
        file_path = os.path.join(UPLOAD_DIR, document_name)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Document {document_name} not found")
        
        # Check if it's an image
        file_extension = os.path.splitext(document_name)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']
        
        if file_extension not in image_extensions:
            raise HTTPException(status_code=400, detail="File is not an image")
        
        # Import OCR functions
        from updated_document_processor import (
            try_tesseract_ocr_emergency, 
            try_apple_vision_ocr_emergency,
            extract_text_from_image
        )
        
        results = {}
        
        if ocr_method == "auto" or ocr_method == "tesseract":
            # Try Tesseract
            tesseract_result = try_tesseract_ocr_emergency(file_path)
            results["tesseract"] = {
                "success": tesseract_result is not None,
                "text": tesseract_result or "Failed to extract text",
                "text_length": len(tesseract_result) if tesseract_result else 0
            }
        
        if ocr_method == "auto" or ocr_method == "apple_vision":
            # Try Apple Vision
            vision_result = try_apple_vision_ocr_emergency(file_path)
            results["apple_vision"] = {
                "success": vision_result is not None,
                "text": vision_result or "Failed to extract text",
                "text_length": len(vision_result) if vision_result else 0
            }
        
        if ocr_method == "auto":
            # Use the best result
            best_method = None
            best_text = ""
            best_length = 0
            
            for method, result in results.items():
                if result["success"] and result["text_length"] > best_length:
                    best_method = method
                    best_text = result["text"]
                    best_length = result["text_length"]
            
            return {
                "document": document_name,
                "reprocessing_method": ocr_method,
                "best_method": best_method,
                "extracted_text": best_text,
                "text_length": best_length,
                "all_results": results
            }
        else:
            # Return specific method result
            if ocr_method in results:
                result = results[ocr_method]
                return {
                    "document": document_name,
                    "reprocessing_method": ocr_method,
                    "extracted_text": result["text"],
                    "text_length": result["text_length"],
                    "success": result["success"]
                }
            else:
                raise HTTPException(status_code=400, detail=f"Unknown OCR method: {ocr_method}")
        
    except Exception as e:
        logger.error(f"Error reprocessing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reprocessing image: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Document Chat Backend with Universal Retrieval + Query Rewriting")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the LLM model")
    parser.add_argument("--disable-smart", action="store_true", help="Disable smart processing by default")
    parser.add_argument("--disable-query-rewriting", action="store_true", help="Disable query rewriting features")
    
    args = parser.parse_args()
    
    # Set default configurations
    default_smart_config = SmartProcessingConfig(
        use_smart_processing=not args.disable_smart,
        extract_entities=True,
        create_smart_chunks=True,
        enable_smart_search=True,
        auto_detect_document_type=True
    )
    app.state.smart_processing_config = default_smart_config.dict()
    
    default_query_rewriting_config = QueryRewritingConfig(
        use_prf=not args.disable_query_rewriting and QUERY_REWRITING_AVAILABLE,
        use_variants=not args.disable_query_rewriting and QUERY_REWRITING_AVAILABLE,
        prf_iterations=1,
        fusion_method="rrf",
        rerank=True
    )
    app.state.query_rewriting_config = default_query_rewriting_config.dict()
    
    # Print startup information
    logger.info(f"🚀 Starting Smart Document Chat Backend on {args.host}:{args.port}")
    logger.info(f"Smart processing: {'enabled' if not args.disable_smart else 'disabled'}")
    logger.info(f"Query rewriting: {'enabled' if QUERY_REWRITING_AVAILABLE and not args.disable_query_rewriting else 'disabled'}")
    
    logger.info("✅ Available features:")
    logger.info("  • Universal document type detection")
    logger.info("  • Multi-strategy entity extraction") 
    logger.info("  • Intelligent chunking with entity awareness")
    logger.info("  • Adaptive query processing")
    logger.info("  • Smart retrieval with priority ranking")
    
    if QUERY_REWRITING_AVAILABLE and not args.disable_query_rewriting:
        logger.info("  • Pseudo Relevance Feedback (PRF)")
        logger.info("  • Query variants generation")
        logger.info("  • Reciprocal Rank Fusion (RRF)")
        logger.info("  • Cross-encoder reranking")
    
    logger.info("\n🎯 Ready to solve the 'Whose CV is this?' problem!")
    
    uvicorn.run(app, host=args.host, port=args.port, workers=1, reload=False)