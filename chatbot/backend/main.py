import sys
import os
sys.path.append(os.path.dirname(__file__))
# mcp_server.py - Message Communication Protocol Server implementation
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Optional, Any
import logging
import os
import shutil
import json
import uuid
import asyncio
from pydantic import BaseModel, Field
from datetime import datetime

# Import existing modules
from document_processor import process_document, chunk_document
from vector_store import VectorStore
from llm_interface import list_ollama_models, stream_ollama_response, _get_ollama_response_sync

# Import advanced RAG modules
from knowledge_graph import KnowledgeGraph
from chain_of_thought import ChainOfThoughtReasoner
from advanced_rag import AdvancedRAG
from hybrid_retriever import HybridRetriever

# Import model download endpoints - make sure model_download.py is in the same directory
try:
    from model_download import add_model_download_routes
except ImportError:
    # Fall back to defining a basic implementation if the module doesn't exist
    def add_model_download_routes(app):
        logging.error("model_download module not found. Model download features will be disabled.")
        
        # Create a router for basic model operations
        from fastapi import APIRouter
        router = APIRouter()
        
        @router.get("/models")
        async def list_models():
            """List available Ollama models without download functionality"""
            try:
                models = list_ollama_models()
                return {
                    "models": models,
                    "downloaded_models": models,  # Assume all listed models are downloaded
                    "model_info": {}  # No detailed info available
                }
            except Exception as e:
                logging.error(f"Error listing models: {str(e)}")
                return {
                    "models": ["deepseek-r1"],
                    "downloaded_models": ["deepseek-r1"],
                    "model_info": {}
                }
        
        app.include_router(router, tags=["models"])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

# Initialize global components
vector_store = VectorStore()

# Initialize knowledge graph with environment variables or defaults
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

knowledge_graph = KnowledgeGraph(
    # uri=NEO4J_URI,
    # username=NEO4J_USERNAME,
    # password=NEO4J_PASSWORD
)

# Initialize reasoner
reasoner = ChainOfThoughtReasoner()

# Initialize hybrid retriever
hybrid_retriever = HybridRetriever(
    vector_store=vector_store,
    knowledge_graph=knowledge_graph
)

# Initialize advanced RAG system
advanced_rag = AdvancedRAG(
    vector_store=vector_store,
    knowledge_graph=knowledge_graph,
    reasoner=reasoner
)

# Storage directory for documents
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="MCP Document Chat Server")

# Add CORS middleware
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    document: str
    use_advanced_rag: bool = True
    use_cot: bool = True
    use_kg: bool = True
    verify_answers: bool = True
    use_multihop: bool = True
    model: str = "mistral"
    temperature: float = 0.7
    context_window: int = 10
    quantization: str = "4bit"

class ConfigRequest(BaseModel):
    config: Dict[str, Any]

class SetDocumentRequest(BaseModel):
    document: str

# Add model download routes to our app
add_model_download_routes(app)

@app.get("/")
async def root():
    """Root endpoint returning server info"""
    return {
        "type": "server_info",
        "message_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "features": [
            {"id": "cot", "name": "Chain of Thought Reasoning", "enabled": advanced_rag.use_cot},
            {"id": "kg", "name": "Knowledge Graph", "enabled": advanced_rag.use_kg},
            {"id": "verification", "name": "Answer Verification", "enabled": advanced_rag.verify_answers},
            {"id": "multihop", "name": "Multi-hop Reasoning", "enabled": advanced_rag.use_multihop},
            {"id": "hybrid", "name": "Hybrid Retrieval", "enabled": True},
            {"id": "streaming", "name": "Streaming Responses", "enabled": True},
            {"id": "model_management", "name": "Model Management", "enabled": True}
        ],
        "server_version": "1.1.0",
        "message": "MCP Document Chat Server"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save the uploaded file
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document text
        document_text = process_document(file_path)
        
        # Chunk the document
        chunks = chunk_document(document_text)
        
        # Add to vector store
        vector_store.add_document(file.filename, chunks)
        
        return {
            "filename": file.filename,
            "chunks": len(chunks),
            "preview": document_text[:300] + "..." if len(document_text) > 300 else document_text,
            "features": {
                "vector_store": True,
                "knowledge_graph": advanced_rag.use_kg
            }
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_stream_response(query, context, model, temperature):
    """Generate streaming response chunks from LLM"""
    # Initialize response data
    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Send initial stream response header
    header = {
        "type": "stream_start",
        "message_id": message_id,
        "timestamp": timestamp
    }
    yield f"data: {json.dumps(header)}\n\n"
    
    # Stream the response content
    accumulated_text = ""
    async for token in stream_ollama_response(
        query=query,
        context=context,
        model=model,
        temperature=temperature,
        stream=True  # Enable streaming in ollama interface
    ):
        accumulated_text += token
        chunk = {
            "type": "stream_token",
            "message_id": message_id,
            "token": token,
            "accumulated_text": accumulated_text,
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)  # Small delay to control streaming pace
    
    # Send final completion message
    footer = {
        "type": "stream_end",
        "message_id": message_id,
        "timestamp": datetime.now().isoformat(),
        "complete_text": accumulated_text
    }
    yield f"data: {json.dumps(footer)}\n\n"

@app.get("/query")
async def handle_query_streaming(
    query: str, 
    document: str, 
    use_advanced_rag: bool = True,
    use_cot: bool = True,
    use_kg: bool = True,
    verify_answers: bool = True,
    use_multihop: bool = True,
    model: str = "mistral",
    temperature: float = 0.7,
    context_window: int = 10,
    quantization: str = "4bit"
):
    """Handle a streaming query request"""
    try:
        logger.info(f"Streaming query request: {query}")
        # Get relevant chunks from vector store
        relevant_chunks = vector_store.search(
            query=query,
            k=int(context_window),
            filter={"source": document}
        )
        
        if not relevant_chunks:
            # For empty results, return a simple error message
            async def error_generator():
                error_msg = {
                    "type": "stream_start",
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_msg)}\n\n"
                
                error_content = "No relevant information found in the document."
                yield f"data: {json.dumps({'type': 'stream_token', 'token': error_content})}\n\n"
                
                yield f"data: {json.dumps({'type': 'stream_end', 'complete_text': error_content})}\n\n"
                
            return StreamingResponse(
                error_generator(),
                media_type="text/event-stream"
            )
        
        # Build context from chunks
        def extract_page_content(chunk):
            # Unpack if it's a (Document, score) tuple
            if isinstance(chunk, tuple):
                chunk = chunk[0]
            # Extract from dict
            if isinstance(chunk, dict):
                return chunk.get("page_content", "")
            # Extract from Document object
            return getattr(chunk, "page_content", "")

        context = "\n\n".join([
            content for chunk in relevant_chunks
            if (content := extract_page_content(chunk))
        ])
        
        logger.info(f"Starting streaming response for query: {query[:50]}...")
        
        # Return streaming response
        return StreamingResponse(
            generate_stream_response(
                query=query,
                context=context,
                model=model,
                temperature=float(temperature)
            ),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error processing streaming query: {str(e)}")
        # Stream an error response in SSE format
        async def error_stream():
            error_msg = {
                "type": "stream_start",
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_msg)}\n\n"
            
            error_content = f"Sorry, I encountered an error: {str(e)}"
            yield f"data: {json.dumps({'type': 'stream_token', 'token': error_content})}\n\n"
            
            yield f"data: {json.dumps({'type': 'stream_end', 'complete_text': error_content})}\n\n"
            
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )

@app.post("/query")
async def handle_query_post(request: QueryRequest):
    """Handle a query request via POST (for compatibility)"""
    try:
        # Get relevant chunks from vector store
        relevant_chunks = vector_store.search(
            query=request.query,
            k=request.context_window,
            filter={"source": request.document}
        )
        
        if not relevant_chunks:
            return {
                "type": "error",
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "error": "No relevant information found in the document.",
                "error_code": "NO_RELEVANT_INFO"
            }
        
        # Build context from chunks
        def extract_page_content(chunk):
            # Unpack if it's a (Document, score) tuple
            if isinstance(chunk, tuple):
                chunk = chunk[0]
            # Extract from dict
            if isinstance(chunk, dict):
                return chunk.get("page_content", "")
            # Extract from Document object
            return getattr(chunk, "page_content", "")

        context = "\n\n".join([
            content for chunk in relevant_chunks
            if (content := extract_page_content(chunk))
        ])
        
        # Return streaming response
        return StreamingResponse(
            generate_stream_response(
                query=request.query,
                context=context,
                model=request.model,
                temperature=request.temperature
            ),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-sync")
async def handle_query_sync(request: QueryRequest):
    """Handle a query request with non-streaming response (fallback)"""
    try:
        logger.info(f"Non-streaming query request: {request.query}")
        # Get relevant chunks from vector store
        relevant_chunks = vector_store.search(
            query=request.query,
            k=request.context_window,
            filter={"source": request.document}
        )
        
        if not relevant_chunks:
            return {
                "type": "error",
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "error": "No relevant information found in the document.",
                "error_code": "NO_RELEVANT_INFO"
            }
        
        # Build context from chunks
        def extract_page_content(chunk):
            # Unpack if it's a (Document, score) tuple
            if isinstance(chunk, tuple):
                chunk = chunk[0]
            # Extract from dict
            if isinstance(chunk, dict):
                return chunk.get("page_content", "")
            # Extract from Document object
            return getattr(chunk, "page_content", "")

        context = "\n\n".join([
            content for chunk in relevant_chunks
            if (content := extract_page_content(chunk))
        ])
        
        # Get response from LLM
        response = _get_ollama_response_sync(
            query=request.query,
            context=context,
            model=request.model,
            temperature=request.temperature
        )
        
        # Get source metadata
        sources = []
        for chunk in relevant_chunks:
            if isinstance(chunk, tuple) and hasattr(chunk[0], 'metadata'):
                sources.append(chunk[0].metadata.get("source", "Unknown"))
            elif hasattr(chunk, 'metadata'):
                sources.append(chunk.metadata.get("source", "Unknown"))
            else:
                sources.append("Unknown")
        
        # Prepare response
        return {
            "type": "query_response",
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "sources": sources,
            "document": request.document
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure")
async def handle_configuration(request: ConfigRequest):
    """Handle configuration updates"""
    try:
        # Update advanced RAG configuration
        for key, value in request.config.items():
            if hasattr(advanced_rag, key):
                setattr(advanced_rag, key, value)
        
        return {
            "type": "configure_response",
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "config": request.config
        }
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_document")
async def handle_set_document(request: SetDocumentRequest):
    """Handle setting the current document"""
    try:
        return {
            "type": "set_document_response",
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "document": request.document
        }
    except Exception as e:
        logger.error(f"Error setting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        models = list_ollama_models()
        return {
            "type": "list_models_response",
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "models": models
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import sys
    import uvicorn
    
    # Use the app object directly, not a string reference
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    else:
        # Development mode
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)




# ...rest of the file remains the same...