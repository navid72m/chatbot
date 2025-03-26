import sys
import os
sys.path.append(os.path.dirname(__file__))
# mcp_server.py - Message Communication Protocol Server implementation
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from llm_interface import  list_ollama_models, stream_ollama_response

# Import advanced RAG modules
from knowledge_graph import KnowledgeGraph
from chain_of_thought import ChainOfThoughtReasoner
from advanced_rag import AdvancedRAG
from hybrid_retriever import HybridRetriever

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
    uri=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
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
            {"id": "hybrid", "name": "Hybrid Retrieval", "enabled": True}
        ],
        "server_version": "1.0.0",
        "message": "MCP Document Chat Server"
    }

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

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Handle a query request"""
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
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # Get response from LLM
        response = stream_ollama_response(
            query=request.query,
            context=context,
            model=request.model,
            temperature=request.temperature
        )
        
        # Prepare response
        return {
            "type": "query_response",
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "sources": [chunk.metadata.get("source", "Unknown") for chunk in relevant_chunks],
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
    import uvicorn
    uvicorn.run("mcp_server:app", host="127.0.0.1", port=8000, reload=True)