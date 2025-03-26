import sys
import os
sys.path.append(os.path.dirname(__file__))
# app_integration.py - Integration of Advanced RAG features into the main application
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any, Union
import logging
import os
import shutil
from pydantic import BaseModel

# Import existing modules
from document_processor import process_document, chunk_document
from vector_store import VectorStore
from llm_interface import query_ollama, list_ollama_models

# Import new advanced RAG modules
from knowledge_graph import KnowledgeGraph
from chain_of_thought import ChainOfThoughtReasoner
from advanced_rag import AdvancedRAG
from hybrid_retriever import HybridRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    model: str = "mistral"
    temperature: float = 0.7
    context_window: int = 5
    quantization: str = "4bit"
    use_advanced_rag: bool = True
    use_cot: bool = True
    use_kg: bool = True
    verify_answers: bool = True
    use_multihop: bool = True
    current_document: Optional[str] = None

class RAGConfigRequest(BaseModel):
    use_advanced_rag: bool = True
    use_cot: bool = True
    use_kg: bool = True
    verify_answers: bool = True
    use_multihop: bool = True
    max_hops: int = 2
    max_kg_results: int = 3
    max_vector_results: int = 5
    vector_weight: float = 0.7
    kg_weight: float = 0.3

class EntityGraphRequest(BaseModel):
    entities: List[str]
    max_hops: int = 2

# Create FastAPI application
app = FastAPI(title="Advanced Document Chat Backend")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced Document Chat Backend API",
        "features": [
            "Vector Search", 
            "Knowledge Graph", 
            "Chain-of-Thought Reasoning",
            "Hybrid Retrieval",
            "Multi-hop Reasoning",
            "Answer Verification"
        ]
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document with advanced features"""
    try:
        # Save the file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document (OCR, text extraction)
        document_text = process_document(file_path)
        
        # Chunk the document
        chunks = chunk_document(document_text)
        
        # Add to vector store
        vector_store.add_document(file.filename, chunks)
        
        # Add to knowledge graph
        # Convert chunks to the format expected by knowledge_graph
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            chunk.metadata["source"] = file.filename
            chunk.metadata["chunk_id"] = i
        
        # knowledge_graph.process_documents(chunks)
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks": len(chunks),
            "preview": document_text[:300] + "..." if len(document_text) > 300 else document_text,
            "features": {
                "vector_store": True,
                "knowledge_graph": True
            }
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query_document(request: QueryRequest):
    """Query the document using advanced RAG system"""
    try:
        # Configure the advanced RAG system based on request
        advanced_rag.use_cot = request.use_cot
        advanced_rag.use_kg = request.use_kg
        advanced_rag.verify_answers = request.verify_answers
        advanced_rag.use_multihop = request.use_multihop
        advanced_rag.model = request.model
        advanced_rag.temperature = request.temperature
        
        if not request.use_advanced_rag:
            # Use the advanced RAG system
            logger.info(f"Using advanced RAG for query: {request.query}")
            response = advanced_rag.answer_query(request.query)
            
            return {
                "response": response["answer"],
                "reasoning": response["reasoning"],
                "sources": response["sources"],
                "confidence": response["confidence"],
                "retrieval_time": response["retrieval_time"],
                "verification": response.get("verification", {})
            }
        else:
            # Fall back to the original implementation
            logger.info(f"Using standard RAG for query: {request.query}")
            
            # Get the current document filename from the request
            current_document = request.current_document
            
            if not current_document:
                return {
                    "response": "Please upload a document first before querying.",
                    "error": "No document selected"
                }
            
            # Retrieve relevant document chunks from the vector store, filtered by current document
            relevant_chunks = vector_store.search(
                query=request.query, 
                k=request.context_window,
                filter={"source": current_document}  # Filter by current document
            )
            
            if not relevant_chunks:
                return {
                    "response": "I don't have enough information to answer this question based on the current document.",
                    "document": current_document
                }
            
            logger.info(f"Relevant chunks from document {current_document}: {relevant_chunks}")
            # Build context from relevant chunks with token limit
            context_parts = []
            total_tokens = 0
            max_tokens = 4096  # Maximum context size
            
            for chunk in relevant_chunks:
                text = chunk.page_content.strip()
                
                # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                chunk_tokens = len(text) // 4
                
                # If adding this chunk would exceed the limit, truncate it
                if total_tokens + chunk_tokens > max_tokens:
                    remaining_tokens = max_tokens - total_tokens
                    # Truncate text to fit remaining tokens
                    text = text[:remaining_tokens * 4] + "..."
                    logger.warning(f"Truncated chunk to fit within token limit")
                
                context_parts.append(text)
                total_tokens += chunk_tokens
                
                # If we've reached the token limit, stop adding chunks
                if total_tokens >= max_tokens:
                    break
            
            context = "\n\n".join(context_parts)
            
            # Query the LLM via Ollama with quantization parameter
            response = query_ollama(
                query=request.query, 
                context=context,
                model=request.model,
                temperature=request.temperature,
                quantization=request.quantization
            )
            
            return {
                "response": response,
                "sources": [chunk.metadata.get("source", "Unknown") for chunk in relevant_chunks],
                "document": current_document,
                "advanced_features": False
            }
            
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

@app.post("/config/rag")
async def configure_rag(config: RAGConfigRequest):
    """Configure Advanced RAG system parameters"""
    try:
        # Update AdvancedRAG configuration
        advanced_rag.use_cot = config.use_cot
        advanced_rag.use_kg = config.use_kg
        advanced_rag.verify_answers = config.verify_answers
        advanced_rag.use_multihop = config.use_multihop
        advanced_rag.max_hops = config.max_hops
        advanced_rag.max_kg_results = config.max_kg_results
        advanced_rag.max_vector_results = config.max_vector_results
        
        # Update Hybrid Retriever configuration
        hybrid_retriever.vector_weight = config.vector_weight
        hybrid_retriever.kg_weight = config.kg_weight
        hybrid_retriever.max_vector_results = config.max_vector_results
        hybrid_retriever.max_kg_results = config.max_kg_results
        
        return {
            "success": True,
            "message": "Advanced RAG configuration updated",
            "config": config.dict()
        }
    except Exception as e:
        logger.error(f"Error updating RAG configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating RAG configuration: {str(e)}")

@app.get("/knowledge-graph/entities")
async def get_entities(
    query: Optional[str] = Query(None, description="Optional query text to extract entities from"),
    limit: int = Query(20, description="Maximum number of entities to return")
):
    """Get entities from the knowledge graph, optionally filtered by query"""
    try:
        if query:
            # Extract entities from the query
            entities = knowledge_graph.extract_entities(query)
            return {
                "entities": [e["text"] for e in entities[:limit]],
                "query": query
            }
        else:
            # Return all entities from the knowledge graph (simplified implementation)
            # In a real implementation, you would query the knowledge graph for all entities
            return {
                "message": "To get entities, provide a query parameter"
            }
    except Exception as e:
        logger.error(f"Error retrieving entities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving entities: {str(e)}")

@app.post("/knowledge-graph/entity-graph")
async def get_entity_graph(request: EntityGraphRequest):
    """Get a subgraph centered around specified entities"""
    try:
        graph_data = knowledge_graph.get_entity_graph(
            entity_names=request.entities,
            max_hops=request.max_hops
        )
        
        return graph_data
    except Exception as e:
        logger.error(f"Error retrieving entity graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving entity graph: {str(e)}")

@app.get("/knowledge-graph/entity/{entity_name}/related")
async def get_related_entities(
    entity_name: str,
    max_hops: int = Query(2, description="Maximum number of hops from the entity")
):
    """Get entities related to the specified entity"""
    try:
        related = knowledge_graph.query_related_entities(
            entity_name=entity_name,
            max_hops=max_hops
        )
        
        return {
            "entity": entity_name,
            "related_entities": related
        }
    except Exception as e:
        logger.error(f"Error retrieving related entities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving related entities: {str(e)}")

@app.get("/knowledge-graph/path")
async def find_path(
    entity1: str = Query(..., description="First entity name"),
    entity2: str = Query(..., description="Second entity name"),
    max_hops: int = Query(3, description="Maximum path length")
):
    """Find paths connecting two entities in the knowledge graph"""
    try:
        paths = knowledge_graph.find_path_between_entities(
            start_entity=entity1,
            end_entity=entity2,
            max_hops=max_hops
        )
        
        return {
            "entity1": entity1,
            "entity2": entity2,
            "paths": paths
        }
    except Exception as e:
        logger.error(f"Error finding path between entities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finding path: {str(e)}")

@app.get("/models")
async def get_models():
    """List available Ollama models"""
    try:
        models = list_ollama_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.get("/quantization-options")
async def get_quantization_options():
    """Get available quantization options"""
    return {
        "options": [
            {"value": "None", "label": "None (Full Precision)"},
            {"value": "8bit", "label": "8-bit Quantization"},
            {"value": "4bit", "label": "4-bit Quantization (Recommended)"},
            {"value": "1bit", "label": "1-bit Quantization (Fastest, Lower Quality)"}
        ]
    }

@app.get("/advanced-rag/features")
async def get_advanced_rag_features():
    """Get information about available advanced RAG features"""
    return {
        "features": [
            {
                "id": "cot",
                "name": "Chain of Thought Reasoning",
                "description": "Enables step-by-step reasoning before answering, reducing hallucination",
                "enabled": advanced_rag.use_cot
            },
            {
                "id": "kg",
                "name": "Knowledge Graph",
                "description": "Uses a graph database to capture entity relationships and enhance retrieval",
                "enabled": advanced_rag.use_kg
            },
            {
                "id": "verification",
                "name": "Answer Verification",
                "description": "Verifies answer accuracy against source documents and identifies unsupported claims",
                "enabled": advanced_rag.verify_answers
            },
            {
                "id": "multihop",
                "name": "Multi-hop Reasoning",
                "description": "Breaks complex queries into simpler sub-questions for better reasoning",
                "enabled": advanced_rag.use_multihop
            },
            {
                "id": "hybrid",
                "name": "Hybrid Retrieval",
                "description": "Combines vector search and knowledge graph for better document retrieval",
                "enabled": True
            }
        ]
    }

# Advanced debug endpoints for development
@app.post("/debug/analyze-query")
async def analyze_query(query: str = Form(...)):
    """Debug endpoint to analyze a query with all available tools"""
    try:
        # Extract entities
        entities = knowledge_graph.extract_entities(query)
        
        # Check if query is complex
        is_complex = advanced_rag._is_complex_query(query)
        
        # Get expanded queries
        expanded = []
        if hybrid_retriever.use_query_expansion:
            expanded = hybrid_retriever._expand_query(query)
        
        return {
            "query": query,
            "entities": entities,
            "is_complex": is_complex,
            "expanded_queries": expanded,
            "word_count": len(query.split())
        }
    except Exception as e:
        logger.error(f"Error analyzing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_integration:app", host="127.0.0.1", port=8000, reload=True)