# app.py - Main FastAPI application with quantization support
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from typing import List, Optional
from pydantic import BaseModel

from document_processor import process_document, chunk_document
from vector_store import VectorStore
from llm_interface import query_ollama, list_ollama_models

app = FastAPI(title="Document Chat Backend")

# Set up CORS to allow requests from Electron
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store
vector_store = VectorStore()

# Storage directory for documents
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    model: str = "mistral"  # Default model
    temperature: float = 0.7
    context_window: int = 5  # Number of document chunks to include
    quantization: str = "4bit"  # Default to 4-bit quantization

@app.get("/")
async def root():
    return {"message": "Document Chat Backend API"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
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
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks": len(chunks),
            "preview": document_text[:300] + "..." if len(document_text) > 300 else document_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query_document(request: QueryRequest):
    """Query the document using the LLM with quantization support"""
    try:
        # Retrieve relevant document chunks from the vector store
        relevant_chunks = vector_store.search(request.query, k=request.context_window)
        
        if not relevant_chunks:
            return {
                "response": "I don't have enough information to answer this question based on the documents you've provided."
            }
        
        # Build context from relevant chunks
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
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
            "sources": [chunk.metadata.get("source", "Unknown") for chunk in relevant_chunks]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

@app.get("/models")
async def get_models():
    """List available Ollama models"""
    try:
        models = list_ollama_models()
        return {"models": models}
    except Exception as e:
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

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)