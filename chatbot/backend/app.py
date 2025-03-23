# app.py - Main FastAPI application
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
    """Query the document using the LLM"""
    try:
        # Retrieve relevant document chunks from the vector store
        relevant_chunks = vector_store.search(request.query, k=request.context_window)
        
        if not relevant_chunks:
            return {
                "response": "I don't have enough information to answer this question based on the documents you've provided."
            }
        
        # Build context from relevant chunks
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # Query the LLM via Ollama
        response = query_ollama(
            query=request.query, 
            context=context,
            model=request.model,
            temperature=request.temperature
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

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)


# document_processor.py - Document processing and chunking
import os
from typing import List, Dict
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_document(file_path: str) -> str:
    """Extract text from a document file"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return process_image(file_path)
    elif file_extension == '.pdf':
        return process_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def process_image(image_path: str) -> str:
    """Extract text from an image using OCR"""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def process_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file"""
    try:
        # First try PyPDF2
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # If we got meaningful text, return it
        if text.strip() and len(text) > 100:
            return text
        
        # If not enough text extracted, try OCR
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image) + "\n\n"
        return text
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        # Fall back to OCR only
        try:
            images = convert_from_path(pdf_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n\n"
            return text
        except Exception as inner_e:
            raise Exception(f"Failed to process PDF: {str(inner_e)}")

def chunk_document(text: str) -> List[Dict]:
    """Split document into chunks for vector storage"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.create_documents([text])
    return chunks


# vector_store.py - Vector storage with ChromaDB
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict, Optional
import os

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize the vector store with HuggingFace embeddings"""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings using sentence-transformers
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    def add_document(self, document_name: str, chunks: List[Document]) -> None:
        """Add a document's chunks to the vector store"""
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata'):
                chunk.metadata = {}
            chunk.metadata["source"] = document_name
            chunk.metadata["chunk_id"] = i
        
        # Add to ChromaDB
        self.db.add_documents(chunks)
        self.db.persist()
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar chunks to the query"""
        results = self.db.similarity_search(query, k=k)
        return results


# llm_interface.py - Interface for Ollama LLM
import requests
import json
from typing import List, Dict, Optional

OLLAMA_API = "http://localhost:11434/api"

def query_ollama(query: str, context: str, model: str = "mistral", temperature: float = 0.7) -> str:
    """Query Ollama with the given prompt"""
    prompt = f"""
Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question: {query}
""".strip()

    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        response = requests.post(f"{OLLAMA_API}/generate", json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error querying Ollama: {str(e)}")

def list_ollama_models() -> List[str]:
    """List available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_API}/tags")
        response.raise_for_status()
        
        result = response.json()
        return [model["name"] for model in result["models"]]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error listing Ollama models: {str(e)}")