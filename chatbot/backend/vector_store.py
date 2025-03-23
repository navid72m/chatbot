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