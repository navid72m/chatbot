"""
A simplified vector store that uses basic numpy operations instead of ML libraries.
This is meant to be a temporary solution for packaging purposes.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

class Document:
    """Simple document class similar to LangChain's Document."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleVectorStore:
    """A simplified vector store that doesn't rely on external ML libraries."""
    
    def __init__(self, persist_directory: str = "simple_db"):
        """Initialize the simple vector store.
        
        Args:
            persist_directory: Directory to persist the database.
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.documents_file = os.path.join(persist_directory, "documents.json")
        self.documents = []
        
        # Load existing documents if they exist
        if os.path.exists(self.documents_file):
            try:
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = [
                        Document(item["page_content"], item["metadata"])
                        for item in data
                    ]
            except Exception as e:
                print(f"Error loading documents: {e}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add.
        """
        self.documents.extend(documents)
        self._save_documents()
    
    def _save_documents(self) -> None:
        """Save documents to disk."""
        data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.documents
        ]
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for documents similar to the query.
        Simple implementation using string matching.
        
        Args:
            query: Query string.
            k: Number of documents to return.
            
        Returns:
            List of Document objects.
        """
        # Simple string matching for now
        scored_docs = []
        
        for doc in self.documents:
            # Calculate a simple score based on word overlap
            query_words = set(query.lower().split())
            doc_words = set(doc.page_content.lower().split())
            common_words = query_words.intersection(doc_words)
            
            if common_words:
                score = len(common_words) / max(len(query_words), len(doc_words))
                scored_docs.append((doc, score))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, _ in scored_docs[:k]]
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for documents similar to the query and return scores.
        
        Args:
            query: Query string.
            k: Number of documents to return.
            
        Returns:
            List of tuples of (Document, score).
        """
        # Simple string matching for now
        scored_docs = []
        
        for doc in self.documents:
            # Calculate a simple score based on word overlap
            query_words = set(query.lower().split())
            doc_words = set(doc.page_content.lower().split())
            common_words = query_words.intersection(doc_words)
            
            if common_words:
                score = len(common_words) / max(len(query_words), len(doc_words))
                scored_docs.append((doc, score))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents with scores
        return scored_docs[:k]
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        self.documents = []
        if os.path.exists(self.documents_file):
            os.remove(self.documents_file)

# For backwards compatibility with existing code
VectorStore = SimpleVectorStore