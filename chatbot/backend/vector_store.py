# vector_store.py - Vector storage with ChromaDB

# from asyncio.log import logger
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.docstore.document import Document
from typing import List, Dict, Optional
import os

import logging
logger = logging.getLogger(__name__)

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
    
    def search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar chunks to the query
        
        Args:
            query: The search query
            k: Number of results to return
            filter: Optional filter dictionary (e.g., {"source": "document_name"})
            
        Returns:
            List of similar Document objects
        """
        try:
            logger.info(f"Starting vector search with query: '{query}', k={k}, filter={filter}")
            
            if filter and "source" in filter:
                # Use where filter in ChromaDB for more efficient filtering
                where_filter = {"source": filter["source"]}
                logger.info(f"Applying filter: {where_filter}")
                
                results = self.db.similarity_search_with_score(
                    query, 
                    k=k,
                    filter=where_filter
                )
                
                # Log detailed information about each result
                logger.info(f"Found {len(results)} results with scores:")
                for doc, score in results:
                    logger.info(f"Score: {score:.4f}")
                    logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    logger.info(f"Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}")
                    logger.info(f"Content preview: {doc.page_content[:100]}...")
                    logger.info("-" * 50)
                
                filtered_docs = [doc for doc, _ in results]
                logger.info(f"Returning {len(filtered_docs)} filtered documents")
                return filtered_docs
            else:
                # If no filter, use regular similarity search
                logger.info("No filter provided, performing regular similarity search")
                results = self.db.similarity_search(query, k=k)
                
                # Log information about unfiltered results
                logger.info(f"Found {len(results)} results:")
                for doc in results:
                    logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    logger.info(f"Chunk ID: {doc.metadata.get('chunk_id', 'Unknown')}")
                    logger.info(f"Content preview: {doc.page_content[:100]}...")
                    logger.info("-" * 50)
                
                logger.info(f"Returning {len(results)} documents")
                return results
                
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}", exc_info=True)
            logger.error(f"Query parameters - query: '{query}', k: {k}, filter: {filter}")
            return []

