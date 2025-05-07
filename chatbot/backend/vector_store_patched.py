import os
import logging
from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Disable ChromaDB telemetry to avoid the posthog dependency
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    A wrapper class for vector store operations using ChromaDB.
    """
    def __init__(self):
        """Initialize the vector store with embeddings."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            # Use persistent storage
            persist_directory = os.path.expanduser("~/Library/Application Support/Document Chat/chroma_db")
            os.makedirs(persist_directory, exist_ok=True)
            
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name="documents"
            )
            logger.info(f"Vector store initialized with persist_directory: {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Create an in-memory vector store as fallback
            try:
                self.vector_store = Chroma(
                    embedding_function=self.embeddings,
                    collection_name="documents"
                )
                logger.info("Fallback to in-memory vector store")
            except Exception as fallback_e:
                logger.error(f"Fallback vector store initialization failed: {fallback_e}")
                raise RuntimeError(f"Failed to initialize vector store: {e}")

    def add_document(self, source: str, chunks: List[str]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            source: Document source/filename
            chunks: List of text chunks to add
        """
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": source,
                "chunk": i
            }
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)
        
        self.vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} chunks from {source} to vector store")

    def search(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional filter to apply
            
        Returns:
            List of document chunks
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        logger.info(f"Found {len(results)} relevant chunks for query: {query}")
        return results

def build_focused_context(query: str, documents: List[Document]) -> str:
    """
    Build a focused context from retrieved documents.
    
    Args:
        query: The user's query
        documents: List of retrieved documents
        
    Returns:
        Formatted context string
    """
    # Sort documents by relevance (assuming they're returned in relevance order)
    context_parts = []
    
    for i, doc in enumerate(documents):
        # Add the document with a header
        source = doc.metadata.get("source", "Unknown")
        chunk_num = doc.metadata.get("chunk", i)
        
        context_part = f"[Document: {source}, Chunk: {chunk_num}]\n{doc.page_content}\n"
        context_parts.append(context_part)
    
    return "\n".join(context_parts)
