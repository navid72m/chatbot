import os
import logging
from typing import List, Optional, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Storage directories
INDEX_STORAGE_DIR = os.path.expanduser("~/Library/Application Support/Document Chat/indices")
os.makedirs(INDEX_STORAGE_DIR, exist_ok=True)

class SimpleVectorStore:
    """A simplified vector store implementation to avoid LlamaIndex dependencies."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict] = None):
        """Add documents and their embeddings to the store."""
        if metadata is None:
            metadata = [{}] * len(texts)
        
        self.documents.extend(texts)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
        
    def search(self, query_embedding: List[float], k: int = 5, filter_dict: Dict = None):
        """Search for similar documents."""
        import numpy as np
        
        if not self.embeddings:
            return []
        
        # Convert query embedding to numpy array
        query_array = np.array(query_embedding)
        
        # Calculate cosine similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            # Apply filter if provided
            if filter_dict and not self._matches_filter(self.metadata[i], filter_dict):
                continue
                
            # Calculate cosine similarity
            emb_array = np.array(emb)
            similarity = np.dot(query_array, emb_array) / (np.linalg.norm(query_array) * np.linalg.norm(emb_array))
            similarities.append((i, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for idx, score in similarities[:k]:
            results.append((self.documents[idx], score, self.metadata[idx]))
            
        return results
    
    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def save(self, path: str):
        """Save the vector store to disk."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
            
    def load(self, path: str):
        """Load vector store from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]


class LlamaIndexRAG:
    """A simplified implementation of RAG using SentenceTransformers without LlamaIndex."""
    
    def __init__(self, llm_model_path: str = None):
        """Initialize the RAG system."""
        # Set up embedding model
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        
        # Store LLM path for later use
        self.llm_path = llm_model_path
        
        # Track processed documents
        self.processed_documents = {}
        
    def _get_index_path(self, document_name: str) -> str:
        """Get storage path for a document index."""
        # Create a safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in document_name)
        return os.path.join(INDEX_STORAGE_DIR, f"{safe_name}_index.json")
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text content from a file."""
        # Check file type
        if file_path.lower().endswith('.pdf'):
            return self._extract_text_from_pdf(file_path)
        else:
            return self._extract_text_from_txt(file_path)
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n\n"
            return text
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n\n"
                return text
            except ImportError:
                raise ImportError("PDF extraction libraries not found. Install PyPDF2 or pdfplumber.")
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from text file."""
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
                
        raise ValueError(f"Could not decode file {file_path} with any encoding")
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into chunks with overlap."""
        # Simple paragraph-based chunking
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                # Add to current chunk
                current_chunk += para + "\n\n"
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk, including overlap from previous
                if overlap > 0 and current_chunk:
                    # Add last ~overlap characters from previous chunk
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-20:]) if len(words) > 20 else current_chunk
                    current_chunk = overlap_text + "\n\n" + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def process_document(self, file_path: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Process a document and create an index."""
        try:
            # Get document name from path
            document_name = os.path.basename(file_path)
            index_path = self._get_index_path(document_name)
            
            # Extract text from file
            document_text = self._extract_text_from_file(file_path)
            
            # Split into chunks
            chunks = self._chunk_text(document_text, chunk_size, overlap)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Create vector store
            vector_store = SimpleVectorStore()
            
            # Create embeddings for chunks
            embeddings = []
            metadata = []
            
            for i, chunk in enumerate(chunks):
                # Create embedding
                embedding = self.embed_model.encode(chunk)
                embeddings.append(embedding.tolist())
                
                # Create metadata
                chunk_metadata = {
                    "source": document_name,
                    "chunk_id": i
                }
                metadata.append(chunk_metadata)
            
            # Add to vector store
            vector_store.add_documents(chunks, embeddings, metadata)
            
            # Save vector store
            vector_store.save(index_path)
            
            # Record this document as processed
            self.processed_documents[document_name] = {
                "chunks": len(chunks),
                "path": file_path
            }
            
            logger.info(f"Created index for {document_name} with {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    def load_index(self, document_name: str) -> bool:
        """Check if an index exists for a document."""
        index_path = self._get_index_path(document_name)
        return os.path.exists(index_path)
    
    def query(self, query_text: str, document_name: str, top_k: int = 5) -> Dict[str, Any]:
        """Query a document."""
        try:
            # Get index path
            index_path = self._get_index_path(document_name)
            
            # Check if index exists
            if not os.path.exists(index_path):
                logger.warning(f"No index found for {document_name}")
                return {
                    "response": f"No index found for document: {document_name}",
                    "chunks_retrieved": [],
                    "sources": []
                }
            
            # Load vector store
            vector_store = SimpleVectorStore()
            vector_store.load(index_path)
            
            # Create query embedding
            query_embedding = self.embed_model.encode(query_text).tolist()
            
            # Search for similar chunks
            results = vector_store.search(
                query_embedding=query_embedding,
                k=top_k,
                filter_dict={"source": document_name}
            )
            
            # Extract chunks and scores
            chunks_retrieved = [(chunk, score) for chunk, score, _ in results]
            sources = [metadata.get("source", "Unknown") for _, _, metadata in results]
            
            # If we have a LLM, we could generate an answer here
            # For now, just return the retrieved chunks
            response = ""
            if self.llm_path and os.path.exists(self.llm_path):
                try:
                    from llama_cpp_interface import stream_llama_cpp_response
                    context = "\n\n".join([chunk for chunk, _ in chunks_retrieved])
                    llm_response = stream_llama_cpp_response(
                        query=query_text,
                        context=context,
                        model="mamba",
                        temperature=0.3
                    )
                    response = llm_response["response"] if isinstance(llm_response, dict) else llm_response
                except Exception as e:
                    logger.error(f"Error generating response with LLM: {str(e)}")
                    response = "Error generating response with LLM."
            
            return {
                "response": response,
                "chunks_retrieved": chunks_retrieved,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            return {
                "response": f"Error processing your query: {str(e)}",
                "chunks_retrieved": [],
                "sources": []
            }
    
    def get_document_list(self) -> List[str]:
        """Get list of documents that have indices."""
        try:
            # Check all JSON files in the index directory
            document_names = []
            
            if os.path.exists(INDEX_STORAGE_DIR):
                for filename in os.listdir(INDEX_STORAGE_DIR):
                    if filename.endswith("_index.json"):
                        doc_name = filename.replace("_index.json", "")
                        document_names.append(doc_name)
            
            return document_names
        except Exception as e:
            logger.error(f"Error listing document indices: {str(e)}")
            return []