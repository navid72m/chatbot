import os
import logging
import time
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
        """Extract text content from a file with optimized performance."""
        # Create a cache directory for extracted text
        cache_dir = os.path.join(INDEX_STORAGE_DIR, "text_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check file type and size
        file_size = os.path.getsize(file_path)
        file_hash = self._get_document_hash(file_path)
        cache_path = os.path.join(cache_dir, f"{file_hash}.txt")
        
        # Check if we have a cached version of the extracted text
        if os.path.exists(cache_path):
            logger.info(f"Using cached text extraction for {os.path.basename(file_path)}")
            with open(cache_path, 'r', encoding='utf-8') as cache_file:
                return cache_file.read()
        
        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            # Optimize PDF processing - use a more efficient library
            try:
                # Try PyMuPDF first (faster than PyPDF2)
                import fitz  # PyMuPDF
                text = ""
                with fitz.open(file_path) as doc:
                    # Process in chunks of 10 pages for memory efficiency
                    total_pages = len(doc)
                    for i in range(0, total_pages, 10):
                        page_text = ""
                        for page_num in range(i, min(i+10, total_pages)):
                            page = doc.load_page(page_num)
                            page_text += page.get_text() + "\n\n"
                        text += page_text
                        
                        # Early check for large documents
                        if len(text) > 1000000:  # ~1MB of text
                            logger.warning(f"PDF is very large, truncating after page {i+10}")
                            text += f"\n[Note: PDF truncated after {i+10} pages due to size]"
                            break
                
            except ImportError:
                # Fall back to original implementation without using super()
                try:
                    import PyPDF2
                    text = ""
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        for page_num in range(len(reader.pages)):
                            text += reader.pages[page_num].extract_text() + "\n\n"
                            
                            # Check if we're exceeding reasonable limits
                            if len(text) > 1000000:  # ~1MB of text
                                logger.warning(f"PDF too large, truncating after page {page_num+1}")
                                text += f"\n[Note: PDF truncated after {page_num+1} pages]"
                                break
                except ImportError:
                    try:
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            text = ""
                            for i, page in enumerate(pdf.pages):
                                text += page.extract_text() + "\n\n"
                                
                                # Check if we're exceeding reasonable limits
                                if len(text) > 1000000:  # ~1MB of text
                                    logger.warning(f"PDF too large, truncating after page {i+1}")
                                    text += f"\n[Note: PDF truncated after {i+1} pages]"
                                    break
                    except ImportError:
                        raise ImportError("PDF extraction libraries not found.")
        else:
            # For text files, use a more efficient approach
            try:
                # For small text files (< 10MB), read all at once
                if file_size < 10 * 1024 * 1024:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    # For large text files, read in chunks
                    text = ""
                    chunk_size = 1024 * 1024  # 1MB chunks
                    with open(file_path, 'r', encoding='utf-8') as f:
                        while chunk := f.read(chunk_size):
                            text += chunk
                            # Break if we have enough text already
                            if len(text) > 2 * 1024 * 1024:  # 2MB of text
                                text += "\n[Note: File truncated due to size]"
                                break
            except UnicodeDecodeError:
                # Try with other encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                            continue
                else:
                    raise ValueError(f"Could not decode file {file_path} with any encoding")
        
        # Cache the extracted text for future use
        with open(cache_path, 'w', encoding='utf-8') as cache_file:
            cache_file.write(text)
        
        return text
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
        """Process a document and create an index with optimized performance."""
        try:
            # Get document name from path
            document_name = os.path.basename(file_path)
            index_path = self._get_index_path(document_name)
            
            # Check for existing index to avoid reprocessing
            if os.path.exists(index_path):
                logger.info(f"Found existing index for {document_name}, loading instead of reprocessing")
                vector_store = SimpleVectorStore()
                vector_store.load(index_path)
                # Return the document chunks without reprocessing
                return vector_store.documents
            
            # Extract text based on file type
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                logger.info(f"Processing image file: {document_name}")
                from document_processor_patched import extract_text_from_image
                document_text = extract_text_from_image(file_path)
            else:
                # Use the _extract_text_from_file method
                document_text = self._extract_text_from_file(file_path)
            
            # Split into chunks using the optimized method
            chunks = self._optimized_chunk_text(document_text, chunk_size, overlap)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Create vector store
            vector_store = SimpleVectorStore()
            
            # Create embeddings in batches for better performance
            batch_size = 16  # Adjust based on available RAM
            embeddings = []
            metadata = []
            
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:min(i+batch_size, len(chunks))]
                
                # Log batch progress
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
                # Encode batch of chunks at once (much faster than one by one)
                batch_embeddings = self.embed_model.encode(batch, show_progress_bar=False)
                
                # Add embeddings and metadata
                for j, embedding in enumerate(batch_embeddings):
                    chunk_index = i + j
                    embeddings.append(embedding.tolist())
                    metadata.append({
                        "source": document_name,
                        "chunk_id": chunk_index
                    })
            
            # Add to vector store (all at once)
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
        
    def _get_document_hash(self, file_path: str) -> str:
        """Generate a content hash for document caching."""
        import hashlib
        
        hasher = hashlib.md5()
        
        # For large files, hash only the first 10MB
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
                if f.tell() > 10 * 1024 * 1024:  # 10MB
                    break
        
        return hasher.hexdigest()

    def _optimized_chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Optimized version of text chunking for better performance."""
        import re
        
        # Pre-process text to remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Quick return for very short texts
        if len(text) <= chunk_size:
            return [text]
        
        # Start with paragraph splitting (faster than sentence splitting)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        # If text is short enough, just return paragraphs
        if sum(len(p) for p in paragraphs) <= chunk_size * 1.5:
            return paragraphs
        
        # For larger documents, use a more efficient chunking approach
        chunks = []
        current_chunk = ""
        
        # Process paragraphs
        for para in paragraphs:
            # If paragraph fits in current chunk, add it
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                # If current chunk plus paragraph is too large
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    # Keep overlap text for next chunk if needed
                    if overlap > 0:
                        # More efficient way to keep overlap text
                        words = current_chunk.split()
                        overlap_word_count = min(20, len(words))  # Cap at 20 words
                        current_chunk = " ".join(words[-overlap_word_count:])
                    else:
                        current_chunk = ""
                
                # If paragraph itself is larger than chunk size, split it
                if len(para) > chunk_size:
                    # Split large paragraphs into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= chunk_size:
                            current_chunk += " " + sentence if current_chunk else sentence
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                                # Keep overlap
                                if overlap > 0:
                                    words = current_chunk.split()
                                    overlap_word_count = min(20, len(words))
                                    current_chunk = " ".join(words[-overlap_word_count:])
                                else:
                                    current_chunk = ""
                            
                            # If sentence itself is too large, force split it
                            if len(sentence) > chunk_size:
                                # Split by whitespace first
                                words = sentence.split()
                                temp_chunk = ""
                                
                                for word in words:
                                    if len(temp_chunk) + len(word) + 1 <= chunk_size:
                                        temp_chunk += " " + word if temp_chunk else word
                                    else:
                                        chunks.append(temp_chunk.strip())
                                        temp_chunk = word
                                
                                if temp_chunk:
                                    current_chunk = temp_chunk
                            else:
                                current_chunk = sentence
                else:
                    # Start a new chunk with this paragraph
                    current_chunk = para
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def load_index(self, document_name: str) -> bool:
        """Check if an index exists for a document."""
        index_path = self._get_index_path(document_name)
        return os.path.exists(index_path)
    def _get_document_hash(self, file_path: str) -> str:
        """Generate a content hash for document caching."""
        import hashlib
        
        hasher = hashlib.md5()
        
        # For large files, hash only the first 10MB
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
                if f.tell() > 10 * 1024 * 1024:  # 10MB
                    break
        
        return hasher.hexdigest()

    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text content from a file with optimized performance."""
        # Create a cache directory for extracted text
        cache_dir = os.path.join(INDEX_STORAGE_DIR, "text_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check file type and size
        file_size = os.path.getsize(file_path)
        file_hash = self._get_document_hash(file_path)
        cache_path = os.path.join(cache_dir, f"{file_hash}.txt")
        
        # Check if we have a cached version of the extracted text
        if os.path.exists(cache_path):
            logger.info(f"Using cached text extraction for {os.path.basename(file_path)}")
            with open(cache_path, 'r', encoding='utf-8') as cache_file:
                return cache_file.read()
        
        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            # Optimize PDF processing - use a more efficient library
            try:
                # Try PyMuPDF first (faster than PyPDF2)
                import fitz  # PyMuPDF
                text = ""
                with fitz.open(file_path) as doc:
                    # Process in chunks of 10 pages for memory efficiency
                    total_pages = len(doc)
                    for i in range(0, total_pages, 10):
                        page_text = ""
                        for page_num in range(i, min(i+10, total_pages)):
                            page = doc.load_page(page_num)
                            page_text += page.get_text() + "\n\n"
                        text += page_text
                        
                        # Early check for large documents
                        if len(text) > 1000000:  # ~1MB of text
                            logger.warning(f"PDF is very large, truncating after page {i+10}")
                            text += f"\n[Note: PDF truncated after {i+10} pages due to size]"
                            break
                
            except ImportError:
                # Fall back to original methods
                logger.info(f"Falling back to original methods for {file_path}")
                text = self._extract_text_from_pdf(file_path)
        else:
            # For text files, use a more efficient approach
            try:
                # For small text files (< 10MB), read all at once
                if file_size < 10 * 1024 * 1024:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    # For large text files, read in chunks
                    text = ""
                    chunk_size = 1024 * 1024  # 1MB chunks
                    with open(file_path, 'r', encoding='utf-8') as f:
                        while chunk := f.read(chunk_size):
                            text += chunk
                            # Break if we have enough text already
                            if len(text) > 2 * 1024 * 1024:  # 2MB of text
                                text += "\n[Note: File truncated due to size]"
                                break
            except UnicodeDecodeError:
                # Try with other encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(f"Could not decode file {file_path} with any encoding")
        
        # Cache the extracted text for future use
        with open(cache_path, 'w', encoding='utf-8') as cache_file:
            cache_file.write(text)
        
        return text
    
    def query(self, query_text: str, document_name: str, top_k: int = 5, 
          temperature: float = 0.3, rerank: bool = True) -> Dict[str, Any]:
        """
        Query a document with optimized performance.
        
        Args:
            query_text: The query text
            document_name: The name of the document to query
            top_k: Number of chunks to retrieve
            temperature: Temperature for LLM response generation
            rerank: Whether to apply reranking to improve relevance
            
        Returns:
            Dictionary with response, chunks retrieved and sources
        """
        query_start_time = time.time()
        try:
            # Initialize cache if not exists
            if not hasattr(self, '_vector_store_cache'):
                self._vector_store_cache = {}
                
            if not hasattr(self, '_query_cache'):
                self._query_cache = {}
                
            # Check cache for this exact query
            cache_key = f"{document_name}:{query_text}:{top_k}"
            if cache_key in self._query_cache:
                logger.info(f"Using cached query result for {document_name}")
                return self._query_cache[cache_key]
                
            # Get index path
            index_path = self._get_index_path(document_name)
            
            # Check if index exists
            if not os.path.exists(index_path):
                logger.warning(f"No index found for {document_name}")
                return {
                    "response": f"No index found for document: {document_name}",
                    "chunks_retrieved": [],
                    "sources": [],
                    "query_time_ms": 0
                }
            
            # Load vector store from cache or disk
            load_start = time.time()
            if document_name in self._vector_store_cache:
                vector_store = self._vector_store_cache[document_name]
                logger.info(f"Using cached vector store for {document_name}")
            else:
                vector_store = SimpleVectorStore()
                vector_store.load(index_path)
                self._vector_store_cache[document_name] = vector_store
                logger.info(f"Loaded vector store for {document_name} in {(time.time() - load_start)*1000:.2f}ms")
            
            # Create query embedding
            embed_start = time.time()
            query_embedding = self.embed_model.encode(query_text).tolist()
            logger.info(f"Created query embedding in {(time.time() - embed_start)*1000:.2f}ms")
            
            # Search for similar chunks
            search_start = time.time()
            results = vector_store.search(
                query_embedding=query_embedding,
                k=top_k * 2 if rerank else top_k,  # Get more results if we'll rerank
                filter_dict={"source": document_name}
            )
            logger.info(f"Vector search completed in {(time.time() - search_start)*1000:.2f}ms")
            
            # Rerank results if requested (and if we have enough results)
            if rerank and len(results) > top_k:
                rerank_start = time.time()
                try:
                    # Import reranker if not already cached
                    if not hasattr(self, '_reranker'):
                        from sentence_transformers import CrossEncoder
                        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    
                    # Prepare pairs for reranking
                    pairs = []
                    for chunk, _, _ in results:
                        pairs.append((query_text, chunk))
                    
                    # Get reranker scores
                    rerank_scores = self._reranker.predict(pairs)
                    
                    # Combine with original results and sort
                    reranked_results = sorted(zip(results, rerank_scores), 
                                            key=lambda x: x[1], reverse=True)
                    
                    # Take top_k results after reranking
                    results = [item[0] for item in reranked_results[:top_k]]
                    logger.info(f"Reranked results in {(time.time() - rerank_start)*1000:.2f}ms")
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}, using original scores")
                    # Fallback to top_k of original results
                    results = results[:top_k]
            else:
                # No reranking, just take top_k
                results = results[:top_k]
            
            # Extract chunks and scores with better formatting
            chunks_retrieved = []
            sources = []
            
            for chunk, score, metadata in results:
                # Format the chunk with its metadata
                chunk_id = metadata.get("chunk_id", "unknown")
                source = metadata.get("source", "Unknown")
                
                # Add to retrieved chunks with more info
                chunks_retrieved.append((
                    chunk,
                    score,
                    {"chunk_id": chunk_id, "source": source}
                ))
                
                # Add to sources list
                sources.append(source)
            
            # Generate response with LLM if available
            response = ""
            llm_time_ms = 0
            
            if self.llm_path and os.path.exists(self.llm_path):
                llm_start = time.time()
                try:
                    from llama_cpp_interface import stream_llama_cpp_response
                    
                    # Create a more structured context with relevance info
                    context_parts = []
                    for i, (chunk, score, _) in enumerate(chunks_retrieved):
                        # Include chunk with its relevance score and position
                        context_parts.append(f"[Document {i+1} (Relevance: {score:.2f})]\n{chunk}")
                    
                    # Join with clear separators
                    context = "\n\n".join(context_parts)
                    
                    # Generate response
                    llm_response = stream_llama_cpp_response(
                        query=query_text,
                        context=context,
                        model="mamba",
                        temperature=temperature
                    )
                    
                    response = llm_response["response"] if isinstance(llm_response, dict) else llm_response
                    llm_time_ms = (time.time() - llm_start) * 1000
                    logger.info(f"Generated LLM response in {llm_time_ms:.2f}ms")
                except Exception as e:
                    logger.error(f"Error generating response with LLM: {str(e)}")
                    response = f"Error generating response: {str(e)}"
            
            # Format chunks for return (without the extra metadata we added)
            final_chunks = [(chunk, score) for chunk, score, _ in chunks_retrieved]
            
            # Calculate total query time
            query_time_ms = (time.time() - query_start_time) * 1000
            
            # Prepare result
            result = {
                "response": response,
                "chunks_retrieved": final_chunks,
                "sources": sources,
                "query_time_ms": query_time_ms,
                "llm_time_ms": llm_time_ms
            }
            
            # Cache the result
            self._query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            query_time_ms = (time.time() - query_start_time) * 1000
            logger.error(f"Error querying document: {str(e)}")
            return {
                "response": f"Error processing your query: {str(e)}",
                "chunks_retrieved": [],
                "sources": [],
                "query_time_ms": query_time_ms,
                "error": str(e)
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