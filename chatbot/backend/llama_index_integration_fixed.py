import os
import logging
import time
from typing import List, Optional, Dict, Any
import torch
from sentence_transformers import SentenceTransformer
import json

# Import our new query rewriting components
from query_rewriting import QueryRewriter, EnhancedRAGRetriever

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


class EnhancedLlamaIndexRAG:
    """Enhanced LlamaIndex RAG system with query rewriting capabilities."""
    
    def __init__(self, llm_model_path: str = None):
        """Initialize the enhanced RAG system."""
        # Set up embedding model
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        
        # Store LLM path for later use
        self.llm_path = llm_model_path
        
        # Track processed documents
        self.processed_documents = {}
        
        # Initialize query rewriter
        self.query_rewriter = QueryRewriter(self.embed_model)
        
        # Cache for vector stores and queries
        self._vector_store_cache = {}
        self._query_cache = {}
        
        # Enhanced retriever will be initialized per document
        self._enhanced_retrievers = {}
    
    def _get_index_path(self, document_name: str) -> str:
        """Get storage path for a document index."""
        # Create a safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in document_name)
        return os.path.join(INDEX_STORAGE_DIR, f"{safe_name}_index.json")
    
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
            text = self._extract_text_from_pdf(file_path)
        else:
            text = self._extract_text_from_txt(file_path)
        
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
                    # Limit size to avoid memory issues
                    if len(text) > 1000000:  # 1MB limit
                        text += "\n[Note: PDF truncated due to size]"
                        break
            return text
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n\n"
                        if len(text) > 1000000:  # 1MB limit
                            text += "\n[Note: PDF truncated due to size]"
                            break
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
        import re
        
        # Pre-process text to remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Quick return for very short texts
        if len(text) <= chunk_size:
            return [text]
        
        # Start with paragraph splitting
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    # Keep overlap text for next chunk
                    if overlap > 0:
                        words = current_chunk.split()
                        overlap_word_count = min(20, len(words))
                        current_chunk = " ".join(words[-overlap_word_count:])
                    else:
                        current_chunk = ""
                
                # If paragraph itself is larger than chunk size, split it
                if len(para) > chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= chunk_size:
                            current_chunk += " " + sentence if current_chunk else sentence
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                # Force split very long sentences
                                words = sentence.split()
                                temp_chunk = ""
                                for word in words:
                                    if len(temp_chunk) + len(word) + 1 <= chunk_size:
                                        temp_chunk += " " + word if temp_chunk else word
                                    else:
                                        if temp_chunk:
                                            chunks.append(temp_chunk.strip())
                                        temp_chunk = word
                                if temp_chunk:
                                    current_chunk = temp_chunk
                else:
                    current_chunk = para
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_document(self, file_path: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Process a document and create an index with enhanced capabilities."""
        try:
            # Get document name from path
            document_name = os.path.basename(file_path)
            index_path = self._get_index_path(document_name)
            
            # Check for existing index
            if os.path.exists(index_path):
                logger.info(f"Found existing index for {document_name}, loading instead of reprocessing")
                vector_store = SimpleVectorStore()
                vector_store.load(index_path)
                
                # Initialize enhanced retriever for this document
                self._enhanced_retrievers[document_name] = EnhancedRAGRetriever(vector_store, self.embed_model)
                
                return vector_store.documents
            
            # Extract text based on file type
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                logger.info(f"Processing image file: {document_name}")
                from document_processor_patched import extract_text_from_image
                document_text = extract_text_from_image(file_path)
            else:
                document_text = self._extract_text_from_file(file_path)
            
            # Split into chunks
            chunks = self._chunk_text(document_text, chunk_size, overlap)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Create vector store
            vector_store = SimpleVectorStore()
            
            # Create embeddings in batches
            batch_size = 16
            embeddings = []
            metadata = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:min(i+batch_size, len(chunks))]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
                batch_embeddings = self.embed_model.encode(batch, show_progress_bar=False)
                
                for j, embedding in enumerate(batch_embeddings):
                    chunk_index = i + j
                    embeddings.append(embedding.tolist())
                    metadata.append({
                        "source": document_name,
                        "chunk_id": chunk_index
                    })
            
            # Add to vector store
            vector_store.add_documents(chunks, embeddings, metadata)
            
            # Save vector store
            vector_store.save(index_path)
            
            # Initialize enhanced retriever for this document
            self._enhanced_retrievers[document_name] = EnhancedRAGRetriever(vector_store, self.embed_model)
            
            # Record this document as processed
            self.processed_documents[document_name] = {
                "chunks": len(chunks),
                "path": file_path
            }
            
            logger.info(f"Created enhanced index for {document_name} with {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    def query_with_rewriting(
        self, 
        query_text: str, 
        document_name: str, 
        top_k: int = 5,
        temperature: float = 0.3,
        use_prf: bool = True,
        use_variants: bool = True,
        prf_iterations: int = 1,
        fusion_method: str = "rrf",
        rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Query a document using advanced query rewriting techniques.
        
        Args:
            query_text: The user's query
            document_name: Name of the document to query
            top_k: Number of chunks to retrieve
            temperature: Temperature for LLM response generation
            use_prf: Whether to use Pseudo Relevance Feedback
            use_variants: Whether to generate query variants
            prf_iterations: Number of PRF iterations
            fusion_method: Method to fuse results ("rrf" or "score")
            rerank: Whether to apply reranking
            
        Returns:
            Dictionary with enhanced response and metadata
        """
        query_start_time = time.time()
        
        try:
            logger.info(f"Enhanced query with rewriting: {query_text}")
            
            # Check if we have an enhanced retriever for this document
            if document_name not in self._enhanced_retrievers:
                # Try to load the document index
                index_path = self._get_index_path(document_name)
                if os.path.exists(index_path):
                    vector_store = SimpleVectorStore()
                    vector_store.load(index_path)
                    self._enhanced_retrievers[document_name] = EnhancedRAGRetriever(vector_store, self.embed_model)
                else:
                    logger.error(f"No index found for document: {document_name}")
                    return {
                        "response": f"No index found for document: {document_name}",
                        "chunks_retrieved": [],
                        "sources": [],
                        "query_rewriting": {"original_query": query_text},
                        "error": "Document not indexed"
                    }
            
            # Get the enhanced retriever for this document
            retriever = self._enhanced_retrievers[document_name]
            
            # Perform enhanced retrieval with query rewriting
            retrieval_result = retriever.retrieve_with_rewriting(
                query=query_text,
                document_name=document_name,
                top_k=top_k,
                use_prf=use_prf,
                use_variants=use_variants,
                prf_iterations=prf_iterations,
                fusion_method=fusion_method
            )
            
            # Extract retrieved documents
            retrieved_docs = retrieval_result["retrieved_documents"]
            
            # Apply reranking if requested
            if rerank and len(retrieved_docs) > 1:
                try:
                    rerank_start = time.time()
                    if not hasattr(self, '_reranker'):
                        from sentence_transformers import CrossEncoder
                        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    
                    # Prepare pairs for reranking
                    pairs = [(query_text, doc) for doc, _ in retrieved_docs]
                    rerank_scores = self._reranker.predict(pairs)
                    
                    # Combine and sort by rerank scores
                    reranked = sorted(zip(retrieved_docs, rerank_scores), key=lambda x: x[1], reverse=True)
                    retrieved_docs = [item[0] for item in reranked]
                    
                    logger.info(f"Reranked results in {(time.time() - rerank_start)*1000:.2f}ms")
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}, using original order")
            
            # Generate response with LLM
            response = ""
            llm_time_ms = 0
            
            if self.llm_path and os.path.exists(self.llm_path):
                llm_start = time.time()
                try:
                    from llama_cpp_interface import stream_llama_cpp_response
                    
                    # Create enhanced context with query rewriting information
                    context_parts = []
                    
                    # Add information about query rewriting
                    rewriting_info = retrieval_result["query_rewriting"]
                    if len(rewriting_info["all_queries"]) > 1:
                        context_parts.append(f"[Query Analysis: Original query was expanded to {len(rewriting_info['all_queries'])} variants for better retrieval]")
                    
                    # Add retrieved documents with relevance scores
                    for i, (chunk, score) in enumerate(retrieved_docs):
                        context_parts.append(f"[Document {i+1} (Relevance: {score:.3f})]\n{chunk}")
                    
                    # Create context with clear separators
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
                    
                    logger.info(f"Generated enhanced response in {llm_time_ms:.2f}ms")
                    
                except Exception as e:
                    logger.error(f"Error generating response with LLM: {str(e)}")
                    response = f"Error generating response: {str(e)}"
            
            # Calculate total query time
            query_time_ms = (time.time() - query_start_time) * 1000
            
            # Prepare final result
            result = {
                "response": response,
                "chunks_retrieved": retrieved_docs,
                "sources": [document_name] * len(retrieved_docs),
                "query_rewriting": retrieval_result["query_rewriting"],
                "retrieval_metadata": retrieval_result["metadata"],
                "individual_results": retrieval_result["individual_results"],
                "fusion_method": fusion_method,
                "query_time_ms": query_time_ms,
                "llm_time_ms": llm_time_ms,
                "techniques_used": {
                    "prf": use_prf,
                    "variants": use_variants,
                    "reranking": rerank,
                    "fusion": fusion_method
                }
            }
            
            return result
            
        except Exception as e:
            query_time_ms = (time.time() - query_start_time) * 1000
            logger.error(f"Error in enhanced query: {str(e)}")
            return {
                "response": f"Error processing your query: {str(e)}",
                "chunks_retrieved": [],
                "sources": [],
                "query_rewriting": {"original_query": query_text, "error": str(e)},
                "query_time_ms": query_time_ms,
                "error": str(e)
            }
    
    def query(self, query_text: str, document_name: str, top_k: int = 5, temperature: float = 0.3, **kwargs) -> Dict[str, Any]:
        """
        Backward compatible query method that uses enhanced querying by default.
        """
        # Extract enhancement parameters from kwargs
        use_prf = kwargs.get('use_prf', True)
        use_variants = kwargs.get('use_variants', True)
        prf_iterations = kwargs.get('prf_iterations', 1)
        fusion_method = kwargs.get('fusion_method', 'rrf')
        rerank = kwargs.get('rerank', True)
        
        # Use enhanced query method
        enhanced_result = self.query_with_rewriting(
            query_text=query_text,
            document_name=document_name,
            top_k=top_k,
            temperature=temperature,
            use_prf=use_prf,
            use_variants=use_variants,
            prf_iterations=prf_iterations,
            fusion_method=fusion_method,
            rerank=rerank
        )
        
        # Return in the expected format for backward compatibility
        return {
            "response": enhanced_result["response"],
            "chunks_retrieved": enhanced_result["chunks_retrieved"],
            "sources": enhanced_result["sources"],
            "query_time_ms": enhanced_result.get("query_time_ms", 0),
            "enhancement_metadata": {
                "query_rewriting": enhanced_result.get("query_rewriting", {}),
                "techniques_used": enhanced_result.get("techniques_used", {}),
                "retrieval_metadata": enhanced_result.get("retrieval_metadata", {})
            }
        }
    
    def get_document_list(self) -> List[str]:
        """Get list of documents that have indices."""
        try:
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
    
    def get_query_rewriting_stats(self, document_name: str = None) -> Dict[str, Any]:
        """Get statistics about query rewriting performance."""
        try:
            stats = {
                "total_documents": len(self.processed_documents),
                "enhanced_retrievers": len(self._enhanced_retrievers),
                "cache_sizes": {
                    "vector_stores": len(self._vector_store_cache),
                    "queries": len(self._query_cache)
                }
            }
            
            if document_name and document_name in self.processed_documents:
                doc_info = self.processed_documents[document_name]
                stats["document_info"] = {
                    "chunks": doc_info["chunks"],
                    "path": doc_info["path"],
                    "has_enhanced_retriever": document_name in self._enhanced_retrievers
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting query rewriting stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_caches(self):
        """Clear all caches to free memory."""
        self._vector_store_cache.clear()
        self._query_cache.clear()
        
        # Clear caches in enhanced retrievers
        for retriever in self._enhanced_retrievers.values():
            if hasattr(retriever, 'retrieval_cache'):
                retriever.retrieval_cache.clear()
        
        logger.info("Cleared all caches")