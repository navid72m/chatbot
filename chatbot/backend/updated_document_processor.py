# Enhanced document_processor_patched.py with smart retrieval
import gc
import os
import logging
import time
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Optional
import torch
from vector_store import VectorStore, build_focused_context
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.schema import Document

# Import our smart retrieval components
from smart_universal_retrieval import SmartRAGPipeline, SmartEntity

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize reranker
try:
    reranker = CrossEncoder("BAAI/bge-reranker-base")
except:
    reranker = None
    logger.warning("Reranker not available")

def rerank_chunks(question, chunks, top_k=5):
    """Enhanced reranking with smart chunks support and better error handling."""
    if not reranker or not chunks:
        # Return original chunks if no reranker or empty input
        return chunks[:top_k]
    
    try:
        pairs = []
        processed_chunks = []
        
        for i, item in enumerate(chunks):
            try:
                # Handle different input formats
                if isinstance(item, tuple):
                    if len(item) >= 2:
                        document, score = item[0], item[1]
                    else:
                        document, score = item[0], 0.5
                else:
                    document, score = item, 0.5
                
                # Extract text for reranking
                text_content = ""
                if hasattr(document, 'page_content'):
                    text_content = document.page_content
                elif isinstance(document, dict):
                    if 'text' in document:
                        text_content = document['text']
                    elif 'page_content' in document:
                        text_content = document['page_content']
                    else:
                        text_content = str(document)
                else:
                    text_content = str(document)
                
                # Add to pairs for reranking
                pairs.append((question, text_content))
                processed_chunks.append((document, score))
                
            except Exception as e:
                logger.warning(f"Error processing chunk {i} for reranking: {e}")
                continue
        
        if not pairs:
            logger.warning("No valid pairs for reranking")
            return chunks[:top_k]
        
        # Get reranker scores
        rerank_scores = reranker.predict(pairs)
        
        # Combine original chunks with rerank scores and sort
        reranked = []
        for i, ((doc, original_score), rerank_score) in enumerate(zip(processed_chunks, rerank_scores)):
            # Combine original and rerank scores
            combined_score = 0.7 * float(rerank_score) + 0.3 * float(original_score)
            reranked.append((doc, combined_score))
        
        # Sort by combined score and return top_k
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Return original chunks on error
        try:
            # Try to return in proper format
            result = []
            for item in chunks[:top_k]:
                if isinstance(item, tuple) and len(item) >= 2:
                    result.append((item[0], item[1]))
                else:
                    result.append((item, 0.5))
            return result
        except:
            return chunks[:top_k]

# Simple SentenceTransformer implementation
class SimpleSentenceTransformer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def encode_single_text(self, text):
        return self.model.encode(text)
    
    def embed_query(self, text):
        result = self.encode_single_text(text)
        return result.tolist()
    
    def embed_documents(self, documents):
        results = []
        texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        embeddings = self.encode(texts, convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]

    def encode(self, texts, convert_to_numpy=True):
        embeddings = []
        for text in texts:
            embeddings.append(self.encode_single_text(text))
        return embeddings

# Direct embedding function
class LocalEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SimpleSentenceTransformer(model_name)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.model.encode([text], convert_to_numpy=True)[0]
        return embeddings.tolist()

# Global variables
embedding_function = None
vector_store = None
smart_rag_pipeline = None

def initialize():
    """Initialize both traditional and smart RAG systems."""
    global embedding_function, vector_store, smart_rag_pipeline
    
    # Initialize traditional components
    embedding_function = SimpleSentenceTransformer()
    vector_store = VectorStore()
    
    # Initialize smart RAG pipeline
    smart_rag_pipeline = SmartRAGPipeline()
    
    if vector_store is None:
        raise ValueError("Vector store is None")
    
    logger.info("Initialized both traditional and smart retrieval systems")

# Function to check if GPU is available for OCR
def is_gpu_available():
    """Check if GPU is available for acceleration."""
    try:
        import paddle
        gpu_available = paddle.device.is_compiled_with_cuda()
        logger.info(f"GPU acceleration available: {gpu_available}")
        return gpu_available
    except:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

# Function to check if a file is a valid image
def is_valid_image(file_path: str) -> bool:
    """Check if the file is a valid image that can be processed."""
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
        
    try:
        from PIL import Image
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            logger.warning(f"File extension not recognized as an image: {file_path}")
            return False
        
        try:
            with Image.open(file_path) as img:
                img.load()
                
                if img.width < 10 or img.height < 10:
                    logger.warning(f"Image is very small: {img.width}x{img.height}")
                    return False
                    
                if img.width > 10000 or img.height > 10000:
                    logger.warning(f"Image is very large: {img.width}x{img.height}")
            
            logger.info(f"Valid image: {file_path} ({img.width}x{img.height})")
            return True
        except Exception as e:
            logger.warning(f"Invalid image file {file_path}: {e}")
            return False
    except ImportError:
        logger.warning("PIL not installed, skipping image validation")
        return False

# Extract text from image using PaddleOCR
def extract_text_from_image(image_path: str, lang: str = 'en') -> str:
    """Extract text from an image using PaddleOCR with proper error handling."""
    if not is_valid_image(image_path):
        return f"[Image validation failed: {os.path.basename(image_path)}]"
    
    try:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.warning("PaddleOCR not installed, returning image metadata only")
            return create_image_document(image_path)
        
        use_gpu = is_gpu_available()
        logger.info(f"Initializing PaddleOCR (lang: {lang}, gpu: {use_gpu})")
        
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False
        )
        
        logger.info(f"Processing image: {image_path}")
        result = ocr.ocr(image_path, cls=True)
        
        extracted_text = ""
        
        if result and len(result) > 0:
            try:
                for page_idx, page_result in enumerate(result):
                    if not page_result:
                        continue
                        
                    if page_idx > 0:
                        extracted_text += "\n\n--- Page " + str(page_idx + 1) + " ---\n\n"
                    
                    for line in page_result:
                        if len(line) >= 2:
                            if isinstance(line[1], tuple) and len(line[1]) >= 2:
                                text = line[1][0]
                                confidence = line[1][1]
                                
                                if confidence > 0.5:
                                    extracted_text += text + " "
            except Exception as e:
                logger.warning(f"Error parsing OCR results: {str(e)}")
                try:
                    for item in result:
                        if isinstance(item, list):
                            for line in item:
                                if isinstance(line, tuple) and len(line) >= 2:
                                    text = str(line[1])
                                    extracted_text += text + " "
                                elif isinstance(line, list) and len(line) >= 2:
                                    text = str(line[1])
                                    extracted_text += text + " "
                except Exception:
                    logger.error("Failed to extract text with fallback method")
        
        extracted_text = extracted_text.strip()
        
        if not extracted_text:
            logger.warning(f"No text extracted from image: {image_path}")
            return create_image_document(image_path)
        
        logger.info(f"Extracted {len(extracted_text)} characters from image {image_path}")
        return extracted_text
        
    except Exception as e:
        logger.error(f"Error during OCR processing: {str(e)}")
        return create_image_document(image_path)

def create_image_document(file_path: str) -> str:
    """Creates a simple document for image files with metadata."""
    try:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_size_kb = file_size / 1024
        
        image_info = ""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                image_info = f"\nImage dimensions: {img.width}x{img.height} pixels\nImage format: {img.format}\nImage mode: {img.mode}"
        except:
            image_info = ""
        
        description = f"[Image file: {file_name}]\nFile size: {file_size_kb:.2f} KB{image_info}\n\nThis is an image file. Install PaddleOCR for text extraction."
        
        return description
    except Exception as e:
        logger.error(f"Error creating image document: {e}")
        return f"[Image file: {os.path.basename(file_path)}]\nError processing image file."

def estimate_token_count(text: str) -> int:
    """Roughly estimate token count - 1 token ≈ 4 chars for English text."""
    if not text:
        return 0
    return len(text) // 4

def truncate_text_for_context_window(text: str, max_tokens: int = 6000) -> str:
    """Truncate text to fit within model's context window with buffer."""
    if not text:
        return ""
        
    estimated_tokens = estimate_token_count(text)
    
    if estimated_tokens <= max_tokens:
        return text
    
    logger.warning(f"Text too long (est. {estimated_tokens} tokens > {max_tokens}), truncating")
    
    truncate_ratio = max_tokens / estimated_tokens
    truncate_chars = int(len(text) * truncate_ratio)
    truncate_chars = int(truncate_chars * 0.9)
    truncated_text = text[:truncate_chars]
    
    truncation_notice = f"\n\n[Note: Text was truncated from {estimated_tokens} tokens to fit model context window]"
    
    return truncated_text + truncation_notice

def sanitize_text(text: str) -> str:
    """Remove control characters and normalize whitespace."""
    if not text:
        return ""
    
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_and_index_file(file_path: str, use_smart_processing: bool = True):
    """
    Enhanced processing that uses both traditional and smart RAG systems.
    """
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    global embedding_function, vector_store, smart_rag_pipeline
    
    # Initialize if needed
    if embedding_function is None or vector_store is None or smart_rag_pipeline is None:
        initialize()
    
    logger.info(f"Processing file with smart processing={'enabled' if use_smart_processing else 'disabled'}: {file_path}")
    
    try:
        # Extract text based on file type
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']:
            logger.info(f"Processing image file: {file_path}")
            text = extract_text_from_image(file_path)
            
        elif file_extension == '.pdf':
            text = extract_pdf_text(file_path)
            
        elif file_extension in ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.js', '.py', '.java', '.c', '.cpp', '.rb', '.php']:
            text = extract_text_file(file_path)
        else:
            text = f"[File: {os.path.basename(file_path)}]\nUnsupported file type for text extraction."
        
        # Clean up text
        text = sanitize_text(text)
        text = truncate_text_for_context_window(text)
        
        if use_smart_processing and smart_rag_pipeline:
            # Use smart processing
            try:
                smart_result = smart_rag_pipeline.process_document_smart(file_path, text)
                
                # Convert smart chunks to traditional format for backward compatibility
                traditional_chunks = convert_smart_chunks_to_traditional(smart_result, file_path)
                
                # Generate enhanced suggestions
                suggested_questions = smart_result.get("suggestions", [])
                
                # Also add to traditional vector store for backward compatibility
                vector_store.add_document(file_path, traditional_chunks)
                
                logger.info(f"Smart processing complete: {len(traditional_chunks)} chunks, "
                          f"{smart_result['entities_found']} entities, "
                          f"document type: {smart_result['document_type']}")
                
                return traditional_chunks, suggested_questions, smart_result
                
            except Exception as e:
                logger.error(f"Smart processing failed: {str(e)}, falling back to traditional")
                use_smart_processing = False
        
        if not use_smart_processing:
            # Traditional processing
            chunks = create_traditional_chunks(text, file_path)
            
            # Generate basic suggestions
            from generate_suggestions import generate_suggested_questions
            safe_text = text[:5000] if len(text) > 5000 else text
            suggested_questions = generate_suggested_questions(safe_text)
            
            # Add to vector store
            if chunks:
                vector_store.add_document(file_path, chunks)
                logger.info(f"Traditional processing complete: {len(chunks)} chunks")
            
            return chunks, suggested_questions
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise e

def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF files."""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                try:
                    page_text = reader.pages[page_num].extract_text() + "\n\n"
                    text += page_text
                    
                    if estimate_token_count(text) > 6000:
                        logger.warning(f"PDF too large, truncating after page {page_num+1}")
                        text += f"\n[Note: PDF truncated after {page_num+1} pages due to size]"
                        break
                except Exception as e:
                    logger.warning(f"Error extracting text from PDF page {page_num}: {e}")
                    text += f"\n[Error extracting page {page_num}]\n\n"
                    
    except ImportError:
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text() + "\n\n"
                        text += page_text
                        
                        if estimate_token_count(text) > 6000:
                            logger.warning(f"PDF too large, truncating after page {i+1}")
                            text += f"\n[Note: PDF truncated after {i+1} pages due to size]"
                            break
                    except Exception as e:
                        logger.warning(f"Error extracting text from PDF page {i}: {e}")
                        text += f"\n[Error extracting page {i}]\n\n"
        except ImportError:
            raise ImportError("PDF extraction libraries not found.")
    
    if not text:
        text = f"[PDF file: {os.path.basename(file_path)}]\nFailed to extract text from this PDF file."
    
    return text

def extract_text_file(file_path: str) -> str:
    """Extract text from text-based files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        return f"[Text file: {os.path.basename(file_path)}]\nCould not decode file with any standard text encoding."

def convert_smart_chunks_to_traditional(smart_result: Dict[str, Any], file_path: str) -> List[Document]:
    """Convert smart chunks to traditional LangChain Document format."""
    traditional_chunks = []
    
    # Get chunks from smart pipeline
    if smart_rag_pipeline and smart_rag_pipeline.vector_store.chunks:
        document_name = smart_result["document_name"]
        
        for i, (chunk, metadata) in enumerate(zip(smart_rag_pipeline.vector_store.chunks, 
                                                 smart_rag_pipeline.vector_store.metadata)):
            if metadata["document"] == document_name:
                # Convert to LangChain Document
                doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        "source": file_path,
                        "chunk_id": i,
                        "chunk_type": chunk.get("type", "content"),
                        "priority": chunk.get("metadata", {}).get("priority", "medium"),
                        "has_entities": len(chunk.get("entities", [])) > 0,
                        "entity_count": len(chunk.get("entities", [])),
                        "document_type": smart_result.get("document_type", "general"),
                        "smart_processed": True
                    }
                )
                traditional_chunks.append(doc)
    
    return traditional_chunks

def create_traditional_chunks(text: str, file_path: str) -> List[Document]:
    """Create traditional chunks for backward compatibility."""
    max_chunk_size = 512
    
    # Split by sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    chunks = []
    current_chunk = ""
    chunk_id = 0
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={
                        "source": file_path, 
                        "chunk_id": chunk_id,
                        "smart_processed": False
                    }
                ))
                chunk_id += 1
                current_chunk = sentence
            else:
                if len(sentence) > max_chunk_size:
                    chunks.append(Document(
                        page_content=sentence[:max_chunk_size].strip(),
                        metadata={
                            "source": file_path, 
                            "chunk_id": chunk_id,
                            "smart_processed": False
                        }
                    ))
                    chunk_id += 1
                else:
                    current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(Document(
            page_content=current_chunk.strip(),
            metadata={
                "source": file_path, 
                "chunk_id": chunk_id,
                "smart_processed": False
            }
        ))
    
    return chunks

def query_index(query, top_k=5, use_smart_retrieval=True):
    """Enhanced query function that can use smart retrieval."""
    global embedding_function, vector_store, smart_rag_pipeline
    
    # Initialize if needed
    if embedding_function is None or vector_store is None:
        initialize()
    
    # Validate query
    if isinstance(query, bool) or not isinstance(query, str):
        if not isinstance(query, str):
            try:
                query = str(query)
                logger.warning(f"Converted non-string query to string: {query}")
            except:
                logger.error(f"Cannot convert {type(query)} to string")
                return []
    
    try:
        if not query.strip():
            logger.warning("Empty query received")
            return []
        
        # Try smart retrieval first if available and enabled
        if use_smart_retrieval and smart_rag_pipeline and hasattr(smart_rag_pipeline, 'vector_store'):
            try:
                # Get current document context (this would need to be passed or stored globally)
                # For now, we'll search across all documents in smart store
                all_documents = set()
                for metadata in smart_rag_pipeline.vector_store.metadata:
                    all_documents.add(metadata["document"])
                
                if all_documents:
                    # Use the first document for now - in practice, you'd pass the current document
                    current_doc = list(all_documents)[0]
                    smart_results = smart_rag_pipeline.query_smart(query, current_doc, top_k)
                    
                    # Convert smart results to traditional format
                    traditional_results = []
                    for chunk, score in smart_results["chunks_retrieved"]:
                        # Create Document object
                        doc = Document(
                            page_content=chunk["text"],
                            metadata={
                                "source": current_doc,
                                "chunk_type": chunk.get("type", "content"),
                                "smart_retrieval": True,
                                "relevance_score": score
                            }
                        )
                        traditional_results.append((doc, score))
                    
                    logger.info(f"Smart retrieval returned {len(traditional_results)} results")
                    return traditional_results
            
            except Exception as e:
                logger.warning(f"Smart retrieval failed: {str(e)}, falling back to traditional")
        
        # Fall back to traditional retrieval
        results = vector_store.search(query, k=top_k)
        
        # Handle different return formats from vector_store.search
        processed_results = []
        for result in results:
            try:
                if isinstance(result, tuple):
                    if len(result) == 2:
                        # Format: (doc, score)
                        doc, score = result
                        processed_results.append((doc, score))
                    elif len(result) == 3:
                        # Format: (doc, score, metadata) - extract first two
                        doc, score, _ = result
                        processed_results.append((doc, score))
                    else:
                        logger.warning(f"Unexpected tuple length in search results: {len(result)}")
                        continue
                else:
                    # Single document, assign default score
                    processed_results.append((result, 0.5))
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
        
        # Apply reranking to processed results
        reranked_results = rerank_chunks(query, processed_results, top_k)
        
        # Ensure total content is manageable
        max_content_length = 1000
        total_length = 0
        pruned_results = []
        
        for doc, score in reranked_results:
            try:
                # Handle different document formats
                if hasattr(doc, 'page_content'):
                    content_length = len(doc.page_content)
                elif isinstance(doc, dict) and 'text' in doc:
                    content_length = len(doc['text'])
                else:
                    content_length = len(str(doc))
                
                if total_length + content_length <= max_content_length:
                    pruned_results.append((doc, score))
                    total_length += content_length
                else:
                    logger.warning(f"Skipping document with length {content_length} to avoid exceeding context window")
            except Exception as e:
                logger.warning(f"Error processing document in pruning: {e}")
                continue
        
        return pruned_results
    
    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def query_index_with_context(query, top_k=5, use_smart_retrieval=True):
    """Enhanced context building with smart retrieval support."""
    global embedding_function, vector_store
    results = query_index(query, top_k, use_smart_retrieval)
    
    content = ""
    for item in results:
        try:
            # Handle different return formats
            if isinstance(item, tuple) and len(item) >= 2:
                doc, score = item[0], item[1]
            else:
                doc = item
                score = 0.5
            
            # Handle different document formats
            if hasattr(doc, 'page_content'):
                content += f"Document: {doc.page_content}\n"
            elif isinstance(doc, dict):
                if 'text' in doc:
                    content += f"Document: {doc['text']}\n"
                elif 'page_content' in doc:
                    content += f"Document: {doc['page_content']}\n"
                else:
                    content += f"Document: {str(doc)}\n"
            else:
                content += f"Document: {str(doc)}\n"
        except Exception as e:
            logger.warning(f"Error processing document in context building: {e}")
            continue
    
    # Final check for context window size
    if estimate_token_count(content) > 1500:
        logger.warning(f"Context too large ({estimate_token_count(content)} est. tokens), truncating")
        content = truncate_text_for_context_window(content, 1500)
        
    return content

def build_focused_context(query, results):
    """Build focused context from results with improved error handling."""
    context = f"Query: {query}\n\n"
    
    for item in results:
        try:
            # Handle different return formats
            if isinstance(item, tuple) and len(item) >= 2:
                doc, score = item[0], item[1]
            else:
                doc = item
                score = 0.5
            
            # Handle different document formats
            if hasattr(doc, 'page_content'):
                context += f"Document: {doc.page_content}\n"
            elif isinstance(doc, dict):
                if 'text' in doc:
                    context += f"Document: {doc['text']}\n"
                elif 'page_content' in doc:
                    context += f"Document: {doc['page_content']}\n"
                else:
                    context += f"Document: {str(doc)}\n"
            else:
                context += f"Document: {str(doc)}\n"
        except Exception as e:
            logger.warning(f"Error processing document in context building: {e}")
            continue
    
    return context

def get_smart_processing_stats():
    """Get statistics from smart processing."""
    if smart_rag_pipeline:
        return smart_rag_pipeline.get_processing_stats()
    else:
        return {"error": "Smart RAG pipeline not initialized"}

def is_valid_chunk(text: str) -> bool:
    """Check if a chunk is valid for embedding."""
    if not text:
        return False
        
    if len(text) < 5:
        return False
        
    # Check for excessive non-printable characters
    non_printable = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
    if non_printable / len(text) > 0.1:
        return False
        
    # Check for very high ratio of digits or special chars
    digits_special = sum(1 for c in text if c.isdigit() or not c.isalnum())
    if digits_special / len(text) > 0.7:
        return False
        
    # Check for known binary signatures
    binary_indicators = ['JFIF', 'PNG', 'PDF', 'ÿØÿà', 'PK\x03\x04']
    for indicator in binary_indicators:
        if indicator in text:
            return False
            
    if len(text) > 10000:
        return False
        
    return True

# Example usage
if __name__ == "__main__":
    pass