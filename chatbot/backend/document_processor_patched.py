import gc
import os
import logging
import time
import numpy as np
import re
from typing import List, Tuple, Dict, Any, Optional
import torch
from vector_store import VectorStore, build_focused_context
from sentence_transformers import SentenceTransformer
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# Set up logging
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
def rerank_chunks(question, chunks, top_k=5):
                # pairs = [(question, chunk) for chunk, _ in chunks]
                pairs = []
                for i, doc in enumerate(chunks):
                    logger.info(f"doc_rerank: {doc}")
                    d, score = doc
                    pairs.append((question, d.page_content))
                scores = reranker.predict(pairs)
                reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
                return reranked[:top_k]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple SentenceTransformer implementation without dependencies
class SimpleSentenceTransformer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def encode_single_text(self, text):
        return self.model.encode(text)
    # Then update embed_query and embed_documents methods
    def embed_query(self, text):
        result = self.encode_single_text(text)
        return result.tolist()
    

    def embed_documents(self, documents):
        results = []
        texts = [doc.page_content for doc in documents]
        for text in texts:
            # Process one at a time
            embedding = self.encode_single_text(text)
            results.append(embedding.tolist())
            # Add small delay between processing to allow memory cleanup
            time.sleep(0.1)
        return results
    def encode(self, texts, convert_to_numpy=True):
        embeddings = []
        for text in texts:
            embeddings.append(self.encode_single_text(text, convert_to_numpy))

        return embeddings
        
# Direct embedding function without any abstract base class
class LocalEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SimpleSentenceTransformer(model_name)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"texts: {texts}")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        logger.info(f"embeddings: {embeddings}")
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.model.encode([text], convert_to_numpy=True)[0]
        return embeddings.tolist()

# Document class compatible with your evaluator expectations
from langchain.schema import Document

# Global variables
embedding_function = None
vector_store = None

# Initialize the models
def initialize():
    global embedding_function, vector_store
    embedding_function = SimpleSentenceTransformer()
    vector_store = VectorStore()
    if vector_store is None:
        raise ValueError("Vector store is None")
    logger.info("Initialized embedding model and vector store")

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
    """
    Check if the file is a valid image that can be processed.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
        
    try:
        from PIL import Image
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            logger.warning(f"File extension not recognized as an image: {file_path}")
            return False
        
        try:
            # Try to open the image
            with Image.open(file_path) as img:
                # Force load the image data to check for corruption
                img.load()
                
                # Check for reasonable dimensions
                if img.width < 10 or img.height < 10:
                    logger.warning(f"Image is very small: {img.width}x{img.height}")
                    return False
                    
                if img.width > 10000 or img.height > 10000:
                    logger.warning(f"Image is very large: {img.width}x{img.height}")
                    # We'll still try to process it, but warn the user
            
            logger.info(f"Valid image: {file_path} ({img.width}x{img.height})")
            return True
        except Exception as e:
            logger.warning(f"Invalid image file {file_path}: {e}")
            return False
    except ImportError:
        logger.warning("PIL not installed, skipping image validation")
        return False

# Extract text from image using PaddleOCR with better error handling
def extract_text_from_image(image_path: str, lang: str = 'en') -> str:
    """
    Extract text from an image using PaddleOCR with proper error handling
    
    Args:
        image_path: Path to the image file
        lang: Language for OCR
        
    Returns:
        Extracted text as string
    """
    # Verify the image first
    if not is_valid_image(image_path):
        return f"[Image validation failed: {os.path.basename(image_path)}]"
    
    try:
        # Check if PaddleOCR is available
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.warning("PaddleOCR not installed, returning image metadata only")
            return create_image_document(image_path)
        
        # Initialize PaddleOCR with appropriate settings
        use_gpu = is_gpu_available()
        logger.info(f"Initializing PaddleOCR (lang: {lang}, gpu: {use_gpu})")
        
        ocr = PaddleOCR(
            use_angle_cls=True,  # Detect text orientation
            lang=lang,
            use_gpu=use_gpu,
            show_log=False
        )
        
        # Run OCR
        logger.info(f"Processing image: {image_path}")
        result = ocr.ocr(image_path, cls=True)
        
        # Extract text from results
        extracted_text = ""
        
        # Process results - handle different PaddleOCR result formats
        if result and len(result) > 0:
            try:
                # For each page in the result
                for page_idx, page_result in enumerate(result):
                    # Skip empty pages
                    if not page_result:
                        continue
                        
                    # Add page separator for multi-page documents
                    if page_idx > 0:
                        extracted_text += "\n\n--- Page " + str(page_idx + 1) + " ---\n\n"
                    
                    # For each line in the page
                    for line in page_result:
                        # Extract text and confidence
                        if len(line) >= 2:
                            # Text and confidence should be in a tuple
                            if isinstance(line[1], tuple) and len(line[1]) >= 2:
                                text = line[1][0]         # Text content
                                confidence = line[1][1]   # Confidence score
                                
                                # Only include text with reasonable confidence
                                if confidence > 0.5:
                                    extracted_text += text + " "
            except Exception as e:
                logger.warning(f"Error parsing OCR results: {str(e)}")
                # Fallback to a more general extraction method
                try:
                    # Just try to extract any text we can find in the results
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
        
        # Clean up text
        extracted_text = extracted_text.strip()
        logger.error(f"extracted_text: {extracted_text}")
        # If we got no text, return metadata
        if not extracted_text:
            logger.warning(f"No text extracted from image: {image_path}")
            return create_image_document(image_path)
        
        logger.info(f"Extracted {len(extracted_text)} characters from image {image_path}")
        return extracted_text
        
    except Exception as e:
        logger.error(f"Error during OCR processing: {str(e)}")
        # Return image metadata on error
        return create_image_document(image_path)

# Create a basic image document with metadata
def create_image_document(file_path: str) -> str:
    """
    Creates a simple document for image files with metadata
    This is a fallback when OCR fails or isn't available
    """
    try:
        # Get basic file information
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_size_kb = file_size / 1024
        
        # Try to get image dimensions if PIL is available
        image_info = ""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                image_info = f"\nImage dimensions: {img.width}x{img.height} pixels\nImage format: {img.format}\nImage mode: {img.mode}"
        except:
            image_info = ""
        
        # Create a simple description
        description = f"[Image file: {file_name}]\nFile size: {file_size_kb:.2f} KB{image_info}\n\nThis is an image file. Install PaddleOCR for text extraction."
        
        return description
    except Exception as e:
        logger.error(f"Error creating image document: {e}")
        return f"[Image file: {os.path.basename(file_path)}]\nError processing image file."

# Estimate token count for context window management
def estimate_token_count(text: str) -> int:
    """Roughly estimate token count - 1 token ≈ 4 chars for English text"""
    if not text:
        return 0
    return len(text) // 4

# Truncate text to fit context window
def truncate_text_for_context_window(text: str, max_tokens: int = 6000) -> str:
    """Truncate text to fit within model's context window with buffer"""
    if not text:
        return ""
        
    estimated_tokens = estimate_token_count(text)
    
    if estimated_tokens <= max_tokens:
        return text
    
    # If text is too long, truncate it
    logger.warning(f"Text too long (est. {estimated_tokens} tokens > {max_tokens}), truncating to fit context window")
    
    # Truncate proportionally to get under token limit
    truncate_ratio = max_tokens / estimated_tokens
    truncate_chars = int(len(text) * truncate_ratio)
    
    # Take first 90% of allowed chars and append a notice
    truncate_chars = int(truncate_chars * 0.9)  # 90% to leave room for the notice
    truncated_text = text[:truncate_chars]
    
    # Add a notice about truncation
    truncation_notice = f"\n\n[Note: Text was truncated from {estimated_tokens} tokens to fit model context window]"
    
    return truncated_text + truncation_notice

# Sanitize text - remove control characters that cause problems
def sanitize_text(text: str) -> str:
    """Remove control characters and normalize whitespace"""
    if not text:
        return ""
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Document processing and chunking
def process_and_index_file(file_path: str):
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    global embedding_function, vector_store
    
    # Initialize if needed
    if embedding_function is None or vector_store is None:
        initialize()
    
    logger.info(f"Processing file: {file_path}")
    
    try:
        # Get file extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Handle each file type differently
        if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp']:
            # For image files - use OCR with better error handling
            logger.info(f"Processing image file with OCR: {file_path}")
            text = extract_text_from_image(file_path)
            
        elif file_extension == '.pdf':
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        try:
                            page_text = reader.pages[page_num].extract_text() + "\n\n"
                            text += page_text
                            
                            # Check if we're exceeding context window limits
                            if estimate_token_count(text) > 6000:
                                logger.warning(f"PDF too large, truncating after page {page_num+1} of {len(reader.pages)}")
                                text += f"\n[Note: PDF truncated after {page_num+1} pages due to size limitations]"
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
                                
                                # Check if we're exceeding context window limits
                                if estimate_token_count(text) > 6000:
                                    logger.warning(f"PDF too large, truncating after page {i+1} of {len(pdf.pages)}")
                                    text += f"\n[Note: PDF truncated after {i+1} pages due to size limitations]"
                                    break
                            except Exception as e:
                                logger.warning(f"Error extracting text from PDF page {i}: {e}")
                                text += f"\n[Error extracting page {i}]\n\n"
                except ImportError:
                    raise ImportError("PDF extraction libraries not found.")
                    
            # If PDF text extraction failed entirely
            if not text:
                text = f"[PDF file: {os.path.basename(file_path)}]\nFailed to extract text from this PDF file."
                        
        elif file_extension in ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.js', '.py', '.java', '.c', '.cpp', '.rb', '.php']:
            # For text files, try UTF-8 first, then other encodings
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Try other encodings
                encodings = ['latin-1', 'cp1252', 'iso-8859-1']
                text = None
                
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                        
                if text is None:
                    # If all encodings fail, report error
                    text = f"[Text file: {os.path.basename(file_path)}]\nCould not decode file with any standard text encoding."
        else:
            # For unknown file types, just create a basic document
            text = f"[File: {os.path.basename(file_path)}]\nUnsupported file type. This file type is not supported for text extraction."
        
        # Clean up text - important to avoid binary data
        text = sanitize_text(text)
        
        # Limit text size for context window
        text = truncate_text_for_context_window(text)
        
        # Split text into chunks
        max_chunk_size = 200  # Small enough to avoid memory issues
        
        # Split by sentences or short paragraphs
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # Skip very long sentences (likely binary data or errors)
            # if len(sentence) > 500:
            #     logger.warning(f"Skipping unusually long sentence: {sentence[:50]}...")
            #     continue
                
            logger.info(f"length of para: {len(sentence)}")
            
            # If this sentence would make the chunk too large, save current chunk and start a new one
            if len(current_chunk) + len(sentence) > max_chunk_size:
                if current_chunk:  # Only add if there's content
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={"source": file_path, "chunk_id": chunk_id}
                    ))
                    chunk_id += 1
                    current_chunk = sentence
                else:
                    # If a single sentence is longer than max_chunk_size, truncate it
                    if len(sentence) > max_chunk_size:
                        chunks.append(Document(
                            page_content=sentence[:max_chunk_size].strip(),
                            metadata={"source": file_path, "chunk_id": chunk_id}
                        ))
                        chunk_id += 1
                    else:
                        current_chunk = sentence
            else:
                # Add to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the final chunk
        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={"source": file_path, "chunk_id": chunk_id}
            ))
        
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Limit total number of chunks to prevent context issues
        if len(chunks) > 30:
            logger.warning(f"Too many chunks ({len(chunks)}), limiting to 30")
            chunks = chunks[:30]
        
        # Validate chunks before adding to vector store
        valid_chunks = []
        for chunk in chunks:
            # Skip any chunks with binary or problematic data
            if is_valid_chunk(chunk.page_content):
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Skipping invalid chunk: {chunk.page_content[:30]}...")
        
        logger.info(f"Using {len(valid_chunks)} valid chunks out of {len(chunks)} total")
        
        # Process each chunk individually
        all_embeddings = []
        
        # for chunk in valid_chunks:
        #     # Process one chunk at a time 
        #     try:
        #         logger.info(f"Processing chunk: {chunk.page_content[:50]}...")
                
        #         # Use the encode method one document at a time
        #         embedding = embedding_function.embed_query(chunk.page_content)
        #         all_embeddings.append(embedding)
                
        #     except Exception as e:
        #         logger.error(f"Error encoding chunk: {e}")
        #         # Create a fallback embedding for this chunk
        #         embedding_dim = 384
        #         all_embeddings.append(np.random.randn(embedding_dim).astype(np.float32).tolist())
        
        # all_embeddings = embedding_function.embed_documents(valid_chunks)
        # Add to vector store - only valid chunks
        from generate_suggestions import generate_suggested_questions
        # Generate sample questions based on safe text
        safe_text = text[:5000] if len(text) > 5000 else text
        suggested_questions = generate_suggested_questions(safe_text)
        
        # Use valid chunks only
        if valid_chunks:
            vector_store.add_document(file_path, valid_chunks)
            logger.info(f"Indexed {len(valid_chunks)} chunks in vector store")
        else:
            logger.warning(f"No valid chunks found for {file_path}, skipping indexing")
        
        return valid_chunks, suggested_questions
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise e

# Validate a chunk before adding to vector store
def is_valid_chunk(text: str) -> bool:
    """Check if a chunk is valid for embedding (not binary or corrupted)"""
    if not text:
        return False
        
    # Check for minimum text length
    if len(text) < 5:
        return False
        
    # Check for excessive non-printable characters
    non_printable = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
    if non_printable / len(text) > 0.1:  # More than 10% non-printable
        return False
        
    # Check for very high ratio of digits or special chars (likely binary)
    digits_special = sum(1 for c in text if c.isdigit() or not c.isalnum())
    if digits_special / len(text) > 0.7:  # More than 70% digits/special
        return False
        
    # Check for known binary signatures as strings
    binary_indicators = [
        'JFIF',
        'PNG',
        'PDF',
        'ÿØÿà',
        'PK\x03\x04'
    ]
    
    for indicator in binary_indicators:
        if indicator in text:
            return False
            
    # Length check - reject extremely long chunks
    if len(text) > 10000:
        return False
        
    return True
        
# Query function
def query_index(query, top_k=5):
    global embedding_function, vector_store
    
    # Initialize if needed
    if embedding_function is None or vector_store is None:
        initialize()
    
    # Check for invalid query types
    if isinstance(query, bool) or not isinstance(query, str):
        if not isinstance(query, str):
            try:
                query = str(query)
                logger.warning(f"Converted non-string query to string: {query}")
            except:
                logger.error(f"Cannot convert {type(query)} to string")
                return []
    
    try:
        # Empty query check
        if not query.strip():
            logger.warning("Empty query received")
            return []
            
        # Search similar documents
        results = vector_store.search(query, k=top_k)
        results = rerank_chunks(query, results, top_k)
        # For LLM context window limitations, ensure total content is manageable
        max_content_length = 1000  # Reduced safe limit for most small models
        total_length = 0
        pruned_results = []
        
        for i, ((doc ,s), score) in enumerate(results):
            logger.info(f"doc_query: {doc}")
            content_length = len(doc.page_content)
            if total_length + content_length <= max_content_length:
                pruned_results.append((doc, score))
                total_length += content_length
            else:
                logger.warning(f"Skipping document with length {content_length} to avoid exceeding context window")
        
        return pruned_results
    
    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        return []

def query_index_with_context(query, top_k=5):
    global embedding_function, vector_store
    results = query_index(query, top_k)
    # context = build_focused_context(query, results)
    content = ""
    for doc, score in results:
        logger.info(f"doc_query_with_context: {doc}")
        logger.info(f"score_query_with_context: {score}")
        content += f"Document: {doc.page_content}\n"
        # content.append(doc.page_content)
    
    # Final check for context window size
    if estimate_token_count(content) > 1500:
        logger.warning(f"Context too large ({estimate_token_count(content)} est. tokens), truncating")
        content = truncate_text_for_context_window(content, 1500)
        
    return content

def build_focused_context(query, results):
    context = f"Query: {query}\n\n"
    for doc, score in results:
        context += f"Document: {doc.page_content}\n"
    return context

 



# Example usage
if __name__ == "__main__":
    pass