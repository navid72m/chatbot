import os
import logging
from typing import List, Optional
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_document(file_path: str) -> str:
    """
    Process a document file and extract its text content.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Extracted text content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension in ['.txt', '.md', '.csv']:
            # Plain text files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_extension in ['.pdf']:
            # PDF files
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                text = "\n\n".join([page.page_content for page in pages])
                return text
            except ImportError:
                logger.warning("PyPDF2 not installed. Falling back to simple text extraction.")
                with open(file_path, 'rb') as f:
                    # Basic extraction for PDF (not optimal but works in a pinch)
                    text = f.read().decode('utf-8', errors='ignore')
                    # Clean up text by removing unusual characters and multiple spaces
                    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    return text
        
        elif file_extension in ['.docx', '.doc']:
            # Word documents
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
                return "\n\n".join([doc.page_content for doc in documents])
            except ImportError:
                logger.warning("docx2txt not installed. Using basic text extraction.")
                with open(file_path, 'rb') as f:
                    # Basic extraction (not optimal but works in a pinch)
                    text = f.read().decode('utf-8', errors='ignore')
                    # Clean up text
                    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    return text
        
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            # Image files - use PaddleOCR for high-quality text extraction
            try:
                # Import PaddleOCR
                from paddleocr import PaddleOCR
                
                logger.info(f"Processing image with PaddleOCR: {file_path}")
                
                # Initialize the PaddleOCR engine
                # use_angle_cls=True enables text orientation detection for rotated text
                # lang='en' specifies English (change as needed - supports multiple languages)
                # use_gpu=True will use GPU acceleration if available
                ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=is_gpu_available())
                
                # Perform OCR on the image
                result = ocr.ocr(file_path, cls=True)
                
                # Process the results
                extracted_text = ""
                
                # PaddleOCR returns a list of detected text regions with coordinates and text
                if result and len(result) > 0:
                    for idx, line_result in enumerate(result):
                        # Handle different output structures based on PaddleOCR version
                        if isinstance(line_result, list):
                            # Extract text from each detected area
                            for item in line_result:
                                # Each item is a tuple of (points, (text, confidence))
                                if len(item) >= 2 and isinstance(item[1], tuple) and len(item[1]) >= 1:
                                    text = item[1][0]  # The text content
                                    confidence = item[1][1]  # The confidence score
                                    
                                    # Only include text with high enough confidence
                                    if confidence > 0.6:
                                        extracted_text += text + " "
                        else:
                            # Older versions of PaddleOCR might have different output structure
                            logger.warning("Unexpected PaddleOCR result structure, attempting to extract text anyway")
                            extracted_text += str(line_result) + " "
                
                # Clean up the text
                extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
                
                logger.info(f"PaddleOCR extracted {len(extracted_text)} characters from image {file_path}")
                
                # Warning if very little text was extracted
                if len(extracted_text) < 20:
                    logger.warning(f"PaddleOCR extracted very little text from {file_path}, might be a low-quality image or no text content")
                
                return extracted_text
                
            except ImportError as e:
                logger.error(f"PaddleOCR not installed: {str(e)}. Please install paddleocr with: pip install paddleocr")
                return f"ERROR: OCR processing requires PaddleOCR. Please install with: pip install paddleocr"
                
            except Exception as e:
                logger.error(f"Error during OCR processing of {file_path}: {str(e)}")
                return f"ERROR: OCR processing failed: {str(e)}"
        
        else:
            # Default case: try to read as text
            logger.warning(f"Unsupported file extension: {file_extension}. Attempting to read as text.")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {str(e)}")
        raise RuntimeError(f"Failed to process document: {str(e)}")

def is_gpu_available():
    """Check if GPU is available for acceleration"""
    try:
        import paddle
        return paddle.device.is_compiled_with_cuda()
    except ImportError:
        # If paddle is not available, we can check for CUDA availability using torch
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # If neither paddle nor torch is available, assume no GPU
            return False

from langchain.schema import Document  # add this at the top

def chunk_document(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split a document text into smaller chunks and wrap them as langchain Document objects.
    
    Args:
        text: The document text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of Document objects
    """
    if not text:
        return []
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split document into {len(chunks)} chunks")

        return [Document(page_content=chunk, metadata={}) for chunk in chunks]
    
    except Exception as e:
        logger.error(f"Error chunking document: {str(e)}")
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(Document(page_content=chunk, metadata={}))
        logger.info(f"Fallback chunking: Split document into {len(chunks)} chunks")
        return chunks