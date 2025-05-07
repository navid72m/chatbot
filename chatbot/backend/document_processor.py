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
        
        else:
            # Default case: try to read as text
            logger.warning(f"Unsupported file extension: {file_extension}. Attempting to read as text.")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {str(e)}")
        raise RuntimeError(f"Failed to process document: {str(e)}")

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
