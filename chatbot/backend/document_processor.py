# document_processor.py - Document processing and chunking
import os
from typing import List, Dict
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_document(file_path: str) -> str:
    """Extract text from a document file"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return process_image(file_path)
    elif file_extension == '.pdf':
        return process_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

def process_image(image_path: str) -> str:
    """Extract text from an image using OCR"""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def process_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file"""
    try:
        # First try PyPDF2
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # If we got meaningful text, return it
        if text.strip() and len(text) > 100:
            return text
        
        # If not enough text extracted, try OCR
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image) + "\n\n"
        return text
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        # Fall back to OCR only
        try:
            images = convert_from_path(pdf_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n\n"
            return text
        except Exception as inner_e:
            raise Exception(f"Failed to process PDF: {str(inner_e)}")

def chunk_document(text: str) -> List[Dict]:
    """Split document into chunks for vector storage"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.create_documents([text])
    return chunks 