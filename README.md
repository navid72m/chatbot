# Document Chat

A powerful document analysis and chat application with advanced OCR and RAG (Retrieval-Augmented Generation) capabilities for natural language interaction with your documents.



## Demo

[![Watch the demo](assets/demo_thumbnail_with_play.png)](https://youtu.be/cIJL3SNN4R4)

*Click on the image above to watch the demo video*







## Features

- **Universal Document Support**: Process PDFs, Word documents, text files, CSV, and images (JPG, PNG, etc.)
- **Advanced OCR**: Extract text from images and scanned documents using PaddleOCR
- **Vector Search**: Find relevant document sections based on semantic meaning, not just keywords
- **Conversational Interface**: Ask questions about your documents in natural language
- **Multi-RAG Support**: Choose between different RAG implementations:
  - Default RAG for basic document retrieval
  - LlamaIndex RAG for advanced context-aware responses
- **Local Processing**: All processing happens on your device - no data is sent to external servers
- **Customizable Models**: Select from different language models and performance settings

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- GPU support is optional but recommended for faster processing

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-chat.git
   cd document-chat
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install OCR dependencies (optional but recommended):
   ```bash
   # For OCR support
   pip install paddleocr paddlepaddle pillow
   
   # For GPU acceleration (if you have a CUDA-compatible GPU)
   pip install paddlepaddle-gpu
   ```

## Configuration

Document Chat can be configured through the `config.json` file or through the application interface.

### Model Settings

- **Model**: Choose from available language models
- **Quantization**: Select quantization level (None, 8-bit, 4-bit, 1-bit)
- **Context Window**: Number of document chunks to include in context (higher = more comprehensive but slower)
- **Temperature**: Control response creativity (lower = more factual, higher = more creative)

### OCR Settings

- **Language**: Primary language for OCR (default: 'en')
- **Use Angle Detection**: Enable text orientation detection (default: True)
- **GPU Acceleration**: Enable GPU support for OCR if available

## Usage

### Starting the Application

1. Start the backend server:
   ```bash
   cd chatbot/backend
   python app_integration_updated.py
   ```

2. Start the frontend (in a new terminal):
   ```bash
   cd chatbot/frontend
   npm install  # First time only
   npm start
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

### Working with Documents

1. **Upload Documents**:
   - Click the upload area or drag-and-drop a document
   - Supported formats: PDF, DOCX, TXT, CSV, JPG, PNG, etc.

2. **Ask Questions**:
   - Type your question in the chat input
   - The system will search the document and provide relevant answers

3. **Advanced Options**:
   - Adjust model settings for different response types
   - Choose RAG method for different retrieval strategies
   - Evaluate RAG performance on your documents

## OCR Capabilities

Document Chat uses PaddleOCR for extracting text from images with these features:

- Multi-language support
- Text orientation detection
- High-accuracy recognition
- GPU acceleration when available

### Supported Image Formats

- JPEG/JPG
- PNG
- BMP
- TIFF
- WebP

## Troubleshooting

### Vector Database Issues

If you experience issues with the vector database, you may need to reset it:

```bash
# Delete the vector database directory
rm -rf ~/Library/Application\ Support/Document\ Chat/
```

Or use the provided script:

```bash
python scripts/delete_vector_db.py
```

### OCR Problems

If OCR is not working or producing poor results:

1. Ensure PaddleOCR is installed:
   ```bash
   pip install paddleocr paddlepaddle
   ```

2. For better performance, install with GPU support:
   ```bash
   pip install paddlepaddle-gpu
   ```

3. For image-heavy workflows, ensure images are:
   - High resolution (300+ DPI recommended)
   - Well-lit with good contrast
   - Properly oriented

### Memory Issues

If you encounter memory errors:

1. Reduce the context window size in settings
2. Use a more efficient quantization (4-bit recommended)
3. Process smaller documents or split large ones
4. Increase system swap space

## API Reference

The backend exposes these main endpoints:

- `POST /upload`: Upload a document
- `POST /query-sync`: Query the document
- `GET /suggestions`: Get suggested questions
- `POST /set_document`: Set the current document
- `POST /evaluate/basic`: Evaluate RAG performance
- `GET /models`: List available models
- `GET /documents`: List uploaded documents
- `POST /reset-database`: Reset the vector database

Complete API documentation is available in the `docs/api.md` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain) for document processing
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for OCR capabilities
- [LlamaIndex](https://github.com/jerryjliu/llama_index) for advanced RAG
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for embeddings
