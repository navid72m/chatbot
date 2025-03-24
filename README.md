# Local LLM Chatbot with Advanced RAG Capabilities

This project is a **fully local document chatbot** powered by Large Language Models (LLMs). It allows users to upload PDFs and ask questions — all without sending data to external servers.

---

## ✨ Features

- **Local LLM Inference**: Runs quantized `.gguf` models (e.g. Mistral) using `llama.cpp`
- **PDF Upload & Chunking**: Parses PDFs into context-aware chunks for retrieval
- **Vector Search**: Embeds chunks using SentenceTransformers & retrieves top-K relevant passages
- **Knowledge Graph Integration**: Uses Neo4j to capture entities and relationships between concepts
- **Chain-of-Thought Reasoning**: LLM breaks down steps before answering
- **Multi-hop Reasoning**: Handles complex queries by chaining sub-questions
- **Answer Verification**: Validates responses against retrieved sources

---

## 🧠 Architecture

```
User ─┬─▶ Upload PDF ─────┬────▶ Document Processor
     │                   │
     │                   └────▶ Chunk & Embed
     │                           │
     │                 ┌────────┴──────┐
     │                 ▼               ▼
     │         Vector Store         Knowledge Graph
     │             │                     │
     │             └────┬────┬──────────┘
     │                  ▼    ▼
     │           Hybrid Retriever
     │                  │
     │          Chain-of-Thought
     │                  │
     │             Final Answer
     ▼
   Chat UI (Electron)
```

---

## 🗂️ Key Components

| File | Purpose |
|------|---------|
| `app_integration.py` | FastAPI server wiring all components |
| `llm_interface.py` | Loads GGUF model using llama.cpp via `ctypes` |
| `vector_store.py` | Chroma vector DB wrapper with embedding logic |
| `knowledge_graph.py` | spaCy + Neo4j entity & relation extractor |
| `chain_of_thought.py` | Implements CoT + reasoning parsing |
| `document_processor.py` | PDF chunking, cleaning, and metadata attachment |
| `frontend/` | Electron + Vite React app to interact with backend |

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.10+
- Node.js 18+
- Neo4j installed locally
- `llama.cpp` compiled as shared library (`libllama.dylib`)
- A `.gguf` model like `mistral-7b-instruct-v0.1.Q4_K_M.gguf`

### 🛠️ Setup (Backend)

```bash
# Clone the repo
git clone https://github.com/yourname/local-llm-chatbot.git
cd local-llm-chatbot/backend

# Install Python deps
pip install -r requirements.txt

# Set Neo4j credentials
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password

# Start backend
python app_integration.py
```

### 🖥️ Setup (Frontend)

```bash
cd ../frontend
npm install
npm run dev  # or: npm run build && npm start
```

---

## 💬 API Quick Reference

| Endpoint | Description |
|----------|-------------|
| `POST /upload` | Upload and process a PDF |
| `POST /query` | Ask a question using hybrid RAG |
| `GET /models` | List available local models |
| `GET /advanced-rag/features` | View enabled features |

---

## 🧪 Example Usage

### Python
```python
from advanced_rag import AdvancedRAG

rag = AdvancedRAG(...)
rag.add_documents([doc1, doc2])
result = rag.answer_query("What is the main finding?")
print(result["answer"])
```

### Curl
```bash
curl -X POST -F "file=@report.pdf" http://localhost:8000/upload

curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "What is the report summary?"}' \
     http://localhost:8000/query
```

---

## 🛠️ Configuration Flags

| Flag | Description | Default |
|------|-------------|---------|
| `use_cot` | Enable step-by-step reasoning | `True` |
| `use_kg` | Enable Neo4j knowledge graph | `True` |
| `use_multihop` | Break down complex queries | `True` |
| `verify_answers` | Verify answer against sources | `True` |
| `temperature` | Model creativity level | `0.7` |

---

## 📈 Performance Tips

- Run models with 4-bit quantization (`Q4_K_M`) for faster inference
- Use ChromaDB with persistence for large corpora
- Enable `llama.cpp` Metal backend for M1/M2 Macs
- Pre-index documents for better first-response latency

---

## 🚧 Limitations

- No GPU acceleration (yet)
- spaCy extraction may miss domain-specific terms
- CoT & multihop increase latency
- Basic Electron UI (for now)



---

## 🪪 License

MIT License

