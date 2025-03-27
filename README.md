# 🧠 Local LLM Desktop Chatbot with Advanced RAG & Ollama Integration

This project is a **fully local desktop application** that enables secure, offline document-based chat using Large Language Models (LLMs). It uses **Ollama** to run quantized models (like `deepseek-coder` or `mistral`) and supports advanced Retrieval-Augmented Generation (RAG) features such as:

- 🧾 Multi-format document upload
- 🔍 Vector + knowledge graph-based hybrid search
- 🔄 Chain-of-thought reasoning
- ✅ Answer verification against retrieved context

---

## ✨ Key Features

- **🔐 100% Local Inference** via [Ollama](https://ollama.com)
- **📄 Multi-format Upload**: PDF, DOCX, TXT, CSV, XLSX
- **📚 Hybrid RAG**: Combines vector search (ChromaDB) + Neo4j knowledge graph
- **🧠 CoT & Multihop Reasoning**: Breaks down complex queries
- **📏 Answer Verification**: Validates LLM output against sources
- **🧑‍💻 Electron App**: Cross-platform frontend (macOS, Windows)

---

## 🧱 Architecture Diagram

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
     │       ┌──────────┴────────────┐
     │       ▼                       ▼
     │   Answer Verifier        Chain-of-Thought
     │       │                       │
     │       └──────────┬────────────┘
     │                  ▼
     │          Local LLM via Ollama
     ▼
   Electron Desktop App
```

---

## 🗂️ Key Components

| File / Dir | Description |
|------------|-------------|
| `backend/` | FastAPI + Python backend with RAG logic |
| `electron/` | Electron main process, starts backend & Ollama |
| `frontend/` | Vite + React UI for chatting, uploading |
| `ollama/` | Local LLM runner and model container |
| `Modelfile` | Ollama configuration file (base model) |

---

## 🧪 Tech Stack

- 🧠 **LLM Inference**: [Ollama](https://ollama.com) with quantized `.gguf` models
- 🧾 **Document Parsing**: PyMuPDF, Pandas
- 💡 **Embeddings**: `sentence-transformers`
- 🔍 **Vector DB**: ChromaDB
- 🧬 **Knowledge Graph**: spaCy + Neo4j
- 🧩 **Frontend**: React + Vite (Electron)
- 🐍 **Backend**: FastAPI with modular RAG pipeline

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.10+
- Node.js 18+
- Neo4j running locally
- Ollama installed (or embedded binary in `electron/bin/ollama`)

### 🛠️ Backend Setup

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password
python app.py  # Or use compiled backend binary
```

### 🖥️ Frontend & Electron App

```bash
cd frontend
npm install
npm run build  # Build static frontend assets

# In project root
npm run electron:dev  # For dev
npm run electron:build  # For packaged macOS/Windows app
```

---

## 📦 Packaging for Distribution

Ensure `electron/bin/backend` and `electron/bin/ollama` exist and are executable.

```bash
npm run electron:build  # Generates DMG/EXE in /release/
```

---

## 🧪 API Reference

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Check server health |
| `POST /upload` | Upload and embed document |
| `POST /query` | Ask a question to LLM |
| `POST /configure` | Change reasoning settings |
| `GET /models` | List supported local models |

---

## ⚙️ Configuration Flags

| Flag | Description | Default |
|------|-------------|---------|
| `use_kg` | Enable Neo4j knowledge graph | `True` |
| `use_cot` | Enable chain-of-thought | `True` |
| `use_multihop` | Multi-hop retrieval | `True` |
| `verify_answers` | Answer validation | `True` |
| `temperature` | Model creativity | `0.7` |
| `context_window` | How many chunks to pass to model | `10` |

---

## 🛠️ Troubleshooting

- **App launches but shows blank screen**: Check index.html paths in `vite.config.js`
- **Backend not connecting**: Ensure the health check route `/health` returns `200`
- **Port conflicts**: Make sure port 8000 isn’t already in use
- **Error: spawn backend ENOENT**: Ensure binary exists and is executable

---

## 📈 Performance Tips

- Use 4-bit quantized models like `mistral.Q4_K_M`
- Pre-warm documents into ChromaDB
- Set `Ollama` to use Metal backend (for macOS M1/M2)
- Reduce `context_window` to speed up answers

---

## 📜 License

MIT License — All documents stay local.

---

Let me know if you'd like me to embed actual screenshots, diagnostic logs, or contribution guidelines.