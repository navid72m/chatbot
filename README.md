# Document Chat — Local‑First Full‑Stack RAG (Electron + FastAPI + React)

A production‑oriented **local document chat** system powered by **GGUF local LLMs** (llama.cpp), **hybrid retrieval** (Vector + BM25), optional **reranking**, and a **navigation‑aware RAG layer** for queries like:

- **“Summarize page 12”**
- **“What is Chapter 8 about?”**

This repo is designed as a complete app, not a notebook demo:
- **Frontend:** React (chat UI)
- **Desktop:** Electron (packaged app)
- **Backend:** FastAPI (document processing + RAG)
- **Retrieval:** Vector search + BM25 + RRF fusion
- **Optional:** Cross‑encoder reranking
- **Evaluation:** recall/precision/MRR/NDCG/MAP + latency

---

## 🎥 Demo

[Demo video](https://youtu.be/cIJL3SNN4R4)

---

## ✨ Key Features

### Core
- Upload and chat with documents (PDF, images, text, JSON/CSV)
- **Local‑first**: run fully offline (privacy‑friendly)
- OCR support for scanned PDFs/images (PaddleOCR)
- Document‑grounded answers with cited snippets
- Suggested questions after upload

### Retrieval & Ranking
- **Hybrid retrieval**:
  - Vector search (semantic)
  - BM25 search (lexical)
  - RRF fusion (robust & stable)
- Optional reranking (cross‑encoder)
- **Token‑budgeted context builder** (prevents context overflow)

### Navigation‑Aware RAG (Important)
Classic RAG often fails on navigation queries (“page/chapter/section”), because retrieval is semantic.

This repo includes **intent routing** that detects such queries and fetches content **directly by page/chapter**, bypassing retrieval when appropriate.

---

## 🧠 Architecture

### RAG Pipeline Diagram

> Save your diagram image as: `assets/rag_pipeline.png`

![Enhanced RAG Pipeline](assets/rag_pipeline.png)

---

## 🔎 How It Works (Short)

### Offline indexing
1. Extract text (OCR if needed)
2. Split into chunks + metadata (page, offsets)
3. Compute embeddings
4. Build indexes:
   - Vector index (FAISS)
   - BM25 index
   - Metadata store

### Online query
1. **Intent detection**
2. Route query:
   - **Semantic QA** → Hybrid Retrieval → (Optional rerank)
   - **Page/Chapter** → Direct chunk fetch (skip rerank)
3. Build context within token budget
4. Generate grounded answer with LLM

---

## 🧩 Repository Structure (Typical)

```text
.
├── backend/                 # FastAPI backend
│   ├── app_integration_updated.py
│   ├── document_processor.py
│   ├── models/              # GGUF models
│   └── ...
├── frontend/                # React UI
├── electron/                # Electron packaging
├── assets/
│   └── rag_pipeline.png     # README diagram
└── fast_rag_evaluation.py   # eval runner
```

---

## 🧰 Tech Stack

- **Backend:** FastAPI, Python
- **LLM:** llama.cpp (GGUF)
- **Embeddings:** sentence-transformers
- **Vector Index:** FAISS
- **Lexical Search:** BM25
- **Reranking (optional):** cross‑encoder / BGE reranker
- **Frontend:** React
- **Desktop:** Electron

---

## 🚀 Quickstart

### 1) Clone
```bash
git clone https://github.com/navid72m/pdf.git
cd pdf
```

### 2) Backend setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If you use OCR:
```bash
pip install paddleocr
```

### 3) Add a GGUF model
Put your model into:
```bash
backend/models/
```

Example:
```text
backend/models/deepseek-r1.Q4_K_M.gguf
```

### 4) Start backend
```bash
python app_integration_updated.py --host 127.0.0.1 --port 8000
```

### 5) Start frontend
```bash
cd ../frontend
npm install
npm run dev
```

(Optional) Electron desktop:
```bash
cd ../electron
npm install
npm start
```

---

## ⚙️ Configuration

Common env vars:

```bash
export LLAMA_CPP_MODEL_PATH="./backend/models/deepseek-r1.Q4_K_M.gguf"
export LLAMA_CTX_SIZE=4096
export LLAMA_THREADS=8
```

---

## 🔌 Backend API

### Upload document
```http
POST /upload
Content-Type: multipart/form-data
```

### Query document
```http
POST /query
Content-Type: application/json

{
  "query": "Summarize page 12",
  "document": "myfile.pdf"
}
```

### Suggested questions
```http
GET /suggestions?document=myfile.pdf
```

---

## 🧪 Evaluation

This repo includes evaluation scripts to compare:

- baseline hybrid retrieval
- enhanced navigation‑aware retrieval

Metrics:
- Recall@K / Precision@K
- MRR / NDCG / MAP
- Latency (mean / p95 / p99)

Run:
```bash
python fast_rag_evaluation.py
```

---

## 🛠️ Troubleshooting

### Context window exceeded
Solutions:
- cap context tokens (e.g., 2500–5000)
- rerank only top 10 candidates
- reduce neighbor expansion
- shorten chunk size

### Enhanced pipeline too slow
- navigation queries should **skip reranking**
- cache embeddings + retrieval results
- rerank only top N (10)

---

## 🗺️ Roadmap

- [ ] Better chapter detection via TOC parsing
- [ ] Hard negative dataset generation for eval
- [ ] Multi‑document workspace
- [ ] Streaming UI improvements
- [ ] Benchmark embedding/reranker choices

---

## 🤝 Contributing

PRs welcome.  
If you change retrieval logic, please attach:
- eval results
- latency impact

---

## 📄 License

MIT
