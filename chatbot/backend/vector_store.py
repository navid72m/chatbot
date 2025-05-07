# vector_store.py - Vector storage using FAISS

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List
import logging
import os
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Dict, Any, Optional

from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import shutil


logger = logging.getLogger(__name__)

class BM25Store:
    def __init__(self, index_dir="bm25_index"):
        self.index_dir = os.path.expanduser(f"~/Library/Application Support/Document Chat/{index_dir}")
        self.schema = Schema(content=TEXT(stored=True), doc_id=ID(stored=True), source=ID(stored=True))
        if os.path.exists(self.index_dir):
            self.ix = index.open_dir(self.index_dir)
        else:
            os.makedirs(self.index_dir, exist_ok=True)
            self.ix = index.create_in(self.index_dir, self.schema)
        


    def add_documents(self, chunks: List[Document]):
        writer = self.ix.writer()
        for chunk in chunks:
            writer.add_document(
                content=chunk.page_content,
                doc_id=str(chunk.metadata.get("chunk_id", 0)),
                source=str(chunk.metadata.get("source", "unknown"))
            )
        writer.commit()
        self.bm25.add_documents(chunks)

    def search(self, query: str, k=5, filter: dict = None) -> List[Document]:
        with self.ix.searcher() as searcher:
            parser = QueryParser("content", self.ix.schema)
            q = parser.parse(query)
            results = searcher.search(q, limit=k)
            return [
                Document(page_content=hit["content"], metadata={"source": hit["source"], "chunk_id": hit["doc_id"]})
                for hit in results
                if not filter or hit["source"] == filter.get("source")
            ]


class LocalEmbeddingFunction(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raw = self.model.encode(texts, convert_to_numpy=True)
        normed = self._normalize(raw)
        return normed.tolist()

    def embed_query(self, text: str) -> List[float]:
        raw = self.model.encode([text], convert_to_numpy=True)[0]
        normed = raw / np.linalg.norm(raw)
        return normed.tolist()

class VectorStore:
    def __init__(self):
        self.embeddings = LocalEmbeddingFunction()
        self.vector_store = None
        self.persist_directory = os.path.expanduser("~/Library/Application Support/Document Chat/faiss_db")
        os.makedirs(self.persist_directory, exist_ok=True)
        self.index_path = os.path.join(self.persist_directory, "index")
        self._initialize_vector_store()
        self.bm25 = BM25Store()
    def _initialize_vector_store(self):
        try:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    folder_path=self.persist_directory,
                    index_name="index",
                    embeddings=self.embeddings
                )
                logger.info("FAISS vector store loaded successfully")
            else:
                self.vector_store = FAISS.from_documents(
                    documents=[Document(page_content="init", metadata={"source": "init"})],
                    embedding=self.embeddings
                )
                self.vector_store.save_local(folder_path=self.persist_directory, index_name="index")
                logger.info("New FAISS vector store created")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise RuntimeError(f"Failed to initialize vector store: {e}")

    def add_document(self, document_name: str, chunks: List[Document]) -> None:
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        for i, chunk in enumerate(chunks):
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["source"] = document_name
            chunk.metadata["chunk_id"] = i

        try:
            self.vector_store.add_documents(chunks)
            self.vector_store.save_local(folder_path=self.persist_directory, index_name="index")
            logger.info(f"Added document {document_name} with {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error adding document {document_name}: {e}")
            raise
    def similarity_search_with_debug(self, query: str, k: int = 5):
        results = self.vector_store.similarity_search_with_score(query, k)
        for doc, score in results:
            print(f"[{doc.metadata.get('source', 'Unknown')}] Cosine Similarity: {score:.4f}")
        return results


    def search(self, query: str, k: int = 5, filter: Optional[dict] = None) -> List[Document]:
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")

        try:
            results = self.vector_store.similarity_search_with_score(query, k)
            logger.info(f"results: {results}")
            
            filtered_results = []
            
            for item in results:
                # Handle both tuple (doc, score) and direct document objects
                logger.info(f"doc_item {item}")
                if isinstance(item, tuple) :
                    doc, score = item
                else:
                    # If the item is not a tuple, assume it's already a Document
                    doc = item
                    score = 0.0  # Default score
                    
                # logger.info(f"vector_store:doc: page_content='{doc.page_content}' metadata={doc.metadata}")
                
                # Apply filtering if needed
                if filter and doc.metadata:
                    if all(doc.metadata.get(k1) == v1 for k1, v1 in filter.items()):
                        filtered_results.append((doc, score))
                else:
                    filtered_results.append((doc, score))
                    
                # Stop if we've reached the desired number of results
                if len(filtered_results) >= k:
                    break
                    
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    def hybrid_search(self, query: str, k: int = 5, filter: dict = None):
        vector_results = self.search(query, k=k, filter=filter)
        bm25_results = self.bm25.search(query, k=k, filter=filter)

        # Combine and deduplicate
        combined = {}
        
        # Process vector results
        for doc_tuple in vector_results:
            doc, score = doc_tuple
            if hasattr(doc, 'metadata') and doc.metadata:  # Check if doc has metadata
                key = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', '0')}"
                combined[key] = doc
            else:
                # Handle documents without metadata
                logger.info(f"doc: {doc}")
                if hasattr(doc, 'page_content'):
                    combined[f"no_metadata_{hash(doc.page_content)}"] = doc
                else:
                    combined[f"no_metadata_{hash(doc)}"] = doc

        # Process BM25 results
        for doc in bm25_results:
            if hasattr(doc, 'metadata') and doc.metadata:  # Check if doc has metadata
                key = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', '0')}"
                if key not in combined:
                    combined[key] = doc
            else:
                # Handle documents without metadata
                if hasattr(doc, 'page_content'):
                    key = f"no_metadata_{hash(doc.page_content)}"
                else:
                    key = f"no_metadata_{hash(doc)}"
                if key not in combined:
                    combined[key] = doc

        return list(combined.values())[:k]

def build_focused_context(query: str, relevant_chunks: List[Document], max_tokens: int = 2048) -> str:
    context_parts = []
    total_tokens = 0

    for i, chunk_data in enumerate(relevant_chunks):
        # Handle both Document objects and (Document, score) tuples
        chunk = chunk_data[0] if isinstance(chunk_data, tuple) else chunk_data
        source = chunk.metadata.get("source", "Unknown")
        chunk_id = chunk.metadata.get("chunk_id", i)
        text = chunk.page_content.strip()

        chunk_tokens = max(1, len(text) // 4)

        if total_tokens + chunk_tokens > max_tokens:
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 50:
                truncated_text = text[:remaining_tokens * 4] + "..."
                formatted_chunk = f"[DOCUMENT: {source}]\n[SECTION: {chunk_id}]\n{truncated_text}\n"
                context_parts.append(formatted_chunk)
                total_tokens = max_tokens
            break

        formatted_chunk = f"[DOCUMENT: {source}]\n[SECTION: {chunk_id}]\n{text}\n"
        context_parts.append(formatted_chunk)
        total_tokens += chunk_tokens
    context_parts = context_parts[-1]

    return context_parts