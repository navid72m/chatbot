# vector_store.py - Vector storage using FAISS

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from typing import List, Optional
import os
import pickle

class VectorStore:
    def __init__(self, persist_path: str = "faiss_store"):
        self.persist_path = persist_path
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Load existing FAISS index if exists
        if os.path.exists(os.path.join(persist_path, "index.faiss")):
            self.db = FAISS.load_local(persist_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.db = None

    def add_document(self, document_name: str, chunks: List[Document]) -> None:
        """Add a document's chunks to the FAISS vector store"""
        for i, chunk in enumerate(chunks):
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["source"] = document_name
            chunk.metadata["chunk_id"] = i

        if self.db is None:
            self.db = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.db.add_documents(chunks)

        self.db.save_local(self.persist_path)

    def search(self, query: str, k: int = 5, filter: Optional[dict] = None) -> List[Document]:
        """Search for similar chunks to the query, optionally applying a metadata filter"""
        if self.db is None:
            return []

        if filter:
            return [doc for doc, _ in self.db.similarity_search_with_score(query, k=k, filter=filter)]
        return self.db.similarity_search(query, k=k)

