# hybrid_retriever.py - Implementation of hybrid retrieval strategies
import logging
from typing import List, Dict, Optional, Any, Union, Tuple
import time
import numpy as np
from langchain.docstore.document import Document

from vector_store import VectorStore
from knowledge_graph import KnowledgeGraph
from llm_interface import stream_ollama_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retrieval system that combines vector search, knowledge graph,
    and query rewriting to improve document retrieval for RAG systems.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        model: str = "mistral",
        temperature: float = 0.7,
        vector_weight: float = 0.7,
        kg_weight: float = 0.3,
        max_vector_results: int = 5,
        max_kg_results: int = 3,
        use_query_expansion: bool = True
    ):
        """Initialize the hybrid retriever"""
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.model = model
        self.temperature = temperature
        
        # Weighting for result combination
        self.vector_weight = vector_weight
        self.kg_weight = kg_weight
        
        # Retrieval limits
        self.max_vector_results = max_vector_results
        self.max_kg_results = max_kg_results
        
        # Feature flags
        self.use_query_expansion = use_query_expansion
        
        logger.info("Hybrid retriever initialized")
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Main retrieval method combining multiple strategies
        
        Args:
            query: The user's query
            
        Returns:
            List of retrieved Documents
        """
        start_time = time.time()
        logger.info(f"Retrieving documents for query: {query}")
        
        # Step 1: Expand/rewrite query if enabled
        expanded_queries = [query]
        if self.use_query_expansion:
            expanded_queries.extend(self._expand_query(query))
            logger.info(f"Expanded to {len(expanded_queries)} queries")
        
        # Step 2: Vector retrieval for each query variation
        all_vector_results = []
        for q in expanded_queries:
            results = self.vector_store.search(q, k=self.max_vector_results)
            all_vector_results.extend(results)
        
        # Deduplicate vector results
        vector_results = self._deduplicate_documents(all_vector_results)
        logger.info(f"Retrieved {len(vector_results)} unique documents from vector store")
        
        # Step 3: Knowledge graph retrieval
        kg_results = self._retrieve_from_kg(query)
        logger.info(f"Retrieved {len(kg_results)} documents from knowledge graph")
        
        # Step 4: Rerank and combine results
        combined_results = self._rerank_and_combine(query, vector_results, kg_results)
        
        elapsed = time.time() - start_time
        logger.info(f"Hybrid retrieval completed in {elapsed:.2f} seconds, returning {len(combined_results)} documents")
        
        return combined_results
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand the query with variations to improve retrieval
        
        Args:
            query: Original query string
            
        Returns:
            List of expanded query strings
        """
        expansion_prompt = f"""
Given this search query: "{query}"

Please generate 2-3 alternative versions of this query that:
1. Use different phrasing or synonyms
2. Make implicit information explicit
3. Add relevant context that might help find better results

Format your response as a list, one query per line.
"""

        try:
            # Get expansion from LLM
            response = stream_ollama_response(
                query=expansion_prompt,
                context="",
                model=self.model,
                temperature=0.7
            )
            
            # Parse response into list of queries
            expansions = []
            for line in response.split('\n'):
                # Remove any numbering, bullets, quotes
                clean_line = line.strip().lstrip('0123456789.-*>• "\'').rstrip('"\'').strip()
                if clean_line and len(clean_line) > 5 and clean_line != query:
                    expansions.append(clean_line)
            
            # Limit number of expansions
            return expansions[:3]  # Max 3 expansions
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return []  # Return empty list on failure
    
    def _retrieve_from_kg(self, query: str) -> List[Document]:
        """
        Retrieve documents using knowledge graph
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved Documents
        """
        # Extract entities from query
        entities = self.knowledge_graph.extract_entities(query)
        
        # Return empty if no entities found
        if not entities:
            return []
        
        # Get document IDs from knowledge graph
        all_doc_ids = set()
        for entity in entities:
            # Get documents mentioning this entity
            doc_ids = self.knowledge_graph.query_entity_documents(entity["text"])
            all_doc_ids.update(doc_ids)
            
            # Also get related entities and their documents
            related = self.knowledge_graph.query_related_entities(entity["text"], max_hops=2)
            for rel in related[:self.max_kg_results]:  # Limit to top related entities
                rel_docs = self.knowledge_graph.query_entity_documents(rel["name"])
                all_doc_ids.update(rel_docs)
        
        # Get the actual documents from vector store by ID
        # This is a simplified implementation - in practice you'd need to map document IDs to actual documents
        kg_documents = []
        for doc_id in all_doc_ids:
            # Create a query to find this document ID
            doc_query = f"source:{doc_id}"
            results = self.vector_store.search(doc_query, k=1)
            kg_documents.extend(results)
        
        # Add documents from entity relationships
        if len(entities) >= 2:
            # Look for relationships between entities
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    entity1 = entities[i]["text"]
                    entity2 = entities[j]["text"]
                    
                    relations = self.knowledge_graph.query_relationship_context(entity1, entity2)
                    
                    if relations:
                        # Create a synthetic document with relationship information
                        rel_text = f"Relationship between {entity1} and {entity2}:\n\n"
                        for rel in relations:
                            rel_text += f"- {rel['predicate']}\n"
                            rel_text += f"  Context: {rel['context']}\n\n"
                        
                        rel_doc = Document(
                            page_content=rel_text,
                            metadata={
                                "source": "knowledge_graph_relation",
                                "entity1": entity1,
                                "entity2": entity2
                            }
                        )
                        kg_documents.append(rel_doc)
        
        return kg_documents
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity"""
        if not documents:
            return []
        
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # Create a simplified representation of content for comparison
            # This is a simple approach - more sophisticated deduplication could be used
            simplified = doc.page_content[:100].lower()
            
            if simplified not in seen_content:
                seen_content.add(simplified)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _rerank_and_combine(
        self,
        query: str,
        vector_results: List[Document],
        kg_results: List[Document]
    ) -> List[Document]:
        """
        Rerank and combine results from different retrieval methods
        
        Args:
            query: Original query
            vector_results: Results from vector search
            kg_results: Results from knowledge graph
            
        Returns:
            Combined and reranked document list
        """
        # Simple reranking strategy: 
        # 1. Score documents by similarity to query (vector store already did this)
        # 2. Boost documents that are both in vector and KG results
        # 3. Interleave results based on the vector_weight and kg_weight
        
        # First, deduplicate across both sets
        all_docs = self._deduplicate_documents(vector_results + kg_results)
        
        # Create a set of document IDs from knowledge graph results for quick lookup
        kg_doc_ids = set()
        for doc in kg_results:
            doc_id = f"{doc.metadata.get('source', '')}-{doc.metadata.get('chunk_id', '')}"
            kg_doc_ids.add(doc_id)
        
        # Score and sort documents
        scored_docs = []
        for i, doc in enumerate(all_docs):
            # Base score: position in vector results (if present)
            vector_score = 0
            if doc in vector_results:
                vector_score = 1.0 - (vector_results.index(doc) / max(1, len(vector_results)))
            
            # Knowledge graph presence score
            doc_id = f"{doc.metadata.get('source', '')}-{doc.metadata.get('chunk_id', '')}"
            kg_score = 1.0 if doc_id in kg_doc_ids else 0.0
            
            # Combined score
            combined_score = (self.vector_weight * vector_score) + (self.kg_weight * kg_score)
            
            scored_docs.append((doc, combined_score))
        
        # Sort by combined score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return the documents, discarding scores
        return [doc for doc, _ in scored_docs]
    
    def retrieve_with_explanation(self, query: str) -> Dict[str, Any]:
        """
        Retrieve documents with explanation of why each was selected
        
        Args:
            query: The user's query
            
        Returns:
            Dict containing documents and explanations
        """
        # Get documents using standard retrieval
        documents = self.retrieve(query)
        
        if not documents:
            return {
                "documents": [],
                "explanations": [],
                "query": query
            }
        
        # Generate explanations for why each document was retrieved
        explanations = []
        
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            
            # Different explanation based on source type
            if "knowledge_graph" in source:
                # This is from the knowledge graph
                entity1 = doc.metadata.get("entity1", "")
                entity2 = doc.metadata.get("entity2", "")
                if entity1 and entity2:
                    explanation = f"Retrieved because it shows the relationship between {entity1} and {entity2}, which are relevant to your query."
                else:
                    explanation = "Retrieved from the knowledge graph based on entity relationships."
            else:
                # This is from vector search
                explanation = f"Retrieved because its content is semantically similar to your query."
            
            explanations.append(explanation)
        
        return {
            "documents": documents,
            "explanations": explanations,
            "query": query
        }


# Test the class if run directly
if __name__ == "__main__":
    print("This module implements hybrid retrieval for RAG systems.")
    print("Import and use this class within your RAG application.")