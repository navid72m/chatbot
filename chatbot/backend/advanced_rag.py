# advanced_rag.py - Main implementation of Advanced RAG system with hybrid retrieval
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import time
from langchain.docstore.document import Document

# Import local modules
from knowledge_graph import KnowledgeGraph
from chain_of_thought import ChainOfThoughtReasoner
from vector_store import VectorStore
from llm_interface import stream_ollama_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRAG:
    """
    Advanced RAG system that integrates knowledge graphs, chain-of-thought reasoning,
    and hybrid retrieval to enhance accuracy and reduce hallucination.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        reasoner: Optional[ChainOfThoughtReasoner] = None,
        model: str = "mistral",
        temperature: float = 0.7,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password"
    ):
        """Initialize the Advanced RAG system"""
        # Initialize components if not provided
        self.vector_store = vector_store or VectorStore()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        self.reasoner = reasoner or ChainOfThoughtReasoner(
            model=model,
            temperature=temperature
        )
        
        self.model = model
        self.temperature = temperature
        
        # Configuration parameters
        self.use_cot = False
        self.use_kg = False
        self.verify_answers = True
        self.use_multihop = False
        self.max_hops = 2
        self.max_kg_results = 3
        self.max_vector_results = 5
        
        logger.info("Advanced RAG system initialized")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Process and add documents to both vector store and knowledge graph
        
        Args:
            documents: List of Document objects to process
        """
        start_time = time.time()
        
        # Add to vector store
        logger.info(f"Adding {len(documents)} documents to vector store")
        for doc in documents:
            self.vector_store.add_document(doc.metadata.get("source", "unknown"), [doc])
        
        # Add to knowledge graph if enabled
        if self.use_kg:
            logger.info(f"Adding {len(documents)} documents to knowledge graph")
            self.knowledge_graph.process_documents(documents)
        
        elapsed = time.time() - start_time
        logger.info(f"Document processing completed in {elapsed:.2f} seconds")
    
    def hybrid_retrieval(self, query: str) -> List[Document]:
        """
        Perform hybrid retrieval using both vector store and knowledge graph
        
        Args:
            query: The user's query
            
        Returns:
            List of retrieved Document objects
        """
        # Get vector search results
        vector_results = self.vector_store.search(query, k=self.max_vector_results)
        
        # If knowledge graph is disabled, return only vector results
        if not self.use_kg:
            return vector_results
        
        # Extract potential entities from query
        query_entities = self._extract_query_entities(query)
        
        # Get knowledge graph results if entities found
        kg_documents = []
        if query_entities:
            for entity in query_entities:
                # Get related entities
                related = self.knowledge_graph.query_related_entities(
                    entity["text"], 
                    max_hops=self.max_hops
                )
                
                # If related entities found, get documents mentioning them
                if related:
                    for rel_entity in related[:self.max_kg_results]:
                        doc_ids = self.knowledge_graph.query_entity_documents(rel_entity["name"])
                        
                        # Find these documents in our vector store
                        for doc_id in doc_ids:
                            # Get document from vector DB (simplified, real implementation would need to retrieve by ID)
                            docs = self.vector_store.search(f"document:{doc_id}", k=1)
                            kg_documents.extend(docs)
        
        # Combine and deduplicate results
        all_documents = self._combine_and_deduplicate(vector_results, kg_documents)
        
        # Get path-based context for entities if available
        if len(query_entities) >= 2:
            # Try to find paths between entities mentioned in the query
            for i in range(len(query_entities)):
                for j in range(i+1, len(query_entities)):
                    entity1 = query_entities[i]["text"]
                    entity2 = query_entities[j]["text"]
                    
                    paths = self.knowledge_graph.find_path_between_entities(
                        entity1, entity2, max_hops=self.max_hops
                    )
                    
                    # Add path context as synthetic documents
                    if paths:
                        path_text = self._format_path_as_text(paths, entity1, entity2)
                        path_doc = Document(
                            page_content=path_text,
                            metadata={
                                "source": "knowledge_graph_path",
                                "entity1": entity1,
                                "entity2": entity2
                            }
                        )
                        all_documents.append(path_doc)
        
        return all_documents
    
    def _extract_query_entities(self, query: str) -> List[Dict]:
        """Extract entities from the query text"""
        return self.knowledge_graph.extract_entities(query)
    
    def _combine_and_deduplicate(self, vector_docs: List[Document], kg_docs: List[Document]) -> List[Document]:
        """Combine and deduplicate document results"""
        # Create a set of document IDs we've seen
        seen_ids = set()
        combined_docs = []
        
        # Process vector store results first (they usually have higher precision)
        for doc in vector_docs:
            doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_id", ""))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_docs.append(doc)
        
        # Then add KG results if not already included
        for doc in kg_docs:
            doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_id", ""))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_docs.append(doc)
        
        return combined_docs
    
    def _format_path_as_text(self, paths: List[Dict], entity1: str, entity2: str) -> str:
        """Format KG path results as readable text"""
        text = f"Relationship between {entity1} and {entity2}:\n\n"
        
        for i, path in enumerate(paths):
            text += f"- {path['from']} {path['relationship']} {path['to']}\n"
            if path['context']:
                text += f"  Context: {path['context']}\n"
        
        return text
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answer a query using the advanced RAG system
        
        Args:
            query: The user's question
            
        Returns:
            Dict containing the answer and supporting information
        """
        # Start timing for performance analysis
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents using hybrid retrieval
        logger.info(f"Performing hybrid retrieval for query: {query}")
        relevant_docs = self.hybrid_retrieval(query)
        
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            return {
                "answer": "I don't have enough information to answer this question based on the documents provided.",
                "reasoning": "No relevant documents found for this query.",
                "sources": [],
                "confidence": "LOW"
            }
        
        # Step 2: Build context from relevant documents
        logger.info(f"Building context from {len(relevant_docs)} documents")
        logger.info(f"Relevant docs: {relevant_docs}")
        context = self._build_context(relevant_docs)
        
        # Step 3: Answer generation based on configuration
        if self.use_multihop and self._is_complex_query(query):
            # For complex queries, use multi-hop reasoning
            logger.info("Using multi-hop reasoning for complex query")
            multihop_result = self.reasoner.multi_hop_reasoning(query, context)
            
            answer = multihop_result["final_answer"]
            reasoning = "Multi-hop reasoning:\n" + "\n".join([
                f"Sub-question: {sa['question']}\nAnswer: {sa['answer']}"
                for sa in multihop_result["sub_answers"]
            ])
        elif self.use_cot:
            # Use chain-of-thought reasoning
            logger.info("Using chain-of-thought reasoning")
            cot_result = self.reasoner.answer_with_reasoning(query, context)
            
            # answer = cot_result["answer"]
            answer = cot_result
            reasoning = "no reasoning"
            # reasoning = f"Entities: {cot_result['entities']}\n\n" + \
                    #    f"Question type: {cot_result['question_type']}\n\n" + \
                    #    f"Reasoning: {cot_result['reasoning']}\n\n" + \
                    #    f"Limitations: {cot_result['limitations']}"
        else:
            # Use standard query
            logger.info("Using standard query processing")
            answer = stream_ollama_response(
                query=query,
                context=context,
                model=self.model,
                temperature=self.temperature
            )
            reasoning = "Standard query processing (no explicit reasoning steps)"
        
        # Step 4: Verify answer if enabled
        confidence = "MEDIUM"  # Default confidence
        verification = {}
        
        if self.verify_answers:
            logger.info("Verifying answer accuracy")
            verification = self.reasoner.verify_factual_accuracy(answer, context)
            confidence = verification["confidence"]
            
            # Add warnings for low confidence or unsupported claims
            if confidence == "LOW" or verification["unsupported_claims"]:
                answer += "\n\nNote: Some statements in this answer may not be fully supported by the source documents."
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        # Prepare sources information
        sources = []
        for doc in relevant_docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources.append(source)
        
        # Prepare the final response
        response = {
            "answer": answer,
            "reasoning": reasoning,
            "sources": sources,
            "confidence": confidence,
            "retrieval_time": elapsed,
            "document_count": len(relevant_docs)
        }
        
        # Add verification details if available
        if verification:
            response["verification"] = verification
        
        return response
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from retrieved documents"""
        # Simple concatenation with document metadata
        try:
            context_parts = []
            total_tokens = 0
            max_tokens = 4096  # Maximum context size
            
            for i, doc in enumerate(documents):
                source = doc.metadata.get("source", "Unknown")
                text = doc.page_content.strip()
                
                # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                doc_tokens = len(text) // 4
                
                # If adding this document would exceed the limit, truncate it
                if total_tokens + doc_tokens > max_tokens:
                    remaining_tokens = max_tokens - total_tokens
                    # Truncate text to fit remaining tokens
                    text = text[:remaining_tokens * 4] + "..."
                    logger.warning(f"Truncated document {source} to fit within token limit")
                
                # Add a header for each document chunk
                if "knowledge_graph_path" in source:
                    # This is a knowledge graph path, format differently
                    context_parts.append(f"[Knowledge Graph Relationship #{i+1}]\n{text}")
                else:
                    # Regular document
                    context_parts.append(f"[Document: {source}, Chunk: {i+1}]\n{text}")
                
                total_tokens += doc_tokens
                
                # If we've reached the token limit, stop adding documents
                if total_tokens >= max_tokens:
                    break
            
            return "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return ""
    
    
    def _is_complex_query(self, query: str) -> bool:
        """
        Determine if a query is complex and would benefit from multi-hop reasoning
        
        Simple heuristics:
        - Contains multiple question marks
        - Contains words like "compare", "relationship", "difference", "connection"
        - Length > 15 words
        - Contains multiple entities
        """
        query_lower = query.lower()
        
        # Check for multiple questions
        if query.count("?") > 1:
            return True
        
        # Check for comparison or relationship words
        complex_terms = ["compare", "relationship", "relation", "difference", 
                         "connection", "impact", "effect", "influence", "versus", "vs",
                         "how does", "in what way", "explain why", "analyze"]
        
        for term in complex_terms:
            if term in query_lower:
                return True
        
        # Check word count (approx)
        if len(query.split()) > 15:
            return True
        
        # Check for multiple entities
        entities = self._extract_query_entities(query)
        if len(entities) > 2:
            return True
        
        return False