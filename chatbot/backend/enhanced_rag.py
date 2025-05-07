# enhanced_rag.py
import logging
from typing import Dict, Any, List, Optional, Tuple
import re
# from bitnet_wrapper import BitNetInterface


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_rag")

class EnhancedRAG:
    """Enhanced RAG system with improved retrieval and generation."""
    
    def __init__(self, vector_store, llm):
        """
        Initialize the enhanced RAG system.
        
        Args:
            vector_store: Vector store for retrieval
            llm: LLM model for generation
        """
        # from bitnet_wrapper import BitNetCppInterface
        # bitnet_cpp = BitNetCppInterface()
        self.vector_store = vector_store
        self.llm = llm
        # from document_processor import process_document, chunk_document
        # document_text = process_document(document_name)
        # chunks = chunk_document(document_text, chunk_size=200, chunk_overlap=50)
        # _=self.vector_store.add_document(document_name, chunks)
        logger.info("Enhanced RAG system initialized")
    
    def query(self, query: str, document_name: str, k: int = 5, temperature: float = 0.3) -> Dict[str, Any]:
        """
        Process a query with enhanced RAG techniques.
        
        Args:
            query: User question
            document_name: Document to search
            k: Number of chunks to retrieve
            temperature: LLM temperature
            
        Returns:
            Dict with response and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Classify question type
        question_type = self._classify_question(query)
        logger.info(f"Question classified as: {question_type}")
        
        # Step 2: Adjust retrieval parameters based on question type
        retrieval_k = k
        if question_type == "multi_hop":
            # Get more chunks for multi-hop questions
            retrieval_k = min(k * 2, 10)
        
        # self.vector_store.add_document(document_name, chunks)
        # Step 3: Retrieve chunks with hybrid search
        chunks = self._hybrid_search(query, document_name, retrieval_k)
        print(f"chunks: {chunks}")
        # Step 4: Build context from chunks
        context = self._build_context(chunks)
        print(f"context: {context}")
        
        # Step 5: Generate answer with specialized prompt
        prompt = self._get_specialized_prompt(query, context, question_type)
        
        # Step 6: Generate and post-process answer
        generated_text = self._generate_answer(prompt, temperature)
        clean_answer = self._post_process_answer(generated_text)
        
        return {
            "response": clean_answer,
            "question_type": question_type,
            "chunks_retrieved": len(chunks),
        }
    
    def _classify_question(self, query: str) -> str:
        """Classify question into type."""
        query = query.lower()
        
        # Factoid patterns
        factoid_patterns = [
            r'^who\s',
            r'^what\s+is\s+',
            r'^what\s+are\s+',
            r'^when\s',
            r'^where\s',
            r'which\s+\w+\s+is',
            r'which\s+\w+\s+are',
            r'how\s+many\s',
            r'how\s+much\s'
        ]
        
        # Multi-hop patterns
        multi_hop_patterns = [
            r'and\s+(?:also|what|how|where|when|why|who)',
            r'as\s+well\s+as',
            r'relationship\s+between',
            r'compare',
            r'both',
            r'together\s+with'
        ]
        
        # Unanswerable patterns
        unanswerable_patterns = [
            r'opinion',
            r'thoughts',
            r'feel',
            r'imagine',
            r'would\s+you',
            r'could\s+you\s+speculate',
            r'what\s+if',
            r'beyond\s+the\s+context',
            r'not\s+mentioned'
        ]
        
        # Check for factoid patterns
        for pattern in factoid_patterns:
            if re.search(pattern, query):
                return "factoid"
        
        # Check for multi-hop patterns
        for pattern in multi_hop_patterns:
            if re.search(pattern, query):
                return "multi_hop"
        
        # Check for unanswerable patterns
        for pattern in unanswerable_patterns:
            if re.search(pattern, query):
                return "unanswerable"
        
        # Count entities as a signal for multi-hop
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        if len(capitalized_words) >= 2:
            return "multi_hop"
        
        # Check complexity by query length
        if len(query.split()) > 15 or "," in query:
            return "multi_hop"
        
        return "general"
    
    def _hybrid_search(self, query: str, document_name: str, k: int) -> List:
        """
        Advanced hybrid search that combines vector similarity, BM25 keywords, and forced retrieval fallbacks.
        
        Args:
            query: User question
            document_name: Document to search
            k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks
        """
        logger.info(f"Performing hybrid search for: {query} in {document_name}")
       
        # Step 1: Try standard vector search first
        semantic_results = self.vector_store.search(
            query=query, 
            k=k, 
            filter={"source": document_name}
        )
        
        logger.info(f"Vector search returned {len(semantic_results)} results")
        
        # If we got results, great!
        if semantic_results and len(semantic_results) > 0:
            # Step 2: Extract keywords for expansion
            keywords = self._extract_important_keywords(query)
            logger.info(f"Extracted keywords: {keywords}")
            
            # Step 3: If we have keywords, get additional chunks with keyword search
            keyword_results = []
            if keywords:
                for keyword in keywords[:2]:  # Use top 2 keywords
                    keyword_chunks = self.vector_store.search(
                        query=keyword,
                        k=3,  # Smaller k for keyword searches
                        filter={"source": document_name}
                    )
                    keyword_results.extend(keyword_chunks)
                
                logger.info(f"Keyword search returned {len(keyword_results)} additional results")
                
                # Step 4: Combine and deduplicate results
                all_results = list(semantic_results)
                
                # Add unique keyword results
                for chunk in keyword_results:
                    if not any(self._chunks_are_similar(chunk, existing) for existing in all_results):
                        all_results.append(chunk)
                
                logger.info(f"Combined search returned {len(all_results)} total results")
                return all_results[:k]  # Return top k combined results
            
            return semantic_results  # Return vector results if no keywords
        
        # If vector search failed, we need stronger fallbacks
        logger.warning("Vector search returned no results. Using fallback retrieval methods.")
        
        # Fallback 1: Try retrieving all chunks from the document
        try:
            all_document_chunks = self._get_all_document_chunks(document_name)
            if all_document_chunks and len(all_document_chunks) > 0:
                logger.info(f"Fallback: Retrieved {len(all_document_chunks)} total chunks from document")
                
                # Score chunks by keyword presence
                scored_chunks = self._score_chunks_by_keywords(all_document_chunks, query)
                
                # Return top k scored chunks or all if fewer than k
                return [chunk for chunk, _ in scored_chunks[:k]]
        except Exception as e:
            logger.error(f"Fallback retrieval error: {str(e)}")
        
        # Fallback 2: If all else fails, create a synthetic chunk
        logger.warning("All retrieval methods failed. Creating synthetic chunk.")
        return [{"content": f"This is a document about {document_name}. No specific information found for: {query}"}]

    def _extract_important_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Remove common stop words
        stop_words = ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "about", 
                    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", 
                    "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must"]
        
        # Extract words longer than 3 characters, not in stop words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Prioritize proper nouns (capitalized words)
        capitalized = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
        
        # Combine and prioritize
        unique_keywords = []
        for word in capitalized + keywords:
            if word.lower() not in [k.lower() for k in unique_keywords]:
                unique_keywords.append(word.lower())
        
        return unique_keywords

    def _chunks_are_similar(self, chunk1, chunk2) -> bool:
        """Check if two chunks are similar (to avoid duplicates)."""
        # Extract text from chunks
        text1 = self._extract_chunk_text(chunk1)
        text2 = self._extract_chunk_text(chunk2)
        
        # Simple similarity check
        if text1 and text2:
            # If chunks are very short, check for exact match
            if len(text1) < 50 or len(text2) < 50:
                return text1 == text2
                
            # For longer chunks, check if one is contained in the other
            if text1 in text2 or text2 in text1:
                return True
                
            # Check for high token overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1.intersection(words2)) / min(len(words1), len(words2))
                return overlap > 0.8
        
        return False

    def _extract_chunk_text(self, chunk) -> str:
        """Extract text content from a chunk."""
        if hasattr(chunk, "page_content"):
            return chunk.page_content
        elif isinstance(chunk, dict) and "content" in chunk:
            return chunk["content"]
        elif isinstance(chunk, dict) and "text" in chunk:
            return chunk["text"]
        elif isinstance(chunk, str):
            return chunk
        else:
            try:
                return str(chunk)
            except:
                return ""

    def _get_all_document_chunks(self, document_name: str) -> List:
        """Get all chunks from a document."""
        # Method 1: Try to get all document chunks using a very generic query
        try:
            return self.vector_store.search(
                query="information document content",
                k=50,  # Get many chunks
                filter={"source": document_name}
            )
        except:
            # Method 2: If your vector store has a specific method for this
            if hasattr(self.vector_store, "get_all_chunks"):
                return self.vector_store.get_all_chunks(document_name)
            else:
                return []

    def _score_chunks_by_keywords(self, chunks: List, query: str) -> List[Tuple]:
        """Score chunks by keyword presence and return sorted list of (chunk, score)."""
        # Extract keywords from query
        keywords = self._extract_important_keywords(query)
        
        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            text = self._extract_chunk_text(chunk)
            
            # Skip empty chunks
            if not text or len(text.strip()) == 0:
                continue
                
            # Calculate score based on keyword presence
            score = 0
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    score += 1
            
            # Add to results if it has any score
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score (highest first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks
    
    def _build_context(self, chunks: List) -> str:
        """Build context string from retrieved chunks with tags for better interpretability."""
        context_parts = []

        for i, chunk in enumerate(chunks):
            label = "[CHUNK]"
            if isinstance(chunk, dict):
                content = chunk.get("content") or chunk.get("text") or str(chunk)
                metadata = chunk.get("metadata", {})

                # Try to infer high-level label from metadata or content
                if any(k in str(content).lower() for k in ["contact", "phone", "email", "address"]):
                    label = "[CONTACT INFO]"
                elif any(k in str(content).lower() for k in ["education", "university", "degree"]):
                    label = "[EDUCATION]"
                elif any(k in str(content).lower() for k in ["intern", "student", "job", "experience", "company"]):
                    label = "[WORK EXPERIENCE]"

                context_parts.append(f"{label}\n{content.strip()}")

            elif hasattr(chunk, "page_content"):
                context_parts.append(f"[CHUNK]\n{chunk.page_content.strip()}")

            elif isinstance(chunk, str):
                context_parts.append(f"[CHUNK]\n{chunk.strip()}")

            else:
                try:
                    context_parts.append(f"[CHUNK]\n{str(chunk).strip()}")
                except:
                    continue

        return "\n\n".join(context_parts)

    
    def _get_specialized_prompt(self, query: str, context: str, question_type: str) -> str:
        """Get specialized prompt based on question type."""

        if question_type == "factoid":
            return f"""
    You are a precise assistant extracting **short factual answers** from context.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    INSTRUCTIONS:
    - Return the **specific fact(s)** directly, without extra explanation.
    - Use 1–2 concise sentences.
    - If the information is **not clearly in the context**, say:
    \"This question cannot be answered based on the provided document.\"
    - Do **not** guess or invent details.

    ANSWER:
    """

        elif question_type == "multi_hop":
            return f"""
    You are answering a question that requires connecting multiple facts in the context.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    INSTRUCTIONS:
    - Carefully combine **multiple parts** of the context.
    - If needed, refer to different people, dates, or entities across chunks.
    - Only use **explicitly stated** information.
    - If **any key fact is missing**, say:
    \"This question cannot be answered based on the provided document.\"

    ANSWER:
    """

        elif question_type == "unanswerable":
            return f"""
    Check whether the following question can be answered using only the provided context.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    INSTRUCTIONS:
    - If all required facts are present, answer concisely.
    - If even one important detail is missing, respond with:
    \"This question cannot be answered based on the provided document.\"
    - Do not guess or infer — only use what's explicitly in the context.

    ANSWER:
    """

        else:  # general
            return f"""
    You are answering a question using the following document context.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    INSTRUCTIONS:
    - Use only information in the context.
    - Answer clearly and directly.
    - If the answer is not present, respond with:
    \"This question cannot be answered based on the provided document.\"

    ANSWER:
    """

    def _generate_answer(self, prompt: str, temperature: float) -> str:
        """Generate answer using LLM."""
        try:
            if self.llm is None:
                logger.error("llm not initialized")
                return "Error generating response."
            response = self.llm(prompt=prompt, max_tokens=2048, temperature=temperature)
            logger.info(f"response: {response}")
            # Extract generated text
            if isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["text"]
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _post_process_answer(self, answer: str) -> str:
        """Clean up the generated answer."""
        if not answer:
            return ""
            
        # Remove numbered thinking steps
        answer = re.sub(r'\d+\.\s+.*?(?=\d+\.|OUTPUT:|RESPONSE:|ANSWER:|$)', '', answer, flags=re.DOTALL)
        
        # Remove section headers
        sections = ["THINKING:", "RESPONSE:", "OUTPUT:", "ANSWER:", "CONCLUSION:"]
        for section in sections:
            if section in answer:
                parts = answer.split(section)
                if len(parts) > 1:
                    answer = parts[1].strip()
            
        # Clean up common phrases
        phrases_to_remove = [
            "Based on the context provided,",
            "Based on the context,",
            "Based on the provided context,",
            "Based on the given context,",
            "According to the context,",
            "From the context,",
            "The context indicates that",
            "As per the context,"
        ]
        
        for phrase in phrases_to_remove:
            answer = answer.replace(phrase, "")
        
        # Clean up whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer