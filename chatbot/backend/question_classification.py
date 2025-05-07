import logging
from typing import List, Dict, Any, Optional, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_question(llm, query: str) -> str:
    """
    Classify a question to determine the best approach for answering.
    
    Args:
        llm: LLM model instance
        query (str): The user's question
        
    Returns:
        str: Question type (factoid, multi_hop, reasoning, unanswerable)
    """
    # Use rule-based classification first for efficiency
    question_type = rule_based_classification(query)
    if question_type:
        logger.info(f"Rule-based classification: {question_type} for query: {query}")
        return question_type
    
    # Fall back to LLM classification if needed
    prompt = f"""
Classify the following question into exactly ONE of these categories:
- FACTOID: Simple questions asking for specific facts, names, dates, or direct information
- MULTI_HOP: Questions requiring connecting information from multiple sources or parts of a document
- REASONING: Questions requiring analysis, inference, comparison, or interpretation
- UNANSWERABLE: Questions that would be difficult to answer from a typical document

Question: {query}

Respond with ONLY ONE word from the above categories.
"""
    
    try:
        response = llm(prompt=prompt, max_tokens=50, temperature=0.1)
        
        # Extract generated text
        classification = response["choices"][0]["text"] if isinstance(response, dict) and "choices" in response else response
        classification = classification.strip().upper()
        
        # Normalize the response
        if "FACTOID" in classification:
            return "factoid"
        elif "MULTI" in classification or "HOP" in classification:
            return "multi_hop"
        elif "REASON" in classification:
            return "reasoning"
        elif "ANSWER" in classification:
            return "unanswerable"
        else:
            # Default to reasoning if classification is unclear
            return "reasoning"
    except Exception as e:
        logger.error(f"Question classification error: {str(e)}")
        # Default to reasoning in case of error
        return "reasoning"

def rule_based_classification(query: str) -> Optional[str]:
    """
    Use rule-based approach to classify questions.
    
    Args:
        query (str): The user's question
        
    Returns:
        Optional[str]: Question type or None if can't determine
    """
    query = query.lower().strip()
    
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
        r'how\s+much\s',
        r'names?\s+of\s',
        r'lists?\s+of\s',
        r'contacts?',
        r'address',
        r'phone',
        r'email',
        r'^in\s+what\s',
    ]
    
    # Multi-hop patterns
    multi_hop_patterns = [
        r'relation\s+between',
        r'relationship\s+between',
        r'compare',
        r'comparison',
        r'difference\s+between',
        r'similarities\s+between',
        r'both',
        r'connect',
        r'combined',
        r'and.*also',
        r'as\s+well\s+as',
        r'multiple',
    ]
    
    # Reasoning patterns
    reasoning_patterns = [
        r'why\s',
        r'how\s+does\s',
        r'how\s+do\s',
        r'explain',
        r'describe',
        r'analyze',
        r'evaluate',
        r'assess',
        r'what\s+would\s+happen',
        r'what\s+might\s+happen',
        r'implications',
        r'consequences',
        r'effects?\s+of',
        r'impacts?\s+of',
        r'benefits?\s+of',
        r'drawbacks?\s+of',
        r'advantages?\s+of',
        r'disadvantages?\s+of',
    ]
    
    # Unanswerable patterns
    unanswerable_patterns = [
        r'could\s+you\s+tell\s+me\s+more',
        r'anything\s+else',
        r'additional\s+information',
        r'other\s+than',
        r'beyond\s+what',
        r'outside\s+of',
        r'not\s+mentioned',
        r'not\s+stated',
        r'not\s+included',
        r'speculate',
        r'predict',
        r'future',
        r'what\s+will\s+happen',
        r'(?:your|you|the|their)\s+thoughts',
        r'(?:your|you|the|their)\s+opinions?',
        r'(?:your|you|the|their)\s+feelings?',
    ]
    
    # Check patterns in order of specificity
    for pattern in unanswerable_patterns:
        if re.search(pattern, query):
            return "unanswerable"
            
    for pattern in multi_hop_patterns:
        if re.search(pattern, query):
            return "multi_hop"
            
    for pattern in reasoning_patterns:
        if re.search(pattern, query):
            return "reasoning"
            
    for pattern in factoid_patterns:
        if re.search(pattern, query):
            return "factoid"
    
    # Count entities as a signal for multi-hop
    entities = extract_entities(query)
    if len(entities) >= 2:
        return "multi_hop"
    
    # No pattern matched
    return None

def extract_entities(text: str) -> List[str]:
    """
    Extract potential entities from text.
    This is a simple implementation - in production, use a proper NER.
    """
    # Simple rule: capitalized words (excluding first word of sentence)
    words = text.split()
    entities = []
    
    # Check for capitalized words (not at start of sentence)
    for i, word in enumerate(words):
        if i > 0 and word[0].isupper():
            # Remove punctuation
            clean_word = word.strip('.,;:?!()"\'')
            if clean_word:
                entities.append(clean_word)
    
    # Also check for quoted terms
    quoted = re.findall(r'"([^"]*)"', text)
    entities.extend(quoted)
    
    return entities

def improved_rag_query(
    llm, 
    vector_store, 
    query: str, 
    document_name: str, 
    k: int = 5, 
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Improved RAG query with question classification and specialized handling.
    
    Args:
        llm: LLM model instance
        vector_store: Vector store instance
        query (str): User question
        document_name (str): Document to query
        k (int): Number of chunks to retrieve
        temperature (float): LLM temperature
        
    Returns:
        Dict: Response with answer and metadata
    """
    logger.info(f"Processing query: {query}")
    
    # Step 1: Classify the question
    question_type = classify_question(llm, query)
    logger.info(f"Question classified as: {question_type}")
    
    # Step 2: Adjust retrieval parameters based on question type
    retrieval_k = k
    if question_type == "multi_hop":
        # Retrieve more chunks for multi-hop questions
        retrieval_k = min(k * 2, 10)
    
    # Step 3: Perform advanced retrieval
    from query_expansion import hybrid_search
    chunks = hybrid_search(
        vector_store=vector_store,
        query=query,
        llm=llm,
        k=retrieval_k,
        filter={"source": document_name}
    )
    
    # Step 4: Check if question might be unanswerable
    if not chunks or (question_type == "unanswerable" and not has_relevant_information(query, chunks)):
        logger.info("Question appears to be unanswerable")
        return {
            "response": f"The document doesn't contain information about {extract_main_topic(query)}.",
            "question_type": question_type,
            "chunks_retrieved": len(chunks),
            "is_answerable": False
        }
    
    # Step 5: Build context with appropriate formatting
    if question_type == "multi_hop":
        context = build_multi_hop_context(chunks)
    else:
        context = build_standard_context(chunks)
    
    # Step 6: Generate answer with specialized prompt
    from improved_prompt_template import specialized_prompt_by_question_type
    answer = specialized_prompt_by_question_type(
        llm=llm,
        query=query,
        context=context,
        question_type=question_type,
        temperature=temperature
    )
    
    return {
        "response": answer,
        "question_type": question_type,
        "chunks_retrieved": len(chunks),
        "is_answerable": True
    }

def has_relevant_information(query: str, chunks: List[Any]) -> bool:
    """
    Check if chunks likely contain information relevant to the query.
    
    Args:
        query (str): User question
        chunks (List): Retrieved chunks
        
    Returns:
        bool: True if likely contains relevant information
    """
    # Extract key terms from query
    query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
    
    # Check if any chunks contain these terms
    for chunk in chunks:
        # Extract text from chunk (handling different formats)
        if hasattr(chunk, "page_content"):
            text = chunk.page_content.lower()
        elif isinstance(chunk, dict) and "content" in chunk:
            text = chunk["content"].lower()
        elif isinstance(chunk, str):
            text = chunk.lower()
        else:
            text = str(chunk).lower()
        
        # Count how many query terms appear in the chunk
        matching_terms = sum(1 for term in query_terms if term in text)
        
        # If chunk contains several key terms, consider it relevant
        if matching_terms >= min(2, len(query_terms) / 2):
            return True
    
    # No chunk contained enough relevant terms
    return False

def extract_main_topic(query: str) -> str:
    """
    Extract the main topic from a question for better "not found" responses.
    
    Args:
        query (str): User question
        
    Returns:
        str: Main topic of the question
    """
    # Remove question words and stopwords
    cleaned = re.sub(r'^(who|what|when|where|why|how|is|are|was|were|do|does|did|can|could|would|should)\s+', '', query.lower())
    
    # Get the first few words as the topic
    words = cleaned.split()
    if len(words) <= 3:
        return cleaned
    else:
        return ' '.join(words[:3]) + "..."

def build_standard_context(chunks: List[Any]) -> str:
    """
    Build a standard context from chunks.
    
    Args:
        chunks (List): Retrieved chunks
        
    Returns:
        str: Formatted context
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks):
        # Extract text from chunk (handling different formats)
        if hasattr(chunk, "page_content"):
            text = chunk.page_content
        elif isinstance(chunk, dict) and "content" in chunk:
            text = chunk["content"]
        elif isinstance(chunk, str):
            text = chunk
        else:
            text = str(chunk)
        
        # Add to context
        context_parts.append(f"CHUNK {i+1}:\n{text}")
    
    return "\n\n".join(context_parts)

def build_multi_hop_context(chunks: List[Any]) -> str:
    """
    Build a context specially formatted for multi-hop questions.
    
    Args:
        chunks (List): Retrieved chunks
        
    Returns:
        str: Formatted context
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks):
        # Extract text from chunk (handling different formats)
        if hasattr(chunk, "page_content"):
            text = chunk.page_content
        elif isinstance(chunk, dict) and "content" in chunk:
            text = chunk["content"]
        elif isinstance(chunk, str):
            text = chunk
        else:
            text = str(chunk)
        
        # Add to context with separator line to make connections more visible
        context_parts.append(f"SECTION {i+1}:\n{text}\n{'='*40}")
    
    header = "IMPORTANT: The answer may require connecting information from multiple sections below.\n\n"
    return header + "\n\n".join(context_parts)