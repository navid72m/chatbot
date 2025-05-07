import logging
from typing import List, Dict, Any, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def expand_query(llm, query: str) -> str:
    """
    Expand a query to improve retrieval by adding synonyms and rephrasing.
    
    Args:
        llm: The LLM model instance
        query (str): The original user query
        
    Returns:
        str: Expanded query for better retrieval
    """
    # If query is very short, use a simpler expansion approach
    if len(query.split()) <= 3:
        expanded_terms = simple_keyword_expansion(query)
        return f"{query} {' '.join(expanded_terms)}"
    
    # For longer queries, use the LLM
    prompt = f"""
Your task is to expand a search query to improve document retrieval.

ORIGINAL QUERY: "{query}"

Provide an expanded version that:
1. Includes synonyms for key terms
2. Rephrases the question in an alternative way 
3. Extracts important entities and concepts

Format your response as:
KEYWORDS: [list key terms and synonyms]
REPHRASING: [alternative phrasing of the query]
ENTITIES: [key entities mentioned]

Keep your response brief and focused.
"""
    
    try:
        response = llm(prompt=prompt, max_tokens=200, temperature=0.3)
        
        # Extract generated text
        expanded_text = response["choices"][0]["text"] if isinstance(response, dict) and "choices" in response else response
        
        # Parse the response to extract useful parts
        keywords = extract_section(expanded_text, "KEYWORDS:")
        rephrasing = extract_section(expanded_text, "REPHRASING:")
        entities = extract_section(expanded_text, "ENTITIES:")
        
        # Combine into expanded query (keeping original query as well)
        expanded_query = f"{query} {keywords} {rephrasing} {entities}"
        
        # Clean up expanded query
        expanded_query = re.sub(r'\s+', ' ', expanded_query).strip()
        
        logger.info(f"Original query: {query}")
        logger.info(f"Expanded query: {expanded_query}")
        
        return expanded_query
    except Exception as e:
        logger.error(f"Query expansion error: {str(e)}")
        # In case of error, return original query
        return query

def extract_section(text: str, section_name: str) -> str:
    """Extract a section from the LLM output."""
    pattern = f"{section_name}(.*?)(?=KEYWORDS:|REPHRASING:|ENTITIES:|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def simple_keyword_expansion(query: str) -> List[str]:
    """
    Simple rule-based expansion for short queries.
    """
    expanded_terms = []
    
    # Common synonyms for question words
    expansions = {
        "who": ["person", "individual", "name"],
        "what": ["description", "definition", "detail"],
        "when": ["time", "date", "period"],
        "where": ["location", "place", "position"],
        "why": ["reason", "cause", "explanation"],
        "how": ["method", "process", "way"],
        # Add domain-specific expansions based on your documents
        "contact": ["phone", "email", "address", "number"],
        "information": ["details", "data", "facts"],
        "location": ["address", "place", "position"],
        "education": ["university", "school", "degree", "studies"],
        "experience": ["work", "job", "career", "history"],
        "skills": ["abilities", "competencies", "expertise"],
        "project": ["work", "assignment", "initiative"],
        "role": ["position", "job", "responsibility"]
    }
    
    # Add expansions for words in the query
    words = query.lower().split()
    for word in words:
        if word in expansions:
            expanded_terms.extend(expansions[word])
    
    return expanded_terms

def hybrid_search(vector_store, query: str, llm, k: int = 5, filter: Dict = None):
    """
    Perform hybrid search with query expansion and reranking.
    
    Args:
        vector_store: Vector store instance
        query (str): Original user query
        llm: LLM instance for query expansion
        k (int): Number of results to retrieve
        filter (Dict): Filters to apply to search
        
    Returns:
        List: Retrieved and reranked chunks
    """
    # Step 1: Expand the query
    expanded_query = expand_query(llm, query)
    
    # Step 2: Retrieve more results than needed for reranking
    retrieve_k = min(k * 2, 10)  # Get more results for filtering
    results = vector_store.search(expanded_query, k=retrieve_k, filter=filter)
    
    # Step 3: Simple reranking - prioritize chunks containing exact query terms
    if len(results) > k:
        reranked_results = rerank_by_term_overlap(query, results)
        return reranked_results[:k]
    
    return results

def rerank_by_term_overlap(query: str, chunks: List[Any]) -> List[Any]:
    """
    Simple reranking based on term overlap between query and chunks.
    
    Args:
        query (str): Original user query
        chunks (List): Retrieved chunks
        
    Returns:
        List: Reranked chunks
    """
    query_terms = set(query.lower().split())
    
    # Calculate term overlap scores
    scored_chunks = []
    for chunk in chunks:
        # Extract text from chunk (handling different formats)
        if hasattr(chunk, "page_content"):
            text = chunk.page_content
        elif isinstance(chunk, dict) and "content" in chunk:
            text = chunk["content"]
        elif isinstance(chunk, str):
            text = chunk
        else:
            text = str(chunk)
        
        # Calculate overlap score
        chunk_terms = set(text.lower().split())
        overlap = len(query_terms.intersection(chunk_terms))
        
        # Store score with chunk
        scored_chunks.append((chunk, overlap))
    
    # Sort by overlap score (descending)
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return reranked chunks
    return [chunk for chunk, score in scored_chunks]