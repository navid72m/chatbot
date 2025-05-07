# hybrid_search.py

def hybrid_search(vector_store, query: str, document_name: str, k: int = 5):
    """
    Perform hybrid search combining vector search with keyword matching.
    
    Args:
        vector_store: Vector store instance
        query: User query
        document_name: Document to search in
        k: Number of results to return
        
    Returns:
        List of retrieved chunks
    """
    # Extract keywords from the query (words longer than 3 chars)
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    keywords = [word for word in keywords if word not in ["what", "where", "when", "which", "about", "with", "from"]]
    
    # Get vector search results
    vector_results = vector_store.search(
        query=query, 
        k=k, 
        filter={"source": document_name}
    )
    
    # For multi-hop questions, we want more diverse results
    if len(keywords) >= 2:
        # Perform additional searches with individual keywords
        keyword_results = []
        for keyword in keywords[:3]:  # Use top 3 keywords
            # Search with just this keyword
            results = vector_store.search(
                query=keyword,
                k=3,  # Smaller k for individual keywords
                filter={"source": document_name}
            )
            keyword_results.extend(results)
        
        # Combine all results
        all_results = vector_results + keyword_results
        
        # Remove duplicates (keeping order)
        unique_results = []
        seen = set()
        
        for chunk in all_results:
            # Create a hash for the chunk (using string representation)
            chunk_hash = str(chunk)
            if chunk_hash not in seen:
                seen.add(chunk_hash)
                unique_results.append(chunk)
        
        # Return top k unique results
        return unique_results[:k]
    
    # For regular questions, just return vector results
    return vector_results