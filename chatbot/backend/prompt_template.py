# prompt_templates.py

def factoid_prompt(query: str, context: str) -> str:
    """Specialized prompt for factoid questions."""
    return f"""
You are answering a FACTOID question that requires a specific, direct answer.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Provide ONLY the specific fact(s) requested
- Be extremely brief and direct - use 1-2 sentences maximum
- If the information is not in the context, state ONLY "The document doesn't mention [topic]"
- DO NOT include reasoning, explanations, or thinking process

ANSWER:
"""

def multi_hop_prompt(query: str, context: str) -> str:
    """Specialized prompt for multi-hop questions requiring connecting information."""
    return f"""
You are answering a question that requires connecting information from MULTIPLE PARTS of the context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Carefully examine different chunks in the context
- Connect relevant information across multiple chunks
- Provide a complete answer that combines all relevant details
- Be concise but thorough
- If parts of the answer are missing from the context, clearly state what information is unavailable

ANSWER:
"""

def unanswerable_prompt(query: str, context: str) -> str:
    """Specialized prompt for potentially unanswerable questions."""
    topic = " ".join(query.split()[:5])  # Extract first few words
    return f"""
You are determining if a question can be answered using the provided context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Carefully check if the context contains ANY information related to the question
- If the context contains NO relevant information, respond ONLY with: "The document doesn't contain information about {topic}."
- If the context contains PARTIAL information, provide what is available and specify what's missing
- Be brief and direct
- Do not invent information not present in the context

ANSWER:
"""

def general_prompt(query: str, context: str) -> str:
    """Default prompt for general questions."""
    return f"""
You are a precise document assistant answering questions based on the provided context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer directly based ONLY on information in the context
- Be concise but complete
- If the information is not in the context, state "The document doesn't mention [topic]"
- DO NOT include any reasoning process in your response
- DO NOT make up information not present in the context
- DO NOT start with phrases like "Based on the context"

ANSWER:
"""