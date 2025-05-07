import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def improved_llama_cpp_response(llm, query: str, context: str, model: str = None, temperature: float = 0.3) -> str:
    """
    Generate a response using llama.cpp LLM with an improved prompt template.
    
    Args:
        llm: The llama.cpp model instance
        query (str): The user's question
        context (str): Retrieved context chunks
        model (str): Ignored (for API compatibility)
        temperature (float): Sampling temperature
        
    Returns:
        str: Generated answer
    """
    # Create a focused prompt that encourages information extraction
    prompt = f"""
You are a precise document assistant answering questions based on the provided context. Your answers should be:
1. Direct and specific
2. Based ONLY on information in the context
3. Brief but complete
4. If the information is not in the context, state "The document doesn't contain information about [topic]"
5.If there is any uncertainty or the context does not contain the answer, respond exactly with:
"The context does not contain the answer."
CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- If the information is explicitly stated in the context, provide the answer directly
- If the information can be inferred from multiple parts of the context, connect those parts clearly
- If the information is NOT in the context, simply state "The document doesn't contain information about [topic]"
- DO NOT include your reasoning process in your response
- DO NOT make up information not present in the context
- DO NOT hedge your answer with phrases like "based on the context"
- If the context does not contain the answer, respond exactly with:
"The context does not contain the answer."

ANSWER:
"""
    
    try:
        logger.info(f"llama.cpp prompt: {prompt}")
        if not llm:
            logger.error("llm not initialized")
            return "Model not available."
            
        response = llm(prompt=prompt, max_tokens=7500, temperature=temperature)
        logger.info(f"llama.cpp response: {response}")
        # Extract generated text
        generated_text = response["choices"][0]["text"] if isinstance(response, dict) and "choices" in response else response
        
        logger.info(f"llama.cpp response generated successfully")
        return generated_text.strip()
    except Exception as e:
        logger.error(f"llama.cpp generation error: {str(e)}")
        return "Error generating response."

def specialized_prompt_by_question_type(llm, query: str, context: str, question_type: str, temperature: float = 0.7) -> str:
    """
    Use specialized prompts based on the question type.
    
    Args:
        llm: The llama.cpp model instance
        query (str): The user's question
        context (str): Retrieved context chunks
        question_type (str): Type of question (factoid, reasoning, multi_hop, unanswerable)
        temperature (float): Sampling temperature
        
    Returns:
        str: Generated answer
    """
    if question_type == "factoid":
        prompt = f"""
You are answering a FACTOID question that requires specific facts from the provided context.

CONTEXT:
{context}

FACTOID QUESTION:
{query}

INSTRUCTIONS:
- Provide a direct, specific answer with exactly the fact(s) requested
- Use only information explicitly stated in the context
- Be concise - the ideal answer is 1-2 sentences
- If the specific fact is not in the context, state "The document doesn't mention [topic]"

ANSWER:
"""
    elif question_type == "multi_hop":
        prompt = f"""
You are answering a question that requires connecting information from MULTIPLE PARTS of the context.

CONTEXT:
{context}

MULTI-HOP QUESTION:
{query}

INSTRUCTIONS:
- Carefully examine different chunks in the context
- Connect relevant information across multiple chunks
- Provide a complete answer that combines all relevant details
- If parts of the answer are missing from the context, clearly state what information is unavailable

ANSWER:
"""
    elif question_type == "unanswerable":
        prompt = f"""
You are determining if a question can be answered using the provided context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Carefully check if the context contains ANY information related to the question
- If the context contains NO relevant information, reply "The document doesn't contain information about [topic]"
- If the context contains PARTIAL information, provide what is available and specify what's missing
- Do not invent information not present in the context

ANSWER:
"""
    else:  # reasoning or default
        prompt = f"""
You are answering a question that requires careful REASONING about the provided context.

CONTEXT:
{context}

REASONING QUESTION:
{query}

INSTRUCTIONS:
- Analyze the information in the context thoroughly
- Make logical inferences based ONLY on what's provided
- Provide a specific answer that directly addresses the question
- If the context doesn't contain enough information, state what's missing

ANSWER:
"""
    
    try:
        logger.info(f"llama.cpp prompt for {question_type} question: {prompt}")
        if not llm:
            logger.error("llm not initialized")
            return "Model not available."
            
        response = llm(prompt=prompt, max_tokens=512, temperature=temperature)
        
        # Extract generated text
        # generated_text = response["choices"][0]["text"] if isinstance(response, dict) and "choices" in response else response
        generated_text = response
        logger.info(f"llama.cpp response generated successfully")
        return generated_text.strip()
    except Exception as e:
        logger.error(f"llama.cpp generation error: {str(e)}")
        return "Error generating response."