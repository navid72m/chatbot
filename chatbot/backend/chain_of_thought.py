# chain_of_thought.py - Implementation of chain-of-thought reasoning for RAG
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from langchain.docstore.document import Document

from llm_interface import query_ollama

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChainOfThoughtReasoner:
    """
    Chain of Thought implementation to reduce hallucination in LLM responses
    through explicit reasoning steps before answering.
    """
    
    def __init__(self, model="mistral", temperature=0.7):
        """Initialize the CoT reasoner with default parameters"""
        self.model = model
        self.temperature = temperature
    
    def answer_with_reasoning(self, query: str, context: str) -> Dict[str, str]:
        """
        Generate a response using explicit chain-of-thought reasoning
        
        Args:
            query: The user question
            context: The context information from retrieved documents
            
        Returns:
            Dict containing both reasoning steps and the final answer
        """
        # Create a specialized prompt for CoT reasoning
        cot_prompt = f"""
Context information is below.
---------------------
{context}
---------------------

To answer the question: "{query}"

Think through this step-by-step:

1. Identify the key entities and concepts in the question.
2. Determine what specific information the question is asking for.
3. Find the relevant information in the context.
4. Reason about how this information answers the question.
5. Check if there are any assumptions or limitations in your answer.
6. Provide your final answer based on this reasoning.

Format your response as follows:
ENTITIES: [List the key entities and concepts from the question]
QUESTION_TYPE: [What type of information is being requested]
RELEVANT_INFO: [Quote specific parts of the context that are relevant]
REASONING: [Connect the relevant information to answer the question]
LIMITATIONS: [Note any missing information or assumptions]
ANSWER: [Provide the final answer based on context only]
"""

        # Get the response using the enhanced CoT prompt
        response = query_ollama(
            query=cot_prompt, 
            context="",  # Context is already in our CoT prompt
            model=self.model, 
            temperature=self.temperature
        )
        
        logger.info(f"Generated CoT response of length {len(response)}")
        
        # Parse the structured response
        parsed_response = self._parse_cot_response(response)
        
        return parsed_response
    
    def _parse_cot_response(self, response: str) -> Dict[str, str]:
        """Parse the structured CoT response into components"""
        # Initialize result dictionary
        result = {
            "entities": "",
            "question_type": "",
            "relevant_info": "",
            "reasoning": "",
            "limitations": "",
            "answer": ""
        }
        
        # Define patterns to extract each section
        patterns = {
            "entities": r"ENTITIES:(.*?)(?=QUESTION_TYPE:|$)",
            "question_type": r"QUESTION_TYPE:(.*?)(?=RELEVANT_INFO:|$)",
            "relevant_info": r"RELEVANT_INFO:(.*?)(?=REASONING:|$)",
            "reasoning": r"REASONING:(.*?)(?=LIMITATIONS:|$)",
            "limitations": r"LIMITATIONS:(.*?)(?=ANSWER:|$)",
            "answer": r"ANSWER:(.*?)(?=$)"
        }
        
        # Extract each section using regex
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
                
        # If parsing failed, return the full response as the answer
        if not result["answer"]:
            result["answer"] = response.strip()
            
        return result
    
    def verify_factual_accuracy(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Verify factual accuracy of an answer against the source context
        
        Args:
            answer: The answer to verify
            context: The source context
            
        Returns:
            Dict with confidence score and explanation
        """
        verification_prompt = f"""
Given this answer: "{answer}"
And this source context: "{context}"

Verify the factual accuracy of the answer by checking:
1. Whether all claims in the answer are supported by the context
2. Whether there are any contradictions between the answer and context
3. Whether the answer makes unsupported assumptions

For each claim in the answer, explicitly quote the supporting evidence from the context.

Format your response as follows:
SUPPORTED_CLAIMS: [List claims that are directly supported by the context]
UNSUPPORTED_CLAIMS: [List claims that are not supported by the context]
CONTRADICTIONS: [List any contradictions between answer and context]
CONFIDENCE: [Rate as HIGH, MEDIUM, or LOW]
EXPLANATION: [Explain your assessment]
"""

        # Get verification response
        verification = query_ollama(
            query=verification_prompt,
            context="",  # Context is in the prompt
            model=self.model,
            temperature=0.5  # Lower temperature for factual assessment
        )
        
        # Parse verification response
        result = self._parse_verification_response(verification)
        
        return result
    
    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse the verification response"""
        result = {
            "supported_claims": [],
            "unsupported_claims": [],
            "contradictions": [],
            "confidence": "LOW",
            "explanation": ""
        }
        
        # Extract confidence rating
        confidence_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", response, re.IGNORECASE)
        if confidence_match:
            result["confidence"] = confidence_match.group(1).upper()
            
        # Extract explanation
        explanation_match = re.search(r"EXPLANATION:(.*?)(?=$)", response, re.DOTALL)
        if explanation_match:
            result["explanation"] = explanation_match.group(1).strip()
            
        # Extract claims lists
        supported_match = re.search(r"SUPPORTED_CLAIMS:(.*?)(?=UNSUPPORTED_CLAIMS:|$)", response, re.DOTALL)
        if supported_match:
            claims_text = supported_match.group(1).strip()
            # Split by newlines or bullet points
            claims = re.split(r'\n+|\* ', claims_text)
            result["supported_claims"] = [c.strip() for c in claims if c.strip()]
            
        unsupported_match = re.search(r"UNSUPPORTED_CLAIMS:(.*?)(?=CONTRADICTIONS:|$)", response, re.DOTALL)
        if unsupported_match:
            claims_text = unsupported_match.group(1).strip()
            claims = re.split(r'\n+|\* ', claims_text)
            result["unsupported_claims"] = [c.strip() for c in claims if c.strip()]
            
        contradictions_match = re.search(r"CONTRADICTIONS:(.*?)(?=CONFIDENCE:|$)", response, re.DOTALL)
        if contradictions_match:
            claims_text = contradictions_match.group(1).strip()
            claims = re.split(r'\n+|\* ', claims_text)
            result["contradictions"] = [c.strip() for c in claims if c.strip()]
            
        return result
    
    def multi_hop_reasoning(self, query: str, context: str) -> Dict[str, Any]:
        """
        Implement multi-hop reasoning for complex queries by breaking them down
        
        Args:
            query: The complex query to break down
            context: The context information
            
        Returns:
            Dict with sub-questions, their answers, and the final answer
        """
        # First, decompose the query into sub-questions
        decomposition_prompt = f"""
Given this complex question: "{query}"
And this context: "{context[:500]}..." (context continues)

Please break down this question into 2-4 simpler sub-questions that:
1. Can be answered individually
2. When answered in sequence, help solve the original question
3. Progress logically from simpler to more complex understanding

Format your response as a JSON array of strings, each containing one sub-question.
"""

        # Get decomposition response
        decomposition_response = query_ollama(
            query=decomposition_prompt,
            context="",  # Context is in the prompt
            model=self.model,
            temperature=0.7
        )
        
        # Extract the sub-questions (handling both JSON and non-JSON formats)
        sub_questions = self._extract_sub_questions(decomposition_response)
        
        # Answer each sub-question
        sub_answers = []
        for i, sq in enumerate(sub_questions):
            logger.info(f"Answering sub-question {i+1}/{len(sub_questions)}: {sq}")
            
            # Answer with basic reasoning
            answer = query_ollama(
                query=sq,
                context=context,
                model=self.model,
                temperature=self.temperature
            )
            
            sub_answers.append({"question": sq, "answer": answer})
        
        # Format the sub-answers into a context for the final answer
        intermediate_results = "\n\n".join([
            f"Sub-question {i+1}: {sa['question']}\nAnswer: {sa['answer']}"
            for i, sa in enumerate(sub_answers)
        ])
        
        # Generate final answer using the sub-answers
        final_prompt = f"""
Original question: "{query}"

I've broken this down into sub-questions and answered each:

{intermediate_results}

Based on these intermediate answers, please provide a comprehensive answer to the original question.
Your answer should synthesize the information from the sub-questions and present a coherent response.
"""

        final_answer = query_ollama(
            query=final_prompt,
            context=context,
            model=self.model,
            temperature=self.temperature
        )
        
        return {
            "original_question": query,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
            "final_answer": final_answer
        }
    
    def _extract_sub_questions(self, response: str) -> List[str]:
        """Extract sub-questions from the decomposition response"""
        # Try to parse as JSON first
        try:
            # Find a JSON array in the response
            json_match = re.search(r'\[.*\]', response.replace('\n', ' '), re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                questions = json.loads(json_str)
                if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                    return questions
        except json.JSONDecodeError:
            pass
        
        # Fallback: Try to extract numbered questions
        questions = []
        # Look for patterns like "1. Question" or "Question 1:" or "Sub-question 1:"
        patterns = [
            r'\d+\.\s*(.*?)(?=\d+\.|$)',  # "1. Question"
            r'(?:Question|Sub-question)\s+\d+:?\s*(.*?)(?=(?:Question|Sub-question)|$)'  # "Question 1:" 
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.DOTALL)
            for match in matches:
                question = match.group(1).strip()
                if question and len(question) > 10:  # Minimum length to be a valid question
                    questions.append(question)
        
        # If still no questions, split by newlines and look for question marks
        if not questions:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if '?' in line and len(line) > 10:
                    # Remove any leading numbers or bullets
                    cleaned = re.sub(r'^[\d\.\*\-\s]+', '', line).strip()
                    questions.append(cleaned)
        
        return questions

# Test the class if run directly
if __name__ == "__main__":
    # Example usage
    reasoner = ChainOfThoughtReasoner()
    
    test_query = "What were the key factors that led to the 2008 financial crisis?"
    test_context = """
    The 2008 financial crisis was primarily caused by deregulation in the financial industry, which permitted banks to engage in hedge fund trading with derivatives. Banks demanded more mortgages to support the profitable sale of these derivatives, which led to the creation of subprime mortgages. When the Federal Reserve raised interest rates in 2006, these subprime mortgages defaulted, causing the housing bubble to burst. The resulting subprime mortgage crisis led to the banking credit crisis, and ultimately to the Great Recession of 2008. Key institutions like Lehman Brothers collapsed, while others required government bailouts.
    """
    
    print("Testing Chain of Thought reasoning...")
    result = reasoner.answer_with_reasoning(test_query, test_context)
    
    print("\nEntities identified:", result["entities"])
    print("\nQuestion type:", result["question_type"])
    print("\nRelevant info:", result["relevant_info"])
    print("\nReasoning:", result["reasoning"])
    print("\nLimitations:", result["limitations"])
    print("\nFinal answer:", result["answer"])