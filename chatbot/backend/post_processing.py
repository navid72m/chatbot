# post_processing.py
import re

def post_process_answer(answer: str) -> str:
    """Remove reasoning steps and make answers direct."""
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