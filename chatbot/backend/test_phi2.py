# test_phi2_rag.py
import logging
from vector_store import VectorStore
from bitnet_wrapper import BitNetModel
from enhanced_rag import EnhancedRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phi2_test")

def test_phi2():
    """Test Phi-2 model directly."""
    # Initialize model
    phi2 = BitNetModel(model_name="microsoft/phi-2")
    
    # Test generation
    prompt = "What is machine learning?"
    logger.info(f"Testing Phi-2 with prompt: {prompt}")
    
    response = phi2(prompt)
    logger.info(f"Phi-2 response: {response['choices'][0]['text']}")
    logger.info(f"Token usage: {response['usage']}")

def test_phi2_rag():
    """Test Phi-2 with RAG."""
    # Initialize components
    vector_store = VectorStore()
    enhanced_rag = EnhancedRAG(
        vector_store=vector_store,
        use_bitnet=True,
        model_name="microsoft/phi-2"
    )
    
    # Test a query
    document_name = "/Users/seyednavidmirnourilangeroudi/Downloads/resume navid.pdf"
    query = "What are Navid's skills?"
    
    logger.info(f"Testing Phi-2 RAG with query: {query}")
    result = enhanced_rag.query(
        query=query,
        document_name=document_name
    )
    
    logger.info(f"Question type: {result['question_type']}")
    logger.info(f"Chunks retrieved: {result['chunks_retrieved']}")
    logger.info(f"Response: {result['response']}")

if __name__ == "__main__":
    logger.info("Testing Phi-2 standalone...")
    test_phi2()
    
    logger.info("\nTesting Phi-2 with RAG...")
    test_phi2_rag()