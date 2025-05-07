import sys
import os
import logging
from fastapi import APIRouter
import json

# Import your existing components
from vector_store import VectorStore
from llama_cpp_interface import stream_llama_cpp_response
from document_processor import process_document, chunk_document

# Import the RAG evaluator
from rag_evaluator import RAGEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_evaluation")

# Create a router for evaluation endpoints
evaluation_router = APIRouter(prefix="/evaluate", tags=["evaluation"])

def integrate_evaluation_framework(app, vector_store, document_storage_path):
    """
    Integrate the RAG evaluation framework with your existing FastAPI app.
    
    Args:
        app: FastAPI app instance
        vector_store: Your VectorStore instance
        document_storage_path: Path where uploaded documents are stored
    """
    # Define the LLM function that will be used for evaluation
    def evaluation_llm_function(query, context):
        """Wrapper around your existing LLM function for evaluation."""
        return stream_llama_cpp_response(
            query=query,
            context=context,
            model="mistral",  # Use your default model
            temperature=0.3   # Lower temperature for more deterministic answers
        )
    
    # Create the evaluator
    evaluator = RAGEvaluator(
        vector_store=vector_store,
        llm_function=evaluation_llm_function,
        document_store_dir=os.path.join(document_storage_path, "evaluation")
    )
    
    # Store the evaluator in app state for access from endpoints
    app.state.rag_evaluator = evaluator
    
    # Register evaluation endpoints
    app.include_router(evaluation_router)
    
    logger.info("RAG evaluation framework integrated successfully")
    return evaluator

# Function to get document text from your storage
def get_document_text(document_name, upload_dir):
    """Get the full text of a document from your storage."""
    try:
        # Find the document file path
        file_path = os.path.join(upload_dir, document_name)
        
        if not os.path.exists(file_path):
            logger.warning(f"Document file not found: {file_path}")
            return None
        
        # Process the document to get text
        document_text = process_document(file_path)
        return document_text
    except Exception as e:
        logger.error(f"Error retrieving document text: {str(e)}")
        return None

# Evaluation endpoints
@evaluation_router.post("/create-dataset")
async def create_evaluation_dataset(request: dict):
    """Create an evaluation dataset for a document."""
    from fastapi import HTTPException, Request
    
    # Access the request object to get app state
    request_obj = Request.scope["fastapi"]
    evaluator = request_obj.app.state.rag_evaluator
    
    try:
        # Get document name
        document_name = request.get("document")
        if not document_name:
            document_name = getattr(request_obj.app.state, "current_document", None)
            if not document_name:
                raise HTTPException(status_code=400, detail="No document specified")
        
        # Get document text
        upload_dir = os.path.expanduser("~/Library/Application Support/Document Chat/uploads")
        document_text = get_document_text(document_name, upload_dir)
        
        if not document_text:
            raise HTTPException(status_code=404, detail=f"Document {document_name} not found or could not be processed")
        
        # Optional parameters
        num_questions = request.get("num_questions", 20)
        question_types = request.get("question_types", ["factoid", "reasoning", "multi_hop", "unanswerable"])
        
        # Create dataset
        dataset = evaluator.create_evaluation_dataset(
            document_name=document_name,
            document_text=document_text,
            num_questions=num_questions,
            question_types=question_types
        )
        
        # Return success with dataset stats
        return {
            "success": True,
            "document": document_name,
            "dataset_size": len(dataset),
            "question_types": {q_type: sum(1 for q in dataset if q["type"] == q_type) 
                              for q_type in set(q["type"] for q in dataset)}
        }
    except Exception as e:
        logger.error(f"Error creating evaluation dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating evaluation dataset: {str(e)}")

@evaluation_router.post("/run")
async def run_evaluation(request: dict):
    """Run evaluation on a document using its dataset."""
    from fastapi import HTTPException, Request
    
    # Access the request object to get app state
    request_obj = Request.scope["fastapi"]
    evaluator = request_obj.app.state.rag_evaluator
    
    try:
        # Get document name
        document_name = request.get("document")
        if not document_name:
            document_name = getattr(request_obj.app.state, "current_document", None)
            if not document_name:
                raise HTTPException(status_code=400, detail="No document specified")
        
        # Check if dataset exists
        dataset = evaluator.load_dataset(document_name)
        if not dataset:
            raise HTTPException(
                status_code=404, 
                detail=f"No evaluation dataset found for {document_name}. Please create one first."
            )
        
        # Optional parameters
        k = request.get("k", 5)  # Number of chunks to retrieve
        
        # Run evaluation
        results = evaluator.evaluate_rag_system(
            document_name=document_name,
            dataset=dataset,
            k=k
        )
        
        # Generate report
        report = evaluator.generate_evaluation_report(results)
        
        # Save results to file for reference
        results_dir = os.path.join(os.path.expanduser("~/Library/Application Support/Document Chat/evaluation"), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"{document_name}_{timestamp}_results.json")
        
        with open(results_file, 'w') as f:
            json.dump({
                "document": document_name,
                "timestamp": timestamp,
                "report": report,
                "detailed_results": results.get("detailed_results", [])
            }, f, indent=2)
        
        return {
            "success": True,
            "document": document_name,
            "report": report,
            "results_file": results_file
        }
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running evaluation: {str(e)}")

@evaluation_router.get("/datasets")
async def list_evaluation_datasets():
    """List all available evaluation datasets."""
    from fastapi import HTTPException, Request
    
    # Access the request object to get app state
    request_obj = Request.scope["fastapi"]
    evaluator = request_obj.app.state.rag_evaluator
    
    try:
        # List all dataset files in the evaluation directory
        datasets_dir = evaluator.document_store_dir
        dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith("_eval_dataset.json")]
        
        datasets = []
        for filename in dataset_files:
            # Extract document name from filename
            doc_name = filename.replace("_eval_dataset.json", "")
            
            # Get file stats
            file_path = os.path.join(datasets_dir, filename)
            file_size = os.path.getsize(file_path)
            file_date = os.path.getmtime(file_path)
            
            # Load dataset to get question count
            try:
                with open(file_path, 'r') as f:
                    dataset = json.load(f)
                    question_count = len(dataset)
                    question_types = {q_type: sum(1 for q in dataset if q.get("type") == q_type) 
                                    for q_type in set(q.get("type", "unknown") for q in dataset)}
            except:
                question_count = 0
                question_types = {}
            
            datasets.append({
                "document": doc_name,
                "file": filename,
                "size_bytes": file_size,
                "created": datetime.datetime.fromtimestamp(file_date).isoformat(),
                "question_count": question_count,
                "question_types": question_types
            })
        
        return {
            "success": True,
            "datasets": datasets
        }
    except Exception as e:
        logger.error(f"Error listing evaluation datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing evaluation datasets: {str(e)}")

# Example: Adding evaluation to your existing app
def add_evaluation_to_app():
    """Example of how to add the evaluation framework to your app."""
    from fastapi import FastAPI
    from vector_store import VectorStore
    
    # Assume your app is already set up
    app = FastAPI(title="Document Chat Backend")
    vector_store = VectorStore()
    upload_dir = os.path.expanduser("~/Library/Application Support/Document Chat/uploads")
    
    # Integrate evaluation framework
    integrate_evaluation_framework(
        app=app,
        vector_store=vector_store,
        document_storage_path=os.path.expanduser("~/Library/Application Support/Document Chat")
    )
    
    return app

# If this script is run directly, demonstrate the evaluation
if __name__ == "__main__":
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(description='RAG Evaluation Tool')
    parser.add_argument('--document', type=str, help='Document name to evaluate')
    parser.add_argument('--questions', type=int, default=20, help='Number of questions to generate')
    parser.add_argument('--create-dataset', action='store_true', help='Create evaluation dataset')
    parser.add_argument('--run-evaluation', action='store_true', help='Run evaluation')
    
    args = parser.parse_args()
    
    if not args.document:
        print("Please specify a document name with --document")
        sys.exit(1)
    
    # Initialize components
    vector_store = VectorStore()
    upload_dir = os.path.expanduser("~/Library/Application Support/Document Chat/uploads")
    
    # Define the LLM function
    def evaluation_llm_function(query, context):
        return stream_llama_cpp_response(
            query=query,
            context=context,
            model="mistral",
            temperature=0.3
        )
    
    # Create evaluator
    evaluator = RAGEvaluator(
        vector_store=vector_store,
        llm_function=evaluation_llm_function,
        document_store_dir=os.path.join(upload_dir, "../evaluation")
    )
    
    # Process commands
    if args.create_dataset:
        # Get document text
        document_text = get_document_text(args.document, upload_dir)
        if not document_text:
            print(f"Document {args.document} not found or could not be processed")
            sys.exit(1)
        
        # Create dataset
        dataset = evaluator.create_evaluation_dataset(
            document_name=args.document,
            document_text=document_text,
            num_questions=args.questions
        )
        
        print(f"Created evaluation dataset with {len(dataset)} questions")
        question_types = {q_type: sum(1 for q in dataset if q["type"] == q_type) 
                          for q_type in set(q["type"] for q in dataset)}
        print(f"Question types: {question_types}")
    
    if args.run_evaluation:
        # Load dataset
        dataset = evaluator.load_dataset(args.document)
        if not dataset:
            print(f"No evaluation dataset found for {args.document}. Please create one first.")
            sys.exit(1)
        
        print(f"Running evaluation on {len(dataset)} questions...")
        
        # Run evaluation
        results = evaluator.evaluate_rag_system(
            document_name=args.document,
            dataset=dataset
        )
        
        # Generate report
        report = evaluator.generate_evaluation_report(results)
        
        # Print summary
        print("\n--- EVALUATION SUMMARY ---")
        print(f"Document: {args.document}")
        print(f"Questions: {results['num_questions']}")
        print(f"Overall Score: {results['overall_score']:.2f} / 1.00")
        print(f"Retrieval Precision: {results['retrieval']['precision_avg']:.2f}")
        print(f"Answer Quality: {results['generation']['semantic_similarity_avg']:.2f}")
        
        # Print recommendations
        print("\n--- RECOMMENDATIONS ---")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")
        
        # Save results to file
        results_dir = os.path.join(upload_dir, "../evaluation/results")
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"{args.document}_{timestamp}_results.json")
        
        with open(results_file, 'w') as f:
            json.dump({
                "document": args.document,
                "timestamp": timestamp,
                "report": report,
                "detailed_results": results.get("detailed_results", [])
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")