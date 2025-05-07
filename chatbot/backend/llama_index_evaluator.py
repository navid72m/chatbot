import os
import logging
import json
from typing import List, Dict, Any, Tuple, Optional

from rag_evaluator import RAGEvaluator
from llama_index_integration import LlamaIndexRAG
from llama_cpp_interface import stream_llama_cpp_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaIndexEvaluator:
    """Evaluation framework for comparing LlamaIndex RAG with original RAG system."""
    
    def __init__(self, llm_model_path: str = None, document_store_dir: str = None):
        """
        Initialize the evaluator.
        
        Args:
            llm_model_path: Path to the LLM model for generation
            document_store_dir: Directory to store evaluation datasets
        """
        # Initialize LlamaIndex RAG system
        self.llama_index_rag = LlamaIndexRAG(llm_model_path)
        
        # Directory for evaluation results
        self.eval_results_dir = os.path.expanduser("~/Library/Application Support/Document Chat/evaluation")
        os.makedirs(self.eval_results_dir, exist_ok=True)
        
        # Create evaluator for LlamaIndex
        self.llama_index_evaluator = RAGEvaluator(
            llm_function=self._llama_index_llm_function,
            document_store_dir=document_store_dir
        )
        
        # Create evaluator for original system
        self.original_evaluator = RAGEvaluator(
            llm_function=stream_llama_cpp_response,
            document_store_dir=document_store_dir
        )
    
    def _llama_index_llm_function(self, query: str, context: str) -> str:
        """LLM function that uses LlamaIndex's query engine."""
        # Extract document name from context if available
        document_name = None
        for line in context.split('\n'):
            if line.startswith("Document:"):
                document_name = line.split("Document:")[1].strip()
                break
        
        if not document_name:
            # If no document name found, use the LLM directly
            return stream_llama_cpp_response(query=query, context=context)
        
        # Use LlamaIndex query engine
        try:
            result = self.llama_index_rag.query(query_text=query, document_name=document_name)
            if result["response"] and len(result["response"]) > 10:
                return result["response"]
            else:
                # Fallback to using the LLM with the context
                chunks_text = "\n\n".join([f"Document: {chunk}" for chunk, _ in result["chunks_retrieved"]])
                if not chunks_text:
                    return "I don't have enough information to answer this question."
                
                response = stream_llama_cpp_response(query=query, context=chunks_text)
                return response["response"] if isinstance(response, dict) else response
        
        except Exception as e:
            logger.error(f"Error in LlamaIndex LLM function: {str(e)}")
            # Fallback to original LLM
            return stream_llama_cpp_response(query=query, context=context)
    
    def compare_systems(self, document_name: str, num_questions: int = 10) -> Dict[str, Any]:
        """
        Compare LlamaIndex RAG with the original RAG system.
        
        Args:
            document_name: Name of the document to evaluate on
            num_questions: Number of questions to generate for evaluation
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Starting comparison of RAG systems for document: {document_name}")
        
        # Ensure both systems have processed the document
        file_path = os.path.join(
            os.path.expanduser("~/Library/Application Support/Document Chat/uploads"), 
            document_name
        )
        
        if not os.path.exists(file_path):
            return {"error": f"Document {document_name} not found in uploads directory"}
        
        try:
            # Process with LlamaIndex if not already done
            if not self.llama_index_rag.load_index(document_name):
                logger.info(f"Processing {document_name} with LlamaIndex")
                self.llama_index_rag.process_document(file_path)
        except Exception as e:
            logger.error(f"Error processing document with LlamaIndex: {str(e)}")
            return {"error": f"Error processing document with LlamaIndex: {str(e)}"}
        
        # Check for existing evaluation dataset or create a new one
        dataset = self.original_evaluator.load_dataset(document_name)
        
        if not dataset or len(dataset) < num_questions:
            logger.info(f"Creating evaluation dataset for {document_name}")
            try:
                # Read document text
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        document_text = f.read()
                except Exception as e:
                    logger.error(f"Failed to read document: {str(e)}")
                    return {"error": f"Failed to read document: {str(e)}"}
            
            # Create dataset
            dataset = self.original_evaluator.create_evaluation_dataset(
                document_name=document_name,
                document_text=document_text,
                num_questions=num_questions
            )
        
        # Run evaluation on original system
        logger.info("Evaluating original RAG system")
        original_results = self.original_evaluator.evaluate_rag_system(
            document_name=document_name,
            dataset=dataset
        )
        
        # Run evaluation on LlamaIndex system
        logger.info("Evaluating LlamaIndex RAG system")
        llama_index_results = self.llama_index_evaluator.evaluate_rag_system(
            document_name=document_name,
            dataset=dataset
        )
        
        # Compare results
        comparison = self._compare_results(original_results, llama_index_results)
        
        # Save comparison results
        self._save_comparison(document_name, comparison)
        
        return comparison
    
    def _compare_results(self, original_results: Dict[str, Any], llama_index_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the results of both systems and identify differences."""
        comparison = {
            "document": original_results["document"],
            "num_questions": original_results["num_questions"],
            "original": {
                "overall_score": original_results["overall_score"],
                "retrieval": {
                    "precision_avg": original_results["retrieval"]["precision_avg"],
                    "recall_avg": original_results["retrieval"]["recall_avg"],
                    "mrr_avg": original_results["retrieval"]["reciprocal_rank_avg"]
                },
                "generation": {
                    "semantic_similarity_avg": original_results["generation"]["semantic_similarity_avg"],
                    "exact_match_avg": original_results["generation"]["exact_match_avg"]
                }
            },
            "llama_index": {
                "overall_score": llama_index_results["overall_score"],
                "retrieval": {
                    "precision_avg": llama_index_results["retrieval"]["precision_avg"],
                    "recall_avg": llama_index_results["retrieval"]["recall_avg"],
                    "mrr_avg": llama_index_results["retrieval"]["reciprocal_rank_avg"]
                },
                "generation": {
                    "semantic_similarity_avg": llama_index_results["generation"]["semantic_similarity_avg"],
                    "exact_match_avg": llama_index_results["generation"]["exact_match_avg"]
                }
            },
            "differences": {
                "overall_score": llama_index_results["overall_score"] - original_results["overall_score"],
                "retrieval": {
                    "precision": llama_index_results["retrieval"]["precision_avg"] - original_results["retrieval"]["precision_avg"],
                    "recall": llama_index_results["retrieval"]["recall_avg"] - original_results["retrieval"]["recall_avg"],
                    "mrr": llama_index_results["retrieval"]["reciprocal_rank_avg"] - original_results["retrieval"]["reciprocal_rank_avg"]
                },
                "generation": {
                    "semantic_similarity": llama_index_results["generation"]["semantic_similarity_avg"] - original_results["generation"]["semantic_similarity_avg"],
                    "exact_match": llama_index_results["generation"]["exact_match_avg"] - original_results["generation"]["exact_match_avg"]
                }
            },
            "question_type_comparison": {},
            "question_level_comparison": []
        }
        
        # Compare by question type
        all_types = set(original_results["by_question_type"].keys()) | set(llama_index_results["by_question_type"].keys())
        
        for q_type in all_types:
            if q_type in original_results["by_question_type"] and q_type in llama_index_results["by_question_type"]:
                orig_type = original_results["by_question_type"][q_type]
                llama_type = llama_index_results["by_question_type"][q_type]
                
                comparison["question_type_comparison"][q_type] = {
                    "count": orig_type["count"],
                    "original": {
                        "retrieval_precision": orig_type["retrieval_precision_avg"],
                        "generation_score": orig_type["generation_score_avg"]
                    },
                    "llama_index": {
                        "retrieval_precision": llama_type["retrieval_precision_avg"],
                        "generation_score": llama_type["generation_score_avg"]
                    },
                    "differences": {
                        "retrieval_precision": llama_type["retrieval_precision_avg"] - orig_type["retrieval_precision_avg"],
                        "generation_score": llama_type["generation_score_avg"] - orig_type["generation_score_avg"]
                    }
                }
        
        # Question-level comparison
        for i, (orig_item, llama_item) in enumerate(zip(
            original_results["detailed_results"], 
            llama_index_results["detailed_results"]
        )):
            if orig_item["question"] != llama_item["question"]:
                continue  # Skip if questions don't match
                
            comparison["question_level_comparison"].append({
                "question": orig_item["question"],
                "question_type": orig_item["question_type"],
                "ground_truth": orig_item["ground_truth"],
                "original": {
                    "answer": orig_item["generated_answer"],
                    "retrieval_precision": orig_item["retrieval_metrics"]["precision"],
                    "semantic_similarity": orig_item["generation_metrics"]["semantic_similarity"]
                },
                "llama_index": {
                    "answer": llama_item["generated_answer"],
                    "retrieval_precision": llama_item["retrieval_metrics"]["precision"],
                    "semantic_similarity": llama_item["generation_metrics"]["semantic_similarity"]
                }
            })
        
        # System recommendations
        comparison["recommendations"] = self._generate_comparison_recommendations(comparison)
        
        return comparison
    
    def _generate_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on the comparison results."""
        recommendations = []
        
        # Overall performance
        diff = comparison["differences"]["overall_score"]
        if diff > 0.05:
            recommendations.append(
                f"LlamaIndex outperforms the original system by {diff:.2f} points overall. "
                "Consider using LlamaIndex as your primary RAG system."
            )
        elif diff < -0.05:
            recommendations.append(
                f"The original system outperforms LlamaIndex by {-diff:.2f} points overall. "
                "Consider keeping the original system or diagnosing LlamaIndex issues."
            )
        else:
            recommendations.append(
                "Both systems perform similarly overall. Consider using both and selecting "
                "based on specific document characteristics or query types."
            )
        
        # Retrieval comparison
        retrieval_diff = comparison["differences"]["retrieval"]["precision"]
        if abs(retrieval_diff) > 0.1:
            better_system = "LlamaIndex" if retrieval_diff > 0 else "The original system"
            recommendations.append(
                f"{better_system} has significantly better retrieval precision. "
                f"This suggests it's better at finding relevant information."
            )
        
        # Generation comparison
        gen_diff = comparison["differences"]["generation"]["semantic_similarity"]
        if abs(gen_diff) > 0.1:
            better_system = "LlamaIndex" if gen_diff > 0 else "The original system"
            recommendations.append(
                f"{better_system} produces answers with higher semantic similarity to ground truth. "
                f"This suggests better answer quality."
            )
        
        # Question type analysis
        for q_type, stats in comparison.get("question_type_comparison", {}).items():
            diff = stats["differences"]["generation_score"]
            if abs(diff) > 0.15:  # Significant difference
                better_system = "LlamaIndex" if diff > 0 else "The original system"
                recommendations.append(
                    f"{better_system} performs significantly better on '{q_type}' questions. "
                    f"Consider using it specifically for these types of queries."
                )
        
        return recommendations
    
    def _save_comparison(self, document_name: str, comparison: Dict[str, Any]) -> None:
        """Save comparison results to disk."""
        clean_name = ''.join(c if c.isalnum() else '_' for c in document_name)
        filepath = os.path.join(self.eval_results_dir, f"{clean_name}_comparison.json")
        
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Saved comparison results to {filepath}")
    
    def load_comparison(self, document_name: str) -> Optional[Dict[str, Any]]:
        """Load saved comparison results."""
        clean_name = ''.join(c if c.isalnum() else '_' for c in document_name)
        filepath = os.path.join(self.eval_results_dir, f"{clean_name}_comparison.json")
        
        if not os.path.exists(filepath):
            logger.warning(f"No comparison results found for {document_name}")
            return None
        
        with open(filepath, 'r') as f:
            comparison = json.load(f)
        
        logger.info(f"Loaded comparison results for {document_name}")
        return comparison

def integrate_evaluation_framework(app, model_path=None):
    """
    Integrate the evaluation framework into the FastAPI app.
    
    Args:
        app: FastAPI app instance
        model_path: Path to LLM model
    """
    from fastapi import HTTPException
    
    # Initialize evaluator
    evaluator = LlamaIndexEvaluator(model_path)
    
    # Store as app state
    app.state.llama_index_evaluator = evaluator
    
    @app.post("/evaluate/compare")
    async def compare_rag_systems(request: dict):
        try:
            document_name = request.get("document")
            if not document_name:
                document_name = getattr(app.state, "current_document", None)
                if not document_name:
                    raise HTTPException(status_code=400, detail="No document specified")
            
            # Optional parameters
            num_questions = request.get("num_questions", 10)
            force_new = request.get("force_new", False)
            
            # Check for existing comparison
            if not force_new:
                existing = evaluator.load_comparison(document_name)
                if existing:
                    return {
                        "success": True,
                        "document": document_name,
                        "comparison": existing,
                        "source": "cached"
                    }
            
            # Run comparison
            comparison = evaluator.compare_systems(
                document_name=document_name,
                num_questions=num_questions
            )
            
            if "error" in comparison:
                raise HTTPException(status_code=500, detail=comparison["error"])
            
            return {
                "success": True,
                "document": document_name,
                "comparison": comparison,
                "source": "new"
            }
            
        except Exception as e:
            logger.error(f"Error comparing RAG systems: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error comparing RAG systems: {str(e)}")
    
    @app.get("/evaluate/comparisons")
    async def list_comparisons():
        """List all available system comparisons."""
        try:
            eval_dir = os.path.expanduser("~/Library/Application Support/Document Chat/evaluation")
            if not os.path.exists(eval_dir):
                return {"comparisons": []}
                
            files = [f for f in os.listdir(eval_dir) if f.endswith("_comparison.json")]
            comparisons = []
            
            for file in files:
                try:
                    with open(os.path.join(eval_dir, file), 'r') as f:
                        data = json.load(f)
                        comparisons.append({
                            "document": data.get("document", file.replace("_comparison.json", "")),
                            "date": os.path.getmtime(os.path.join(eval_dir, file)),
                            "original_score": data.get("original", {}).get("overall_score", 0),
                            "llama_index_score": data.get("llama_index", {}).get("overall_score", 0)
                        })
                except Exception as e:
                    logger.error(f"Error reading comparison file {file}: {str(e)}")
            
            return {"comparisons": sorted(comparisons, key=lambda x: x["date"], reverse=True)}
            
        except Exception as e:
            logger.error(f"Error listing comparisons: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error listing comparisons: {str(e)}")
    
    return evaluator