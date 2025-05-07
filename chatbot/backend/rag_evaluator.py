import os
import json
import logging
import random
import sys
from typing import List, Dict, Any, Tuple, Optional
import time
from sentence_transformers import util
# from vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_evaluator")

class RAGEvaluator:
    """Framework for evaluating RAG (Retrieval-Augmented Generation) systems."""
    
    def __init__(self,  llm_function,  document_store_dir=None):
        """
        Initialize the RAG evaluator.
        
        Args:
            vector_store: The vector store instance used in your RAG system
            llm_function: Function to generate answers using an LLM
            document_store_dir: Directory to store evaluation datasets
        """
        # self.vector_store = vector_store
        self.llm_function = llm_function
        self.document_store_dir = document_store_dir or os.path.expanduser("~/rag_evaluation")
        os.makedirs(self.document_store_dir, exist_ok=True)
        # self.enhanced_rag = enhanced_rag
        # Cache for generated datasets
        self.datasets = {}
    
    def create_evaluation_dataset(
        self, 
        document_name: str, 
        document_text: str, 
        num_questions: int = 20,
        question_types: List[str] = None,
        save_to_disk: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Create an evaluation dataset from a document.
        
        Args:
            document_name: Name of the document
            document_text: Full text content of the document
            num_questions: Number of questions to generate
            question_types: Types of questions to generate (factoid, reasoning, etc.)
            save_to_disk: Whether to save the dataset to disk
            
        Returns:
            List of evaluation items with questions and answers
        """
        logger.info(f"Creating evaluation dataset for document: {document_name}")
        
        # Default question types if none specified
        if question_types is None:
            question_types = ["factoid", "reasoning", "multi_hop", "unanswerable"]
        
        # Split document into chunks (reuse existing chunking function)
        from document_processor import chunk_document
        chunks = chunk_document(document_text, chunk_size=200, chunk_overlap=50)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # Assign chunk IDs for reference
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict) and "content" in chunk:
                chunk["chunk_id"] = i
            else:
                # If chunks are strings, convert to dict with content and chunk_id
                chunks[i] = {"content": chunk, "chunk_id": i}

        logger.info(f"Chunks: {chunks}")
        
        # Calculate how many questions to generate per type
        questions_per_type = max(1, num_questions // len(question_types))
        
        # Generate questions for each type
        dataset = []
        
        for q_type in question_types:
            type_questions = self._generate_questions_by_type(
                document_name=document_name,
                chunks=chunks,
                question_type=q_type,
                num_questions=questions_per_type
            )
            dataset.extend(type_questions)
        
        logger.info(f"Dataset: {dataset}")
        # Ensure we have enough questions
        while len(dataset) < num_questions and chunks:
            # Generate additional questions if needed
            random_type = random.choice(question_types)
            additional_question = self._generate_questions_by_type(
                document_name=document_name,
                chunks=chunks,
                question_type=random_type,
                num_questions=1
            )
            if additional_question:
                dataset.extend(additional_question)
        
        logger.info(f"Generated {len(dataset)} questions for evaluation")
        
        # Save dataset if requested
        if save_to_disk:
            self._save_dataset(document_name, dataset)
        
        # Cache the dataset
        self.datasets[document_name] = dataset
        
        return dataset
    
    def _generate_questions_by_type(
        self, 
        document_name: str,
        chunks: List[Dict[str, Any]], 
        question_type: str,
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Generate questions of a specific type from document chunks."""
        questions = []
        
        # Different strategies based on question type
        if question_type == "factoid":
            # Factoid questions (who, what, when, where)
            questions = self._generate_factoid_questions(document_name, chunks, num_questions)
        
        elif question_type == "reasoning":
            # Complex reasoning questions requiring analysis
            questions = self._generate_reasoning_questions(document_name, chunks, num_questions)
        
        elif question_type == "multi_hop":
            # Questions requiring information from multiple chunks
            questions = self._generate_multi_hop_questions(document_name, chunks, num_questions)
        
        elif question_type == "unanswerable":
            # Questions that can't be answered from the document
            questions = self._generate_unanswerable_questions(document_name, chunks, num_questions)
        
        # Add question type to each item
        for q in questions:
            q["type"] = question_type
        
        return questions
    
    def _generate_factoid_questions(
        self, 
        document_name: str,
        chunks: List[Dict[str, Any]], 
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Generate factoid questions from document chunks."""
        questions = []
        
        # Get a sample of chunks to use
        sample_chunks = random.sample(chunks, min(num_questions, len(chunks)))
        
        for chunk in sample_chunks:
            # Extract chunk content and ID
            chunk_content = chunk["content"] if isinstance(chunk, dict) else chunk
            chunk_id = chunk.get("chunk_id", 0) if isinstance(chunk, dict) else 0
            
            # Generate question using LLM
            prompt = f"""
            I'm going to show you a passage from a document. Create ONE specific factoid question (who, what, when, where)
            that can be answered directly using information in this passage. The question should be answerable with a 
            specific fact from the text.
            
            PASSAGE:
            {chunk_content}
            
            TASK:
            1. Generate ONE factoid question that can be answered from the passage.
            2. Provide the answer to this question, citing the specific part of the text that contains the answer.
            
            Format your response as:
            QUESTION: [your factoid question]
            ANSWER: [the answer to the question]
            EVIDENCE: [the specific text that answers the question]
            """
            
            response = self.llm_function(query="Generate factoid question", context=prompt)
            
            # Parse response
            question, answer, evidence = self._parse_question_generation_response(response)
            
            if question and answer:
                questions.append({
                    "document": document_name,
                    "question": question,
                    "ground_truth": answer,
                    "evidence": evidence,
                    "relevant_chunks": [chunk_id],
                    "type": "factoid"
                })
            
            # Exit if we have enough questions
            if len(questions) >= num_questions:
                break
        
        return questions
    
    def _generate_reasoning_questions(
        self, 
        document_name: str,
        chunks: List[Any], 
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Generate reasoning questions that require analysis and inference."""
        questions = []
        
        # Don't extract content yet - keep the original Document objects
        # Just create pairs of chunks to use
        combined_chunks = []
        
        for i in range(0, len(chunks), 2):
            if i + 1 < len(chunks):
                # Store pairs of document objects
                combined_chunks.append({
                    "first_chunk": chunks[i],
                    "second_chunk": chunks[i+1],
                    "chunk_id": [i, i+1]
                })
            else:
                # Just use a single chunk if at the end
                combined_chunks.append({
                    "first_chunk": chunks[i],
                    "chunk_id": [i]
                })
        
        # Sample from combined chunks
        sample_size = min(num_questions, len(combined_chunks))
        sample_chunks = random.sample(combined_chunks, sample_size) if sample_size > 0 else []
        
        for chunk_pair in sample_chunks:
            # Now extract and combine content only when needed for the prompt
            prompt_text = ""
            
            # First chunk
            first_chunk = chunk_pair["first_chunk"]
            if hasattr(first_chunk, "page_content"):
                prompt_text += first_chunk.page_content
            elif hasattr(first_chunk, "text"):
                prompt_text += first_chunk.text
            else:
                prompt_text += str(first_chunk)
            
            # Add second chunk if it exists
            if "second_chunk" in chunk_pair:
                prompt_text += " "  # Add space between chunks
                second_chunk = chunk_pair["second_chunk"]
                if hasattr(second_chunk, "page_content"):
                    prompt_text += second_chunk.page_content
                elif hasattr(second_chunk, "text"):
                    prompt_text += second_chunk.text
                else:
                    prompt_text += str(second_chunk)
            
            # Generate reasoning question using LLM
            prompt = f"""
            I'm going to show you a passage from a document. Create ONE reasoning question that requires analysis
            or inference based on information in this passage. The question should not be answerable with a simple fact
            but require deeper understanding or connecting multiple pieces of information.
            
            PASSAGE:
            {prompt_text}
            
            TASK:
            1. Generate ONE reasoning question that requires analysis of the passage.
            2. Provide the answer to this question, explaining the reasoning process.
            3. Identify the specific parts of the text that support this reasoning.
            
            Format your response as:
            QUESTION: [your reasoning question]
            ANSWER: [detailed answer with reasoning]
            EVIDENCE: [parts of the text that support this reasoning]
            """
            
            response = self.llm_function(query="Generate reasoning question", context=prompt)
            
            # Parse response
            question, answer, evidence = self._parse_question_generation_response(response)
            
            if question and answer:
                questions.append({
                    "document": document_name,
                    "question": question,
                    "ground_truth": answer,
                    "evidence": evidence,
                    "relevant_chunks": chunk_pair["chunk_id"],
                    "type": "reasoning"
                })
            
            # Exit if we have enough questions
            if len(questions) >= num_questions:
                break
        
        return questions
    
    def _generate_multi_hop_questions(
        self, 
        document_name: str,
        chunks: List[Dict[str, Any]], 
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Generate questions requiring information from multiple non-consecutive chunks."""
        questions = []
        
        # Need at least 3 chunks for multi-hop questions
        if len(chunks) < 3:
            return questions
        
        # Try to generate up to the requested number of questions
        attempts = 0
        max_attempts = num_questions * 3  # Allow some failures
        
        while len(questions) < num_questions and attempts < max_attempts:
            attempts += 1
            
            # Select 2-3 random chunks
            num_chunks_to_use = random.randint(2, min(3, len(chunks)))
            selected_chunks = random.sample(chunks, num_chunks_to_use)
            
            # Extract content and IDs
            contents = [c["content"] if isinstance(c, dict) else c for c in selected_chunks]
            chunk_ids = [c.get("chunk_id", i) for i, c in enumerate(selected_chunks)]
            
            # Prepare information from each chunk
            chunk_info = ""
            for i, content in enumerate(contents):
                chunk_info += f"CHUNK {i+1}:\n{content}\n\n"
            
            # Generate multi-hop question using LLM
            prompt = f"""
            I'm going to show you multiple separate chunks from a document. Create ONE question that requires connecting
            information across these different chunks. The question should not be answerable from any single chunk alone.
            
            {chunk_info}
            
            TASK:
            1. Generate ONE question that requires information from at least 2 different chunks to answer correctly.
            2. Provide the answer to this question, explaining which information comes from which chunk.
            
            Format your response as:
            QUESTION: [your multi-hop question]
            ANSWER: [answer that connects information from multiple chunks]
            EVIDENCE: [specify which parts from which chunks support the answer]
            """
            
            response = self.llm_function(query="Generate multi-hop question", context=prompt)
            
            # Parse response
            question, answer, evidence = self._parse_question_generation_response(response)
            
            # Validate that it's truly multi-hop
            if question and answer and "CHUNK" in evidence:
                questions.append({
                    "document": document_name,
                    "question": question,
                    "ground_truth": answer,
                    "evidence": evidence,
                    "relevant_chunks": chunk_ids,
                    "type": "multi_hop"
                })
        
        return questions
    
    def _generate_unanswerable_questions(
        self, 
        document_name: str,
        chunks: List[Dict[str, Any]], 
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Generate questions that cannot be answered from the document."""
        questions = []
        
        # Sample chunks to base unanswerable questions on
        sample_chunks = random.sample(chunks, min(num_questions, len(chunks)))
        
        for chunk in sample_chunks:
            # Extract chunk content
            chunk_content = chunk["content"] if isinstance(chunk, dict) else chunk
            
            # Generate unanswerable question using LLM
            prompt = f"""
            I'm going to show you a passage from a document. Create ONE question that is related to the topic of the passage
            but CANNOT be answered using the information provided. The question should seem relevant but require information
            that is not present in the text.
            
            PASSAGE:
            {chunk_content}
            
            TASK:
            1. Generate ONE question that cannot be answered from the passage but is on the same topic.
            2. Explain why this question cannot be answered with the given information.
            
            Format your response as:
            QUESTION: [your unanswerable question]
            EXPLANATION: [why this cannot be answered from the passage]
            """
            
            response = self.llm_function(query="Generate unanswerable question", context=prompt)
            
            # Parse response - different format for unanswerable
            question = None
            explanation = None
            
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("QUESTION:"):
                    question = line[len("QUESTION:"):].strip()
                elif line.startswith("EXPLANATION:"):
                    explanation = line[len("EXPLANATION:"):].strip()
            
            if question:
                questions.append({
                    "document": document_name,
                    "question": question,
                    "ground_truth": "This question cannot be answered based on the provided document.",
                    "evidence": explanation if explanation else "No relevant information in the document.",
                    "relevant_chunks": [],  # No relevant chunks for unanswerable
                    "type": "unanswerable"
                })
            
            # Exit if we have enough questions
            if len(questions) >= num_questions:
                break
        
        return questions
    
    def _parse_question_generation_response(self, response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse the LLM response to extract question, answer, and evidence."""
        question = None
        answer = None
        evidence = None
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("QUESTION:"):
                question = line[len("QUESTION:"):].strip()
            elif line.startswith("ANSWER:"):
                answer = line[len("ANSWER:"):].strip()
            elif line.startswith("EVIDENCE:"):
                evidence = line[len("EVIDENCE:"):].strip()
        
        return question, answer, evidence
    
    def _save_dataset(self, document_name: str, dataset: List[Dict[str, Any]]) -> None:
        """Save evaluation dataset to disk."""
        # Clean filename
        clean_name = ''.join(c if c.isalnum() else '_' for c in document_name)
        filepath = os.path.join(self.document_store_dir, f"{clean_name}_eval_dataset.json")
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved evaluation dataset to {filepath}")
    
    def load_dataset(self, document_name: str) -> List[Dict[str, Any]]:
        """Load evaluation dataset from disk."""
        # Check if already in memory
        if document_name in self.datasets:
            return self.datasets[document_name]
        
        # Clean filename
        clean_name = ''.join(c if c.isalnum() else '_' for c in document_name)
        filepath = os.path.join(self.document_store_dir, f"{clean_name}_eval_dataset.json")
        
        if not os.path.exists(filepath):
            logger.warning(f"No evaluation dataset found for {document_name}")
            return []
        
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        
        # Cache dataset
        self.datasets[document_name] = dataset
        
        logger.info(f"Loaded evaluation dataset for {document_name} with {len(dataset)} questions")
        return dataset
    def simple_evaluate_system(self, document_name: str, dataset: List[Dict[str, Any]] = None, k: int = 5):
        """A simplified, robust evaluation function that avoids key errors."""
        # Load dataset if not provided
        if dataset is None:
            dataset = self.load_dataset(document_name)
            if not dataset:
                logger.error(f"No evaluation dataset found for {document_name}")
                return self._create_empty_results(document_name)
        
        logger.info(f"Evaluating RAG system on {len(dataset)} questions for {document_name}")
        
        # Initialize result structure with all required fields
        results = {
            "document": document_name,
            "num_questions": len(dataset),
            "retrieval": {
                "precision": [],
                "recall": [],
                "reciprocal_rank": [],
                # Pre-initialize avg fields to avoid modification during iteration
                "precision_avg": 0.0,
                "recall_avg": 0.0,
                "reciprocal_rank_avg": 0.0
            },
            "generation": {
                "exact_match": [],
                "semantic_similarity": [],
                "contains_answer": [],
                # Pre-initialize avg fields to avoid modification during iteration
                "exact_match_avg": 0.0, 
                "semantic_similarity_avg": 0.0,
                "contains_answer_avg": 0.0
            },
            "by_question_type": {},
            "detailed_results": []
        }
        
        # [Rest of evaluation code as before]
        
        # Calculate averages AFTER all metrics have been collected
        # This is a separate step to avoid changing dict size during iteration
        for category in ["retrieval", "generation"]:
            for metric in list(results[category].keys()):  # Use list() to create a copy
                if not metric.endswith("_avg") and results[category][metric]:
                    results[category][f"{metric}_avg"] = sum(results[category][metric]) / len(results[category][metric])
        
        # Calculate overall score
        results["overall_score"] = (
            results["retrieval"]["precision_avg"] * 0.5 + 
            results["generation"]["semantic_similarity_avg"] * 0.5
        )
        
        logger.info(f"Evaluation complete. Overall score: {results['overall_score']:.2f}")
        return results
    def evaluate_rag_system(
        self, 
        document_name: str,
        dataset: List[Dict[str, Any]] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG system on a given dataset.
        
        Args:
            document_name: Name of the document
            dataset: Evaluation dataset (will load from disk if None)
            k: Number of chunks to retrieve for each question
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load dataset if not provided
        if dataset is None:
            dataset = self.load_dataset(document_name)
            if not dataset:
                logger.error(f"No evaluation dataset found for {document_name}")
                return {"error": "No evaluation dataset found"}
        
        logger.info(f"Evaluating RAG system on {len(dataset)} questions for {document_name}")
        
        # Prepare results structure
        results = {
            "document": document_name,
            "num_questions": len(dataset),
            "retrieval": {
                "precision": [],
                "recall": [],
                "reciprocal_rank": []
            },
            "generation": {
                "exact_match": [],
                "semantic_similarity": [],
                "contains_answer": []
            },
            "by_question_type": {},
            "detailed_results": []
        }
        
        # Create structure for each question type
        question_types = set(item["type"] for item in dataset)
        for q_type in question_types:
            results["by_question_type"][q_type] = {
                "count": 0,
                "retrieval_precision": [],
                "generation_score": []
            }
        
        # Process each question
        for item in dataset:
            question = item["question"]
            ground_truth = item["ground_truth"]
            relevant_chunk_ids = item.get("relevant_chunks", [])
            question_type = item.get("type", "general")
            
            # Update type counter
            results["by_question_type"][question_type]["count"] += 1
            
            # Measure retrieval time
            retrieval_start = time.time()
            
            # Retrieval step
            # retrieved_chunks = self.vector_store.search(
            #     query=question, 
            #     k=k, 
            #     filter={"source": document_name}
            # )
            # from hybrid_search import hybrid_search
            # retrieved_chunks = hybrid_search(
            #     vector_store=self.vector_store,
            #     query=question, 
            #     k=k, 
            #     document_name=document_name

            # )
            from document_processor_patched import query_index, query_index_with_context
            # retrieved_context = query_index_with_context(question, k)
            # retrieved_chunks,_ = process_and_index_file(document_name)
            contexts = query_index_with_context(question, k)
            retrieved_chunks = []
            for doc in contexts:
                # doc, score = context
                retrieved_chunks.append(doc)
            # retrieved_chunks = chunks
            # _=self.vector_store.add_document(document_name, chunks)
            # retrieved_chunks = self.enhanced_rag._hybrid_search(
            #     query=question,
            #     k=k,
            #     document_name=document_name
            # )
            # logger.info(f"retrieved_chunks: {retrieved_chunks}")
            
            retrieval_time = time.time() - retrieval_start
            
            # Get chunk IDs from retrieved chunks
            retrieved_chunk_ids = []
            for chunk in retrieved_chunks:
                # Handle different chunk metadata formats
                if hasattr(chunk, "metadata") and chunk.metadata:
                    chunk_id = chunk.metadata.get("chunk_id")
                    if chunk_id is not None:
                        retrieved_chunk_ids.append(chunk_id)
                elif isinstance(chunk, dict):
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id is not None:
                        retrieved_chunk_ids.append(chunk_id)
            
            # Ensure all IDs are properly formatted
            retrieved_chunk_ids = [int(cid) if isinstance(cid, str) and cid.isdigit() else cid 
                                  for cid in retrieved_chunk_ids]
            
            # For unanswerable questions, having no relevant chunks is correct
            if question_type == "unanswerable":
                # For unanswerable questions, precision is 1.0 if nothing relevant is retrieved
                # (since no chunks should contain the answer)
                precision = 1.0 if not retrieved_chunk_ids else 0.0
                recall = 1.0  # Recall is always 1.0 for unanswerable (there are no relevant chunks to retrieve)
                mrr = 1.0  # MRR also 1.0 for the same reason
            else:
                # Calculate retrieval metrics for answerable questions
                if not relevant_chunk_ids or not retrieved_chunk_ids:
                    precision = 0.0
                    recall = 0.0
                    mrr = 0.0
                else:
                    # Convert to sets for intersection
                    retrieved_set = set(retrieved_chunk_ids)
                    relevant_set = set(relevant_chunk_ids)
                    
                    # Calculate precision and recall
                    precision = len(retrieved_set.intersection(relevant_set)) / len(retrieved_set) if retrieved_set else 0.0
                    recall = len(retrieved_set.intersection(relevant_set)) / len(relevant_set) if relevant_set else 0.0
                    
                    # Calculate Mean Reciprocal Rank (MRR)
                    # Find the rank of the first relevant chunk
                    mrr = 0.0
                    for i, chunk_id in enumerate(retrieved_chunk_ids):
                        if chunk_id in relevant_set:
                            mrr = 1.0 / (i + 1)  # Rank is 1-indexed
                            break
            
            # Store retrieval metrics
            results["retrieval"]["precision"].append(precision)
            results["retrieval"]["recall"].append(recall)
            results["retrieval"]["reciprocal_rank"].append(mrr)
            results["by_question_type"][question_type]["retrieval_precision"].append(precision)
            
            # Build context from retrieved chunks
            contexts_second = query_index_with_context(question, k)
            logger.info(f"context: {contexts_second}")
            

            
            context = ""
            for doc in contexts_second:

                # doc , score= ctx_tuple
                context += "\n" + doc
            logger.info(f"context_second: {context}")
            # Measure generation time
            generation_start = time.time()


            
            # Generate answer
            generated_answer = stream_llama_cpp_response(question, context)
            # generated_answer = generated_answer_dict["response"]

            
            generation_time = time.time() - generation_start
            
            # Calculate generation metrics
            exact_match = self._normalized_text(generated_answer) == self._normalized_text(ground_truth)
            contains_answer = self._normalized_text(ground_truth) in self._normalized_text(generated_answer)
            embed_pred = embedding_function.embed_query(generated_answer)
            embed_ground = embedding_function.embed_query(ground_truth)
            semantic_sim = util.cos_sim(embed_pred, embed_ground).item()
            # semantic_sim = self._calculate_semantic_similarity(generated_answer, ground_truth)
            # semantic_sim = embedding_function.similarity(generated_answer, ground_truth)
            # For unanswerable questions, check if the system correctly identifies it as unanswerable
            if question_type == "unanswerable":
                # Keywords indicating the system recognizes the question can't be answered
                if "context does not contain" in generated_answer.lower():
                    semantic_sim =1.0
                    exact_match = True
                else:
                    semantic_sim = 0.0
                    exact_match = False
            if question_type == "factoid":
                # Custom token-based Jaccard similarity
                gt_tokens = set(self._normalized_text(ground_truth).split())
                pred_tokens = set(self._normalized_text(generated_answer).split())
                token_overlap = len(gt_tokens & pred_tokens) / len(gt_tokens | pred_tokens) if gt_tokens | pred_tokens else 0.0
                
                # Log it or add it to results if needed
                results["detailed_results"][-1]["generation_metrics"]["jaccard_token_similarity"] = token_overlap
                
                # Boost semantic similarity if Jaccard is decent but not exact
                if token_overlap > 0.5 and semantic_sim < 0.5:
                    semantic_sim = (semantic_sim + token_overlap) / 2

                            
            
            results["generation"]["exact_match"].append(1 if exact_match else 0)
            results["generation"]["contains_answer"].append(1 if contains_answer else 0)
            results["generation"]["semantic_similarity"].append(semantic_sim)
            results["by_question_type"][question_type]["generation_score"].append(semantic_sim)
            
            # Store detailed result for this question
            results["detailed_results"].append({
                "question": question,
                "question_type": question_type,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "retrieval_metrics": {
                    "precision": precision,
                    "recall": recall,
                    "mrr": mrr,
                    "retrieved_chunks": retrieved_chunk_ids,
                    "relevant_chunks": relevant_chunk_ids,
                    "time_seconds": retrieval_time
                },
                "generation_metrics": {
                    "exact_match": exact_match,
                    "contains_answer": contains_answer,
                    "semantic_similarity": semantic_sim,
                    "time_seconds": generation_time
                }
            })
        
        # Calculate averages for overall metrics
        metrics_to_average = {
            "retrieval": list(results["retrieval"].keys()),
            "generation": list(results["generation"].keys())
        }

        for category in ["retrieval", "generation"]:
            for metric in metrics_to_average[category]:
                if not metric.endswith("_avg") and results[category][metric]:
                    results[category][f"{metric}_avg"] = sum(results[category][metric]) / len(results[category][metric])
                elif not metric.endswith("_avg"):
                    results[category][f"{metric}_avg"] = 0.0
        
        # Calculate averages for question type metrics
        for q_type in results["by_question_type"]:
            type_data = results["by_question_type"][q_type]
            if type_data["retrieval_precision"]:
                type_data["retrieval_precision_avg"] = sum(type_data["retrieval_precision"]) / len(type_data["retrieval_precision"])
            else:
                type_data["retrieval_precision_avg"] = 0.0
                
            if type_data["generation_score"]:
                type_data["generation_score_avg"] = sum(type_data["generation_score"]) / len(type_data["generation_score"])
            else:
                type_data["generation_score_avg"] = 0.0
        
        # Add an overall score (combined metric)
        results["overall_score"] = (
            results["retrieval"]["precision_avg"] * 0.5 + 
            results["generation"]["semantic_similarity_avg"] * 0.5
        )
        
        logger.info(f"Evaluation complete. Overall score: {results['overall_score']:.2f}")
        return results
    
    def _build_context(self, chunks: List) -> str:
        """Build context string from retrieved chunks."""
        # Handle different chunk formats
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, "page_content"):
                # For LangChain document format
                context_parts.append(f"CHUNK {i+1}:\n{chunk.page_content}")
            elif isinstance(chunk, dict) and "content" in chunk:
                # Dictionary with content key
                context_parts.append(f"CHUNK {i+1}:\n{chunk['content']}")
            elif isinstance(chunk, dict) and "text" in chunk:
                # Dictionary with text key
                context_parts.append(f"CHUNK {i+1}:\n{chunk['text']}")
            elif isinstance(chunk, str):
                # Plain string
                context_parts.append(f"CHUNK {i+1}:\n{chunk}")
            else:
                # Try to extract content
                try:
                    content = str(chunk)
                    context_parts.append(f"CHUNK {i+1}:\n{content}")
                except:
                    logger.warning(f"Could not extract content from chunk {i}")
        
        return "\n\n".join(context_parts)
    
    def _normalized_text(self, text: str) -> str:
        """Normalize text for comparison."""
        import re
        # Convert to lowercase, remove extra whitespace, and normalize punctuation
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return normalized
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        This is a simple implementation using word overlap (Jaccard similarity).
        For production use, consider using embedding-based similarity.
        """
        words1 = set(self._normalized_text(text1).split())
        words2 = set(self._normalized_text(text2).split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a human-readable evaluation report.

        Args:
            results: Results from evaluate_rag_system

        Returns:
            Dictionary with report sections
        """
        if "error" in results:
            return {"error": results["error"]}

        report = {
            "summary": {
                "document": results["document"],
                "num_questions": results["num_questions"],
                "overall_score": f"{results['overall_score']:.2f} / 1.00",
                "retrieval": {
                    "precision": f"{results['retrieval']['precision_avg']:.2f}",
                    "recall": f"{results['retrieval']['recall_avg']:.2f}",
                    "mrr": f"{results['retrieval']['reciprocal_rank_avg']:.2f}"
                },
                "generation": {
                    "exact_match_rate": f"{results['generation']['exact_match_avg']:.2f}",
                    "answer_inclusion_rate": f"{results['generation']['contains_answer_avg']:.2f}",
                    "semantic_similarity": f"{results['generation']['semantic_similarity_avg']:.2f}",
                    "token_overlap_jaccard": f"{results['generation'].get('jaccard_token_similarity_avg', 0.0):.2f}"
                }
            },
            "by_question_type": {},
            "recommendations": self._generate_recommendations(results),
            "error_analysis": self._analyze_errors(results)
        }

        # Question type analysis
        for q_type, stats in results["by_question_type"].items():
            report["by_question_type"][q_type] = {
                "count": stats["count"],
                "retrieval_precision": f"{stats['retrieval_precision_avg']:.2f}",
                "generation_score": f"{stats['generation_score_avg']:.2f}"
            }

        return report

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Analyze retrieval performance
        retrieval_precision = results["retrieval"]["precision_avg"]
        retrieval_recall = results["retrieval"]["recall_avg"]
        
        if retrieval_precision < 0.5:
            recommendations.append(
                "Improve retrieval precision by enhancing query processing or using hybrid search."
            )
        
        if retrieval_recall < 0.5:
            recommendations.append(
                "Improve recall by increasing the number of retrieved chunks or implementing query expansion."
            )
        
        # Analyze generation performance
        semantic_similarity = results["generation"]["semantic_similarity_avg"]
        
        if semantic_similarity < 0.7:
            recommendations.append(
                "Improve answer generation by refining prompt design or enhancing context formatting."
            )
        
        # Analyze by question type
        for q_type, stats in results["by_question_type"].items():
            if stats["generation_score_avg"] < 0.6:
                recommendations.append(
                    f"Focus on improving performance for '{q_type}' questions which currently score below average."
                )
        
        # Check for any significantly better/worse question types
        type_scores = [(q_type, stats["generation_score_avg"]) 
                      for q_type, stats in results["by_question_type"].items()]
        if type_scores:
            best_type = max(type_scores, key=lambda x: x[1])
            worst_type = min(type_scores, key=lambda x: x[1])
            
            if best_type[1] - worst_type[1] > 0.3:  # Significant difference
                recommendations.append(
                    f"System performs much better on '{best_type[0]}' questions ({best_type[1]:.2f}) than '{worst_type[0]}' questions ({worst_type[1]:.2f}). Consider tailoring your approach for different question types."
                )
        
        # If few recommendations, add general ones
        if len(recommendations) < 2:
            recommendations.append(
                "Consider implementing hybrid search (combining vector search with keyword search) for better retrieval."
            )
            recommendations.append(
                "Experiment with different chunking strategies to improve contextual relevance."
            )
        
        return recommendations
    
    def _analyze_errors(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns in the evaluation results."""
        error_analysis = {
            "worst_performing_questions": [],
            "retrieval_errors": 0,
            "generation_errors": 0,
            "error_by_type": {}
        }
        
        # Count errors by type
        for q_type in results["by_question_type"]:
            error_analysis["error_by_type"][q_type] = {
                "count": 0,
                "percentage": 0.0
            }
        
        # Analyze detailed results for errors
        detailed_results = results.get("detailed_results", [])
        
        # Sort by performance (worst first)
        sorted_results = sorted(
            detailed_results, 
            key=lambda x: x["generation_metrics"]["semantic_similarity"]
        )
        
        # Count different error types
        for item in detailed_results:
            question_type = item["question_type"]
            
            # Check for retrieval error
            if item["retrieval_metrics"]["precision"] < 0.3:
                error_analysis["retrieval_errors"] += 1
            
            # Check for generation error
            if item["generation_metrics"]["semantic_similarity"] < 0.5:
                error_analysis["generation_errors"] += 1
                error_analysis["error_by_type"][question_type]["count"] += 1
        
        # Calculate percentages
        for q_type, stats in error_analysis["error_by_type"].items():
            type_count = results["by_question_type"][q_type]["count"]
            if type_count > 0:
                stats["percentage"] = (stats["count"] / type_count) * 100
        
        # Add worst performing questions (up to 5)
        for item in sorted_results[:5]:
            if item["generation_metrics"]["semantic_similarity"] < 0.7:  # Only include actual poor performers
                error_analysis["worst_performing_questions"].append({
                    "question": item["question"],
                    "question_type": item["question_type"],
                    "ground_truth": item["ground_truth"],
                    "generated_answer": item["generated_answer"],
                    "similarity_score": item["generation_metrics"]["semantic_similarity"]
                })
        
        return error_analysis

# Integration with FastAPI
def setup_evaluation_endpoints(app, vector_store, llm_function):
    """
    Add RAG evaluation endpoints to a FastAPI app.
    
    Args:
        app: FastAPI app instance
        vector_store: Vector store instance
        llm_function: Function to generate answers using an LLM
    """
    from fastapi import HTTPException
    import logging
    
    logger = logging.getLogger("rag_evaluator")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(vector_store, llm_function)
    
    # Store as app state
    app.state.rag_evaluator = evaluator
    
    @app.post("/evaluate/create-dataset")
    async def create_evaluation_dataset(request: dict):
        try:
            document_name = request.get("document")
            if not document_name:
                document_name = getattr(app.state, "current_document", None)
                if not document_name:
                    raise HTTPException(status_code=400, detail="No document specified")
            
            # Get document text (implement this based on your storage)
            document_text = get_document_text(document_name)
            if not document_text:
                raise HTTPException(status_code=404, detail=f"Document {document_name} not found")
            
            # Optional parameters
            num_questions = request.get("num_questions", 20)
            question_types = request.get("question_types", None)
            
            # Create dataset
            dataset = evaluator.create_evaluation_dataset(
                document_name=document_name,
                document_text=document_text,
                num_questions=num_questions,
                question_types=question_types
            )
            
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
    
    @app.post("/evaluate/run")
    async def run_evaluation(request: dict):
        try:
            document_name = request.get("document")
            if not document_name:
                document_name = getattr(app.state, "current_document", None)
                if not document_name:
                    raise HTTPException(status_code=400, detail="No document specified")
            
            # Load dataset
            dataset = evaluator.load_dataset(document_name)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"No evaluation dataset found for {document_name}")
            
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
            
            return {
                "success": True,
                "document": document_name,
                "report": report,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error running evaluation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error running evaluation: {str(e)}")

# Helper function to get document text (implement based on your storage)
def get_document_text(document_name):
    """
    Retrieve the full text of a document by name.
    Implement this based on your document storage mechanism.
    """
    # This is a placeholder - implement based on your storage
    # For example, you might load from a file, database, etc.
    from document_processor import process_document
    return process_document(document_name)

from vector_store import VectorStore
from document_processor_patched import SimpleSentenceTransformer
from llama_cpp_interface import stream_llama_cpp_response
from improved_prompt_template import improved_llama_cpp_response

def enhanced_llm_function(query, context):
    """Wrapper for the improved LLM function."""
    return improved_llama_cpp_response( query, context, temperature=0.3)


# Example usage:
if __name__ == "__main__":
    # This is an example of how to use the RAG evaluator
    # You would need to implement these components based on your actual system
    
    # class MockVectorStore:
    #     def search(self, query, k=5, filter=None):
    #         # Mock implementation for testing
    #         return [{"content": f"This is chunk {i}", "chunk_id": i} for i in range(k)]
    
    # def mock_llm_function(query, context):
    #     # Mock implementation for testing
    #     return f"Answer to: {query}\nBased on: {context[:50]}..."
    vector_store = VectorStore()
    embedding_function = SimpleSentenceTransformer()
    from llama_cpp_interface import improved_llama_cpp_response
    # vector_store = VectorStore()
    MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    # MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    # MODEL_PATH = os.environ.get("LLAMA_CPP_MODEL_PATH", "../bitnet/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    CTX_SIZE = int(os.environ.get("LLAMA_CTX_SIZE", 2048))
    N_THREADS = int(os.environ.get("LLAMA_THREADS", os.cpu_count() ))
    N_GPU_LAYERS = int(os.environ.get("LLAMA_GPU_LAYERS", -1))  # -1 means use all available GPU layers
    from llama_cpp import Llama
    try:
        # Add n_gpu_layers parameter for Apple Silicon
        llm = Llama(
            model_path=MODEL_PATH, 
            n_ctx=CTX_SIZE, 
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS  # Use Metal on Apple Silicon
        )
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load LLaMA model: {str(e)}")
        llm = None
    from enhanced_rag import EnhancedRAG
    document_name = sys.argv[2]
    # enhanced_rag = EnhancedRAG(vector_store, llm, document_name)
    # Initialize evaluator
    evaluator = RAGEvaluator(
       
        llm_function=stream_llama_cpp_response
       
    )
    
    # Create a test dataset
    # test_doc = "This is a test document. It contains some information about RAG systems."
    # dataset = evaluator.create_evaluation_dataset(
    #     document_name="test_doc",
    #     document_text=test_doc,
    #     num_questions=5
    # )
   
    # document_text = get_document_text(document_name)
    # dataset = evaluator.create_evaluation_dataset(
    #     document_name=document_name,
    #     document_text=document_text,
    #     num_questions=5
    # )
    dataset = evaluator.load_dataset(document_name)
    # logger.info(f"dataset: {dataset}")
    # from document_processor import chunk_document
    # chunks = chunk_document(document_text, chunk_size=200, chunk_overlap=50)
    
    # vector_store.add_document(document_name, chunks)
    from document_processor_patched import process_and_index_file
    chunks=process_and_index_file(document_name)
    # Run evaluation
    results = evaluator.evaluate_rag_system(
        document_name=document_name,
        dataset=dataset
    )
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    print(report)
    
    print(f"Evaluation complete. Overall score: {results['overall_score']:.2f}")