import gc
import os
import logging
import time
import numpy as np
from typing import List
import torch
from vector_store import VectorStore,build_focused_context
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# Set up logging
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
def rerank_chunks(question, chunks, top_k=5):
                # pairs = [(question, chunk) for chunk, _ in chunks]
                pairs = []
                for i ,doc in enumerate(chunks):
                    logger.info(f"doc_rerank: {doc}")
                    d, score = doc
                    # doc, score = chunk
                    # logger.info(f"doc: {doc}")
                    # logger.info(f"type of doc: {type(doc)}")
                    # # doc = Document(doc)
                    # page_content, (source, chunk_id) = doc
                    pairs.append((question, d.page_content))
                scores = reranker.predict(pairs)
                reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
                return reranked[:top_k]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple SentenceTransformer implementation without dependencies
# class SimpleSentenceTransformer:
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         logger.info(f"Loading model: {model_name}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.model.eval()
        
#     def encode(self, sentences, convert_to_numpy=True, normalize=True):
#         if isinstance(sentences, str):
#             sentences = [sentences]
            
#         logger.info(f"sentences: {sentences}")
#         # Tokenize
#         encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#         logger.info(f"encoded_input: {encoded_input}")
#         # Get model output
#         with torch.no_grad():
#             model_output = self.model(**encoded_input)

#         logger.info(f"model_output: {model_output}")
#         # Mean Pooling
#         token_embeddings = model_output[0] 
#         attention_mask = encoded_input['attention_mask']
        
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         logger.info(f"embeddings: {embeddings}")
#         # Normalize
#         if normalize:
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
#         # Convert to numpy if requested
#         if convert_to_numpy:
#             embeddings = embeddings.numpy()
            
#         return embeddings
class SimpleSentenceTransformer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device="cpu")
        
    def encode_single_text(self, text):
        return self.model.encode(text)
    # Then update embed_query and embed_documents methods
    def embed_query(self, text):
        result = self.encode_single_text(text)
        return result.tolist()

    def embed_documents(self, texts):
        results = []
        for text in texts:
            # Process one at a time
            embedding = self.encode_single_text(text)
            results.append(embedding.tolist())
            # Add small delay between processing to allow memory cleanup
            time.sleep(0.1)
        return results
    def encode(self, texts, convert_to_numpy=True):
        embeddings = []
        for text in texts:
            embeddings.append(self.encode_single_text(text, convert_to_numpy))

        return embeddings
# Direct embedding function without any abstract base class
class LocalEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SimpleSentenceTransformer(model_name)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"texts: {texts}")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        logger.info(f"embeddings: {embeddings}")
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.model.encode([text], convert_to_numpy=True)[0]
        return embeddings.tolist()

# Simple vector store implementation
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents, embeddings):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        
    def similarity_search_with_score(self, query_embedding, k=5):
        if not self.embeddings:
            return []
            
        scores = []
        for emb in self.embeddings:
            # Compute cosine similarity
            emb_array = np.array(emb)
            query_array = np.array(query_embedding)
            similarity = np.dot(query_array, emb_array) / (np.linalg.norm(query_array) * np.linalg.norm(emb_array))
            scores.append(similarity)
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Return top k documents and scores
        return [(self.documents[i], scores[i]) for i in top_k_indices]

# Document class compatible with your evaluator expectations
from langchain.schema import Document

# Global variables
embedding_function = None
vector_store = None

# Initialize the models
def initialize():
    global embedding_function, vector_store
    embedding_function = SimpleSentenceTransformer()
    vector_store = VectorStore()
    if vector_store is None:
        raise ValueError("Vector store is None")
    logger.info("Initialized embedding model and vector store")

# Document processing and chunking
def process_and_index_file(file_path):
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    global embedding_function, vector_store
    
    # Initialize if needed
    if embedding_function is None or vector_store is None:
        initialize()
    
    logger.info(f"Processing file: {file_path}")
    
    try:
        # Extract text from PDF
        if file_path.lower().endswith('.pdf'):
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text() + "\n\n"
            except ImportError:
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        text = ""
                        for page in pdf.pages:
                            text += page.extract_text() + "\n\n"
                except ImportError:
                    raise ImportError("PDF extraction libraries not found.")
        else:
            # For text files, try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            text = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
                    
            if text is None:
                raise ValueError("Could not decode file with any encoding")
        
        # Split text into very small chunks to avoid memory issues
        max_chunk_size = 256  # Very small chunks
        
        # Split by sentences or short paragraphs
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            logger.info(f"length of para: {len(sentence)}")
            
            # If this sentence would make the chunk too large, save current chunk and start a new one
            if len(current_chunk) + len(sentence) > max_chunk_size:
                if current_chunk:  # Only add if there's content
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={"source": file_path, "chunk_id": chunk_id}
                    ))
                    chunk_id += 1
                    current_chunk = sentence
                else:
                    # If a single sentence is longer than max_chunk_size, truncate it
                    if len(sentence) > max_chunk_size:
                        chunks.append(Document(
                            page_content=sentence[:max_chunk_size].strip(),
                            metadata={"source": file_path, "chunk_id": chunk_id}
                        ))
                        chunk_id += 1
                    else:
                        current_chunk = sentence
            else:
                # Add to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the final chunk
        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={"source": file_path, "chunk_id": chunk_id}
            ))
        
        logger.info(f"Split document into {len(chunks)} chunks")
        logger.info(f"chunks: {chunks}")
        
        # Process each chunk individually
        all_embeddings = []
        
        for chunk in chunks:
            # Process one chunk at a time 
            try:
                logger.info(f"Processing chunk: {chunk.page_content[:50]}...")
                
                # Use the encode method one document at a time
                embedding = embedding_function.embed_query(chunk.page_content)
                all_embeddings.append(embedding)
                
                # Force garbage collection
                # import gc
                # gc.collect()
                # torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"Error encoding chunk: {e}")
                # Create a fallback embedding for this chunk
                embedding_dim = 384
                all_embeddings.append(np.random.randn(embedding_dim).astype(np.float32).tolist())
            
        # Add to vector store
        from generate_suggestions import generate_suggested_questions
        suggested_questions = generate_suggested_questions(text)
        # for chunk in chunks:
        # vector_store.add_document(chunk.page_content, chunk.metadata)
        # vector_store.add_documents(chunks, all_embeddings)
        vector_store.add_document(file_path, chunks)
        logger.info(f"Indexed {len(chunks)} chunks in vector store")
        
        return chunks, suggested_questions
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise e
# Query function
def query_index(query, top_k=5):
    global embedding_function, vector_store
    
    # Initialize if needed
    if embedding_function is None or vector_store is None:
        initialize()
    
    # Check for invalid query types
    if isinstance(query, bool) or not isinstance(query, str):
        if not isinstance(query, str):
            try:
                query = str(query)
                logger.warning(f"Converted non-string query to string: {query}")
            except:
                logger.error(f"Cannot convert {type(query)} to string")
                return []
    
    try:
        # Empty query check
        if not query.strip():
            logger.warning("Empty query received")
            return []
            
        # Get query embedding
        # query_embedding = embedding_function.embed_query(query)
        
        # Search similar documents
        results = vector_store.search(query, k=top_k)
        results = rerank_chunks(query, results, top_k)
        # For LLM context window limitations, ensure total content is manageable
        max_content_length = 1500  # Safe limit for most small models
        total_length = 0
        pruned_results = []
        
        for i, ((doc ,s), score) in enumerate(results):
            logger.info(f"doc_query: {doc}")
            content_length = len(doc.page_content)
            if total_length + content_length <= max_content_length:
                pruned_results.append((doc, score))
                total_length += content_length
            else:
                logger.warning(f"Skipping document with length {content_length} to avoid exceeding context window")
        
        return pruned_results
    
    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        return []

def query_index_with_context(query, top_k=5):
    global embedding_function, vector_store
    results = query_index(query, top_k)
    # context = build_focused_context(query, results)
    content = ""
    for doc, score in results:
        logger.info(f"doc_query_with_context: {doc}")
        logger.info(f"score_query_with_context: {score}")
        content += f"Document: {doc.page_content}\n"
        # content.append(doc.page_content)
    return content

def build_focused_context(query, results):
    context = f"Query: {query}\n\n"
    for doc, score in results:
        context += f"Document: {doc.page_content}\n"
    return context
# Example usage
if __name__ == "__main__":
    # process_and_index_file("example.txt")
    # results = query_index("What is this document about?")
    # for doc, score in results:
    #     print(f"Score: {score:.4f}, Text: {doc.page_content[:100]}...")
    pass