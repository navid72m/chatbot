# Advanced RAG System with Knowledge Graph Integration

This project extends a standard Retrieval-Augmented Generation (RAG) chatbot with advanced features to reduce hallucination and improve accuracy:

- **Knowledge Graph Integration**: Captures entity relationships in Neo4j to enhance retrieval
- **Chain-of-Thought Reasoning**: Implements explicit reasoning steps before answering
- **Multi-hop Reasoning**: Breaks complex queries into sub-questions
- **Hybrid Retrieval**: Combines vector search with graph traversal for better document retrieval
- **Answer Verification**: Cross-checks generated answers against source material

## Architecture Overview

![Advanced RAG Architecture](https://raw.githubusercontent.com/yourusername/advanced-rag-system/main/docs/architecture.svg)

```
┌───────────────────┐     ┌────────────────────┐
│                   │     │                    │
│    Documents      │────▶│  Document          │
│                   │     │  Processor         │
└───────────────────┘     └──────┬─────────────┘
                                 │
                                 ▼
      ┌────────────────────────────────────────┐
      │                                        │
      ▼                                        ▼
┌───────────────┐                    ┌─────────────────┐
│               │                    │                 │
│ Vector Store  │                    │ Knowledge Graph │
│ (ChromaDB)    │                    │ (Neo4j)         │
│               │                    │                 │
└───────┬───────┘                    └────────┬────────┘
        │                                     │
        ▼                                     ▼
┌───────────────┐                    ┌─────────────────┐
│               │                    │                 │
│Vector Retrieval│                    │ Graph Retrieval │
│               │                    │                 │
└───────┬───────┘                    └────────┬────────┘
        │                                     │
        └─────────────────┬──────────────────┘
                          │
                          ▼
                  ┌────────────────┐       ┌─────────────────┐
                  │                │       │                 │
                  │Hybrid Retriever│──────▶│Chain-of-Thought │
                  │                │       │Reasoning        │
                  └────────────────┘       └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │                 │
                                           │   Multi-hop     │
                                           │   Reasoning     │
                                           │                 │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │                 │
                                           │    Answer       │
                                           │  Verification   │
                                           │                 │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │                 │
                                           │ LLM Interface   │
                                           │   (Ollama)      │
                                           │                 │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │                 │
                                           │  Final Answer   │
                                           │                 │
                                           └─────────────────┘
```

The system integrates three key components:

1. **Vector Store**: Semantic search using Chroma and sentence transformers
2. **Knowledge Graph**: Entity and relationship extraction with Neo4j
3. **Reasoning Layer**: Chain-of-thought and multi-hop reasoning with LLMs

## Components

### Core Files

- `advanced_rag.py`: Main implementation of the Advanced RAG system
- `knowledge_graph.py`: Neo4j integration for entity and relationship management
- `chain_of_thought.py`: Implementation of chain-of-thought reasoning
- `hybrid_retriever.py`: Combined retrieval from vector store and knowledge graph
- `app_integration.py`: FastAPI server integrating all components

### Support Files

- `vector_store.py`: Vector store implementation using Chroma
- `document_processor.py`: Document processing and chunking
- `llm_interface.py`: Interface to Ollama LLM models
- `setup_advanced_rag.py`: Setup script for dependencies
- `example_usage.py`: Example demonstration script

## Setup Instructions

### Prerequisites

- Python 3.9+
- Neo4j (for knowledge graph features)
- Ollama (for LLM access)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-rag-system.git
   cd advanced-rag-system
   ```

2. Run the setup script:
   ```bash
   python setup_advanced_rag.py
   ```

3. Ensure Neo4j is running:
   ```bash
   # You can use Docker for easy setup
   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
   ```

4. Ensure Ollama is running with the required model:
   ```bash
   ollama serve
   ollama pull mistral
   ```

5. Start the API server:
   ```bash
   python app_integration.py
   ```

## Usage Examples

### Basic Query

```python
from advanced_rag import AdvancedRAG
from vector_store import VectorStore
from knowledge_graph import KnowledgeGraph
from chain_of_thought import ChainOfThoughtReasoner

# Initialize components
vector_store = VectorStore()
knowledge_graph = KnowledgeGraph()
reasoner = ChainOfThoughtReasoner()

# Create Advanced RAG instance
rag = AdvancedRAG(
    vector_store=vector_store,
    knowledge_graph=knowledge_graph,
    reasoner=reasoner,
    model="mistral"
)

# Add documents
documents = [...] # Your documents here
rag.add_documents(documents)

# Query
response = rag.answer_query("What causes climate change?")
print(response["answer"])
```

### API Usage

The system exposes a FastAPI interface:

```bash
# Upload a document
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload

# Query
curl -X POST -H "Content-Type: application/json" \
    -d '{"query": "What is climate change?", "use_advanced_rag": true}' \
    http://localhost:8000/query
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_cot` | Enable chain-of-thought reasoning | `True` |
| `use_kg` | Enable knowledge graph features | `True` |
| `verify_answers` | Cross-check answers against sources | `True` |
| `use_multihop` | Enable breaking complex queries | `True` |
| `max_hops` | Maximum path length in knowledge graph | `2` |
| `vector_weight` | Weight for vector search results | `0.7` |
| `kg_weight` | Weight for knowledge graph results | `0.3` |
| `max_vector_results` | Maximum vector search results | `5` |
| `max_kg_results` | Maximum knowledge graph results | `3` |
| `temperature` | LLM temperature parameter | `0.7` |

## API Endpoints

The system provides the following API endpoints:

### Document Management

- `POST /upload`: Upload a document for processing
- `GET /`: Root endpoint with API information

### Query Processing

- `POST /query`: Query documents with optional advanced RAG features
- `GET /models`: List available Ollama models
- `GET /quantization-options`: Get available quantization options

### Knowledge Graph Endpoints

- `GET /knowledge-graph/entities`: Get entities extracted from a query
- `POST /knowledge-graph/entity-graph`: Get a subgraph around specified entities
- `GET /knowledge-graph/entity/{entity_name}/related`: Get entities related to a specific entity
- `GET /knowledge-graph/path`: Find paths between two entities

### Configuration

- `POST /config/rag`: Update Advanced RAG configuration
- `GET /advanced-rag/features`: Get information about available features

### Debug

- `POST /debug/analyze-query`: Debug endpoint to analyze a query

## How It Works

### Knowledge Graph Integration

The system extracts entities and relationships from documents:

1. **Entity Extraction**: Uses spaCy NLP to identify people, organizations, locations, and concepts
2. **Relationship Extraction**: Identifies subject-verb-object patterns to establish connections
3. **Graph Storage**: Stores everything in Neo4j with proper indexing for fast retrieval

During retrieval, the system:

1. Extracts entities from the query
2. Finds related entities in the knowledge graph
3. Retrieves documents mentioning these entities
4. Identifies paths between entities mentioned in the query
5. Combines with vector search results

### Chain-of-Thought Reasoning

The system implements structured reasoning:

1. Identifies key entities and concepts in the question
2. Determines what specific information is being requested
3. Finds relevant information in the context
4. Connects information to develop a reasoned answer
5. Checks for assumptions or limitations
6. Provides the final answer based on explicit reasoning

### Multi-hop Reasoning for Complex Queries

For complex questions, the system:

1. Decomposes the question into 2-4 simpler sub-questions
2. Answers each sub-question sequentially
3. Synthesizes the final answer using all intermediate answers

### Answer Verification

To reduce hallucination, the system:

1. Generates an answer based on retrieved documents
2. Cross-checks each claim against the source material
3. Identifies unsupported claims and contradictions
4. Assigns a confidence score (HIGH/MEDIUM/LOW)
5. Highlights limitations or uncertainties in the answer

## Performance Considerations

- **Memory Usage**: The system requires significant RAM, especially with larger knowledge graphs
- **Neo4j Performance**: For large document collections, proper Neo4j indexing is crucial
- **LLM Latency**: Chain-of-thought and multi-hop reasoning involve multiple LLM calls, increasing response time
- **Entity Extraction**: spaCy processing can be CPU-intensive for large documents

## Limitations and Future Work

- **Knowledge Graph Quality**: Entity and relationship extraction is imperfect and may miss complex relationships
- **Cross-document Reasoning**: Current implementation has limited ability to connect information across many documents
- **Temporal Reasoning**: Limited support for time-based relationships and reasoning
- **Question Complexity**: Very complex questions may still yield incomplete answers

Future improvements could include:

- Implementing more sophisticated entity and relationship extraction
- Adding temporal awareness to the knowledge graph
- Supporting multilingual documents and queries
- Implementing more advanced multi-document reasoning strategies
- Adding interactive clarification for ambiguous queries

## Contributors

This project was developed by [Your Team Name] with contributions from:

- Your Name - Core architecture and integration
- Contributor 1 - Knowledge graph implementation
- Contributor 2 - Chain-of-thought reasoning

## License

This project is licensed under the MIT License - see the LICENSE file for details.