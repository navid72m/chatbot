#!/usr/bin/env python
# example_usage.py - Example script demonstrating usage of the Advanced RAG system

import logging
import os
import sys
import time
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Make sure modules are in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import Advanced RAG components
from knowledge_graph import KnowledgeGraph
from chain_of_thought import ChainOfThoughtReasoner
from advanced_rag import AdvancedRAG
from vector_store import VectorStore

def main():
    """Main example function demonstrating Advanced RAG usage"""
    print("\n" + "=" * 60)
    print(" Advanced RAG Example Usage")
    print("=" * 60 + "\n")
    
    # Step 1: Initialize components
    print("Initializing components...")
    
    # Vector store
    vector_store = VectorStore()
    
    # Knowledge graph (using default connection settings)
    try:
        knowledge_graph = KnowledgeGraph(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
        kg_available = True
    except Exception as e:
        logger.error(f"Could not connect to Neo4j: {e}")
        logger.warning("Knowledge graph features will be disabled")
        knowledge_graph = None
        kg_available = False
    
    # Reasoner
    reasoner = ChainOfThoughtReasoner(model="mistral")
    
    # Advanced RAG
    rag = AdvancedRAG(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        reasoner=reasoner,
        model="mistral"
    )
    
    # Configure RAG options
    rag.use_kg = kg_available
    rag.use_cot = True
    rag.verify_answers = True
    rag.use_multihop = True
    
    # Step 2: Create sample documents
    print("\nCreating sample documents...")
    
    sample_texts = [
        {
            "title": "Climate Change Basics",
            "content": """
            Climate change refers to long-term shifts in temperatures and weather patterns. 
            These shifts may be natural, such as through variations in the solar cycle. 
            But since the 1800s, human activities have been the main driver of climate change, 
            primarily due to burning fossil fuels like coal, oil and gas, which produces heat-trapping gases.
            
            Greenhouse gases such as carbon dioxide, methane, and nitrous oxide trap heat in the Earth's atmosphere, 
            causing the greenhouse effect. While this effect is natural and necessary to support life on Earth, 
            the increased concentration of these gases due to human activities is causing global temperatures to rise.
            
            The Intergovernmental Panel on Climate Change (IPCC), which includes more than 1,300 scientists from the 
            United States and other countries, forecasts a temperature rise of 2.5 to 10 degrees Fahrenheit over the 
            next century. According to the IPCC, the extent of climate change effects on individual regions will vary 
            over time and with the ability of different societal and environmental systems to mitigate or adapt to change.
            """
        },
        {
            "title": "Climate Change Impacts",
            "content": """
            The impacts of climate change are already being observed across the globe. Rising global temperatures 
            have been accompanied by changes in weather and climate. Many places have seen changes in rainfall, 
            resulting in more floods, droughts, or intense rain, as well as more frequent and severe heat waves.
            
            Sea levels are rising due to thermal expansion of warming ocean water and melting land ice, threatening 
            coastal communities and ecosystems. The oceans are also absorbing increased carbon dioxide, leading to 
            ocean acidification, which harms marine life, especially organisms with calcium carbonate shells or skeletons.
            
            Ecosystems and biodiversity are being disrupted by climate change. Many land, freshwater, and marine species 
            have already moved to new locations or altered their seasonal activities in response to climate change. Some 
            species face increased extinction risk due to climate change, especially those that cannot adapt quickly enough.
            
            Human systems are also vulnerable to climate change. Agriculture may be affected as temperature and precipitation 
            patterns change, impacting crop yields. Human health can be impacted through increased heat stress, waterborne 
            diseases, poor air quality, and diseases transmitted by insects and rodents. Infrastructure and transportation 
            systems may be damaged by extreme weather events.
            """
        },
        {
            "title": "Climate Change Mitigation",
            "content": """
            Climate change mitigation involves reducing greenhouse gas emissions and enhancing sinks that absorb greenhouse gases. 
            Mitigation strategies include transitioning to renewable energy sources such as solar, wind, and hydroelectric power, 
            improving energy efficiency, and changing management practices in agriculture and forestry.
            
            The Paris Agreement, adopted in 2015, aims to strengthen the global response to climate change by keeping global 
            temperature rise this century well below 2 degrees Celsius above pre-industrial levels, and to pursue efforts to 
            limit the temperature increase even further to 1.5 degrees Celsius. To achieve this goal, countries have submitted 
            nationally determined contributions (NDCs) outlining their post-2020 climate actions.
            
            Carbon pricing is a tool used by some governments and businesses to reduce emissions. This approach puts a price on 
            carbon dioxide emissions, encouraging polluters to reduce emissions and invest in clean energy and low-carbon growth.
            
            Carbon capture and storage (CCS) technologies can capture CO2 produced from the use of fossil fuels in electricity 
            generation and industrial processes, preventing the CO2 from entering the atmosphere. The captured CO2 can be put to 
            productive use or stored deep underground in geological formations.
            
            Individual actions also play a role in mitigating climate change. These include reducing energy use, using renewable 
            energy, adopting sustainable transportation methods, reducing waste, and supporting climate-friendly businesses and policies.
            """
        }
    ]
    
    # Create document objects
    documents = []
    for item in sample_texts:
        doc = Document(
            page_content=item["content"],
            metadata={"source": item["title"]}
        )
        documents.append(doc)
    
    # Step 3: Add documents to the system
    print("Adding documents to Advanced RAG system...")
    rag.add_documents(documents)
    
    # Step 4: Ask some questions
    questions = [
        "What is climate change and what causes it?",
        "How does climate change affect the environment?",
        "What is the relationship between fossil fuels and global warming?",
        "What are some strategies to mitigate climate change?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'-' * 40}")
        print(f"Question {i}: {question}")
        print(f"{'-' * 40}")
        
        # Time the query
        start_time = time.time()
        
        # Get the answer
        response = rag.answer_query(question)
        
        # Print the answer
        print(f"\nAnswer: {response['answer']}")
        print(f"\nSources: {', '.join(response['sources'])}")
        print(f"Confidence: {response['confidence']}")
        print(f"Time taken: {response['retrieval_time']:.2f} seconds")
        
        # Print reasoning if available
        if response.get('reasoning'):
            print(f"\nReasoning: {response['reasoning'][:300]}...")
        
        # Wait between questions
        if i < len(questions):
            print("\nWaiting for next question...")
            time.sleep(2)
    
    # Step 5: Demonstrate multi-hop reasoning with a complex question
    complex_question = "How might renewable energy adoption help address the impacts of climate change on coastal communities?"
    
    print(f"\n{'=' * 60}")
    print(f" Complex Question: {complex_question}")
    print(f"{'=' * 60}")
    
    # Set up multi-hop reasoning
    rag.use_multihop = True
    
    # Get the answer
    start_time = time.time()
    response = rag.answer_query(complex_question)
    
    # Print the answer
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources: {', '.join(response['sources'])}")
    print(f"Confidence: {response['confidence']}")
    print(f"Time taken: {response['retrieval_time']:.2f} seconds")
    
    # Clean up
    if kg_available:
        knowledge_graph.close()
    
    print("\n" + "=" * 60)
    print(" Example completed")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()