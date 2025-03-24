# knowledge_graph.py - Neo4j knowledge graph integration for enhanced RAG
import logging
from typing import List, Dict, Optional, Set, Tuple
from neo4j import GraphDatabase
import spacy
from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Knowledge graph integration using Neo4j for enhanced RAG capabilities"""
    
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="password"):
        """Initialize the knowledge graph connection"""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
        # Load NLP model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy NLP model for entity extraction")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            logger.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")
        
        # Connect to Neo4j
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            logger.info(f"Connected to Neo4j at {self.uri}")
            
            # Create constraints and indexes for better performance
            with self.driver.session() as session:
                # Create constraints on Entity nodes
                session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                # Create constraint on Document nodes
                session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
                
                logger.info("Created Neo4j constraints and indexes")
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}")
            raise
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text using spaCy"""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Extract noun chunks for concept entities
        for chunk in doc.noun_chunks:
            # Filter out single pronouns and very short chunks
            if len(chunk.text.split()) > 1 or (len(chunk.text) > 3 and not chunk.text.lower() in ["he", "she", "it", "they", "we", "you"]):
                entities.append({
                    "text": chunk.text,
                    "label": "CONCEPT",
                    "start": chunk.start_char,
                    "end": chunk.end_char
                })
        
        return entities
    
    def extract_relationships(self, text: str) -> List[Dict]:
        """Extract potential relationships between entities"""
        doc = self.nlp(text)
        relationships = []
        
        for sent in doc.sents:
            entities_in_sent = [(e.text, e.start_char, e.end_char, e.label_) for e in sent.ents]
            
            # Skip if fewer than 2 entities in sentence
            if len(entities_in_sent) < 2:
                continue
                
            # Extract subject-verb-object patterns
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    # Find subject
                    subject = None
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = child
                            break
                    
                    # Find object
                    obj = None
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj", "attr"]:
                            obj = child
                            break
                    
                    # If we have both subject and object, create relationship
                    if subject and obj:
                        # Expand to include compound phrases
                        full_subject = self._expand_compound(subject)
                        full_object = self._expand_compound(obj)
                        
                        relationships.append({
                            "subject": full_subject,
                            "predicate": token.text,
                            "object": full_object,
                            "sentence": sent.text
                        })
        
        return relationships
    
    def _expand_compound(self, token):
        """Expand a token to include its compound phrases"""
        result = token.text
        
        # Check for compound dependencies
        for child in token.children:
            if child.dep_ == "compound":
                result = child.text + " " + result
        
        # Also check for adjectival modifiers
        for child in token.children:
            if child.dep_ == "amod":
                result = child.text + " " + result
        
        return result
    
    def add_document(self, document: Document) -> None:
        """Process a document and add it to the knowledge graph"""
        doc_id = document.metadata.get("source", "unknown")
        text = document.page_content
        
        # Extract entities and relationships
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text)
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from document {doc_id}")
        
        # Add to Neo4j
        with self.driver.session() as session:
            # Create document node
            session.run(
                "MERGE (d:Document {doc_id: $doc_id}) SET d.text = $text",
                doc_id=doc_id, text=text[:1000]  # Store preview of text
            )
            
            # Create entity nodes and connect to document
            for entity in entities:
                session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type
                MERGE (d:Document {doc_id: $doc_id})
                MERGE (e)-[:MENTIONED_IN]->(d)
                """,
                name=entity["text"].lower(),
                type=entity["label"],
                doc_id=doc_id
                )
            
            # Create relationships between entities
            for rel in relationships:
                session.run("""
                MATCH (s:Entity {name: $subject})
                MATCH (o:Entity {name: $object})
                MERGE (s)-[:RELATES_TO {predicate: $predicate, sentence: $sentence, doc_id: $doc_id}]->(o)
                """,
                subject=rel["subject"].lower(),
                object=rel["object"].lower(),
                predicate=rel["predicate"],
                sentence=rel["sentence"],
                doc_id=doc_id
                )
    
    def process_documents(self, documents: List[Document]) -> None:
        """Process multiple documents and add them to the knowledge graph"""
        for document in documents:
            self.add_document(document)
    
    def query_related_entities(self, entity_name: str, max_hops: int = 2) -> List[Dict]:
        """Query for entities related to the given entity"""
        with self.driver.session() as session:
            result = session.run(f"""
            MATCH (e:Entity {{name: $name}})
            MATCH (e)-[r:RELATES_TO*1..{max_hops}]-(related:Entity)
            RETURN related.name AS name, related.type AS type, 
                   count(*) AS strength
            ORDER BY strength DESC
            LIMIT 10
            """, name=entity_name.lower())
            
            return [{"name": record["name"], "type": record["type"], "strength": record["strength"]} 
                    for record in result]
    
    def query_entity_documents(self, entity_name: str) -> List[str]:
        """Find documents mentioning the given entity"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(d:Document)
            RETURN d.doc_id AS doc_id
            """, name=entity_name.lower())
            
            return [record["doc_id"] for record in result]
    
    def query_relationship_context(self, subject: str, object_entity: str) -> List[Dict]:
        """Find relationship context between two entities"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (s:Entity {name: $subject})-[r:RELATES_TO]->(o:Entity {name: $object})
            RETURN r.predicate AS predicate, r.sentence AS context, r.doc_id AS source
            UNION
            MATCH (o:Entity {name: $object})-[r:RELATES_TO]->(s:Entity {name: $subject})
            RETURN r.predicate AS predicate, r.sentence AS context, r.doc_id AS source
            """, subject=subject.lower(), object=object_entity.lower())
            
            return [{"predicate": record["predicate"], 
                     "context": record["context"],
                     "source": record["source"]} 
                   for record in result]
    
    def find_path_between_entities(self, start_entity: str, end_entity: str, max_hops: int = 3) -> List[Dict]:
        """Find paths connecting two entities"""
        with self.driver.session() as session:
            result = session.run(f"""
            MATCH path = shortestPath((s:Entity {{name: $start}})-[r:RELATES_TO*1..{max_hops}]-(e:Entity {{name: $end}}))
            UNWIND relationships(path) AS rel
            RETURN startNode(rel).name AS from_entity, 
                   endNode(rel).name AS to_entity,
                   rel.predicate AS relationship,
                   rel.sentence AS context
            """, start=start_entity.lower(), end=end_entity.lower())
            
            paths = []
            for record in result:
                paths.append({
                    "from": record["from_entity"],
                    "to": record["to_entity"],
                    "relationship": record["relationship"],
                    "context": record["context"]
                })
            
            return paths
    
    def get_entity_graph(self, entity_names: List[str], max_hops: int = 2) -> Dict:
        """Get a subgraph centered around specified entities"""
        entities_str = "', '".join([e.lower() for e in entity_names])
        
        with self.driver.session() as session:
            # Get nodes
            nodes_result = session.run(f"""
            MATCH (start:Entity)
            WHERE start.name IN ['{entities_str}']
            MATCH (related:Entity)-[*0..{max_hops}]-(start)
            RETURN related.name AS name, related.type AS type, 
                   CASE WHEN related.name IN ['{entities_str}'] THEN true ELSE false END AS is_source
            """)
            
            nodes = [{"id": record["name"], 
                      "type": record["type"], 
                      "source": record["is_source"]} 
                    for record in nodes_result]
            
            # Get edges
            edges_result = session.run(f"""
            MATCH (start:Entity)
            WHERE start.name IN ['{entities_str}']
            MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
            WHERE (source)-[*0..{max_hops}]-(start) AND (target)-[*0..{max_hops}]-(start)
            RETURN source.name AS source, target.name AS target, 
                   r.predicate AS relationship, r.doc_id AS doc_id
            """)
            
            edges = [{"source": record["source"], 
                      "target": record["target"],
                      "relationship": record["relationship"],
                      "doc_id": record["doc_id"]}
                    for record in edges_result]
            
            return {"nodes": nodes, "edges": edges}

# Test function if script is run directly
if __name__ == "__main__":
    try:
        # Create test knowledge graph
        kg = KnowledgeGraph()
        
        # Test entity extraction
        test_text = "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California. They created the first Apple computer in 1976."
        entities = kg.extract_entities(test_text)
        print("Extracted entities:", entities)
        
        # Test relationship extraction
        relationships = kg.extract_relationships(test_text)
        print("Extracted relationships:", relationships)
        
        # Close connection
        kg.close()
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}")