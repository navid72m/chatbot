"""
A simplified knowledge graph module that doesn't rely on spaCy.
This is meant to be a temporary solution for packaging purposes.
"""
import os
import json
import logging
import re
from typing import List, Dict, Any, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """A simplified knowledge graph that doesn't rely on spaCy."""
    
    def __init__(self, storage_path: str = "knowledge_db"):
        """Initialize the simple knowledge graph.
        
        Args:
            storage_path: Directory to persist the knowledge graph.
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.graph_file = os.path.join(storage_path, "knowledge_graph.json")
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
        
        # Load existing knowledge graph if it exists
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.entities = data.get("entities", {})
                    self.relationships = data.get("relationships", [])
                logger.info(f"Loaded knowledge graph with {len(self.entities)} entities and {len(self.relationships)} relationships")
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {e}")
    
    def _save_graph(self) -> None:
        """Save the knowledge graph to disk."""
        data = {
            "entities": self.entities,
            "relationships": self.relationships
        }
        with open(self.graph_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using simple rules.
        
        Args:
            text: Input text.
            
        Returns:
            List of extracted entities.
        """
        # Simple entity extraction using capitalized words as a heuristic
        # This is a simplified approach and not as accurate as spaCy
        words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
        
        # Filter out common words that start with capital letters but aren't entities
        stop_words = {"I", "The", "A", "An", "This", "That", "These", "Those"}
        entities = [word for word in words if word not in stop_words]
        
        return list(set(entities))
    
    def add_document(self, document: str, source: str = "unknown") -> None:
        """Process a document and add entities and relationships to the knowledge graph.
        
        Args:
            document: Document text.
            source: Source of the document.
        """
        # Extract entities
        entities = self.extract_entities(document)
        
        # Add entities to the graph
        for entity in entities:
            if entity not in self.entities:
                self.entities[entity] = {
                    "name": entity,
                    "sources": [source],
                    "count": 1
                }
            else:
                if source not in self.entities[entity]["sources"]:
                    self.entities[entity]["sources"].append(source)
                self.entities[entity]["count"] += 1
        
        # Add co-occurrence relationships
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if relationship already exists
                relationship_exists = False
                for rel in self.relationships:
                    if (rel["source"] == entity1 and rel["target"] == entity2) or \
                       (rel["source"] == entity2 and rel["target"] == entity1):
                        rel["weight"] += 1
                        relationship_exists = True
                        break
                
                # Add new relationship if it doesn't exist
                if not relationship_exists:
                    self.relationships.append({
                        "source": entity1,
                        "target": entity2,
                        "type": "co-occurrence",
                        "weight": 1
                    })
        
        # Save the updated graph
        self._save_graph()
    
    def get_entity(self, entity_name: str) -> Dict[str, Any]:
        """Get information about an entity.
        
        Args:
            entity_name: Name of the entity.
            
        Returns:
            Entity information.
        """
        return self.entities.get(entity_name, {"name": entity_name, "sources": [], "count": 0})
    
    def get_related_entities(self, entity_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get entities related to the given entity.
        
        Args:
            entity_name: Name of the entity.
            max_results: Maximum number of related entities to return.
            
        Returns:
            List of related entities with relationship information.
        """
        related = []
        
        for rel in self.relationships:
            if rel["source"] == entity_name:
                related.append({
                    "entity": rel["target"],
                    "relationship": rel["type"],
                    "weight": rel["weight"]
                })
            elif rel["target"] == entity_name:
                related.append({
                    "entity": rel["source"],
                    "relationship": rel["type"],
                    "weight": rel["weight"]
                })
        
        # Sort by weight and limit results
        related.sort(key=lambda x: x["weight"], reverse=True)
        return related[:max_results]
    
    def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """Search for entities matching the query.
        
        Args:
            query: Search query.
            
        Returns:
            List of matching entities.
        """
        query = query.lower()
        results = []
        
        for entity_name, entity_data in self.entities.items():
            if query in entity_name.lower():
                results.append(entity_data)
        
        # Sort by count (popularity)
        results.sort(key=lambda x: x["count"], reverse=True)
        return results

# For backwards compatibility with existing code
KnowledgeGraph = KnowledgeGraph