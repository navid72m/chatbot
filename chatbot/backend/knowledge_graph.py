# knowledge_graph.py

import spacy
import networkx as nx
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def extract_entities_and_relations(self, text: str) -> List[Tuple[str, str, str]]:
        doc = nlp(text)
        triples = []
        for sent in doc.sents:
            ents = [ent.text.strip() for ent in sent.ents if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LOC", "FAC")]
            if len(ents) >= 2:
                for i in range(len(ents) - 1):
                    triples.append((ents[i], "related_to", ents[i+1]))
        logger.info(f"Triples: {triples}")
        return triples

    def add_document(self, text: str):
        triples = self.extract_entities_and_relations(text)
        for subj, rel, obj in triples:
            self.graph.add_edge(subj, obj, relation=rel)
        logger.info(f"Graph: {self.graph.edges()}")

    def get_neighbors(self, entity: str) -> List[str]:
        return list(self.graph.neighbors(entity)) if entity in self.graph else []

    def get_all_triples(self) -> List[Tuple[str, str, str]]:
        return [(u, d['relation'], v) for u, v, d in self.graph.edges(data=True)]
    def get_kg_context(self, query: str, max_relations: int = 5) -> str:
        doc = nlp(query)
        entity_contexts = []
        seen = set()

        for ent in doc.ents:
            entity = ent.text.strip()
            if entity not in self.graph or entity in seen:
                continue
            seen.add(entity)
            neighbors = self.get_neighbors(entity)
            for neighbor in neighbors[:max_relations]:
                relation = self.graph[entity][neighbor]['relation']
                entity_contexts.append(f"{entity} {relation} {neighbor}.")

        return " ".join(entity_contexts[:max_relations])
