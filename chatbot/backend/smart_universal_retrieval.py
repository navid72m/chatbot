import os
import re
import logging
import spacy
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict, Counter
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SmartEntity:
    """Universal entity representation for any document type."""
    text: str
    label: str  # PERSON, ORG, DATE, SKILL, etc.
    confidence: float
    start_pos: int
    end_pos: int
    context: str
    document_type: str = "general"

class UniversalEntityExtractor:
    """
    Smart entity extractor that works for any document type.
    Automatically detects document type and applies appropriate extraction strategies.
    """
    
    def __init__(self):
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Universal patterns for all document types
        self.universal_patterns = {
            'person_names': [
                # Pattern 1: Capitalized names (2-3 words)
                r'\b([A-Z][a-z]{1,15}\s+[A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)\b',
                # Pattern 2: ALL CAPS names
                r'\b([A-Z]{2,15}\s+[A-Z]{2,15}(?:\s+[A-Z]{2,15})?)\b',
                # Pattern 3: Names with titles
                r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
                # Pattern 4: Names in signature contexts
                r'(?:sincerely|regards|best regards),?\s*[\n\r]\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            ],
            'contact_info': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',  # Phone
                r'(?:linkedin\.com/in/|linkedin\.com/pub/)([a-zA-Z0-9\-]+)',  # LinkedIn
            ],
            'organizations': [
                r'\b([A-Z][a-zA-Z\s&.,]+?)\s+(?:Inc|LLC|Corp|Company|Technologies|University|College|Institute)\b',
                r'\b(?:at|@)\s+([A-Z][a-zA-Z\s&.,]+?)(?:\s*[-–—]\s*|\s*,|\s*\n|\s*$)',
            ],
            'dates': [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
                r'\b\d{4}[-–—]\d{4}\b',  # Year ranges
            ],
            'skills_tech': [
                # Programming languages
                r'\b(?:Python|JavaScript|Java|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin|TypeScript|Scala|R|MATLAB)\b',
                # Frameworks
                r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel|Rails|ASP\.NET)\b',
                # Databases
                r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|SQLite|Oracle|SQL Server)\b',
                # Cloud/DevOps
                r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|GitHub|GitLab|Terraform)\b',
                # AI/ML
                r'\b(?:TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy|OpenCV|Keras|NLTK)\b',
            ]
        }
        
        # Document type indicators
        self.doc_type_indicators = {
            'resume': ['objective', 'summary', 'experience', 'education', 'skills', 'employment', 'qualification'],
            'report': ['executive summary', 'methodology', 'findings', 'conclusion', 'analysis', 'research'],
            'contract': ['agreement', 'party', 'whereas', 'terms', 'conditions', 'obligations'],
            'manual': ['instructions', 'procedure', 'step', 'guide', 'tutorial', 'how to'],
            'academic': ['abstract', 'introduction', 'literature review', 'methodology', 'results', 'discussion']
        }
        
        # Words that are definitely NOT names
        self.non_name_words = {
            'resume', 'curriculum', 'vitae', 'objective', 'summary', 'experience', 'education',
            'skills', 'employment', 'work', 'history', 'professional', 'personal', 'career',
            'contact', 'information', 'phone', 'email', 'address', 'linkedin', 'github',
            'references', 'available', 'request', 'upon', 'thank', 'you', 'sincerely'
        }
    
    def detect_document_type(self, text: str) -> str:
        """Automatically detect document type based on content."""
        text_lower = text.lower()
        
        type_scores = {}
        for doc_type, indicators in self.doc_type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            type_scores[doc_type] = score
        
        # Return the type with highest score, or 'general' if none
        if type_scores:
            detected_type = max(type_scores, key=type_scores.get)
            if type_scores[detected_type] >= 2:  # Minimum threshold
                return detected_type
        
        return 'general'
    
    def extract_smart_entities(self, text: str, document_type: str = None) -> List[SmartEntity]:
        """Extract entities using multiple intelligent approaches."""
        
        if document_type is None:
            document_type = self.detect_document_type(text)
        
        logger.info(f"Extracting entities for document type: {document_type}")
        
        entities = []
        
        # Method 1: Pattern-based extraction
        pattern_entities = self._extract_with_patterns(text, document_type)
        entities.extend(pattern_entities)
        
        # Method 2: spaCy NER (if available)
        if self.nlp:
            spacy_entities = self._extract_with_spacy(text, document_type)
            entities.extend(spacy_entities)
        
        # Method 3: Context-aware extraction
        context_entities = self._extract_contextual_entities(text, document_type)
        entities.extend(context_entities)
        
        # Method 4: Position-based extraction (for structured docs)
        position_entities = self._extract_positional_entities(text, document_type)
        entities.extend(position_entities)
        
        # Deduplicate and rank
        final_entities = self._deduplicate_and_rank(entities)
        
        return final_entities
    
    def _extract_with_patterns(self, text: str, document_type: str) -> List[SmartEntity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for category, patterns in self.universal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    entity_text = entity_text.strip()
                    
                    if self._is_valid_entity(entity_text, category, document_type):
                        # Get context
                        start_pos = max(0, match.start() - 50)
                        end_pos = min(len(text), match.end() + 50)
                        context = text[start_pos:end_pos].strip()
                        
                        # Determine label
                        label = self._categorize_entity(entity_text, category, context, document_type)
                        
                        # Calculate confidence
                        confidence = self._calculate_confidence(entity_text, category, context, document_type)
                        
                        entities.append(SmartEntity(
                            text=entity_text,
                            label=label,
                            confidence=confidence,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            context=context,
                            document_type=document_type
                        ))
        
        return entities
    
    def _extract_with_spacy(self, text: str, document_type: str) -> List[SmartEntity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Skip very short or very long entities
                if len(ent.text.strip()) < 2 or len(ent.text.strip()) > 100:
                    continue
                
                # Get context
                start_token = max(0, ent.start - 10)
                end_token = min(len(doc), ent.end + 10)
                context = doc[start_token:end_token].text
                
                # Enhance spaCy labels with document context
                enhanced_label = self._enhance_spacy_label(ent.label_, ent.text, document_type)
                
                entities.append(SmartEntity(
                    text=ent.text.strip(),
                    label=enhanced_label,
                    confidence=0.75,  # Base confidence for spaCy
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    context=context,
                    document_type=document_type
                ))
        
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")
        
        return entities
    
    def _extract_contextual_entities(self, text: str, document_type: str) -> List[SmartEntity]:
        """Extract entities based on document structure and context."""
        entities = []
        lines = text.split('\n')
        
        # For any document, look for names in key positions
        if document_type in ['resume', 'general']:
            # Check first few lines for names (common in many document types)
            for i, line in enumerate(lines[:5]):
                line = line.strip()
                if self._looks_like_person_name(line, document_type):
                    entities.append(SmartEntity(
                        text=line,
                        label="PERSON",
                        confidence=0.85 - (i * 0.1),  # Higher confidence for earlier lines
                        start_pos=text.find(line),
                        end_pos=text.find(line) + len(line),
                        context=f"Found in line {i+1} of document header",
                        document_type=document_type
                    ))
        
        # Extract section headers as potential topics/categories
        for line in lines:
            if self._looks_like_section_header(line):
                entities.append(SmartEntity(
                    text=line.strip(),
                    label="SECTION_HEADER",
                    confidence=0.6,
                    start_pos=text.find(line),
                    end_pos=text.find(line) + len(line),
                    context="Document section header",
                    document_type=document_type
                ))
        
        return entities
    
    def _extract_positional_entities(self, text: str, document_type: str) -> List[SmartEntity]:
        """Extract entities based on their position in structured documents."""
        entities = []
        
        if document_type == 'resume':
            # Look for names at document start
            first_line = text.split('\n')[0].strip() if text.split('\n') else ""
            if self._looks_like_person_name(first_line, document_type):
                entities.append(SmartEntity(
                    text=first_line,
                    label="PERSON_PRIMARY",
                    confidence=0.95,
                    start_pos=0,
                    end_pos=len(first_line),
                    context="Primary name at document start",
                    document_type=document_type
                ))
        
        # Look for key-value pairs (Name: John Doe)
        kv_pattern = r'([A-Za-z\s]+):\s*([A-Za-z0-9\s@.\-]+)'
        matches = re.finditer(kv_pattern, text)
        
        for match in matches:
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            
            if key in ['name', 'full name', 'candidate name']:
                entities.append(SmartEntity(
                    text=value,
                    label="PERSON",
                    confidence=0.9,
                    start_pos=match.start(2),
                    end_pos=match.end(2),
                    context=f"Extracted from '{key}:' field",
                    document_type=document_type
                ))
        
        return entities
    
    def _looks_like_person_name(self, text: str, document_type: str) -> bool:
        """Enhanced name detection logic."""
        if not text or len(text.strip()) < 3:
            return False
        
        # Clean text
        text = re.sub(r'[•·▪▫■□○●\-_=]+', '', text).strip()
        words = text.split()
        
        # Must be 2-3 words for names
        if len(words) < 2 or len(words) > 3:
            return False
        
        # Each word should be capitalized and mostly alphabetic
        for word in words:
            if not word[0].isupper():
                return False
            if not re.match(r"^[A-Za-z'\-]{2,20}$", word):
                return False
        
        # Check against non-name words
        if any(word.lower() in self.non_name_words for word in words):
            return False
        
        # Additional context-based validation
        text_lower = text.lower()
        
        # Should not contain obvious non-name indicators
        non_name_indicators = ['@', '.com', 'phone', 'email', 'address', ':', ';', '|', 'www']
        if any(indicator in text_lower for indicator in non_name_indicators):
            return False
        
        return True
    
    def _looks_like_section_header(self, line: str) -> bool:
        """Check if line looks like a section header."""
        line = line.strip()
        
        # Must be short enough to be a header
        if len(line) < 3 or len(line) > 50:
            return False
        
        # Should be mostly uppercase or title case
        if line.isupper() or line.istitle():
            return True
        
        # Check for common section patterns
        if line.endswith(':') and len(line.split()) <= 3:
            return True
        
        return False
    
    def _is_valid_entity(self, text: str, category: str, document_type: str) -> bool:
        """Validate if extracted text is a valid entity."""
        if not text or len(text.strip()) < 2:
            return False
        
        text_clean = text.strip()
        
        if category == 'person_names':
            return self._looks_like_person_name(text_clean, document_type)
        elif category == 'contact_info':
            return True  # Contact patterns are already strict
        elif category == 'skills_tech':
            return len(text_clean) >= 2 and len(text_clean) <= 30
        elif category == 'organizations':
            return len(text_clean) >= 3 and len(text_clean) <= 50
        
        return True
    
    def _categorize_entity(self, text: str, pattern_category: str, context: str, document_type: str) -> str:
        """Determine the specific label for an entity."""
        
        if pattern_category == 'person_names':
            # Check context for more specific labeling
            if any(word in context.lower() for word in ['author', 'by', 'written']):
                return "AUTHOR"
            elif document_type == 'resume':
                return "PERSON_CANDIDATE"
            else:
                return "PERSON"
        
        elif pattern_category == 'contact_info':
            if '@' in text:
                return "EMAIL"
            elif re.search(r'\d{3}.*\d{3}.*\d{4}', text):
                return "PHONE"
            elif 'linkedin' in text.lower():
                return "LINKEDIN"
            else:
                return "CONTACT"
        
        elif pattern_category == 'organizations':
            if any(word in text.lower() for word in ['university', 'college', 'institute']):
                return "EDUCATION_ORG"
            else:
                return "ORGANIZATION"
        
        elif pattern_category == 'skills_tech':
            return "SKILL_TECHNICAL"
        
        elif pattern_category == 'dates':
            return "DATE"
        
        return pattern_category.upper()
    
    def _calculate_confidence(self, text: str, category: str, context: str, document_type: str) -> float:
        """Calculate confidence score for entity extraction."""
        base_confidence = 0.6
        
        # Boost confidence based on category
        if category == 'contact_info':
            base_confidence = 0.9  # Contact patterns are very reliable
        elif category == 'person_names':
            base_confidence = 0.7
            # Boost if found in typical name positions
            if 'line 1' in context or 'document start' in context:
                base_confidence += 0.2
        
        # Boost based on document type context
        if document_type == 'resume' and category == 'person_names':
            base_confidence += 0.1
        
        # Penalize very short or very long text
        if len(text) < 3:
            base_confidence -= 0.2
        elif len(text) > 50:
            base_confidence -= 0.1
        
        return min(1.0, max(0.1, base_confidence))
    
    def _enhance_spacy_label(self, spacy_label: str, text: str, document_type: str) -> str:
        """Enhance spaCy labels with context."""
        
        label_mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',
            'DATE': 'DATE',
            'MONEY': 'MONEY',
            'PRODUCT': 'PRODUCT'
        }
        
        enhanced_label = label_mapping.get(spacy_label, spacy_label)
        
        # Add document-specific enhancements
        if enhanced_label == 'PERSON' and document_type == 'resume':
            enhanced_label = 'PERSON_CANDIDATE'
        
        return enhanced_label
    
    def _deduplicate_and_rank(self, entities: List[SmartEntity]) -> List[SmartEntity]:
        """Remove duplicates and rank entities by confidence."""
        
        # Group similar entities
        groups = defaultdict(list)
        
        for entity in entities:
            # Create key for grouping (normalized text + label)
            key = (entity.text.lower().strip(), entity.label)
            groups[key].append(entity)
        
        # Select best entity from each group
        final_entities = []
        for group_entities in groups.values():
            # Sort by confidence and take the best
            best_entity = max(group_entities, key=lambda x: x.confidence)
            final_entities.append(best_entity)
        
        # Sort all entities by confidence
        final_entities.sort(key=lambda x: x.confidence, reverse=True)
        
        return final_entities


class SmartChunker:
    """
    Intelligent chunking that creates specialized chunks based on content and entities.
    """
    
    def __init__(self, entity_extractor: UniversalEntityExtractor):
        self.entity_extractor = entity_extractor
    
    def create_smart_chunks(self, text: str, document_type: str = None, 
                          chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """Create intelligent chunks with entity awareness."""
        
        # Detect document type if not provided
        if document_type is None:
            document_type = self.entity_extractor.detect_document_type(text)
        
        # Extract entities
        entities = self.entity_extractor.extract_smart_entities(text, document_type)
        
        chunks = []
        
        # Chunk 1: Smart Summary with key entities
        summary_chunk = self._create_smart_summary(text, entities, document_type)
        chunks.append(summary_chunk)
        
        # Chunk 2: Entity-focused chunks for high-priority entities
        entity_chunks = self._create_entity_focused_chunks(text, entities)
        chunks.extend(entity_chunks)
        
        # Chunk 3: Regular content chunks with entity awareness
        content_chunks = self._create_smart_content_chunks(text, entities, chunk_size, overlap)
        chunks.extend(content_chunks)
        
        # Chunk 4: FAQ-style chunks for common questions
        faq_chunks = self._create_faq_chunks(entities, document_type)
        chunks.extend(faq_chunks)
        
        return chunks
    
    def _create_smart_summary(self, text: str, entities: List[SmartEntity], document_type: str) -> Dict[str, Any]:
        """Create an intelligent summary chunk."""
        
        summary_parts = []
        
        # Find primary person (highest confidence PERSON entity)
        primary_person = self._get_primary_person(entities)
        
        if document_type == 'resume':
            if primary_person:
                summary_parts.extend([
                    f"This is the resume of {primary_person.text}.",
                    f"The candidate's name is {primary_person.text}.",
                    f"This CV belongs to {primary_person.text}."
                ])
            
            # Add contact info
            contact_entities = [e for e in entities if e.label in ['EMAIL', 'PHONE', 'LINKEDIN']]
            for contact in contact_entities[:2]:  # Top 2 contact methods
                summary_parts.append(f"{contact.label.title()}: {contact.text}")
            
            # Add top skills
            skill_entities = [e for e in entities if 'SKILL' in e.label]
            if skill_entities:
                skills_text = ", ".join([e.text for e in skill_entities[:5]])
                summary_parts.append(f"Key skills: {skills_text}")
        
        else:
            # General document summary
            if primary_person:
                summary_parts.append(f"Key person mentioned: {primary_person.text}")
            
            # Add organizations
            org_entities = [e for e in entities if 'ORG' in e.label]
            if org_entities:
                orgs_text = ", ".join([e.text for e in org_entities[:3]])
                summary_parts.append(f"Organizations: {orgs_text}")
        
        summary_text = " ".join(summary_parts) if summary_parts else f"This is a {document_type} document."
        
        return {
            "text": summary_text,
            "type": "smart_summary",
            "entities": entities,
            "metadata": {
                "priority": "highest",
                "document_type": document_type,
                "entity_count": len(entities),
                "has_primary_person": primary_person is not None
            }
        }
    
    def _create_entity_focused_chunks(self, text: str, entities: List[SmartEntity]) -> List[Dict[str, Any]]:
        """Create chunks focused on high-priority entities."""
        chunks = []
        
        # Focus on high-confidence, important entities
        priority_entities = [e for e in entities if e.confidence > 0.8 and 
                           e.label in ['PERSON', 'PERSON_CANDIDATE', 'PERSON_PRIMARY', 'EMAIL', 'PHONE']]
        
        for entity in priority_entities[:3]:  # Top 3 priority entities
            # Create extended context around entity
            start_pos = max(0, entity.start_pos - 300)
            end_pos = min(len(text), entity.end_pos + 300)
            chunk_text = text[start_pos:end_pos].strip()
            
            chunks.append({
                "text": chunk_text,
                "type": "entity_focused",
                "primary_entity": entity,
                "metadata": {
                    "priority": "high",
                    "entity_type": entity.label,
                    "entity_text": entity.text,
                    "confidence": entity.confidence
                }
            })
        
        return chunks
    
    def _create_smart_content_chunks(self, text: str, entities: List[SmartEntity], 
                                   chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Create content chunks that respect entity boundaries."""
        chunks = []
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        current_entities = []
        
        for sentence in sentences:
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Create chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "type": "smart_content",
                    "entities": current_entities.copy(),
                    "metadata": {
                        "priority": "medium",
                        "entity_count": len(current_entities),
                        "has_entities": len(current_entities) > 0
                    }
                })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-overlap//4:] if len(words) > overlap//4 else words
                current_chunk = " ".join(overlap_words) + " " + sentence
                current_entities = []
            else:
                current_chunk += " " + sentence if current_chunk else sentence
            
            # Find entities in current sentence
            sentence_entities = [e for e in entities if sentence in e.context]
            current_entities.extend(sentence_entities)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "type": "smart_content",
                "entities": current_entities,
                "metadata": {
                    "priority": "medium",
                    "entity_count": len(current_entities),
                    "has_entities": len(current_entities) > 0
                }
            })
        
        return chunks
    
    def _create_faq_chunks(self, entities: List[SmartEntity], document_type: str) -> List[Dict[str, Any]]:
        """Create FAQ-style chunks for common questions."""
        chunks = []
        
        # Get primary person
        primary_person = self._get_primary_person(entities)
        
        if document_type == 'resume' and primary_person:
            # Create FAQ chunk for name-related questions
            faq_content = [
                f"Q: Whose resume is this? A: This resume belongs to {primary_person.text}.",
                f"Q: What is the candidate's name? A: The candidate's name is {primary_person.text}.",
                f"Q: Who is this CV for? A: This CV is for {primary_person.text}."
            ]
            
            # Add contact info to FAQ
            contact_entities = [e for e in entities if e.label in ['EMAIL', 'PHONE']]
            for contact in contact_entities:
                faq_content.append(f"Q: What is the contact {contact.label.lower()}? A: {contact.text}")
            
            chunks.append({
                "text": " ".join(faq_content),
                "type": "faq",
                "metadata": {
                    "priority": "high",
                    "question_types": ["identity", "contact"],
                    "primary_person": primary_person.text
                }
            })
        
        return chunks
    
    def _get_primary_person(self, entities: List[SmartEntity]) -> Optional[SmartEntity]:
        """Get the primary person from entities."""
        person_entities = [e for e in entities if 'PERSON' in e.label]
        
        if person_entities:
            # Sort by confidence and specific labels
            person_entities.sort(key=lambda x: (
                x.confidence,
                1 if x.label == 'PERSON_PRIMARY' else 0,
                1 if x.label == 'PERSON_CANDIDATE' else 0
            ), reverse=True)
            
            return person_entities[0]
        
        return None


class SmartVectorStore:
    """
    Enhanced vector store with intelligent search capabilities.
    """
    
    def __init__(self, embed_model: SentenceTransformer = None):
        self.embed_model = embed_model or SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Storage
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        
        # Smart indices
        self.entity_index = defaultdict(list)  # entity_text -> chunk_indices
        self.type_index = defaultdict(list)    # chunk_type -> chunk_indices
        self.priority_index = defaultdict(list) # priority -> chunk_indices
        
        # Load reranker if available
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except:
            self.reranker = None
    
    def add_smart_chunks(self, chunks: List[Dict[str, Any]], document_name: str):
        """Add smart chunks with comprehensive indexing."""
        
        start_idx = len(self.chunks)
        
        for i, chunk in enumerate(chunks):
            chunk_idx = start_idx + i
            
            # Store chunk
            self.chunks.append(chunk)
            
            # Create embedding
            embedding = self.embed_model.encode(chunk["text"])
            self.embeddings.append(embedding)
            
            # Store metadata
            chunk_metadata = {
                "document": document_name,
                "chunk_id": chunk_idx,
                "type": chunk.get("type", "content"),
                **chunk.get("metadata", {})
            }
            self.metadata.append(chunk_metadata)
            
            # Index by type
            chunk_type = chunk.get("type", "content")
            self.type_index[chunk_type].append(chunk_idx)
            
            # Index by priority
            priority = chunk.get("metadata", {}).get("priority", "medium")
            self.priority_index[priority].append(chunk_idx)
            
            # Index entities
            entities = chunk.get("entities", [])
            for entity in entities:
                self.entity_index[entity.text.lower()].append(chunk_idx)
            
            # Index primary entity if exists
            primary_entity = chunk.get("primary_entity")
            if primary_entity:
                self.entity_index[primary_entity.text.lower()].append(chunk_idx)
    
    def smart_search(self, query: str, document_name: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Intelligent search that adapts to query type and content."""
        
        # Analyze query
        query_analysis = self._analyze_query(query)
        
        # Get candidate chunks using multiple strategies
        candidates = []
        
        # Strategy 1: Priority-based retrieval for factual queries
        if query_analysis["is_factual"]:
            priority_candidates = self._get_priority_chunks(document_name, query_analysis["query_type"])
            candidates.extend(priority_candidates)
        
        # Strategy 2: Entity-based retrieval
        entity_candidates = self._get_entity_matching_chunks(query, document_name)
        candidates.extend(entity_candidates)
        
        # Strategy 3: Semantic similarity search
        semantic_candidates = self._get_semantic_candidates(query, document_name, top_k * 2)
        candidates.extend(semantic_candidates)
        
        # Strategy 4: Type-specific retrieval
        type_candidates = self._get_type_specific_chunks(query_analysis["query_type"], document_name)
        candidates.extend(type_candidates)
        
        # Deduplicate candidates
        unique_candidates = self._deduplicate_candidates(candidates)
        
        # Score and rank candidates
        scored_candidates = self._score_candidates(query, unique_candidates, query_analysis)
        
        # Rerank if reranker is available
        if self.reranker and len(scored_candidates) > 1:
            scored_candidates = self._rerank_candidates(query, scored_candidates)
        
        # Return top_k results
        return scored_candidates[:top_k]
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine search strategy."""
        query_lower = query.lower()
        
        # Determine if factual query
        factual_keywords = ['who', 'whose', 'what', 'where', 'when', 'which', 'name', 'contact']
        is_factual = any(keyword in query_lower for keyword in factual_keywords)
        
        # Determine query type
        query_type = "general"
        if any(word in query_lower for word in ['whose', 'who', 'name']):
            query_type = "identity"
        elif any(word in query_lower for word in ['contact', 'email', 'phone']):
            query_type = "contact"
        elif any(word in query_lower for word in ['skill', 'technology', 'experience']):
            query_type = "skills"
        elif any(word in query_lower for word in ['education', 'degree', 'university']):
            query_type = "education"
        elif any(word in query_lower for word in ['work', 'job', 'employment']):
            query_type = "experience"
        
        # Check for entity mentions in query
        mentioned_entities = []
        for entity_text in self.entity_index.keys():
            if entity_text in query_lower:
                mentioned_entities.append(entity_text)
        
        return {
            "is_factual": is_factual,
            "query_type": query_type,
            "mentioned_entities": mentioned_entities,
            "keywords": query_lower.split()
        }
    
    def _get_priority_chunks(self, document_name: str, query_type: str) -> List[Tuple[int, float, str]]:
        """Get high-priority chunks for factual queries."""
        candidates = []
        
        # Always prioritize smart_summary and faq chunks for factual queries
        for chunk_type in ["smart_summary", "faq", "entity_focused"]:
            if chunk_type in self.type_index:
                for chunk_idx in self.type_index[chunk_type]:
                    if self.metadata[chunk_idx]["document"] == document_name:
                        score = 0.9 if chunk_type == "smart_summary" else 0.85
                        candidates.append((chunk_idx, score, "priority"))
        
        return candidates
    
    def _get_entity_matching_chunks(self, query: str, document_name: str) -> List[Tuple[int, float, str]]:
        """Get chunks that contain entities mentioned in the query."""
        candidates = []
        query_lower = query.lower()
        
        for entity_text, chunk_indices in self.entity_index.items():
            if entity_text in query_lower:
                for chunk_idx in chunk_indices:
                    if self.metadata[chunk_idx]["document"] == document_name:
                        # Higher score for exact entity matches
                        score = 0.8
                        candidates.append((chunk_idx, score, "entity_match"))
        
        return candidates
    
    def _get_semantic_candidates(self, query: str, document_name: str, top_k: int) -> List[Tuple[int, float, str]]:
        """Get candidates using semantic similarity."""
        query_embedding = self.embed_model.encode(query)
        
        similarities = []
        for i, (chunk, embedding, metadata) in enumerate(zip(self.chunks, self.embeddings, self.metadata)):
            if metadata["document"] != document_name:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            similarities.append((i, float(similarity), "semantic"))
        
        # Sort by similarity and return top candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _get_type_specific_chunks(self, query_type: str, document_name: str) -> List[Tuple[int, float, str]]:
        """Get chunks of specific types based on query."""
        candidates = []
        
        # Map query types to preferred chunk types
        type_preferences = {
            "identity": ["smart_summary", "faq", "entity_focused"],
            "contact": ["smart_summary", "entity_focused"],
            "skills": ["smart_content"],
            "experience": ["smart_content"],
            "education": ["smart_content"]
        }
        
        preferred_types = type_preferences.get(query_type, ["smart_content"])
        
        for chunk_type in preferred_types:
            if chunk_type in self.type_index:
                for chunk_idx in self.type_index[chunk_type]:
                    if self.metadata[chunk_idx]["document"] == document_name:
                        score = 0.7
                        candidates.append((chunk_idx, score, "type_specific"))
        
        return candidates
    
    def _deduplicate_candidates(self, candidates: List[Tuple[int, float, str]]) -> List[Tuple[int, float, str]]:
        """Remove duplicate chunk indices, keeping highest scores."""
        seen = {}
        
        for chunk_idx, score, source in candidates:
            if chunk_idx not in seen or score > seen[chunk_idx][1]:
                seen[chunk_idx] = (chunk_idx, score, source)
        
        return list(seen.values())
    
    def _score_candidates(self, query: str, candidates: List[Tuple[int, float, str]], 
                         query_analysis: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """Score and rank candidates using multiple factors."""
        
        scored_results = []
        
        for chunk_idx, base_score, source in candidates:
            chunk = self.chunks[chunk_idx]
            metadata = self.metadata[chunk_idx]
            
            # Start with base score
            final_score = base_score
            
            # Boost for query type alignment
            if query_analysis["query_type"] == "identity" and chunk.get("type") in ["smart_summary", "faq"]:
                final_score += 0.2
            
            # Boost for entity matches
            chunk_entities = chunk.get("entities", [])
            for entity in chunk_entities:
                if entity.text.lower() in query.lower():
                    final_score += 0.1 * entity.confidence
            
            # Boost for primary entity matches
            primary_entity = chunk.get("primary_entity")
            if primary_entity and primary_entity.text.lower() in query.lower():
                final_score += 0.15
            
            # Boost for high-priority chunks
            if metadata.get("priority") == "highest":
                final_score += 0.1
            elif metadata.get("priority") == "high":
                final_score += 0.05
            
            # Penalize very long chunks for factual queries
            if query_analysis["is_factual"] and len(chunk["text"]) > 1000:
                final_score -= 0.05
            
            scored_results.append((chunk, min(1.0, final_score)))
        
        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results
    
    def _rerank_candidates(self, query: str, candidates: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """Rerank candidates using cross-encoder."""
        if not self.reranker or len(candidates) <= 1:
            return candidates
        
        try:
            # Prepare pairs for reranking
            pairs = []
            for chunk, _ in candidates:
                pairs.append((query, chunk["text"]))
            
            # Get reranker scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine original scores with rerank scores
            reranked = []
            for i, (chunk, original_score) in enumerate(candidates):
                # Weighted combination of original and rerank scores
                final_score = 0.7 * original_score + 0.3 * rerank_scores[i]
                reranked.append((chunk, final_score))
            
            # Sort by combined score
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates


class SmartRAGPipeline:
    """
    Complete smart RAG pipeline that works universally for all document types.
    """
    
    def __init__(self):
        self.entity_extractor = UniversalEntityExtractor()
        self.chunker = SmartChunker(self.entity_extractor)
        self.vector_store = SmartVectorStore()
        
        # Statistics
        self.processing_stats = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "chunks_created": 0
        }
    
    def process_document_smart(self, file_path: str, text: str = None) -> Dict[str, Any]:
        """Process any document with smart extraction and chunking."""
        
        if text is None:
            # Use existing text extraction logic
            text = self._extract_text_from_file(file_path)
        
        # Detect document type
        document_type = self.entity_extractor.detect_document_type(text)
        
        # Extract entities
        entities = self.entity_extractor.extract_smart_entities(text, document_type)
        
        # Create smart chunks
        chunks = self.chunker.create_smart_chunks(text, document_type)
        
        # Add to vector store
        document_name = os.path.basename(file_path)
        self.vector_store.add_smart_chunks(chunks, document_name)
        
        # Update statistics
        self.processing_stats["documents_processed"] += 1
        self.processing_stats["entities_extracted"] += len(entities)
        self.processing_stats["chunks_created"] += len(chunks)
        
        # Generate smart suggestions
        suggestions = self._generate_smart_suggestions(entities, document_type)
        
        return {
            "document_name": document_name,
            "document_type": document_type,
            "entities_found": len(entities),
            "key_entities": entities[:10],
            "chunks_created": len(chunks),
            "suggestions": suggestions,
            "processing_successful": True
        }
    
    def query_smart(self, query: str, document_name: str, top_k: int = 5) -> Dict[str, Any]:
        """Smart query processing with adaptive retrieval."""
        
        # Perform smart search
        results = self.vector_store.smart_search(query, document_name, top_k)
        
        # Build enhanced context
        context_parts = []
        for i, (chunk, score) in enumerate(results):
            context_part = f"[Chunk {i+1}, Relevance: {score:.3f}, Type: {chunk.get('type', 'content')}]\n{chunk['text']}"
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        
        # Analyze query for metadata
        query_analysis = self.vector_store._analyze_query(query)
        
        # Extract entity information from results
        entity_info = self._extract_entity_info_from_results(results)
        
        return {
            "context": context,
            "chunks_retrieved": results,
            "query_analysis": query_analysis,
            "entity_info": entity_info,
            "total_chunks_searched": len(self.vector_store.chunks),
            "retrieval_strategy": "smart_adaptive"
        }
    
    def _generate_smart_suggestions(self, entities: List[SmartEntity], document_type: str) -> List[str]:
        """Generate intelligent suggestions based on extracted entities."""
        suggestions = []
        
        # Find primary person
        primary_person = None
        for entity in entities:
            if 'PERSON' in entity.label and entity.confidence > 0.7:
                primary_person = entity
                break
        
        if document_type == 'resume':
            if primary_person:
                suggestions.extend([
                    "Whose resume is this?",
                    "What is the candidate's name?",
                    f"What are {primary_person.text}'s qualifications?",
                    f"What skills does {primary_person.text} have?",
                    f"What is {primary_person.text}'s contact information?"
                ])
            else:
                suggestions.extend([
                    "Whose CV is this?",
                    "What is the candidate's name?",
                    "What are the main qualifications?",
                    "What skills are mentioned?",
                    "What contact information is provided?"
                ])
        
        elif document_type == 'report':
            suggestions.extend([
                "What is the main topic of this report?",
                "Who are the key people mentioned?",
                "What are the main findings?",
                "What methodology was used?",
                "What are the conclusions?"
            ])
        
        else:  # General document
            if primary_person:
                suggestions.append(f"Who is {primary_person.text}?")
            
            suggestions.extend([
                "What is this document about?",
                "Who are the key people mentioned?",
                "What are the main topics?",
                "What organizations are mentioned?",
                "What important dates are referenced?"
            ])
        
        # Add entity-specific suggestions
        skill_entities = [e for e in entities if 'SKILL' in e.label]
        if skill_entities:
            suggestions.append("What technical skills are mentioned?")
        
        org_entities = [e for e in entities if 'ORG' in e.label]
        if org_entities:
            suggestions.append("What organizations are referenced?")
        
        return suggestions[:5]
    
    def _extract_entity_info_from_results(self, results: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Extract entity information from search results."""
        
        all_entities = []
        entity_types = set()
        
        for chunk, score in results:
            chunk_entities = chunk.get("entities", [])
            all_entities.extend(chunk_entities)
            
            for entity in chunk_entities:
                entity_types.add(entity.label)
        
        # Find highest confidence entities by type
        best_entities = {}
        for entity in all_entities:
            if entity.label not in best_entities or entity.confidence > best_entities[entity.label].confidence:
                best_entities[entity.label] = entity
        
        return {
            "total_entities": len(all_entities),
            "entity_types": list(entity_types),
            "best_entities": {label: entity.text for label, entity in best_entities.items()},
            "has_person": any('PERSON' in label for label in entity_types),
            "has_contact": any(label in ['EMAIL', 'PHONE'] for label in entity_types)
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from file - placeholder for your existing implementation."""
        # This should call your existing text extraction logic
        # For now, return empty string as placeholder
        return ""