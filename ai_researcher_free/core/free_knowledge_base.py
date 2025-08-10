"""
Free Knowledge Base System
Uses SQLite for lightweight, free storage and processing
"""

import sqlite3
import json
import arxiv
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import gc
import re
import logging
from datetime import datetime, timedelta
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeKnowledgeBase:
    """
    Free knowledge base for research papers and concepts
    Uses only free resources: SQLite, arXiv, free embeddings
    """
    
    def __init__(self, db_path: str = "data/knowledge_base.db"):
        self.db_path = db_path
        self.embedder = None
        self.setup_database()
        self._load_embedder()
        
    def _load_embedder(self):
        """Load lightweight sentence transformer (free)"""
        try:
            logger.info("Loading free embedding model...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedder: {e}")
            self.embedder = None
    
    def setup_database(self):
        """Setup SQLite database schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Papers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT,
                authors TEXT,
                published TEXT,
                categories TEXT,
                url TEXT,
                embedding BLOB,
                processed_date TEXT,
                citation_count INTEGER DEFAULT 0
            )
        ''')
        
        # Concepts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT UNIQUE,
                frequency INTEGER DEFAULT 1,
                category TEXT,
                first_seen TEXT,
                last_updated TEXT
            )
        ''')
        
        # Paper-concept relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_concepts (
                paper_id TEXT,
                concept TEXT,
                relevance_score REAL DEFAULT 1.0,
                FOREIGN KEY (paper_id) REFERENCES papers (id)
            )
        ''')
        
        # Research gaps table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gap_description TEXT,
                domain TEXT,
                evidence TEXT,
                confidence_score REAL,
                identified_date TEXT
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_categories ON papers(categories)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_category ON concepts(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_concepts_paper ON paper_concepts(paper_id)')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database setup complete")
    
    def ingest_literature(self, query: str, max_papers: int = 50, 
                         batch_size: int = 10) -> Dict[str, Any]:
        """
        Ingest research papers from arXiv (free)
        
        Args:
            query: Search query for arXiv
            max_papers: Maximum number of papers to download
            batch_size: Process papers in batches to manage memory
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting literature ingestion for query: '{query}'")
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers_processed = 0
            papers_added = 0
            papers_updated = 0
            batch = []
            
            for paper in search.results():
                paper_data = {
                    'id': paper.entry_id,
                    'title': paper.title.strip(),
                    'abstract': paper.summary.strip(),
                    'authors': json.dumps([author.name for author in paper.authors]),
                    'published': paper.published.isoformat(),
                    'categories': json.dumps(paper.categories),
                    'url': paper.entry_id
                }
                
                batch.append(paper_data)
                
                if len(batch) >= batch_size:
                    stats = self._process_batch(batch)
                    papers_added += stats['added']
                    papers_updated += stats['updated']
                    papers_processed += len(batch)
                    batch = []
                    
                    # Memory cleanup and progress update
                    gc.collect()
                    logger.info(f"Processed {papers_processed}/{max_papers} papers...")
                    
                    # Rate limiting to be nice to arXiv
                    time.sleep(0.5)
            
            # Process remaining papers
            if batch:
                stats = self._process_batch(batch)
                papers_added += stats['added']
                papers_updated += stats['updated']
                papers_processed += len(batch)
            
            result = {
                'total_processed': papers_processed,
                'papers_added': papers_added,
                'papers_updated': papers_updated,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Ingestion complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during literature ingestion: {e}")
            return {'error': str(e), 'total_processed': 0}
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process a batch of papers"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        papers_added = 0
        papers_updated = 0
        
        for paper in batch:
            try:
                # Check if paper already exists
                cursor.execute('SELECT id FROM papers WHERE id = ?', (paper['id'],))
                exists = cursor.fetchone()
                
                # Generate embedding if embedder is available
                embedding_blob = None
                if self.embedder:
                    text = f"{paper['title']} {paper['abstract']}"
                    embedding = self.embedder.encode(text)
                    embedding_blob = embedding.tobytes()
                
                if exists:
                    # Update existing paper
                    cursor.execute('''
                        UPDATE papers SET 
                        title=?, abstract=?, authors=?, published=?, 
                        categories=?, url=?, embedding=?, processed_date=?
                        WHERE id=?
                    ''', (
                        paper['title'], paper['abstract'], paper['authors'],
                        paper['published'], paper['categories'], paper['url'],
                        embedding_blob, datetime.now().isoformat(), paper['id']
                    ))
                    papers_updated += 1
                else:
                    # Insert new paper
                    cursor.execute('''
                        INSERT INTO papers 
                        (id, title, abstract, authors, published, categories, 
                         url, embedding, processed_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        paper['id'], paper['title'], paper['abstract'],
                        paper['authors'], paper['published'], paper['categories'],
                        paper['url'], embedding_blob, datetime.now().isoformat()
                    ))
                    papers_added += 1
                
                # Extract and store concepts
                self._extract_and_store_concepts(paper, cursor)
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('id', 'unknown')}: {e}")
        
        conn.commit()
        conn.close()
        
        return {'added': papers_added, 'updated': papers_updated}
    
    def _extract_and_store_concepts(self, paper: Dict[str, Any], cursor):
        """Extract concepts from paper and store in database"""
        text = f"{paper['title']} {paper['abstract']}"
        concepts = self.extract_concepts(text)
        
        for concept in concepts:
            # Insert or update concept frequency
            cursor.execute('''
                INSERT OR IGNORE INTO concepts (concept, category, first_seen, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (concept, self._classify_concept(concept), 
                  datetime.now().isoformat(), datetime.now().isoformat()))
            
            cursor.execute('''
                UPDATE concepts SET 
                frequency = frequency + 1,
                last_updated = ?
                WHERE concept = ?
            ''', (datetime.now().isoformat(), concept))
            
            # Link paper to concept
            cursor.execute('''
                INSERT OR REPLACE INTO paper_concepts (paper_id, concept, relevance_score)
                VALUES (?, ?, ?)
            ''', (paper['id'], concept, self._calculate_relevance(concept, text)))
    
    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text using rule-based NLP
        This is completely free and works well for ML papers
        """
        concepts = set()
        
        # ML/AI specific terms
        ml_patterns = [
            r'\b(?:neural\s+)?networks?\b',
            r'\b(?:deep\s+)?learning\b',
            r'\battention\s+mechanisms?\b',
            r'\btransformers?\b',
            r'\bconvolutional\s+neural\s+networks?\b',
            r'\bCNNs?\b',
            r'\bRNNs?\b',
            r'\bLSTMs?\b',
            r'\bGRUs?\b',
            r'\bVision\s+Transformers?\b',
            r'\bViTs?\b',
            r'\bmulti-head\s+attention\b',
            r'\bself-attention\b',
            r'\bcross-attention\b',
            r'\bcomputer\s+vision\b',
            r'\bnatural\s+language\s+processing\b',
            r'\bNLP\b',
            r'\bmachine\s+learning\b',
            r'\bartificial\s+intelligence\b',
            r'\bfeature\s+extraction\b',
            r'\brepresentation\s+learning\b'
        ]
        
        # Extract ML-specific terms
        for pattern in ml_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concepts.add(match.group().lower().strip())
        
        # Extract capitalized technical terms
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for term in technical_terms:
            if self._is_technical_concept(term):
                concepts.add(term.lower())
        
        # Extract acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        for acronym in acronyms:
            if len(acronym) <= 6:  # Reasonable acronym length
                concepts.add(acronym)
        
        return list(concepts)[:50]  # Limit to prevent explosion
    
    def _is_technical_concept(self, term: str) -> bool:
        """Check if a term is likely a technical concept"""
        technical_keywords = [
            'neural', 'network', 'learning', 'algorithm', 'model',
            'optimization', 'gradient', 'activation', 'convolution',
            'attention', 'transformer', 'embedding', 'feature',
            'classification', 'regression', 'clustering', 'detection'
        ]
        
        return any(keyword in term.lower() for keyword in technical_keywords)
    
    def _classify_concept(self, concept: str) -> str:
        """Classify concept into categories"""
        concept_lower = concept.lower()
        
        if any(word in concept_lower for word in ['vision', 'image', 'visual', 'cnn']):
            return 'computer_vision'
        elif any(word in concept_lower for word in ['language', 'text', 'nlp', 'bert']):
            return 'natural_language_processing'
        elif any(word in concept_lower for word in ['attention', 'transformer']):
            return 'attention_mechanisms'
        elif any(word in concept_lower for word in ['neural', 'network', 'deep']):
            return 'neural_networks'
        else:
            return 'general_ml'
    
    def _calculate_relevance(self, concept: str, text: str) -> float:
        """Calculate relevance score of concept to text"""
        # Simple frequency-based relevance
        concept_count = text.lower().count(concept.lower())
        text_length = len(text.split())
        
        # Normalize by text length
        relevance = min(1.0, concept_count / max(1, text_length / 100))
        return relevance
    
    def find_research_gaps(self, domain: str = "machine learning", 
                          min_papers: int = 3) -> List[Dict[str, Any]]:
        """
        Identify potential research gaps using frequency analysis
        
        Args:
            domain: Research domain to focus on
            min_papers: Minimum papers for a concept to be considered established
            
        Returns:
            List of potential research gaps with metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find concepts with low frequency (potential gaps)
        cursor.execute('''
            SELECT c.concept, c.frequency, c.category, COUNT(pc.paper_id) as paper_count
            FROM concepts c
            LEFT JOIN paper_concepts pc ON c.concept = pc.concept
            WHERE c.category LIKE ? OR c.concept LIKE ?
            GROUP BY c.concept
            HAVING paper_count < ?
            ORDER BY c.frequency ASC, paper_count ASC
            LIMIT 20
        ''', (f'%{domain}%', f'%{domain}%', min_papers))
        
        gaps = []
        for row in cursor.fetchall():
            concept, frequency, category, paper_count = row
            
            # Generate gap description
            gap_description = self._generate_gap_description(concept, category, paper_count)
            
            gaps.append({
                'concept': concept,
                'description': gap_description,
                'frequency': frequency,
                'paper_count': paper_count,
                'category': category,
                'confidence_score': self._calculate_gap_confidence(frequency, paper_count)
            })
        
        conn.close()
        
        # Store identified gaps
        self._store_research_gaps(gaps, domain)
        
        logger.info(f"Found {len(gaps)} potential research gaps in {domain}")
        return gaps
    
    def _generate_gap_description(self, concept: str, category: str, paper_count: int) -> str:
        """Generate a description for a research gap"""
        templates = {
            'computer_vision': [
                f"Limited exploration of {concept} in visual recognition tasks",
                f"Underinvestigated application of {concept} for image understanding",
                f"Sparse research on {concept} integration with vision models"
            ],
            'natural_language_processing': [
                f"Insufficient research on {concept} for language understanding",
                f"Limited investigation of {concept} in text processing",
                f"Underexplored {concept} applications in NLP tasks"
            ],
            'attention_mechanisms': [
                f"Novel {concept} attention patterns remain unexplored",
                f"Limited research on {concept} in attention-based models",
                f"Underinvestigated {concept} for improving attention efficiency"
            ],
            'neural_networks': [
                f"Limited exploration of {concept} in neural architectures",
                f"Insufficient research on {concept} optimization techniques",
                f"Underexplored {concept} for improving network performance"
            ],
            'general_ml': [
                f"Limited investigation of {concept} in machine learning",
                f"Underexplored applications of {concept} in ML algorithms",
                f"Insufficient research on {concept} optimization"
            ]
        }
        
        import random
        category_templates = templates.get(category, templates['general_ml'])
        return random.choice(category_templates)
    
    def _calculate_gap_confidence(self, frequency: int, paper_count: int) -> float:
        """Calculate confidence score for a research gap"""
        # Higher confidence for concepts with very low frequency/paper count
        freq_score = max(0, 1.0 - frequency / 10.0)
        paper_score = max(0, 1.0 - paper_count / 5.0)
        
        return (freq_score + paper_score) / 2.0
    
    def _store_research_gaps(self, gaps: List[Dict[str, Any]], domain: str):
        """Store identified research gaps in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for gap in gaps:
            cursor.execute('''
                INSERT OR REPLACE INTO research_gaps 
                (gap_description, domain, evidence, confidence_score, identified_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                gap['description'],
                domain,
                json.dumps({
                    'concept': gap['concept'],
                    'frequency': gap['frequency'],
                    'paper_count': gap['paper_count'],
                    'category': gap['category']
                }),
                gap['confidence_score'],
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def get_similar_papers(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find papers similar to query using embeddings"""
        if not self.embedder:
            logger.warning("No embedder available, using keyword matching")
            return self._keyword_based_search(query_text, top_k)
        
        # Generate query embedding
        query_embedding = self.embedder.encode(query_text)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all papers with embeddings
        cursor.execute('SELECT id, title, abstract, embedding FROM papers WHERE embedding IS NOT NULL')
        papers = cursor.fetchall()
        
        similarities = []
        for paper in papers:
            try:
                paper_embedding = np.frombuffer(paper[3], dtype=np.float32)
                similarity = np.dot(query_embedding, paper_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(paper_embedding)
                )
                similarities.append((similarity, paper[0], paper[1], paper[2]))
            except Exception as e:
                logger.error(f"Error calculating similarity for paper {paper[0]}: {e}")
        
        # Sort by similarity
        similarities.sort(reverse=True)
        conn.close()
        
        return [
            {
                'id': paper[1],
                'title': paper[2],
                'abstract': paper[3][:500] + "..." if len(paper[3]) > 500 else paper[3],
                'similarity': float(paper[0])
            }
            for paper in similarities[:top_k]
        ]
    
    def _keyword_based_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword-based search when embeddings are not available"""
        keywords = query_text.lower().split()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search in titles and abstracts
        search_query = '''
            SELECT id, title, abstract, 
                   (CASE 
                    WHEN lower(title) LIKE ? THEN 3
                    WHEN lower(abstract) LIKE ? THEN 2
                    ELSE 1 
                   END) as relevance_score
            FROM papers 
            WHERE lower(title) LIKE ? OR lower(abstract) LIKE ?
            ORDER BY relevance_score DESC
            LIMIT ?
        '''
        
        search_term = f"%{' '.join(keywords)}%"
        cursor.execute(search_query, (search_term, search_term, search_term, search_term, top_k))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'title': row[1],
                'abstract': row[2][:500] + "..." if len(row[2]) > 500 else row[2],
                'similarity': row[3] / 3.0  # Normalize score
            })
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count papers
        cursor.execute('SELECT COUNT(*) FROM papers')
        paper_count = cursor.fetchone()[0]
        
        # Count concepts
        cursor.execute('SELECT COUNT(*) FROM concepts')
        concept_count = cursor.fetchone()[0]
        
        # Count research gaps
        cursor.execute('SELECT COUNT(*) FROM research_gaps')
        gap_count = cursor.fetchone()[0]
        
        # Get category distribution
        cursor.execute('''
            SELECT category, COUNT(*) 
            FROM concepts 
            GROUP BY category 
            ORDER BY COUNT(*) DESC
        ''')
        category_distribution = dict(cursor.fetchall())
        
        # Get recent papers
        cursor.execute('''
            SELECT COUNT(*) 
            FROM papers 
            WHERE processed_date > datetime('now', '-7 days')
        ''')
        recent_papers = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_papers': paper_count,
            'total_concepts': concept_count,
            'research_gaps': gap_count,
            'recent_papers_week': recent_papers,
            'category_distribution': category_distribution,
            'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        }
    
    def clear_database(self):
        """Clear all data from database (use with caution)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM paper_concepts')
        cursor.execute('DELETE FROM research_gaps')
        cursor.execute('DELETE FROM concepts')
        cursor.execute('DELETE FROM papers')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Database cleared")
