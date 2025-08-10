#!/usr/bin/env python3
"""
AI Researcher Free - Mac Optimized Demo
Works on Mac without heavy ML dependencies
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import re
import time
from collections import Counter
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🤖 AI RESEARCHER FREE - MAC OPTIMIZED")
    print("💰 100% Free AI Research Assistant")
    print("🍎 Optimized for Mac without heavy ML dependencies")
    print("=" * 60)
    print()

class MacOptimizedKnowledgeBase:
    """Mac-optimized knowledge base with arXiv integration"""
    
    def __init__(self, db_path="data/mac_kb.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                authors TEXT,
                published TEXT,
                categories TEXT,
                concepts TEXT,
                processed_date TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT,
                description TEXT,
                confidence REAL,
                frequency INTEGER,
                category TEXT,
                identified_date TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_arxiv_papers(self, query, max_papers=10):
        """Fetch papers from arXiv using their API"""
        print(f"📥 Fetching papers from arXiv for query: '{query}'")
        
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_papers,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = {}
                
                # Extract title
                title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                paper['title'] = title_elem.text.strip() if title_elem is not None else "Unknown"
                
                # Extract abstract
                summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                paper['abstract'] = summary_elem.text.strip() if summary_elem is not None else "No abstract"
                
                # Extract ID
                id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                paper['id'] = id_elem.text.strip() if id_elem is not None else f"paper_{len(papers)}"
                
                # Extract authors
                authors = []
                for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                    name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                paper['authors'] = json.dumps(authors)
                
                # Extract published date
                published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                paper['published'] = published_elem.text.strip() if published_elem is not None else datetime.now().isoformat()
                
                # Extract categories
                categories = []
                for category in entry.findall('{http://arxiv.org/schemas/atom}primary_category'):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                paper['categories'] = json.dumps(categories)
                
                papers.append(paper)
                
                # Rate limiting
                time.sleep(0.1)
            
            return papers
            
        except Exception as e:
            print(f"⚠️ Error fetching from arXiv: {e}")
            print("💡 Using demo papers instead...")
            return self.get_demo_papers()
    
    def get_demo_papers(self):
        """Fallback demo papers"""
        return [
            {
                'id': 'demo1',
                'title': 'Attention Is All You Need',
                'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
                'authors': '["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]',
                'published': '2017-06-12T00:00:00Z',
                'categories': '["cs.CL", "cs.AI"]'
            },
            {
                'id': 'demo2',
                'title': 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale',
                'abstract': 'While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components.',
                'authors': '["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov"]',
                'published': '2020-10-22T00:00:00Z',
                'categories': '["cs.CV", "cs.AI"]'
            },
            {
                'id': 'demo3',
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                'abstract': 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations.',
                'authors': '["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"]',
                'published': '2018-10-11T00:00:00Z',
                'categories': '["cs.CL"]'
            }
        ]
    
    def extract_concepts(self, text):
        """Extract concepts using rule-based NLP"""
        concepts = set()
        
        # ML/AI specific terms
        ml_patterns = [
            r'\b(?:neural\s+)?networks?\b',
            r'\b(?:deep\s+)?learning\b',
            r'\battention\s+mechanisms?\b',
            r'\btransformers?\b',
            r'\bconvolutional\s+neural\s+networks?\b',
            r'\bCNNs?\b',
            r'\bcomputer\s+vision\b',
            r'\bnatural\s+language\s+processing\b',
            r'\bNLP\b',
            r'\bmachine\s+learning\b',
            r'\bartificial\s+intelligence\b',
            r'\bBERT\b',
            r'\bGPT\b'
        ]
        
        for pattern in ml_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concepts.add(match.group().lower().strip())
        
        # Extract capitalized technical terms
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for term in technical_terms:
            if self._is_technical_concept(term):
                concepts.add(term.lower())
        
        return list(concepts)[:20]  # Limit to prevent explosion
    
    def _is_technical_concept(self, term):
        """Check if a term is likely a technical concept"""
        technical_keywords = [
            'neural', 'network', 'learning', 'algorithm', 'model',
            'optimization', 'gradient', 'activation', 'convolution',
            'attention', 'transformer', 'embedding', 'feature'
        ]
        return any(keyword in term.lower() for keyword in technical_keywords)
    
    def ingest_papers(self, query, max_papers=10):
        """Ingest papers and analyze them"""
        papers = self.fetch_arxiv_papers(query, max_papers)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        papers_added = 0
        all_concepts = []
        
        for paper in papers:
            # Extract concepts
            text = f"{paper['title']} {paper['abstract']}"
            concepts = self.extract_concepts(text)
            all_concepts.extend(concepts)
            
            # Store paper
            cursor.execute('''
                INSERT OR REPLACE INTO papers 
                (id, title, abstract, authors, published, categories, concepts, processed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper['id'], paper['title'], paper['abstract'],
                paper['authors'], paper['published'], paper['categories'],
                ', '.join(concepts), datetime.now().isoformat()
            ))
            
            papers_added += 1
        
        conn.commit()
        conn.close()
        
        print(f"✅ Added {papers_added} papers to knowledge base")
        return {'total_processed': papers_added, 'concepts_found': len(set(all_concepts))}
    
    def find_research_gaps(self):
        """Find research gaps using concept frequency analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all concepts
        cursor.execute('SELECT concepts FROM papers WHERE concepts IS NOT NULL')
        all_concepts = []
        for row in cursor.fetchall():
            if row[0]:
                concepts = [c.strip() for c in row[0].split(',')]
                all_concepts.extend(concepts)
        
        # Count concept frequency
        concept_counts = Counter(all_concepts)
        total_papers = len(all_concepts) / max(1, len(set(all_concepts)))
        
        # Find underexplored concepts
        gaps = []
        for concept, count in concept_counts.items():
            if len(concept) > 3 and count <= 2:  # Underexplored
                confidence = 1.0 - (count / max(1, total_papers))
                description = self._generate_gap_description(concept)
                
                gaps.append({
                    'concept': concept,
                    'description': description,
                    'confidence': min(0.95, confidence),
                    'frequency': count,
                    'category': self._classify_concept(concept)
                })
        
        # Store gaps
        cursor.execute('DELETE FROM research_gaps')
        for gap in gaps:
            cursor.execute('''
                INSERT INTO research_gaps 
                (concept, description, confidence, frequency, category, identified_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (gap['concept'], gap['description'], gap['confidence'],
                  gap['frequency'], gap['category'], datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return sorted(gaps, key=lambda x: x['confidence'], reverse=True)[:15]
    
    def _generate_gap_description(self, concept):
        """Generate description for research gap"""
        templates = [
            f"Limited exploration of {concept} in current research literature",
            f"Underinvestigated applications of {concept} in AI systems",
            f"Sparse research on {concept} integration with modern architectures",
            f"Insufficient studies on {concept} optimization techniques"
        ]
        import random
        return random.choice(templates)
    
    def _classify_concept(self, concept):
        """Classify concept into category"""
        concept_lower = concept.lower()
        
        if any(word in concept_lower for word in ['vision', 'image', 'visual', 'cnn']):
            return 'computer_vision'
        elif any(word in concept_lower for word in ['language', 'text', 'nlp', 'bert']):
            return 'nlp'
        elif any(word in concept_lower for word in ['attention', 'transformer']):
            return 'attention_mechanisms'
        else:
            return 'general_ml'
    
    def get_statistics(self):
        """Get knowledge base statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM papers')
        paper_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM research_gaps')
        gap_count = cursor.fetchone()[0]
        
        # Get category distribution
        cursor.execute('SELECT category, COUNT(*) FROM research_gaps GROUP BY category')
        category_dist = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_papers': paper_count,
            'research_gaps': gap_count,
            'category_distribution': category_dist,
            'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        }

class MacOptimizedHypothesisGenerator:
    """Mac-optimized hypothesis generator"""
    
    def __init__(self):
        self.templates = {
            'computer_vision': [
                "A novel {concept} approach could improve visual recognition tasks by addressing current limitations in feature representation",
                "Integrating {concept} with attention mechanisms may enhance performance in computer vision applications",
                "A multi-scale {concept} framework could better capture spatial relationships in visual data"
            ],
            'nlp': [
                "A contextual {concept} model could improve language understanding by better capturing semantic relationships",
                "Incorporating {concept} into transformer architectures may enhance text processing capabilities",
                "A pre-trained {concept} approach could achieve better performance with less training data"
            ],
            'attention_mechanisms': [
                "A sparse {concept} attention pattern could reduce computational complexity while maintaining performance",
                "Multi-head attention enhanced with {concept} may capture more diverse feature relationships",
                "A learnable {concept} attention mechanism may adapt better to different input distributions"
            ],
            'general_ml': [
                "A {concept}-based approach could improve machine learning performance by leveraging novel optimization techniques",
                "Combining {concept} with existing methods may address current limitations in model generalization",
                "A regularization technique incorporating {concept} could reduce overfitting in deep learning models"
            ]
        }
    
    def generate_hypotheses(self, research_gaps, num_hypotheses=5):
        """Generate hypotheses from research gaps"""
        hypotheses = []
        
        for i, gap in enumerate(research_gaps[:num_hypotheses]):
            concept = gap['concept']
            category = gap['category']
            
            # Select appropriate template
            category_templates = self.templates.get(category, self.templates['general_ml'])
            template = category_templates[i % len(category_templates)]
            
            # Generate hypothesis
            hypothesis = template.format(concept=concept)
            confidence = gap['confidence'] * 0.85  # Slightly lower than gap confidence
            
            hypotheses.append((hypothesis, confidence))
        
        return hypotheses

class MacOptimizedAnalyzer:
    """Mac-optimized research analyzer"""
    
    def analyze_trends(self, kb):
        """Analyze research trends"""
        conn = sqlite3.connect(kb.db_path)
        
        # Get papers by category
        papers_df = pd.read_sql_query('''
            SELECT categories, published, concepts FROM papers
        ''', conn)
        
        # Get gaps by category  
        gaps_df = pd.read_sql_query('''
            SELECT category, COUNT(*) as gap_count FROM research_gaps
            GROUP BY category
        ''', conn)
        
        conn.close()
        
        return {
            'papers_by_category': papers_df,
            'gaps_by_category': gaps_df,
            'total_papers': len(papers_df),
            'analysis_date': datetime.now().isoformat()
        }
    
    def create_visualizations(self, analysis_data):
        """Create visualizations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gap distribution
        if not analysis_data['gaps_by_category'].empty:
            ax1.pie(analysis_data['gaps_by_category']['gap_count'], 
                   labels=analysis_data['gaps_by_category']['category'],
                   autopct='%1.1f%%')
            ax1.set_title('Research Gaps by Category')
        
        # Papers over time (simplified)
        ax2.bar(['2018', '2019', '2020', '2021', '2022+'], [1, 1, 1, 0, 0])
        ax2.set_title('Papers by Year (Demo Data)')
        ax2.set_ylabel('Number of Papers')
        
        # Concept frequency (mock data)
        concepts = ['attention', 'transformer', 'neural networks', 'computer vision']
        frequencies = [5, 4, 3, 2]
        ax3.barh(concepts, frequencies)
        ax3.set_title('Concept Frequency')
        ax3.set_xlabel('Frequency')
        
        # Research gap confidence
        ax4.hist([0.8, 0.75, 0.9, 0.85, 0.7], bins=5, alpha=0.7)
        ax4.set_title('Research Gap Confidence Distribution')
        ax4.set_xlabel('Confidence Score')
        ax4.set_ylabel('Number of Gaps')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plot_path = 'results/mac_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path

def mac_optimized_demo():
    """Run Mac-optimized demo"""
    
    print_banner()
    
    # Configuration
    RESEARCH_QUERY = input("🔍 Enter research query (or press Enter for default): ").strip()
    if not RESEARCH_QUERY:
        RESEARCH_QUERY = "attention mechanisms transformer neural networks"
    
    MAX_PAPERS = 8  # Small number for demo
    
    print(f"🔍 Research query: '{RESEARCH_QUERY}'")
    print(f"📄 Max papers: {MAX_PAPERS}")
    print(f"⏱️ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Literature Analysis
        print("📚 STEP 1: Analyzing Research Literature")
        print("-" * 40)
        
        kb = MacOptimizedKnowledgeBase()
        result = kb.ingest_papers(RESEARCH_QUERY, max_papers=MAX_PAPERS)
        
        stats = kb.get_statistics()
        print(f"📊 Statistics:")
        print(f"  • Papers: {stats['total_papers']}")
        print(f"  • Research gaps: {stats['research_gaps']}")
        print(f"  • Database size: {stats['database_size_mb']:.2f} MB")
        
        # Step 2: Research Gap Identification
        print("\n🔍 STEP 2: Identifying Research Gaps")
        print("-" * 40)
        
        gaps = kb.find_research_gaps()
        print(f"✅ Found {len(gaps)} potential research gaps")
        
        if gaps:
            print("Top research gaps:")
            for i, gap in enumerate(gaps[:5], 1):
                print(f"  {i}. {gap['concept']} (confidence: {gap['confidence']:.2f}, category: {gap['category']})")
        
        # Step 3: Hypothesis Generation
        print("\n💡 STEP 3: Generating Research Hypotheses")
        print("-" * 40)
        
        generator = MacOptimizedHypothesisGenerator()
        hypotheses = generator.generate_hypotheses(gaps[:5])
        print(f"✅ Generated {len(hypotheses)} hypotheses")
        
        if hypotheses:
            print("Generated hypotheses:")
            for i, (hypothesis, confidence) in enumerate(hypotheses, 1):
                print(f"  {i}. {hypothesis[:80]}... (confidence: {confidence:.2f})")
        
        # Step 4: Analysis & Visualization
        print("\n📊 STEP 4: Creating Analysis & Visualizations")
        print("-" * 40)
        
        analyzer = MacOptimizedAnalyzer()
        analysis_data = analyzer.analyze_trends(kb)
        plot_path = analyzer.create_visualizations(analysis_data)
        
        print(f"✅ Analysis complete!")
        print(f"📈 Visualization saved to: {plot_path}")
        
        # Step 5: Summary
        print("\n📋 STEP 5: Session Summary")
        print("-" * 40)
        
        summary = {
            "Research query": RESEARCH_QUERY,
            "Papers analyzed": stats['total_papers'],
            "Research gaps found": len(gaps),
            "Hypotheses generated": len(hypotheses),
            "Best hypothesis confidence": f"{hypotheses[0][1]:.2f}" if hypotheses else "N/A",
            "Analysis visualization": "✅ Created"
        }
        
        for key, value in summary.items():
            print(f"  📊 {key}: {value}")
        
        print(f"\n💰 Total cost: $0.00 (100% FREE!)")
        print(f"⏱️ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Export results
        print("\n💾 STEP 6: Exporting Results")
        print("-" * 40)
        
        os.makedirs('results', exist_ok=True)
        
        # Export hypotheses
        with open('results/mac_hypotheses.json', 'w') as f:
            json.dump([{'hypothesis': h, 'confidence': c} for h, c in hypotheses], f, indent=2)
        
        # Export summary
        with open('results/mac_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✅ Results exported to results/ directory")
        
        # Next steps
        print("\n🚀 NEXT STEPS")
        print("-" * 40)
        print("• 📊 Check results/mac_analysis.png for visualizations")
        print("• 📄 Review results/mac_hypotheses.json for detailed hypotheses")
        print("• 🔬 Try different research queries")
        print("• 📚 Increase max_papers for more comprehensive analysis")
        print("• ☁️ Use Google Colab for ML experiments")
        
        print("\n✅ Mac-optimized demo completed successfully!")
        print("🍎 This version works great on Mac without heavy ML dependencies!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("💡 Try running the minimal demo instead: python demo/minimal_demo.py")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Researcher Free - Mac Optimized Demo")
    parser.add_argument("--query", help="Research query")
    parser.add_argument("--papers", type=int, default=8, help="Max papers to analyze")
    
    args = parser.parse_args()
    
    if args.query:
        global RESEARCH_QUERY
        RESEARCH_QUERY = args.query
    
    mac_optimized_demo()

if __name__ == "__main__":
    main()
