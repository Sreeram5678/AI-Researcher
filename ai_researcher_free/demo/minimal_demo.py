#!/usr/bin/env python3
"""
AI Researcher by Sreeram - Minimal Demo
Lightweight version that works with basic dependencies
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import re
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🤖 AI RESEARCHER BY SREERAM - MINIMAL DEMO")
    print("💰 100% Free AI Research Assistant")
    print("=" * 60)
    print()

class MinimalKnowledgeBase:
    """Simplified knowledge base that doesn't require heavy dependencies"""
    
    def __init__(self, db_path="data/minimal_kb.db"):
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
                identified_date TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_demo_papers(self):
        """Add some demo papers for testing"""
        demo_papers = [
            {
                'id': 'demo1',
                'title': 'Attention Is All You Need',
                'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.',
                'concepts': 'attention, transformer, neural networks, sequence modeling'
            },
            {
                'id': 'demo2', 
                'title': 'Vision Transformer for Image Recognition',
                'abstract': 'While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. We show that reliance on CNNs is not necessary.',
                'concepts': 'vision transformer, computer vision, image recognition, attention'
            },
            {
                'id': 'demo3',
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                'abstract': 'We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations.',
                'concepts': 'BERT, bidirectional, transformers, language understanding'
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for paper in demo_papers:
            cursor.execute('''
                INSERT OR REPLACE INTO papers (id, title, abstract, concepts, processed_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (paper['id'], paper['title'], paper['abstract'], 
                  paper['concepts'], datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return len(demo_papers)
    
    def find_research_gaps(self):
        """Find potential research gaps using simple text analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all concepts
        cursor.execute('SELECT concepts FROM papers')
        all_concepts = []
        for row in cursor.fetchall():
            concepts = row[0].split(', ')
            all_concepts.extend(concepts)
        
        # Count concept frequency
        concept_counts = Counter(all_concepts)
        
        # Find underexplored concepts (low frequency)
        gaps = []
        for concept, count in concept_counts.items():
            if count <= 2 and len(concept) > 3:  # Underexplored concepts
                confidence = 1.0 - (count / len(concept_counts))
                gaps.append({
                    'concept': concept,
                    'description': f"Limited exploration of {concept} in current research",
                    'confidence': confidence,
                    'paper_count': count
                })
        
        # Store gaps
        cursor.execute('DELETE FROM research_gaps')  # Clear old gaps
        for gap in gaps:
            cursor.execute('''
                INSERT INTO research_gaps (concept, description, confidence, identified_date)
                VALUES (?, ?, ?, ?)
            ''', (gap['concept'], gap['description'], gap['confidence'], 
                  datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return gaps[:10]  # Return top 10 gaps
    
    def get_statistics(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM papers')
        paper_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM research_gaps')
        gap_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_papers': paper_count,
            'research_gaps': gap_count,
            'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        }

class MinimalHypothesisGenerator:
    """Simplified hypothesis generator using templates"""
    
    def __init__(self):
        self.templates = [
            "A novel {concept} approach could improve performance by addressing current limitations",
            "Integrating {concept} with attention mechanisms may enhance model capabilities", 
            "A multi-scale {concept} framework could better capture complex patterns",
            "Combining {concept} with transfer learning might achieve better results",
            "A lightweight {concept} architecture could maintain accuracy while reducing complexity"
        ]
    
    def generate_hypotheses(self, research_gaps, num_hypotheses=3):
        """Generate hypotheses from research gaps"""
        hypotheses = []
        
        for i, gap in enumerate(research_gaps[:num_hypotheses]):
            concept = gap['concept']
            template = self.templates[i % len(self.templates)]
            hypothesis = template.format(concept=concept)
            confidence = gap['confidence'] * 0.8  # Slightly lower than gap confidence
            
            hypotheses.append((hypothesis, confidence))
        
        return hypotheses

class MinimalExperimentRunner:
    """Simplified experiment runner for demo purposes"""
    
    def __init__(self):
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_mock_experiment(self, hypothesis):
        """Run a mock experiment (simulation)"""
        print(f"🧪 Running mock experiment for hypothesis:")
        print(f"   {hypothesis[:80]}...")
        
        # Simulate experiment results
        import random
        random.seed(hash(hypothesis) % 1000)  # Deterministic but varied results
        
        accuracy = random.uniform(65, 85)  # Reasonable accuracy range
        loss = random.uniform(0.3, 0.8)
        parameters = random.randint(50000, 200000)
        
        results = {
            'hypothesis': hypothesis,
            'test_results': {
                'accuracy': accuracy,
                'loss': loss,
                'samples_tested': 1000
            },
            'model_info': {
                'total_parameters': parameters,
                'model_size_mb': parameters * 4 / (1024 * 1024),
                'model_type': 'simulated_cnn'
            },
            'timestamp': datetime.now().isoformat(),
            'status': 'mock_experiment'
        }
        
        # Save results
        result_file = os.path.join(self.results_dir, f"mock_experiment_{int(datetime.now().timestamp())}.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def minimal_demo():
    """Run minimal demo without heavy dependencies"""
    
    print_banner()
    
    print("🔧 Initializing minimal AI research system...")
    print("💡 This demo uses lightweight components to avoid dependency issues")
    print()
    
    try:
        # Step 1: Setup knowledge base
        print("📚 STEP 1: Setting up Knowledge Base")
        print("-" * 40)
        
        kb = MinimalKnowledgeBase()
        papers_added = kb.add_demo_papers()
        print(f"✅ Added {papers_added} demo papers to knowledge base")
        
        stats = kb.get_statistics()
        print(f"📊 Database stats: {stats['total_papers']} papers, {stats['database_size_mb']:.2f} MB")
        
        # Step 2: Find research gaps
        print("\n🔍 STEP 2: Identifying Research Gaps")
        print("-" * 40)
        
        gaps = kb.find_research_gaps()
        print(f"✅ Found {len(gaps)} potential research gaps")
        
        if gaps:
            print("Top research gaps:")
            for i, gap in enumerate(gaps[:5], 1):
                print(f"  {i}. {gap['concept']} (confidence: {gap['confidence']:.2f})")
        
        # Step 3: Generate hypotheses
        print("\n💡 STEP 3: Generating Research Hypotheses")
        print("-" * 40)
        
        generator = MinimalHypothesisGenerator()
        hypotheses = generator.generate_hypotheses(gaps[:3])
        print(f"✅ Generated {len(hypotheses)} hypotheses")
        
        if hypotheses:
            print("Generated hypotheses:")
            for i, (hypothesis, confidence) in enumerate(hypotheses, 1):
                print(f"  {i}. {hypothesis} (confidence: {confidence:.2f})")
        
        # Step 4: Mock experiment
        print("\n🧪 STEP 4: Running Mock Experiment")
        print("-" * 40)
        
        if hypotheses:
            runner = MinimalExperimentRunner()
            best_hypothesis = hypotheses[0][0]
            
            results = runner.run_mock_experiment(best_hypothesis)
            
            print(f"✅ Mock experiment completed!")
            print(f"📊 Results:")
            print(f"   • Accuracy: {results['test_results']['accuracy']:.2f}%")
            print(f"   • Loss: {results['test_results']['loss']:.4f}")
            print(f"   • Parameters: {results['model_info']['total_parameters']:,}")
            print(f"   • Model Size: {results['model_info']['model_size_mb']:.2f} MB")
        
        # Step 5: Summary
        print("\n📋 STEP 5: Demo Summary")
        print("-" * 40)
        
        summary = {
            "Mode": "Minimal Demo (no heavy dependencies)",
            "Papers processed": stats['total_papers'],
            "Research gaps found": len(gaps),
            "Hypotheses generated": len(hypotheses),
            "Mock experiment": "✅ Completed"
        }
        
        for key, value in summary.items():
            print(f"  📊 {key}: {value}")
        
        print(f"\n💰 Total cost: $0.00 (100% FREE!)")
        print(f"⏱️ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n🚀 NEXT STEPS")
        print("-" * 40)
        print("• 🔧 Fix dependency issues to run full version")
        print("• 📦 Install missing packages: brew install xz")
        print("• 🐍 Try with different Python version")
        print("• ☁️ Use Google Colab for full functionality")
        print("• 📝 Check USAGE.md for detailed instructions")
        
        print("\n✅ Minimal demo completed successfully!")
        print("💡 This shows the core AI research workflow without heavy ML dependencies")
        
    except Exception as e:
        print(f"\n❌ Error during minimal demo: {e}")
        print("💡 This is a simplified version - some features may be limited")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Researcher by Sreeram - Minimal Demo")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("🔍 Running in verbose mode")
    
    minimal_demo()

if __name__ == "__main__":
    main()
