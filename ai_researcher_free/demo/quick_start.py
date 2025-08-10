#!/usr/bin/env python3
"""
AI Researcher Free - Quick Start Demo
Complete research workflow in one script
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import modules
from core.free_knowledge_base import FreeKnowledgeBase
from core.template_generator import TemplateHypothesisGenerator
from core.free_llm_generator import FreeLLMGenerator
from core.experiment_runner import FreeExperimentRunner

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🤖 AI RESEARCHER FREE - QUICK START DEMO")
    print("💰 100% Free AI Research Assistant")
    print("=" * 60)
    print()

def quick_start_demo():
    """Run complete AI research workflow"""
    
    print_banner()
    
    # Configuration
    RESEARCH_QUERY = "attention mechanisms transformer neural networks"
    MAX_PAPERS = 15  # Small for quick demo
    
    print(f"🔍 Research domain: '{RESEARCH_QUERY}'")
    print(f"📄 Papers to analyze: {MAX_PAPERS}")
    print(f"⏱️ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Literature Analysis
        print("📚 STEP 1: Analyzing Research Literature")
        print("-" * 40)
        
        kb = FreeKnowledgeBase()
        print("✅ Knowledge base initialized")
        
        result = kb.ingest_literature(RESEARCH_QUERY, max_papers=MAX_PAPERS)
        print(f"✅ Processed {result.get('total_processed', 0)} papers")
        
        stats = kb.get_statistics()
        print(f"📊 Statistics: {stats['total_papers']} papers, {stats['total_concepts']} concepts")
        
        # Step 2: Research Gap Identification
        print("\n🔍 STEP 2: Identifying Research Gaps")
        print("-" * 40)
        
        gaps = kb.find_research_gaps("machine learning")
        print(f"✅ Found {len(gaps)} potential research gaps")
        
        if gaps:
            print("Top 3 research gaps:")
            for i, gap in enumerate(gaps[:3], 1):
                print(f"  {i}. {gap['concept']} (confidence: {gap['confidence_score']:.2f})")
        
        # Step 3: Hypothesis Generation
        print("\n💡 STEP 3: Generating Research Hypotheses")
        print("-" * 40)
        
        generator = TemplateHypothesisGenerator()
        hypotheses = generator.generate_hypotheses(gaps[:3], num_hypotheses=3)
        print(f"✅ Generated {len(hypotheses)} hypotheses")
        
        if hypotheses:
            print("Generated hypotheses:")
            for i, (hypothesis, confidence) in enumerate(hypotheses, 1):
                print(f"  {i}. {hypothesis[:80]}... (confidence: {confidence:.2f})")
        
        # Step 4: Quick Experiment
        print("\n🧪 STEP 4: Running Quick ML Experiment")
        print("-" * 40)
        
        if hypotheses:
            runner = FreeExperimentRunner()
            best_hypothesis = hypotheses[0][0]
            
            # Quick experiment config
            config = {
                'epochs': 3,  # Very quick for demo
                'batch_size': 32,
                'train_subset_size': 200,
                'test_subset_size': 50
            }
            
            print(f"🎯 Testing: {best_hypothesis[:60]}...")
            print("⚡ Running quick experiment (this may take 1-2 minutes)...")
            
            results = runner.run_vision_experiment(best_hypothesis, config)
            
            print(f"✅ Experiment completed!")
            print(f"📊 Results: {results['test_results']['accuracy']:.1f}% accuracy")
            print(f"⚡ Model: {results['model_info']['model_class']}")
            print(f"🔢 Parameters: {results['model_info']['total_parameters']:,}")
        
        # Step 5: Summary
        print("\n📋 STEP 5: Session Summary")
        print("-" * 40)
        
        summary = {
            "Papers analyzed": stats['total_papers'],
            "Concepts extracted": stats['total_concepts'],
            "Research gaps found": len(gaps),
            "Hypotheses generated": len(hypotheses),
            "Best accuracy": f"{results['test_results']['accuracy']:.1f}%" if 'results' in locals() else "N/A"
        }
        
        for key, value in summary.items():
            print(f"  📊 {key}: {value}")
        
        print(f"\n💰 Total cost: $0.00 (100% FREE!)")
        print(f"⏱️ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Next steps
        print("\n🚀 NEXT STEPS")
        print("-" * 40)
        print("• 🌐 Try the web interface: streamlit run demo/streamlit_app.py")
        print("• 📓 Explore the Jupyter notebook: demo/jupyter_demo.ipynb")
        print("• 🔬 Run more experiments with different hypotheses")
        print("• 📚 Analyze more papers by changing the research query")
        print("• ☁️ Use Google Colab for free GPU access")
        
        print("\n✅ Quick start demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("💡 This might be due to:")
        print("  • Network issues (arXiv access)")
        print("  • Missing dependencies")
        print("  • Insufficient memory")
        print("\n🔧 Try:")
        print("  • pip install -r requirements.txt")
        print("  • Reduce MAX_PAPERS to 5-10")
        print("  • Check your internet connection")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Researcher Free - Quick Start Demo")
    parser.add_argument("--query", default="attention mechanisms transformer", 
                       help="Research query for literature analysis")
    parser.add_argument("--papers", type=int, default=15,
                       help="Maximum papers to analyze")
    parser.add_argument("--quick", action="store_true",
                       help="Extra quick mode (fewer papers and epochs)")
    
    args = parser.parse_args()
    
    # Update global configuration
    global RESEARCH_QUERY, MAX_PAPERS
    RESEARCH_QUERY = args.query
    MAX_PAPERS = args.papers
    
    if args.quick:
        MAX_PAPERS = min(MAX_PAPERS, 8)
        print("⚡ Quick mode enabled - using minimal resources")
    
    quick_start_demo()

if __name__ == "__main__":
    main()
