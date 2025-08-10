"""
AI Researcher by Sreeram - Streamlit Web Interface
100% Free AI Research Assistant
"""

import streamlit as st
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import core modules
try:
    from core.free_knowledge_base import FreeKnowledgeBase
    from core.template_generator import TemplateHypothesisGenerator
    from core.free_llm_generator import FreeLLMGenerator
    from core.experiment_runner import FreeExperimentRunner
    from core.paper_analyzer import PaperAnalyzer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Researcher by Sreeram",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
            <h1>🤖 AI Researcher by Sreeram</h1>
    <p>Autonomous AI Research Assistant - 100% Free Edition</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🎛️ Configuration")

# Initialize session state
if 'kb' not in st.session_state:
    st.session_state.kb = None
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Sidebar settings
st.sidebar.subheader("Free LLM Settings")
use_llm = st.sidebar.checkbox("Enable Free LLM Generation", help="Uses Ollama if available, otherwise templates")

if use_llm:
    hf_token = st.sidebar.text_input(
        "Hugging Face Token (Optional)", 
        type="password",
        help="Get free token from https://huggingface.co/settings/tokens"
    )
else:
    hf_token = None

st.sidebar.subheader("Experiment Settings")
use_gpu = st.sidebar.checkbox("Use GPU if available", value=True)
max_papers = st.sidebar.slider("Max papers to analyze", 10, 100, 30)
quick_mode = st.sidebar.checkbox("Quick mode (faster, smaller datasets)", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Cost: $0.00")
st.sidebar.success("✅ Everything is 100% FREE!")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 Literature Analysis", 
    "💡 Hypothesis Generation", 
    "🧪 Experiment Runner",
    "📊 Analysis Dashboard",
    "ℹ️ System Status"
])

# Tab 1: Literature Analysis
with tab1:
    st.header("📚 Research Literature Analysis")
    st.write("Analyze research papers from arXiv using free tools")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        research_query = st.text_input(
            "Research Query",
            value="attention mechanisms computer vision",
            help="Enter keywords to search arXiv"
        )
    
    with col2:
        st.metric("Max Papers", max_papers, "Free tier")
    
    if st.button("🔍 Start Literature Analysis", type="primary"):
        if not research_query.strip():
            st.error("Please enter a research query")
        else:
            with st.spinner("Initializing knowledge base..."):
                try:
                    kb = FreeKnowledgeBase()
                    st.session_state.kb = kb
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Literature ingestion
                    status_text.text("📖 Downloading papers from arXiv...")
                    progress_bar.progress(20)
                    
                    result = kb.ingest_literature(research_query, max_papers=max_papers)
                    progress_bar.progress(60)
                    
                    status_text.text("🔍 Analyzing papers and extracting concepts...")
                    
                    # Get statistics
                    stats = kb.get_statistics()
                    progress_bar.progress(80)
                    
                    # Find research gaps
                    gaps = kb.find_research_gaps("machine learning")
                    progress_bar.progress(100)
                    
                    status_text.text("✅ Analysis complete!")
                    
                    # Display results
                    st.success(f"Successfully processed {result.get('total_processed', 0)} papers!")
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📄 Papers", stats['total_papers'])
                    with col2:
                        st.metric("🔬 Concepts", stats['total_concepts'])
                    with col3:
                        st.metric("🕳️ Research Gaps", len(gaps))
                    with col4:
                        st.metric("💾 DB Size", f"{stats['database_size_mb']:.1f} MB")
                    
                    # Research gaps
                    if gaps:
                        st.subheader("🔍 Identified Research Gaps")
                        for i, gap in enumerate(gaps[:10], 1):
                            with st.expander(f"Gap {i}: {gap['concept']}", expanded=i<=3):
                                st.write(f"**Description:** {gap['description']}")
                                st.write(f"**Category:** {gap['category']}")
                                st.write(f"**Confidence:** {gap['confidence_score']:.2f}")
                                st.write(f"**Papers mentioning this:** {gap['paper_count']}")
                        
                        # Store gaps for next steps
                        st.session_state.research_gaps = gaps
                    
                    # Category distribution
                    if stats['category_distribution']:
                        st.subheader("📊 Research Category Distribution")
                        
                        categories = list(stats['category_distribution'].keys())
                        counts = list(stats['category_distribution'].values())
                        
                        fig = px.pie(
                            values=counts, 
                            names=categories,
                            title="Distribution of Research Categories"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.info("This might be due to network issues or arXiv rate limiting. Try again with fewer papers.")

# Tab 2: Hypothesis Generation
with tab2:
    st.header("💡 Research Hypothesis Generation")
    
    if 'research_gaps' not in st.session_state:
        st.info("👈 Please run Literature Analysis first to identify research gaps")
    else:
        st.write("Generate novel research hypotheses based on identified gaps")
        
        # Hypothesis generation method
        col1, col2 = st.columns(2)
        with col1:
            generation_method = st.selectbox(
                "Generation Method",
                ["Template-based (Free)", "Free LLM (Ollama/HF)", "Hybrid (Best)"],
                help="Template-based is always free, LLM requires setup"
            )
        
        with col2:
            num_hypotheses = st.slider("Number of hypotheses", 1, 10, 5)
        
        if st.button("🧠 Generate Hypotheses", type="primary"):
            with st.spinner("Generating research hypotheses..."):
                try:
                    if generation_method.startswith("Template") or not use_llm:
                        # Template-based generation
                        generator = TemplateHypothesisGenerator()
                        hypotheses = generator.generate_hypotheses(
                            st.session_state.research_gaps, 
                            num_hypotheses
                        )
                        method_used = "Template-based"
                        
                    else:
                        # LLM-based generation
                        generator = FreeLLMGenerator(hf_token)
                        hypotheses = generator.generate_hypotheses(
                            st.session_state.research_gaps,
                            num_hypotheses=num_hypotheses
                        )
                        method_used = "Free LLM"
                    
                    if hypotheses:
                        st.success(f"Generated {len(hypotheses)} hypotheses using {method_used}")
                        
                        # Display hypotheses
                        for i, (hypothesis, confidence) in enumerate(hypotheses, 1):
                            with st.expander(f"Hypothesis {i} (Confidence: {confidence:.2f})", expanded=i<=2):
                                st.write(hypothesis)
                                
                                # Additional details for template-based
                                if hasattr(generator, 'generate_methodology_description'):
                                    methodology = generator.generate_methodology_description(hypothesis)
                                    st.write(f"**Methodology:** {methodology}")
                                
                                # Research questions
                                if hasattr(generator, 'generate_research_questions'):
                                    questions = generator.generate_research_questions(hypothesis)
                                    st.write("**Research Questions:**")
                                    for j, question in enumerate(questions[:3], 1):
                                        st.write(f"{j}. {question}")
                        
                        # Store best hypothesis for experiments
                        if hypotheses:
                            st.session_state.best_hypothesis = hypotheses[0][0]
                            st.session_state.all_hypotheses = hypotheses
                        
                        # Export option
                        if st.button("📥 Export Hypotheses"):
                            if hasattr(generator, 'export_hypotheses'):
                                filename = generator.export_hypotheses(hypotheses)
                                st.success(f"Hypotheses exported to {filename}")
                    
                    else:
                        st.warning("No hypotheses were generated. Try adjusting the settings.")
                        
                except Exception as e:
                    st.error(f"Error generating hypotheses: {e}")

# Tab 3: Experiment Runner
with tab3:
    st.header("🧪 ML Experiment Runner")
    
    if 'best_hypothesis' not in st.session_state:
        st.info("👈 Please generate hypotheses first")
    else:
        st.write("Run experiments to test your research hypothesis")
        
        # Display selected hypothesis
        st.subheader("Selected Hypothesis")
        st.write(st.session_state.best_hypothesis)
        
        # Experiment configuration
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_type = st.selectbox(
                "Experiment Type",
                ["Computer Vision", "Natural Language Processing"],
                help="Choose the type of ML experiment"
            )
            
            if experiment_type == "Computer Vision":
                dataset = st.selectbox("Dataset", ["CIFAR-10", "MNIST"])
                model_type = st.selectbox("Model Type", ["CNN", "Attention-CNN", "Simple-ViT", "ResNet"])
            else:
                dataset = "Synthetic Text"
                model_type = st.selectbox("Model Type", ["Simple-NLP", "LSTM", "Transformer"])
        
        with col2:
            epochs = st.slider("Training Epochs", 1, 50, 10 if quick_mode else 20)
            batch_size = st.slider("Batch Size", 16, 128, 32)
            learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.0001], index=0)
        
        # Resource estimation
        st.subheader("💻 Resource Estimation")
        if quick_mode:
            st.info("🚀 Quick mode: ~2-5 minutes, CPU-friendly")
        else:
            st.warning("🐌 Full mode: ~10-30 minutes, GPU recommended")
        
        if st.button("🚀 Run Experiment", type="primary"):
            with st.spinner("Running ML experiment..."):
                try:
                    # Initialize experiment runner
                    runner = FreeExperimentRunner(use_gpu=use_gpu)
                    
                    # Configure experiment
                    config = {
                        'dataset': dataset.lower().replace('-', ''),
                        'model_type': model_type.lower(),
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'train_subset_size': 500 if quick_mode else 2000,
                        'test_subset_size': 100 if quick_mode else 500
                    }
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔧 Setting up experiment...")
                    progress_bar.progress(10)
                    
                    # Run experiment
                    if experiment_type == "Computer Vision":
                        results = runner.run_vision_experiment(
                            st.session_state.best_hypothesis, 
                            config
                        )
                    else:
                        results = runner.run_nlp_experiment(
                            st.session_state.best_hypothesis,
                            config
                        )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Experiment completed!")
                    
                    # Display results
                    st.success("🎉 Experiment completed successfully!")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Test Accuracy", f"{results['test_results']['accuracy']:.2f}%")
                    with col2:
                        st.metric("📉 Test Loss", f"{results['test_results']['loss']:.4f}")
                    with col3:
                        st.metric("⚡ Model Size", f"{results['model_info']['model_size_mb']:.2f} MB")
                    with col4:
                        st.metric("🔢 Parameters", f"{results['model_info']['total_parameters']:,}")
                    
                    # Training history
                    if results['training_history']:
                        st.subheader("📈 Training Progress")
                        
                        history_df = pd.DataFrame(results['training_history'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_acc = px.line(history_df, x='epoch', y='accuracy', 
                                            title='Training Accuracy')
                            st.plotly_chart(fig_acc, use_container_width=True)
                        
                        with col2:
                            fig_loss = px.line(history_df, x='epoch', y='loss',
                                             title='Training Loss')
                            st.plotly_chart(fig_loss, use_container_width=True)
                    
                    # Experiment report
                    if st.button("📄 Generate Report"):
                        report = runner.create_experiment_report(results)
                        st.text_area("Experiment Report", report, height=400)
                    
                    # Store results
                    st.session_state.experiment_results.append(results)
                    
                except Exception as e:
                    st.error(f"Experiment failed: {e}")
                    st.info("Try reducing the dataset size or switching to CPU mode")

# Tab 4: Analysis Dashboard
with tab4:
    st.header("📊 Research Analysis Dashboard")
    
    if st.session_state.kb is None:
        st.info("👈 Please run Literature Analysis first")
    else:
        st.write("Comprehensive analysis of research trends and patterns")
        
        if st.button("🔬 Run Paper Analysis", type="primary"):
            with st.spinner("Analyzing research papers..."):
                try:
                    # Get papers from knowledge base
                    conn = st.session_state.kb._setup_database()
                    # This is a simplified version - you'd query the actual papers
                    
                    # Initialize analyzer
                    analyzer = PaperAnalyzer()
                    
                    # Mock analysis for demo (replace with real data)
                    mock_papers = [
                        {
                            'id': 'paper1',
                            'title': 'Attention Is All You Need',
                            'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...',
                            'published': '2017-06-12T00:00:00Z'
                        },
                        {
                            'id': 'paper2', 
                            'title': 'Vision Transformer for Image Classification',
                            'abstract': 'While the Transformer architecture has become the de-facto standard for natural language processing...',
                            'published': '2020-10-22T00:00:00Z'
                        }
                    ]
                    
                    # Run analysis
                    collection_analysis = analyzer.analyze_paper_collection(mock_papers)
                    st.session_state.analysis_results = collection_analysis
                    
                    st.success("Paper analysis completed!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Papers Analyzed", collection_analysis['total_papers'])
                    with col2:
                        st.metric("🔬 Research Types", len(collection_analysis['research_type_distribution']))
                    with col3:
                        st.metric("💡 Emerging Concepts", len(collection_analysis['emerging_concepts']))
                    
                    # Domain distribution
                    if collection_analysis['domain_distribution']:
                        st.subheader("🌐 Research Domain Distribution")
                        
                        domain_df = pd.DataFrame(
                            list(collection_analysis['domain_distribution'].items()),
                            columns=['Domain', 'Score']
                        )
                        
                        fig = px.bar(domain_df, x='Domain', y='Score',
                                   title='Domain Influence Scores')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Research types
                    if collection_analysis['research_type_distribution']:
                        st.subheader("📊 Research Type Distribution")
                        
                        type_df = pd.DataFrame(
                            list(collection_analysis['research_type_distribution'].items()),
                            columns=['Type', 'Count']
                        )
                        
                        fig = px.pie(type_df, values='Count', names='Type',
                                   title='Research Contribution Types')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Emerging concepts
                    if collection_analysis['emerging_concepts']:
                        st.subheader("🌟 Emerging Concepts")
                        concepts_text = ", ".join(collection_analysis['emerging_concepts'])
                        st.write(concepts_text)
                    
                    # Generate report
                    if st.button("📄 Generate Trend Report"):
                        report = analyzer.generate_trend_report(collection_analysis)
                        st.text_area("Research Trend Report", report, height=400)
                
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    
    # Experiment history
    if st.session_state.experiment_results:
        st.subheader("🧪 Experiment History")
        
        for i, result in enumerate(st.session_state.experiment_results, 1):
            with st.expander(f"Experiment {i}: {result['test_results']['accuracy']:.1f}% accuracy"):
                st.write(f"**Hypothesis:** {result['hypothesis']}")
                st.write(f"**Model:** {result['model_info']['model_class']}")
                st.write(f"**Accuracy:** {result['test_results']['accuracy']:.2f}%")
                st.write(f"**Parameters:** {result['model_info']['total_parameters']:,}")

# Tab 5: System Status
with tab5:
    st.header("ℹ️ System Status & Information")
    
    # System status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖥️ Compute Resources")
        
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.success(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            st.info("💻 Using CPU (GPU not available)")
        
        st.write(f"**PyTorch Version:** {torch.__version__}")
        st.write(f"**Device Count:** {torch.cuda.device_count()}")
        
        # Memory usage
        import psutil
        memory = psutil.virtual_memory()
        st.write(f"**RAM Usage:** {memory.percent}% ({memory.used/1e9:.1f}GB / {memory.total/1e9:.1f}GB)")
    
    with col2:
        st.subheader("🤖 AI Provider Status")
        
        # Check LLM providers
        if use_llm:
            try:
                llm_gen = FreeLLMGenerator(hf_token)
                provider_status = llm_gen.get_provider_status()
                
                for provider, status in provider_status.items():
                    if status['available']:
                        st.success(f"✅ {provider.title()}: Available")
                    else:
                        st.warning(f"❌ {provider.title()}: Not available")
            except Exception as e:
                st.error(f"Error checking LLM providers: {e}")
        else:
            st.info("🎨 Using template-based generation (always available)")
    
    # Project information
    st.subheader("📋 Project Information")
    
    project_info = {
        "Version": "1.0.0",
        "License": "Custom",
        "Cost": "$0.00 (100% Free)",
        "Dependencies": "PyTorch, Transformers, Streamlit",
        "Data Sources": "arXiv (free), Public datasets",
        "Compute": "Local CPU/GPU, Free cloud options"
    }
    
    for key, value in project_info.items():
        st.write(f"**{key}:** {value}")
    
    # Quick actions
    st.subheader("🔧 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Knowledge Base"):
            if st.session_state.kb:
                st.session_state.kb.clear_database()
                st.success("Knowledge base cleared")
    
    with col2:
        if st.button("📁 Clear Results"):
            st.session_state.experiment_results = []
            st.session_state.analysis_results = None
            st.success("Results cleared")
    
    with col3:
        if st.button("🔄 Reset Session"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.success("Session reset")
            st.experimental_rerun()
    
    # Usage tips
    st.subheader("💡 Usage Tips")
    
    tips = [
        "🔍 Start with specific research queries for better results",
        "⚡ Use quick mode for faster experiments on CPU",
        "🤖 Install Ollama for better local LLM generation",
        "☁️ Use Google Colab for free GPU access",
        "📊 Export results for further analysis",
        "🔄 Try different model architectures for comparison"
    ]
    
    for tip in tips:
        st.write(tip)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
            <p>🤖 AI Researcher by Sreeram - Autonomous Research Assistant</p>
    <p>💰 100% Free • 🔓 Open Source • 🚀 No API Keys Required</p>
</div>
""", unsafe_allow_html=True)
