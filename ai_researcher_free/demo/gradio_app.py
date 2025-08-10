"""
AI Researcher by Sreeram - Gradio Interface
Alternative web interface using Gradio
"""

import gradio as gr
import sys
import os
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import core modules
from core.free_knowledge_base import FreeKnowledgeBase
from core.template_generator import TemplateHypothesisGenerator
from core.free_llm_generator import FreeLLMGenerator
from core.experiment_runner import FreeExperimentRunner
from core.paper_analyzer import PaperAnalyzer

# Global state
app_state = {
    'kb': None,
    'research_gaps': [],
    'hypotheses': [],
    'experiment_results': []
}

def analyze_literature(research_query, max_papers, progress=gr.Progress()):
    """Analyze literature from arXiv"""
    try:
        progress(0.1, desc="Initializing knowledge base...")
        
        kb = FreeKnowledgeBase()
        app_state['kb'] = kb
        
        progress(0.3, desc="Downloading papers from arXiv...")
        
        result = kb.ingest_literature(research_query, max_papers=max_papers)
        
        progress(0.7, desc="Analyzing papers and extracting concepts...")
        
        # Get statistics
        stats = kb.get_statistics()
        
        # Find research gaps
        gaps = kb.find_research_gaps("machine learning")
        app_state['research_gaps'] = gaps
        
        progress(1.0, desc="Analysis complete!")
        
        # Format results
        summary = f"""
## Literature Analysis Results

### 📊 Statistics
- **Papers Processed:** {result.get('total_processed', 0)}
- **Total Papers in DB:** {stats['total_papers']}
- **Concepts Extracted:** {stats['total_concepts']}
- **Research Gaps Found:** {len(gaps)}
- **Database Size:** {stats['database_size_mb']:.1f} MB

### 🔍 Top Research Gaps
"""
        
        for i, gap in enumerate(gaps[:5], 1):
            summary += f"""
**{i}. {gap['concept']}**
- Description: {gap['description']}
- Category: {gap['category']}
- Confidence: {gap['confidence_score']:.2f}
- Papers: {gap['paper_count']}
"""
        
        # Create domain distribution chart
        if stats['category_distribution']:
            categories = list(stats['category_distribution'].keys())
            counts = list(stats['category_distribution'].values())
            
            plt.figure(figsize=(10, 6))
            plt.pie(counts, labels=categories, autopct='%1.1f%%')
            plt.title('Research Category Distribution')
            chart_path = "results/domain_distribution.png"
            os.makedirs("results", exist_ok=True)
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return summary, chart_path
        
        return summary, None
        
    except Exception as e:
        error_msg = f"❌ Error during literature analysis: {str(e)}"
        return error_msg, None

def generate_hypotheses(generation_method, num_hypotheses, hf_token, progress=gr.Progress()):
    """Generate research hypotheses"""
    try:
        if not app_state['research_gaps']:
            return "❌ Please run literature analysis first to identify research gaps."
        
        progress(0.2, desc="Initializing hypothesis generator...")
        
        if generation_method == "Template-based (Free)":
            generator = TemplateHypothesisGenerator()
            hypotheses = generator.generate_hypotheses(
                app_state['research_gaps'], 
                num_hypotheses
            )
            method_used = "Template-based"
            
        elif generation_method == "Free LLM":
            generator = FreeLLMGenerator(hf_token if hf_token else None)
            progress(0.5, desc="Generating with free LLM...")
            hypotheses = generator.generate_hypotheses(
                app_state['research_gaps'],
                num_hypotheses=num_hypotheses
            )
            method_used = "Free LLM"
        
        else:  # Hybrid
            # Try LLM first, fallback to template
            try:
                generator = FreeLLMGenerator(hf_token if hf_token else None)
                hypotheses = generator.generate_hypotheses(
                    app_state['research_gaps'],
                    num_hypotheses=num_hypotheses
                )
                method_used = "Hybrid (LLM)"
            except:
                generator = TemplateHypothesisGenerator()
                hypotheses = generator.generate_hypotheses(
                    app_state['research_gaps'], 
                    num_hypotheses
                )
                method_used = "Hybrid (Template fallback)"
        
        progress(0.9, desc="Formatting results...")
        
        if not hypotheses:
            return "❌ No hypotheses were generated. Try adjusting settings."
        
        app_state['hypotheses'] = hypotheses
        
        # Format results
        result_text = f"""
## 💡 Generated Research Hypotheses
**Method Used:** {method_used}
**Number of Hypotheses:** {len(hypotheses)}

"""
        
        for i, (hypothesis, confidence) in enumerate(hypotheses, 1):
            result_text += f"""
### Hypothesis {i} (Confidence: {confidence:.2f})
{hypothesis}

"""
            
            # Add methodology if available
            if hasattr(generator, 'generate_methodology_description'):
                methodology = generator.generate_methodology_description(hypothesis)
                result_text += f"**Suggested Methodology:** {methodology}\n\n"
        
        progress(1.0, desc="Hypotheses generated!")
        return result_text
        
    except Exception as e:
        return f"❌ Error generating hypotheses: {str(e)}"

def run_experiment(experiment_type, dataset, model_type, epochs, batch_size, learning_rate, quick_mode, use_gpu, progress=gr.Progress()):
    """Run ML experiment"""
    try:
        if not app_state['hypotheses']:
            return "❌ Please generate hypotheses first."
        
        best_hypothesis = app_state['hypotheses'][0][0]  # Use best hypothesis
        
        progress(0.1, desc="Initializing experiment runner...")
        
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
        
        progress(0.3, desc="Setting up model and data...")
        
        # Run experiment
        if experiment_type == "Computer Vision":
            results = runner.run_vision_experiment(best_hypothesis, config)
        else:
            results = runner.run_nlp_experiment(best_hypothesis, config)
        
        progress(0.9, desc="Generating report...")
        
        app_state['experiment_results'].append(results)
        
        # Create training plots
        if results['training_history']:
            history_df = pd.DataFrame(results['training_history'])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Accuracy plot
            ax1.plot(history_df['epoch'], history_df['accuracy'], 'b-', marker='o')
            ax1.set_title('Training Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.grid(True, alpha=0.3)
            
            # Loss plot
            ax2.plot(history_df['epoch'], history_df['loss'], 'r-', marker='o')
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            chart_path = f"results/experiment_{int(time.time())}.png"
            os.makedirs("results", exist_ok=True)
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            chart_path = None
        
        # Format results
        result_text = f"""
## 🧪 Experiment Results

### Hypothesis Tested
{best_hypothesis}

### Configuration
- **Experiment Type:** {experiment_type}
- **Dataset:** {dataset}
- **Model:** {model_type}
- **Epochs:** {epochs}
- **Batch Size:** {batch_size}
- **Learning Rate:** {learning_rate}

### Results
- **Test Accuracy:** {results['test_results']['accuracy']:.2f}%
- **Test Loss:** {results['test_results']['loss']:.4f}
- **Model Parameters:** {results['model_info']['total_parameters']:,}
- **Model Size:** {results['model_info']['model_size_mb']:.2f} MB
- **Device Used:** {results['device_used']}

### Training Progress
"""
        
        for epoch_data in results['training_history'][-5:]:  # Last 5 epochs
            result_text += f"- Epoch {epoch_data['epoch']}: {epoch_data['accuracy']:.2f}% accuracy, {epoch_data['loss']:.4f} loss\n"
        
        progress(1.0, desc="Experiment completed!")
        
        return result_text, chart_path
        
    except Exception as e:
        return f"❌ Experiment failed: {str(e)}", None

def analyze_papers(progress=gr.Progress()):
    """Analyze research papers for trends"""
    try:
        if not app_state['kb']:
            return "❌ Please run literature analysis first."
        
        progress(0.2, desc="Initializing paper analyzer...")
        
        analyzer = PaperAnalyzer()
        
        # Mock papers for demo (replace with real data from KB)
        mock_papers = [
            {
                'id': 'paper1',
                'title': 'Attention Is All You Need',
                'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism.',
                'published': '2017-06-12T00:00:00Z'
            },
            {
                'id': 'paper2',
                'title': 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale',
                'abstract': 'While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited.',
                'published': '2020-10-22T00:00:00Z'
            }
        ]
        
        progress(0.6, desc="Analyzing paper collection...")
        
        collection_analysis = analyzer.analyze_paper_collection(mock_papers)
        
        progress(0.9, desc="Creating visualizations...")
        
        # Create visualizations
        plots = analyzer.create_visualizations(collection_analysis)
        
        # Generate report
        report = analyzer.generate_trend_report(collection_analysis)
        
        progress(1.0, desc="Analysis complete!")
        
        chart_path = plots[0] if plots else None
        
        return report, chart_path
        
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}", None

def get_system_status():
    """Get system status information"""
    try:
        import torch
        import psutil
        
        # GPU info
        if torch.cuda.is_available():
            gpu_info = f"✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)"
        else:
            gpu_info = "💻 Using CPU (GPU not available)"
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = f"🖥️ RAM: {memory.percent}% used ({memory.used/1e9:.1f}GB / {memory.total/1e9:.1f}GB)"
        
        # LLM providers
        llm_status = "🤖 LLM Providers:\n"
        try:
            llm_gen = FreeLLMGenerator()
            provider_status = llm_gen.get_provider_status()
            
            for provider, status in provider_status.items():
                if status['available']:
                    llm_status += f"  ✅ {provider.title()}: Available\n"
                else:
                    llm_status += f"  ❌ {provider.title()}: Not available\n"
        except:
            llm_status += "  🎨 Template-based generation (always available)\n"
        
        # Application state
        state_info = f"""
📊 Application State:
- Knowledge Base: {'✅ Loaded' if app_state['kb'] else '❌ Not loaded'}
- Research Gaps: {len(app_state['research_gaps'])} found
- Hypotheses: {len(app_state['hypotheses'])} generated
- Experiments: {len(app_state['experiment_results'])} completed
"""
        
        system_info = f"""
## 🖥️ System Status

{gpu_info}
{memory_info}

{llm_status}

{state_info}

## 📋 Project Information
- **Version:** 1.0.0
- **License:** Apache 2.0
- **Cost:** $0.00 (100% Free)
- **Dependencies:** PyTorch, Transformers, Gradio
- **Data Sources:** arXiv (free), Public datasets
"""
        
        return system_info
        
    except Exception as e:
        return f"❌ Error getting system status: {str(e)}"

def clear_data(data_type):
    """Clear application data"""
    try:
        if data_type == "Knowledge Base":
            if app_state['kb']:
                app_state['kb'].clear_database()
            app_state['kb'] = None
            app_state['research_gaps'] = []
            return "✅ Knowledge base cleared"
        
        elif data_type == "Hypotheses":
            app_state['hypotheses'] = []
            return "✅ Hypotheses cleared"
        
        elif data_type == "Experiments":
            app_state['experiment_results'] = []
            return "✅ Experiment results cleared"
        
        elif data_type == "All Data":
            if app_state['kb']:
                app_state['kb'].clear_database()
            app_state['kb'] = None
            app_state['research_gaps'] = []
            app_state['hypotheses'] = []
            app_state['experiment_results'] = []
            return "✅ All data cleared"
        
        return "❌ Unknown data type"
        
    except Exception as e:
        return f"❌ Error clearing data: {str(e)}"

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="AI Researcher by Sreeram", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.Markdown("""
        # 🤖 AI Researcher by Sreeram
        ## Autonomous AI Research Assistant - 100% Free Edition
        
        Discover research gaps, generate hypotheses, and run experiments using only free resources!
        """)
        
        with gr.Tabs():
            
            # Tab 1: Literature Analysis
            with gr.Tab("📚 Literature Analysis"):
                gr.Markdown("### Analyze research papers from arXiv")
                
                with gr.Row():
                    research_query = gr.Textbox(
                        label="Research Query",
                        value="attention mechanisms computer vision",
                        placeholder="Enter keywords to search arXiv"
                    )
                    max_papers = gr.Slider(
                        label="Max Papers",
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=10
                    )
                
                analyze_btn = gr.Button("🔍 Analyze Literature", variant="primary")
                
                with gr.Row():
                    literature_output = gr.Markdown(label="Analysis Results")
                    literature_chart = gr.Image(label="Domain Distribution")
                
                analyze_btn.click(
                    analyze_literature,
                    inputs=[research_query, max_papers],
                    outputs=[literature_output, literature_chart]
                )
            
            # Tab 2: Hypothesis Generation
            with gr.Tab("💡 Hypothesis Generation"):
                gr.Markdown("### Generate novel research hypotheses")
                
                with gr.Row():
                    generation_method = gr.Dropdown(
                        label="Generation Method",
                        choices=["Template-based (Free)", "Free LLM", "Hybrid"],
                        value="Template-based (Free)"
                    )
                    num_hypotheses = gr.Slider(
                        label="Number of Hypotheses",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )
                
                hf_token = gr.Textbox(
                    label="Hugging Face Token (Optional)",
                    type="password",
                    placeholder="Get free token from https://huggingface.co/settings/tokens"
                )
                
                generate_btn = gr.Button("🧠 Generate Hypotheses", variant="primary")
                hypothesis_output = gr.Markdown(label="Generated Hypotheses")
                
                generate_btn.click(
                    generate_hypotheses,
                    inputs=[generation_method, num_hypotheses, hf_token],
                    outputs=[hypothesis_output]
                )
            
            # Tab 3: Experiment Runner
            with gr.Tab("🧪 Experiment Runner"):
                gr.Markdown("### Run ML experiments to test hypotheses")
                
                with gr.Row():
                    with gr.Column():
                        experiment_type = gr.Dropdown(
                            label="Experiment Type",
                            choices=["Computer Vision", "Natural Language Processing"],
                            value="Computer Vision"
                        )
                        dataset = gr.Dropdown(
                            label="Dataset",
                            choices=["CIFAR-10", "MNIST"],
                            value="CIFAR-10"
                        )
                        model_type = gr.Dropdown(
                            label="Model Type",
                            choices=["CNN", "Attention-CNN", "Simple-ViT", "ResNet"],
                            value="CNN"
                        )
                    
                    with gr.Column():
                        epochs = gr.Slider(label="Epochs", minimum=1, maximum=50, value=10, step=1)
                        batch_size = gr.Slider(label="Batch Size", minimum=16, maximum=128, value=32, step=16)
                        learning_rate = gr.Dropdown(
                            label="Learning Rate",
                            choices=[0.001, 0.01, 0.0001],
                            value=0.001
                        )
                
                with gr.Row():
                    quick_mode = gr.Checkbox(label="Quick Mode (faster, smaller datasets)", value=True)
                    use_gpu = gr.Checkbox(label="Use GPU if available", value=True)
                
                run_btn = gr.Button("🚀 Run Experiment", variant="primary")
                
                with gr.Row():
                    experiment_output = gr.Markdown(label="Experiment Results")
                    experiment_chart = gr.Image(label="Training Progress")
                
                run_btn.click(
                    run_experiment,
                    inputs=[experiment_type, dataset, model_type, epochs, batch_size, learning_rate, quick_mode, use_gpu],
                    outputs=[experiment_output, experiment_chart]
                )
            
            # Tab 4: Analysis Dashboard
            with gr.Tab("📊 Analysis Dashboard"):
                gr.Markdown("### Research trend analysis and insights")
                
                analyze_papers_btn = gr.Button("🔬 Analyze Papers", variant="primary")
                
                with gr.Row():
                    analysis_output = gr.Markdown(label="Analysis Report")
                    analysis_chart = gr.Image(label="Trend Visualization")
                
                analyze_papers_btn.click(
                    analyze_papers,
                    outputs=[analysis_output, analysis_chart]
                )
            
            # Tab 5: System Status
            with gr.Tab("ℹ️ System Status"):
                gr.Markdown("### System information and controls")
                
                status_btn = gr.Button("🔄 Refresh Status")
                status_output = gr.Markdown(label="System Status")
                
                status_btn.click(
                    get_system_status,
                    outputs=[status_output]
                )
                
                gr.Markdown("### 🧹 Data Management")
                
                with gr.Row():
                    clear_type = gr.Dropdown(
                        label="Clear Data Type",
                        choices=["Knowledge Base", "Hypotheses", "Experiments", "All Data"],
                        value="Knowledge Base"
                    )
                    clear_btn = gr.Button("🗑️ Clear Data", variant="secondary")
                
                clear_output = gr.Textbox(label="Status")
                
                clear_btn.click(
                    clear_data,
                    inputs=[clear_type],
                    outputs=[clear_output]
                )
                
                # Load status on startup
                interface.load(get_system_status, outputs=[status_output])
        
        # Footer
        gr.Markdown("""
        ---
        🤖 **AI Researcher by Sreeram** - 100% Free Autonomous Research Assistant  
        💰 No costs • 🔓 Open source • 🚀 No API keys required
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()
    
    # Launch with sharing enabled for easy access
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed errors
    )
