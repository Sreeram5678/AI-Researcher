# 🚀 AI Researcher Free - Usage Guide

Complete guide to using the AI Researcher Free system for autonomous research.

## 🎯 Quick Start Options

### Option 1: Command Line Demo (Fastest)
```bash
# Run the quick start demo
python demo/quick_start.py

# With custom settings
python demo/quick_start.py --query "computer vision attention" --papers 20
```

### Option 2: Web Interface (Recommended)
```bash
# Streamlit interface (most user-friendly)
streamlit run demo/streamlit_app.py

# Gradio interface (alternative)
python demo/gradio_app.py
```

### Option 3: Jupyter Notebook (Interactive)
```bash
# Launch Jupyter
jupyter notebook demo/jupyter_demo.ipynb

# Or use JupyterLab
jupyter lab demo/jupyter_demo.ipynb
```

### Option 4: Programmatic API
```python
# Use in your own Python scripts
from core import FreeKnowledgeBase, TemplateHypothesisGenerator
# ... your code here
```

## 📚 Complete Workflow Guide

### Step 1: Literature Analysis

**Goal**: Analyze research papers from arXiv to build a knowledge base

```python
from core.free_knowledge_base import FreeKnowledgeBase

# Initialize
kb = FreeKnowledgeBase()

# Analyze literature
result = kb.ingest_literature(
    query="your research domain",  # e.g., "attention mechanisms computer vision"
    max_papers=50                  # Adjust based on your needs
)

# Check results
stats = kb.get_statistics()
print(f"Processed {stats['total_papers']} papers")
print(f"Extracted {stats['total_concepts']} concepts")
```

**Tips**:
- Start with 20-30 papers for quick results
- Use specific keywords for focused analysis
- Check `stats['category_distribution']` to see research areas

### Step 2: Research Gap Identification

**Goal**: Find underexplored areas in your research domain

```python
# Find gaps
gaps = kb.find_research_gaps(
    domain="machine learning",  # Adjust to your domain
    min_papers=3               # Minimum papers for established concepts
)

# Examine gaps
for gap in gaps[:10]:
    print(f"Gap: {gap['concept']}")
    print(f"Description: {gap['description']}")
    print(f"Confidence: {gap['confidence_score']:.2f}")
    print(f"Papers: {gap['paper_count']}")
    print()
```

**Understanding Results**:
- `confidence_score`: Higher = more likely to be a real gap
- `paper_count`: Lower = less explored area
- `category`: Research domain classification

### Step 3: Hypothesis Generation

**Goal**: Generate testable research hypotheses based on identified gaps

#### Option A: Template-Based (Always Free)
```python
from core.template_generator import TemplateHypothesisGenerator

generator = TemplateHypothesisGenerator()
hypotheses = generator.generate_hypotheses(
    research_gaps=gaps[:5],    # Use top 5 gaps
    num_hypotheses=5           # Generate 5 hypotheses
)

for i, (hypothesis, confidence) in enumerate(hypotheses, 1):
    print(f"Hypothesis {i} (Confidence: {confidence:.2f}):")
    print(hypothesis)
    print()
```

#### Option B: Free LLM (Better Quality)
```python
from core.free_llm_generator import FreeLLMGenerator

# Initialize with optional Hugging Face token
generator = FreeLLMGenerator(huggingface_token="your_free_token")

# Check available providers
status = generator.get_provider_status()
print("Available providers:", status)

# Generate hypotheses
hypotheses = generator.generate_hypotheses(
    research_gaps=gaps[:3],
    num_hypotheses=3
)
```

**Free LLM Setup**:
1. **Ollama (Recommended)**: 
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama2:7b
   ollama serve
   ```

2. **Hugging Face (Free tier)**:
   - Get token: https://huggingface.co/settings/tokens
   - 30,000 free requests/month

### Step 4: Experiment Design & Execution

**Goal**: Test your hypotheses with machine learning experiments

```python
from core.experiment_runner import FreeExperimentRunner

runner = FreeExperimentRunner(use_gpu=True)  # Set False for CPU

# Select best hypothesis
best_hypothesis = hypotheses[0][0]

# Configure experiment
config = {
    'dataset': 'cifar10',        # or 'mnist'
    'model_type': 'attention',   # Based on hypothesis
    'epochs': 20,                # Training epochs
    'batch_size': 32,           # Batch size
    'learning_rate': 0.001,     # Learning rate
    'train_subset_size': 2000,  # Subset size for speed
    'test_subset_size': 400
}

# Run experiment
results = runner.run_vision_experiment(best_hypothesis, config)

# View results
print(f"Test Accuracy: {results['test_results']['accuracy']:.2f}%")
print(f"Model Size: {results['model_info']['model_size_mb']:.2f} MB")
```

**Experiment Types**:
- **Computer Vision**: Image classification with CIFAR-10/MNIST
- **Natural Language Processing**: Text classification with synthetic data
- **Custom**: Add your own datasets and models

**Resource Optimization**:
```python
# For faster experiments (CPU-friendly)
quick_config = {
    'epochs': 5,
    'train_subset_size': 500,
    'test_subset_size': 100,
    'batch_size': 16
}

# For better results (GPU recommended)
full_config = {
    'epochs': 50,
    'train_subset_size': 5000,
    'test_subset_size': 1000,
    'batch_size': 64
}
```

### Step 5: Analysis & Insights

**Goal**: Analyze results and generate research insights

```python
from core.paper_analyzer import PaperAnalyzer

analyzer = PaperAnalyzer()

# Analyze your paper collection
# (In practice, extract papers from knowledge base)
collection_analysis = analyzer.analyze_paper_collection(papers)

# Generate insights
print("Domain Distribution:", collection_analysis['domain_distribution'])
print("Methodology Trends:", collection_analysis['methodology_trends'])
print("Emerging Concepts:", collection_analysis['emerging_concepts'])
print("Research Gaps:", collection_analysis['research_gaps'])

# Generate report
report = analyzer.generate_trend_report(collection_analysis)
print(report)

# Create visualizations
plots = analyzer.create_visualizations(collection_analysis)
```

## 🔧 Configuration & Settings

### Environment Variables
```bash
# Optional: Hugging Face token for free LLM access
export HUGGINGFACE_API_KEY="your_free_token"

# Optional: Custom data directory
export AI_RESEARCHER_DATA_DIR="/path/to/data"

# Optional: Enable debug logging
export AI_RESEARCHER_DEBUG=1
```

### Configuration File (.env)
```bash
# Create .env file in project root
PROJECT_NAME=ai_researcher_free
DATA_DIR=./data
MODELS_DIR=./models
RESULTS_DIR=./results

# Free tier settings
MAX_PAPERS_PER_QUERY=50
MAX_HYPOTHESES=5
BATCH_SIZE=32
USE_FREE_MODELS=true

# Optional API keys
HUGGINGFACE_API_KEY=your_free_token_here
```

### Model Selection
```python
# Available models based on hypothesis
model_mapping = {
    'attention': 'Attention-enhanced CNN',
    'transformer': 'Simple Vision Transformer',
    'cnn': 'Standard CNN',
    'resnet': 'ResNet-inspired architecture'
}

# The system automatically selects based on hypothesis keywords
# You can also specify explicitly:
config['model_type'] = 'attention'
```

## 🚀 Performance Optimization

### CPU Optimization
```python
# Reduce memory usage
config.update({
    'batch_size': 16,           # Smaller batches
    'train_subset_size': 500,   # Smaller datasets
    'test_subset_size': 100,
    'epochs': 5                 # Fewer epochs
})

# Enable CPU optimizations
import torch
torch.set_num_threads(4)       # Use available CPU cores
```

### GPU Acceleration
```python
# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
# Use larger batches and datasets with GPU
gpu_config = {
    'batch_size': 64,
    'train_subset_size': 5000,
    'test_subset_size': 1000,
    'epochs': 20
}
```

### Cloud Resources (Free)
```python
# Google Colab setup
!git clone https://github.com/yourusername/ai-researcher-free.git
%cd ai-researcher-free
!pip install -r requirements.txt

# Check GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# Run with full GPU power
runner = FreeExperimentRunner(use_gpu=True)
```

## 📊 Understanding Results

### Knowledge Base Statistics
```python
stats = kb.get_statistics()

# Interpret the results
print(f"Papers: {stats['total_papers']}")           # Total papers analyzed
print(f"Concepts: {stats['total_concepts']}")       # Unique concepts found
print(f"Recent: {stats['recent_papers_week']}")     # Papers from last week
print(f"Size: {stats['database_size_mb']:.1f}MB")   # Database storage

# Category breakdown
for category, count in stats['category_distribution'].items():
    print(f"  {category}: {count} concepts")
```

### Research Gap Analysis
```python
# Gap quality indicators
for gap in gaps:
    confidence = gap['confidence_score']
    
    if confidence > 0.8:
        print(f"🔥 High-confidence gap: {gap['concept']}")
    elif confidence > 0.6:
        print(f"⚡ Medium-confidence gap: {gap['concept']}")
    else:
        print(f"💡 Low-confidence gap: {gap['concept']}")
```

### Hypothesis Quality
```python
# Hypothesis evaluation
for hypothesis, confidence in hypotheses:
    print(f"Hypothesis: {hypothesis}")
    
    # Quality indicators
    if confidence > 0.8:
        print("  Quality: Excellent ✅")
    elif confidence > 0.6:
        print("  Quality: Good 👍")
    else:
        print("  Quality: Fair 🤔")
    
    # Generate research questions
    questions = generator.generate_research_questions(hypothesis)
    print("  Key Questions:")
    for q in questions[:3]:
        print(f"    • {q}")
```

### Experiment Results
```python
# Understanding experiment metrics
results = runner.run_vision_experiment(hypothesis, config)

accuracy = results['test_results']['accuracy']
loss = results['test_results']['loss']
params = results['model_info']['total_parameters']

print(f"Accuracy: {accuracy:.2f}%")
if accuracy > 70:
    print("  Performance: Excellent! ✅")
elif accuracy > 50:
    print("  Performance: Good 👍")
else:
    print("  Performance: Needs improvement 🔧")

print(f"Model Complexity: {params:,} parameters")
if params < 100000:
    print("  Complexity: Lightweight 🪶")
elif params < 1000000:
    print("  Complexity: Medium ⚖️")
else:
    print("  Complexity: Heavy 🏋️")
```

## 🔄 Iterative Research Process

### Refining Your Research
```python
# 1. Start broad, then narrow down
initial_query = "machine learning"
refined_query = "attention mechanisms transformer networks"
specific_query = "multi-head attention computer vision"

# 2. Iterative gap analysis
for iteration in range(3):
    gaps = kb.find_research_gaps(domain=f"iteration_{iteration}")
    # Analyze and refine
    
# 3. Hypothesis refinement
base_hypotheses = generator.generate_hypotheses(gaps[:3], 3)
refined_hypotheses = generator.generate_hypotheses(gaps[3:6], 3)
# Compare and select best

# 4. Experiment comparison
baseline_results = runner.run_vision_experiment(baseline_hypothesis, config)
improved_results = runner.run_vision_experiment(improved_hypothesis, config)
# Compare performance
```

### Building on Results
```python
# Save and load previous work
import pickle

# Save knowledge base
with open('my_research_kb.pkl', 'wb') as f:
    pickle.dump(kb, f)

# Save hypotheses
generator.export_hypotheses(hypotheses, 'my_hypotheses.json')

# Save experiment results
with open('my_experiments.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Load and continue
with open('my_research_kb.pkl', 'rb') as f:
    kb = pickle.load(f)
```

## 🎯 Advanced Usage Patterns

### Research Domain Exploration
```python
# Explore multiple related domains
domains = [
    "computer vision attention mechanisms",
    "natural language processing transformers", 
    "reinforcement learning neural networks"
]

all_gaps = []
for domain in domains:
    result = kb.ingest_literature(domain, max_papers=20)
    gaps = kb.find_research_gaps(domain)
    all_gaps.extend(gaps)

# Find cross-domain opportunities
cross_domain_gaps = [gap for gap in all_gaps if gap['confidence_score'] > 0.7]
```

### Comparative Analysis
```python
# Compare different approaches
approaches = ['cnn', 'attention', 'transformer']
results_comparison = {}

for approach in approaches:
    config['model_type'] = approach
    result = runner.run_vision_experiment(hypothesis, config)
    results_comparison[approach] = result['test_results']['accuracy']

# Find best approach
best_approach = max(results_comparison, key=results_comparison.get)
print(f"Best approach: {best_approach} ({results_comparison[best_approach]:.2f}%)")
```

### Batch Processing
```python
# Process multiple hypotheses
batch_results = []

for i, (hypothesis, confidence) in enumerate(hypotheses):
    print(f"Testing hypothesis {i+1}/{len(hypotheses)}")
    
    config['epochs'] = 5  # Quick experiments
    result = runner.run_vision_experiment(hypothesis, config)
    
    batch_results.append({
        'hypothesis': hypothesis,
        'confidence': confidence,
        'accuracy': result['test_results']['accuracy'],
        'parameters': result['model_info']['total_parameters']
    })

# Sort by performance
batch_results.sort(key=lambda x: x['accuracy'], reverse=True)
print("Best performing hypotheses:")
for result in batch_results[:3]:
    print(f"  {result['accuracy']:.1f}%: {result['hypothesis'][:60]}...")
```

## ❓ Troubleshooting Common Issues

### Memory Issues
```python
# Reduce memory usage
import gc
import torch

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Force garbage collection
gc.collect()

# Use smaller configurations
minimal_config = {
    'epochs': 3,
    'batch_size': 8,
    'train_subset_size': 200,
    'test_subset_size': 50
}
```

### Network/API Issues
```python
# Handle network timeouts
import time

def robust_literature_analysis(query, max_papers, retries=3):
    for attempt in range(retries):
        try:
            return kb.ingest_literature(query, max_papers)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)  # Wait before retry
            else:
                raise

# Use smaller batches
result = robust_literature_analysis("your query", max_papers=10)
```

### Performance Issues
```python
# Profile your code
import cProfile

def profile_experiment():
    return runner.run_vision_experiment(hypothesis, config)

cProfile.run('profile_experiment()')

# Monitor resource usage
import psutil
import time

def monitor_resources():
    while True:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        print(f"CPU: {cpu}%, Memory: {memory}%")
        time.sleep(1)
```

This usage guide should help you get the most out of AI Researcher Free! Remember, everything is 100% free - no API keys, no subscriptions, no limits. Happy researching! 🚀
