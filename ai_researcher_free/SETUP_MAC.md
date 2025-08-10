# 🍎 AI Researcher by Sreeram - Mac Setup Guide

Quick setup guide specifically for Mac users with 8GB RAM and 256GB storage.

## 🚀 Quick Start (Recommended)

### Option 1: Mac Optimized Version
```bash
# Navigate to the project
cd ai_researcher_free

# Install basic dependencies
pip install requests matplotlib pandas seaborn

# Run Mac-optimized demo
python run_mac.py
# Choose option 1: Mac Optimized Demo
```

### Option 2: Minimal Version
```bash
# Run minimal demo (no external dependencies)
python demo/minimal_demo.py
```

## 📦 Dependency Issues & Solutions

### Common Error: `ModuleNotFoundError: No module named '_lzma'`

**Solution:**
```bash
# Install xz library
brew install xz

# Reinstall Python packages
pip install --upgrade pip
pip install requests matplotlib pandas
```

### PyTorch/Transformers Issues

**Solution: Use Mac-optimized version**
- Skip heavy ML dependencies
- Use arXiv API for literature analysis
- Use rule-based concept extraction
- Still get full research workflow!

## 🔧 Installation Options

### Option A: Basic Setup (Recommended for Mac)
```bash
# Essential packages only
pip install requests matplotlib pandas seaborn arxiv beautifulsoup4

# Run Mac optimized demo
python demo/mac_optimized_demo.py
```

### Option B: Full Setup (If you want ML features)
```bash
# Install system dependency first
brew install xz

# Try installing full requirements
pip install -r requirements.txt

# If it fails, use Option A instead
```

### Option C: Virtual Environment
```bash
# Create fresh environment
python3 -m venv mac_env
source mac_env/bin/activate

# Install basic packages
pip install requests matplotlib pandas

# Test with minimal demo
python demo/minimal_demo.py
```

## 🎯 What Works on Your Mac

### ✅ **Fully Functional (Mac Optimized)**
- 📚 **Literature Analysis**: Downloads papers from arXiv
- 🔍 **Research Gap Detection**: Finds underexplored areas
- 💡 **Hypothesis Generation**: Creates research hypotheses
- 📊 **Trend Analysis**: Analyzes research patterns
- 📈 **Visualizations**: Creates charts and graphs
- 💾 **Data Export**: Saves results to JSON/images

### ⚠️ **Limited (Requires Cloud)**
- 🧪 **ML Experiments**: Use Google Colab instead
- 🤖 **Local LLM**: Install Ollama separately
- 🌐 **Web Interface**: May need dependency fixes

### ❌ **Not Recommended**
- Large dataset experiments (use cloud)
- Heavy transformer models (8GB RAM limit)

## 🚀 Recommended Workflow for Mac

### Step 1: Quick Test
```bash
python demo/minimal_demo.py
```

### Step 2: Enhanced Analysis
```bash
python demo/mac_optimized_demo.py
```

### Step 3: Custom Research
```bash
python demo/mac_optimized_demo.py --query "your research topic" --papers 10
```

### Step 4: Export & Analyze
- Results saved to `results/` directory
- Visualizations as PNG files
- Hypotheses as JSON files

## 📊 Performance on Your Mac

### Expected Performance:
- **Literature Analysis**: 5-10 papers in ~30 seconds
- **Gap Detection**: Near-instant
- **Hypothesis Generation**: ~5 seconds
- **Visualization Creation**: ~10 seconds
- **Memory Usage**: ~2-3GB RAM
- **Storage**: ~100MB for data

### Tips for Better Performance:
- Start with 5-10 papers
- Use specific research queries
- Clear results/ directory periodically
- Close other applications during analysis

## 🔧 Troubleshooting

### Issue: "No module named 'xyz'"
```bash
pip install xyz
```

### Issue: "Connection timeout"
```bash
# Try with fewer papers
python demo/mac_optimized_demo.py --papers 3
```

### Issue: "Out of memory"
```bash
# Use minimal demo
python demo/minimal_demo.py
```

### Issue: "Permission denied"
```bash
chmod +x run_mac.py
chmod +x demo/*.py
```

## 🌟 Advanced Options

### Local LLM (Optional)
```bash
# Install Ollama
brew install ollama

# Download model
ollama pull llama2:7b

# Start server
ollama serve

# Use with hypothesis generation
python demo/mac_optimized_demo.py
```

### Google Colab Integration
1. Upload `demo/jupyter_demo.ipynb` to Colab
2. Get free GPU access
3. Run full ML experiments

### Web Interface (If dependencies work)
```bash
pip install streamlit
streamlit run demo/streamlit_app.py
```

## 📋 Mac-Specific Features

### Optimized for Mac:
- ✅ Uses system Python
- ✅ Lightweight dependencies
- ✅ Native file handling
- ✅ Energy efficient
- ✅ Works with 8GB RAM
- ✅ Respects storage limits

### Mac Integration:
- 📁 Results open in Finder
- 🖼️ Images open in Preview
- 📊 CSV files open in Numbers
- 📝 JSON files open in TextEdit

## 🎯 Quick Commands Summary

```bash
# Test basic functionality
python demo/minimal_demo.py

# Run full Mac-optimized version
python run_mac.py

# Custom research query
python demo/mac_optimized_demo.py --query "AI topic"

# Check what's installed
python run_mac.py
# Choose option 3

# Get help
python run_mac.py
# Choose option 4
```

## 📞 Support for Mac Users

### If Nothing Works:
1. Use `python demo/minimal_demo.py` - always works
2. Check Python version: `python --version` (need 3.8+)
3. Try in Google Colab for full features
4. Contact support with Mac version: `sw_vers`

### Success Indicators:
- ✅ Minimal demo runs without errors
- ✅ Can analyze 3-5 papers
- ✅ Generates research hypotheses
- ✅ Creates visualizations
- ✅ Exports results to files

---

**🍎 Mac users: You're all set! The system is optimized for your hardware and will work beautifully for AI research exploration.**

**💰 Remember: Everything is 100% FREE - no API keys, no subscriptions, no limits!**
