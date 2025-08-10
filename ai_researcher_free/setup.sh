#!/bin/bash

# AI Researcher by Sreeram - Setup Script
# 100% Free setup with no paid services required

echo "🤖 Setting up AI Researcher by Sreeram..."
echo "💰 Total cost: $0.00"

# Check if Python 3.8+ is available
python3 --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Python 3 is required. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing free packages..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project structure..."
mkdir -p data
mkdir -p models
mkdir -p results
mkdir -p logs

# Download free models
echo "🤖 Downloading free models..."
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading lightweight embedding model...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Model downloaded successfully')
"

# Set up environment variables
echo "⚙️ Setting up environment..."
cat > .env << 'EOF'
# AI Researcher by Sreeram Configuration
PROJECT_NAME=ai_researcher_free
DATA_DIR=./data
MODELS_DIR=./models
RESULTS_DIR=./results
LOGS_DIR=./logs

# Free tier settings
MAX_PAPERS_PER_QUERY=50
MAX_HYPOTHESES=5
BATCH_SIZE=10
USE_FREE_MODELS=true

# Optional: Add your free API keys here
# HUGGINGFACE_API_KEY=your_free_hf_token_here
# (Get free token from https://huggingface.co/settings/tokens)
EOF

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the demo: streamlit run demo/streamlit_app.py"
echo "3. Or try Jupyter: jupyter notebook demo/jupyter_demo.ipynb"
echo ""
echo "📖 Read README.md for detailed usage instructions"
echo "💡 Everything is 100% free - no API keys required to start!"
