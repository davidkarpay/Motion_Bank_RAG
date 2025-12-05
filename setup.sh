#!/bin/bash
# Motion RAG Setup Script
# =======================

set -e

echo "🔧 Setting up Motion RAG Framework..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
required_version="3.10"

if [[ $(echo -e "$python_version\n$required_version" | sort -V | head -n1) != "$required_version" ]]; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "📥 Downloading spaCy English model..."
python -m spacy download en_core_web_sm || true

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/motions
mkdir -p data/uploads
mkdir -p data/embeddings

# Check for Qdrant
echo "🔍 Checking for Qdrant..."
if command -v docker &> /dev/null; then
    if ! docker ps | grep -q qdrant; then
        echo "🐳 Starting Qdrant via Docker..."
        docker run -d -p 6333:6333 -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant
        echo "⏳ Waiting for Qdrant to start..."
        sleep 5
    else
        echo "✅ Qdrant already running"
    fi
else
    echo "⚠️  Docker not found. Please install Qdrant manually:"
    echo "   docker run -d -p 6333:6333 qdrant/qdrant"
fi

# Check for Ollama
echo "🔍 Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found"
    
    # Check if model is available
    if ! ollama list | grep -q "phi3"; then
        echo "📥 Pulling Phi-3 model (this may take a while)..."
        ollama pull phi3:14b
    else
        echo "✅ Phi-3 model available"
    fi
else
    echo "⚠️  Ollama not found. Please install from: https://ollama.ai"
    echo "   Then run: ollama pull phi3:14b"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Start API server: python -m app.api.main"
echo "  3. Start UI (new terminal): streamlit run app/chat/streamlit_app.py"
echo ""
echo "Or use: ./scripts/start.sh"
