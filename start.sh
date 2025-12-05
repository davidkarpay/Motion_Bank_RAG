#!/bin/bash
# Motion RAG Start Script
# =======================

set -e

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if Qdrant is running
if ! curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "⚠️  Qdrant not responding. Starting..."
    docker run -d -p 6333:6333 -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant
    sleep 5
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama not responding. Please start Ollama:"
    echo "   ollama serve"
    exit 1
fi

echo "🚀 Starting Motion RAG..."

# Start API server in background
echo "Starting API server on http://localhost:8000..."
python -m app.api.main &
API_PID=$!

# Wait for API to be ready
sleep 3

# Start Streamlit
echo "Starting Streamlit UI on http://localhost:8501..."
streamlit run app/chat/streamlit_app.py &
UI_PID=$!

echo ""
echo "✅ Motion RAG is running!"
echo "   API: http://localhost:8000"
echo "   UI:  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services..."

# Handle shutdown
trap "kill $API_PID $UI_PID 2>/dev/null; exit" SIGINT SIGTERM

# Wait for processes
wait
