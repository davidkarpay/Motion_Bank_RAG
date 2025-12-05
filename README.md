# ⚖️ Motion RAG Framework

A locally-hosted RAG (Retrieval-Augmented Generation) system for criminal defense attorneys to strategize and draft motions based on a searchable database of past successful motions.

## 🎯 Purpose

This framework enables public defenders and criminal defense attorneys to:

- **Search** their motion database using natural language queries
- **Analyze** success patterns by motion type, judge, and legal issue  
- **Strategize** through AI-assisted chat grounded in successful precedents
- **Draft** new motions based on patterns from winning filings

All data stays local—no client information ever leaves your infrastructure.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  Search  │ │ Strategy │ │  Draft   │ │Analytics │           │
│  │  Motions │ │   Chat   │ │  Motion  │ │Dashboard │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   Ingestion  │ │   Retrieval  │ │  Generation  │            │
│  │   Pipeline   │ │   Pipeline   │ │   Pipeline   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└────────┬────────────────────┬────────────────────┬──────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   BGE-M3     │    │    Qdrant    │    │   Ollama     │
│  Embeddings  │    │  Vector DB   │    │  (Phi-3/     │
│              │    │              │    │   Mistral)   │
└──────────────┘    └──────────────┘    └──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- Ollama (for local LLM)
- 16GB+ RAM recommended

### Installation

```bash
# Clone the repository
cd motion-rag

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or manual setup:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Qdrant (vector database)
docker run -d -p 6333:6333 -v ./data/qdrant:/qdrant/storage qdrant/qdrant

# Install Ollama and pull model
# Visit https://ollama.ai for installation
ollama pull phi3:14b
```

### Launch

```bash
# Start all services
./scripts/start.sh

# Or manually:
# Terminal 1: API Server
python -m app.api.main

# Terminal 2: Streamlit UI
streamlit run app/chat/streamlit_app.py
```

Access the UI at `http://localhost:8501`

## 📁 Project Structure

```
motion-rag/
├── app/
│   ├── api/              # FastAPI backend
│   │   └── main.py       # API endpoints
│   ├── chat/             # Streamlit frontend
│   │   └── streamlit_app.py
│   ├── core/             # Data models
│   │   └── models.py
│   ├── ingestion/        # Document processing
│   │   └── document_processor.py
│   ├── retrieval/        # Search & retrieval
│   │   ├── vector_store.py
│   │   └── hybrid_retriever.py
│   └── generation/       # LLM & drafting
│       └── llm_service.py
├── config/
│   └── settings.py       # Configuration
├── data/
│   ├── motions/          # Motion files
│   ├── uploads/          # Uploaded files
│   └── metadata.json     # Motion metadata
├── scripts/
│   ├── setup.sh
│   └── start.sh
└── requirements.txt
```

## 🔧 Configuration

Edit `config/settings.py` to customize:

```python
# Embedding model (default: BGE-M3)
embedding.model_name = "BAAI/bge-m3"

# Vector database
vector_db.host = "localhost"
vector_db.port = 6333

# LLM (via Ollama)
llm.model = "phi3:14b"  # or "mistral:7b" for smaller footprint
llm.temperature = 0.3   # Lower for legal precision

# Retrieval settings
retrieval.top_k_final = 8
retrieval.min_relevance_score = 0.5
```

## 📖 Usage Guide

### 1. Ingesting Motions

**Via UI:**
Navigate to "Manage Database" → "Upload Motion"

**Via API:**
```bash
# Single file
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "file=@motion.pdf" \
  -F "outcome=granted" \
  -F "judge=Hon. Smith"

# Batch import
curl -X POST "http://localhost:8000/ingest/batch" \
  -d '"/path/to/motions/folder"'
```

**Supported formats:** PDF, DOCX, TXT

### 2. Searching Motions

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Fourth Amendment traffic stop consent search",
    "motion_type": "motion_to_suppress",
    "outcome": "granted",
    "top_k": 5
  }'
```

### 3. Strategy Sessions

Create a session and chat with AI grounded in your motion database:

```bash
# Create session
curl -X POST "http://localhost:8000/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "motion_type": "motion_to_suppress",
    "key_facts": "Traffic stop for broken taillight..."
  }'

# Chat
curl -X POST "http://localhost:8000/sessions/{session_id}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What grounds exist for suppression?",
    "use_retrieval": true
  }'
```

### 4. Generating Drafts

```bash
curl -X POST "http://localhost:8000/draft" \
  -H "Content-Type: application/json" \
  -d '{
    "motion_type": "motion_to_suppress",
    "facts": "On January 15, 2024, officers pulled over defendant...",
    "legal_issues": ["warrantless search", "lack of consent"]
  }'
```

## 📊 Supported Motion Types

- Motion to Suppress Evidence
- Motion to Suppress Statements
- Motion to Dismiss
- Speedy Trial Demand
- Motion for Discovery
- Motion for Bond Reduction
- Motion to Sever
- Richardson Motion (FL Discovery Violation)
- Nelson Motion (Self-Representation)
- Sentencing Memorandum
- And more...

## 🔐 Security Considerations

This framework is designed for local deployment to maintain attorney-client privilege:

- **No cloud dependencies** for core functionality
- **Case segregation** via vector database filtering
- **Audit logging** for compliance
- **No training on your data** - models are pre-trained

### Recommended Security Measures

1. Run on isolated network or air-gapped system
2. Enable disk encryption for data directory
3. Implement access controls at OS level
4. Regular backup of `data/` directory
5. Review audit logs periodically

## 🔬 Technical Details

### Embedding Model
BGE-M3 provides:
- 1024-dimensional dense vectors
- 8192 token context window
- 100+ language support
- Optimized for retrieval tasks

### Retrieval Pipeline
1. **Semantic Search** - Dense vector similarity via Qdrant
2. **Keyword Search** - BM25 for precise legal terms
3. **Hybrid Fusion** - Weighted combination (70/30 default)
4. **Cross-Encoder Reranking** - Fine-grained relevance scoring
5. **Parent Expansion** - Return full context for matched chunks

### Generation
- Phi-3 14B for complex legal reasoning
- Mistral 7B as lightweight alternative
- Temperature 0.3 for factual accuracy
- Grounded in retrieved context with citations

## 🛠️ Development

```bash
# Run tests
pytest tests/

# Format code
black app/
ruff check app/

# Type checking
mypy app/
```

## 📈 Roadmap

- [ ] DOCX export for generated motions
- [ ] Florida case law integration
- [ ] Multi-modal support (body cam transcripts)
- [ ] Fine-tuning on successful motions
- [ ] LegalServer integration
- [ ] Batch outcome tracking

## ⚠️ Disclaimer

**This tool is for attorney use only.** All AI-generated content requires thorough attorney review before filing. The system may produce inaccurate citations or arguments. Always verify:

- Case citations exist and are correctly quoted
- Statutory references are current
- Arguments are appropriate for your jurisdiction
- Facts are accurately represented

## 📄 License

[Specify your license]

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md before submitting PRs.

---

Built for public defenders who deserve the same tools as well-funded prosecution offices.
