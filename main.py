"""
Motion RAG API
FastAPI backend for motion strategy and drafting
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.models import (
    MotionMetadata, MotionDraft, ChatMessage, 
    StrategySession, MotionOutcome
)
from app.ingestion.document_processor import MotionIngestionPipeline
from app.retrieval.vector_store import EmbeddingService, VectorDBService, MetadataStore
from app.retrieval.hybrid_retriever import HybridRetriever, MotionAnalyzer
from app.generation.llm_service import MotionGenerator, ChatService
from config.settings import settings, MotionTypeEnum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Motion RAG API",
    description="RAG-powered motion strategy and drafting for criminal defense",
    version="0.1.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (lazy loading)
_services = {}

def get_services():
    """Lazy initialization of services"""
    if not _services:
        logger.info("Initializing services...")
        
        embedding_service = EmbeddingService()
        vector_db = VectorDBService()
        metadata_store = MetadataStore()
        
        retriever = HybridRetriever()
        analyzer = MotionAnalyzer()
        generator = MotionGenerator(retriever, analyzer)
        chat_service = ChatService(retriever, generator)
        ingestion = MotionIngestionPipeline()
        
        _services.update({
            "embedding": embedding_service,
            "vector_db": vector_db,
            "metadata": metadata_store,
            "retriever": retriever,
            "analyzer": analyzer,
            "generator": generator,
            "chat": chat_service,
            "ingestion": ingestion
        })
        
        logger.info("Services initialized")
    
    return _services


# ============== Request/Response Models ==============

class IngestRequest(BaseModel):
    file_path: str
    outcome: Optional[str] = None
    judge: Optional[str] = None
    charge_type: Optional[str] = None
    court: Optional[str] = None

class IngestResponse(BaseModel):
    motion_id: str
    title: str
    motion_type: str
    chunk_count: int
    message: str

class SearchRequest(BaseModel):
    query: str
    motion_type: Optional[str] = None
    outcome: Optional[str] = None
    judge: Optional[str] = None
    charge_type: Optional[str] = None
    top_k: int = 5

class SearchResult(BaseModel):
    motion_id: str
    title: str
    motion_type: str
    outcome: Optional[str]
    section: Optional[str]
    text_preview: str
    relevance_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    filters_applied: Dict[str, Any]

class DraftRequest(BaseModel):
    motion_type: str
    facts: str
    legal_issues: Optional[List[str]] = None

class DraftResponse(BaseModel):
    draft_id: str
    motion_type: str
    title: str
    content: str
    sections: Dict[str, str]
    cited_cases: List[str]
    cited_statutes: List[str]
    source_motion_ids: List[str]

class ChatRequest(BaseModel):
    message: str
    use_retrieval: bool = True

class ChatResponse(BaseModel):
    message_id: str
    content: str
    sources_used: int
    timestamp: str

class SessionRequest(BaseModel):
    motion_type: Optional[str] = None
    charge_type: Optional[str] = None
    key_facts: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    motion_type: Optional[str]
    charge_type: Optional[str]
    created_at: str

class AnalyticsResponse(BaseModel):
    motion_type: str
    total_motions: int
    success_rate: float
    by_judge: Dict[str, Dict[str, Any]]
    common_issues: Dict[str, int]
    top_cases: Dict[str, int]
    top_statutes: Dict[str, int]


# ============== Health & Status ==============

@app.get("/health")
async def health_check():
    """Check system health"""
    services = get_services()
    
    # Check vector DB
    try:
        stats = services["vector_db"].get_stats()
        vector_ok = stats["status"] == "green"
    except Exception:
        vector_ok = False
    
    # Check LLM
    try:
        from app.generation.llm_service import OllamaService
        llm = OllamaService()
        llm_ok = llm.check_health()
    except Exception:
        llm_ok = False
    
    return {
        "status": "healthy" if (vector_ok and llm_ok) else "degraded",
        "vector_db": "ok" if vector_ok else "error",
        "llm": "ok" if llm_ok else "error",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    services = get_services()
    
    vector_stats = services["vector_db"].get_stats()
    motions = services["metadata"].list_motions()
    
    motion_types = {}
    outcomes = {}
    
    for m in motions:
        motion_types[m.motion_type] = motion_types.get(m.motion_type, 0) + 1
        if m.outcome:
            outcomes[m.outcome.value] = outcomes.get(m.outcome.value, 0) + 1
    
    return {
        "total_vectors": vector_stats["total_vectors"],
        "total_motions": len(motions),
        "motion_types": motion_types,
        "outcomes": outcomes
    }


# ============== Ingestion ==============

@app.post("/ingest", response_model=IngestResponse)
async def ingest_motion(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest a motion document"""
    services = get_services()
    
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    try:
        # Process document
        metadata, parent_chunks, child_chunks = services["ingestion"].process_file(file_path)
        
        # Add manual metadata
        if request.outcome:
            metadata.outcome = MotionOutcome(request.outcome)
        if request.judge:
            metadata.judge = request.judge
        if request.charge_type:
            metadata.charge_type = request.charge_type
        if request.court:
            metadata.court = request.court
        
        # Embed chunks
        all_chunks = parent_chunks + child_chunks
        services["embedding"].embed_chunks(all_chunks)
        
        # Store in vector DB
        services["vector_db"].upsert_chunks(all_chunks, metadata)
        
        # Save metadata
        services["metadata"].save_metadata(metadata)
        
        return IngestResponse(
            motion_id=metadata.motion_id,
            title=metadata.title,
            motion_type=metadata.motion_type,
            chunk_count=len(all_chunks),
            message="Motion ingested successfully"
        )
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    outcome: Optional[str] = None,
    judge: Optional[str] = None,
    charge_type: Optional[str] = None
):
    """Ingest an uploaded motion file"""
    services = get_services()
    
    # Save uploaded file
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Use existing ingest logic
    return await ingest_motion(IngestRequest(
        file_path=str(file_path),
        outcome=outcome,
        judge=judge,
        charge_type=charge_type
    ), BackgroundTasks())

@app.post("/ingest/batch")
async def ingest_batch(directory: str, background_tasks: BackgroundTasks):
    """Ingest all motions in a directory"""
    services = get_services()
    
    dir_path = Path(directory)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
    
    results = services["ingestion"].process_directory(dir_path)
    
    ingested = 0
    for metadata, parent_chunks, child_chunks in results:
        try:
            all_chunks = parent_chunks + child_chunks
            services["embedding"].embed_chunks(all_chunks)
            services["vector_db"].upsert_chunks(all_chunks, metadata)
            services["metadata"].save_metadata(metadata)
            ingested += 1
        except Exception as e:
            logger.error(f"Failed to ingest {metadata.title}: {e}")
    
    return {
        "processed": len(results),
        "ingested": ingested,
        "failed": len(results) - ingested
    }


# ============== Search ==============

@app.post("/search", response_model=SearchResponse)
async def search_motions(request: SearchRequest):
    """Search for relevant motions"""
    services = get_services()
    
    filters = {}
    if request.motion_type:
        filters[settings.vector_db.motion_type_field] = request.motion_type
    if request.outcome:
        filters[settings.vector_db.outcome_field] = request.outcome
    if request.judge:
        filters[settings.vector_db.judge_field] = request.judge
    if request.charge_type:
        filters[settings.vector_db.charge_type_field] = request.charge_type
    
    context = services["retriever"].retrieve(
        query=request.query,
        filters=filters if filters else None,
        top_k=request.top_k
    )
    
    # Build results
    results = []
    for chunk, score in zip(context.chunks, context.relevance_scores):
        # Find matching metadata
        motion_meta = next(
            (m for m in context.metadata if m.motion_id == chunk.motion_id),
            None
        )
        
        results.append(SearchResult(
            motion_id=chunk.motion_id,
            title=motion_meta.title if motion_meta else "Unknown",
            motion_type=chunk.motion_type or "unknown",
            outcome=chunk.outcome,
            section=chunk.section,
            text_preview=chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
            relevance_score=score
        ))
    
    return SearchResponse(
        results=results,
        query=request.query,
        filters_applied=filters
    )

@app.get("/search/similar/{motion_id}")
async def find_similar(motion_id: str, top_k: int = 5):
    """Find motions similar to a given motion"""
    services = get_services()
    
    context = services["retriever"].find_similar_motions(motion_id, top_k=top_k)
    
    results = []
    for chunk, score in zip(context.chunks, context.relevance_scores):
        motion_meta = next(
            (m for m in context.metadata if m.motion_id == chunk.motion_id),
            None
        )
        
        results.append({
            "motion_id": chunk.motion_id,
            "title": motion_meta.title if motion_meta else "Unknown",
            "motion_type": chunk.motion_type,
            "outcome": chunk.outcome,
            "relevance_score": score
        })
    
    return {"similar_motions": results}


# ============== Drafting ==============

@app.post("/draft", response_model=DraftResponse)
async def generate_draft(request: DraftRequest):
    """Generate a motion draft"""
    services = get_services()
    
    try:
        draft = services["generator"].generate_draft(
            motion_type=request.motion_type,
            facts=request.facts,
            legal_issues=request.legal_issues
        )
        
        return DraftResponse(
            draft_id=draft.draft_id,
            motion_type=draft.motion_type,
            title=draft.title,
            content=draft.content,
            sections=draft.sections,
            cited_cases=draft.cited_cases,
            cited_statutes=draft.cited_statutes,
            source_motion_ids=draft.source_motion_ids
        )
    
    except Exception as e:
        logger.error(f"Draft generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/draft/{draft_id}/refine")
async def refine_draft(draft_id: str, feedback: str, additional_query: Optional[str] = None):
    """Refine an existing draft based on feedback"""
    # Note: Would need draft storage for full implementation
    raise HTTPException(status_code=501, detail="Draft refinement not yet implemented")


# ============== Chat Sessions ==============

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """Create a new strategy session"""
    services = get_services()
    
    session = services["chat"].create_session(
        motion_type=request.motion_type,
        charge_type=request.charge_type,
        key_facts=request.key_facts
    )
    
    return SessionResponse(
        session_id=session.session_id,
        motion_type=session.motion_type,
        charge_type=session.charge_type,
        created_at=session.created_at.isoformat()
    )

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    services = get_services()
    
    session = services["chat"].get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "motion_type": session.motion_type,
        "charge_type": session.charge_type,
        "key_facts": session.key_facts,
        "message_count": len(session.messages),
        "draft_ids": session.draft_ids,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat()
    }

@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, request: ChatRequest):
    """Send a message in a strategy session"""
    services = get_services()
    
    try:
        response = services["chat"].chat(
            session_id=session_id,
            message=request.message,
            use_retrieval=request.use_retrieval
        )
        
        return ChatResponse(
            message_id=response.message_id,
            content=response.content,
            sources_used=len(response.retrieved_context.chunks) if response.retrieved_context else 0,
            timestamp=response.timestamp.isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    """Get chat history for a session"""
    services = get_services()
    
    session = services["chat"].get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat()
            }
            for m in session.messages
        ]
    }

@app.post("/sessions/{session_id}/generate-draft")
async def generate_draft_from_session(session_id: str, additional_facts: Optional[str] = None):
    """Generate a draft based on session discussion"""
    services = get_services()
    
    try:
        draft = services["chat"].generate_draft_from_session(
            session_id=session_id,
            additional_facts=additional_facts
        )
        
        return DraftResponse(
            draft_id=draft.draft_id,
            motion_type=draft.motion_type,
            title=draft.title,
            content=draft.content,
            sections=draft.sections,
            cited_cases=draft.cited_cases,
            cited_statutes=draft.cited_statutes,
            source_motion_ids=draft.source_motion_ids
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============== Analytics ==============

@app.get("/analytics/{motion_type}", response_model=AnalyticsResponse)
async def get_analytics(motion_type: str):
    """Get analytics for a motion type"""
    services = get_services()
    
    motions = services["metadata"].list_motions(motion_type=motion_type)
    
    if not motions:
        raise HTTPException(status_code=404, detail="No motions found for this type")
    
    # Calculate success rate
    granted = sum(1 for m in motions if m.outcome in [MotionOutcome.GRANTED, MotionOutcome.GRANTED_IN_PART])
    denied = sum(1 for m in motions if m.outcome == MotionOutcome.DENIED)
    decided = granted + denied
    
    success_rate = granted / decided if decided > 0 else 0.0
    
    # Get judge analysis
    by_judge = services["analyzer"].get_success_rate_by_judge(motion_type)
    
    # Get common patterns
    common_issues = services["analyzer"].get_common_arguments(motion_type)
    top_cases, top_statutes = services["analyzer"].get_frequently_cited(motion_type)
    
    return AnalyticsResponse(
        motion_type=motion_type,
        total_motions=len(motions),
        success_rate=success_rate,
        by_judge=by_judge,
        common_issues=common_issues,
        top_cases=top_cases,
        top_statutes=top_statutes
    )

@app.get("/motion-types")
async def list_motion_types():
    """List available motion types"""
    return {
        "motion_types": [
            {"id": "motion_to_suppress", "name": "Motion to Suppress Evidence"},
            {"id": "motion_to_suppress_statements", "name": "Motion to Suppress Statements"},
            {"id": "motion_to_dismiss", "name": "Motion to Dismiss"},
            {"id": "motion_speedy_trial", "name": "Speedy Trial Demand"},
            {"id": "motion_discovery", "name": "Motion for Discovery"},
            {"id": "motion_bond_reduction", "name": "Motion for Bond Reduction"},
            {"id": "motion_continuance", "name": "Motion for Continuance"},
            {"id": "motion_to_sever", "name": "Motion to Sever"},
            {"id": "motion_change_venue", "name": "Motion for Change of Venue"},
            {"id": "richardson_motion", "name": "Richardson Motion (Discovery Violation)"},
            {"id": "nelson_motion", "name": "Nelson Motion (Self-Representation)"},
            {"id": "sentencing_memorandum", "name": "Sentencing Memorandum"},
        ]
    }


# ============== Metadata Management ==============

@app.get("/motions/{motion_id}")
async def get_motion(motion_id: str):
    """Get motion metadata"""
    services = get_services()
    
    metadata = services["metadata"].get_metadata(motion_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Motion not found")
    
    return metadata.model_dump()

@app.patch("/motions/{motion_id}/outcome")
async def update_outcome(motion_id: str, outcome: str, notes: Optional[str] = None):
    """Update motion outcome"""
    services = get_services()
    
    services["metadata"].update_outcome(motion_id, outcome, notes)
    
    return {"message": "Outcome updated", "motion_id": motion_id, "outcome": outcome}

@app.delete("/motions/{motion_id}")
async def delete_motion(motion_id: str):
    """Delete a motion and its vectors"""
    services = get_services()
    
    services["vector_db"].delete_motion(motion_id)
    
    return {"message": "Motion deleted", "motion_id": motion_id}


# ============== Run ==============

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
