"""
Motion RAG Configuration
Aligned with local-first architecture for attorney-client privilege protection
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional
import os

class EmbeddingSettings(BaseSettings):
    """Embedding model configuration - BGE-M3 for legal document retrieval"""
    model_name: str = "BAAI/bge-m3"
    dimension: int = 1024
    max_length: int = 8192  # Full legal document sections
    batch_size: int = 32
    device: str = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
    normalize: bool = True

class VectorDBSettings(BaseSettings):
    """Qdrant configuration for motion vectors with case segregation"""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "motions"
    # Metadata fields for filtering
    motion_type_field: str = "motion_type"
    outcome_field: str = "outcome"
    judge_field: str = "judge"
    charge_type_field: str = "charge_type"
    court_field: str = "court"
    date_field: str = "filing_date"
    attorney_field: str = "attorney"

class LLMSettings(BaseSettings):
    """Local LLM configuration via Ollama"""
    base_url: str = "http://localhost:11434"
    model: str = "phi3:14b"  # Good balance for legal reasoning
    fallback_model: str = "mistral:7b"
    temperature: float = 0.3  # Lower for legal precision
    max_tokens: int = 4096
    context_window: int = 8192
    
class ChunkingSettings(BaseSettings):
    """Motion-specific chunking for parent-child retrieval"""
    # Child chunks for precise retrieval
    child_chunk_size: int = 400
    child_overlap: int = 50
    # Parent chunks for generation context
    parent_chunk_size: int = 1600
    parent_overlap: int = 200
    # Motion section boundaries
    section_headers: list = Field(default=[
        "MOTION", "MEMORANDUM", "STATEMENT OF FACTS", "LEGAL ARGUMENT",
        "ARGUMENT", "CONCLUSION", "CERTIFICATE OF SERVICE", "WHEREFORE",
        "COMES NOW", "INTRODUCTION", "BACKGROUND", "STANDARD OF REVIEW",
        "DISCUSSION", "RELIEF REQUESTED", "PRAYER FOR RELIEF"
    ])

class RetrievalSettings(BaseSettings):
    """Hybrid retrieval configuration"""
    top_k_semantic: int = 20
    top_k_keyword: int = 10
    top_k_final: int = 8
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    min_relevance_score: float = 0.5

class MotionTypeEnum:
    """Standard Florida criminal motion types"""
    SUPPRESS_EVIDENCE = "motion_to_suppress"
    SUPPRESS_STATEMENTS = "motion_to_suppress_statements"
    DISMISS = "motion_to_dismiss"
    SPEEDY_TRIAL = "motion_speedy_trial"
    DISCOVERY = "motion_discovery"
    BOND_REDUCTION = "motion_bond_reduction"
    CONTINUANCE = "motion_continuance"
    SEVER = "motion_to_sever"
    CHANGE_VENUE = "motion_change_venue"
    RECONSIDER = "motion_reconsider"
    WITHDRAW_PLEA = "motion_withdraw_plea"
    COMPETENCY = "motion_competency"
    RICHARDSON = "richardson_motion"  # FL discovery violation
    NELSON = "nelson_motion"  # Self-representation inquiry
    FARETTA = "faretta_motion"  # Pro se request
    SENTENCING = "sentencing_memorandum"
    OTHER = "other"

class Settings(BaseSettings):
    """Main application settings"""
    app_name: str = "Motion RAG Assistant"
    debug: bool = False
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    motions_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "motions")
    
    # Component settings
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    
    # Security
    require_case_filter: bool = True  # Enforce case segregation
    audit_all_queries: bool = True
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

settings = Settings()
