"""
Core data models for motion documents and metadata
"""
from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid

class MotionOutcome(str, Enum):
    """Possible motion outcomes for tracking success patterns"""
    GRANTED = "granted"
    DENIED = "denied"
    GRANTED_IN_PART = "granted_in_part"
    MOOT = "moot"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    UNKNOWN = "unknown"

class CourtLevel(str, Enum):
    """Florida court hierarchy"""
    COUNTY = "county"
    CIRCUIT = "circuit"
    DCA = "district_court_appeal"
    SUPREME = "florida_supreme"

class MotionMetadata(BaseModel):
    """
    Comprehensive metadata for motion documents.
    Enables filtering, outcome analysis, and pattern recognition.
    """
    motion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Document identification
    title: str
    motion_type: str  # From MotionTypeEnum
    file_path: Optional[str] = None
    
    # Case information (anonymized for database storage)
    case_number_hash: Optional[str] = None  # Hashed for privacy
    charge_type: Optional[str] = None  # e.g., "burglary", "dui", "possession"
    charge_degree: Optional[str] = None  # e.g., "1st_felony", "misdemeanor"
    
    # Court information
    court: Optional[str] = None  # e.g., "15th Judicial Circuit"
    court_level: Optional[CourtLevel] = CourtLevel.CIRCUIT
    judge: Optional[str] = None
    
    # Outcome tracking
    outcome: Optional[MotionOutcome] = MotionOutcome.PENDING
    outcome_date: Optional[date] = None
    outcome_notes: Optional[str] = None
    
    # Filing information
    filing_date: Optional[date] = None
    attorney: Optional[str] = None
    
    # Content analysis (populated during ingestion)
    legal_issues: List[str] = Field(default_factory=list)
    statutes_cited: List[str] = Field(default_factory=list)
    cases_cited: List[str] = Field(default_factory=list)
    constitutional_claims: List[str] = Field(default_factory=list)
    
    # Processing metadata
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    has_exhibits: bool = False
    
    # Quality indicators
    success_score: Optional[float] = None  # 0-1 based on similar motion outcomes
    
    class Config:
        use_enum_values = True

class DocumentChunk(BaseModel):
    """Individual chunk from a motion document"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    motion_id: str
    
    # Content
    text: str
    section: Optional[str] = None  # e.g., "LEGAL ARGUMENT", "STATEMENT OF FACTS"
    
    # Position
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    
    # Parent-child relationship
    parent_chunk_id: Optional[str] = None
    is_parent: bool = False
    
    # Embedding (populated during indexing)
    embedding: Optional[List[float]] = None
    
    # Metadata copy for filtering
    motion_type: Optional[str] = None
    outcome: Optional[str] = None
    judge: Optional[str] = None
    charge_type: Optional[str] = None

class RetrievedContext(BaseModel):
    """Context retrieved for generation"""
    chunks: List[DocumentChunk]
    metadata: List[MotionMetadata]
    relevance_scores: List[float]
    query: str
    filters_applied: Dict[str, Any]

class MotionDraft(BaseModel):
    """Generated motion draft"""
    draft_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    motion_type: str
    
    # Generated content
    title: str
    content: str
    sections: Dict[str, str] = Field(default_factory=dict)
    
    # Sources used
    source_motion_ids: List[str] = Field(default_factory=list)
    cited_cases: List[str] = Field(default_factory=list)
    cited_statutes: List[str] = Field(default_factory=list)
    
    # Generation metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    prompt_used: Optional[str] = None
    model_used: Optional[str] = None
    
    # Review status
    reviewed: bool = False
    reviewer_notes: Optional[str] = None

class ChatMessage(BaseModel):
    """Chat message for the strategy interface"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context used for assistant responses
    retrieved_context: Optional[RetrievedContext] = None
    
    # If this message generated a draft
    draft_id: Optional[str] = None

class StrategySession(BaseModel):
    """A strategy session with an attorney"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Session context
    motion_type: Optional[str] = None
    charge_type: Optional[str] = None
    key_facts: Optional[str] = None
    
    # Conversation history
    messages: List[ChatMessage] = Field(default_factory=list)
    
    # Generated artifacts
    draft_ids: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
