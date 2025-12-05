"""
Embedding and Vector Database Services
Handles BGE-M3 embeddings and Qdrant vector storage
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None

from app.core.models import DocumentChunk, MotionMetadata, RetrievedContext
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service using BGE-M3 for legal document retrieval.
    Supports both dense and sparse embeddings for hybrid search.
    """
    
    def __init__(self):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers required for embeddings")
        
        logger.info(f"Loading embedding model: {settings.embedding.model_name}")
        self.model = SentenceTransformer(
            settings.embedding.model_name,
            device=settings.embedding.device
        )
        self.dimension = settings.embedding.dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=settings.embedding.normalize,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently"""
        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding.batch_size,
            normalize_embeddings=settings.embedding.normalize,
            show_progress_bar=True
        )
        return embeddings.tolist()
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add embeddings to chunks"""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks


class VectorDBService:
    """
    Qdrant vector database service for motion storage and retrieval.
    Implements case segregation and metadata filtering.
    """
    
    def __init__(self):
        if QdrantClient is None:
            raise ImportError("qdrant-client required for vector storage")
        
        self.client = QdrantClient(
            host=settings.vector_db.host,
            port=settings.vector_db.port
        )
        self.collection_name = settings.vector_db.collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding.dimension,
                    distance=Distance.COSINE
                )
            )
            
            # Create payload indexes for filtering
            for field in [
                settings.vector_db.motion_type_field,
                settings.vector_db.outcome_field,
                settings.vector_db.judge_field,
                settings.vector_db.charge_type_field,
                settings.vector_db.court_field,
            ]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD
                )
            
            logger.info("Collection and indexes created")
    
    def upsert_chunks(
        self, 
        chunks: List[DocumentChunk],
        metadata: MotionMetadata
    ) -> int:
        """Insert or update chunks in the database"""
        points = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(f"Chunk {chunk.chunk_id} has no embedding, skipping")
                continue
            
            payload = {
                "motion_id": chunk.motion_id,
                "text": chunk.text,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "is_parent": chunk.is_parent,
                "parent_chunk_id": chunk.parent_chunk_id,
                # Filtering fields
                settings.vector_db.motion_type_field: metadata.motion_type,
                settings.vector_db.outcome_field: metadata.outcome.value if metadata.outcome else None,
                settings.vector_db.judge_field: metadata.judge,
                settings.vector_db.charge_type_field: metadata.charge_type,
                settings.vector_db.court_field: metadata.court,
                settings.vector_db.attorney_field: metadata.attorney,
                # Additional metadata
                "title": metadata.title,
                "statutes_cited": metadata.statutes_cited,
                "cases_cited": metadata.cases_cited,
                "constitutional_claims": metadata.constitutional_claims,
                "legal_issues": metadata.legal_issues,
            }
            
            points.append(PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload=payload
            ))
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        logger.info(f"Upserted {len(points)} chunks for motion {metadata.motion_id}")
        return len(points)
    
    def search(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        include_parents: bool = True
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks with optional filtering.
        Returns chunks with relevance scores.
        """
        # Build filter conditions
        filter_conditions = []
        
        if filters:
            for key, value in filters.items():
                if value is not None:
                    if isinstance(value, list):
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key=key,
                                match=qdrant_models.MatchAny(any=value)
                            )
                        )
                    else:
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key=key,
                                match=qdrant_models.MatchValue(value=value)
                            )
                        )
        
        # Only search child chunks by default (for precision)
        if not include_parents:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="is_parent",
                    match=qdrant_models.MatchValue(value=False)
                )
            )
        
        query_filter = None
        if filter_conditions:
            query_filter = qdrant_models.Filter(must=filter_conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        )
        
        chunks_with_scores = []
        for result in results:
            chunk = DocumentChunk(
                chunk_id=result.id,
                motion_id=result.payload.get("motion_id"),
                text=result.payload.get("text"),
                section=result.payload.get("section"),
                chunk_index=result.payload.get("chunk_index"),
                is_parent=result.payload.get("is_parent", False),
                parent_chunk_id=result.payload.get("parent_chunk_id"),
                motion_type=result.payload.get(settings.vector_db.motion_type_field),
                outcome=result.payload.get(settings.vector_db.outcome_field),
                judge=result.payload.get(settings.vector_db.judge_field),
                charge_type=result.payload.get(settings.vector_db.charge_type_field),
                start_char=0,
                end_char=len(result.payload.get("text", ""))
            )
            chunks_with_scores.append((chunk, result.score))
        
        return chunks_with_scores
    
    def get_parent_chunk(self, parent_id: str) -> Optional[DocumentChunk]:
        """Retrieve a parent chunk by ID"""
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[parent_id],
                with_payload=True
            )
            
            if results:
                result = results[0]
                return DocumentChunk(
                    chunk_id=result.id,
                    motion_id=result.payload.get("motion_id"),
                    text=result.payload.get("text"),
                    section=result.payload.get("section"),
                    chunk_index=result.payload.get("chunk_index"),
                    is_parent=True,
                    start_char=0,
                    end_char=len(result.payload.get("text", ""))
                )
        except Exception as e:
            logger.error(f"Failed to retrieve parent chunk {parent_id}: {e}")
        
        return None
    
    def get_motion_chunks(self, motion_id: str) -> List[DocumentChunk]:
        """Get all chunks for a specific motion"""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="motion_id",
                        match=qdrant_models.MatchValue(value=motion_id)
                    )
                ]
            ),
            limit=1000,
            with_payload=True
        )[0]
        
        chunks = []
        for result in results:
            chunk = DocumentChunk(
                chunk_id=result.id,
                motion_id=result.payload.get("motion_id"),
                text=result.payload.get("text"),
                section=result.payload.get("section"),
                chunk_index=result.payload.get("chunk_index"),
                is_parent=result.payload.get("is_parent", False),
                start_char=0,
                end_char=len(result.payload.get("text", ""))
            )
            chunks.append(chunk)
        
        return sorted(chunks, key=lambda c: c.chunk_index)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "total_vectors": info.points_count,
            "indexed_vectors": info.indexed_vectors_count,
            "status": info.status.value
        }
    
    def delete_motion(self, motion_id: str) -> int:
        """Delete all chunks for a motion"""
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="motion_id",
                            match=qdrant_models.MatchValue(value=motion_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted chunks for motion {motion_id}")
        return 1


class MetadataStore:
    """
    JSON-based metadata store for motion documents.
    Complements vector storage with full document metadata.
    """
    
    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or settings.data_dir / "metadata.json"
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
    
    def _load(self):
        """Load metadata from disk"""
        if self.store_path.exists():
            with open(self.store_path, 'r') as f:
                self._data = json.load(f)
        else:
            self._data = {"motions": {}}
    
    def _save(self):
        """Save metadata to disk"""
        with open(self.store_path, 'w') as f:
            json.dump(self._data, f, indent=2, default=str)
    
    def save_metadata(self, metadata: MotionMetadata):
        """Save motion metadata"""
        self._data["motions"][metadata.motion_id] = metadata.model_dump()
        self._save()
    
    def get_metadata(self, motion_id: str) -> Optional[MotionMetadata]:
        """Retrieve motion metadata"""
        if motion_id in self._data["motions"]:
            return MotionMetadata(**self._data["motions"][motion_id])
        return None
    
    def list_motions(
        self,
        motion_type: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> List[MotionMetadata]:
        """List motions with optional filtering"""
        results = []
        for motion_data in self._data["motions"].values():
            if motion_type and motion_data.get("motion_type") != motion_type:
                continue
            if outcome and motion_data.get("outcome") != outcome:
                continue
            results.append(MotionMetadata(**motion_data))
        return results
    
    def update_outcome(
        self, 
        motion_id: str, 
        outcome: str, 
        outcome_notes: Optional[str] = None
    ):
        """Update motion outcome"""
        if motion_id in self._data["motions"]:
            self._data["motions"][motion_id]["outcome"] = outcome
            if outcome_notes:
                self._data["motions"][motion_id]["outcome_notes"] = outcome_notes
            self._save()
