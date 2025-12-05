"""
Hybrid Retrieval Pipeline
Combines semantic search with keyword matching and cross-encoder reranking
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from app.core.models import DocumentChunk, MotionMetadata, RetrievedContext
from app.retrieval.vector_store import EmbeddingService, VectorDBService, MetadataStore
from config.settings import settings

logger = logging.getLogger(__name__)


class KeywordSearcher:
    """
    BM25-based keyword search for legal terminology.
    Complements semantic search for precise legal terms and citations.
    """
    
    def __init__(self):
        self.corpus = []
        self.chunk_ids = []
        self.bm25 = None
        self._tokenize_pattern = re.compile(r'\w+')
    
    def build_index(self, chunks: List[DocumentChunk]):
        """Build BM25 index from chunks"""
        if BM25Okapi is None:
            logger.warning("rank_bm25 not installed, keyword search disabled")
            return
        
        self.corpus = []
        self.chunk_ids = []
        
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.corpus.append(tokens)
            self.chunk_ids.append(chunk.chunk_id)
        
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            logger.info(f"Built BM25 index with {len(self.corpus)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        return self._tokenize_pattern.findall(text.lower())
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for matching chunks, returns (chunk_id, score) pairs"""
        if self.bm25 is None:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k results
        indexed_scores = [(self.chunk_ids[i], scores[i]) for i in range(len(scores))]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:top_k]


class Reranker:
    """
    Cross-encoder reranker for improving retrieval precision.
    Evaluates query-document pairs for fine-grained relevance scoring.
    """
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model"""
        if CrossEncoder is None:
            logger.warning("CrossEncoder not available, reranking disabled")
            return
        
        try:
            self.model = CrossEncoder(
                settings.retrieval.rerank_model,
                max_length=512
            )
            logger.info(f"Loaded reranker: {settings.retrieval.rerank_model}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
    
    def rerank(
        self, 
        query: str, 
        chunks: List[DocumentChunk],
        top_k: Optional[int] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Rerank chunks based on query relevance"""
        if self.model is None or not chunks:
            return [(c, 1.0) for c in chunks]
        
        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.text) for chunk in chunks]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Combine chunks with scores
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            scored = scored[:top_k]
        
        return scored


class HybridRetriever:
    """
    Hybrid retrieval combining semantic search, keyword search, and reranking.
    Implements parent-child retrieval for precise matching with full context.
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBService()
        self.metadata_store = MetadataStore()
        self.keyword_searcher = KeywordSearcher()
        self.reranker = Reranker()
        
        self.semantic_weight = settings.retrieval.semantic_weight
        self.keyword_weight = settings.retrieval.keyword_weight
    
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        expand_to_parents: bool = True
    ) -> RetrievedContext:
        """
        Perform hybrid retrieval with optional parent expansion.
        
        Args:
            query: Search query
            filters: Metadata filters (motion_type, outcome, judge, etc.)
            top_k: Number of results to return
            expand_to_parents: Whether to include parent chunks for context
            
        Returns:
            RetrievedContext with chunks, metadata, and scores
        """
        top_k = top_k or settings.retrieval.top_k_final
        
        # 1. Semantic search on child chunks
        query_embedding = self.embedding_service.embed_text(query)
        semantic_results = self.vector_db.search(
            query_vector=query_embedding,
            filters=filters,
            top_k=settings.retrieval.top_k_semantic,
            include_parents=False  # Search children for precision
        )
        
        # 2. Keyword search (if index built)
        keyword_results = self.keyword_searcher.search(
            query, 
            top_k=settings.retrieval.top_k_keyword
        )
        keyword_scores = {chunk_id: score for chunk_id, score in keyword_results}
        
        # 3. Combine scores with weighting
        chunk_scores = {}
        chunk_map = {}
        
        for chunk, semantic_score in semantic_results:
            chunk_scores[chunk.chunk_id] = self.semantic_weight * semantic_score
            chunk_map[chunk.chunk_id] = chunk
        
        for chunk_id, keyword_score in keyword_results:
            if chunk_id in chunk_scores:
                # Normalize keyword score to 0-1 range
                normalized_kw = min(1.0, keyword_score / 10.0)
                chunk_scores[chunk_id] += self.keyword_weight * normalized_kw
            elif chunk_id in chunk_map:
                # Add new chunk from keyword search
                chunk_scores[chunk_id] = self.keyword_weight * min(1.0, keyword_score / 10.0)
        
        # 4. Get top candidates
        sorted_chunks = sorted(
            chunk_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:settings.retrieval.top_k_semantic]
        
        candidate_chunks = [chunk_map[chunk_id] for chunk_id, _ in sorted_chunks if chunk_id in chunk_map]
        
        # 5. Rerank for final precision
        reranked = self.reranker.rerank(query, candidate_chunks, top_k=top_k)
        
        # 6. Expand to parent chunks for generation context
        final_chunks = []
        final_scores = []
        seen_parents = set()
        
        for chunk, score in reranked:
            if score < settings.retrieval.min_relevance_score:
                continue
            
            if expand_to_parents and chunk.parent_chunk_id:
                if chunk.parent_chunk_id not in seen_parents:
                    parent = self.vector_db.get_parent_chunk(chunk.parent_chunk_id)
                    if parent:
                        final_chunks.append(parent)
                        final_scores.append(score)
                        seen_parents.add(chunk.parent_chunk_id)
            else:
                final_chunks.append(chunk)
                final_scores.append(score)
        
        # 7. Collect metadata for retrieved motions
        motion_ids = list(set(c.motion_id for c in final_chunks))
        metadata_list = []
        for motion_id in motion_ids:
            meta = self.metadata_store.get_metadata(motion_id)
            if meta:
                metadata_list.append(meta)
        
        return RetrievedContext(
            chunks=final_chunks,
            metadata=metadata_list,
            relevance_scores=final_scores,
            query=query,
            filters_applied=filters or {}
        )
    
    def retrieve_by_motion_type(
        self,
        query: str,
        motion_type: str,
        outcome: Optional[str] = None,
        top_k: int = 5
    ) -> RetrievedContext:
        """Convenience method for motion-type filtered retrieval"""
        filters = {
            settings.vector_db.motion_type_field: motion_type
        }
        if outcome:
            filters[settings.vector_db.outcome_field] = outcome
        
        return self.retrieve(query, filters=filters, top_k=top_k)
    
    def retrieve_successful_motions(
        self,
        query: str,
        motion_type: Optional[str] = None,
        top_k: int = 5
    ) -> RetrievedContext:
        """Retrieve only granted/successful motions"""
        filters = {
            settings.vector_db.outcome_field: ["granted", "granted_in_part"]
        }
        if motion_type:
            filters[settings.vector_db.motion_type_field] = motion_type
        
        return self.retrieve(query, filters=filters, top_k=top_k)
    
    def find_similar_motions(
        self,
        motion_id: str,
        top_k: int = 5
    ) -> RetrievedContext:
        """Find motions similar to a given motion"""
        # Get the motion's chunks
        chunks = self.vector_db.get_motion_chunks(motion_id)
        
        if not chunks:
            return RetrievedContext(
                chunks=[],
                metadata=[],
                relevance_scores=[],
                query=f"similar to {motion_id}",
                filters_applied={}
            )
        
        # Use the legal argument section for similarity
        legal_chunks = [c for c in chunks if c.section == "LEGAL ARGUMENT"]
        if not legal_chunks:
            legal_chunks = chunks[:3]
        
        # Combine text for query
        query_text = " ".join(c.text[:500] for c in legal_chunks[:3])
        
        # Search, excluding the original motion
        query_embedding = self.embedding_service.embed_text(query_text)
        results = self.vector_db.search(
            query_vector=query_embedding,
            top_k=top_k * 3  # Get more to filter
        )
        
        # Filter out chunks from the original motion
        filtered = [(c, s) for c, s in results if c.motion_id != motion_id]
        
        # Deduplicate by motion
        seen_motions = set()
        final_chunks = []
        final_scores = []
        
        for chunk, score in filtered:
            if chunk.motion_id not in seen_motions and len(final_chunks) < top_k:
                final_chunks.append(chunk)
                final_scores.append(score)
                seen_motions.add(chunk.motion_id)
        
        # Get metadata
        metadata_list = []
        for motion_id in seen_motions:
            meta = self.metadata_store.get_metadata(motion_id)
            if meta:
                metadata_list.append(meta)
        
        return RetrievedContext(
            chunks=final_chunks,
            metadata=metadata_list,
            relevance_scores=final_scores,
            query=f"similar to {motion_id}",
            filters_applied={}
        )


class MotionAnalyzer:
    """
    Analyzes patterns across motions for strategic insights.
    """
    
    def __init__(self):
        self.metadata_store = MetadataStore()
    
    def get_success_rate_by_judge(
        self, 
        motion_type: str,
        min_motions: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze success rates by judge for a motion type"""
        motions = self.metadata_store.list_motions(motion_type=motion_type)
        
        judge_stats = defaultdict(lambda: {"granted": 0, "denied": 0, "total": 0})
        
        for motion in motions:
            if motion.judge and motion.outcome:
                judge_stats[motion.judge]["total"] += 1
                if motion.outcome in ["granted", "granted_in_part"]:
                    judge_stats[motion.judge]["granted"] += 1
                elif motion.outcome == "denied":
                    judge_stats[motion.judge]["denied"] += 1
        
        # Calculate rates and filter
        results = {}
        for judge, stats in judge_stats.items():
            if stats["total"] >= min_motions:
                results[judge] = {
                    **stats,
                    "success_rate": stats["granted"] / stats["total"] if stats["total"] > 0 else 0
                }
        
        return dict(sorted(results.items(), key=lambda x: x[1]["success_rate"], reverse=True))
    
    def get_common_arguments(
        self, 
        motion_type: str,
        outcome: Optional[str] = "granted"
    ) -> Dict[str, int]:
        """Find common legal issues in successful motions"""
        motions = self.metadata_store.list_motions(motion_type=motion_type, outcome=outcome)
        
        issue_counts = defaultdict(int)
        for motion in motions:
            for issue in motion.legal_issues:
                issue_counts[issue] += 1
        
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_frequently_cited(
        self, 
        motion_type: str,
        outcome: Optional[str] = "granted"
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Get frequently cited cases and statutes in successful motions"""
        motions = self.metadata_store.list_motions(motion_type=motion_type, outcome=outcome)
        
        case_counts = defaultdict(int)
        statute_counts = defaultdict(int)
        
        for motion in motions:
            for case in motion.cases_cited:
                case_counts[case] += 1
            for statute in motion.statutes_cited:
                statute_counts[statute] += 1
        
        top_cases = dict(sorted(case_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        top_statutes = dict(sorted(statute_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        
        return top_cases, top_statutes
