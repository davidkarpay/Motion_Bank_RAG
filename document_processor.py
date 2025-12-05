"""
Motion Document Processor
Handles ingestion of PDF and DOCX motion files with legal-specific parsing
"""
import re
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import logging

# Document parsing
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# OCR fallback
try:
    import pytesseract
    from PIL import Image
    import io
except ImportError:
    pytesseract = None

from app.core.models import MotionMetadata, DocumentChunk, MotionOutcome
from config.settings import settings

logger = logging.getLogger(__name__)

class MotionParser:
    """
    Parses motion documents with legal-domain awareness.
    Extracts structure, citations, and metadata.
    """
    
    # Florida-specific patterns
    FLORIDA_STATUTE_PATTERN = r'(?:§|Section|Fla\.?\s*Stat\.?)\s*(\d+\.\d+(?:\(\d+\))?)'
    FLORIDA_RULE_PATTERN = r'(?:Fla\.?\s*R\.?\s*(?:Crim\.?\s*P\.?|Cr\.?\s*P\.?))\s*(\d+\.\d+)'
    CASE_CITATION_PATTERN = r'(\d+)\s+(So\.?\s*(?:2d|3d)?|Fla\.?|F\.?\s*(?:2d|3d|Supp\.?\s*(?:2d|3d)?)?)\s+(\d+)'
    
    # Constitutional claims
    FOURTH_AMENDMENT_KEYWORDS = [
        "search and seizure", "unreasonable search", "warrant", "probable cause",
        "terry stop", "exclusionary rule", "fruit of the poisonous tree",
        "consent search", "automobile exception", "exigent circumstances"
    ]
    FIFTH_AMENDMENT_KEYWORDS = [
        "miranda", "self-incrimination", "right to remain silent",
        "custodial interrogation", "involuntary statement", "coercion"
    ]
    SIXTH_AMENDMENT_KEYWORDS = [
        "right to counsel", "effective assistance", "speedy trial",
        "confrontation", "compulsory process", "massiah", "strickland"
    ]
    
    # Motion section headers
    SECTION_PATTERNS = [
        (r'(?i)^(?:I+\.|[A-Z]\.|INTRODUCTION)', "INTRODUCTION"),
        (r'(?i)^(?:STATEMENT\s+OF\s+(?:THE\s+)?FACTS?|FACTUAL\s+BACKGROUND)', "STATEMENT OF FACTS"),
        (r'(?i)^(?:LEGAL\s+)?ARGUMENT', "LEGAL ARGUMENT"),
        (r'(?i)^(?:STANDARD\s+OF\s+REVIEW)', "STANDARD OF REVIEW"),
        (r'(?i)^(?:CONCLUSION|WHEREFORE)', "CONCLUSION"),
        (r'(?i)^(?:PRAYER\s+FOR\s+RELIEF|RELIEF\s+REQUESTED)', "RELIEF REQUESTED"),
        (r'(?i)^(?:CERTIFICATE\s+OF\s+SERVICE)', "CERTIFICATE OF SERVICE"),
    ]
    
    def __init__(self):
        self.current_section = "PREAMBLE"
    
    def extract_text_from_pdf(self, file_path: Path) -> Tuple[str, int]:
        """Extract text from PDF, with OCR fallback for scanned documents"""
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) required for PDF processing")
        
        doc = fitz.open(file_path)
        full_text = []
        page_count = len(doc)
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # If page has minimal text, try OCR
            if len(text.strip()) < 100 and pytesseract:
                logger.info(f"OCR fallback for page {page_num + 1}")
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))
                text = pytesseract.image_to_string(img)
            
            full_text.append(f"[PAGE {page_num + 1}]\n{text}")
        
        doc.close()
        return "\n\n".join(full_text), page_count
    
    def extract_text_from_docx(self, file_path: Path) -> Tuple[str, int]:
        """Extract text from DOCX preserving structure"""
        if DocxDocument is None:
            raise ImportError("python-docx required for DOCX processing")
        
        doc = DocxDocument(file_path)
        paragraphs = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                # Check if it's a heading
                if para.style and 'Heading' in para.style.name:
                    paragraphs.append(f"\n## {text}\n")
                else:
                    paragraphs.append(text)
        
        # Estimate page count (rough approximation)
        full_text = "\n".join(paragraphs)
        page_count = max(1, len(full_text) // 3000)
        
        return full_text, page_count
    
    def extract_metadata(self, text: str, file_path: Path) -> MotionMetadata:
        """Extract metadata from motion text"""
        
        # Detect motion type from title/content
        motion_type = self._detect_motion_type(text)
        
        # Extract title (usually first substantive line)
        title = self._extract_title(text)
        
        # Extract citations
        statutes = self._extract_statutes(text)
        cases = self._extract_case_citations(text)
        
        # Detect constitutional claims
        constitutional = self._detect_constitutional_claims(text)
        
        # Extract legal issues
        legal_issues = self._extract_legal_issues(text, motion_type)
        
        # Word count
        word_count = len(text.split())
        
        return MotionMetadata(
            title=title,
            motion_type=motion_type,
            file_path=str(file_path),
            statutes_cited=statutes,
            cases_cited=cases,
            constitutional_claims=constitutional,
            legal_issues=legal_issues,
            word_count=word_count,
            outcome=MotionOutcome.UNKNOWN
        )
    
    def _detect_motion_type(self, text: str) -> str:
        """Detect motion type from content"""
        text_lower = text.lower()
        
        type_patterns = [
            (r'motion\s+to\s+suppress\s+(?:evidence|physical)', "motion_to_suppress"),
            (r'motion\s+to\s+suppress\s+(?:statement|confession)', "motion_to_suppress_statements"),
            (r'motion\s+to\s+dismiss', "motion_to_dismiss"),
            (r'demand\s+for\s+speedy\s+trial|speedy\s+trial', "motion_speedy_trial"),
            (r'motion\s+(?:for|to)\s+(?:compel\s+)?discovery', "motion_discovery"),
            (r'motion\s+(?:for|to)\s+reduce\s+bond|bond\s+reduction', "motion_bond_reduction"),
            (r'motion\s+(?:for|to)\s+(?:a\s+)?continuance', "motion_continuance"),
            (r'motion\s+to\s+sever', "motion_to_sever"),
            (r'motion\s+(?:for|to)\s+change\s+(?:of\s+)?venue', "motion_change_venue"),
            (r'motion\s+(?:for|to)\s+reconsider', "motion_reconsider"),
            (r'motion\s+to\s+withdraw\s+plea', "motion_withdraw_plea"),
            (r'motion\s+(?:for|to)\s+(?:determine\s+)?competency', "motion_competency"),
            (r'richardson\s+(?:hearing|motion|inquiry)', "richardson_motion"),
            (r'nelson\s+(?:hearing|inquiry)', "nelson_motion"),
            (r'faretta\s+(?:hearing|inquiry|waiver)', "faretta_motion"),
            (r'sentencing\s+memorandum', "sentencing_memorandum"),
        ]
        
        for pattern, motion_type in type_patterns:
            if re.search(pattern, text_lower):
                return motion_type
        
        return "other"
    
    def _extract_title(self, text: str) -> str:
        """Extract motion title"""
        lines = text.split('\n')
        
        for line in lines[:30]:  # Check first 30 lines
            line = line.strip()
            if not line:
                continue
            
            # Look for motion title patterns
            if re.match(r'(?i)^(?:DEFENDANT\'?S?\s+)?MOTION', line):
                return line[:200]  # Truncate long titles
            if re.match(r'(?i)^MEMORANDUM\s+(?:OF\s+LAW\s+)?IN\s+SUPPORT', line):
                return line[:200]
        
        return "Untitled Motion"
    
    def _extract_statutes(self, text: str) -> List[str]:
        """Extract Florida statute citations"""
        statutes = set()
        
        # Florida Statutes
        for match in re.finditer(self.FLORIDA_STATUTE_PATTERN, text):
            statutes.add(f"Fla. Stat. § {match.group(1)}")
        
        # Florida Rules of Criminal Procedure
        for match in re.finditer(self.FLORIDA_RULE_PATTERN, text):
            statutes.add(f"Fla. R. Crim. P. {match.group(1)}")
        
        return sorted(list(statutes))
    
    def _extract_case_citations(self, text: str) -> List[str]:
        """Extract case citations"""
        cases = set()
        
        # Standard citation format
        for match in re.finditer(self.CASE_CITATION_PATTERN, text):
            citation = f"{match.group(1)} {match.group(2)} {match.group(3)}"
            cases.add(citation)
        
        # Named case patterns (e.g., "State v. Smith")
        named_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+v\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        for match in re.finditer(named_pattern, text):
            case_name = f"{match.group(1)} v. {match.group(2)}"
            if len(case_name) < 50:  # Sanity check
                cases.add(case_name)
        
        return sorted(list(cases))[:50]  # Limit to 50 citations
    
    def _detect_constitutional_claims(self, text: str) -> List[str]:
        """Detect constitutional claims in the motion"""
        text_lower = text.lower()
        claims = []
        
        if any(kw in text_lower for kw in self.FOURTH_AMENDMENT_KEYWORDS):
            claims.append("Fourth Amendment")
        if any(kw in text_lower for kw in self.FIFTH_AMENDMENT_KEYWORDS):
            claims.append("Fifth Amendment")
        if any(kw in text_lower for kw in self.SIXTH_AMENDMENT_KEYWORDS):
            claims.append("Sixth Amendment")
        if "due process" in text_lower:
            claims.append("Due Process")
        if "equal protection" in text_lower:
            claims.append("Equal Protection")
        
        return claims
    
    def _extract_legal_issues(self, text: str, motion_type: str) -> List[str]:
        """Extract key legal issues"""
        issues = []
        text_lower = text.lower()
        
        # Motion-type-specific issues
        if motion_type == "motion_to_suppress":
            if "warrantless" in text_lower:
                issues.append("warrantless search")
            if "consent" in text_lower:
                issues.append("consent validity")
            if "traffic stop" in text_lower or "vehicle" in text_lower:
                issues.append("vehicle search")
            if "terry" in text_lower or "stop and frisk" in text_lower:
                issues.append("Terry stop")
        
        elif motion_type == "motion_to_suppress_statements":
            if "miranda" in text_lower:
                issues.append("Miranda violation")
            if "coercion" in text_lower or "involuntary" in text_lower:
                issues.append("involuntary statement")
            if "custodial" in text_lower:
                issues.append("custodial interrogation")
        
        elif motion_type == "motion_to_dismiss":
            if "speedy trial" in text_lower:
                issues.append("speedy trial violation")
            if "statute of limitations" in text_lower:
                issues.append("statute of limitations")
            if "double jeopardy" in text_lower:
                issues.append("double jeopardy")
            if "insufficient" in text_lower and "evidence" in text_lower:
                issues.append("insufficient evidence")
        
        return issues
    
    def identify_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Identify section boundaries in motion text"""
        sections = []
        current_section = "PREAMBLE"
        current_start = 0
        
        lines = text.split('\n')
        char_pos = 0
        
        for line in lines:
            for pattern, section_name in self.SECTION_PATTERNS:
                if re.match(pattern, line.strip()):
                    # Save previous section
                    if char_pos > current_start:
                        sections.append((current_section, current_start, char_pos))
                    current_section = section_name
                    current_start = char_pos
                    break
            
            char_pos += len(line) + 1  # +1 for newline
        
        # Add final section
        sections.append((current_section, current_start, len(text)))
        
        return sections


class MotionChunker:
    """
    Creates parent-child chunks for motion documents.
    Respects section boundaries for semantic coherence.
    """
    
    def __init__(self):
        self.parser = MotionParser()
        self.child_size = settings.chunking.child_chunk_size
        self.child_overlap = settings.chunking.child_overlap
        self.parent_size = settings.chunking.parent_chunk_size
        self.parent_overlap = settings.chunking.parent_overlap
    
    def chunk_document(
        self, 
        text: str, 
        motion_id: str,
        metadata: MotionMetadata
    ) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
        """
        Create parent and child chunks from motion text.
        Returns (parent_chunks, child_chunks)
        """
        # Identify sections
        sections = self.parser.identify_sections(text)
        
        parent_chunks = []
        child_chunks = []
        parent_idx = 0
        child_idx = 0
        
        for section_name, start, end in sections:
            section_text = text[start:end]
            
            if len(section_text.strip()) < 50:
                continue
            
            # Create parent chunks for this section
            section_parents = self._create_chunks(
                section_text, 
                self.parent_size, 
                self.parent_overlap,
                start_offset=start
            )
            
            for parent_text, parent_start, parent_end in section_parents:
                parent_chunk = DocumentChunk(
                    motion_id=motion_id,
                    text=parent_text,
                    section=section_name,
                    chunk_index=parent_idx,
                    start_char=parent_start,
                    end_char=parent_end,
                    is_parent=True,
                    motion_type=metadata.motion_type,
                    outcome=metadata.outcome.value if metadata.outcome else None,
                    judge=metadata.judge,
                    charge_type=metadata.charge_type
                )
                parent_chunks.append(parent_chunk)
                
                # Create child chunks within this parent
                children = self._create_chunks(
                    parent_text,
                    self.child_size,
                    self.child_overlap,
                    start_offset=parent_start
                )
                
                for child_text, child_start, child_end in children:
                    child_chunk = DocumentChunk(
                        motion_id=motion_id,
                        text=child_text,
                        section=section_name,
                        chunk_index=child_idx,
                        start_char=child_start,
                        end_char=child_end,
                        is_parent=False,
                        parent_chunk_id=parent_chunk.chunk_id,
                        motion_type=metadata.motion_type,
                        outcome=metadata.outcome.value if metadata.outcome else None,
                        judge=metadata.judge,
                        charge_type=metadata.charge_type
                    )
                    child_chunks.append(child_chunk)
                    child_idx += 1
                
                parent_idx += 1
        
        return parent_chunks, child_chunks
    
    def _create_chunks(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int,
        start_offset: int = 0
    ) -> List[Tuple[str, int, int]]:
        """Create overlapping chunks from text"""
        chunks = []
        
        # Try to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        chunk_start = start_offset
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_end = chunk_start + len(chunk_text)
                chunks.append((chunk_text, chunk_start, chunk_end))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                
                current_chunk = overlap_sentences
                current_length = overlap_len
                chunk_start = chunk_end - overlap_len
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, chunk_start, chunk_start + len(chunk_text)))
        
        return chunks


class MotionIngestionPipeline:
    """
    Complete ingestion pipeline for motion documents.
    Processes files, extracts metadata, chunks content, and prepares for indexing.
    """
    
    def __init__(self):
        self.parser = MotionParser()
        self.chunker = MotionChunker()
    
    def process_file(self, file_path: Path) -> Tuple[MotionMetadata, List[DocumentChunk], List[DocumentChunk]]:
        """
        Process a single motion file.
        Returns (metadata, parent_chunks, child_chunks)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text based on file type
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            text, page_count = self.parser.extract_text_from_pdf(file_path)
        elif suffix in ['.docx', '.doc']:
            text, page_count = self.parser.extract_text_from_docx(file_path)
        elif suffix == '.txt':
            text = file_path.read_text(encoding='utf-8')
            page_count = max(1, len(text) // 3000)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # Extract metadata
        metadata = self.parser.extract_metadata(text, file_path)
        metadata.page_count = page_count
        
        # Create chunks
        parent_chunks, child_chunks = self.chunker.chunk_document(
            text, 
            metadata.motion_id,
            metadata
        )
        
        logger.info(
            f"Processed {file_path.name}: "
            f"{len(parent_chunks)} parents, {len(child_chunks)} children"
        )
        
        return metadata, parent_chunks, child_chunks
    
    def process_directory(self, dir_path: Path) -> List[Tuple[MotionMetadata, List[DocumentChunk], List[DocumentChunk]]]:
        """Process all motion files in a directory"""
        dir_path = Path(dir_path)
        results = []
        
        for file_path in dir_path.rglob('*'):
            if file_path.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt']:
                try:
                    result = self.process_file(file_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        return results
