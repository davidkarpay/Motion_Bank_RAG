"""
LLM Service and Motion Generation Pipeline
Handles local LLM inference and motion drafting with retrieved context
"""
import logging
from typing import List, Optional, Dict, Any, Generator
from datetime import datetime
import json
import re

try:
    import httpx
except ImportError:
    httpx = None

from app.core.models import (
    DocumentChunk, MotionMetadata, RetrievedContext, 
    MotionDraft, ChatMessage, StrategySession
)
from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """
    Local LLM service via Ollama.
    Supports streaming responses and context management.
    """
    
    def __init__(self):
        if httpx is None:
            raise ImportError("httpx required for Ollama communication")
        
        self.base_url = settings.llm.base_url
        self.model = settings.llm.model
        self.client = httpx.Client(timeout=120.0)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response from the LLM"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or settings.llm.temperature,
                    "num_predict": max_tokens or settings.llm.max_tokens
                }
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama error: {response.text}")
            raise Exception(f"LLM request failed: {response.status_code}")
        
        return response.json()["message"]["content"]
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Stream response tokens"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        with self.client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature or settings.llm.temperature
                }
            }
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
    
    def check_health(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                return any(self.model in name for name in model_names)
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
        return False


class PromptBuilder:
    """
    Builds prompts for motion strategy and drafting.
    Incorporates retrieved context and legal formatting requirements.
    """
    
    SYSTEM_PROMPT = """You are an experienced criminal defense attorney assistant specializing in Florida criminal law. You help attorneys analyze cases, develop motion strategies, and draft legal documents.

Your responses should:
1. Be grounded in the provided context from successful motions
2. Cite specific cases and statutes where relevant
3. Use proper legal formatting and terminology
4. Identify potential weaknesses and counterarguments
5. Be practical and actionable

Always indicate when you're drawing from the provided examples vs. general legal knowledge. Never fabricate case citations."""

    STRATEGY_PROMPT_TEMPLATE = """Based on the following context from similar motions, help the attorney with their question.

## Retrieved Motion Context:
{context}

## Motion Metadata:
{metadata}

## Attorney's Question:
{question}

Provide strategic analysis and recommendations. If drafting language is requested, base it on successful examples from the context."""

    DRAFT_PROMPT_TEMPLATE = """Draft a {motion_type} motion based on the following:

## Case Facts:
{facts}

## Legal Issues:
{legal_issues}

## Successful Motion Examples:
{examples}

## Frequently Cited Authorities:
Cases: {top_cases}
Statutes: {top_statutes}

Draft a complete motion following Florida criminal procedure format:
1. Caption and title
2. Introduction/Statement of Facts
3. Legal Argument (with proper citations)
4. Conclusion/Prayer for Relief
5. Certificate of Service placeholder

Base your arguments on the successful examples provided. Use proper legal citation format."""

    REFINEMENT_PROMPT_TEMPLATE = """Refine the following motion draft based on the attorney's feedback:

## Current Draft:
{current_draft}

## Attorney's Feedback:
{feedback}

## Additional Context (if any):
{additional_context}

Revise the draft incorporating the feedback while maintaining proper legal format and citation style."""

    def build_strategy_prompt(
        self,
        question: str,
        context: RetrievedContext
    ) -> tuple[str, str]:
        """Build prompt for strategy discussion"""
        # Format context chunks
        context_text = self._format_chunks(context.chunks)
        
        # Format metadata
        metadata_text = self._format_metadata(context.metadata)
        
        prompt = self.STRATEGY_PROMPT_TEMPLATE.format(
            context=context_text,
            metadata=metadata_text,
            question=question
        )
        
        return self.SYSTEM_PROMPT, prompt
    
    def build_draft_prompt(
        self,
        motion_type: str,
        facts: str,
        legal_issues: List[str],
        context: RetrievedContext,
        top_cases: Dict[str, int],
        top_statutes: Dict[str, int]
    ) -> tuple[str, str]:
        """Build prompt for motion drafting"""
        # Format examples
        examples_text = self._format_chunks(context.chunks)
        
        # Format authorities
        cases_text = ", ".join(f"{case} (cited {count}x)" for case, count in list(top_cases.items())[:10])
        statutes_text = ", ".join(f"{stat} (cited {count}x)" for stat, count in list(top_statutes.items())[:10])
        
        prompt = self.DRAFT_PROMPT_TEMPLATE.format(
            motion_type=motion_type.replace("_", " ").title(),
            facts=facts,
            legal_issues=", ".join(legal_issues),
            examples=examples_text,
            top_cases=cases_text or "None available",
            top_statutes=statutes_text or "None available"
        )
        
        return self.SYSTEM_PROMPT, prompt
    
    def build_refinement_prompt(
        self,
        current_draft: str,
        feedback: str,
        additional_context: Optional[RetrievedContext] = None
    ) -> tuple[str, str]:
        """Build prompt for draft refinement"""
        context_text = ""
        if additional_context:
            context_text = self._format_chunks(additional_context.chunks)
        
        prompt = self.REFINEMENT_PROMPT_TEMPLATE.format(
            current_draft=current_draft,
            feedback=feedback,
            additional_context=context_text or "None"
        )
        
        return self.SYSTEM_PROMPT, prompt
    
    def _format_chunks(self, chunks: List[DocumentChunk]) -> str:
        """Format chunks for prompt inclusion"""
        formatted = []
        
        for i, chunk in enumerate(chunks, 1):
            section = chunk.section or "General"
            formatted.append(f"### Example {i} ({section}):\n{chunk.text}\n")
        
        return "\n".join(formatted) if formatted else "No relevant examples found."
    
    def _format_metadata(self, metadata: List[MotionMetadata]) -> str:
        """Format metadata for prompt inclusion"""
        formatted = []
        
        for meta in metadata[:5]:  # Limit to 5
            outcome = meta.outcome.value if meta.outcome else "unknown"
            formatted.append(
                f"- {meta.title}: Type={meta.motion_type}, "
                f"Outcome={outcome}, Judge={meta.judge or 'N/A'}"
            )
        
        return "\n".join(formatted) if formatted else "No metadata available."


class MotionGenerator:
    """
    Complete motion generation pipeline.
    Retrieves context, builds prompts, and generates drafts.
    """
    
    def __init__(self, retriever, analyzer):
        self.llm = OllamaService()
        self.prompt_builder = PromptBuilder()
        self.retriever = retriever
        self.analyzer = analyzer
    
    def generate_draft(
        self,
        motion_type: str,
        facts: str,
        legal_issues: Optional[List[str]] = None
    ) -> MotionDraft:
        """Generate a complete motion draft"""
        legal_issues = legal_issues or []
        
        # Retrieve successful examples
        query = f"{motion_type} {' '.join(legal_issues)} {facts[:200]}"
        context = self.retriever.retrieve_successful_motions(
            query=query,
            motion_type=motion_type,
            top_k=5
        )
        
        # Get common citations
        top_cases, top_statutes = self.analyzer.get_frequently_cited(
            motion_type=motion_type,
            outcome="granted"
        )
        
        # Build and execute prompt
        system_prompt, user_prompt = self.prompt_builder.build_draft_prompt(
            motion_type=motion_type,
            facts=facts,
            legal_issues=legal_issues,
            context=context,
            top_cases=top_cases,
            top_statutes=top_statutes
        )
        
        content = self.llm.generate(user_prompt, system_prompt=system_prompt)
        
        # Parse sections from generated content
        sections = self._parse_sections(content)
        
        return MotionDraft(
            motion_type=motion_type,
            title=f"Motion to {motion_type.replace('_', ' ').title()}",
            content=content,
            sections=sections,
            source_motion_ids=[m.motion_id for m in context.metadata],
            cited_cases=list(top_cases.keys())[:10],
            cited_statutes=list(top_statutes.keys())[:10],
            prompt_used=user_prompt,
            model_used=settings.llm.model
        )
    
    def refine_draft(
        self,
        draft: MotionDraft,
        feedback: str,
        additional_query: Optional[str] = None
    ) -> MotionDraft:
        """Refine a draft based on attorney feedback"""
        additional_context = None
        
        if additional_query:
            additional_context = self.retriever.retrieve(
                query=additional_query,
                top_k=3
            )
        
        system_prompt, user_prompt = self.prompt_builder.build_refinement_prompt(
            current_draft=draft.content,
            feedback=feedback,
            additional_context=additional_context
        )
        
        refined_content = self.llm.generate(user_prompt, system_prompt=system_prompt)
        
        return MotionDraft(
            motion_type=draft.motion_type,
            title=draft.title,
            content=refined_content,
            sections=self._parse_sections(refined_content),
            source_motion_ids=draft.source_motion_ids,
            cited_cases=draft.cited_cases,
            cited_statutes=draft.cited_statutes,
            prompt_used=user_prompt,
            model_used=settings.llm.model
        )
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """Parse motion content into sections"""
        sections = {}
        current_section = "HEADER"
        current_content = []
        
        section_patterns = [
            (r'(?i)^#+\s*(INTRODUCTION|STATEMENT\s+OF\s+FACTS)', "STATEMENT OF FACTS"),
            (r'(?i)^#+\s*(LEGAL\s+)?ARGUMENT', "LEGAL ARGUMENT"),
            (r'(?i)^#+\s*CONCLUSION', "CONCLUSION"),
            (r'(?i)^#+\s*(CERTIFICATE|WHEREFORE)', "CONCLUSION"),
        ]
        
        for line in content.split('\n'):
            matched = False
            for pattern, section_name in section_patterns:
                if re.match(pattern, line.strip()):
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    current_section = section_name
                    current_content = [line]
                    matched = True
                    break
            
            if not matched:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections


class ChatService:
    """
    Manages strategy chat sessions with attorneys.
    Maintains conversation context and integrates retrieval.
    """
    
    def __init__(self, retriever, generator):
        self.llm = OllamaService()
        self.prompt_builder = PromptBuilder()
        self.retriever = retriever
        self.generator = generator
        self.sessions: Dict[str, StrategySession] = {}
    
    def create_session(
        self,
        motion_type: Optional[str] = None,
        charge_type: Optional[str] = None,
        key_facts: Optional[str] = None
    ) -> StrategySession:
        """Create a new strategy session"""
        session = StrategySession(
            motion_type=motion_type,
            charge_type=charge_type,
            key_facts=key_facts
        )
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[StrategySession]:
        """Retrieve an existing session"""
        return self.sessions.get(session_id)
    
    def chat(
        self,
        session_id: str,
        message: str,
        use_retrieval: bool = True
    ) -> ChatMessage:
        """Process a chat message and generate response"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Add user message
        user_msg = ChatMessage(role="user", content=message)
        session.messages.append(user_msg)
        
        # Build context from retrieval if enabled
        context = None
        if use_retrieval:
            filters = {}
            if session.motion_type:
                filters[settings.vector_db.motion_type_field] = session.motion_type
            if session.charge_type:
                filters[settings.vector_db.charge_type_field] = session.charge_type
            
            context = self.retriever.retrieve(
                query=message,
                filters=filters if filters else None,
                top_k=5
            )
        
        # Build prompt with conversation history
        history = self._format_history(session.messages[-10:])  # Last 10 messages
        
        system_prompt, base_prompt = self.prompt_builder.build_strategy_prompt(
            question=message,
            context=context or RetrievedContext(
                chunks=[], metadata=[], relevance_scores=[], 
                query=message, filters_applied={}
            )
        )
        
        # Add session context
        if session.key_facts:
            base_prompt = f"## Case Background:\n{session.key_facts}\n\n{base_prompt}"
        
        if history:
            base_prompt = f"## Conversation History:\n{history}\n\n{base_prompt}"
        
        # Generate response
        response_text = self.llm.generate(base_prompt, system_prompt=system_prompt)
        
        # Create assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=response_text,
            retrieved_context=context
        )
        session.messages.append(assistant_msg)
        session.updated_at = datetime.utcnow()
        
        return assistant_msg
    
    def generate_draft_from_session(
        self,
        session_id: str,
        additional_facts: Optional[str] = None
    ) -> MotionDraft:
        """Generate a motion draft based on session discussion"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if not session.motion_type:
            raise ValueError("Session must have a motion type to generate draft")
        
        # Compile facts from session
        facts = session.key_facts or ""
        if additional_facts:
            facts = f"{facts}\n\n{additional_facts}"
        
        # Extract legal issues discussed
        legal_issues = self._extract_discussed_issues(session)
        
        # Generate draft
        draft = self.generator.generate_draft(
            motion_type=session.motion_type,
            facts=facts,
            legal_issues=legal_issues
        )
        
        session.draft_ids.append(draft.draft_id)
        return draft
    
    def _format_history(self, messages: List[ChatMessage]) -> str:
        """Format conversation history for prompt"""
        formatted = []
        for msg in messages[:-1]:  # Exclude current message
            role = "Attorney" if msg.role == "user" else "Assistant"
            formatted.append(f"{role}: {msg.content[:500]}...")
        return "\n\n".join(formatted)
    
    def _extract_discussed_issues(self, session: StrategySession) -> List[str]:
        """Extract legal issues discussed in session"""
        issues = set()
        
        # Keywords to look for
        issue_keywords = [
            "fourth amendment", "fifth amendment", "sixth amendment",
            "miranda", "search", "seizure", "warrant", "consent",
            "probable cause", "reasonable suspicion", "terry stop",
            "speedy trial", "due process", "discovery", "brady"
        ]
        
        for msg in session.messages:
            content_lower = msg.content.lower()
            for keyword in issue_keywords:
                if keyword in content_lower:
                    issues.add(keyword.title())
        
        return list(issues)
