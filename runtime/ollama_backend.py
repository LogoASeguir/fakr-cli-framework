"""
Ollama Multi-Model Backend: Route queries to 3B/5B/8B models based on complexity.

Architecture:
  - 3B model (always running): Fast conversational responses
  - 5B model: Reasoning tasks, complexity > threshold 1
  - 8B model (OFFLINE): Deep knowledge synthesis, files/URLs → fractal knowledge
  
Query complexity scoring:
  - Token count
  - Semantic depth (via router)
  - Task type classification
  - Required reasoning steps estimate
  
Benefits:
  - 3B fast-path for normal conversation
  - 5B for analysis needing reasoning
  - 8B only for deep learning tasks
  - Reduced latency + memory usage
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import requests
from pathlib import Path
import time
import os

try:
    import aiohttp
    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


# ===================================================================
# Configuration & Model Metadata
# ===================================================================

class ModelSize(str, Enum):
    """Model size categories."""
    SMALL = "3B"      # Always available: ~2GB RAM, 10ms latency
    MEDIUM = "5B"     # Default reasoning: ~4GB RAM, 30ms latency
    LARGE = "8B"      # Heavy lifting: ~6GB RAM, 100ms latency


@dataclass
class ModelMetadata:
    """Metadata for a single model in Ollama."""
    name: str
    size: ModelSize
    context_window: int  # tokens
    quantization: str    # e.g., "q4_K_M"
    ollama_model_id: str  # what to request from Ollama
    memory_requirement_gb: float
    avg_latency_ms: float
    
    def __hash__(self) -> int:
        return hash(self.name)


# Standard 3-tier setup
DEFAULT_MODELS = {
    "phi3-mini": ModelMetadata(
        name="phi3-mini",
        size=ModelSize.SMALL,
        context_window=4096,
        quantization="q4_K_M",
        ollama_model_id="phi3-mini:3.8b",
        memory_requirement_gb=2.0,
        avg_latency_ms=10,
    ),
    "neural-chat5b": ModelMetadata(
        name="neural-chat5b",
        size=ModelSize.MEDIUM,
        context_window=8192,
        quantization="q4_K_M",
        ollama_model_id="neural-chat:5b",
        memory_requirement_gb=4.0,
        avg_latency_ms=30,
    ),
    "llama2-8b": ModelMetadata(
        name="llama2-8b",
        size=ModelSize.LARGE,
        context_window=4096,
        quantization="q4_K_M",
        ollama_model_id="llama2:8b",
        memory_requirement_gb=6.0,
        avg_latency_ms=100,
    ),
}


# ===================================================================
# Query Complexity Scoring
# ===================================================================

class TaskType(str, Enum):
    """Classify task to estimate complexity."""
    CHAT = "chat"           # Simple conversation
    QUESTION_ANSWERING = "qa"  # Factual questions
    SUMMARIZATION = "summarize"  # Condense text
    REASONING = "reasoning"  # Logic, math, code
    SYNTHESIS = "synthesis"  # Create from knowledge
    LEARNING = "learn"      # Ingest new knowledge (OFFLINE)


@dataclass
class ComplexityScore:
    """Multi-dimensional complexity metric."""
    token_count: int
    semantic_depth: float  # 0.0-1.0 from semantic router
    task_type: TaskType
    reasoning_steps_estimate: int  # 1-5
    overall_score: float  # 0-100
    recommended_model: ModelSize
    
    def __repr__(self) -> str:
        return (
            f"ComplexityScore(score={self.overall_score:.1f}, "
            f"model={self.recommended_model}, task={self.task_type})"
        )


class ComplexityAnalyzer:
    """Estimate query complexity for routing."""
    
    # Thresholds for model selection
    THRESHOLD_3B_TO_5B = 30  # complexity score
    THRESHOLD_5B_TO_8B = 65
    
    def __init__(self) -> None:
        self.task_scores: Dict[TaskType, float] = {
            TaskType.CHAT: 10,
            TaskType.QUESTION_ANSWERING: 20,
            TaskType.SUMMARIZATION: 25,
            TaskType.REASONING: 50,
            TaskType.SYNTHESIS: 70,
            TaskType.LEARNING: 85,
        }
        
        self.metrics = {
            "total_queries": 0,
            "routed_to_3b": 0,
            "routed_to_5b": 0,
            "routed_to_8b": 0,
        }
    
    def estimate_reasoning_steps(self, query: str) -> int:
        """Estimate how many reasoning steps query needs."""
        keywords = {
            "why": 2,
            "how": 2,
            "explain": 2,
            "define": 1,
            "summarize": 1,
            "analyze": 3,
            "design": 4,
            "implement": 4,
            "optimize": 3,
            "debug": 3,
        }
        
        query_lower = query.lower()
        max_steps = 1
        for keyword, steps in keywords.items():
            if keyword in query_lower:
                max_steps = max(max_steps, steps)
        
        return max_steps
    
    def classify_task(self, query: str) -> TaskType:
        """Determine task type from query."""
        query_lower = query.lower()
        
        # Heuristic classification
        if any(w in query_lower for w in ["why", "how", "explain", "reason"]):
            return TaskType.REASONING
        elif any(w in query_lower for w in ["summarize", "summary", "condense"]):
            return TaskType.SUMMARIZATION
        elif any(w in query_lower for w in ["create", "design", "build", "write"]):
            return TaskType.SYNTHESIS
        elif any(w in query_lower for w in ["learn", "understand", "ingest", "import"]):
            return TaskType.LEARNING
        elif "?" in query:
            return TaskType.QUESTION_ANSWERING
        else:
            return TaskType.CHAT
    
    def analyze(
        self,
        query: str,
        semantic_depth: float = 0.5,
    ) -> ComplexityScore:
        """
        Analyze query complexity and recommend model.
        
        semantic_depth: 0.0 (simple) to 1.0 (complex)
        """
        # Token count (rough: ~4 chars per token)
        token_count = len(query) // 4 + 1
        
        # Task type
        task_type = self.classify_task(query)
        task_score = self.task_scores.get(task_type, 20)
        
        # Reasoning steps
        reasoning_steps = self.estimate_reasoning_steps(query)
        
        # Combine scores (weighted average)
        score = (
            (token_count / 10) * 0.2 +        # Token count (0-30 range)
            (semantic_depth * 30) * 0.3 +      # Semantic depth
            task_score * 0.4 +                  # Task complexity
            (reasoning_steps * 10) * 0.1        # Reasoning estimate
        )
        score = min(100, score)  # Cap at 100
        
        # Recommend model
        if score < self.THRESHOLD_3B_TO_5B:
            recommended = ModelSize.SMALL
            self.metrics["routed_to_3b"] += 1
        elif score < self.THRESHOLD_5B_TO_8B:
            recommended = ModelSize.MEDIUM
            self.metrics["routed_to_5b"] += 1
        else:
            recommended = ModelSize.LARGE
            self.metrics["routed_to_8b"] += 1
        
        self.metrics["total_queries"] += 1
        
        return ComplexityScore(
            token_count=token_count,
            semantic_depth=semantic_depth,
            task_type=task_type,
            reasoning_steps_estimate=reasoning_steps,
            overall_score=score,
            recommended_model=recommended,
        )


# ===================================================================
# Ollama Client
# ===================================================================

@dataclass
class OllamaResponse:
    """Response from an Ollama API call."""
    model_used: str
    text: str
    tokens_generated: int
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    """Interface to Ollama API with multi-model support."""
    
    def __init__(
        
        self,
        base_url: str = OLLAMA_BASE_URL,
        models: Optional[Dict[str, ModelMetadata]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models = models or DEFAULT_MODELS.copy()
        self.analyzer = ComplexityAnalyzer()
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "total_latency_ms": 0,
        }
    
    def available_models(self) -> List[str]:
        """List registered models."""
        return list(self.models.keys())
    
    def select_model(self, query: str, semantic_depth: float = 0.5) -> str:
        """
        Choose best model for query via complexity analysis.
        
        Returns: model name
        """
        score = self.analyzer.analyze(query, semantic_depth)
        
        # Find first model matching recommended size
        for name, meta in self.models.items():
            if meta.size == score.recommended_model:
                return name
        
        # Fallback to 3B
        return "phi3-mini"
    
    def generate(
        self,
        query: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        auto_select: bool = True,
    ) -> OllamaResponse:
        """
        Generate response using specified or auto-selected model.
        
        auto_select: If True, choose model based on query complexity
        """
        start_time = time.time()
        
        # Select model if not specified
        if not model:
            if auto_select:
                model = self.select_model(query, semantic_depth=0.5)
            else:
                model = "phi3-mini"  # default
        
        if model not in self.models:
            raise ValueError(f"Unknown model: {model}")
        
        meta = self.models[model]
        
        # Prepare request
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": meta.ollama_model_id,
            "prompt": query,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "stream": False,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            self.metrics["requests"] += 1
            self.metrics["total_latency_ms"] += latency_ms
            
            return OllamaResponse(
                model_used=model,
                text=data.get("response", ""),
                tokens_generated=data.get("eval_count", 0),
                latency_ms=latency_ms,
                metadata={
                    "model_id": meta.ollama_model_id,
                    "context_window": meta.context_window,
                },
            )
        
        except Exception as e:
            self.metrics["errors"] += 1
            
            return OllamaResponse(
                model_used=model,
                text=f"[Error calling Ollama: {e}]",
                tokens_generated=0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )
    
    def generate_with_context(
        self,
        query: str,
        context_blocks: List[str],
        model: Optional[str] = None,
    ) -> OllamaResponse:
        """
        Generate response with knowledge blocks as context.
        
        Useful for rag-style retrieval + generation.
        """
        # Construct contextualized prompt
        context_str = "\n\n---\n\n".join(context_blocks)
        full_prompt = f"{context_str}\n\nQuestion: {query}"
        
        return self.generate(full_prompt, model=model)
    
    def create_session(
        self,
        session_id: str,
        model: str = "phi3-mini",
        system_prompt: str = "",
    ) -> None:
        """Create a conversation session with context preservation."""
        self.active_sessions[session_id] = {
            "model": model,
            "system_prompt": system_prompt,
            "history": [],
        }
    
    def chat(self, session_id: str, message: str) -> OllamaResponse:
        """Continue conversation in a session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"No session: {session_id}")
        
        session = self.active_sessions[session_id]
        
        # Build conversation history
        history_str = ""
        for role, text in session["history"]:
            history_str += f"{role}: {text}\n"
        
        # Prompt
        full_prompt = (
            f"{session['system_prompt']}\n\n"
            f"{history_str}\n"
            f"User: {message}\n"
            f"Assistant:"
        )
        
        response = self.generate(full_prompt, model=session["model"])
        
        # Store in history
        session["history"].append(("User", message))
        session["history"].append(("Assistant", response.text))
        
        # Keep history bounded
        if len(session["history"]) > 20:
            session["history"] = session["history"][-20:]
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return performance metrics."""
        avg_latency = (
            self.metrics["total_latency_ms"] / max(self.metrics["requests"], 1)
        )
        
        return {
            "total_requests": self.metrics["requests"],
            "total_errors": self.metrics["errors"],
            "avg_latency_ms": round(avg_latency, 1),
            "error_rate": (
                self.metrics["errors"] / max(self.metrics["requests"], 1)
            ),
            "complexity_routing": self.analyzer.metrics.copy(),
        }
    
    def health_check(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# ===================================================================
# Example/Test
# ===================================================================

if __name__ == "__main__":
    client = OllamaClient()
    
    # Test routing
    test_queries = [
        "Hello, how are you?",
        "Explain quantum entanglement",
        "Design a REST API for a task manager",
        "Summarize the contents of philosophy_book.pdf",
    ]
    
    print("Query Complexity Analysis & Model Routing:\n")
    for query in test_queries:
        score = client.analyzer.analyze(query, semantic_depth=0.5)
        model = client.select_model(query)
        print(f"Query: {query}")
        print(f"  → Complexity: {score.overall_score:.1f}/100")
        print(f"  → Task: {score.task_type.value}")
        print(f"  → Model: {model} ({score.recommended_model.value})")
        print()
    
    print("Routing Metrics:")
    print(json.dumps(client.analyzer.metrics, indent=2))
