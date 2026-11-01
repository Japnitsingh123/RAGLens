"""
RAGLens Schemas: Pydantic models for the 5-Strategy RAG Evaluation Platform.
Benchmarking: Naive, Hybrid BM25, Cohere Rerank, HyDE, Parent-Child via LangGraph.
Evaluation: Ragas (Faithfulness, Correctness, Completeness) & Failure Attribution.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

class StrategyType(str, Enum):
    NAIVE = "naive"
    HYBRID_BM25 = "hybrid_bm25"
    COHERE_RERANK = "cohere_rerank"
    HYDE = "hyde"
    PARENT_CHILD = "parent_child"

class RootCauseType(str, Enum):
    NONE = "none"
    CONTEXT_LIMITS = "context_limits"
    EMBEDDING_WEAKNESS = "embedding_weakness"
    HALLUCINATION = "hallucination"

class RagasMetrics(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Claims supported by retrieved context (0-1.0)")
    answer_correctness: float = Field(..., ge=0.0, le=1.0, description="Semantic and factual accuracy vs ground truth (0-1.0)")
    answer_completeness: float = Field(..., ge=0.0, le=1.0, description="Coverage of necessary facets in query (0-1.0)")
    reasoning: Dict[str, str] = Field(default_factory=dict, description="Explanations for each metric score")

class FailureAttribution(BaseModel):
    primary_root_cause: RootCauseType = Field(..., description="Diagnosed primary failure mode")
    context_limits_score: float = Field(..., ge=0.0, le=1.0, description="Risk/severity of context truncation or limit issues")
    embedding_weakness_score: float = Field(..., ge=0.0, le=1.0, description="Risk/severity of vocabulary or semantic retrieval mismatch")
    hallucination_score: float = Field(..., ge=0.0, le=1.0, description="Risk/severity of unsupported claims in generated text")
    diagnosis_details: str = Field(..., description="Detailed explanation of what failed and why")
    recommendation: str = Field(..., description="Actionable architectural fix for this strategy")

class StrategyResult(BaseModel):
    strategy: StrategyType
    strategy_name: str
    description: str
    answer: str
    contexts: List[str]
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_tokens: int
    ragas_metrics: RagasMetrics
    failure_attribution: FailureAttribution
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BenchmarkRequest(BaseModel):
    query: str = Field(..., min_length=2, json_schema_extra={"example": "What encryption standard is used for data at rest and in transit?"})
    ground_truth: Optional[str] = Field(None, json_schema_extra={"example": "AES-256-GCM for data at rest and TLS 1.3 with forward secrecy for transit."})
    top_k: int = Field(3, ge=1, le=10)

class BenchmarkResponse(BaseModel):
    query: str
    ground_truth: Optional[str]
    strategies_evaluated: int
    execution_mode: str = "LangGraph Parallel Execution"
    results: List[StrategyResult]
    best_strategy: str
    executive_summary: str
    mysql_vector_status: str = "MySQL 8.4 VECTOR(768) Engine Active"
