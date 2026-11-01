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
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Proportion of answer claims directly supported by retrieved context (0-1.0)")
    answer_relevancy: float = Field(..., ge=0.0, le=1.0, description="Semantic relevance of answer to user question (0-1.0)")
    context_precision: float = Field(..., ge=0.0, le=1.0, description="Proportion of relevant context chunks ranked near the top (0-1.0)")
    context_recall: float = Field(..., ge=0.0, le=1.0, description="Proportion of ground truth reference facts covered by retrieved context (0-1.0)")
    ragas_score: float = Field(default=0.85, ge=0.0, le=1.0, description="Overall harmonic/mean Ragas benchmark score (0-1.0)")
    answer_correctness: Optional[float] = Field(None, description="Legacy alias for context_precision / factual accuracy")
    answer_completeness: Optional[float] = Field(None, description="Legacy alias for context_recall / facet coverage")
    reasoning: Dict[str, str] = Field(default_factory=dict, description="Explanations for each of the 4 Ragas metrics")

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
    vector_store_status: str = "ChromaDB Embedded Vector Engine Active"

class GeneratedQuestion(BaseModel):
    question: str
    ground_truth: str
    target_strategy_focus: str
    aspect: str

class DocUploadJsonRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    content: str = Field(..., min_length=20)

class DocumentUploadResponse(BaseModel):
    status: str = "success"
    title: str
    parents_ingested: int
    chunks_created: int
    generated_questions: List[GeneratedQuestion]
    message: str
