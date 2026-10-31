from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Pipeline Configuration ---
class PipelineConfig(BaseModel):
    mode: str = Field(default="builtin", description="'builtin' or 'external'")
    # Built-in RAG options
    search_type: str = Field(default="hybrid", description="'vector' or 'hybrid'")
    embedding_model: str = Field(default="text-embedding-3-small", description="Model label for embeddings")
    chunk_size: int = Field(default=512, description="Chunk size in characters")
    chunk_overlap: int = Field(default=64, description="Chunk overlap in characters")
    top_k: int = Field(default=4, description="Number of retrieved chunks")
    semantic_cache_enabled: bool = Field(default=True, description="Enable semantic caching")
    cache_similarity_threshold: float = Field(default=0.88, description="Cosine similarity threshold for cache hit")
    security_guardrails_enabled: bool = Field(default=True, description="Enable input & document guardrails")
    llm_model: str = Field(default="gpt-4o", description="LLM used for generation / cost calculation")

    # External RAG API options
    external_url: Optional[str] = Field(default=None, description="External RAG endpoint (e.g. http://localhost:8080/rag)")
    external_headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Headers e.g. Authorization")
    request_template: Optional[Dict[str, Any]] = Field(default=None, description="JSON body payload template")
    response_answer_key: Optional[str] = Field(default="answer", description="JSON key for answer in response")
    response_contexts_key: Optional[str] = Field(default="contexts", description="JSON key for retrieved contexts")

# --- Test Items ---
class TestCaseItem(BaseModel):
    id: str
    question: str
    ground_truth: str
    contexts: Optional[List[str]] = None
    generated_answer: Optional[str] = None
    category: Optional[str] = "general"

class AdversarialSecurityTest(BaseModel):
    id: str
    name: str
    category: str  # 'prompt_injection', 'system_override', 'poisoned_document'
    attack_payload: str
    expected_behavior: str
    target_field: str = "query"  # 'query' or 'document'
    passed: bool = False
    verdict: str = "Pending"  # 'SAFE', 'VULNERABLE', 'BLOCKED'
    risk_level: str = "HIGH"  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    evidence: str = ""

# --- Quality Metric Schemas ---
class QualityItemResult(BaseModel):
    test_id: str
    question: str
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    context_recall_score: float
    hallucination_detected: bool
    unsupported_claims: List[str] = Field(default_factory=list)
    supported_claims: List[str] = Field(default_factory=list)
    retrieved_contexts: List[str] = Field(default_factory=list)
    generated_answer: str

class QualitySummary(BaseModel):
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    overall_quality_score: float
    items: List[QualityItemResult] = Field(default_factory=list)

# --- Performance Metric Schemas ---
class PerformanceSummary(BaseModel):
    avg_response_latency_ms: float
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_total_tokens: float
    total_tokens_consumed: int
    estimated_cost_usd: float
    estimated_cost_per_1k_queries: float
    semantic_cache_hit_rate_pct: float
    cache_hits: int
    cache_misses: int
    latency_breakdown: Dict[str, float] = Field(default_factory=dict)

# --- Security Metric Schemas ---
class SecuritySummary(BaseModel):
    overall_security_score: float  # 0.0 to 1.0 (1.0 = 100% passed)
    total_tests: int
    passed_tests: int
    failed_tests: int
    risk_rating: str  # 'Secure', 'Medium Risk', 'High Risk', 'Critical Vulnerabilities'
    test_results: List[AdversarialSecurityTest] = Field(default_factory=list)
    vulnerability_breakdown: Dict[str, int] = Field(default_factory=dict)

# --- Benchmark Schemas ---
class BenchmarkComparisonItem(BaseModel):
    configuration_name: str
    variant: str
    retrieval_latency_ms: float
    context_precision: float
    context_recall: float
    token_usage: int
    cost_per_query: float
    description: str

class BenchmarkGroupResult(BaseModel):
    benchmark_type: str  # 'search_mode', 'chunk_size', 'top_k', 'caching'
    title: str
    winner: str
    recommendation: str
    variants: List[BenchmarkComparisonItem]

class BenchmarkSummary(BaseModel):
    groups: List[BenchmarkGroupResult] = Field(default_factory=list)

# --- Full Evaluation Report ---
class FullEvaluationReport(BaseModel):
    run_id: str
    timestamp: str
    pipeline_mode: str
    raglens_score: float  # 0 - 100
    status: str  # 'Production Ready', 'Needs Improvement', 'High Risk'
    quality: QualitySummary
    performance: PerformanceSummary
    security: SecuritySummary
    benchmarking: BenchmarkSummary
    executive_summary: str
    what_is_working: List[str] = Field(default_factory=list)
    what_is_failing: List[str] = Field(default_factory=list)
    what_needs_improvement: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

# --- Evaluation Request ---
class RunEvaluationRequest(BaseModel):
    pipeline_config: PipelineConfig = Field(default_factory=PipelineConfig)
    suites: List[str] = Field(default_factory=lambda: ["quality", "performance", "security", "benchmarking"])
    custom_test_items: Optional[List[TestCaseItem]] = None
    custom_documents: Optional[List[str]] = None
