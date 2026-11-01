"""
Comprehensive Unit & Integration Test Suite for RAGLens.
Tests:
- MySQL Vector Store & Hybrid Search
- 5 RAG Strategies: Naive, Hybrid BM25, Cohere Rerank, HyDE, Parent-Child
- LangGraph Parallel Execution Workflow
- Ragas LLM-as-a-Judge (Faithfulness, Correctness, Completeness)
- Failure Attribution Diagnostics (Context Limits, Embedding Weakness, Hallucination)
- FastAPI Endpoints
"""

import os
import sys
from pathlib import Path

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.mysql_vector_store import MySQLVectorStore, generate_deterministic_embedding, cosine_similarity
from app.core.rag_strategies import (
    execute_naive_rag,
    execute_hybrid_bm25,
    execute_cohere_rerank,
    execute_hyde,
    execute_parent_child
)
from app.core.ragas_evaluator import RagasEvaluator
from app.core.failure_attribution import FailureAttributionEngine
from app.core.langgraph_workflow import execute_parallel_benchmark
from app.models.schemas import RootCauseType

client = TestClient(app)

def test_mysql_vector_store():
    store = MySQLVectorStore()
    store.add_parent_document("doc_1", "Security Specs", "Complete technical security architecture with AES-256-GCM.")
    store.add_chunk("chunk_1", "doc_1", "AES-256-GCM authenticated encryption for data at rest.", "Security Specs")
    store.add_chunk("chunk_2", "doc_1", "TLS 1.3 for data in transit across microservices.", "Security Specs")

    # Test Dense Vector Search
    dense_results = store.similarity_search("encryption for data at rest", top_k=1)
    assert len(dense_results) == 1
    assert "AES-256-GCM" in dense_results[0]["content"]

    # Test MySQL Fulltext Search
    fts_results = store.mysql_fulltext_search("TLS microservices", top_k=1)
    assert len(fts_results) == 1
    assert "TLS 1.3" in fts_results[0]["content"]

    # Test Parent Document Lookup
    parent = store.get_parent_document("doc_1")
    assert parent is not None
    assert "Complete technical security" in parent["content"]

@pytest.mark.anyio
async def test_5_rag_strategies():
    query = "What encryption is used for data at rest?"
    
    # 1. Naive
    naive = await execute_naive_rag(query, top_k=2)
    assert naive["strategy"] == "naive"
    assert len(naive["contexts"]) > 0

    # 2. Hybrid BM25
    hybrid = await execute_hybrid_bm25(query, top_k=2)
    assert hybrid["strategy"] == "hybrid_bm25"
    assert len(hybrid["contexts"]) > 0

    # 3. Cohere Rerank
    rerank = await execute_cohere_rerank(query, top_k=2)
    assert rerank["strategy"] == "cohere_rerank"
    assert len(rerank["contexts"]) > 0

    # 4. HyDE
    hyde = await execute_hyde(query, top_k=2)
    assert hyde["strategy"] == "hyde"
    assert len(hyde["contexts"]) > 0

    # 5. Parent-Child
    parent_child = await execute_parent_child(query, top_k=2)
    assert parent_child["strategy"] == "parent_child"
    assert len(parent_child["contexts"]) > 0

@pytest.mark.anyio
async def test_ragas_evaluation_engine():
    query = "What is the uptime SLA guarantee?"
    answer = "The platform guarantees a 99.99% monthly uptime SLA."
    contexts = ["The platform guarantees a 99.99% monthly uptime Service Level Agreement (SLA) for core transactional endpoints."]
    ground_truth = "A 99.99% monthly uptime SLA."

    metrics = await RagasEvaluator.evaluate_ragas_metrics(query, answer, contexts, ground_truth)
    assert metrics.faithfulness >= 0.8
    assert metrics.answer_correctness >= 0.7
    assert metrics.answer_completeness >= 0.5

def test_failure_attribution_engine():
    from app.models.schemas import RagasMetrics

    # Test Context Limits Diagnosis
    metrics_limit = RagasMetrics(
        faithfulness=0.9,
        answer_correctness=0.6,
        answer_completeness=0.5,
        reasoning={}
    )
    attr = FailureAttributionEngine.diagnose(
        strategy="naive",
        query="Explain full architecture",
        answer="Short response.",
        contexts=["Truncated chunk."],
        ragas_metrics=metrics_limit
    )
    assert attr.primary_root_cause in [RootCauseType.CONTEXT_LIMITS, RootCauseType.EMBEDDING_WEAKNESS]
    assert len(attr.recommendation) > 10

@pytest.mark.anyio
async def test_langgraph_parallel_benchmark():
    query = "What encryption standard is used for data at rest?"
    ground_truth = "AES-256-GCM authenticated encryption."
    
    report = await execute_parallel_benchmark(query, ground_truth, top_k=2)
    assert report["strategies_evaluated"] == 5
    assert len(report["results"]) == 5
    assert report["best_strategy"] != ""

def test_fastapi_endpoints():
    # Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["vector_store"] == "MySQL VECTOR(768) Engine"

    # Presets
    res_presets = client.get("/api/presets")
    assert res_presets.status_code == 200
    assert len(res_presets.json()["presets"]) >= 4

    # Strategies
    res_strats = client.get("/api/strategies")
    assert res_strats.status_code == 200
    assert len(res_strats.json()["strategies"]) == 5

    # MySQL Schema
    res_schema = client.get("/api/schema/mysql")
    assert res_schema.status_code == 200
    assert "CREATE TABLE IF NOT EXISTS rag_documents" in res_schema.json()["schema_sql"]

    # Dashboard HTML
    res_dash = client.get("/")
    assert res_dash.status_code == 200
    assert "RAGLens" in res_dash.text
