"""
RAGLens: Production RAG Evaluation & Benchmarking Platform.
Tech Stack: Next.js 14 Aesthetic • FastAPI • MySQL VECTOR(768) • LangGraph • Ragas LLM-as-a-Judge

Benchmarks 5 RAG strategies in parallel via LangGraph:
1. Naive Dense RAG
2. Hybrid BM25
3. Cohere Cross-Encoder Rerank
4. HyDE (Hypothetical Document Embeddings)
5. Parent-Child Document Expansion

Evaluates Ragas Metrics:
- Faithfulness
- Answer Correctness
- Answer Completeness

Failure Attribution Engine:
- Diagnoses Root Causes: Context Limits, Embedding Weakness, Hallucinations
- Provides Actionable Engineering Recommendations
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

# Ensure app directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import BenchmarkRequest, BenchmarkResponse
from app.core.langgraph_workflow import execute_parallel_benchmark
from app.core.mysql_vector_store import MYSQL_DDL_SCHEMA
from app.data.sample_knowledge_base import BENCHMARK_PRESET_QUERIES, SAMPLE_ENTERPRISE_DOCS

app = FastAPI(
    title="RAGLens: RAG Evaluation & Benchmarking Platform",
    description="Benchmarking 5 RAG strategies in parallel via LangGraph with MySQL VECTOR and Ragas LLM-as-a-Judge.",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 Templates
templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the Next.js 14 styled benchmarking dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "presets": BENCHMARK_PRESET_QUERIES,
            "docs_count": len(SAMPLE_ENTERPRISE_DOCS)
        }
    )

@app.post("/api/benchmark", response_model=BenchmarkResponse)
async def run_parallel_benchmark(request: BenchmarkRequest):
    """
    Executes all 5 RAG strategies in parallel via LangGraph:
    - Naive Dense RAG
    - Hybrid BM25 (Dense + Fulltext RRF)
    - Cohere Cross-Encoder Rerank
    - HyDE (Hypothetical Document Embeddings)
    - Parent-Child Chunk Expansion

    Evaluates Ragas metrics (Faithfulness, Correctness, Completeness)
    and attributes failure modes (Context Limits vs Embedding Weakness vs Hallucination).
    """
    try:
        report = await execute_parallel_benchmark(
            query=request.query,
            ground_truth=request.ground_truth,
            top_k=request.top_k
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmarking execution failed: {str(e)}")

@app.get("/api/presets")
async def get_presets():
    """Returns curated preset benchmark queries targeting specific failure modes."""
    return {"presets": BENCHMARK_PRESET_QUERIES}

@app.get("/api/strategies")
async def get_strategies():
    """Returns the 5 benchmarked RAG architectures."""
    return {
        "strategies": [
            {
                "id": "naive",
                "name": "Naive Dense RAG",
                "storage": "MySQL VECTOR(768)",
                "focus": "Direct cosine similarity passage retrieval."
            },
            {
                "id": "hybrid_bm25",
                "name": "Hybrid BM25 RAG",
                "storage": "MySQL VECTOR + MySQL FULLTEXT",
                "focus": "Solves lexical mismatch on exact acronyms and ciphers using Reciprocal Rank Fusion."
            },
            {
                "id": "cohere_rerank",
                "name": "Cohere Cross-Encoder Rerank",
                "storage": "MySQL VECTOR Candidate Pool",
                "focus": "Deep cross-attention scoring between query and passages to prioritize top relevance."
            },
            {
                "id": "hyde",
                "name": "HyDE (Hypothetical Doc Embeddings)",
                "storage": "Zero-shot LLM + MySQL VECTOR",
                "focus": "Bridges semantic query-to-document asymmetry by generating hypothetical answer first."
            },
            {
                "id": "parent_child",
                "name": "Parent-Child Document Expansion",
                "storage": "MySQL rag_documents & rag_parent_documents",
                "focus": "Searches small child chunks (100 tokens), expands to parent (500 tokens) to eliminate context boundaries."
            }
        ]
    }

@app.get("/api/schema/mysql")
async def get_mysql_schema():
    """Returns the production MySQL 8.4+ DDL schema for VECTOR and FULLTEXT search."""
    return {"schema_sql": MYSQL_DDL_SCHEMA}

@app.get("/api/health")
async def health_check():
    """Service health and engine status."""
    return {
        "status": "healthy",
        "service": "RAGLens",
        "version": "2.0.0",
        "vector_store": "MySQL VECTOR(768) Engine",
        "orchestrator": "LangGraph Parallel StateGraph",
        "evaluator": "Ragas LLM-as-a-Judge",
        "gemini_api_configured": bool(os.getenv("GEMINI_API_KEY", "").strip())
    }
