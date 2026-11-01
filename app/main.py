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
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

# Ensure app directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import BenchmarkRequest, BenchmarkResponse, DocumentUploadResponse, DocUploadJsonRequest, GeneratedQuestion
from app.core.langgraph_workflow import execute_parallel_benchmark
from app.core.chroma_vector_store import global_chroma_store
from app.core.document_ingest import extract_text_from_bytes, ingest_document_into_store, generate_benchmark_questions_from_document
from app.data.sample_knowledge_base import BENCHMARK_PRESET_QUERIES, SAMPLE_ENTERPRISE_DOCS

app = FastAPI(
    title="RAGLens: RAG Evaluation & Benchmarking Platform",
    description="Benchmarking 5 RAG strategies in parallel via LangGraph with ChromaDB and Ragas LLM-as-a-Judge.",
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

@app.post("/api/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    content: Optional[str] = Form(None)
):
    """
    Ingests user document (PDF, TXT, MD) into ChromaDB vector store (parent-child chunks)
    and automatically generates a comprehensive benchmark question testset covering multiple aspects.
    """
    try:
        doc_title = title or "Custom Technical Document"
        full_text = ""

        if file and file.filename:
            content_bytes = await file.read()
            doc_title = title or file.filename
            full_text = extract_text_from_bytes(file.filename, content_bytes)
        elif content:
            full_text = content.strip()
        else:
            raise HTTPException(status_code=400, detail="Please upload a file (.pdf, .txt, .md) or provide text content.")

        if len(full_text.strip()) < 30:
            raise HTTPException(status_code=400, detail="Document content is too short (minimum 30 characters required).")

        # 1. Ingest into ChromaDB knowledge base
        parents_count, chunks_count = ingest_document_into_store(title=doc_title, text=full_text)

        # 2. Synthesize benchmark questions with LLM
        questions = await generate_benchmark_questions_from_document(doc_title=doc_title, full_text=full_text)

        return DocumentUploadResponse(
            status="success",
            title=doc_title,
            parents_ingested=parents_count,
            chunks_created=chunks_count,
            generated_questions=[GeneratedQuestion(**q) for q in questions],
            message=f"Successfully ingested '{doc_title}' into ChromaDB vector store ({parents_count} parents, {chunks_count} chunks) and generated {len(questions)} evaluation questions."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@app.post("/api/upload-document-json", response_model=DocumentUploadResponse)
async def upload_document_json(req: DocUploadJsonRequest):
    """JSON alternative for document ingestion."""
    try:
        parents_count, chunks_count = ingest_document_into_store(title=req.title, text=req.content)
        questions = await generate_benchmark_questions_from_document(doc_title=req.title, full_text=req.content)
        return DocumentUploadResponse(
            status="success",
            title=req.title,
            parents_ingested=parents_count,
            chunks_created=chunks_count,
            generated_questions=[GeneratedQuestion(**q) for q in questions],
            message=f"Successfully ingested '{req.title}' into ChromaDB vector store and generated {len(questions)} evaluation questions."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

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
                "storage": "ChromaDB Dense HNSW Index",
                "focus": "Direct cosine similarity passage retrieval."
            },
            {
                "id": "hybrid_bm25",
                "name": "Hybrid BM25 RAG",
                "storage": "ChromaDB Dense + BM25 Lexical",
                "focus": "Solves lexical mismatch on exact acronyms and ciphers using Reciprocal Rank Fusion."
            },
            {
                "id": "cohere_rerank",
                "name": "Cross-Encoder / Cohere Rerank",
                "storage": "ChromaDB Pool + ms-marco-MiniLM-L-6-v2 CrossEncoder",
                "focus": "Deep cross-attention scoring between query and passages to prioritize top relevance using neural reranker model."
            },
            {
                "id": "hyde",
                "name": "HyDE (Hypothetical Doc Embeddings)",
                "storage": "Zero-shot LLM + ChromaDB Vector Index",
                "focus": "Bridges semantic query-to-document asymmetry by generating hypothetical answer first."
            },
            {
                "id": "parent_child",
                "name": "Parent-Child Document Expansion",
                "storage": "ChromaDB Child Chunks & Parent Metadata",
                "focus": "Searches small child chunks (100 tokens), expands to parent (500 tokens) to eliminate context boundaries."
            }
        ]
    }

@app.get("/api/chroma/stats")
async def get_chroma_stats():
    """Returns ChromaDB collection metadata and document counts."""
    return {
        "collection_name": global_chroma_store.collection.name,
        "total_chunks": global_chroma_store.count(),
        "parent_documents_stored": len(global_chroma_store.parents),
        "embedding_dimensions": 384,
        "embedding_model": "BAAI/bge-small-en-v1.5"
    }

@app.get("/api/health")
async def health_check():
    """Service health and engine status."""
    return {
        "status": "healthy",
        "service": "RAGLens",
        "version": "2.1.0",
        "vector_store": "ChromaDB (BAAI/bge-small-en-v1.5)",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "orchestrator": "LangGraph Parallel StateGraph",
        "evaluator": "Ragas 4-Metric Engine (Faithfulness, Relevancy, Precision, Recall)",
        "gemini_api_configured": bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip())
    }
