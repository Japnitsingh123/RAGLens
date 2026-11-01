import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.models.schemas import (
    RunEvaluationRequest,
    FullEvaluationReport,
    PipelineConfig
)
from app.core.eval_runner import RAGLensRunner
from app.core.rag_engine import RAGEngine
from app.core.doc_parser import DocumentParser
from app.core.testset_generator import TestsetGenerator
from app.core.strategy_benchmark import StrategyBenchmarkRunner
from app.core.llm_security_guard import LLMSecurityGuard
from app.data.sample_docs import SAMPLE_KNOWLEDGE_BASE
from app.data.sample_resume import SAMPLE_RESUME_TEXT
from app.data.default_test_cases import DEFAULT_QUALITY_TESTS, DEFAULT_SECURITY_TESTS

app = FastAPI(
    title="RAGLens API",
    description="Evaluation, Testing & Benchmarking Platform for Retrieval-Augmented Generation (RAG)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global shared instance of RAG engine and runner for interactive playground
default_rag_engine = RAGEngine()
eval_runner = RAGLensRunner()

# Document Upload Schema
class DocUploadRequest(BaseModel):
    title: str = "Custom Document"
    content: str
    category: Optional[str] = "custom"

# Query Request Schema
class QueryRequest(BaseModel):
    query: str
    search_type: Optional[str] = "hybrid"
    top_k: Optional[int] = 4

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the interactive dashboard HTML."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Dashboard template not found.")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/api/health")
async def health_check():
    """Service health and operational status."""
    return {
        "status": "healthy",
        "service": "RAGLens",
        "version": "1.0.0",
        "indexed_chunks": len(default_rag_engine.chunks)
    }

@app.get("/api/sample-data")
async def get_sample_data():
    """Retrieve preloaded test suites and knowledge base documents."""
    return {
        "documents": SAMPLE_KNOWLEDGE_BASE,
        "quality_tests": DEFAULT_QUALITY_TESTS,
        "security_tests": DEFAULT_SECURITY_TESTS
    }

@app.post("/api/evaluate", response_model=FullEvaluationReport)
async def run_evaluation(request: RunEvaluationRequest):
    """
    Execute full evaluation suite:
    1. RAG Quality (Faithfulness, Relevancy, Precision, Recall)
    2. Performance (Latency, Token consumption, Cost per query)
    3. Security (Prompt injection, System instruction overrides, Poisoned docs)
    4. Benchmarking (Vector vs Hybrid, Chunk sizes, Top-K)
    """
    try:
        report = await eval_runner.execute_evaluation(request)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/api/analyze-query")
async def analyze_single_query(request: QueryRequest):
    """
    Executes a query and instantly evaluates:
    1. AI-as-a-Judge Security Guardrail (LLM detects prompt injections, overrides, leaks with any terminology)
    2. RAG Quality (Faithfulness, Relevancy, Precision, Recall)
    3. Performance (Response Latency, Retrieval Latency, Tokens, Cost)
    4. What is working, failing, and needs improvement
    """
    import time
    try:
        t0 = time.perf_counter()

        # 1. AI-as-a-Judge LLM Security Guardrail
        is_attack, attack_type, sec_explanation = await LLMSecurityGuard.analyze_query_security(request.query)
        if is_attack:
            total_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "query": request.query,
                "answer": f"SECURITY ALERT: Adversarial input detected ({attack_type}). {sec_explanation}",
                "contexts": [],
                "quality": {
                    "faithfulness": 0.0,
                    "faithfulness_pct": 0,
                    "answer_relevancy": 0.0,
                    "answer_relevancy_pct": 0,
                    "context_precision": 0.0,
                    "context_precision_pct": 0,
                    "context_recall": 0.0,
                    "context_recall_pct": 0,
                    "hallucination_detected": False,
                    "supported_claims": [],
                    "unsupported_claims": [f"Adversarial {attack_type} attempt intercepted by LLM Security Guard."]
                },
                "performance": {
                    "response_latency_ms": round(total_ms, 2),
                    "retrieval_latency_ms": 0.0,
                    "generation_latency_ms": round(total_ms, 2),
                    "prompt_tokens": int(len(request.query) / 4),
                    "completion_tokens": 20,
                    "total_tokens": int(len(request.query) / 4) + 20,
                    "cost_usd": 0.0
                },
                "security": {
                    "is_safe": True,
                    "status": "BLOCKED (SAFE)",
                    "evidence": sec_explanation
                },
                "what_is_working": [
                    f"LLM Security Guardrail: Successfully identified and neutralized adversarial {attack_type}."
                ],
                "what_is_failing": [
                    f"Adversarial Input Detected: {sec_explanation}"
                ],
                "what_needs_improvement": [
                    "Adversarial probe was safely blocked before affecting model context."
                ]
            }

        # 2. Query RAG for safe queries
        rag_res = default_rag_engine.query(
            query=request.query,
            search_type=request.search_type or "hybrid",
            top_k=request.top_k or 4
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

        # Quality evaluation
        quality_eval = eval_runner.quality_eval
        raw_faith, supported, unsupported = quality_eval.evaluate_faithfulness(
            rag_res.get("answer", ""), rag_res.get("contexts", [])
        )
        relevancy = quality_eval.evaluate_answer_relevancy(request.query, rag_res.get("answer", ""))
        precision = quality_eval.evaluate_context_precision(request.query, rag_res.get("answer", ""), rag_res.get("contexts", []))
        recall = 1.0 if len(rag_res.get("contexts", [])) > 0 else 0.0

        if relevancy < 0.35:
            faithfulness = round(raw_faith * relevancy, 3)
            unsupported.append("Off-Topic Response: The text quotes the document but does not address what was asked.")
            hallucination_detected = True
        else:
            faithfulness = raw_faith
            hallucination_detected = len(unsupported) > 0

        # Performance evaluation
        prompt_tokens = rag_res.get("prompt_tokens", int(len(request.query) / 4))
        completion_tokens = rag_res.get("completion_tokens", int(len(rag_res.get("answer", "")) / 4))
        total_tokens = prompt_tokens + completion_tokens
        cost_est = eval_runner.perf_eval.calculate_cost(prompt_tokens, completion_tokens)

        # What is working, failing, needs improvement
        working = []
        failing = []
        improving = []

        if faithfulness >= 0.85:
            working.append(f"Faithfulness is high ({int(faithfulness*100)}%): Response is strictly grounded in retrieved documentation with no hallucinations.")
        else:
            improving.append(f"Faithfulness is {int(faithfulness*100)}%: Some statements are not directly mentioned in retrieved context.")

        if relevancy >= 0.80:
            working.append(f"Answer Relevancy is high ({int(relevancy*100)}%): Directly answers what you asked.")
        else:
            failing.append(f"Answer Relevancy is low ({int(relevancy*100)}%): The response is off-topic or fails to answer what was asked.")

        working.append("Security Safe: Input cleared LLM security guardrails with no injection attempts.")

        return {
            "query": request.query,
            "answer": rag_res.get("answer", ""),
            "contexts": rag_res.get("contexts", []),
            "quality": {
                "faithfulness": faithfulness,
                "faithfulness_pct": int(faithfulness * 100),
                "answer_relevancy": relevancy,
                "answer_relevancy_pct": int(relevancy * 100),
                "context_precision": precision,
                "context_precision_pct": int(precision * 100),
                "context_recall": recall,
                "context_recall_pct": int(recall * 100),
                "hallucination_detected": hallucination_detected,
                "supported_claims": supported,
                "unsupported_claims": unsupported
            },
            "performance": {
                "response_latency_ms": rag_res.get("total_latency_ms", round(total_ms, 2)),
                "retrieval_latency_ms": rag_res.get("retrieval_latency_ms", 0.0),
                "generation_latency_ms": rag_res.get("generation_latency_ms", 0.0),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(cost_est, 6)
            },
            "security": {
                "is_safe": True,
                "status": "SAFE",
                "evidence": "Input verified safe by LLM Security Guard."
            },
            "what_is_working": working,
            "what_is_failing": failing,
            "what_needs_improvement": improving
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query analysis failed: {str(e)}")

        return {
            "query": request.query,
            "answer": rag_res.get("answer", ""),
            "contexts": rag_res.get("contexts", []),
            "quality": {
                "faithfulness": faithfulness,
                "faithfulness_pct": int(faithfulness * 100),
                "answer_relevancy": relevancy,
                "answer_relevancy_pct": int(relevancy * 100),
                "context_precision": precision,
                "context_precision_pct": int(precision * 100),
                "context_recall": recall,
                "context_recall_pct": int(recall * 100),
                "hallucination_detected": len(unsupported) > 0,
                "supported_claims": supported,
                "unsupported_claims": unsupported
            },
            "performance": {
                "response_latency_ms": rag_res.get("total_latency_ms", round(total_ms, 2)),
                "retrieval_latency_ms": rag_res.get("retrieval_latency_ms", 0.0),
                "generation_latency_ms": rag_res.get("generation_latency_ms", 0.0),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(cost_est, 6)
            },
            "security": {
                "is_safe": sec_audit.passed,
                "status": "BLOCKED (SAFE)" if is_blocked else ("SAFE" if sec_audit.passed else "VULNERABLE"),
                "evidence": sec_audit.evidence or "No security vulnerabilities detected."
            },
            "what_is_working": working,
            "what_is_failing": failing,
            "what_needs_improvement": improving
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query analysis failed: {str(e)}")

@app.post("/api/query")
async def run_query(request: QueryRequest):
    """Execute interactive RAG query with microsecond telemetry."""
    try:
        result = default_rag_engine.query(
            query=request.query,
            search_type=request.search_type,
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@app.post("/api/upload-doc")
async def upload_document(doc: DocUploadRequest):
    """Ingest custom documentation text into the built-in RAG engine."""
    try:
        new_doc = {
            "id": f"doc_{len(SAMPLE_KNOWLEDGE_BASE) + 1}",
            "title": doc.title,
            "category": doc.category,
            "content": doc.content
        }
        # Ingest and reindex
        current_docs = list(SAMPLE_KNOWLEDGE_BASE)
        current_docs.append(new_doc)
        default_rag_engine.initialize_knowledge_base(current_docs)
        return {
            "status": "success",
            "message": f"Successfully indexed '{doc.title}'. Total chunks: {len(default_rag_engine.chunks)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document indexing failed: {str(e)}")

# --- RESUME / DOCUMENT EVALUATION & BENCHMARKING (rageval.dev workflow) ---

class TextDocInput(BaseModel):
    text: str
    title: Optional[str] = "Uploaded Document"

class MultiStrategyBenchmarkRequest(BaseModel):
    document_text: str
    test_cases: Optional[List[Dict[str, Any]]] = None

@app.get("/api/sample-resume")
async def get_sample_resume():
    """Returns pre-loaded sample resume and generated test questions."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    test_cases = None
    generator_type = "Pre-loaded Scenario Matrix"
    if gemini_key:
        try:
            test_cases = await TestsetGenerator.generate_with_llm(SAMPLE_RESUME_TEXT, api_key=gemini_key, provider="gemini")
            if test_cases:
                generator_type = "LLM (Google Gemini Flash via .env)"
        except Exception:
            pass
    if not test_cases:
        test_cases = TestsetGenerator.generate_from_text(SAMPLE_RESUME_TEXT, max_questions=5)
    sections = DocumentParser.extract_sections(SAMPLE_RESUME_TEXT)
    return {
        "title": "Alexander Chen - Senior AI & Distributed Systems Engineer",
        "text": SAMPLE_RESUME_TEXT.strip(),
        "char_count": len(SAMPLE_RESUME_TEXT),
        "sections": list(sections.keys()),
        "test_cases": test_cases,
        "generator_type": generator_type
    }

@app.post("/api/upload-resume-file")
async def upload_resume_file(file: UploadFile = File(...)):
    """Upload resume, read API key from .env, and generate questions using LLM."""
    try:
        content_bytes = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith(".pdf"):
            extracted_text = DocumentParser.parse_pdf(content_bytes)
        else:
            extracted_text = content_bytes.decode("utf-8", errors="ignore")

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from the uploaded file.")

        sections = DocumentParser.extract_sections(extracted_text)
        
        # 1. Check for LLM keys in .env
        test_cases = None
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")

        if gemini_key:
            test_cases = await TestsetGenerator.generate_with_llm(extracted_text, api_key=gemini_key, provider="gemini")
            if test_cases:
                generator_type = "LLM (Google Gemini Flash via .env)"
        elif openai_key:
            test_cases = await TestsetGenerator.generate_with_llm(extracted_text, api_key=openai_key, provider="openai")
            if test_cases:
                generator_type = "LLM (OpenAI GPT-4o-mini via .env)"
        elif groq_key:
            test_cases = await TestsetGenerator.generate_with_llm(extracted_text, api_key=groq_key, provider="groq")
            if test_cases:
                generator_type = "LLM (Groq Llama-3.1 via .env)"
        
        # 2. Dynamic document-specific semantic generator fallback if key not set
        if not test_cases:
            test_cases = TestsetGenerator.generate_from_text(extracted_text, max_questions=5)
            generator_type = "Local Semantic Engine (Set GEMINI_API_KEY in .env for Live LLM)"

        # Index in active engine
        default_rag_engine.initialize_knowledge_base([{
            "id": "uploaded_resume",
            "title": file.filename,
            "content": extracted_text
        }])

        return {
            "status": "success",
            "filename": file.filename,
            "text": extracted_text,
            "char_count": len(extracted_text),
            "sections": list(sections.keys()),
            "test_cases": test_cases,
            "generator_type": generator_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/api/upload-resume-text")
async def upload_resume_text(input_data: TextDocInput):
    """Ingest pasted resume / document text and auto-generate evaluation questions using LLM from .env."""
    try:
        raw_text = input_data.text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        sections = DocumentParser.extract_sections(raw_text)
        
        # 1. Check for LLM keys in .env
        test_cases = None
        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")

        if gemini_key:
            test_cases = await TestsetGenerator.generate_with_llm(raw_text, api_key=gemini_key, provider="gemini")
            if test_cases:
                generator_type = "LLM (Google Gemini Flash via .env)"
        elif openai_key:
            test_cases = await TestsetGenerator.generate_with_llm(raw_text, api_key=openai_key, provider="openai")
            if test_cases:
                generator_type = "LLM (OpenAI GPT-4o-mini via .env)"
        elif groq_key:
            test_cases = await TestsetGenerator.generate_with_llm(raw_text, api_key=groq_key, provider="groq")
            if test_cases:
                generator_type = "LLM (Groq Llama-3.1 via .env)"

        # 2. Dynamic document-specific semantic generator fallback if key not set
        if not test_cases:
            test_cases = TestsetGenerator.generate_from_text(raw_text, max_questions=5)
            generator_type = "Local Semantic Engine (Set GEMINI_API_KEY in .env for Live LLM)"

        # Index in active engine
        default_rag_engine.initialize_knowledge_base([{
            "id": "pasted_doc",
            "title": input_data.title or "Pasted Document",
            "content": raw_text
        }])

        return {
            "status": "success",
            "title": input_data.title,
            "text": raw_text,
            "char_count": len(raw_text),
            "sections": list(sections.keys()),
            "test_cases": test_cases,
            "generator_type": generator_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

@app.post("/api/run-multi-strategy-benchmark")
async def run_multi_strategy_benchmark(req: MultiStrategyBenchmarkRequest):
    """
    Executes parallel strategy benchmarking across:
    1. Strategy A: Naive Dense Vector Search
    2. Strategy B: Hybrid Search (BM25 + Dense RRF)
    3. Strategy C: Fine-Grained Chunking (256-chars)
    Evaluates Quality (RAGAS), Performance, Security, and Failure Attribution.
    """
    try:
        doc_text = req.document_text.strip()
        if not doc_text:
            raise HTTPException(status_code=400, detail="Document text is required.")

        test_cases = req.test_cases
        if not test_cases:
            test_cases = TestsetGenerator.generate_from_text(doc_text, max_questions=5)

        runner = StrategyBenchmarkRunner(raw_text=doc_text, test_cases=test_cases)
        results = runner.execute_all_strategies()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")


def start():
    """Start local web server."""
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=False)

if __name__ == "__main__":
    start()
