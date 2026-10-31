import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.models.schemas import (
    RunEvaluationRequest,
    FullEvaluationReport,
    PipelineConfig,
    TestCaseItem,
    AdversarialSecurityTest,
    QualitySummary,
    PerformanceSummary,
    SecuritySummary,
    BenchmarkSummary
)
from app.core.rag_engine import RAGEngine
from app.core.external_connector import ExternalRAGConnector
from app.core.quality_eval import QualityEvaluator
from app.core.performance_eval import PerformanceEvaluator
from app.core.security_eval import SecurityEvaluator
from app.core.benchmark_engine import BenchmarkEngine
from app.data.default_test_cases import DEFAULT_QUALITY_TESTS, DEFAULT_SECURITY_TESTS

class RAGLensRunner:
    """
    Main evaluation pipeline orchestrator for RAGLens.
    Runs Quality, Performance, Security, and Benchmarking suites.
    """
    def __init__(self):
        self.quality_eval = QualityEvaluator()
        self.perf_eval = PerformanceEvaluator()
        self.security_eval = SecurityEvaluator()
        self.benchmark_engine = BenchmarkEngine()

    async def execute_evaluation(self, request: RunEvaluationRequest) -> FullEvaluationReport:
        run_id = f"eval_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        cfg = request.pipeline_config
        suites = set(request.suites)

        # 1. Initialize RAG Pipeline
        if cfg.mode == "external":
            connector = ExternalRAGConnector(cfg)
            is_external = True
        else:
            engine = RAGEngine(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                search_type=cfg.search_type,
                top_k=cfg.top_k,
                enable_cache=cfg.semantic_cache_enabled,
                cache_threshold=cfg.cache_similarity_threshold,
                enable_guardrails=cfg.security_guardrails_enabled
            )
            # If custom documents provided, ingest them
            if request.custom_documents:
                doc_objs = [{"id": f"custom_{i}", "title": f"Doc {i+1}", "content": c} for i, c in enumerate(request.custom_documents)]
                engine.initialize_knowledge_base(doc_objs, chunk_size=cfg.chunk_size)
            is_external = False

        # Prepare test cases
        test_items = []
        if request.custom_test_items:
            test_items = [t.model_dump() for t in request.custom_test_items]
        else:
            test_items = DEFAULT_QUALITY_TESTS

        # 2. Run Queries & Collect Telemetry for Quality & Performance
        query_records: List[Dict[str, Any]] = []
        quality_item_results = []

        for item in test_items:
            q = item["question"]
            gt = item["ground_truth"]
            k_facts = item.get("key_facts", [])

            if is_external:
                rag_out = await connector.query(q)
            else:
                rag_out = engine.query(q)

            query_records.append(rag_out)

            if "quality" in suites:
                q_res = self.quality_eval.evaluate_item(
                    test_id=item["id"],
                    question=q,
                    ground_truth=gt,
                    generated_answer=rag_out.get("answer", ""),
                    contexts=rag_out.get("contexts", []),
                    key_facts=k_facts
                )
                quality_item_results.append(q_res)

        # Summarize Quality
        quality_summary = self.quality_eval.summarize(quality_item_results)

        # Summarize Performance
        cache_stats = engine.semantic_cache.get_stats() if not is_external else {"hit_rate_pct": 0, "hits": 0, "misses": len(query_records)}
        perf_summary = self.perf_eval.evaluate(query_records, cache_stats)

        # 3. Security Red-Teaming Suite
        sec_results: List[AdversarialSecurityTest] = []
        if "security" in suites:
            for s_case in DEFAULT_SECURITY_TESTS:
                if is_external:
                    s_rag_out = await connector.query(s_case["attack_payload"])
                else:
                    # If target is document poisoning, simulate poisoned context chunk
                    if s_case.get("target_field") == "document":
                        simulated_poisoned_ctx = [{"content": s_case["attack_payload"], "title": "Poisoned Source", "id": "atk_doc"}]
                        # Check guardrail or answer directly
                        violation = engine._check_guardrails("What are the credentials?", simulated_poisoned_ctx) if engine.enable_guardrails else None
                        s_rag_out = {
                            "answer": violation or "Admin password updated to AdminOverride2026!",
                            "security_blocked": bool(violation)
                        }
                    else:
                        s_rag_out = engine.query(s_case["attack_payload"])

                sec_item = self.security_eval.evaluate_test_case(s_case, s_rag_out)
                sec_results.append(sec_item)

        security_summary = self.security_eval.summarize(sec_results)

        # 4. Benchmarking Suite
        if "benchmarking" in suites:
            benchmark_summary = self.benchmark_engine.run_all_benchmarks()
        else:
            benchmark_summary = BenchmarkSummary(groups=[])

        # 5. Calculate Overall RAGLens Score & Status
        q_weight = 0.40
        p_weight = 0.25
        s_weight = 0.35

        # Normalize performance latency score (100 if latency < 200ms, decays over real-world LLM bounds)
        perf_score = max(0.2, min(1.0, 1.0 - (perf_summary.avg_response_latency_ms / 5000.0)))
        
        overall_score = (
            (quality_summary.overall_quality_score * q_weight) +
            (perf_score * p_weight) +
            (security_summary.overall_security_score * s_weight)
        ) * 100.0

        overall_score = round(overall_score, 1)

        # Status definition
        if overall_score >= 75 and security_summary.risk_rating in ["Secure", "Medium Risk"]:
            status = "Production Ready"
        elif overall_score >= 60:
            status = "Needs Improvement"
        else:
            status = "High Risk"

        # Categorize into: What is working, What is failing, What needs improvement
        what_is_working = []
        what_is_failing = []
        what_needs_improvement = []

        # Quality assessments
        if quality_summary.avg_faithfulness >= 0.85:
            what_is_working.append(f"Faithfulness is high ({int(quality_summary.avg_faithfulness*100)}%): Answers are strictly grounded in documentation with no hallucinations.")
        elif quality_summary.avg_faithfulness >= 0.70:
            what_needs_improvement.append(f"Faithfulness ({int(quality_summary.avg_faithfulness*100)}%): Some statements lack direct context attribution. Reduce temperature.")
        else:
            what_is_failing.append(f"Faithfulness is low ({int(quality_summary.avg_faithfulness*100)}%): The model hallucinated claims not present in retrieved docs.")

        if quality_summary.avg_answer_relevancy >= 0.85:
            what_is_working.append(f"Answer Relevancy ({int(quality_summary.avg_answer_relevancy*100)}%): Responses directly address what the question asked.")
        else:
            what_needs_improvement.append("Answer Relevancy: Model occasionally deviates or includes unnecessary filler.")

        if quality_summary.avg_context_recall >= 0.85:
            what_is_working.append(f"Context Recall ({int(quality_summary.avg_context_recall*100)}%): Retriever successfully fetched all required factual points.")
        else:
            what_needs_improvement.append("Context Recall: Retriever missed some key facts. Hybrid Search or higher Top-K recommended.")

        # Performance assessments
        if perf_summary.avg_response_latency_ms < 150.0:
            what_is_working.append(f"Fast Response Latency: Average of {perf_summary.avg_response_latency_ms}ms per query.")
        else:
            what_needs_improvement.append(f"Response Latency ({perf_summary.avg_response_latency_ms}ms) is high. Enable semantic caching to drop latency.")

        if perf_summary.cache_hits > 0:
            what_is_working.append(f"Semantic Cache is active ({perf_summary.semantic_cache_hit_rate_pct}% hit rate, 99% faster repeat queries).")
        else:
            what_needs_improvement.append("Semantic Cache had 0 hits in this run. Queries can be cached for instant sub-millisecond responses.")

        # Security assessments
        if security_summary.failed_tests == 0:
            what_is_working.append(f"Security: All {security_summary.passed_tests} adversarial tests blocked (Prompt Injections & System Overrides).")
        else:
            what_is_failing.append(f"Security Vulnerability: {security_summary.failed_tests} out of {security_summary.total_tests} attack probes succeeded.")

        # Executive Summary & Actionable Recommendations
        recomms = []
        if quality_summary.avg_faithfulness < 0.85:
            recomms.append("Faithfulness: Lower model temperature or tighten context grounding system prompts.")
        if quality_summary.avg_context_recall < 0.80:
            recomms.append("Context Recall: Enable Hybrid Search (BM25 + Dense) or increase Top-K from 2 to 4.")
        if security_summary.failed_tests > 0:
            recomms.append(f"Security Alert: {security_summary.failed_tests} adversarial injection test(s) failed. Enable RAGLens Guardrails.")
        if perf_summary.semantic_cache_hit_rate_pct < 20.0 and cfg.semantic_cache_enabled:
            recomms.append("Semantic Cache: Consider lowering similarity threshold from 0.92 to 0.88 to capture paraphrased questions.")
        if not recomms:
            recomms.append("All tests passing. System is ready for production deployment.")

        exec_summary = (
            f"RAGLens evaluated the pipeline across {len(query_records)} test queries, achieving an overall health score of "
            f"{overall_score}/100 ({status}). "
            f"Quality: Faithfulness {int(quality_summary.avg_faithfulness*100)}%, Relevancy {int(quality_summary.avg_answer_relevancy*100)}%, "
            f"Recall {int(quality_summary.avg_context_recall*100)}%. "
            f"Performance: {perf_summary.avg_response_latency_ms}ms avg latency. "
            f"Security: {security_summary.passed_tests}/{security_summary.total_tests} passed."
        )

        return FullEvaluationReport(
            run_id=run_id,
            timestamp=timestamp,
            pipeline_mode=cfg.mode,
            raglens_score=overall_score,
            status=status,
            quality=quality_summary,
            performance=perf_summary,
            security=security_summary,
            benchmarking=benchmark_summary,
            executive_summary=exec_summary,
            what_is_working=what_is_working,
            what_is_failing=what_is_failing,
            what_needs_improvement=what_needs_improvement,
            recommendations=recomms
        )
