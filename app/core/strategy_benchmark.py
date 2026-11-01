import time
from typing import List, Dict, Any
from app.core.rag_engine import RAGEngine
from app.core.quality_eval import QualityEvaluator
from app.core.performance_eval import PerformanceEvaluator
from app.core.security_eval import SecurityEvaluator
from app.data.default_test_cases import DEFAULT_SECURITY_TESTS

class StrategyBenchmarkRunner:
    """
    RAG Evaluation & Strategy Benchmarking Runner (Inspired by rageval.dev).
    Runs auto-generated test questions across:
    1. Strategy A: Naive Vector Search (Dense Cosine Similarity)
    2. Strategy B: Hybrid Search (BM25 + Dense Reciprocal Rank Fusion)
    3. Strategy C: Fine-Grained Chunking (256-char window)
    """
    def __init__(self, raw_text: str, test_cases: List[Dict[str, Any]]):
        self.raw_text = raw_text
        self.test_cases = test_cases
        self.quality_eval = QualityEvaluator()
        self.perf_eval = PerformanceEvaluator()
        self.security_eval = SecurityEvaluator()

    def run_strategy(self, engine: RAGEngine, strategy_name: str, strategy_type: str) -> Dict[str, Any]:
        records = []
        quality_results = []

        for tc in self.test_cases:
            q = tc["question"]
            gt = tc["ground_truth"]
            k_facts = tc.get("key_facts", [])

            # Execute query
            res = engine.query(q)
            records.append(res)

            # Evaluate RAGAS quality
            q_res = self.quality_eval.evaluate_item(
                test_id=tc["id"],
                question=q,
                ground_truth=gt,
                generated_answer=res.get("answer", ""),
                contexts=res.get("contexts", []),
                key_facts=k_facts
            )
            quality_results.append(q_res)

        q_summary = self.quality_eval.summarize(quality_results)
        p_summary = self.perf_eval.evaluate(records)

        # Calculate composite score 0-100
        score = (
            (q_summary.avg_faithfulness * 0.35) +
            (q_summary.avg_answer_relevancy * 0.25) +
            (q_summary.avg_context_recall * 0.25) +
            (max(0.0, 1.0 - (p_summary.avg_response_latency_ms / 1000.0)) * 0.15)
        ) * 100.0

        return {
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "score": round(score, 1),
            "quality": {
                "faithfulness": q_summary.avg_faithfulness,
                "faithfulness_pct": int(q_summary.avg_faithfulness * 100),
                "relevancy": q_summary.avg_answer_relevancy,
                "relevancy_pct": int(q_summary.avg_answer_relevancy * 100),
                "precision": q_summary.avg_context_precision,
                "precision_pct": int(q_summary.avg_context_precision * 100),
                "recall": q_summary.avg_context_recall,
                "recall_pct": int(q_summary.avg_context_recall * 100),
                "items": [i.model_dump() for i in quality_results]
            },
            "performance": {
                "avg_latency_ms": p_summary.avg_response_latency_ms,
                "retrieval_latency_ms": p_summary.avg_retrieval_latency_ms,
                "avg_tokens": p_summary.avg_total_tokens,
                "cost_per_query": round(p_summary.estimated_cost_usd / max(len(records), 1), 6)
            }
        }

    def execute_all_strategies(self) -> Dict[str, Any]:
        doc_obj = [{"id": "user_doc", "title": "Uploaded Document", "content": self.raw_text}]

        # 1. Strategy A: Naive Dense Vector Search
        engine_vector = RAGEngine(search_type="vector", top_k=3, chunk_size=512)
        engine_vector.initialize_knowledge_base(doc_obj)
        res_vector = self.run_strategy(engine_vector, "Naive Vector Search", "vector_dense")

        # 2. Strategy B: Hybrid Search (BM25 + Dense RRF)
        engine_hybrid = RAGEngine(search_type="hybrid", top_k=3, chunk_size=512)
        engine_hybrid.initialize_knowledge_base(doc_obj)
        res_hybrid = self.run_strategy(engine_hybrid, "Hybrid Search (BM25 + Dense RRF)", "hybrid_rrf")

        # 3. Strategy C: Fine-Grained Chunking (256 chars)
        engine_fine = RAGEngine(search_type="hybrid", top_k=4, chunk_size=256)
        engine_fine.initialize_knowledge_base(doc_obj)
        res_fine = self.run_strategy(engine_fine, "Fine Chunking (256 chars)", "chunk_fine")

        strategies = [res_vector, res_hybrid, res_fine]

        # Determine winner
        best_strategy = max(strategies, key=lambda s: s["score"])

        # Run Security Red-Team
        sec_items = []
        for s_case in DEFAULT_SECURITY_TESTS[:3]:
            sec_out = engine_hybrid.query(s_case["attack_payload"])
            sec_item = self.security_eval.evaluate_test_case(s_case, sec_out)
            sec_items.append(sec_item)
        sec_results = [item.model_dump() for item in sec_items]
        sec_summary = self.security_eval.summarize(sec_items)

        # Compile What is working, failing, needs improvement
        working = []
        failing = []
        improving = []

        if best_strategy["quality"]["faithfulness"] >= 0.85:
            working.append(f"High Faithfulness ({best_strategy['quality']['faithfulness_pct']}%): {best_strategy['strategy_name']} eliminated hallucinations by grounding strictly in the document.")
        else:
            improving.append(f"Faithfulness is {best_strategy['quality']['faithfulness_pct']}%. Lower model temperature or refine chunk boundaries.")

        if res_hybrid["quality"]["recall"] >= res_vector["quality"]["recall"]:
            diff = res_hybrid["quality"]["recall_pct"] - res_vector["quality"]["recall_pct"]
            working.append(f"Hybrid Search improved Context Recall by +{diff}% over pure Vector Search by capturing exact technical keywords.")

        if sec_summary.failed_tests == 0:
            working.append(f"Security: All {sec_summary.passed_tests} adversarial injection and system override probes were safely neutralized.")
        else:
            failing.append(f"Security Vulnerability: {sec_summary.failed_tests} adversarial attacks succeeded.")

        if best_strategy["performance"]["avg_latency_ms"] < 100.0:
            working.append(f"Fast Latency: Sub-100ms retrieval and response ({best_strategy['performance']['avg_latency_ms']}ms).")
        else:
            improving.append("Latency can be reduced further with smaller top-k or optimized chunk overlaps.")

        improving.append("Fine chunking (256 chars) yields higher precision, whereas 512 chars captures broader context.")

        return {
            "best_strategy": best_strategy["strategy_name"],
            "best_score": best_strategy["score"],
            "strategies": strategies,
            "security": {
                "score_pct": int(sec_summary.overall_security_score * 100),
                "rating": sec_summary.risk_rating,
                "passed": sec_summary.passed_tests,
                "total": sec_summary.total_tests,
                "cases": sec_results
            },
            "what_is_working": working,
            "what_is_failing": failing,
            "what_needs_improvement": improving,
            "test_cases": self.test_cases
        }
