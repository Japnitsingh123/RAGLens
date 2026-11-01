"""
LangGraph Parallel Execution Workflow.
Orchestrates parallel benchmarking of the 5 RAG strategies using LangGraph StateGraph:
- Naive Dense RAG
- Hybrid BM25
- Cohere Cross-Encoder Rerank
- HyDE
- Parent-Child Chunk Expansion

All strategies execute concurrently from START, fan into an aggregation and Ragas evaluation node,
and terminate at END.
"""

import asyncio
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from app.core.rag_strategies import (
    execute_naive_rag,
    execute_hybrid_bm25,
    execute_cohere_rerank,
    execute_hyde,
    execute_parent_child
)
from app.core.ragas_evaluator import RagasEvaluator
from app.core.failure_attribution import FailureAttributionEngine
from app.models.schemas import StrategyResult, StrategyType

class BenchmarkGraphState(TypedDict, total=False):
    query: str
    ground_truth: Optional[str]
    top_k: int
    naive_result: Dict[str, Any]
    hybrid_bm25_result: Dict[str, Any]
    cohere_rerank_result: Dict[str, Any]
    hyde_result: Dict[str, Any]
    parent_child_result: Dict[str, Any]
    final_results: List[Dict[str, Any]]
    best_strategy: str
    executive_summary: str

# Strategy Execution Nodes
async def node_naive(state: BenchmarkGraphState) -> Dict[str, Any]:
    res = await execute_naive_rag(state["query"], top_k=state.get("top_k", 3))
    return {"naive_result": res}

async def node_hybrid_bm25(state: BenchmarkGraphState) -> Dict[str, Any]:
    res = await execute_hybrid_bm25(state["query"], top_k=state.get("top_k", 3))
    return {"hybrid_bm25_result": res}

async def node_cohere_rerank(state: BenchmarkGraphState) -> Dict[str, Any]:
    res = await execute_cohere_rerank(state["query"], top_k=state.get("top_k", 3))
    return {"cohere_rerank_result": res}

async def node_hyde(state: BenchmarkGraphState) -> Dict[str, Any]:
    res = await execute_hyde(state["query"], top_k=state.get("top_k", 3))
    return {"hyde_result": res}

async def node_parent_child(state: BenchmarkGraphState) -> Dict[str, Any]:
    res = await execute_parent_child(state["query"], top_k=state.get("top_k", 3))
    return {"parent_child_result": res}

# Aggregation and Ragas Evaluation Node
async def node_evaluate_and_attribute(state: BenchmarkGraphState) -> Dict[str, Any]:
    query = state["query"]
    ground_truth = state.get("ground_truth")

    raw_results = [
        state.get("naive_result"),
        state.get("hybrid_bm25_result"),
        state.get("cohere_rerank_result"),
        state.get("hyde_result"),
        state.get("parent_child_result")
    ]
    raw_results = [r for r in raw_results if r is not None]

    async def evaluate_single_strategy(r):
        strat = r["strategy"]
        contexts = r["contexts"]
        answer = r["answer"]

        # 1. Ragas 4-Metric Evaluation
        ragas_m = await RagasEvaluator.evaluate_ragas_metrics(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )

        # 2. Failure Attribution
        attribution = FailureAttributionEngine.diagnose(
            strategy=strat,
            query=query,
            answer=answer,
            contexts=contexts,
            ragas_metrics=ragas_m,
            ground_truth=ground_truth
        )

        return StrategyResult(
            strategy=StrategyType(strat),
            strategy_name=r["strategy_name"],
            description=r["description"],
            answer=answer,
            contexts=contexts,
            latency_ms=r["total_latency_ms"],
            retrieval_latency_ms=r["retrieval_latency_ms"],
            generation_latency_ms=r["generation_latency_ms"],
            total_tokens=r["tokens"],
            ragas_metrics=ragas_m,
            failure_attribution=attribution
        )

    # Evaluate all 5 strategies concurrently in parallel
    strategy_results: List[StrategyResult] = await asyncio.gather(*[evaluate_single_strategy(r) for r in raw_results])

    best_score = -1.0
    best_strategy_name = "Parent-Child Document Expansion"
    for strat_obj in strategy_results:
        rm = strat_obj.ragas_metrics
        composite_score = rm.ragas_score or ((rm.faithfulness + rm.answer_relevancy + rm.context_precision + rm.context_recall) / 4.0)
        if composite_score > best_score:
            best_score = composite_score
            best_strategy_name = strat_obj.strategy_name

    summary = (
        f"Benchmarked 5 strategies concurrently via LangGraph with BGE-small-en-v1.5 and Neural Cross-Encoder. "
        f"Top performing architecture: '{best_strategy_name}' (Composite Ragas Score: {round(best_score * 100, 1)}%)."
    )

    return {
        "final_results": [s.model_dump() for s in strategy_results],
        "best_strategy": best_strategy_name,
        "executive_summary": summary
    }

def build_langgraph_workflow():
    """Builds the compiled LangGraph StateGraph."""
    workflow = StateGraph(BenchmarkGraphState)

    # Add 5 strategy nodes
    workflow.add_node("run_naive", node_naive)
    workflow.add_node("run_hybrid_bm25", node_hybrid_bm25)
    workflow.add_node("run_cohere_rerank", node_cohere_rerank)
    workflow.add_node("run_hyde", node_hyde)
    workflow.add_node("run_parent_child", node_parent_child)

    # Add evaluation node
    workflow.add_node("evaluate_and_attribute", node_evaluate_and_attribute)

    # Fan out in parallel from START to all 5 strategies
    workflow.add_edge(START, "run_naive")
    workflow.add_edge(START, "run_hybrid_bm25")
    workflow.add_edge(START, "run_cohere_rerank")
    workflow.add_edge(START, "run_hyde")
    workflow.add_edge(START, "run_parent_child")

    # Fan in to evaluation aggregator
    workflow.add_edge("run_naive", "evaluate_and_attribute")
    workflow.add_edge("run_hybrid_bm25", "evaluate_and_attribute")
    workflow.add_edge("run_cohere_rerank", "evaluate_and_attribute")
    workflow.add_edge("run_hyde", "evaluate_and_attribute")
    workflow.add_edge("run_parent_child", "evaluate_and_attribute")

    workflow.add_edge("evaluate_and_attribute", END)
    return workflow.compile()

# Precompile graph for instant invocation
compiled_benchmark_graph = build_langgraph_workflow()

async def execute_parallel_benchmark(query: str, ground_truth: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """
    Executes all 5 strategies in parallel via LangGraph.
    Falls back gracefully to asyncio.gather if graph invocation encounters constraints.
    """
    initial_state: BenchmarkGraphState = {
        "query": query,
        "ground_truth": ground_truth,
        "top_k": top_k
    }

    try:
        final_state = await compiled_benchmark_graph.ainvoke(initial_state)
        return {
            "query": query,
            "ground_truth": ground_truth,
            "strategies_evaluated": len(final_state.get("final_results", [])),
            "execution_mode": "LangGraph Parallel StateGraph",
            "results": final_state.get("final_results", []),
            "best_strategy": final_state.get("best_strategy", "Parent-Child Document Expansion"),
            "executive_summary": final_state.get("executive_summary", "Completed 5-strategy parallel evaluation.")
        }
    except Exception:
        # Fallback to direct concurrent execution
        results = await asyncio.gather(
            execute_naive_rag(query, top_k),
            execute_hybrid_bm25(query, top_k),
            execute_cohere_rerank(query, top_k),
            execute_hyde(query, top_k),
            execute_parent_child(query, top_k)
        )
        fake_state: BenchmarkGraphState = {
            "query": query,
            "ground_truth": ground_truth,
            "naive_result": results[0],
            "hybrid_bm25_result": results[1],
            "cohere_rerank_result": results[2],
            "hyde_result": results[3],
            "parent_child_result": results[4]
        }
        eval_state = await node_evaluate_and_attribute(fake_state)
        return {
            "query": query,
            "ground_truth": ground_truth,
            "strategies_evaluated": 5,
            "execution_mode": "Asyncio Concurrent Parallel Execution",
            "results": eval_state.get("final_results", []),
            "best_strategy": eval_state.get("best_strategy", "Parent-Child Document Expansion"),
            "executive_summary": eval_state.get("executive_summary", "Completed parallel evaluation.")
        }
