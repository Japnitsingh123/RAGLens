from typing import List, Dict, Any, Optional
from app.models.schemas import PerformanceSummary

# Model pricing in USD per 1 Million Tokens (Prompt, Completion)
MODEL_PRICING = {
    "gpt-4o": {"prompt": 2.50, "completion": 10.00, "embedding": 0.02},
    "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00, "embedding": 0.02},
    "gemini-1-5-pro": {"prompt": 1.25, "completion": 5.00, "embedding": 0.02},
    "llama-3-70b": {"prompt": 0.79, "completion": 0.79, "embedding": 0.01},
    "default": {"prompt": 2.00, "completion": 8.00, "embedding": 0.02}
}

class PerformanceEvaluator:
    """
    Performance Testing Suite:
    - Response latency & breakdown (retrieval vs generation)
    - Token consumption
    - Transparent cost per query calculation
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.pricing = MODEL_PRICING.get(model_name.lower(), MODEL_PRICING["default"])

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost_prompt = (prompt_tokens / 1_000_000.0) * self.pricing["prompt"]
        cost_comp = (completion_tokens / 1_000_000.0) * self.pricing["completion"]
        # Add small embedding cost
        cost_emb = (prompt_tokens / 1_000_000.0) * self.pricing["embedding"]
        return cost_prompt + cost_comp + cost_emb

    def evaluate(self, run_records: List[Dict[str, Any]]) -> PerformanceSummary:
        if not run_records:
            return PerformanceSummary(
                avg_response_latency_ms=0.0,
                avg_retrieval_latency_ms=0.0,
                avg_generation_latency_ms=0.0,
                avg_prompt_tokens=0.0,
                avg_completion_tokens=0.0,
                avg_total_tokens=0.0,
                total_tokens_consumed=0,
                estimated_cost_usd=0.0,
                estimated_cost_per_1k_queries=0.0,
                latency_breakdown={"retrieval": 0.0, "generation": 0.0, "overhead": 0.0}
            )

        n = len(run_records)
        total_resp_latency = sum(r.get("total_latency_ms", 0.0) for r in run_records)
        total_retrieval_latency = sum(r.get("retrieval_latency_ms", 0.0) for r in run_records)
        total_gen_latency = sum(r.get("generation_latency_ms", 0.0) for r in run_records)

        total_prompt_tok = sum(r.get("prompt_tokens", 0) for r in run_records)
        total_comp_tok = sum(r.get("completion_tokens", 0) for r in run_records)
        total_tok = total_prompt_tok + total_comp_tok

        total_cost = self.calculate_cost(total_prompt_tok, total_comp_tok)
        cost_per_1k = (total_cost / n) * 1000.0 if n > 0 else 0.0

        avg_resp = total_resp_latency / n
        avg_ret = total_retrieval_latency / n
        avg_gen = total_gen_latency / n
        avg_overhead = max(0.0, avg_resp - (avg_ret + avg_gen))

        return PerformanceSummary(
            avg_response_latency_ms=round(avg_resp, 2),
            avg_retrieval_latency_ms=round(avg_ret, 2),
            avg_generation_latency_ms=round(avg_gen, 2),
            avg_prompt_tokens=round(total_prompt_tok / n, 1),
            avg_completion_tokens=round(total_comp_tok / n, 1),
            avg_total_tokens=round(total_tok / n, 1),
            total_tokens_consumed=total_tok,
            estimated_cost_usd=round(total_cost, 6),
            estimated_cost_per_1k_queries=round(cost_per_1k, 4),
            latency_breakdown={
                "retrieval_ms": round(avg_ret, 2),
                "generation_ms": round(avg_gen, 2),
                "system_overhead_ms": round(avg_overhead, 2)
            }
        )
