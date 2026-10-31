import time
import httpx
from typing import Dict, Any, List, Tuple
from app.models.schemas import PipelineConfig

class ExternalRAGConnector:
    """
    Connects to any external developer RAG REST API to benchmark and evaluate it.
    """
    def __init__(self, config: PipelineConfig):
        self.config = config

    async def query(self, query: str) -> Dict[str, Any]:
        if not self.config.external_url:
            raise ValueError("External URL must be provided for external mode.")

        headers = self.config.external_headers or {}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        # Build payload
        if self.config.request_template:
            payload = {}
            for k, v in self.config.request_template.items():
                if isinstance(v, str) and "{query}" in v:
                    payload[k] = v.replace("{query}", query)
                else:
                    payload[k] = v
        else:
            payload = {"query": query, "question": query}

        start_time = time.perf_counter()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.config.external_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract answer
        answer_key = self.config.response_answer_key or "answer"
        answer = data.get(answer_key, "")
        if not answer and "response" in data:
            answer = data["response"]
        elif not answer and "output" in data:
            answer = data["output"]

        # Extract contexts
        contexts_key = self.config.response_contexts_key or "contexts"
        raw_contexts = data.get(contexts_key, [])
        contexts: List[str] = []
        if isinstance(raw_contexts, list):
            for item in raw_contexts:
                if isinstance(item, str):
                    contexts.append(item)
                elif isinstance(item, dict):
                    contexts.append(item.get("content") or item.get("text") or str(item))

        # Estimate tokens
        prompt_tokens = int((len(query) + sum(len(c) for c in contexts)) / 3.8)
        completion_tokens = int(len(answer) / 3.8)

        return {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "retrieval_latency_ms": round(total_latency_ms * 0.4, 2),  # Estimated split if not provided
            "generation_latency_ms": round(total_latency_ms * 0.6, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cache_hit": False,
            "security_blocked": False
        }
