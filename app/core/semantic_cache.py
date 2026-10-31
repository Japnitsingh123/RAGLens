import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class SemanticCache:
    """
    In-memory semantic cache with cosine similarity matching,
    LRU eviction, and operational telemetry.
    """
    def __init__(self, max_size: int = 200, default_threshold: float = 0.88):
        self.max_size = max_size
        self.default_threshold = default_threshold
        # Stores: query_hash -> { 'query': str, 'embedding': np.ndarray, 'response': Dict[str, Any], 'timestamp': float, 'hits': int }
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Telemetry
        self.total_lookups = 0
        self.hits = 0
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.estimated_latency_saved_ms = 0.0
        self.estimated_tokens_saved = 0

    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def get(self, query: str, query_embedding: np.ndarray, threshold: Optional[float] = None) -> Tuple[Optional[Dict[str, Any]], str, float]:
        """
        Lookup a query in the cache.
        Returns (response_data, match_type, similarity_score).
        match_type: 'exact', 'semantic', or 'miss'
        """
        self.total_lookups += 1
        thresh = threshold if threshold is not None else self.default_threshold
        normalized_q = self._normalize(query)

        # 1. Exact match check
        for key, entry in self._cache.items():
            if self._normalize(entry["query"]) == normalized_q:
                self.hits += 1
                self.exact_hits += 1
                entry["hits"] += 1
                entry["last_accessed"] = time.time()
                self.estimated_latency_saved_ms += entry.get("original_latency_ms", 180.0)
                self.estimated_tokens_saved += entry.get("tokens", 250)
                return entry["response"], "exact", 1.0

        # 2. Semantic vector similarity check
        best_match_entry = None
        best_sim = -1.0

        for key, entry in self._cache.items():
            cached_emb = entry["embedding"]
            sim = self._cosine_similarity(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_match_entry = entry

        if best_match_entry is not None and best_sim >= thresh:
            self.hits += 1
            self.semantic_hits += 1
            best_match_entry["hits"] += 1
            best_match_entry["last_accessed"] = time.time()
            self.estimated_latency_saved_ms += best_match_entry.get("original_latency_ms", 180.0)
            self.estimated_tokens_saved += best_match_entry.get("tokens", 250)
            return best_match_entry["response"], "semantic", best_sim

        # 3. Cache Miss
        self.misses += 1
        return None, "miss", max(0.0, best_sim)

    def set(self, query: str, query_embedding: np.ndarray, response: Dict[str, Any], original_latency_ms: float = 200.0, tokens: int = 250):
        """Store an entry in the cache with LRU eviction."""
        if len(self._cache) >= self.max_size:
            # Evict least recently accessed
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].get("last_accessed", 0))
            del self._cache[oldest_key]

        cache_key = f"k_{len(self._cache)}_{int(time.time() * 1000)}"
        self._cache[cache_key] = {
            "query": query,
            "embedding": query_embedding,
            "response": response,
            "timestamp": time.time(),
            "last_accessed": time.time(),
            "hits": 0,
            "original_latency_ms": original_latency_ms,
            "tokens": tokens
        }

    def get_stats(self) -> Dict[str, Any]:
        hit_rate = (self.hits / self.total_lookups * 100.0) if self.total_lookups > 0 else 0.0
        return {
            "total_lookups": self.total_lookups,
            "hits": self.hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 1),
            "cached_entries_count": len(self._cache),
            "estimated_latency_saved_ms": round(self.estimated_latency_saved_ms, 1),
            "estimated_tokens_saved": self.estimated_tokens_saved
        }

    def clear(self):
        self._cache.clear()
        self.total_lookups = 0
        self.hits = 0
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.estimated_latency_saved_ms = 0.0
        self.estimated_tokens_saved = 0
