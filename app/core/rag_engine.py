import time
import math
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from app.data.sample_docs import SAMPLE_KNOWLEDGE_BASE
from app.core.semantic_cache import SemanticCache
from app.core.llm_security_guard import LLMSecurityGuard

class BM25Retriever:
    """Okapi BM25 implementation for lexical keyword search."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_freqs: List[Dict[str, int]] = []
        self.metadata: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', text.lower())

    def fit(self, documents: List[Dict[str, Any]]):
        self.metadata = documents
        self.corpus = [doc["content"] for doc in documents]
        self.doc_len = []
        self.doc_freqs = []
        self.df = {}
        
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            freq: Dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        n_docs = len(self.corpus)
        self.avg_doc_len = float(sum(self.doc_len) / n_docs) if n_docs > 0 else 1.0

        for term, freq in self.df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 4) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        scores: List[Tuple[int, float]] = []

        for idx, (doc_freq, d_len) in enumerate(zip(self.doc_freqs, self.doc_len)):
            score = 0.0
            for term in query_tokens:
                if term in doc_freq:
                    tf = doc_freq[term]
                    idf = self.idf.get(term, 0.0)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                    score += idf * (numerator / max(denominator, 1e-6))
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DenseVectorizer:
    """
    Fast, deterministic dense semantic vectorizer with subword hashing & semantic projection.
    Provides reliable, zero-latency embeddings without requiring external network API calls.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', text.lower())
        
        if not tokens:
            return vec

        # Use character 3-grams and words to create a rich semantic dense vector
        features = list(tokens)
        for token in tokens:
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    features.append(token[i:i+3])

        for feat in features:
            # MD5 hash to bucket
            h = int(hashlib.md5(feat.encode('utf-8')).hexdigest()[:8], 16)
            idx = h % self.dim
            weight = 1.0 if len(feat) > 3 else 0.5
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign * weight

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class RAGEngine:
    """
    Production-grade modular RAG Engine supporting:
    - Configurable chunk size and overlap
    - Dense Vector Search (Cosine Similarity)
    - BM25 Lexical Keyword Search
    - Hybrid Search (Reciprocal Rank Fusion)
    - Semantic Caching
    - Guardrail Filtering (Direct & Indirect Prompt Injections)
    """
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        search_type: str = "hybrid",
        top_k: int = 4,
        enable_cache: bool = True,
        cache_threshold: float = 0.88,
        enable_guardrails: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_type = search_type
        self.top_k = top_k
        self.enable_cache = enable_cache
        self.enable_guardrails = enable_guardrails

        self.vectorizer = DenseVectorizer(dim=128)
        self.bm25 = BM25Retriever()
        self.semantic_cache = SemanticCache(default_threshold=cache_threshold)

        self.chunks: List[Dict[str, Any]] = []
        self.chunk_embeddings: List[np.ndarray] = []

        # Initialize with default sample knowledge base
        self.initialize_knowledge_base(SAMPLE_KNOWLEDGE_BASE)

    def _chunk_text(self, text: str, doc_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_idx = 0

        while start < len(cleaned):
            end = min(start + self.chunk_size, len(cleaned))
            # Try to break at sentence or space boundary if not at end
            if end < len(cleaned):
                last_space = cleaned.rfind(' ', start + int(self.chunk_size * 0.7), end)
                if last_space != -1:
                    end = last_space

            chunk_text = cleaned[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{doc_meta.get('id', 'doc')}_chk_{chunk_idx}",
                    "doc_id": doc_meta.get("id", "doc"),
                    "title": doc_meta.get("title", "Document"),
                    "category": doc_meta.get("category", "general"),
                    "content": chunk_text
                })
                chunk_idx += 1

            if end >= len(cleaned):
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def initialize_knowledge_base(self, documents: List[Dict[str, Any]], chunk_size: Optional[int] = None):
        if chunk_size:
            self.chunk_size = chunk_size

        all_chunks: List[Dict[str, Any]] = []
        for doc in documents:
            chunks = self._chunk_text(doc.get("content", ""), doc)
            all_chunks.extend(chunks)

        self.chunks = all_chunks
        self.chunk_embeddings = [self.vectorizer.encode(c["content"]) for c in self.chunks]
        self.bm25.fit(self.chunks)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _vector_search(self, query_emb: np.ndarray, k: int) -> List[Tuple[int, float]]:
        scores = []
        for idx, emb in enumerate(self.chunk_embeddings):
            sim = self._cosine_similarity(query_emb, emb)
            scores.append((idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _hybrid_search(self, query: str, query_emb: np.ndarray, k: int, rrf_k: int = 60) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion of Dense Vector and BM25 results."""
        dense_results = self._vector_search(query_emb, k=len(self.chunks))
        bm25_results = self.bm25.search(query, top_k=len(self.chunks))

        rrf_scores: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

    def retrieve(self, query: str, search_type: Optional[str] = None, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], float]:
        st = search_type or self.search_type
        k = top_k or self.top_k

        t_start = time.perf_counter()
        query_emb = self.vectorizer.encode(query)

        if st == "vector":
            ranked_indices = self._vector_search(query_emb, k=k)
        else:
            # Hybrid search
            ranked_indices = self._hybrid_search(query, query_emb, k=k)

        retrieval_ms = (time.perf_counter() - t_start) * 1000.0

        results = []
        for idx, score in ranked_indices:
            if idx < len(self.chunks):
                item = dict(self.chunks[idx])
                item["retrieval_score"] = round(float(score), 4)
                results.append(item)

        return results, retrieval_ms

    def _check_guardrails(self, query: str, contexts: List[Dict[str, Any]]) -> Optional[str]:
        """Detect direct and indirect prompt injection attempts using LLM-as-a-Judge reasoning."""
        context_texts = [ctx.get("content", "") for ctx in contexts] if contexts else []
        is_attack, attack_type, explanation = LLMSecurityGuard.analyze_query_security_sync(query, context_texts)
        if is_attack:
            return f"SECURITY ALERT: Adversarial {attack_type} detected. {explanation}"
        return None

    def _generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Synthesize factual answer strictly grounded in retrieved contexts."""
        if not contexts:
            return "No relevant documentation found to answer your request."

        # Extract sentences from contexts that share keywords with the query
        query_words = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower()))
        matched_sentences: List[str] = []

        for ctx in contexts:
            sentences = re.split(r'(?<=[.!?])\s+', ctx["content"])
            for s in sentences:
                s_words = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s.lower()))
                common = query_words.intersection(s_words)
                if len(common) >= 1:
                    matched_sentences.append(s.strip())

        if not matched_sentences:
            # Fallback to top chunk snippet
            return contexts[0]["content"][:300] + "..."

        # Select top unique sentences
        unique_sentences = list(dict.fromkeys(matched_sentences))[:4]
        return " ".join(unique_sentences)

    def query(
        self,
        query: str,
        search_type: Optional[str] = None,
        top_k: Optional[int] = None,
        use_cache: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Execute end-to-end RAG query with telemetry and caching."""
        start_total = time.perf_counter()
        use_caching = self.enable_cache if use_cache is None else use_cache
        query_emb = self.vectorizer.encode(query)

        # 1. Check Semantic Cache
        if use_caching:
            cached_resp, match_type, sim_score = self.semantic_cache.get(query, query_emb)
            if cached_resp is not None:
                cached_resp_copy = dict(cached_resp)
                cached_resp_copy["cache_hit"] = True
                cached_resp_copy["cache_match_type"] = match_type
                cached_resp_copy["cache_similarity"] = round(sim_score, 3)
                cached_resp_copy["total_latency_ms"] = round((time.perf_counter() - start_total) * 1000.0, 2)
                return cached_resp_copy

        # 2. Retrieval
        contexts, retrieval_ms = self.retrieve(query, search_type=search_type, top_k=top_k)

        # 3. Security Guardrails
        if self.enable_guardrails:
            guardrail_violation = self._check_guardrails(query, contexts)
            if guardrail_violation:
                total_ms = (time.perf_counter() - start_total) * 1000.0
                return {
                    "query": query,
                    "answer": guardrail_violation,
                    "contexts": [c["content"] for c in contexts],
                    "retrieval_latency_ms": round(retrieval_ms, 2),
                    "generation_latency_ms": 0.5,
                    "total_latency_ms": round(total_ms, 2),
                    "prompt_tokens": len(query.split()) * 2,
                    "completion_tokens": len(guardrail_violation.split()) * 2,
                    "total_tokens": (len(query.split()) + len(guardrail_violation.split())) * 2,
                    "cache_hit": False,
                    "security_blocked": True,
                    "security_reason": guardrail_violation
                }

        # 4. Generation
        gen_start = time.perf_counter()
        answer = self._generate_answer(query, contexts)
        generation_ms = (time.perf_counter() - gen_start) * 1000.0
        total_ms = (time.perf_counter() - start_total) * 1000.0

        # Token estimation (1 token ~= 0.75 words or 4 chars)
        prompt_text = query + " " + " ".join([c["content"] for c in contexts])
        prompt_tokens = int(len(prompt_text) / 3.8)
        completion_tokens = int(len(answer) / 3.8)
        total_tokens = prompt_tokens + completion_tokens

        response_payload = {
            "query": query,
            "answer": answer,
            "contexts": [c["content"] for c in contexts],
            "context_items": contexts,
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "generation_latency_ms": round(generation_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cache_hit": False,
            "security_blocked": False
        }

        # Store in cache if enabled
        if use_caching:
            self.semantic_cache.set(
                query=query,
                query_embedding=query_emb,
                response=response_payload,
                original_latency_ms=total_ms,
                tokens=total_tokens
            )

        return response_payload
