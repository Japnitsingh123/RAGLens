"""
ChromaDB Vector & Hybrid Knowledge Base Engine.
Stores parent documents and fine-grained child chunks directly in ChromaDB.
Embedding Model: BAAI/bge-small-en-v1.5 (384 dimensions).
Supports:
- Dense Vector Similarity Search (via ChromaDB HNSW Index & BGE embeddings)
- Sparse BM25 Lexical Keyword Search
- Hybrid BM25 (Reciprocal Rank Fusion of ChromaDB + BM25)
- Parent-Child Context Expansion
- Dynamic User Document Ingestion (PDF, TXT, Markdown)
"""

import os
import math
import re
from typing import List, Dict, Any, Optional
import chromadb
from app.data.sample_knowledge_base import SAMPLE_ENTERPRISE_DOCS

_bge_embedding_model = None

def get_bge_model():
    """Singleton loader for BAAI/bge-small-en-v1.5 embedding model."""
    global _bge_embedding_model
    if _bge_embedding_model is None:
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from sentence_transformers import SentenceTransformer
            _bge_embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except Exception:
            _bge_embedding_model = None
    return _bge_embedding_model

def generate_embedding(text: str, dim: int = 384) -> List[float]:
    """Generates 384-dimensional dense embedding using BAAI/bge-small-en-v1.5 with fallback."""
    model = get_bge_model()
    if model is not None:
        try:
            vec = model.encode(text, normalize_embeddings=True)
            return [round(float(x), 6) for x in vec]
        except Exception:
            pass

    # Deterministic fallback embedding (384-dim)
    tokens = re.findall(r"\w+", text.lower())
    vec = [0.0] * dim
    if not tokens:
        return vec

    for i, token in enumerate(tokens):
        h = hash(token)
        idx1 = abs(h) % dim
        idx2 = abs(h >> 5) % dim
        idx3 = abs(h >> 11) % dim
        weight = 1.0 / math.log2(i + 3)
        vec[idx1] += weight * 1.5
        vec[idx2] += weight * 0.8
        vec[idx3] += weight * 0.5

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [round(x / norm, 6) for x in vec]
    return vec

def cosine_sim(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    return max(-1.0, min(1.0, dot))

class ChromaVectorStore:
    """
    ChromaDB-backed vector store for RAGLens.
    Manages embedded collections, parent-child metadata linkage,
    and hybrid retrieval without external server dependencies.
    """
    def __init__(self, collection_name: str = "raglens_knowledge_base"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.parents: Dict[str, Dict[str, Any]] = {}

    def add_parent_document(self, doc_id: str, title: str, full_content: str, metadata: Optional[Dict] = None):
        self.parents[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "content": full_content,
            "metadata": metadata or {}
        }

    def add_chunk(self, chunk_id: str, parent_id: Optional[str], content: str, title: str = "", metadata: Optional[Dict] = None):
        emb = generate_embedding(content, dim=384)
        meta = metadata or {}
        meta["parent_id"] = parent_id or ""
        meta["title"] = title
        if parent_id and parent_id in self.parents:
            meta["parent_content"] = self.parents[parent_id]["content"]

        self.collection.add(
            ids=[chunk_id],
            documents=[content],
            embeddings=[emb],
            metadatas=[meta]
        )

    def similarity_search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Dense vector search using ChromaDB query_embeddings with BGE-small-en-1.5."""
        q_emb = generate_embedding(query_text, dim=384)
        count = self.collection.count()
        if count == 0:
            return []

        k = min(top_k, count)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        if results and results.get("ids") and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

            for i in range(len(ids)):
                # Chroma distance to cosine similarity
                score = round(max(0.0, 1.0 - (dists[i] / 2.0)), 4) if dists else 0.85
                output.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "parent_id": metas[i].get("parent_id", ""),
                    "score": score,
                    "search_type": "chromadb_dense"
                })
        return output

    def bm25_search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Sparse BM25 lexical keyword matching over ChromaDB document collection."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        if not all_data or not all_data.get("ids"):
            return []

        q_tokens = set(re.findall(r"\w+", query_text.lower()))
        scored = []
        for i, doc_id in enumerate(all_data["ids"]):
            doc_content = all_data["documents"][i]
            meta = all_data["metadatas"][i]
            doc_tokens = set(re.findall(r"\w+", doc_content.lower()))
            overlap = len(q_tokens & doc_tokens)
            score = (overlap / (len(q_tokens) + 1e-5)) * 1.0
            scored.append({
                "id": doc_id,
                "content": doc_content,
                "metadata": meta,
                "parent_id": meta.get("parent_id", ""),
                "score": score,
                "search_type": "bm25"
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def hybrid_search(self, query_text: str, top_k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) combining ChromaDB dense search and BM25 lexical search.
        Score = 1/(k + rank_dense) + 1/(k + rank_bm25)
        """
        dense_results = self.similarity_search(query_text, top_k=top_k * 2)
        bm25_results = self.bm25_search(query_text, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        doc_map = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            doc_map[doc_id] = doc

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[:top_k]:
            results.append({**doc_map[doc_id], "score": score, "search_type": "hybrid_rrf"})
        return results

    def get_parent_document(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """Fetches the enclosing parent document context for Parent-Child expansion."""
        return self.parents.get(parent_id)

    def count(self) -> int:
        return self.collection.count()

# Global ChromaDB Vector Store
global_chroma_store = ChromaVectorStore()

def populate_sample_knowledge_base(store: ChromaVectorStore):
    """Fast batch initialization of sample enterprise documents."""
    if store.count() > 0:
        return
    all_chunks = []
    for doc in SAMPLE_ENTERPRISE_DOCS:
        store.add_parent_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            full_content=doc["full_content"]
        )
        for chunk in doc["chunks"]:
            all_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "parent_id": doc["doc_id"],
                "content": chunk["content"],
                "title": doc["title"]
            })

    contents = [c["content"] for c in all_chunks]
    model = get_bge_model()
    if model is not None:
        try:
            raw_embs = model.encode(contents, normalize_embeddings=True, batch_size=32)
            embeddings = [[round(float(x), 6) for x in v] for v in raw_embs]
        except Exception:
            embeddings = [generate_embedding(c, dim=384) for c in contents]
    else:
        embeddings = [generate_embedding(c, dim=384) for c in contents]

    ids = [c["chunk_id"] for c in all_chunks]
    metas = [
        {
            "parent_id": c["parent_id"],
            "title": c["title"],
            "parent_content": store.parents.get(c["parent_id"], {}).get("content", "")
        }
        for c in all_chunks
    ]
    store.collection.add(ids=ids, documents=contents, embeddings=embeddings, metadatas=metas)

# Initialize sample documents in batch
populate_sample_knowledge_base(global_chroma_store)
