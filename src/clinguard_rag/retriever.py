"""Hybrid Clinical Retriever combining BM25Okapi and Dense TF-IDF Cosine via RRF."""

from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from clinguard_rag.schemas import GuidelineChunk, RetrievedChunk


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


class HybridClinicalRetriever:
    """In-memory BM25 + TF-IDF Cosine Reciprocal Rank Fusion retriever for clinical passages."""

    def __init__(self, chunks: List[GuidelineChunk], rrf_k: int = 60) -> None:
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.doc_tokens = [
            _tokenize(f"{c.title} {c.text} {' '.join(c.clinical_keywords)}")
            for c in chunks
        ]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))

        vocab_set = sorted({tok for toks in self.doc_tokens for tok in toks})
        self.vocab = {w: i for i, w in enumerate(vocab_set)}
        self.idf: Dict[str, float] = {}
        n_docs = len(chunks)
        for w in vocab_set:
            df = sum(1 for toks in self.doc_tokens if w in toks)
            self.idf[w] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        self.tfidf_matrix = np.zeros((n_docs, len(self.vocab)), dtype=np.float64)
        for d_idx, toks in enumerate(self.doc_tokens):
            for tok in toks:
                v_idx = self.vocab[tok]
                self.tfidf_matrix[d_idx, v_idx] += self.idf[tok]
            norm = np.linalg.norm(self.tfidf_matrix[d_idx])
            if norm > 0:
                self.tfidf_matrix[d_idx] /= norm

    @classmethod
    def from_json(cls, path: str | Path) -> "HybridClinicalRetriever":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([GuidelineChunk(**item) for item in raw])

    def _bm25_score(self, q_tokens: List[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        toks = self.doc_tokens[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0
        for q in q_tokens:
            if q not in self.idf:
                continue
            tf = toks.count(q)
            if tf == 0:
                continue
            num = tf * (k1 + 1.0)
            den = tf + k1 * (1.0 - b + b * (dl / self.avgdl))
            score += self.idf[q] * (num / den)
        return score

    def _dense_score(self, q_tokens: List[str], doc_idx: int) -> float:
        q_vec = np.zeros(len(self.vocab), dtype=np.float64)
        for q in q_tokens:
            if q in self.vocab:
                q_vec[self.vocab[q]] += self.idf[q]
        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return 0.0
        q_vec /= norm
        return float(np.dot(q_vec, self.tfidf_matrix[doc_idx]))

    def retrieve(
        self,
        query: str,
        allowed_domains: Optional[List[str]] = None,
        top_k: int = 2,
    ) -> List[RetrievedChunk]:
        q_tokens = _tokenize(query)
        candidates = [
            i for i, c in enumerate(self.chunks)
            if not allowed_domains or c.domain in allowed_domains
        ]
        if not candidates:
            return []

        bm25_scores = {i: self._bm25_score(q_tokens, i) for i in candidates}
        dense_scores = {i: self._dense_score(q_tokens, i) for i in candidates}

        bm25_ranked = sorted(candidates, key=lambda i: bm25_scores[i], reverse=True)
        dense_ranked = sorted(candidates, key=lambda i: dense_scores[i], reverse=True)

        bm25_rank_map = {doc_i: r + 1 for r, doc_i in enumerate(bm25_ranked)}
        dense_rank_map = {doc_i: r + 1 for r, doc_i in enumerate(dense_ranked)}

        results: List[RetrievedChunk] = []
        for i in candidates:
            rrf = (1.0 / (self.rrf_k + bm25_rank_map[i])) + (
                1.0 / (self.rrf_k + dense_rank_map[i])
            )
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[i],
                    bm25_score=round(bm25_scores[i], 4),
                    dense_score=round(dense_scores[i], 4),
                    rrf_score=round(rrf, 6),
                )
            )

        results.sort(key=lambda r: (r.rrf_score, r.bm25_score), reverse=True)
        return results[:top_k]