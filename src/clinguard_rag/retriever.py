"""Stage 2: Hybrid BM25 + TF-IDF Cosine Reciprocal Rank Fusion (RRF) Clinical Retriever."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from clinguard_rag.schemas import ClinicalChunk, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


class HybridClinicalRetriever:
    """Combines lexical BM25Okapi with L2-normalized IDF vector cosine similarity via RRF."""

    def __init__(self, chunks: list[ClinicalChunk], rrf_k: int = 60) -> None:
        self.chunks = chunks
        self.rrf_k = rrf_k
        self._tokenized = [_tokenize(f"{c.title} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

        # Build vocabulary & IDF-weighted normalized vectors
        df: Counter[str] = Counter()
        for toks in self._tokenized:
            for t in set(toks):
                df[t] += 1
        self._vocab: dict[str, int] = {t: i for i, t in enumerate(sorted(df.keys()))}
        n_docs = max(1, len(chunks))
        self._idf = np.zeros(len(self._vocab), dtype=np.float64)
        for term, idx in self._vocab.items():
            self._idf[idx] = math.log((n_docs + 1.0) / (df[term] + 0.5)) + 1.0

        self._doc_vecs = np.zeros((len(chunks), len(self._vocab)), dtype=np.float64)
        for i, toks in enumerate(self._tokenized):
            self._doc_vecs[i] = self._encode(toks)

    def _encode(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        counts = Counter(tokens)
        for t, c in counts.items():
            if t in self._vocab:
                idx = self._vocab[t]
                vec[idx] = (1.0 + math.log(c)) * self._idf[idx]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    @classmethod
    def from_json(cls, path: str | Path) -> HybridClinicalRetriever:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks = [ClinicalChunk(**item) for item in raw]
        return cls(chunks=chunks)

    def retrieve(
        self,
        query: str,
        allowed_domains: list[str] | None = None,
        top_k: int = 2,
    ) -> list[RetrievedChunk]:
        # Fail-closed when an explicitly empty domain list is passed (L2 fix)
        if allowed_domains is not None and len(allowed_domains) == 0:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        bm25_all = np.array(self._bm25.get_scores(q_tokens), dtype=np.float64)
        q_vec = self._encode(q_tokens)
        dense_all = self._doc_vecs @ q_vec

        candidate_indices = [
            i
            for i, c in enumerate(self.chunks)
            if (allowed_domains is None or c.domain in allowed_domains)
            and (bm25_all[i] > 0.0 or dense_all[i] > 0.0)  # Filter out zero-overlap chunks (M5 fix)
        ]
        if not candidate_indices:
            return []

        bm25_ranked = sorted(candidate_indices, key=lambda idx: bm25_all[idx], reverse=True)
        dense_ranked = sorted(candidate_indices, key=lambda idx: dense_all[idx], reverse=True)

        bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
        dense_rank_map = {idx: rank + 1 for rank, idx in enumerate(dense_ranked)}

        rrf_scores: dict[int, float] = {}
        for idx in candidate_indices:
            rrf_scores[idx] = (1.0 / (self.rrf_k + bm25_rank_map[idx])) + (
                1.0 / (self.rrf_k + dense_rank_map[idx])
            )

        final_indices = sorted(candidate_indices, key=lambda idx: rrf_scores[idx], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        for idx in final_indices:
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    bm25_score=round(float(bm25_all[idx]), 4),
                    dense_score=round(float(dense_all[idx]), 4),
                    rrf_score=round(float(rrf_scores[idx]), 6),
                    bm25_rank=bm25_rank_map[idx],
                    dense_rank=dense_rank_map[idx],
                )
            )
        return results
