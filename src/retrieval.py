"""
Embedding-based retrieval using sentence-transformers and cosine similarity.

Model choice: all-MiniLM-L6-v2
  - 22 million parameters, ~80 MB on disk
  - Runs on CPU; no GPU required
  - Encodes the full 918-chunk corpus in ~60 seconds
  - Apache 2.0 licence, freely available
  - Produces 384-dimensional embeddings with strong semantic quality

The index is cached to disk so the encoding step is only performed once.
Subsequent runs load the cache in under one second.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

Chunk = Dict[str, Any]


class EmbeddingRetriever:
    """Dense cosine-similarity retriever over a list of text chunks."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        print(f"[Retriever] Loading embedding model: {model_name}")
        self.model      = SentenceTransformer(model_name)
        self.chunks:    List[Chunk]    = []
        self.embeddings: np.ndarray | None = None

    # ── Index management ─────────────────────────────────────

    def build_index(
        self,
        chunks: List[Chunk],
        cache_path: Path | None = None,
    ) -> None:
        """
        Encode all chunks and store the resulting matrix.

        If cache_path is given and an existing cache matches the current
        chunk count, the cache is loaded instead of re-encoding.
        """
        if not chunks:
            raise ValueError("No chunks provided to build_index().")

        if cache_path and cache_path.exists():
            print(f"[Retriever] Loading cached index from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            cached_chunks = data["chunks"].tolist()
            if len(cached_chunks) == len(chunks):
                self.embeddings = data["embeddings"]
                self.chunks     = cached_chunks
                return
            print("[Retriever] Cache size mismatch — rebuilding index.")

        print(f"[Retriever] Encoding {len(chunks)} chunks …")
        texts           = [c["text"] for c in chunks]
        self.embeddings = np.asarray(
            self.model.encode(texts, show_progress_bar=True, batch_size=64)
        )
        self.chunks = chunks

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                embeddings=self.embeddings,
                chunks=np.array(chunks, dtype=object),
            )
            print(f"[Retriever] Index saved to {cache_path}")

    # ── Retrieval ────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Tuple[Chunk, float]]:
        """
        Return the top-k most semantically similar chunks for *query*.

        Diversity filter: at most one speaker-window chunk from the
        same (play, act, scene) appears in the result set, preventing
        overlapping windows from dominating the retrieved evidence.
        """
        if self.embeddings is None:
            raise RuntimeError("Call build_index() before retrieve().")

        q_emb  = np.asarray(self.model.encode([query]))
        scores = cosine_similarity(q_emb, self.embeddings)[0]
        ranked = np.argsort(scores)[::-1]

        results: List[Tuple[Chunk, float]] = []
        seen_scene_window: set              = set()

        for idx in ranked:
            if len(results) >= top_k:
                break
            chunk = self.chunks[idx]
            # Deduplication key for speaker-window chunks
            if chunk["chunk_type"] == "speaker_window":
                key = (chunk["play"], chunk["act"], chunk["scene"])
                if key in seen_scene_window:
                    continue
                seen_scene_window.add(key)
            results.append((chunk, float(scores[idx])))

        return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config       import EMBEDDING_MODEL_NAME, DEFAULT_TOP_K, INDEX_CACHE
    from data_loader  import load_scene_chunks, load_utterances
    from chunking     import create_chunks, format_chunk_for_display

    scenes     = load_scene_chunks()
    utterances = load_utterances()
    chunks     = create_chunks(scenes, utterances)

    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    retriever.build_index(chunks, cache_path=INDEX_CACHE)

    for q in [
        "Why does Macbeth kill Duncan?",
        "Who is Hamlet?",
        "What is the conflict between the Montagues and the Capulets?",
    ]:
        print(f"\nQuery: {q}")
        for rank, (chunk, score) in enumerate(
            retriever.retrieve(q, top_k=DEFAULT_TOP_K), 1
        ):
            print(f"  [{rank}] {chunk['play']} A{chunk['act']}S{chunk['scene']}"
                  f" ({chunk['chunk_type']}) score={score:.4f}")
