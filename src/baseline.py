"""
Baseline system: TF-IDF keyword retrieval with minimal prompting.

This baseline deliberately avoids semantic embeddings. Instead it uses
TF-IDF scoring to find relevant scene-level passages and passes only
the top-2 to the language model with a bare, unguided system prompt.

The baseline is designed to be a fair and honest comparison point:
  - It represents the effort a practitioner might make before investing
    in a full embedding pipeline.
  - It works adequately for questions whose key words appear prominently
    in the right scenes.
  - It fails for synonym-heavy, thematic, or minor-character questions.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import anthropic

from config      import GENERATION_MODEL
from data_loader import load_scene_chunks, load_utterances
from chunking    import create_chunks

Chunk = Dict[str, Any]

BASELINE_SYSTEM_PROMPT = (
    "You are an assistant with knowledge of Shakespeare's plays. "
    "Answer the question using the provided text excerpts. "
    "Be concise and factual."
)
BASELINE_TOP_K = 2


# ── TF-IDF keyword retriever ─────────────────────────────────

def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _build_tfidf(
    chunks: List[Chunk],
) -> Tuple[List[Counter], Dict[str, float]]:
    tf_list: List[Counter] = []
    df:      Counter        = Counter()
    for chunk in chunks:
        tokens = _tokenise(chunk.get("display_text") or chunk.get("text", ""))
        tf     = Counter(tokens)
        tf_list.append(tf)
        df.update(set(tokens))
    N   = len(chunks)
    idf = {
        term: math.log((N + 1) / (count + 1)) + 1
        for term, count in df.items()
    }
    return tf_list, idf


class KeywordRetriever:
    """TF-IDF retriever over scene-level chunks."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks  = chunks
        self.tf_list, self.idf = _build_tfidf(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = BASELINE_TOP_K,
    ) -> List[Tuple[Chunk, float]]:
        q_tokens = _tokenise(query)
        scores   = [
            sum(tf.get(t, 0) * self.idf.get(t, 0) for t in q_tokens)
            for tf in self.tf_list
        ]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], scores[i]) for i in ranked[:top_k]]


# ── Baseline system ──────────────────────────────────────────

class BaselineSystem:
    """TF-IDF keyword baseline with minimal prompting."""

    def __init__(self):
        scenes     = load_scene_chunks()
        utterances = load_utterances()
        all_chunks = create_chunks(scenes, utterances)
        # Use scene-level chunks only (no speaker windows)
        self.chunks    = [c for c in all_chunks if c["chunk_type"] == "scene"]
        self.retriever = KeywordRetriever(self.chunks)
        self.client    = anthropic.Anthropic()

    def answer(self, query: str) -> Dict[str, Any]:
        retrieved = self.retriever.retrieve(query, top_k=BASELINE_TOP_K)

        context_parts: List[str] = []
        for rank, (chunk, _) in enumerate(retrieved, 1):
            text = chunk.get("display_text") or chunk.get("text", "")
            if len(text) > 600:
                text = text[:600] + "…"
            play  = chunk.get("play",  "Unknown")
            act   = chunk.get("act",   "?")
            scene = chunk.get("scene", "?")
            context_parts.append(f"[{play} Act {act}, Scene {scene}]\n{text}")

        context  = "\n\n".join(context_parts)
        user_msg = f"Text excerpts:\n{context}\n\nQuestion: {query}"

        response = self.client.messages.create(
            model     = GENERATION_MODEL,
            max_tokens= 400,
            system    = BASELINE_SYSTEM_PROMPT,
            messages  = [{"role": "user", "content": user_msg}],
        )
        return {
            "query"    : query,
            "retrieved": retrieved,
            "answer"   : response.content[0].text,
        }


if __name__ == "__main__":
    system = BaselineSystem()
    for q in ["Who is Hamlet?", "Why does Macbeth kill Duncan?"]:
        result = system.answer(q)
        print(f"\nQuestion : {result['query']}")
        print(f"Answer   : {result['answer'][:200]}…")
        print(f"Passages : {[(c['play'], c['act'], c['scene']) for c,_ in result['retrieved']]}")
