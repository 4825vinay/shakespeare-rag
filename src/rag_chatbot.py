"""
Shakespeare RAG Chatbot
=======================
Full Retrieval-Augmented Generation pipeline for answering questions
about Hamlet, Macbeth, and Romeo and Juliet.

Usage
-----
Interactive mode:
    python rag_chatbot.py

Single query:
    python rag_chatbot.py --query "Why does Hamlet delay taking revenge?"

With custom top-k:
    python rag_chatbot.py --query "Who is Macbeth?" --top_k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import anthropic

from config    import (DEFAULT_TOP_K, EMBEDDING_MODEL_NAME,
                       GENERATION_MODEL, INDEX_CACHE,
                       MAX_TOKENS, PROMPTS_DIR)
from chunking  import create_chunks, format_chunk_for_display
from data_loader import load_scene_chunks, load_utterances
from retrieval import EmbeddingRetriever

Chunk = Dict[str, Any]


# ── Prompt helpers ───────────────────────────────────────────

def load_system_prompt() -> str:
    path = PROMPTS_DIR / "system_prompt.txt"
    return path.read_text(encoding="utf-8").strip()


def build_user_message(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
) -> str:
    """Build the user message that embeds retrieved evidence."""
    blocks: List[str] = []
    for rank, (chunk, score) in enumerate(retrieved, 1):
        play    = chunk.get("play", "Unknown")
        act     = chunk.get("act",  "?")
        scene   = chunk.get("scene","?")
        summary = chunk.get("summary") or ""
        text    = chunk.get("display_text") or chunk.get("text", "")

        meta = f"{play}, Act {act}, Scene {scene}"
        if summary:
            meta += f" — {summary}"

        if len(text) > 1000:
            text = text[:1000] + "… [truncated]"

        blocks.append(
            f"[Evidence {rank} | relevance={score:.3f} | {meta}]\n{text}"
        )

    context = "\n\n---\n\n".join(blocks)
    return (
        f"Retrieved passages from the plays:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        f"Please answer using the evidence above."
    )


# ── Generation ───────────────────────────────────────────────

def generate_answer(system_prompt: str, user_message: str) -> str:
    """Send the RAG prompt to the language model and return the answer."""
    client   = anthropic.Anthropic()
    response = client.messages.create(
        model     = GENERATION_MODEL,
        max_tokens= MAX_TOKENS,
        system    = system_prompt,
        messages  = [{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ── Main pipeline ─────────────────────────────────────────────

class ShakespeareRAG:
    """End-to-end RAG system: load, index, retrieve, generate."""

    def __init__(self):
        print("[System] Loading corpus …")
        scenes     = load_scene_chunks()
        utterances = load_utterances()
        self.chunks = create_chunks(scenes, utterances)
        print(f"[System] {len(self.chunks)} retrieval chunks ready.")

        self.retriever     = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
        self.retriever.build_index(self.chunks, cache_path=INDEX_CACHE)
        self.system_prompt = load_system_prompt()

    def answer(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """Run the full pipeline and return a result dict."""
        retrieved    = self.retriever.retrieve(query, top_k=top_k)
        user_message = build_user_message(query, retrieved)
        answer_text  = generate_answer(self.system_prompt, user_message)
        return {"query": query, "retrieved": retrieved, "answer": answer_text}

    @staticmethod
    def display(result: Dict[str, Any]) -> None:
        """Pretty-print a result to stdout."""
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  Question: {result['query']}")
        print(sep)

        print("\n  Retrieved Evidence")
        print("  " + "-" * 68)
        for rank, (chunk, score) in enumerate(result["retrieved"], 1):
            print(f"\n  [{rank}] Relevance score: {score:.4f}")
            for line in format_chunk_for_display(chunk).splitlines():
                print(f"      {line}")

        print("\n  Generated Answer")
        print("  " + "-" * 68)
        for line in result["answer"].splitlines():
            print(f"  {line}")
        print()


def interactive_mode(rag: ShakespeareRAG) -> None:
    print("\n  Shakespeare-Aware RAG Chatbot")
    print("  Covering: Hamlet · Macbeth · Romeo and Juliet")
    print("  Type 'quit' or press Ctrl-C to exit.\n")
    while True:
        try:
            query = input("  Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye.")
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            break
        result = rag.answer(query)
        rag.display(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shakespeare RAG Chatbot")
    parser.add_argument("--query",  type=str, help="Run a single query")
    parser.add_argument("--top_k",  type=int, default=DEFAULT_TOP_K,
                        help="Number of passages to retrieve")
    args = parser.parse_args()

    rag = ShakespeareRAG()

    if args.query:
        result = rag.answer(args.query, top_k=args.top_k)
        rag.display(result)
    else:
        interactive_mode(rag)


if __name__ == "__main__":
    main()
