"""
Configuration for the Shakespeare RAG system.

Adjust EMBEDDING_MODEL_NAME or GENERATION_MODEL here if you wish
to swap to a different model without changing any other file.
"""

from pathlib import Path

PROJECT_ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR           = PROJECT_ROOT / "data" / "processed"
PROMPTS_DIR        = PROJECT_ROOT / "prompts"
RESULTS_DIR        = PROJECT_ROOT / "results"

# ── Dataset files ────────────────────────────────────────────
SCENE_CHUNK_FILES = {
    "hamlet":          DATA_DIR / "hamlet_scene_chunks.jsonl",
    "macbeth":         DATA_DIR / "macbeth_scene_chunks.jsonl",
    "romeo_and_juliet":DATA_DIR / "romeo_and_juliet_scene_chunks.jsonl",
}

UTTERANCE_FILES = {
    "hamlet":          DATA_DIR / "hamlet_utterances.jsonl",
    "macbeth":         DATA_DIR / "macbeth_utterances.jsonl",
    "romeo_and_juliet":DATA_DIR / "romeo_and_juliet_utterances.jsonl",
}

# ── Model settings ───────────────────────────────────────────
# Embedding model: 22M params, runs on CPU, Apache 2.0 licence.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Generation model: small hosted language model via Anthropic API.
# Swap this string (or replace generate_answer() in rag_chatbot.py)
# to use a different or locally hosted model.
GENERATION_MODEL = "claude-haiku-4-5-20251001"

MAX_TOKENS    = 512
DEFAULT_TOP_K = 4

# ── Index cache ──────────────────────────────────────────────
# Built once (~60 s on CPU), then reloaded in <1 s on every
# subsequent run. Delete this file to force a full rebuild.
INDEX_CACHE = DATA_DIR / "index_cache.npz"
