"""
Chunking strategy for the Shakespeare corpus.

Two-tier hybrid approach
------------------------
1. Scene chunks (primary)
   One chunk per Act/Scene. The scene_summary is prepended to the full
   scene text so the index captures both the topic (summary keywords)
   and the content (spoken lines). This handles broad thematic and
   event-level queries well.

2. Speaker-window chunks (secondary)
   Utterance records are grouped into overlapping windows of 8 turns
   with a stride of 4 (50 % overlap), excluding stage directions.
   These provide finer evidence for character-attribution and
   dialogue-level queries.

A diversity filter in retrieval.py ensures that at most one
speaker-window chunk from the same (play, act, scene) appears in
any single result set, preventing near-duplicate windows from
crowding out results from other scenes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

Record = Dict[str, Any]
Chunk  = Dict[str, Any]

_EXCLUDED_SPEAKERS = {"STAGE_DIRECTION", "ELSINORE", "FLOURISH"}

WINDOW_SIZE   = 8
WINDOW_STRIDE = 4   # 50 % overlap


# ── Scene chunks ────────────────────────────────────────────

def _build_scene_chunk(record: Record) -> Chunk:
    play    = record.get("play", record.get("play_key", "Unknown"))
    act     = record.get("act", "?")
    scene   = record.get("scene", "?")
    summary = record.get("scene_summary", "").strip()
    text    = record.get("text", "").strip()

    # Prepend summary so the embedding index captures the topic
    enriched = f"[{play}, Act {act}, Scene {scene}]\nSummary: {summary}\n\n{text}"

    return {
        "chunk_id"    : record.get("scene_id") or f"{play}_{act}_{scene}",
        "chunk_type"  : "scene",
        "play"        : play,
        "act"         : act,
        "scene"       : scene,
        "speaker"     : None,
        "text"        : enriched,
        "display_text": text,
        "summary"     : summary,
        "metadata"    : record,
    }


# ── Speaker-window chunks ────────────────────────────────────

def _build_speaker_window(records: List[Record], window_id: int) -> Optional[Chunk]:
    lines = []
    play  = records[0].get("play", records[0].get("play_key", "Unknown"))
    act   = records[0].get("act", "?")
    scene = records[0].get("scene", "?")

    for r in records:
        spk = r.get("speaker", "")
        txt = r.get("text", "").strip()
        if spk in _EXCLUDED_SPEAKERS or not txt:
            continue
        lines.append(f"{spk}: {txt}")

    if not lines:
        return None

    block    = "\n".join(lines)
    enriched = f"[{play}, Act {act}, Scene {scene}]\n{block}"

    return {
        "chunk_id"    : f"{play}_{act}_{scene}_w{window_id:04d}",
        "chunk_type"  : "speaker_window",
        "play"        : play,
        "act"         : act,
        "scene"       : scene,
        "speaker"     : None,
        "text"        : enriched,
        "display_text": block,
        "summary"     : None,
        "metadata"    : {"play": play, "act": act, "scene": scene},
    }


# ── Public API ───────────────────────────────────────────────

def create_chunks(
    scene_records: List[Record],
    utterance_records: List[Record],
) -> List[Chunk]:
    """
    Build the full retrieval corpus from scene and utterance records.

    Parameters
    ----------
    scene_records     : output of data_loader.load_scene_chunks()
    utterance_records : output of data_loader.load_utterances()

    Returns
    -------
    List of chunk dicts ready for embedding and indexing.
    """
    chunks: List[Chunk] = []

    # 1. Scene-level chunks
    for record in scene_records:
        chunks.append(_build_scene_chunk(record))

    # 2. Speaker-window chunks
    grouped: Dict[tuple, List[Record]] = defaultdict(list)
    for u in utterance_records:
        key = (
            u.get("play", u.get("play_key", "")),
            u.get("act"),
            u.get("scene"),
        )
        grouped[key].append(u)

    for recs in grouped.values():
        recs.sort(key=lambda r: r.get("utterance_id", ""))
        for start in range(0, len(recs), WINDOW_STRIDE):
            window = recs[start : start + WINDOW_SIZE]
            chunk  = _build_speaker_window(window, start)
            if chunk is not None:
                chunks.append(chunk)

    return chunks


def format_chunk_for_display(chunk: Chunk) -> str:
    """Return a human-readable citation block for a retrieved chunk."""
    play    = chunk.get("play", "Unknown")
    act     = chunk.get("act",   "?")
    scene   = chunk.get("scene", "?")
    summary = chunk.get("summary") or ""
    text    = chunk.get("display_text") or chunk.get("text", "")

    header = f"{play}, Act {act}, Scene {scene}"
    if summary:
        header += f"\nScene summary: {summary}"

    # Truncate long texts for display
    if len(text) > 800:
        text = text[:800] + "… [truncated]"

    return f"[Source: {header}]\n{text}"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_scene_chunks, load_utterances

    scenes     = load_scene_chunks()
    utterances = load_utterances()
    chunks     = create_chunks(scenes, utterances)

    scene_c  = sum(1 for c in chunks if c["chunk_type"] == "scene")
    window_c = sum(1 for c in chunks if c["chunk_type"] == "speaker_window")
    print(f"Total chunks   : {len(chunks)}")
    print(f"  Scene chunks : {scene_c}")
    print(f"  Speaker wins : {window_c}")
