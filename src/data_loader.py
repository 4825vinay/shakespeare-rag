"""
Data loading utilities for the Shakespeare dataset.

We use scene-level chunks as the primary retrieval unit and
utterance records as the source for speaker-window chunks.
Both are loaded here and passed to chunking.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from config import SCENE_CHUNK_FILES, UTTERANCE_FILES

Record = Dict[str, Any]


def _load_jsonl(path: Path) -> List[Record]:
    """Load every line of a .jsonl file into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Ensure all six JSONL files are present in data/processed/."
        )
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_scene_chunks() -> List[Record]:
    """Load scene-level chunks for all three plays."""
    all_chunks: List[Record] = []
    for play_key, path in SCENE_CHUNK_FILES.items():
        records = _load_jsonl(path)
        for r in records:
            r["play_key"]   = play_key
            r["chunk_type"] = "scene"
        all_chunks.extend(records)
    return all_chunks


def load_utterances() -> List[Record]:
    """Load utterance-level records for all three plays."""
    all_utterances: List[Record] = []
    for play_key, path in UTTERANCE_FILES.items():
        records = _load_jsonl(path)
        for r in records:
            r["play_key"]   = play_key
            r["chunk_type"] = "utterance"
        all_utterances.extend(records)
    return all_utterances


if __name__ == "__main__":
    scenes     = load_scene_chunks()
    utterances = load_utterances()
    print(f"Scene chunks : {len(scenes)}")
    print(f"Utterances   : {len(utterances)}")
    print("\nSample scene chunk keys :", list(scenes[0].keys()))
    print("Sample utterance keys   :", list(utterances[0].keys()))
