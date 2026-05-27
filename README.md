# Shakespeare-Aware Question-Answering System
### CSCI433/933 Assignment 2 — University of Wollongong

---

## Overview

This system allows a user with no prior knowledge of Shakespeare to ask
questions about *Hamlet*, *Macbeth*, and *Romeo and Juliet* and receive
answers that are grounded in the actual text of the plays.

It uses a Retrieval-Augmented Generation (RAG) pipeline:

1. The corpus is split into 918 retrievable chunks (scene-level + speaker-window).
2. Each chunk is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`.
3. A user question is encoded with the same model and compared against all chunks.
4. The top-4 most relevant chunks are retrieved and placed into a prompt.
5. A hosted language model generates a grounded answer.
6. Both the answer and the source passages are displayed.

A TF-IDF keyword baseline is also included for comparison.

---

## Project Structure

```
shakespeare_rag/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── prompts/
│   └── system_prompt.txt            # System prompt used for generation
├── data/
│   └── processed/                   # Dataset JSONL files go here
│       ├── hamlet_scene_chunks.jsonl
│       ├── hamlet_utterances.jsonl
│       ├── macbeth_scene_chunks.jsonl
│       ├── macbeth_utterances.jsonl
│       ├── romeo_and_juliet_scene_chunks.jsonl
│       └── romeo_and_juliet_utterances.jsonl
├── src/
│   ├── config.py          # File paths and model settings
│   ├── data_loader.py     # Load scene and utterance JSONL files
│   ├── chunking.py        # Two-tier chunking strategy
│   ├── retrieval.py       # Embedding index and cosine similarity search
│   ├── rag_chatbot.py     # Full RAG question-answering pipeline
│   ├── baseline.py        # TF-IDF keyword baseline system
│   └── evaluate.py        # Evaluation runner and scoring
├── results/
│   ├── instructor_questions.json    # The 5 instructor-provided questions
│   ├── all_answers.json             # Full system outputs for all 15 questions
│   └── evaluation_summary.csv      # Scored evaluation table
└── report/
    ├── report.tex                   # LaTeX source of the report
    └── report.pdf                   # Compiled PDF report
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `sentence-transformers >= 2.2.0`
- `scikit-learn >= 1.3.0`
- `numpy >= 1.24.0`
- `anthropic >= 0.25.0`

Python 3.10 or higher is required.

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 3. Place the dataset files

The six JSONL dataset files must be present in `data/processed/`.
They are included in this submission.

---

## Running the System

### Interactive chatbot (RAG system)

```bash
cd src
python rag_chatbot.py
```

Type a question at the prompt and press Enter. The system will display
the four retrieved source passages and the generated answer.
Type `quit` to exit.

### Single question (non-interactive)

```bash
cd src
python rag_chatbot.py --query "Why does Hamlet delay taking revenge?"
```

### Baseline system

```bash
cd src
python baseline.py
```

### Run full evaluation (all 15 questions, both systems)

```bash
cd src
python evaluate.py
```

Outputs are written to `results/evaluation_results.json` and
`results/evaluation_summary.csv`.

---

## Index caching

The embedding index is built on the first run (~60 seconds on CPU) and
cached to `data/processed/index_cache.npz`. Every subsequent run loads
the cache in under one second. No recomputation is needed during
assessment.

---

## System design summary

| Component         | Choice                              | Reason                                              |
|-------------------|-------------------------------------|-----------------------------------------------------|
| Chunking          | Hybrid: scene + speaker-window      | Balances context richness with retrieval precision  |
| Embedding model   | all-MiniLM-L6-v2 (22M params, CPU) | Lightweight, no GPU required, Apache 2.0 licence    |
| Retrieval         | Flat cosine similarity + diversity filter | Simple, fast, no external service needed       |
| Generation model  | Small hosted LM via API             | No GPU required; easily swapped for local model     |
| Baseline          | TF-IDF keyword matching             | Honest, interpretable comparison point              |

---

## Swapping to a fully local model

To run the system with no external API, replace the body of
`generate_answer()` in `src/rag_chatbot.py` with a call to a locally
served model via Ollama:

```python
import requests

def generate_answer(system_prompt: str, user_message: str) -> str:
    payload = {
        "model": "phi3:mini",
        "prompt": f"{system_prompt}\n\n{user_message}",
        "stream": False
    }
    r = requests.post("http://localhost:11434/api/generate", json=payload)
    return r.json()["response"]
```

No other changes are required.

---

## Notes

- The dataset files are sourced from the provided assignment dataset.
- All supplementary material (scene summaries, keywords) comes from the
  dataset itself; no external sources were used.
- The system prompt is in `prompts/system_prompt.txt` and can be edited
  without changing any code.
