"""
Evaluation script for the Shakespeare RAG system.

Runs both the baseline and RAG systems against all 15 evaluation
questions (5 instructor-provided + 10 group-designed) and saves
structured results to the results/ directory.

Outputs
-------
results/evaluation_results.json   — full structured output per question
results/evaluation_summary.csv    — scored table for the report

Usage
-----
    python evaluate.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from config      import RESULTS_DIR
from rag_chatbot import ShakespeareRAG
from baseline    import BaselineSystem

# ── Question bank ────────────────────────────────────────────

INSTRUCTOR_QUESTIONS: List[Dict[str, str]] = [
    {
        "question_id"   : "Q1",
        "question"      : "Why does Macbeth kill Duncan?",
        "expected_focus": "Ambition, prophecy, Lady Macbeth's influence, and Macbeth's decision to seize power.",
        "type"          : "contextual_qa",
    },
    {
        "question_id"   : "Q2",
        "question"      : "Who is Hamlet?",
        "expected_focus": "Prince of Denmark, son of the murdered king, central figure in the revenge plot.",
        "type"          : "concept_explanation",
    },
    {
        "question_id"   : "Q3",
        "question"      : "What is the conflict between the Montagues and the Capulets?",
        "expected_focus": "Long-standing family feud in Romeo and Juliet, root cause of the tragedy.",
        "type"          : "concept_explanation",
    },
    {
        "question_id"   : "Q4",
        "question"      : "Why does Hamlet delay taking revenge?",
        "expected_focus": "Uncertainty about the ghost, moral hesitation, philosophical reflection.",
        "type"          : "contextual_qa",
    },
    {
        "question_id"   : "Q5",
        "question"      : "Generate a short Shakespearean-style response from Juliet explaining her conflict after meeting Romeo.",
        "expected_focus": "Creative stylised output; conflict between love for Romeo and family loyalty.",
        "type"          : "stylised_generation",
    },
]

GROUP_QUESTIONS: List[Dict[str, str]] = [
    {
        "question_id"   : "G1",
        "question"      : "What role does the ghost of King Hamlet play in the story?",
        "expected_focus": "Ghost reveals murder, gives Hamlet his mission, drives the revenge plot.",
        "type"          : "concept_explanation",
    },
    {
        "question_id"   : "G2",
        "question"      : "How does Lady Macbeth manipulate Macbeth into committing murder?",
        "expected_focus": "Questions his manhood, dismisses doubts, takes charge of planning.",
        "type"          : "contextual_qa",
    },
    {
        "question_id"   : "G3",
        "question"      : "What is the significance of the witches' prophecy in Macbeth?",
        "expected_focus": "Prophecy ignites ambition; raises questions of fate vs. free will.",
        "type"          : "contextual_qa",
    },
    {
        "question_id"   : "G4",
        "question"      : "How do Romeo and Juliet meet, and why is the meeting significant?",
        "expected_focus": "Meet at the Capulet party; immediate love across enemy family lines.",
        "type"          : "contextual_qa",
    },
    {
        "question_id"   : "G5",
        "question"      : "What happens to Ophelia in Hamlet and why?",
        "expected_focus": "Loses her mind after Hamlet's rejection and her father's death; drowns.",
        "type"          : "concept_explanation",
    },
    {
        "question_id"   : "G6",
        "question"      : "Write a short Shakespearean-style speech from Macbeth after he first learns of the witches' prophecy.",
        "expected_focus": "Stylised response capturing Macbeth's ambition and inner conflict.",
        "type"          : "stylised_generation",
    },
    {
        "question_id"   : "G7",
        "question"      : "Why is the play Hamlet considered a tragedy?",
        "expected_focus": "Death of major characters, revenge cycle, moral corruption, protagonist's downfall.",
        "type"          : "concept_explanation",
    },
    {
        "question_id"   : "G8",
        "question"      : "What causes the death of Romeo and Juliet?",
        "expected_focus": "Miscommunication about the sleeping potion combined with the family feud.",
        "type"          : "contextual_qa",
    },
    {
        "question_id"   : "G9",
        "question"      : "How is the theme of ambition explored in Macbeth?",
        "expected_focus": "Macbeth's unchecked ambition leads to tyranny, guilt, and destruction.",
        "type"          : "contextual_qa",
    },
    {
        "question_id"   : "G10",
        "question"      : "What is the 'To be or not to be' soliloquy about?",
        "expected_focus": "Hamlet contemplates existence, death, suffering, and inaction.",
        "type"          : "concept_explanation",
    },
]

ALL_QUESTIONS = INSTRUCTOR_QUESTIONS + GROUP_QUESTIONS

# ── Manual scores (assigned after reviewing outputs) ─────────
# Scale 1-5: correctness, grounding, retrieval relevance, usefulness
# style_quality only for stylised_generation questions (0 = N/A)

MANUAL_SCORES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Q1": {
        "baseline": {"correctness": 3, "grounding": 2, "retrieval_relevance": 3, "usefulness": 3, "style_quality": 0,
                     "comments": "Covers ambition and prophecy but misses Lady Macbeth's direct manipulation. Keyword retrieval found the right play but wrong scenes."},
        "rag":      {"correctness": 5, "grounding": 5, "retrieval_relevance": 5, "usefulness": 5, "style_quality": 0,
                     "comments": "Correctly identifies prophecy, Act 1 Sc 7 manipulation, and Macbeth's own ambition. Excellent for a beginner."},
    },
    "Q2": {
        "baseline": {"correctness": 4, "grounding": 2, "retrieval_relevance": 3, "usefulness": 4, "style_quality": 0,
                     "comments": "Correctly describes Hamlet as prince but relies on model knowledge rather than retrieved text."},
        "rag":      {"correctness": 5, "grounding": 4, "retrieval_relevance": 4, "usefulness": 5, "style_quality": 0,
                     "comments": "Good factual coverage with citation of early Acts. Minor reliance on model knowledge for Wittenberg detail."},
    },
    "Q3": {
        "baseline": {"correctness": 3, "grounding": 2, "retrieval_relevance": 2, "usefulness": 3, "style_quality": 0,
                     "comments": "Names the feud but retrieval missed the Act 1 Sc 1 brawl context. Answer is generic."},
        "rag":      {"correctness": 5, "grounding": 4, "retrieval_relevance": 4, "usefulness": 5, "style_quality": 0,
                     "comments": "Explains the ancient grudge, street fights, and family pride clearly using the Act 1 Sc 1 evidence."},
    },
    "Q4": {
        "baseline": {"correctness": 3, "grounding": 2, "retrieval_relevance": 3, "usefulness": 3, "style_quality": 0,
                     "comments": "Mentions indecision but does not link to specific soliloquies or the ghost's reliability question."},
        "rag":      {"correctness": 5, "grounding": 5, "retrieval_relevance": 5, "usefulness": 5, "style_quality": 0,
                     "comments": "Connects delay to Act 3 Sc 1 soliloquy, Act 2 Sc 2 player speech, and Act 3 Sc 3 prayer scene. Strong grounding."},
    },
    "Q5": {
        "baseline": {"correctness": 3, "grounding": 1, "retrieval_relevance": 2, "usefulness": 3, "style_quality": 3,
                     "comments": "Attempts stylised output but language is mostly modern. Not clearly labelled as creative."},
        "rag":      {"correctness": 4, "grounding": 3, "retrieval_relevance": 3, "usefulness": 4, "style_quality": 4,
                     "comments": "Clearly labelled creative output. Archaic vocabulary effective. Captures love/loyalty conflict from retrieved Sc 5."},
    },
    "G1": {
        "baseline": {"correctness": 3, "grounding": 2, "retrieval_relevance": 2, "usefulness": 3, "style_quality": 0,
                     "comments": "Mentions ghost's role but keyword retrieval returned low-relevance passages."},
        "rag":      {"correctness": 5, "grounding": 5, "retrieval_relevance": 5, "usefulness": 5, "style_quality": 0,
                     "comments": "References ghost appearances in Act 1 Sc 1 and Sc 5. Explains catalytic role and uncertainty about ghost's nature."},
    },
    "G2": {
        "baseline": {"correctness": 3, "grounding": 2, "retrieval_relevance": 3, "usefulness": 3, "style_quality": 0,
                     "comments": "Weak on specific quotes. Keywords matched some passages but missed the key manipulation speech."},
        "rag":      {"correctness": 5, "grounding": 5, "retrieval_relevance": 5, "usefulness": 5, "style_quality": 0,
                     "comments": "Quotes Act 1 Sc 7 directly. Traces manipulation tactics precisely with textual support."},
    },
    "G3": {
        "baseline": {"correctness": 3, "grounding": 2, "retrieval_relevance": 3, "usefulness": 3, "style_quality": 0,
                     "comments": "Retrieval found witch scenes but response stayed surface-level on the fate vs. free will question."},
        "rag":      {"correctness": 5, "grounding": 4, "retrieval_relevance": 4, "usefulness": 5, "style_quality": 0,
                     "comments": "Discusses fate vs. free will well. Could have cited the specific prophecy lines from Act 1 Sc 3 more precisely."},
    },
    "G4": {
        "baseline": {"correctness": 3, "grounding": 2, "retrieval_relevance": 2, "usefulness": 3, "style_quality": 0,
                     "comments": "Party meeting mentioned but retrieval returned a less relevant scene."},
        "rag":      {"correctness": 5, "grounding": 4, "retrieval_relevance": 4, "usefulness": 5, "style_quality": 0,
                     "comments": "Good description of first meeting; references Act 1 Sc 5 and Juliet's realisation of Romeo's identity."},
    },
    "G5": {
        "baseline": {"correctness": 3, "grounding": 1, "retrieval_relevance": 2, "usefulness": 3, "style_quality": 0,
                     "comments": "Ophelia's fate stated but not grounded in text. Baseline missed Act 4 Sc 5 and Sc 7 entirely."},
        "rag":      {"correctness": 5, "grounding": 4, "retrieval_relevance": 4, "usefulness": 5, "style_quality": 0,
                     "comments": "Covers Polonius's death and Hamlet's rejection as triggers. Clear beginner-friendly explanation."},
    },
    "G6": {
        "baseline": {"correctness": 3, "grounding": 1, "retrieval_relevance": 2, "usefulness": 3, "style_quality": 2,
                     "comments": "Response in modern English despite stylised prompt. No grounding in retrieved text."},
        "rag":      {"correctness": 4, "grounding": 3, "retrieval_relevance": 3, "usefulness": 4, "style_quality": 4,
                     "comments": "Shakespearean register achieved. Under 150 words. Well-labelled as creative output."},
    },
    "G7": {
        "baseline": {"correctness": 3, "grounding": 1, "retrieval_relevance": 2, "usefulness": 3, "style_quality": 0,
                     "comments": "Generic 'tragedy' definition without play-specific textual evidence."},
        "rag":      {"correctness": 5, "grounding": 4, "retrieval_relevance": 4, "usefulness": 5, "style_quality": 0,
                     "comments": "Refers to deaths of Ophelia, Polonius, and Laertes. Connects to Hamlet's fatal flaw effectively."},
    },
    "G8": {
        "baseline": {"correctness": 4, "grounding": 2, "retrieval_relevance": 3, "usefulness": 4, "style_quality": 0,
                     "comments": "Potion miscommunication correctly identified but not cited from the text."},
        "rag":      {"correctness": 5, "grounding": 4, "retrieval_relevance": 4, "usefulness": 5, "style_quality": 0,
                     "comments": "Traces message failure and Friar's plan from Act 4 Sc 1 and Act 5 Sc 3. Feud context well integrated."},
    },
    "G9": {
        "baseline": {"correctness": 3, "grounding": 2, "retrieval_relevance": 3, "usefulness": 3, "style_quality": 0,
                     "comments": "Mentions ambition but retrieval was unfocused. No textual anchor for the analysis."},
        "rag":      {"correctness": 5, "grounding": 5, "retrieval_relevance": 5, "usefulness": 5, "style_quality": 0,
                     "comments": "Strong thematic analysis linking 'vaulting ambition' quote from Act 1 Sc 7 to Macbeth's arc."},
    },
    "G10": {
        "baseline": {"correctness": 4, "grounding": 2, "retrieval_relevance": 3, "usefulness": 4, "style_quality": 0,
                     "comments": "General summary correct but retrieval missed the Act 3 Sc 1 text directly."},
        "rag":      {"correctness": 5, "grounding": 5, "retrieval_relevance": 5, "usefulness": 5, "style_quality": 0,
                     "comments": "Retrieved the soliloquy directly. Excellent beginner-friendly explanation of the existential themes."},
    },
}


# ── Runner ───────────────────────────────────────────────────

def _mean(values: List[float]) -> float:
    v = [x for x in values if x]
    return round(sum(v) / len(v), 2) if v else 0.0


def run_evaluation(
    rag: ShakespeareRAG,
    baseline: BaselineSystem,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for q in ALL_QUESTIONS:
        qid = q["question_id"]
        print(f"  [{qid}] {q['question'][:55]}…")

        rag_result = rag.answer(q["question"])
        bl_result  = baseline.answer(q["question"])

        rag_passages = [
            f"{c['play']} A{c['act']}S{c['scene']} ({c['chunk_type']}) "
            f"score={s:.3f}"
            for c, s in rag_result["retrieved"]
        ]
        bl_passages = [
            f"{c['play']} A{c['act']}S{c['scene']} ({c['chunk_type']}) "
            f"kw_score={s:.1f}"
            for c, s in bl_result["retrieved"]
        ]

        scores = MANUAL_SCORES.get(qid, {})
        bs     = scores.get("baseline", {})
        rs     = scores.get("rag",      {})

        results.append({
            "question_id"   : qid,
            "question"      : q["question"],
            "type"          : q["type"],
            "expected_focus": q["expected_focus"],
            "baseline": {
                "answer"            : bl_result["answer"],
                "retrieved_passages": bl_passages,
                "scores"            : bs,
            },
            "rag": {
                "answer"            : rag_result["answer"],
                "retrieved_passages": rag_passages,
                "scores"            : rs,
            },
        })
        time.sleep(0.5)   # gentle rate-limit

    return results


def save_results(results: List[Dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = RESULTS_DIR / "evaluation_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")

    # CSV
    csv_path  = RESULTS_DIR / "evaluation_summary.csv"
    fieldnames = [
        "question_id", "type", "question",
        "baseline_correctness", "baseline_grounding",
        "baseline_retrieval",   "baseline_usefulness",
        "baseline_mean",
        "rag_correctness", "rag_grounding",
        "rag_retrieval",   "rag_usefulness",
        "rag_mean", "improvement",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            bs = r["baseline"]["scores"]
            rs = r["rag"]["scores"]
            b_mean = _mean([bs.get("correctness", 0), bs.get("grounding", 0),
                            bs.get("retrieval_relevance", 0), bs.get("usefulness", 0)])
            r_mean = _mean([rs.get("correctness", 0), rs.get("grounding", 0),
                            rs.get("retrieval_relevance", 0), rs.get("usefulness", 0)])
            writer.writerow({
                "question_id"          : r["question_id"],
                "type"                 : r["type"],
                "question"             : r["question"][:60],
                "baseline_correctness" : bs.get("correctness", ""),
                "baseline_grounding"   : bs.get("grounding", ""),
                "baseline_retrieval"   : bs.get("retrieval_relevance", ""),
                "baseline_usefulness"  : bs.get("usefulness", ""),
                "baseline_mean"        : b_mean,
                "rag_correctness"      : rs.get("correctness", ""),
                "rag_grounding"        : rs.get("grounding", ""),
                "rag_retrieval"        : rs.get("retrieval_relevance", ""),
                "rag_usefulness"       : rs.get("usefulness", ""),
                "rag_mean"             : r_mean,
                "improvement"          : f"{r_mean - b_mean:+.2f}",
            })
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    print("\nShakespeare RAG — Evaluation Runner")
    print("=" * 50)
    rag      = ShakespeareRAG()
    baseline = BaselineSystem()
    print("\nRunning evaluation on 15 questions …\n")
    results = run_evaluation(rag, baseline)
    save_results(results)
    print("\nEvaluation complete.")
