"""
generate_dataset.py
===================
Dataset Generation & Expansion Tool for OS-Tutor.

This script extracts chunks from the local ChromaDB vector store (or course notes)
and uses a local LLM (via Ollama) to draft candidate fine-tuning pairs in Alpaca format:
    {"instruction": ..., "input": "", "output": ...}

Each generated candidate follows the professor's signature teaching pedagogy:
  - Concept Explanations: Definition -> Intuitive Explanation -> Real-World Example -> Exam Tip
  - Multiple Choice (MCQ): Question Stem -> 4 Options (A-D) -> Correct Answer -> Explanation -> Exam Tip
  - Short Answer / Viva: Direct answer -> Technical detail -> Key distinction / Exam Trap

Usage:
  python generate_dataset.py --generate --limit 5   # Draft 5 new candidates to candidates_review.json
  python generate_dataset.py --merge                # Merge candidates_review.json into dataset.json
"""

import json
import os
import argparse
import chromadb
import ollama

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "os_notes"
REVIEW_FILE = "candidates_review.json"
DATASET_FILE = "dataset.json"

EXPLANATION_PROMPT = """You are an experienced Operating Systems Professor creating high-yield training data for an AI Teaching Assistant.
Based on the course excerpt below, create ONE question and an exemplary teaching answer.

Course Excerpt:
\"\"\"{chunk}\"\"\"

Format your output STRICTLY as valid JSON with three keys:
{{
  "instruction": "<Clear student question about a core concept in the text>",
  "input": "",
  "output": "<Definition of the concept>. <Intuitive explanation of how it works>. Example: <concrete real-world or programming example>. Exam tip: <common student misconception or high-yield exam pitfall>."
}}

Respond ONLY with the JSON object. Do not add markdown wrappers or explanation.
"""

MCQ_PROMPT = """You are an Operating Systems Professor crafting a multiple-choice exam question based on this course excerpt.

Course Excerpt:
\"\"\"{chunk}\"\"\"

Format your output STRICTLY as valid JSON with three keys:
{{
  "instruction": "<Question stem with 4 options labeled A), B), C), D)>",
  "input": "",
  "output": "Correct Answer: <Option letter and text>. Explanation: <Why the correct choice is right and others are wrong>. Exam tip: <Key trick or terminology to watch out for>."
}}

Respond ONLY with the JSON object. Do not add markdown wrappers or explanation.
"""

VIVA_PROMPT = """You are an Operating Systems Professor conducting an oral viva or technical interview.
Based on the course excerpt below, craft a concise technical viva question and the ideal response.

Course Excerpt:
\"\"\"{chunk}\"\"\"

Format your output STRICTLY as valid JSON with three keys:
{{
  "instruction": "<Direct technical viva/interview question>",
  "input": "",
  "output": "<Direct one-sentence technical answer>. <Key underlying OS mechanism or tradeoff>. Exam tip: <Precise terminology examiners look for>."
}}

Respond ONLY with the JSON object. Do not add markdown wrappers or explanation.
"""

def get_chroma_chunks(limit=10):
    """Retrieve random or sequential chunks from persistent ChromaDB collection."""
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"ChromaDB not found at {CHROMA_PATH}. Run build_db.py first.")
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION_NAME)
    results = col.get()
    
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    chunks = []
    for doc, meta in zip(docs[:limit], metas[:limit]):
        chunks.append({"text": doc, "meta": meta})
    return chunks

def draft_candidates(limit=5, model="qwen2.5:3b"):
    """
    Generate candidate examples across Explanation, MCQ, and Viva styles using Ollama.
    Appends generated candidates to candidates_review.json for manual inspection.
    """
    chunks = get_chroma_chunks(limit=limit)
    candidates = []
    
    if os.path.exists(REVIEW_FILE):
        try:
            with open(REVIEW_FILE, "r", encoding="utf-8") as f:
                candidates = json.load(f)
        except Exception:
            candidates = []

    print(f"Drafting {len(chunks)} candidate questions using {model}...")
    
    prompts = [EXPLANATION_PROMPT, MCQ_PROMPT, VIVA_PROMPT]
    
    for i, item in enumerate(chunks):
        chunk_text = item["text"]
        prompt_template = prompts[i % len(prompts)]
        formatted_prompt = prompt_template.format(chunk=chunk_text[:700])
        
        try:
            res = ollama.generate(model=model, prompt=formatted_prompt)
            raw = res["response"].strip()
            # Clean possible markdown formatting
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            
            entry = json.loads(raw)
            if "instruction" in entry and "output" in entry:
                entry["input"] = entry.get("input", "")
                entry["_source_file"] = item["meta"].get("file", "unknown")
                entry["_source_page"] = item["meta"].get("page", 0)
                candidates.append(entry)
                print(f"[{i+1}/{len(chunks)}] Generated candidate: {entry['instruction'][:60]}...")
        except Exception as e:
            print(f"[{i+1}/{len(chunks)}] Skipped chunk due to parsing error: {e}")

    with open(REVIEW_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)
    
    print(f"\nSaved {len(candidates)} candidates to '{REVIEW_FILE}'.")
    print("Review them manually or run `python generate_dataset.py --merge` to merge into dataset.json.")

def merge_review_file():
    """Merge reviewed candidates into dataset.json without duplicates."""
    if not os.path.exists(REVIEW_FILE):
        print(f"No {REVIEW_FILE} found. Nothing to merge.")
        return

    with open(REVIEW_FILE, "r", encoding="utf-8") as f:
        new_items = json.load(f)

    existing = []
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_instructions = {item["instruction"].strip().lower() for item in existing}
    added_count = 0

    for item in new_items:
        clean_item = {
            "instruction": item["instruction"].strip(),
            "input": item.get("input", "").strip(),
            "output": item["output"].strip()
        }
        if clean_item["instruction"].lower() not in existing_instructions:
            existing.append(clean_item)
            existing_instructions.add(clean_item["instruction"].lower())
            added_count += 1

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    print(f"Merged {added_count} new examples into '{DATASET_FILE}'. Total dataset count: {len(existing)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and review OS-Tutor fine-tuning dataset.")
    parser.add_argument("--generate", action="store_true", help="Generate candidate questions using Ollama")
    parser.add_argument("--limit", type=int, default=5, help="Number of chunks to generate from")
    parser.add_argument("--merge", action="store_true", help="Merge approved candidates into dataset.json")
    args = parser.parse_args()

    if args.generate:
        draft_candidates(limit=args.limit)
    elif args.merge:
        merge_review_file()
    else:
        parser.print_help()
