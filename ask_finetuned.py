"""
ask_finetuned.py
================
RAG + Fine-Tuned Model Inference Pipeline for OS-Tutor.

Features:
1. ChromaDB Retrieval: Encodes user query with all-MiniLM-L6-v2 and retrieves top-3 chunks.
2. Confidence/Similarity Threshold: Rejects irrelevant queries (distance > threshold)
   with "Not found in course material."
3. Intent Routing: Classifies request into explain / mcq / lab / viva and applies tailored prompt templates.
4. Model Loading: Loads base model + PEFT LoRA adapter (CUDA 4-bit QLoRA, or CPU/MPS fallback).
   Also supports local Ollama inference fallback if running on low-resource environments.
5. Citing: Appends exact [file, page] citations.

Usage:
  python ask_finetuned.py                  # Interactive terminal loop
  python ask_finetuned.py --mode mcq       # Direct MCQ generation mode
"""

import os
import re
import json
import argparse
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Force offline mode for transformers/huggingface to avoid DNS timeouts
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Configuration
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "os_notes"
DEFAULT_ADAPTER_PATH = "./os-tutor-adapter"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DISTANCE_THRESHOLD = 1.35  # Cosine distance cutoff (lower is more similar)

def extract_core_topic(query):
    """
    Strips out conversational prefix/suffix noise (e.g. 'Explain...', '...with an example')
    and expands key operating systems abbreviations to maximize vector retrieval accuracy.
    """
    clean = query.strip()
    prefix_patterns = [
        r"^(please\s+)?(generate|create|make|give\s+me|ask\s+me)\s+(an?\s+)?(mcq|quiz|multiple\s+choice|question|viva\s+question|lab\s+exercise|coding\s+task)\s+(on|about|for)?\s*",
        r"^(can\s+you\s+)?(explain|describe|tell\s+me\s+about|quiz\s+me\s+on)\s*",
        r"^(what\s+is|what\s+are|define)\s+",
        r"^(give\s+me\s+a\s+viva\s+question\s+(on|about)?)\s*",
    ]
    for p in prefix_patterns:
        clean = re.sub(p, "", clean, flags=re.IGNORECASE).strip()

    suffix_patterns = [
        r"\s+(with\s+(an?\s+)?(example|calculations?|steps?|diagram|gantt\s+chart)).*$",
        r"\s+(using\s+(an?\s+)?(example|scenario)).*$",
        r"\s+(in\s+detail|briefly|step\s+by\s+step).*$",
    ]
    for p in suffix_patterns:
        clean = re.sub(p, "", clean, flags=re.IGNORECASE).strip()

    # Expand common OS abbreviations and synonyms to bridge terminology in textbook chunks
    synonyms = {
        r"\bround[-\s]robin(\s+scheduling)?\b": "Round-Robin RR time-slicing scheduling",
        r"\brr\s+scheduling\b": "Round-Robin RR scheduling time slice",
        r"\bsjf(\s+scheduling)?\b": "Shortest Job First SJF scheduling",
        r"\bstcf(\s+scheduling)?\b": "Shortest Time-to-Completion First STCF PSJF",
        r"\bmlfq(\s+scheduling)?\b": "Multi-Level Feedback Queue MLFQ scheduling",
        r"\bfifo(\s+scheduling)?\b": "First-In First-Out FIFO FCFS scheduling",
        r"\btlb\b": "Translation Lookaside Buffer TLB paging",
        r"\bpcb\b": "Process Control Block PCB task_struct",
    }
    for pat, rep in synonyms.items():
        if re.search(pat, clean, flags=re.IGNORECASE):
            clean = re.sub(pat, rep, clean, flags=re.IGNORECASE)
            break

    return clean if len(clean) >= 3 else query

# Initialize sentence transformer and ChromaDB client
print("Initializing retrieval engine...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

# Concise, Guardrailed Prompt Templates by Mode
TEMPLATES = {
    "explain": """You are OS-Tutor, an Operating Systems Teaching Assistant.
Answer the student's question using ONLY the course context below.

Formatting & Accuracy Rules:
1. Definition: Provide a direct, precise definition in 1 to 3 sentences.
2. Explanation: Provide a single short paragraph explaining only the essential mechanism. Do NOT repeat or rephrase the definition.
3. Example (if asked or helpful): Use a small, logically and mathematically accurate example (e.g., 2-3 processes).
   - For CPU scheduling: Strictly respect Process Arrival Times (T_arr), remaining Burst Times (T_rem), Time Quantum (Q), and FIFO Ready Queue order.
   - A process executes for at most min(Q, T_rem) and NEVER beyond its remaining burst time. When T_rem reaches 0, the process terminates immediately.
   - Trace time steps accurately (e.g. [0-2], [2-4]) with exact remaining times.
4. Do NOT include exam tips, extraneous commentary, or unrelated information.
5. Always cite the exact source: [os.pdf, page X].
If the topic is not covered in the context, say "Not found in course material."

Context:
{context}

Question: {question}

Answer:""",

    "mcq": """You are OS-Tutor, an Operating Systems Professor.
Based ONLY on the course context below, generate a challenging multiple-choice question testing this concept.

Format your response exactly as:
Question: <Question text>
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>
Correct Answer: <Letter and full text of correct choice>
Explanation: <Why the correct choice is right and others are wrong, citing [os.pdf, page X]>

Context:
{context}

Topic / Request: {question}

Response:""",

    "lab": """You are OS-Tutor, an Operating Systems Lab Instructor.
Based ONLY on the course context below, create a practical systems programming lab question.

Format your response exactly as:
Lab Objective: <What the student will implement or observe>
Problem Statement: <Coding or command-line exercise in C / UNIX, e.g., fork(), exec(), wait(), or /proc>
Hints & System Calls: <Key system calls, headers, and flags to use>
Expected Behavior: <How to test and verify the output>
Reference: [os.pdf, page X]

Context:
{context}

Topic / Request: {question}

Response:""",

    "viva": """You are OS-Tutor, an Operating Systems Professor conducting an oral viva or technical interview.
Based ONLY on the course context below, generate a rigorous viva question.

Format your response exactly as:
Viva Question: <Direct technical question examiners ask>
Ideal Answer: <Concise 1-2 sentence response demonstrating deep understanding>
Key Technical Distinction: <Crucial concept examiners check for>
Reference: [os.pdf, page X]

Context:
{context}

Topic / Request: {question}

Response:"""
}

def classify_intent(query):
    """
    Classifies user prompt into one of: explain, mcq, lab, viva.
    Uses regex keyword matching with default to 'explain'.
    """
    q_lower = query.lower()
    if re.search(r"\b(mcq|multiple choice|quiz|quiz me|options|choose the correct)\b", q_lower):
        return "mcq"
    if re.search(r"\b(lab|code|coding|c program|implement|exercise|assignment|hands-on)\b", q_lower):
        return "lab"
    if re.search(r"\b(viva|interview|oral|examiner|trap|oral question)\b", q_lower):
        return "viva"
    return "explain"

class ModelManager:
    """Manages loading base model + LoRA adapter or falling back to Ollama."""
    def __init__(self, adapter_path=DEFAULT_ADAPTER_PATH):
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.base_model_name = self._resolve_base_model()

    def _resolve_base_model(self):
        config_path = os.path.join(self.adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return cfg.get("base_model_name_or_path", DEFAULT_BASE_MODEL)
            except Exception:
                pass
        return DEFAULT_BASE_MODEL

    def load_model(self):
        """Loads HuggingFace model with LoRA adapter."""
        if self.model is not None:
            return

        print(f"Loading tokenizer and model ({self.base_model_name}) with adapter ({self.adapter_path})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path if os.path.exists(self.adapter_path) else self.base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # CPU / Apple Silicon load
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

        if os.path.exists(self.adapter_path) and os.path.exists(os.path.join(self.adapter_path, "adapter_config.json")):
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            print("LoRA adapter loaded successfully!")
        else:
            print("No adapter found at path; using base model directly.")
            self.model = base_model

        self.model.eval()

    def generate(self, prompt, max_new_tokens=250):
        """Generate concise response using the loaded PEFT model."""
        self.load_model()
        formatted = f"### Instruction:\n{prompt.strip()}\n\n### Response:\n"
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Stop at Alpaca prompt delimiters to avoid repetition
        for stop_marker in ["\n###", "### Instruction:", "### Response:", "### Context:", "### Question:"]:
            if stop_marker in raw_output:
                raw_output = raw_output.split(stop_marker)[0].strip()

        return raw_output


# Global model manager singleton
model_manager = ModelManager()

def retrieve_context(question, n_results=3):
    """
    Queries ChromaDB and applies distance threshold checking.
    Uses extract_core_topic to remove intent prefixes so vector similarity accurately
    matches course notes.
    Returns: (context_string, citations_list, min_distance, is_confident)
    """
    search_query = extract_core_topic(question)
    q_embedding = embed_model.encode([search_query]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=n_results)

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if "distances" in results and results["distances"] else []

    if not docs:
        return "", [], float("inf"), False

    min_distance = min(distances) if distances else 0.0
    is_confident = (min_distance <= DISTANCE_THRESHOLD)

    context = ""
    citations = []
    for doc, meta, dist in zip(docs, metas, distances):
        context += f"[{meta['file']}, page {meta['page']}]: {doc}\n\n"
        citations.append({
            "file": meta.get("file", "unknown"),
            "page": meta.get("page", 0),
            "distance": round(float(dist), 4)
        })

    return context, citations, min_distance, is_confident

def answer_question(question, mode=None, use_ollama_fallback=False):
    """
    Full pipeline: Intent routing -> Retrieval -> Threshold check -> Prompting -> Generation.
    """
    if mode is None:
        mode = classify_intent(question)
    mode = mode.lower()
    if mode not in TEMPLATES:
        mode = "explain"

    context, citations, min_dist, is_confident = retrieve_context(question, n_results=3)

    # Confidence Threshold Check:
    if not is_confident:
        return {
            "answer": "Not found in course material.",
            "mode": mode,
            "citations": [],
            "distance": min_dist,
            "found": False
        }

    template = TEMPLATES[mode]
    prompt = template.format(context=context, question=question)

    answer = None
    if use_ollama_fallback:
        try:
            import ollama
            res = ollama.generate(model="qwen2.5:3b", prompt=prompt)
            answer = res["response"].strip()
        except Exception:
            # Fall back to local fine-tuned PEFT model if Ollama server is offline
            answer = model_manager.generate(prompt)
    else:
        answer = model_manager.generate(prompt)

    return {
        "answer": answer,
        "mode": mode,
        "citations": citations,
        "distance": min_dist,
        "found": True
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OS-Tutor RAG + Fine-Tuned Chatbot")
    parser.add_argument("--mode", type=str, choices=["explain", "mcq", "lab", "viva", "auto"], default="auto")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama backend instead of local HF PEFT")
    parser.add_argument("--threshold", type=float, default=1.25, help="Distance cutoff threshold")
    args = parser.parse_args()

    if args.threshold:
        DISTANCE_THRESHOLD = args.threshold

    print("\n" + "=" * 60)
    print("      OS-TUTOR: AI TEACHING ASSISTANT FOR OPERATING SYSTEMS")
    print(f"      Mode: {args.mode.upper()} | Backend: {'Ollama' if args.ollama else 'Fine-Tuned Adapter'}")
    print("=" * 60)
    print("Ask any OS question (type 'exit' to quit):\n")

    while True:
        try:
            q = input("\nStudent: ")
            if not q.strip():
                continue
            if q.lower() in ["exit", "quit", "q"]:
                print("Exiting OS-Tutor. Goodbye!")
                break

            selected_mode = None if args.mode == "auto" else args.mode
            result = answer_question(q, mode=selected_mode, use_ollama_fallback=args.ollama)

            print(f"\n[OS-Tutor ({result['mode'].upper()})]:")
            print(result["answer"])
            if result["citations"]:
                print("\nSources Cited:")
                for c in result["citations"]:
                    print(f"  • {c['file']}, Page {c['page']} (distance: {c['distance']})")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
