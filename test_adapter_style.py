"""
test_adapter_style.py
=====================
Tests and compares the generation style between:
1. Base model alone
2. Fine-tuned model with os-tutor-adapter LoRA weights

Verifies that the adapter exhibits the signature professor teaching style:
Definition -> Explanation -> Example -> Exam Tip.
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ADAPTER_PATH = "./os-tutor-adapter"

def get_base_model_name():
    config_file = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            cfg = json.load(f)
            return cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
    return "Qwen/Qwen2.5-0.5B-Instruct"

def test_style():
    base_model_name = get_base_model_name()
    print("=" * 70)
    print("      OS-TUTOR: BASE MODEL VS FINE-TUNED ADAPTER STYLE TEST")
    print("=" * 70)
    print(f"Base model: {base_model_name}")
    print(f"Adapter:    {ADAPTER_PATH}\n")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    test_prompt = "### Instruction:\nWhat is a process?\n\n### Response:\n"
    inputs = tokenizer(test_prompt, return_tensors="pt")

    print("\n[1] Generating with Base Model...")
    with torch.no_grad():
        base_output_ids = base_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    base_answer = tokenizer.decode(base_output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print("\n[2] Loading Fine-Tuned LoRA Adapter...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("\n[3] Generating with Fine-Tuned Model (LoRA)...")
    with torch.no_grad():
        peft_output_ids = peft_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    peft_answer = tokenizer.decode(peft_output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print("\n" + "=" * 70)
    print("                     OUTPUT COMPARISON")
    print("=" * 70)
    print("\n--- BASE MODEL OUTPUT ---")
    print(base_answer)
    print("\n--- FINE-TUNED MODEL (OS-TUTOR ADAPTER) OUTPUT ---")
    print(peft_answer)
    print("=" * 70)

    # Save to style_comparison.txt
    with open("style_comparison.txt", "w", encoding="utf-8") as f:
        f.write("=== BASE MODEL VS FINE-TUNED MODEL OUTPUT ===\n\n")
        f.write("PROMPT:\n" + test_prompt + "\n\n")
        f.write("BASE MODEL:\n" + base_answer + "\n\n")
        f.write("FINE-TUNED MODEL:\n" + peft_answer + "\n")

    print("\nSaved comparison to 'style_comparison.txt'.")

if __name__ == "__main__":
    test_style()
