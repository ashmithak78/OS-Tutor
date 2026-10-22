"""
train.py
========
End-to-End QLoRA / LoRA Fine-Tuning Pipeline for OS-Tutor.

This script fine-tunes Qwen2.5 on the OS teaching assistant dataset:
- On CUDA (e.g. Google Colab / Linux GPU): Uses 4-bit QLoRA via bitsandbytes (NF4 quantization)
- On Apple Silicon (MPS) or CPU: Automatically falls back to LoRA in float16/float32 without crashing
- Targets LoRA attention projections: ["q_proj", "k_proj", "v_proj", "o_proj"]
- Formats prompts with Alpaca instruction-response templates
- Handles modern TRL (1.x) and pinned TRL (0.11.4) SFTConfig/SFTTrainer parameter compatibility
- Logs loss at every step to verify decreasing loss
- Exports the trained adapter to `os-tutor-adapter/`

Usage:
  python train.py --epochs 3 --batch_size 2 --lr 2e-4
"""

import os
import argparse
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import trl
from trl import SFTTrainer

# Check TRL version for parameter compatibility
TRL_VERSION = tuple(map(int, trl.__version__.split(".")[:2]))

def format_prompts(batch):
    """
    Format instruction-input-output triplets into the prompt template
    matching Alpaca / OS-Tutor inference expectations.
    """
    texts = []
    for instruction, input_text, output in zip(batch["instruction"], batch["input"], batch["output"]):
        if input_text and input_text.strip():
            text = (
                f"### Instruction:\n{instruction.strip()}\n\n"
                f"### Input:\n{input_text.strip()}\n\n"
                f"### Response:\n{output.strip()}"
            )
        else:
            text = (
                f"### Instruction:\n{instruction.strip()}\n\n"
                f"### Response:\n{output.strip()}"
            )
        texts.append(text)
    return {"text": texts}

def load_data(train_path="train.json", val_path="val.json"):
    """Load JSON files and convert to HuggingFace Dataset objects."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}. Run split_dataset.py first.")

    with open(train_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    train_ds = Dataset.from_list(train_raw).map(format_prompts, batched=True)

    val_ds = None
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            val_raw = json.load(f)
        val_ds = Dataset.from_list(val_raw).map(format_prompts, batched=True)

    print(f"Loaded {len(train_ds)} training samples and {len(val_ds) if val_ds else 0} validation samples.")
    return train_ds, val_ds

def run_training(args):
    print("=" * 60)
    print("           OS-TUTOR FINE-TUNING PIPELINE")
    print("=" * 60)
    print(f"Base model: {args.model_name}")
    print(f"Adapter output directory: {args.output_dir}")

    # Determine hardware acceleration and quantization
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    print(f"Hardware detection: CUDA={has_cuda}, MPS={has_mps}")

    bnb_config = None
    if has_cuda:
        try:
            from transformers import BitsAndBytesConfig
            print("Configuring 4-bit NF4 QLoRA quantization for CUDA GPU...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            print("bitsandbytes not installed; proceeding with standard precision.")
    else:
        print("Non-CUDA environment detected (Apple Silicon MPS / CPU).")
        print("Using standard precision with LoRA adapters (no bitsandbytes requirement).")

    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Base Model
    print("Loading base model...")
    model_kwargs = {
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        # Stable CPU load for macOS Python 3.14 to prevent MPS backend C++ segfault
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("Configuring LoRA adapter hyperparameters...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and prepare datasets
    train_ds, val_ds = load_data(args.train_file, args.val_file)

    # Setup SFTConfig dynamically supporting both TRL 0.11.4 and modern TRL versions
    from trl import SFTConfig
    import inspect
    sft_sig = inspect.signature(SFTConfig.__init__).parameters

    config_kwargs = {
        "output_dir": "./training_checkpoints",
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "eval_strategy": "no",
        "fp16": has_cuda,
        "report_to": "none",
        "optim": "adamw_torch",
    }
    if "dataset_text_field" in sft_sig:
        config_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_sig:
        config_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sft_sig:
        config_kwargs["max_length"] = args.max_seq_length

    training_args = SFTConfig(**config_kwargs)

    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "args": training_args,
    }
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    class OSProfessorTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Fall back directly to Transformers Trainer compute_loss to avoid triton fused_linear on non-CUDA
            try:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)
            except AttributeError:
                from transformers import Trainer
                return Trainer.compute_loss(self, model, inputs, return_outputs=return_outputs)

    trainer = OSProfessorTrainer(**trainer_kwargs)

    print("\nStarting training loop...")
    train_result = trainer.train()

    print("\nTraining complete! Metrics:")
    for key, val in train_result.metrics.items():
        print(f"  {key}: {val}")

    # Save and Export Adapter
    print(f"\nSaving LoRA adapter to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Also write a metadata card for project tracking
    metadata = {
        "base_model": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "train_loss": train_result.training_loss,
        "train_samples": len(train_ds),
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully exported fine-tuned adapter to {args.output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune OS-Tutor with QLoRA/LoRA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base HuggingFace model")
    parser.add_argument("--train_file", type=str, default="train.json", help="Path to train JSON")
    parser.add_argument("--val_file", type=str, default="val.json", help="Path to val JSON")
    parser.add_argument("--output_dir", type=str, default="os-tutor-adapter", help="Directory to save adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (-1 for full epochs)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log loss every N steps")
    args = parser.parse_args()

    run_training(args)
