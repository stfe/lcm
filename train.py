import argparse, os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
from constants import (
    DEFAULT_LOCAL_BASE_MODEL_PATH,
    DEFAULT_TOKENIZED_DATA_DIR,
    DEFAULT_TRAIN_OUT_DIR,
    DEFAULT_LR,
    DEFAULT_EPOCHS,
    DEFAULT_BSZ,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_LORA_R,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_BNB_4BIT,
    DEFAULT_BF16,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=DEFAULT_LOCAL_BASE_MODEL_PATH)
    ap.add_argument("--data_dir", default=DEFAULT_TOKENIZED_DATA_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_TRAIN_OUT_DIR)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--bsz", type=int, default=DEFAULT_BSZ)
    ap.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    ap.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    ap.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    ap.add_argument("--bnb_4bit", action="store_true", default=DEFAULT_BNB_4BIT)
    ap.add_argument("--bf16", action="store_true", default=DEFAULT_BF16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    ds = load_from_disk(args.data_dir)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Model with 4-bit quantization (QLoRA)
    bnb_config = None
    model_kwargs = {}
    if args.bnb_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], # broad, covers most archs
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=400,
        save_steps=800,
        save_total_limit=2,
        bf16=args.bf16,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        report_to="none",
        gradient_checkpointing=True,
    )

    def data_collator(features):
        # Left as default for causal LM; relying on tokenizer padding
        batch = {k: [f[k] for f in features] for k in features[0].keys()}
        batch = tok.pad(batch, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        batch["labels"] = labels
        return batch

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        tokenizer=tok,
    )

    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Saved adapter + tokenizer to", args.out_dir)

if __name__ == "__main__":
    main()
