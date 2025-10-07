import socket
# download_model.py
# Monkey-patch to force IPv4
original_getaddrinfo = socket.getaddrinfo

def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = ipv4_only_getaddrinfo

import argparse, os
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
from huggingface_hub.constants import HF_HOME
from constants import (
    DEFAULT_OUT_DIR,
    DEFAULT_PREPARE_DATA_MODEL_ID,
)
from common import build_prompt

DATASETS_CACHE_DIR = os.path.join(HF_HOME, "datasets")
TRAIN_CACHE_DIR = f"{DATASETS_CACHE_DIR}/westenfelder___nl2_sh-alfa/train"
TEST_CACHE_DIR = f"{DATASETS_CACHE_DIR}/westenfelder___nl2_sh-alfa/test"


def main():
    ap = argparse.ArgumentParser(description="Prepare NL2SH-ALFA dataset (tokenized)")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Base output directory (default: outputs)")
    ap.add_argument("--model", default=DEFAULT_PREPARE_DATA_MODEL_ID, help="Tokenizer model name or path")
    ap.add_argument("--local_only", default=True, action="store_true", help="Use only local tokenizer files (no network)")
    args = ap.parse_args()

    save_dir = os.path.join(args.out_dir, "tokenized_nl2sh_alpha")
    os.makedirs(save_dir, exist_ok=True)

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=args.local_only)

    # Load datasets from Hugging Face as requested
    train_dataset = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
    test_dataset = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")
    # train_dataset = load_from_disk(f"{DATASETS_CACHE_DIR}/westenfelder___nl2_sh-alfa/train")
    # test_dataset = load_from_disk(f"{DATASETS_CACHE_DIR}/westenfelder___nl2_sh-alfa/test")

    # Map rows to prompt + target text
    def map_rows_rowwise(example):
        prompt = build_prompt(example['nl'])
        target = (example['bash'] or "").strip() + "\n"
        return {"text": prompt + target}

    train_text = train_dataset.map(map_rows_rowwise)
    test_text = test_dataset.map(map_rows_rowwise)

    ds = DatasetDict({
        "train": Dataset.from_dict({"text": train_text["text"]}),
        "test": Dataset.from_dict({"text": test_text["text"]}),
    })

    # Tokenize
    def tokenize(examples):
        return tok(examples["text"], truncation=True, max_length=512)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])  # drops raw text

    # Save to disk
    ds_tok.save_to_disk(save_dir)
    print("Saved tokenized dataset to:", save_dir)


if __name__ == "__main__":
    main()
