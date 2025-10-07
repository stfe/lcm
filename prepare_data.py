import argparse, json, os, random
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from constants import (
    DEFAULT_TRAIN_JSONL,
    DEFAULT_VAL_JSONL,
    DEFAULT_OUT_DIR,
    DEFAULT_PREPARE_DATA_MODEL_ID,
)
from common import build_prompt

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=DEFAULT_TRAIN_JSONL)
    ap.add_argument("--val", default=DEFAULT_VAL_JSONL)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", default=DEFAULT_PREPARE_DATA_MODEL_ID)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True) # set local_files_only to False when you need to load tokenizer from network
    # print(tok.special_tokens_map)
    # print(tok.all_special_tokens)


    def map_rows(rows):
        texts = []
        for r in rows:
            prompt = build_prompt(r["instruction"])
            target = r["command"].strip() + "\n"
            # Full supervised target (prompt + answer)
            text = prompt + target
            texts.append({"text": text})
        return texts

    train_rows = map_rows(load_jsonl(args.train))
    val_rows = map_rows(load_jsonl(args.val))

    ds = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(val_rows),
    })

    def tokenize(examples):
        return tok(examples["text"], truncation=True, max_length=512)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds_tok.save_to_disk(os.path.join(args.out_dir, "tokenized"))
    print("Saved tokenized dataset to:", os.path.join(args.out_dir, "tokenized"))

if __name__ == "__main__":
    main()
