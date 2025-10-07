
# llm-linux-cmdgen

Tiny LLM that turns natural language into a single Linux command. This README explains:
1) How to prepare data (two scripts, with examples)
2) How to train
3) How to run inference via infer.py, infert_test.py, and cli.py
4) How to convert a trained model + adapter to GGUF
5) A table describing each file in this project

## Setup
- Python 3.10+
- Create and activate a virtual environment, then install requirements:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- (Optional) Download the base model locally to speed up repeated runs:

```bash
python download_model.py
```

The defaults target TinyLlama/TinyLlama-1.1B-Chat-v1.0. You can change defaults in constants.py.

## Prepare data
We support two data preparation paths: a local JSONL pair format and the public NL2SH-ALFA dataset.

### Local JSONL format (prepare_data.py)
- Expected files:
  - data/train.jsonl
  - data/val.jsonl
- Each line format:

```json
{"instruction": "list files including hidden", "command": "ls -la"}
```

- Run:

```bash
# Uses tokenizer from constants.DEFAULT_PREPARE_DATA_MODEL_ID
python prepare_data.py \
  --train data/train.jsonl \
  --val data/val.jsonl \
  --out_dir outputs \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Result: tokenized dataset saved to outputs/tokenized
```

Notes:
- Prompts are built via common.build_prompt to force “one line only”.
- The tokenized dataset is saved using Hugging Face Datasets (save_to_disk).

### NL2SH-ALFA dataset (prepare_data_nl2sh_alpha.py)
This script pulls the public westenfelder/NL2SH-ALFA dataset and converts it to our tokenized format.

```bash
# By default uses local tokenizer files; pass --local_only False to allow download
python prepare_data_nl2sh_alpha.py \
  --out_dir outputs \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local_only False

# Result: tokenized dataset saved to outputs/tokenized_nl2sh_alpha
```

## Train
Training uses QLoRA (4-bit base + LoRA adapters). Defaults are in constants.py.

```bash
# Minimal example using tokenized NL2SH-ALFA produced above
python train.py \
  --base_model /path/to/local/TinyLlama-1.1B-Chat-v1.0 \
  --data_dir outputs/tokenized_nl2sh_alpha \
  --out_dir outputs/tinyllama_cmdgen_nl2sh_alpha \
  --epochs 5 --bsz 8 --grad_accum 4 --lr 2e-4 --bf16 --bnb_4bit
```

Outputs:
- The PEFT adapter and tokenizer are saved to the out_dir you provide (e.g., outputs/tinyllama_cmdgen_nl2sh_alpha).

## Inference
You can run inference with either infer.py directly, the small CLI wrapper, or via a simple unit test harness.

### infer.py
```bash
python infer.py \
  --adapter outputs/tinyllama_cmdgen_nl2sh_alpha \
  --prompt "show listening TCP ports"
# Optional: override base model if not auto-detected
# --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
Behavior:
- Returns exactly one command line (first non-empty line of the model output, post-processed).
- safe_filter.py blocks obviously destructive commands; edit the list to suit your environment.

### CLI wrapper (cli.py)
This wraps infer.py for convenience.

```bash
python cli.py "list all files including hidden" \
  --adapter outputs/tinyllama_cmdgen_nl2sh_alpha

# Add --run to execute the produced command locally (use with care!)
python cli.py "show kernel version" --adapter outputs/tinyllama_cmdgen_nl2sh_alpha --run
```

### Inference test (infert_test.py)
A simple unittest that invokes infer.py as a subprocess to validate that a plausible command is returned.

```bash
python -m unittest infert_test.py
```

Note: The file is named infert_test.py (with a ‘t’). If you expected infer_test.py, this is the correct filename in this repo.

## Convert trained model + adapter to GGUF
To deploy with llama.cpp, first merge the LoRA adapter into the base model, then convert to GGUF.

Step 1: Merge adapter into base weights
```bash
python merge_and_prepare_gguf.py \
  --adapter outputs/tinyllama_cmdgen_nl2sh_alpha \
  --output outputs/merged_model
# Optionally specify --base_model if it cannot be auto-detected
```

Step 2: Convert to GGUF and quantize (requires llama.cpp)
```bash
# Clone and build llama.cpp once
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make && cd -

# Convert merged HF model to GGUF (FP16 example)
python llama.cpp/convert_hf_to_gguf.py outputs/merged_model \
  --outfile outputs/tinyllama_cmdgen.gguf \
  --outtype f16

# Quantize to Q4_K_M (good balance)
./llama.cpp/llama-quantize outputs/tinyllama_cmdgen.gguf outputs/tinyllama_cmdgen-Q4_K_M.gguf Q4_K_M
```

## Project files
A brief description of each top-level file and how to use it.

| File | Description / How to use |
| --- | --- |
| README.md | This document. |
| requirements.txt | Python dependencies; install with pip install -r requirements.txt. |
| constants.py | Centralized defaults (paths, model IDs, hyperparameters). Adjust to your setup. |
| common.py | Utility to build the standardized prompt used across data prep and inference. |
| data/ | Place your train.jsonl and val.jsonl here (for local data prep). |
| prepare_data.py | Converts local JSONL pairs to a tokenized dataset; see section 1.a. |
| prepare_data_nl2sh_alpha.py | Downloads and tokenizes the NL2SH-ALFA dataset; see section 1.b. |
| download_model.py | Downloads the base model snapshot to a local directory (see constants.DEFAULT_LOCAL_DOWNLOAD_DIR). |
| train.py | QLoRA training script; saves a PEFT adapter and tokenizer; see section 2. |
| infer.py | Generates exactly one Linux command from a natural-language prompt; see section 3.a. |
| cli.py | Thin wrapper around infer.py; adds optional --run to execute the produced command; see section 3.b. |
| infert_test.py | Unittest that sanity-checks inference; run with python -m unittest infert_test.py; see section 3.c. |
| safe_filter.py | Simple denylist filter that blocks destructive commands at inference time; edit as needed. |
| merge_and_prepare_gguf.py | Merges LoRA adapter into base model and prints next steps for GGUF; see section 4. |
| outputs/ | Default output directory for tokenized data, training artifacts, and merged models. |

## Data and safety tips
- Add diverse examples that reflect your environment (package manager, service manager, filesystem layout).
- Prefer single commands; use && or pipes only when truly necessary.
- Review and customize safe_filter.py. Consider adding an explicit override mechanism if you need one.

## License
MIT
