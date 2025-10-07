# Centralized default constants for the project

# Prompting
SYSTEM_PROMPT = (
    "You are a command generator. Respond with ONE Linux command only. "
    "No quotes, no code fences, no explanations."
)

# Inference defaults
DEFAULT_ADAPTER_DIR = "outputs/tinyllama_cmdgen_nl2sh_alpha"
DEFAULT_BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9

# Data preparation defaults
DEFAULT_TRAIN_JSONL = "data/train.jsonl"
DEFAULT_VAL_JSONL = "data/val.jsonl"
DEFAULT_OUT_DIR = "outputs"
DEFAULT_PREPARE_DATA_MODEL_ID = DEFAULT_BASE_MODEL_ID

# Training defaults
DEFAULT_LOCAL_BASE_MODEL_PATH = f"base_models/{DEFAULT_BASE_MODEL_ID}"
DEFAULT_TOKENIZED_DATA_DIR = "outputs/tokenized_nl2sh_alpha"
DEFAULT_TRAIN_OUT_DIR = "outputs/tinyllama_cmdgen_nl2sh_alpha"
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 5
DEFAULT_BSZ = 8
DEFAULT_GRAD_ACCUM = 4
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_BNB_4BIT = True
DEFAULT_BF16 = True

# Model download defaults
DEFAULT_REPO_ID = DEFAULT_BASE_MODEL_ID
DEFAULT_LOCAL_DOWNLOAD_DIR = f"downloads/models/{DEFAULT_BASE_MODEL_ID}"

# Merging and GGUF conversion defaults
DEFAULT_MERGED_OUTPUT_DIR = "outputs/merged_model"
DEFAULT_DEVICE = "auto"
