import argparse, sys, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from safe_filter import is_safe
from constants import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from common import build_prompt

def postprocess_to_single_command(text: str) -> str:
    text = text.replace("```", "").replace("\u200b", " ")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    cmd = lines[0]
    if (cmd.startswith("'") and cmd.endswith("'")) or (cmd.startswith('"') and cmd.endswith('"')):
        cmd = cmd[1:-1].strip()
    cmd = re.sub(r"\s+#.*$", "", cmd)
    if "\n" in cmd:
        cmd = cmd.split("\n")[0].strip()
    return cmd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to fine-tuned adapter dir (train.py out_dir)")
    ap.add_argument("--base_model", default=None, help="Optional base model path if not saved with adapter")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    args = ap.parse_args()

    base = args.base_model or DEFAULT_BASE_MODEL_ID

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", quantization_config=bnb_config)
    model = PeftModel.from_pretrained(base_model, args.adapter)

    prompt = build_prompt(args.prompt)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)

    gen = full.split("<|assistant|>")[-1].strip()
    cmd = postprocess_to_single_command(gen)

    if not is_safe(cmd):
        print(f"Blocked potentially destructive command: {cmd}")
        sys.exit(0)

    print(cmd)

if __name__ == "__main__":
    main()
