#!/usr/bin/env python3
"""
Merge LoRA adapter with base model and prepare for GGUF conversion.

Usage:
    python merge_and_prepare_gguf.py --adapter outputs/tinyllama_cmdgen --output outputs/merged_model
    
Then convert to GGUF using llama.cpp tools:
    python /path/to/llama.cpp/convert_hf_to_gguf.py outputs/merged_model --outfile outputs/tinyllama_cmdgen.gguf --outtype q4_k_m
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from constants import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_MERGED_OUTPUT_DIR,
    DEFAULT_DEVICE,
)


def merge_adapter_to_base(adapter_path: str, base_model_path: str, output_path: str, device: str = "auto"):
    """
    Merge a LoRA adapter into the base model and save as a standard HuggingFace model.
    
    Args:
        adapter_path: Path to the trained adapter directory
        base_model_path: Path to base model (can be None to auto-detect from adapter config)
        output_path: Where to save the merged model
        device: Device to load model on
    """
    print(f"Loading base model from {base_model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    
    # Load base model WITHOUT quantization (we need full precision to merge)
    # This will take more RAM but is necessary for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # Use fp16 to save memory
        device_map=device,
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading adapter from {adapter_path}...")
    # Load the PEFT model with adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter weights into base model...")
    # Merge adapter weights into the base model
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    # Save the merged model
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ Merged model saved to {output_path}")
    print(f"\nNext steps:")
    print(f"1. Clone llama.cpp if you haven't:")
    print(f"   git clone https://github.com/ggerganov/llama.cpp.git")
    print(f"   cd llama.cpp && make")
    print(f"\n2. Convert to GGUF format:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {output_path} --outfile {output_path}.gguf --outtype f16")
    print(f"\n3. Quantize to Q4_K_M (recommended for quality/size balance):")
    print(f"   ./llama.cpp/llama-quantize {output_path}.gguf {output_path}-Q4_K_M.gguf Q4_K_M")
    print(f"\n4. Or quantize to other formats (Q2_K, Q4_0, Q5_K_M, Q8_0, etc.):")
    print(f"   ./llama.cpp/llama-quantize {output_path}.gguf {output_path}-Q2_K.gguf Q2_K")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model for GGUF conversion"
    )
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER_DIR,
        help="Path to adapter directory (e.g., outputs/tinyllama_cmdgen)"
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Base model path or HF model ID (default: auto-detect from adapter config)"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_MERGED_OUTPUT_DIR,
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Device to use (auto, cpu, cuda, cuda:0, etc.)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect base model if not provided
    base_model = args.base_model
    if base_model is None:
        # Try to read from adapter config
        import json
        config_path = os.path.join(args.adapter, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path")
        
        if base_model is None:
            base_model = DEFAULT_BASE_MODEL_ID
            print(f"⚠ Could not auto-detect base model, using default: {base_model}")
        else:
            print(f"Auto-detected base model: {base_model}")
    
    merge_adapter_to_base(
        adapter_path=args.adapter,
        base_model_path=base_model,
        output_path=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()