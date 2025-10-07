import argparse, subprocess, sys, shutil, os
from constants import DEFAULT_ADAPTER_DIR, DEFAULT_UNSAFE_FLAG

def main():
    ap = argparse.ArgumentParser(description="Natural-language -> Linux command (model-backed)")
    ap.add_argument("request", type=str, help="Your request, e.g., 'list open ports for nginx'")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER_DIR)
    ap.add_argument("--base_model", default=None) # The default base model is auto-detected from adapter config
    ap.add_argument("--run", action="store_true", help="If set, executes the produced command locally")
    args = ap.parse_args()

    py = shutil.which("python") or "python"
    cmd = [
        py, "infer.py",
        "--adapter", args.adapter,
        "--prompt", args.request,
        "--unsafe", args.unsafe
    ]
    if args.base_model:
        cmd += ["--base_model", args.base_model]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout.strip():
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)

    out_cmd = result.stdout.strip()
    print(out_cmd)

    if args.run:
        os.system(out_cmd)

if __name__ == "__main__":
    main()
