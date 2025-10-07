import socket
# download_model.py
# Monkey-patch to force IPv4
original_getaddrinfo = socket.getaddrinfo

def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = ipv4_only_getaddrinfo

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from constants import DEFAULT_REPO_ID, DEFAULT_LOCAL_DOWNLOAD_DIR

def main():
    print("Starting download...")
    try:
        snapshot_download(
            repo_id=DEFAULT_REPO_ID,
            local_dir=DEFAULT_LOCAL_DOWNLOAD_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            tqdm_class=tqdm
        )
        print("\n✓ Download complete!")
        print(f"Location: {DEFAULT_LOCAL_DOWNLOAD_DIR}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
