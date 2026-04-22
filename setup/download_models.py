"""T0.2 — Download Qwen2.5-7B-Instruct GGUF models (q4_0, q8_0, f16).

Usage:
    python setup/download_models.py [--dry_run]
"""

import argparse
import os
import sys
from pathlib import Path

MODELS = {
    # (repo, filename, description)
    "q4_0": (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-Q4_0.gguf",
        "Q4_0 quantization (~4.0 GB)",
    ),
    "q8_0": (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-Q8_0.gguf",
        "Q8_0 quantization (~7.5 GB)",
    ),
    "f16": (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-f16.gguf",
        "F16 original (~14 GB)",
    ),
}


def download_model(repo: str, filename: str, dest_dir: str):
    """Download a single GGUF file from Hugging Face."""
    from huggingface_hub import hf_hub_download

    print(f"  Downloading {filename} from {repo} ...")
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=dest_dir,
    )
    print(f"  Done -> {os.path.join(dest_dir, filename)}")


def main():
    parser = argparse.ArgumentParser(description="Download GGUF model files")
    parser.add_argument(
        "--dry_run", action="store_true", help="Only list planned downloads"
    )
    parser.add_argument(
        "--models-dir", default=None, help="Override models directory"
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    models_dir = args.models_dir or str(project_dir / "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Target directory: {models_dir}\n")

    if args.dry_run:
        print("[DRY RUN] Planned downloads:")
        for key, (repo, filename, desc) in MODELS.items():
            print(f"  [{key}] {desc}")
            print(f"       Repo: {repo}")
            print(f"       File: {filename}")
        return

    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    for key, (repo, filename, desc) in MODELS.items():
        dest_path = os.path.join(models_dir, filename)
        if os.path.exists(dest_path):
            print(f"  [SKIP] {filename} already exists.")
            continue
        download_model(repo, filename, models_dir)

    print("\n[DONE] All downloads complete.")


if __name__ == "__main__":
    main()
