"""T0.4 — Verify llama-cli can run a hello world inference.

Usage:
    python setup/verify_install.py [--dry_run]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_llama_cli(project_dir: Path) -> Path:
    """Locate llama-cli.exe (Windows)."""
    candidates = [
        project_dir / "llama.cpp" / "build" / "bin" / "llama-cli.exe",
        project_dir / "llama.cpp" / "build" / "bin" / "llama-cli",  # Linux fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Verify llama-cli inference")
    parser.add_argument("--dry_run", action="store_true", help="Only check paths")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    cli = get_llama_cli(project_dir)

    if cli is None:
        print("[FAIL] llama-cli not found.")
        print("  Expected at: llama.cpp/build/bin/llama-cli.exe")
        print("  Please run: bash setup/build_llama.sh")
        sys.exit(1)

    print(f"[OK] llama-cli found at: {cli}")

    # Check models
    models_dir = project_dir / "models"
    if not models_dir.exists() or not list(models_dir.glob("*.gguf")):
        print("[WARN] No GGUF models found in ./models/.")
        print("  Please run: python setup/download_models.py")
        sys.exit(1)

    model_file = next(models_dir.glob("*.gguf"))
    print(f"[OK] Model found: {model_file.name}")

    if args.dry_run:
        print("[DRY RUN] Would run:")
        print(f"  {cli} --model {model_file} --prompt \"Hello, world!\" --n_predict 10 --temp 0")
        return

    # Run hello world inference
    print("\nRunning hello world inference...")
    try:
        result = subprocess.run(
            [
                str(cli),
                "--model", str(model_file),
                "--prompt", "Hello, world!",
                "--n_predict", "10",
                "--temp", "0",
                "--no-display-prompt",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        # llama-cli may print to stderr; check for any output
        if output.strip():
            print(f"[OK] Inference succeeded. Output excerpt:\n{output.strip()[:300]}")
        else:
            print(f"[WARN] llama-cli returned no output (exit code: {result.returncode})")
    except subprocess.TimeoutExpired:
        print("[FAIL] Inference timed out after 60s.")
        sys.exit(1)

    print("\n[DONE] Phase T0.4 finished — environment verified.")


if __name__ == "__main__":
    main()
