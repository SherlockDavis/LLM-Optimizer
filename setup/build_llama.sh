#!/bin/bash
# T0.1 — Build llama.cpp with CUDA support
# Usage: bash setup/build_llama.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"

# Step 1: Clone llama.cpp if not present
if [ -d "$LLAMA_DIR" ]; then
    echo "[INFO] llama.cpp already exists, skipping clone."
else
    echo "[INFO] Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

# Step 2: Build with CUDA
echo "[INFO] Building llama.cpp with CUDA backend..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DLLAMA_CUDA=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build . --config Release -j "$(nproc)"

echo "[INFO] Build complete. llama-cli.exe should be at: $BUILD_DIR/bin/llama-cli.exe"
echo "[DONE] Phase T0.1 finished."
