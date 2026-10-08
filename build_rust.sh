#!/bin/bash
# Build vLLM Rust artifacts and install them into the vllm package.
# Usage: ./build_rust.sh [--debug]
#
# By default builds in release mode. Pass --debug for faster compile times
# during development.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

PYTHON="${VIRTUAL_ENV:-$REPO_ROOT/.venv}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Create the project environment with uv venv --python 3.12 first." >&2
    exit 1
fi

# Read the required toolchain from rust-toolchain.toml.
TOOLCHAIN=$("$PYTHON" -c \
    'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["toolchain"]["channel"])' \
    "$REPO_ROOT/rust-toolchain.toml")

# Ensure rustup and the required toolchain are available.
if ! command -v rustup &>/dev/null; then
    echo "rustup not found, installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    source "$HOME/.cargo/env"
fi

if ! rustup run "$TOOLCHAIN" rustc --version &>/dev/null; then
    echo "Installing Rust toolchain: $TOOLCHAIN"
    rustup toolchain install "$TOOLCHAIN"
fi

if [[ "${1:-}" == "--debug" ]]; then
    PROFILE_ARG="--debug"
else
    PROFILE_ARG="--release"
fi

"$PYTHON" "$REPO_ROOT/tools/build_rust.py" "$PROFILE_ARG"
