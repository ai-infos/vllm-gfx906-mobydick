#!/bin/bash
set -e

# Settings
# NB: run this script from this branch with:
# chmod +x ./build_and_push_docker.sh && docker login -u aiinfos && ./build_and_push_docker.sh
# it will default to current branch name: v0.23.1rc0.x or from another branch: e.g. "./build_and_push_docker.sh v0.17.1rc0.x"
# NB2: you can force ROCM/PyTorch versions with:
# MAX_JOBS=16 sudo ./build_and_push_docker.sh v0.23.1rc0.x 7.2.1 2.11.0
IMAGE_NAME="aiinfos/vllm-gfx906-mobydick"
IMAGE_TAG="${1:-v0.23.1rc0.x}"
ROCM_VERSION="${2:-6.3.3}"
PYTORCH_VERSION="${3:-2.11.0}"
TRANSFORMERS_VERSION="5.7.0"
MAX_JOBS="${MAX_JOBS:-16}"

# Determine amdsmi package version based on ROCm version
if [[ $ROCM_VERSION == 7* ]]; then
    AMDSMI_PKG="amdsmi==7.0.2"
else
    # Default for ROCm 6.x
    AMDSMI_PKG="amdsmi>=6.3,<6.4"
fi

BASE_IMAGE="docker.io/mixa3607/pytorch-gfx906:v${PYTORCH_VERSION}-rocm-${ROCM_VERSION}"

echo "Using base image: ${BASE_IMAGE}"
echo "Limiting native builds to ${MAX_JOBS} parallel jobs per build stage."
echo "Creating Dockerfile..."

cat << EOF > Dockerfile
# syntax=docker/dockerfile:1.4
FROM ${BASE_IMAGE} AS rocm_base

# Global environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV USE_ROCM=1
ENV PYTORCH_ROCM_ARCH=gfx906

# Fix 'Cannot uninstall PyJWT, RECORD file not found' caused by Debian system packages
RUN pip install --upgrade --ignore-installed pyjwt

# Install amdsmi early (often needed for vLLM)
RUN pip install "${AMDSMI_PKG}"

# ==========================================
# Build stage: Install build dependencies
# ==========================================
FROM rocm_base AS build_base

# Limit concurrent compiler processes to avoid exhausting host memory.
ENV MAX_JOBS=${MAX_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# Install build tools + Rust/protobuf dependencies required by vLLM Rust extensions.
# protobuf-compiler provides protoc, required by prost-build in vllm-server.
RUN apt-get update && apt-get install -y \\
    git \\
    wget \\
    curl \\
    cmake \\
    ninja-build \\
    build-essential \\
    pkg-config \\
    protobuf-compiler \\
    libprotobuf-dev \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install \\
    ninja \\
    'cmake>=3.26.1,<4' \\
    pybind11 \\
    packaging \\
    'setuptools>=77.0.3,<80.0.0' \\
    'setuptools-scm>=8' \\
    'setuptools-rust>=1.9.0' \\
    wheel

# ==========================================
# Build Triton-gfx906
# ==========================================
FROM build_base AS build_triton
WORKDIR /app
RUN git clone --branch v3.6.0+gfx906 https://github.com/ai-infos/triton-gfx906.git triton-gfx906 && \\
    cd triton-gfx906 && \\
    pip install -r python/requirements.txt && \\
    TRITON_CODEGEN_BACKENDS="amd" pip wheel --no-build-isolation -w /dist .

# ==========================================
# Build Flash-Attention-gfx906
# ==========================================
FROM build_base AS build_flash_attn
WORKDIR /app
RUN git clone https://github.com/ai-infos/flash-attention-gfx906.git flash-attention-gfx906 && \\
    cd flash-attention-gfx906 && \\
    FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip wheel --no-build-isolation -w /dist .

# ==========================================
# Build vLLM-gfx906-mobydick
# ==========================================
FROM build_base AS build_vllm
WORKDIR /app
RUN git clone https://github.com/ai-infos/vllm-gfx906-mobydick.git vllm-gfx906-mobydick && \\
    cd vllm-gfx906-mobydick && \\
    ./build_rust.sh && \\
    pip install -r requirements/rocm.txt && \\
    pip wheel --no-build-isolation -v -w /dist .

# ==========================================
# Final minimal image
# ==========================================
FROM rocm_base AS final
WORKDIR /workspace/vllm-gfx906-mobydick

# Set recommended runtime environment variables
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
ENV VLLM_LOGGING_LEVEL=INFO

# Bind mount the wheels directly and install them without bloating the layer.
# Install everything in a single pip command so pip resolves them together and doesn't fetch CUDA versions.
RUN --mount=type=bind,from=build_triton,src=/dist/,target=/dist_triton \\
    --mount=type=bind,from=build_flash_attn,src=/dist/,target=/dist_flash_attn \\
    --mount=type=bind,from=build_vllm,src=/dist/,target=/dist_vllm \\
    pip install transformers=="${TRANSFORMERS_VERSION}" /dist_triton/triton-*.whl /dist_flash_attn/flash_attn-*.whl /dist_vllm/vllm-*.whl

CMD ["/bin/bash"]
EOF

echo "Dockerfile has been created."

echo "Building the Docker image: ${IMAGE_NAME}:${IMAGE_TAG}-rocm${ROCM_VERSION}-pytorch${PYTORCH_VERSION} ..."
# Build the multi-stage image. Buildkit handles parallel stage execution.
# Use the no-cache-filter line below when you want to force-refresh vLLM/Rust build layers
# without rebuilding Triton and Flash-Attention.
# DOCKER_BUILDKIT=1 docker build --no-cache-filter build_base --no-cache-filter build_vllm -t ${IMAGE_NAME}:${IMAGE_TAG}-rocm${ROCM_VERSION}-pytorch${PYTORCH_VERSION} -t ${IMAGE_NAME}:latest .
DOCKER_BUILDKIT=1 docker build -t ${IMAGE_NAME}:${IMAGE_TAG}-rocm${ROCM_VERSION}-pytorch${PYTORCH_VERSION} -t ${IMAGE_NAME}:latest .

echo "We need to authenticate with Docker Hub before pushing."
echo "If you haven't logged in yet, it will prompt for your Docker Hub credentials."
docker login -u aiinfos

echo "Pushing the Docker image to Docker Hub at ${IMAGE_NAME}:${IMAGE_TAG}-rocm${ROCM_VERSION}-pytorch${PYTORCH_VERSION} and ${IMAGE_NAME}:latest ..."
docker push ${IMAGE_NAME}:${IMAGE_TAG}-rocm${ROCM_VERSION}-pytorch${PYTORCH_VERSION}
docker push ${IMAGE_NAME}:latest

echo "Process completed successfully! The images are pushed to Docker Hub."
