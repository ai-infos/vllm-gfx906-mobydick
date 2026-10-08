#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=docker/gfx906-build.env
source "$REPO_ROOT/docker/gfx906-build.env"

IMAGE_NAME="${IMAGE_NAME:-aiinfos/vllm-gfx906-mobydick}"
IMAGE_TAG="v0.30.0.x-rocm7.14-pytorch2.13.0"
MAX_JOBS="${MAX_JOBS:-16}"
PUSH=0
TAG_SET=0

while (($#)); do
    case "$1" in
        --push) PUSH=1 ;;
        --help|-h)
            echo "Usage: $0 [image-tag] [--push]"
            echo "Builds this checkout. MAX_JOBS and IMAGE_NAME may be overridden."
            exit 0
            ;;
        --*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            if ((TAG_SET)); then
                echo "Only an image tag is accepted; dependency pins are in docker/gfx906-build.env." >&2
                exit 2
            fi
            IMAGE_TAG="$1"
            TAG_SET=1
            ;;
    esac
    shift
done

if [[ ! "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer." >&2
    exit 2
fi
if [[ ! "$IMAGE_TAG" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ ]]; then
    echo "Invalid Docker image tag: $IMAGE_TAG" >&2
    exit 2
fi
if [[ "$IMAGE_TAG" == latest ]]; then
    echo "Use a versioned image tag; latest is not published by this helper." >&2
    exit 2
fi

VLLM_SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
VLLM_VERSION="0.30.0+gfx906.${VLLM_SOURCE_COMMIT:0:12}"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]]; then
    VLLM_VERSION+=".dirty"
    if ((PUSH)); then
        echo "Commit or remove pending changes before publishing a reproducible image." >&2
        exit 2
    fi
fi

DOCKER_BUILDKIT=1 docker build \
    --file "$REPO_ROOT/docker/Dockerfile.gfx906" \
    --build-arg "BASE_IMAGE=$GFX906_BASE_IMAGE" \
    --build-arg "UV_VERSION=$GFX906_UV_VERSION" \
    --build-arg "UV_SHA256=$GFX906_UV_SHA256" \
    --build-arg "TRITON_REV=$GFX906_TRITON_REV" \
    --build-arg "FLASH_ATTN_REV=$GFX906_FLASH_ATTN_REV" \
    --build-arg "MAX_JOBS=$MAX_JOBS" \
    --build-arg "VLLM_VERSION=$VLLM_VERSION" \
    --build-arg "VLLM_SOURCE_COMMIT=$VLLM_SOURCE_COMMIT" \
    --tag "$IMAGE_NAME:$IMAGE_TAG" "$REPO_ROOT"

if ((PUSH)); then
    docker push "$IMAGE_NAME:$IMAGE_TAG"
fi
