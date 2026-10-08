# Detailed guidances to build and publish the Docker image

The build host needs Linux x86_64, Git, Docker Engine with BuildKit/Buildx,
network access, disk space and RAM for compilation. A gfx906 GPU, host ROCm
installation and GPU device mounts are not required for building or importing
the package. The Dockerfile explicitly targets gfx906 and supplies ROCm inside
the pinned base image. Inference still needs a compatible GPU host.

Run this after the desired changes have been committed and pushed to `main`.
The helper builds the selected checkout, not a separately cloned vLLM branch.
The current stack is vLLM 0.30.0, ROCm 7.14, Torch 2.13 and source-built Triton
3.8.0; immutable inputs are in `docker/gfx906-build.env`.

```bash
set -euo pipefail

uname -m                         # Must be x86_64.
docker version                   # Both client and server must respond.
docker buildx version
free -h
df -h
docker system df

mkdir -p "$HOME/src"
cd "$HOME/src"
if [[ ! -d vllm-gfx906-mobydick ]]; then
    git clone https://github.com/ai-infos/vllm-gfx906-mobydick.git
fi
cd vllm-gfx906-mobydick

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    git status --short
    printf 'Preserve pending changes before publishing.\n' >&2
    exit 1
fi
git switch main
git pull --ff-only origin main
git log -1 --oneline

BUILD_COMMIT="$(git rev-parse HEAD)"
BUILD_TAG="v0.30.0.x-rocm7.14-pytorch2.13.0-${BUILD_COMMIT:0:12}"


MAX_JOBS=64 IMAGE_NAME=aiinfos/vllm-gfx906-mobydick \
sudo bash build_and_push_docker.sh "$BUILD_TAG" \
  2>&1 | tee "/tmp/vllm-build-${BUILD_COMMIT:0:12}.log"

IMAGE_COMMIT="$(sudo docker image inspect "$IMAGE_NAME:$BUILD_TAG" \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
test "$IMAGE_COMMIT" = "$BUILD_COMMIT"

sudo docker run --rm "$IMAGE_NAME:$BUILD_TAG" \
    /opt/vllm-venv/bin/python -c \
    'import importlib.metadata as m; import torch, triton; from vllm import _gfx906_fa_C; assert torch.version.hip.startswith("7.14"); assert triton.__version__ == "3.8.0"; print("vLLM:", m.version("vllm")); print("Torch:", torch.__version__, "HIP:", torch.version.hip); print("FA:", _gfx906_fa_C.forward)'

# Use your own Docker Hub username if it has write access to aiinfos.
# Supply an access token when Docker prompts for a password.
docker login --username aiinfos
sudo docker push "$IMAGE_NAME:$BUILD_TAG"
sudo docker buildx imagetools inspect "$IMAGE_NAME:$BUILD_TAG"
```

Record the registry digest printed by the last command. Downstream machines can
pull `aiinfos/vllm-gfx906-mobydick@sha256:<digest>` to select that exact image.
Docker Hub publication does not start a cloud inference service.

If you want the helper to build and push together after login:

```bash
MAX_JOBS=4 IMAGE_NAME=aiinfos/vllm-gfx906-mobydick \
    bash build_and_push_docker.sh "$BUILD_TAG" --push
```

The helper defaults to `MAX_JOBS=16` and the tag
`v0.30.0.x-rocm7.14-pytorch2.13.0`. It does not log in, rejects `latest`, and
requires a clean checkout for `--push`. The commit-suffixed tag above avoids
overwriting an earlier release image. `IMAGE_TAG` is a positional argument;
setting an environment variable named `IMAGE_TAG` does not override the default.

If the compiler is killed for lack of RAM, retry with `MAX_JOBS=2` and the same
tag. Keep Docker's cache; completed layers can be reused. For a stopped Docker
daemon on a systemd host, use `sudo systemctl start docker`. If Docker rejects
access to its socket, use the host's configured Docker access method.

The build asserts native attention and Rust payload presence and checks imports.
Those checks establish packaging and linkage only. GPU correctness, graph
replay, model quality, memory use and serving performance remain unverified
without a gfx906 device. Apt and transitive Python dependencies are not fully
locked; successful builds record dependency freezes under
`/usr/share/vllm-gfx906/` inside the image.
