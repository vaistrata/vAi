#!/usr/bin/env bash
#==============================================================================
# Installs the TensorRT Edge-LLM the pi0.5 benches compare against: pinned tag,
# its C++ libraries, the experimental pi0.5 runtime and the policy tools.
#
# Installs to ${XDG_CACHE_HOME:-~/.cache}/vai/TensorRT-Edge-LLM-<tag> and links
# it at ${XDG_CACHE_HOME:-~/.cache}/vai/TensorRT-Edge-LLM, where the bench's
# CMake looks for it: Edge-LLM has no CMake package. Builds no model files;
# benchmarks/pi05/scripts/export-trt.sh builds the engines, with the exporter
# it installs from the source this script keeps.
#
# Needs Linux, git, cmake, a CUDA-compatible C++17 compiler, Python 3 with venv
# (Ubuntu: python3-venv), CUDA 12 or 13 (nvcc; CUDA_HOME if it is not on PATH),
# a TensorRT C++ SDK built for that CUDA, with its plugins and ONNX parser
# (TRT_PACKAGE_DIR for an extracted SDK, default /usr), and network access for
# the sources and the CuTe DSL packages. Builds the CuTe DSL kernels for the SM
# of GPU GPU_INDEX (default 0), or CUDA_ARCH, and records it in .cuda-arch for
# the bench's device link; upstream builds its other kernels for a fixed list.
#
# What it downloads stays in ${XDG_CACHE_HOME:-~/.cache}/vai/TensorRT-Edge-LLM-<tag>-src
# for later runs: the clone with its submodules and the Python environment that
# builds the CuTe DSL kernels. A later run reuses them without the network; only
# the build tree is removed after the install.
#
#   ./benchmarks/scripts/setup-trt-edge-llm.sh
#   ./benchmarks/scripts/setup-trt-edge-llm.sh --uninstall     # remove the install, downloads + link
#==============================================================================
set -euo pipefail

# The pinned release; benchmarks/pi05/CMakeLists.txt warns on any other.
TAG=v0.11.1

[ "$(uname -s)" = Linux ] || { echo "this Edge-LLM build is set up for Linux only" >&2; exit 1; }
[ -n "${HOME:-}" ] || { echo "HOME is not set: the install lives under \$HOME/.cache; set it and re-run" >&2; exit 2; }
if [ "$(id -u)" -eq 0 ]; then
    echo "warning: running as root, so this installs into root's cache ($HOME), not yours" >&2
fi

# Deletes nothing outside these: the prefix, its source and staging trees, the link
# to it and, on --uninstall, the earlier install of this script (TensorRT-Edge-LLM-latest).
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
PREFIX="$CACHE_DIR/vai/TensorRT-Edge-LLM-$TAG"
SRC="$PREFIX-src"
LINK="$CACHE_DIR/vai/TensorRT-Edge-LLM"
STAGE="$PREFIX-stage"
JOBS="$(nproc 2>/dev/null || echo 4)"
KEEP_BUILD=0
FORCE=0
UNINSTALL=0

usage() {
    sed -n '2,/^#===/p' "$0" | sed 's/^#//'
    cat <<USAGE

Options:
  --jobs N, -j N parallel build jobs (default: $JOBS)
  --keep-build   keep the build tree too, for a faster rebuild
  --force        rebuild even if this tag is already installed (the downloads are reused)
  --uninstall    remove the install, its downloads, the link and the earlier install
  -h, --help     this text
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --jobs|-j)   JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        --keep-build) KEEP_BUILD=1; shift ;;
        --force)     FORCE=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs needs a positive integer, got '$JOBS'" >&2; exit 2; }

if [ $UNINSTALL -eq 1 ]; then
    if [ "$(readlink "$LINK" 2>/dev/null || true)" = "$PREFIX" ]; then
        rm -f "$LINK"
    fi
    rm -rf "$PREFIX" "$SRC" "$STAGE"
    echo "removed $PREFIX and $SRC"
    # The earlier install of this script, which followed upstream's main branch,
    # told apart from other folders by the commit file only this script writes.
    OLD="$CACHE_DIR/vai/TensorRT-Edge-LLM-latest"
    if [ -f "$OLD/.installed" ]; then
        if [ "$(readlink "$LINK" 2>/dev/null || true)" = "$OLD" ]; then
            rm -f "$LINK"
        fi
        rm -rf "$OLD" "$OLD-src" "$OLD-stage"
        echo "removed $OLD, an earlier install, and its source tree"
    fi
    exit 0
fi

# Preserve a checkout or other existing directory occupying the public link.
link_install() {
    if [ -e "$LINK" ] && [ ! -L "$LINK" ]; then
        local backup
        backup="$(mktemp -d "${LINK}.backup.XXXXXX")"
        mv -- "$LINK" "$backup/original"
        echo "existing $LINK preserved at $backup/original"
    fi
    ln -sfnT "$PREFIX" "$LINK"
}

# Already installed? The prefix is named after the tag.
if [ $FORCE -eq 0 ] && [ -f "$PREFIX/.installed" ]; then
    link_install
    echo "TensorRT Edge-LLM $TAG is already installed at $PREFIX ($(cat "$PREFIX/.installed"))"
    echo "(re-run with --force to rebuild it)"
    exit 0
fi

#------------------------------------------------------------------------------
# Prerequisites, checked before anything is cloned or built
#------------------------------------------------------------------------------
for tool in git cmake python3; do
    command -v "$tool" >/dev/null 2>&1 || { echo "$tool not found" >&2; exit 1; }
done
python3 -c 'import ensurepip, venv' 2>/dev/null || {
    echo "python3 cannot create a virtual environment: install its venv module (Ubuntu: python3-venv)" >&2; exit 1;
}
NVCC="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
[ -x "$NVCC" ] || NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || { echo "nvcc not found: install CUDA or set CUDA_HOME" >&2; exit 1; }
CUDA_DIR="$(dirname "$(dirname "$(readlink -f "$NVCC")")")"
export PATH="$CUDA_DIR/bin:$PATH"
CUDA_VERSION="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
CUDA_MAJOR="${CUDA_VERSION%%.*}"
[[ "$CUDA_MAJOR" == 12 || "$CUDA_MAJOR" == 13 ]] || { echo "CUDA 12 or 13 required" >&2; exit 1; }
TRT_PACKAGE_DIR="$(realpath -e "${TRT_PACKAGE_DIR:-/usr}")"
[ -n "$(find "$TRT_PACKAGE_DIR/include" -name NvInfer.h -print -quit)" ] || {
    echo "TensorRT headers missing: set TRT_PACKAGE_DIR to the TensorRT C++ SDK" >&2; exit 1;
}
if [ -z "${CUDA_ARCH:-}" ]; then
    command -v nvidia-smi >/dev/null 2>&1 || { echo "set CUDA_ARCH (e.g. 120) or provide nvidia-smi" >&2; exit 1; }
    CUDA_ARCH="$(nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:].')"
fi
[[ "$CUDA_ARCH" =~ ^[0-9]+$ ]] || { echo "CUDA_ARCH must be a numeric SM (e.g. 120)" >&2; exit 2; }
GENERATOR=()
command -v ninja >/dev/null 2>&1 && GENERATOR=(-G Ninja)

run_logged() {
    local log=$1
    shift
    "$@" > "$log" 2>&1 || {
        echo "command failed, tail of $log:" >&2
        tail -30 "$log" >&2
        exit 1
    }
}

echo "=============================================================="
echo " TensorRT Edge-LLM $TAG -> $PREFIX"
echo " CUDA $CUDA_VERSION | SM $CUDA_ARCH | $JOBS jobs"
echo " TensorRT SDK: $TRT_PACKAGE_DIR"
echo "=============================================================="

#------------------------------------------------------------------------------
# Source
#------------------------------------------------------------------------------
mkdir -p "$CACHE_DIR/vai"
if [ -d "$SRC/.git" ] && git -C "$SRC" rev-parse -q --verify "refs/tags/$TAG^{commit}" >/dev/null; then
    echo "[1/4] reusing the $TAG clone in $SRC"
else
    echo "[1/4] fetching $TAG..."
fi
if [ -d "$SRC/.git" ]; then
    [ "$(git -C "$SRC" remote get-url origin)" = "https://github.com/NVIDIA/TensorRT-Edge-LLM.git" ] || {
        echo "unexpected origin in $SRC" >&2; exit 1;
    }
    git -C "$SRC" diff --quiet && git -C "$SRC" diff --cached --quiet || {
        echo "tracked source changes in $SRC; commit or stash them before updating" >&2; exit 1;
    }
    # A kept clone already holding the tag needs no network.
    git -C "$SRC" rev-parse -q --verify "refs/tags/$TAG^{commit}" >/dev/null \
        || git -C "$SRC" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
    PREVIOUS_COMMIT="$(git -C "$SRC" rev-parse HEAD)"
    git -C "$SRC" checkout --quiet --detach "$TAG"
    # Old archives must not leak into a build of another commit.
    if [ "$PREVIOUS_COMMIT" != "$(git -C "$SRC" rev-parse HEAD)" ]; then
        rm -rf "$SRC/build" "$SRC/.venv"
    fi
else
    [ ! -e "$SRC" ] || { echo "not a Git checkout: $SRC" >&2; exit 1; }
    # Ctrl-C mid-clone drops the half-made clone; once complete, it is kept.
    trap 'rm -rf "$SRC"; exit 130' INT
    trap 'rm -rf "$SRC"; exit 143' TERM
    git clone --depth 1 --branch "$TAG" https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$SRC"
    trap - INT TERM
fi
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
echo "      $TAG commit: $COMMIT"

git -C "$SRC" submodule update --init --recursive --depth 1

#------------------------------------------------------------------------------
# CuTe DSL artifacts (isolated build-only Python environment)
#------------------------------------------------------------------------------
echo "[2/4] preparing CuTe DSL kernels..."
python3 -m venv "$SRC/.venv"
PYTHON="$SRC/.venv/bin/python"
if [ "$CUDA_MAJOR" = 13 ]; then CUPY=cupy-cuda13x==13.6.0; else CUPY=cupy-cuda12x==12.3.0; fi
run_logged "$SRC/python.log" "$PYTHON" -m pip install 'nvidia-cutlass-dsl[cu'"$CUDA_MAJOR"']==4.8.0' "$CUPY"
(
    cd "$SRC"
    run_logged "$SRC/kernels.log" "$PYTHON" kernelSrcs/build_cutedsl.py \
        --kernels fmha --gpu_arch "sm_$CUDA_ARCH" --arch "$(uname -m)" \
        --cuda-version "$CUDA_MAJOR" -j "$JOBS"
)

#------------------------------------------------------------------------------
# Build + install (upstream does not provide a complete SDK install target)
#------------------------------------------------------------------------------
echo "[3/4] building..."
run_logged "$SRC/configure.log" cmake -S "$SRC" -B "$SRC/build" "${GENERATOR[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCUDA_DIR="$CUDA_DIR" -DCUDA_CTK_VERSION="$CUDA_VERSION" \
    -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR" \
    -DENABLE_CUTE_DSL=fmha -DCUTE_DSL_ARTIFACT_TAG="sm_$CUDA_ARCH" \
    -DBUILD_EXPERIMENTAL_MODELS=ON -DBUILD_UNIT_TESTS=OFF
run_logged "$SRC/build.log" cmake --build "$SRC/build" -j "$JOBS" --target \
    NvInfer_edgellm_plugin pi05_policy_build pi05_policy_inference

echo "[4/4] installing..."
rm -rf "$STAGE"
mkdir -p "$STAGE/include/tensorrt_edgellm" "$STAGE/lib" "$STAGE/bin"
# Keep core and pi0.5 headers in the locations searched by the bench CMake.
cp -R "$SRC/cpp/." "$STAGE/include/tensorrt_edgellm/"
cp -R "$SRC/experimental_models/pi05/cpp/." "$STAGE/include/"
cp -R "$SRC/3rdParty/nlohmannJson/include/nlohmann" "$STAGE/include/"
# cp -P keeps the plugin's version symlinks as links, not three copies of it.
find "$SRC/build" \( -type f -o -type l \) \
    \( -name '*.a' -o -name '*.so' -o -name '*.so.*' \) \
    -exec cp -P -t "$STAGE/lib" {} +
for tool in pi05_policy_build pi05_policy_inference; do
    found="$(find "$SRC/build" -type f -name "$tool" -executable -print -quit)"
    [ -n "$found" ] || { echo "build finished but $tool is missing" >&2; exit 1; }
    cp "$found" "$STAGE/bin/"
done
for lib in libpi05Experimental.a libedgellmCore.a libedgellmChatTemplate.a libxgrammarCore.a \
           libNvInfer_edgellm_plugin.so; do
    [ -f "$STAGE/lib/$lib" ] || { echo "build finished but $lib is missing" >&2; exit 1; }
done
cp "$SRC/LICENSE" "$STAGE/"
printf '%s\n' "$CUDA_ARCH" > "$STAGE/.cuda-arch"
printf '%s\n' "$COMMIT" > "$STAGE/.installed"
# Replace the old prefix only after the complete SDK has been staged.
trap '' INT TERM
rm -rf "$PREFIX"
mv "$STAGE" "$PREFIX"
link_install
trap - INT TERM
# The downloads stay; the build tree goes unless --keep-build.
if [ $KEEP_BUILD -eq 0 ]; then
    rm -rf "$SRC/build"
fi

#------------------------------------------------------------------------------
# Done
#------------------------------------------------------------------------------
echo
echo "installed TensorRT Edge-LLM $TAG ($COMMIT, $(du -sh "$PREFIX" | cut -f1)) at:"
echo "  $PREFIX"
echo "linked from $LINK"
echo "downloads kept in $SRC ($(du -sh "$SRC" | cut -f1)); --uninstall removes them"
echo "bench CMake: -DTensorRTEdgeLLM_ROOT=$LINK -DTENSORRT_ROOT=$TRT_PACKAGE_DIR"
