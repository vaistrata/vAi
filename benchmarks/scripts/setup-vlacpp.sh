#!/usr/bin/env bash
#==============================================================================
# Installs the vla.cpp the pi0.5 benches compare against: pinned tag, CUDA
# backend, shared libraries. vla.cpp has no Vulkan backend for pi0.5.
#
# Installs to ${XDG_CACHE_HOME:-~/.cache}/vai/vla.cpp-<tag> and links it at
# ${XDG_CACHE_HOME:-~/.cache}/vai/vla.cpp, where the bench's CMake looks for it:
# vla.cpp has no CMake package. The pi0.5 GGUF converter from the same tag is
# kept at <prefix>/convert, whose pyproject.toml records the release the
# bench's CMake checks.
#
# What it downloads stays in ${XDG_CACHE_HOME:-~/.cache}/vai/vla.cpp-<tag>-src
# for later runs: the vla.cpp clone and, in its _deps/, the llama.cpp vla.cpp
# fetches (and patches) at configure time. A later run reuses them without the
# network; only the build trees are removed after the install.
#
# Needs Linux, git, cmake 3.22+, a C++17 compiler, CUDA 12 or 13 (nvcc;
# CUDA_HOME if it is not on PATH) and, on the first run, network access.
#
#   ./benchmarks/scripts/setup-vlacpp.sh                 # ~20 min
#   ./benchmarks/scripts/setup-vlacpp.sh --uninstall     # remove the install, downloads + link
#==============================================================================
set -euo pipefail

# The pinned release; benchmarks/pi05/CMakeLists.txt warns on any other.
TAG=v0.4.0

[ "$(uname -s)" = Linux ] || { echo "vla.cpp's CUDA build is set up for Linux only" >&2; exit 1; }
[ -n "${HOME:-}" ] || { echo "HOME is not set: the install lives under \$HOME/.cache; set it and re-run" >&2; exit 2; }
if [ "$(id -u)" -eq 0 ]; then
    echo "warning: running as root, so this installs into root's cache ($HOME), not yours" >&2
fi

# Deletes nothing outside these: the prefix, its source tree and the link to it.
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
PREFIX="$CACHE_DIR/vai/vla.cpp-$TAG"
SRC="$PREFIX-src"
LINK="$CACHE_DIR/vai/vla.cpp"
CONVERT="$PREFIX/convert"
# Where FetchContent keeps llama.cpp: next to the source, so it outlives the build tree.
DEPS="$SRC/_deps"

JOBS="$(nproc 2>/dev/null || echo 4)"
KEEP_BUILD=0
FORCE=0
UNINSTALL=0

usage() {
    sed -n '2,/^#===/p' "$0" | sed 's/^#//'
    cat <<USAGE

Options:
  --jobs N, -j N parallel build jobs (default: $JOBS)
  --keep-build   keep the build trees too, for a faster rebuild
  --force        reinstall even if this tag is already installed (the downloads are reused)
  --uninstall    remove the install, its downloads and the link to it
  -h, --help     this text
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --jobs|-j)   JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        --keep-build) KEEP_BUILD=1; shift ;;
        --force)     FORCE=1;      shift ;;
        --uninstall) UNINSTALL=1;  shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs needs a positive integer, got '$JOBS'" >&2; exit 2; }

if [ $UNINSTALL -eq 1 ]; then
    [ "$(readlink "$LINK" 2>/dev/null || true)" = "$PREFIX" ] && rm -f "$LINK"
    rm -rf "$PREFIX" "$SRC"
    echo "removed $PREFIX and $SRC"
    exit 0
fi

# The pi0.5 converter and the modules it imports; its pyproject.toml carries the
# release the bench's CMake reads.
keep_converter() {
    mkdir -p "$CONVERT"
    cp "$SRC"/scripts/convert_pi05_to_gguf.py \
       "$SRC"/scripts/gguf_common.py "$SRC"/scripts/gguf_blocks.py \
       "$SRC"/scripts/quantize_gguf.py "$SRC"/LICENSE.md "$SRC"/pyproject.toml "$CONVERT/"
}

#------------------------------------------------------------------------------
# Already installed?
#------------------------------------------------------------------------------
if [ $FORCE -eq 0 ] && [ -f "$PREFIX/lib/libvla.so" ] && [ -f "$PREFIX/include/vla.h" ] \
   && [ -f "$CONVERT/pyproject.toml" ]; then
    ln -sfn "$PREFIX" "$LINK"
    echo "vla.cpp $TAG is already installed at $PREFIX"
    echo "(re-run with --force to rebuild it)"
    exit 0
fi

#------------------------------------------------------------------------------
# Prerequisites, checked before anything is cloned or built
#------------------------------------------------------------------------------
for tool in git cmake; do
    command -v "$tool" >/dev/null 2>&1 || { echo "$tool not found" >&2; exit 1; }
done
NVCC="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
[ -x "$NVCC" ] || NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || { echo "nvcc not found: install CUDA 12 or 13, or set CUDA_HOME" >&2; exit 1; }

GENERATOR=()
command -v ninja >/dev/null 2>&1 && GENERATOR=(-G Ninja)

echo "=============================================================="
echo " vla.cpp $TAG -> $PREFIX"
echo " CUDA ($NVCC) | $JOBS jobs"
echo "=============================================================="

#------------------------------------------------------------------------------
# Source
#
# The kept clone is reused only if it is at $TAG.
#------------------------------------------------------------------------------
mkdir -p "$CACHE_DIR/vai"

srcTag="$(git -C "$SRC" rev-parse -q --verify "refs/tags/$TAG^{commit}" 2>/dev/null || true)"
srcHead="$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)"

if [ -n "$srcTag" ] && [ "$srcTag" = "$srcHead" ]; then
    echo "[1/3] reusing source at $SRC"
else
    echo "[1/3] cloning $TAG..."
    rm -rf "$SRC"
    # Ctrl-C mid-clone drops the half-made clone; once complete, it is kept.
    trap 'rm -rf "$SRC"; exit 130' INT
    trap 'rm -rf "$SRC"; exit 143' TERM
    git clone --quiet --depth 1 --branch "$TAG" https://github.com/VinRobotics/vla.cpp "$SRC" 2>/dev/null
    trap - INT TERM
fi

#------------------------------------------------------------------------------
# Build + install
#
# VLA_BUILD_SERVER=OFF: the servers need protobuf and libzmq; the benches do not.
# VLA_SPM=OFF: SentencePiece needs protobuf; the benches pass token ids.
# FETCHCONTENT_BASE_DIR keeps the llama.cpp clone, and the stamps of its patch, in
# $DEPS; FETCHCONTENT_UPDATES_DISCONNECTED reuses it without the network.
#------------------------------------------------------------------------------
echo "[2/3] building (this is the slow part, about 20 minutes)..."
cmake -S "$SRC" -B "$SRC/build" "${GENERATOR[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DVLA_BUILD_SERVER=OFF \
    -DVLA_BUILD_TESTS=OFF \
    -DVLA_SPM=OFF \
    -DFETCHCONTENT_BASE_DIR="$DEPS" \
    -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" > "$SRC/configure.log" 2>&1 || {
        echo "configure failed, tail of $SRC/configure.log:" >&2
        tail -20 "$SRC/configure.log" >&2
        exit 1
    }

cmake --build "$SRC/build" -j "$JOBS" > "$SRC/build.log" 2>&1 || {
    echo "build failed, tail of $SRC/build.log:" >&2
    tail -30 "$SRC/build.log" >&2
    exit 1
}

# The old install is removed only after a successful build; signals are held
# off meanwhile so a Ctrl-C cannot leave a half-installed prefix.
echo "[3/3] installing..."
trap '' INT TERM
rm -rf "$PREFIX"
cmake --install "$SRC/build" > "$SRC/install.log" 2>&1 || {
    echo "install failed, tail of $SRC/install.log:" >&2
    tail -20 "$SRC/install.log" >&2
    exit 1
}

# llama.cpp's own install rules may leave out libraries libvla.so loads at run
# time, so the prefix is completed from the build tree.
for lib in libvla.so libvla_core.so libllama.so libggml.so libggml-base.so libggml-cpu.so \
           libggml-cuda.so libmtmd.so libllama-common.so; do
    [ -e "$PREFIX/lib/$lib" ] && continue
    found="$(find "$SRC/build" "$DEPS" -name "$lib" -type f 2>/dev/null | head -1 || true)"
    [ -n "$found" ] && cp -L "$found" "$PREFIX/lib/"
done
[ -f "$PREFIX/include/vla.h" ] || cp "$SRC/include/vla.h" "$PREFIX/include/"
keep_converter
trap - INT TERM

for lib in libvla.so libvla_core.so libllama.so; do
    [ -e "$PREFIX/lib/$lib" ] || { echo "install finished but $PREFIX/lib/$lib is missing" >&2; exit 1; }
done
ln -sfn "$PREFIX" "$LINK"

# The downloads stay; the build trees go unless --keep-build.
if [ $KEEP_BUILD -eq 0 ]; then
    rm -rf "$SRC/build" "$DEPS"/*-build
fi

#------------------------------------------------------------------------------
# Done
#------------------------------------------------------------------------------
echo
echo "installed vla.cpp $TAG ($(du -sh "$PREFIX" | cut -f1)) at:"
echo "  $PREFIX"
echo "linked from $LINK"
echo "GGUF converter at $CONVERT"
echo "downloads kept in $SRC ($(du -sh "$SRC" | cut -f1)); --uninstall removes them"
