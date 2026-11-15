#!/usr/bin/env bash
#==============================================================================
# Installs the llama.cpp the benches compare against: pinned tag, Vulkan
# backend, static by default (--shared for shared libraries).
#
# Installs to ${XDG_CACHE_HOME:-~/.cache}/vai/llama.cpp-<tag> and registers it
# in CMake's user package registry, so find_package(llama) finds it without
# flags. The GGUF converter from the same tag is kept at <prefix>/convert.
#
# Needs git, cmake, SPIRV-Headers (Ubuntu: apt install spirv-headers) and a
# glslc that supports GL_NV_cooperative_matrix2, such as the LunarG Vulkan
# SDK's: llama.cpp builds its NVIDIA coopmat2 kernels only then.
#
#   ./benchmarks/scripts/setup-llamacpp.sh                 # ~2-3 min
#   ./benchmarks/scripts/setup-llamacpp.sh --uninstall     # remove build + registration
#
# To use your own build instead, configure vAi with
# -Dllama_DIR=<prefix>/lib/cmake/llama.
#==============================================================================
set -euo pipefail

# The pinned release; benchmarks/*/CMakeLists.txt warn on any other build.
# The tag has no build number, so it is that of b11960 (same commit as v0.8.0).
TAG=v0.8.0
BUILD_NUMBER=11960

[ -n "${HOME:-}" ] || { echo "HOME is not set: the registry lives under \$HOME/.cmake; set it and re-run" >&2; exit 2; }
if [ "$(id -u)" -eq 0 ]; then
    echo "warning: running as root, so this installs into root's cache ($HOME), not yours" >&2
fi

# Deletes nothing outside these: the prefix, its source tree, its vai-* registry entries
# and, on --uninstall, the earlier pins it installed.
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
PREFIX="$CACHE_DIR/vai/llama.cpp-$TAG"
SRC="$PREFIX-src"
CONFIG="$PREFIX/lib/cmake/llama/llama-config.cmake"
REG_DIR="$HOME/.cmake/packages/llama"

JOBS="$(nproc 2>/dev/null || echo 4)"
SHARED=OFF
KEEP_SRC=0
FORCE=0
UNINSTALL=0

usage() {
    sed -n '2,/^#===/p' "$0" | sed 's/^#//'
    cat <<USAGE

Options:
  --jobs N, -j N parallel build jobs (default: $JOBS)
  --shared       shared libraries instead of static
  --keep-src     keep the source and build trees (~600 MB) for re-runs
  --force        reinstall even if this tag is already installed
  --uninstall    remove the build, its source tree, the registry entry and earlier pins
  -h, --help     this text
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --jobs|-j)   JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        --shared)    SHARED=ON;    shift ;;
        --keep-src)  KEEP_SRC=1;   shift ;;
        --force)     FORCE=1;      shift ;;
        --uninstall) UNINSTALL=1;  shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs needs a positive integer, got '$JOBS'" >&2; exit 2; }

variantName() {
    case "$1" in
        ON)  echo shared ;;
        OFF) echo static ;;
        *)   echo "unknown linkage" ;;
    esac
}
VARIANT="$(variantName $SHARED)"

# One vai-* registry entry only: with several, which one find_package picks is
# undefined. -Dllama_DIR still takes precedence over it.
register() {
    mkdir -p "$REG_DIR"
    rm -f "$REG_DIR"/vai-*
    printf '%s\n' "$PREFIX/lib/cmake/llama" > "$REG_DIR/vai-$TAG"
}

# Keeps the GGUF converter once the source is removed. It imports the sibling
# conversion/ package, and its requirements pull in the legacy_llama file.
CONVERT="$PREFIX/convert"
keep_converter() {
    mkdir -p "$CONVERT"
    cp -R "$SRC/convert_hf_to_gguf.py" "$SRC/conversion" "$SRC/gguf-py" "$SRC/LICENSE" "$CONVERT/"
    cp "$SRC/requirements/requirements-convert_hf_to_gguf.txt" \
       "$SRC/requirements/requirements-convert_legacy_llama.txt" "$CONVERT/"
}

# Checks every part, so a partial converter from an older run gets repaired.
converterComplete() {
    [ -f "$CONVERT/convert_hf_to_gguf.py" ] \
        && [ -d "$CONVERT/conversion" ] \
        && [ -d "$CONVERT/gguf-py" ] \
        && [ -f "$CONVERT/requirements-convert_hf_to_gguf.txt" ] \
        && [ -f "$CONVERT/requirements-convert_legacy_llama.txt" ]
}

if [ $UNINSTALL -eq 1 ]; then
    rm -rf "$PREFIX" "$SRC"
    rm -f "$REG_DIR"/vai-*
    echo "removed $PREFIX, $SRC and the vai-* entries in $REG_DIR"
    # Earlier pins this script installed, named llama.cpp-b<build> before the v0.8.0
    # tag, told apart from other llama.cpp builds by the converter only it copies in.
    for old in "$CACHE_DIR"/vai/llama.cpp-b*; do
        [[ "$(basename "$old")" =~ ^llama\.cpp-b[0-9]+$ ]] || continue
        [ -f "$old/convert/convert_hf_to_gguf.py" ] && [ -d "$old/lib/cmake/llama" ] || continue
        rm -rf "$old" "$old-src"
        echo "removed $old, an earlier pin, and its source tree"
    done
    exit 0
fi

#------------------------------------------------------------------------------
# Already installed?
#
# Rebuilds unless the installed build number and linkage both match. A rebuild
# wipes the prefix: a leftover .so next to a new .a would win find_library().
#------------------------------------------------------------------------------
if [ -f "$CONFIG" ]; then
    installed="$(sed -n 's/^set(LLAMA_BUILD_NUMBER *\([0-9]*\)).*/\1/p' "$CONFIG" | head -1)"
    installedShared="$(sed -n 's/^set(LLAMA_SHARED_LIB *\([A-Za-z]*\)).*/\1/p' "$CONFIG" | head -1)"

    if [ $FORCE -eq 0 ] \
       && [ "$installed" = "$BUILD_NUMBER" ] \
       && [ "$installedShared" = "$SHARED" ] \
       && [ -f "$PREFIX/include/mtmd.h" ]; then
        register    # restore the entry if it was removed by hand
        echo "llama.cpp $TAG ($VARIANT) is already installed at $PREFIX"
        # Complete a missing or partial converter from a --keep-src tree.
        if ! converterComplete; then
            if [ -f "$SRC/convert_hf_to_gguf.py" ] && [ -d "$SRC/conversion" ]; then
                keep_converter
                echo "completed the GGUF converter at $CONVERT"
            else
                echo "note: the GGUF converter at $CONVERT is missing or incomplete"
            fi
        fi
        echo "(re-run with --force to rebuild it)"
        exit 0
    fi
fi

#------------------------------------------------------------------------------
# Prerequisites, checked before anything is cloned or built
#------------------------------------------------------------------------------
missing=""
for tool in git cmake glslc; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
    echo "missing required tool(s):$missing" >&2
    echo "  glslc ships with the LunarG Vulkan SDK; its glslc builds llama.cpp's NVIDIA coopmat2 kernels" >&2
    exit 1
fi

# ggml-vulkan needs the SPIRV-Headers CMake package, which libvulkan-dev and
# glslc do not bring. Probed by configuring an empty project.
hasSpirvHeaders() {
    local dir rc=0
    dir="$(mktemp -d)"
    cat > "$dir/CMakeLists.txt" <<'PROBE'
cmake_minimum_required(VERSION 3.16)
project(spirv-headers-probe NONE)
find_package(SPIRV-Headers REQUIRED)
PROBE
    cmake -S "$dir" -B "$dir/build" >/dev/null 2>&1 || rc=1
    rm -rf "$dir"
    return $rc
}

if ! hasSpirvHeaders; then
    cat >&2 <<'MISSING'
missing the SPIRV-Headers CMake package: the Vulkan backend requires it, and
libvulkan-dev/glslc alone do not provide it:
  Ubuntu/Debian   apt install spirv-headers
  Fedora          dnf install spirv-headers-devel
  Arch            pacman -S spirv-headers
  or install the LunarG Vulkan SDK, which ships it
MISSING
    exit 1
fi

GENERATOR=()
command -v ninja >/dev/null 2>&1 && GENERATOR=(-G Ninja)

if [ -f "$CONFIG" ]; then
    echo "replacing llama.cpp build ${installed:-unknown} ($(variantName "${installedShared:-}")) at $PREFIX"
fi

echo "=============================================================="
echo " llama.cpp $TAG -> $PREFIX"
echo " $VARIANT | Vulkan | $JOBS jobs"
echo "=============================================================="

#------------------------------------------------------------------------------
# Source
#
# A --keep-src tree is reused only if it is at $TAG: the build number is
# stamped from $BUILD_NUMBER, so an older tree would pass as the new build.
#------------------------------------------------------------------------------
mkdir -p "$CACHE_DIR/vai"

srcTag="$(git -C "$SRC" rev-parse -q --verify "refs/tags/$TAG^{commit}" 2>/dev/null || true)"
srcHead="$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)"

if [ -n "$srcTag" ] && [ "$srcTag" = "$srcHead" ]; then
    echo "[1/3] reusing source at $SRC"
else
    if [ -d "$SRC" ]; then
        echo "[1/3] source at $SRC is not $TAG, re-cloning..."
    else
        echo "[1/3] cloning $TAG..."
    fi
    rm -rf "$SRC"
    # Ctrl-C mid-clone or mid-build drops the half-made tree (unless --keep-src).
    [ $KEEP_SRC -eq 1 ] || { trap 'rm -rf "$SRC"; exit 130' INT; trap 'rm -rf "$SRC"; exit 143' TERM; }
    git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi

#------------------------------------------------------------------------------
# Build + install
#
# LLAMA_BUILD_APP=OFF: the unified binary does not build with the tools off.
# LLAMA_BUILD_MTMD=ON: the multimodal library the planned Gemma 3 VLM bench links.
# LLAMA_BUILD_NUMBER: a shallow clone cannot derive it, and the benches check it.
# LLAMA_BUILD_IS_DEV=OFF: report version 0.8.0, not 0.8.0-dev.
# LLAMA_OPENSSL=OFF: no HTTPS client; nothing here downloads.
#------------------------------------------------------------------------------
echo "[2/3] building (this is the slow part, a few minutes)..."
# shellcheck disable=SC2016  # $ORIGIN is for the linker, not the shell
cmake -S "$SRC" -B "$SRC/build" "${GENERATOR[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS="$SHARED" \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_MTMD=ON \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_IS_DEV=OFF \
    -DLLAMA_BUILD_NUMBER="$BUILD_NUMBER" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' > "$SRC/configure.log" 2>&1 || {
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
trap - INT TERM

[ -f "$CONFIG" ] || { echo "install finished but $CONFIG is missing" >&2; exit 1; }
register

keep_converter

if [ $KEEP_SRC -eq 0 ]; then
    rm -rf "$SRC"
else
    echo "      source kept at $SRC"
fi

#------------------------------------------------------------------------------
# Done
#------------------------------------------------------------------------------
echo
echo "installed llama.cpp $TAG ($(du -sh "$PREFIX" | cut -f1)) at:"
echo "  $PREFIX"
echo "registered for find_package(llama) in $REG_DIR"
echo "GGUF converter at $CONVERT (see benchmarks/SETUP.md, llama.cpp)"
echo
echo "Next:"
echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON"
echo "  cmake --build build"
echo "  build/bin/bench-gpt2-vai"
