#!/usr/bin/env bash
#==============================================================================
# Installs the MIGraphX the pi0.5 bench compares against: release 2.18.0 and
# the ROCm 10.1.0 runtime it ships with, from AMD's prebuilt files (no build).
#
# Unpacks, without apt and without root, into
# ${XDG_CACHE_HOME:-~/.cache}/vai/migraphx-<version>:
#   - the amdrocm10-migraphx and amdrocm10-migraphx-dev Ubuntu 24.04 packages;
#   - from the ROCm 10.1.0 tarball of this GPU, the runtime MIGraphX loads
#     (HIP, hiprtc, comgr, HSA, MIOpen, rocBLAS, hipBLASLt and their kernels),
#     into the same lib folder, about 1.2 GB of the 2 GB download.
# MIGraphX looks for its ROCm libraries next to itself, and the bench's RUNPATH
# names that folder before the system ROCm's, so the bench runs on ROCm 10.1
# whichever ROCm is installed; that one only provides the HIP headers to build.
# It registers the prefix in CMake's user package registry: the bench asks for
# 2.18, so find_package(migraphx) passes over another release ROCm carries.
#
# The bench then compiles its graphs with MIGraphX's MLIR GEMM path (CMake's
# MIGRAPHX_GEMM_PROVIDER=mlir, the default) on the bundle export-onnx.py writes,
# its constants folded (the default): the pairing that runs fastest.
#
# Needs curl, dpkg-deb, tar and sha256sum, on Linux x86_64, and the amdgpu
# kernel driver of a ROCm 10.1 GPU (the tarball is per GPU, --arch).
#
#   ./benchmarks/scripts/setup-migraphx.sh                 # ~2.1 GB download, ~1.6 GB installed
#   ./benchmarks/scripts/setup-migraphx.sh --uninstall     # remove the install + registration
#
# To use another install of the pinned release instead, configure vAi with
# -Dmigraphx_DIR=<prefix>/lib/cmake/migraphx.
#==============================================================================
set -euo pipefail

# The pinned release; benchmarks/pi05/CMakeLists.txt skips the bench on any other.
VERSION=2.18.0
REVISION=1
REPO=https://stable.repo.amd.com/rocm/migraphx/packages/ubuntu2404
# Package -> SHA256, from the repository's Packages index for $VERSION-$REVISION.
declare -A SHA256=(
    [amdrocm10-migraphx]=dd6d810ad9a81916b44e442a922255125c550218df1750d7e820e7c42b21a090
    [amdrocm10-migraphx-dev]=134a80e24ba5950f62578eeffb996e6817dbc325c5a9e13ba0860126b22103fe
)
PACKAGES=(amdrocm10-migraphx amdrocm10-migraphx-dev)
# The folder both packages install under /opt/rocm, which becomes the prefix.
PKG_ROOT=opt/rocm/extras-10

# The ROCm release MIGraphX 2.18 ships with, one tarball per GPU. AMD publishes
# no checksum for them: these are the SHA256 of the files this script was
# written against, and an architecture without one is installed unverified.
ROCM_VERSION=10.1.0
ROCM_REPO=https://stable.repo.amd.com/rocm/core/tarball
declare -A ROCM_SHA256=(
    [gfx1151]=50deb993035bc2f6a5aba5e3de8d476d8c512a2eec2cbfd5aa569d71e0b8b337
)

[ -n "${HOME:-}" ] || { echo "HOME is not set: the registry lives under \$HOME/.cmake; set it and re-run" >&2; exit 2; }
if [ "$(id -u)" -eq 0 ]; then
    echo "warning: running as root, so this installs into root's cache ($HOME), not yours" >&2
fi

# Deletes nothing outside these: the prefix, its download folder and its vai-* registry entries.
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
PREFIX="$CACHE_DIR/vai/migraphx-$VERSION"
DOWNLOAD="$PREFIX-download"
STAMP="$PREFIX/.vai-pin"
CONFIG="$PREFIX/lib/cmake/migraphx/migraphx-config.cmake"
REG_DIR="$HOME/.cmake/packages/migraphx"

ARCH=""
FORCE=0
UNINSTALL=0

usage() {
    sed -n '2,/^#===/p' "$0" | sed 's/^#//'
    cat <<USAGE

Options:
  --arch NAME    the GPU whose ROCm tarball to take, e.g. gfx1151 (default: the first GPU)
  --force        reinstall even if this release is already installed
  --uninstall    remove the install, its download folder and the registry entry
  -h, --help     this text
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --arch)      ARCH="${2:?--arch needs a name, e.g. gfx1151}"; shift 2 ;;
        --force)     FORCE=1;     shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# One vai-* registry entry only: with several, which one find_package picks is
# undefined. -Dmigraphx_DIR still takes precedence over it.
register() {
    mkdir -p "$REG_DIR"
    rm -f "$REG_DIR"/vai-*
    printf '%s\n' "$PREFIX/lib/cmake/migraphx" > "$REG_DIR/vai-$VERSION"
}

if [ $UNINSTALL -eq 1 ]; then
    rm -rf "$PREFIX" "$DOWNLOAD"
    rm -f "$REG_DIR"/vai-*
    echo "removed $PREFIX and the vai-* entries in $REG_DIR"
    exit 0
fi

# The first GPU's LLVM target, from the amdgpu driver's topology: gfx_target_version
# 110501 is gfx1151 (major, then minor and stepping in hex).
detectArch() {
    local node version
    for node in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        version="$(sed -n 's/^gfx_target_version \([0-9]*\)$/\1/p' "$node" 2>/dev/null)"
        if [ -n "$version" ] && [ "$version" -gt 0 ]; then
            printf 'gfx%d%x%x\n' $((version / 10000)) $((version / 100 % 100)) $((version % 100))
            return
        fi
    done
}
[ -n "$ARCH" ] || ARCH="$(detectArch)"
[ -n "$ARCH" ] || { echo "no AMD GPU under /sys/class/kfd: name one with --arch, e.g. --arch gfx1151" >&2; exit 1; }
PIN="$VERSION-$REVISION rocm-$ROCM_VERSION $ARCH"

#------------------------------------------------------------------------------
# Already installed?
#------------------------------------------------------------------------------
if [ $FORCE -eq 0 ] && [ -f "$CONFIG" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$PIN" ]; then
    register    # restore the entry if it was removed by hand
    echo "MIGraphX $VERSION with ROCm $ROCM_VERSION ($ARCH) is already installed at $PREFIX"
    echo "(re-run with --force to reinstall it)"
    exit 0
fi

#------------------------------------------------------------------------------
# Prerequisites, checked before anything is downloaded
#------------------------------------------------------------------------------
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    echo "MIGraphX's packages are for Linux x86_64 only" >&2
    exit 1
fi

missing=""
for tool in curl dpkg-deb tar sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -n "$missing" ]; then
    echo "missing required tool(s):$missing" >&2
    exit 1
fi

ROCM_TARBALL="therock-dist-linux-$ARCH-$ROCM_VERSION.tar.gz"
if [ "$(curl -sIo /dev/null -w '%{http_code}' "$ROCM_REPO/$ROCM_TARBALL")" != 200 ]; then
    echo "AMD publishes no ROCm $ROCM_VERSION tarball for $ARCH ($ROCM_REPO/$ROCM_TARBALL)" >&2
    exit 1
fi

if [ -f "$CONFIG" ]; then
    echo "replacing the MIGraphX install at $PREFIX"
fi

echo "=============================================================="
echo " MIGraphX $VERSION + ROCm $ROCM_VERSION runtime ($ARCH) -> $PREFIX"
echo "=============================================================="

#------------------------------------------------------------------------------
# MIGraphX packages, checked against the pinned hashes
#------------------------------------------------------------------------------
mkdir -p "$CACHE_DIR/vai"
rm -rf "$DOWNLOAD"
mkdir -p "$DOWNLOAD"
trap 'rm -rf "$DOWNLOAD"' EXIT
staging="$DOWNLOAD/root"

echo "[1/3] MIGraphX $VERSION packages..."
for pkg in "${PACKAGES[@]}"; do
    deb="${pkg}_${VERSION}-${REVISION}_amd64.deb"
    curl -fL --retry 3 --progress-bar -o "$DOWNLOAD/$deb" "$REPO/pool/main/$deb"
    actual="$(sha256sum "$DOWNLOAD/$deb" | cut -d' ' -f1)"
    if [ "$actual" != "${SHA256[$pkg]}" ]; then
        echo "checksum mismatch for $deb: expected ${SHA256[$pkg]}, got $actual" >&2
        exit 1
    fi
    dpkg-deb -x "$DOWNLOAD/$deb" "$staging"
    rm -f "$DOWNLOAD/$deb"
done
[ -f "$staging/$PKG_ROOT/lib/cmake/migraphx/migraphx-config.cmake" ] \
    || { echo "the packages carry no $PKG_ROOT/lib/cmake/migraphx: an unexpected layout" >&2; exit 1; }

#------------------------------------------------------------------------------
# ROCm runtime, streamed: only what MIGraphX loads is written to disk
#
# The libraries are the closure of MIGraphX's NEEDED entries through the
# tarball's own RUNPATHs (lib, lib/llvm/lib, lib/rocm_sysdeps/lib), plus what
# they open at run time: hiprtc's builtin headers, the kernel libraries of
# rocBLAS, hipBLASLt and MIOpen, their .kpack device code, and the device
# libraries and clang headers that runtime compilation reads.
#------------------------------------------------------------------------------
ROCM_MEMBERS=(
    './lib/libamdhip64.so.7*' './lib/libhiprtc.so.7*' './lib/libhiprtc-builtins.so.7*'
    './lib/libamd_comgr.so.3*' './lib/libhsa-runtime64.so.1*' './lib/librocm_kpack.so.0*'
    './lib/librocprofiler-register.so.0*' './lib/libroctx64.so.4*'
    './lib/libMIOpen.so.1*' './lib/librocblas.so.5*' './lib/libhipblaslt.so.1*'
    './lib/liborigami.so.1*' './lib/librocroller.so.1*'
    './lib/llvm/lib/libLLVM.so.*' './lib/llvm/lib/libclang-cpp.so.*' './lib/llvm/lib/clang/*'
    './lib/llvm/amdgcn/*' './amdgcn'
    './lib/rocm_sysdeps/lib/librocm_sysdeps_bz2.so*' './lib/rocm_sysdeps/lib/librocm_sysdeps_drm.so.2*'
    './lib/rocm_sysdeps/lib/librocm_sysdeps_drm_amdgpu.so.1*' './lib/rocm_sysdeps/lib/librocm_sysdeps_elf.so.1*'
    './lib/rocm_sysdeps/lib/librocm_sysdeps_liblzma.so.5*' './lib/rocm_sysdeps/lib/librocm_sysdeps_numa.so.1*'
    './lib/rocm_sysdeps/lib/librocm_sysdeps_sqlite3.so*' './lib/rocm_sysdeps/lib/librocm_sysdeps_z.so.1*'
    './lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1*'
    './lib/rocblas/library/*' './lib/hipblaslt/library/*' './share/miopen/*'
    "./.kpack/blas_lib_$ARCH.kpack" "./.kpack/miopen_lib_$ARCH.kpack"
)

echo "[2/3] ROCm $ROCM_VERSION runtime from $ROCM_TARBALL (2 GB, streamed)..."
# Next to MIGraphX: its RUNPATH and the bench's both name this lib folder first.
rocm="$staging/$PKG_ROOT"
mkfifo "$DOWNLOAD/stream"
sha256sum < "$DOWNLOAD/stream" | cut -d' ' -f1 > "$DOWNLOAD/rocm.sha256" &
hasher=$!
curl -fL --retry 3 --progress-bar "$ROCM_REPO/$ROCM_TARBALL" | tee "$DOWNLOAD/stream" \
    | tar -xzf - -C "$rocm" --wildcards "${ROCM_MEMBERS[@]}"
wait "$hasher"
actual="$(cat "$DOWNLOAD/rocm.sha256")"
expected="${ROCM_SHA256[$ARCH]:-}"
if [ -z "$expected" ]; then
    echo "warning: no pinned SHA256 for $ROCM_TARBALL; it hashed to $actual" >&2
elif [ "$actual" != "$expected" ]; then
    echo "checksum mismatch for $ROCM_TARBALL: expected $expected, got $actual" >&2
    exit 1
fi
[ -e "$rocm/lib/libamdhip64.so.7" ] && [ -e "$rocm/lib/libMIOpen.so.1" ] \
    || { echo "$ROCM_TARBALL lacks the expected runtime libraries" >&2; exit 1; }

#------------------------------------------------------------------------------
# Swap the prefix in
#
# The old install is removed only once everything is unpacked; signals are held
# off meanwhile so a Ctrl-C cannot leave a half-installed prefix.
#------------------------------------------------------------------------------
echo "[3/3] installing..."
trap '' INT TERM
rm -rf "$PREFIX"
mv "$staging/$PKG_ROOT" "$PREFIX"
printf '%s\n' "$PIN" > "$STAMP"
trap - INT TERM
register

#------------------------------------------------------------------------------
# Done
#------------------------------------------------------------------------------
echo
echo "installed MIGraphX $VERSION with the ROCm $ROCM_VERSION runtime for $ARCH ($(du -sh "$PREFIX" | cut -f1)) at:"
echo "  $PREFIX"
echo "registered for find_package(migraphx) in $REG_DIR"
echo
echo "Next (see benchmarks/pi05/README.md):"
echo "  python benchmarks/pi05/scripts/export-onnx.py     # FP16 bundle, constants folded"
echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAI_BUILD_BENCH=ON -DMIGRAPHX_GEMM_PROVIDER=mlir"
echo "  cmake --build build --target bench-pi05-migraphx"
echo "  build/bin/bench-pi05-migraphx"
