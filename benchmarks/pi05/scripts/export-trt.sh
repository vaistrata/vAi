#!/usr/bin/env bash
#==============================================================================
# Builds the TensorRT Edge-LLM engines the pi0.5 TensorRT bench reads, from the
# checkpoint in data/ckpts/pi05/lerobot/<size>/, in four steps:
#   1. installs Edge-LLM's Python exporter (trtedge) from the source
#      setup-trt-edge-llm.sh keeps in ${XDG_CACHE_HOME:-~/.cache}/vai/
#      TensorRT-Edge-LLM-v<version>-src into the virtual environment
#      ${XDG_CACHE_HOME:-~/.cache}/vai/venv-trt-export, once
#   2. exports the checkpoint to FP16 ONNX with it, as the openpi configuration
#      Edge-LLM's pi0.5 guide pairs with it (libero: pi05_libero, the bench's
#      configuration; 3B: pi05_aloha, pi05_base with the 50-action chunk), into
#      data/benchmarks/pi05/<size>/trt-onnx/
#   3. adds the normalization statistics the guide names for that configuration
#   4. builds the engines on this GPU with the installed pi05_policy_build into
#      data/benchmarks/pi05/<size>/engines/, where the bench reads them
#
# The exporter environment and a finished export are reused, so a failed build
# does not repeat them; an engine folder already there is never overwritten.
#
# Needs Linux, curl, Python 3.10-3.12 with venv, and TensorRT Edge-LLM from
# setup-trt-edge-llm.sh: its install (TensorRTEdgeLLM_ROOT for another) and the
# source it keeps (TensorRTEdgeLLM_SRC for another checkout of the same tag).
# TRT_EXPORT_VENV names another exporter environment. Set TRT_PACKAGE_DIR for an
# extracted TensorRT SDK, as for the setup script.
#
#   ./benchmarks/pi05/scripts/export-trt.sh                  # --size libero
#   ./benchmarks/pi05/scripts/export-trt.sh --size <size>
#==============================================================================
set -euo pipefail

# The release setup-trt-edge-llm.sh pins; the exporter must be the same one.
VERSION=0.11.1

[ -n "${HOME:-}" ] || { echo "HOME is not set" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/vai"
EDGELLM="${TensorRTEdgeLLM_ROOT:-$CACHE_DIR/TensorRT-Edge-LLM}"
EDGELLM_SRC="${TensorRTEdgeLLM_SRC:-$CACHE_DIR/TensorRT-Edge-LLM-v$VERSION-src}"
VENV="${TRT_EXPORT_VENV:-$CACHE_DIR/venv-trt-export}"
SIZE=libero

usage() {
    sed -n '3,/^#===/p' "$0" | sed 's/^#//'
    cat <<USAGE

Options:
  --size NAME   checkpoint folder (default: $SIZE), one of: libero 3B
  -h, --help    this text
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="${2:?--size needs a name}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
# Edge-LLM's openpi configuration for the checkpoint, and the normalization
# statistics its pi0.5 guide names for that configuration.
OPENPI_ASSETS=https://storage.googleapis.com/openpi-assets/checkpoints
case "$SIZE" in
    libero)
        POLICY_CONFIG=pi05_libero
        NORM_STATS_URL=$OPENPI_ASSETS/pi05_libero/assets/physical-intelligence/libero/norm_stats.json ;;
    3B)
        POLICY_CONFIG=pi05_aloha
        NORM_STATS_URL=$OPENPI_ASSETS/pi05_base/assets/trossen/norm_stats.json ;;
    *) echo "unknown --size '$SIZE', one of: libero 3B" >&2; exit 2 ;;
esac

CHECKPOINT="$ROOT/data/ckpts/pi05/lerobot/$SIZE"
ONNX_DIR="$ROOT/data/benchmarks/pi05/$SIZE/trt-onnx"
ENGINE_DIR="$ROOT/data/benchmarks/pi05/$SIZE/engines"
BUILDER="$EDGELLM/bin/pi05_policy_build"
PLUGIN="$EDGELLM/lib/libNvInfer_edgellm_plugin.so"
# Written into the export once it finishes: what it was made with.
STAMP="$ONNX_DIR/.export"
STAMP_TEXT="$POLICY_CONFIG tensorrt-edgellm $VERSION"

#------------------------------------------------------------------------------
# Prerequisites, checked before anything is exported or built
#------------------------------------------------------------------------------
[ "$(uname -s)" = Linux ] || { echo "TensorRT Edge-LLM engines are built on Linux only" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl not found" >&2; exit 1; }
for file in model.safetensors config.json; do
    [ -f "$CHECKPOINT/$file" ] || {
        echo "missing $CHECKPOINT/$file (see benchmarks/pi05/README.md)" >&2; exit 1;
    }
done
if [ ! -x "$BUILDER" ] || [ ! -f "$PLUGIN" ]; then
    echo "TensorRT Edge-LLM not found at $EDGELLM:" >&2
    echo "  run benchmarks/scripts/setup-trt-edge-llm.sh, or set TensorRTEdgeLLM_ROOT" >&2
    exit 1
fi
exporter_version() {
    "$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("tensorrt-edgellm"))' 2>/dev/null || true
}
# The exporter is installed from the kept source only when the environment lacks it.
if [ -x "$VENV/bin/trtedge" ]; then
    EXPORTER_VERSION="$(exporter_version)"
    [ "$EXPORTER_VERSION" = "$VERSION" ] || {
        echo "exporter in $VENV is tensorrt-edgellm ${EXPORTER_VERSION:-(unknown)}, not $VERSION" >&2
        echo "  remove $VENV to install it again from $EDGELLM_SRC" >&2
        exit 1
    }
else
    [ "$(git -C "$EDGELLM_SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" = \
      "$(git -C "$EDGELLM_SRC" rev-parse -q --verify "refs/tags/v$VERSION^{commit}" 2>/dev/null || echo none)" ] || {
        echo "TensorRT Edge-LLM v$VERSION source not found at $EDGELLM_SRC:" >&2
        echo "  run benchmarks/scripts/setup-trt-edge-llm.sh, which keeps it, or set TensorRTEdgeLLM_SRC" >&2
        exit 1
    }
    command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }
    python3 -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))' || {
        echo "the exporter needs Python 3.10-3.12; python3 is $(python3 -V 2>&1)" >&2; exit 1;
    }
    python3 -c 'import ensurepip, venv' 2>/dev/null || {
        echo "python3 cannot create a virtual environment: install its venv module (Ubuntu: python3-venv)" >&2; exit 1;
    }
fi
[ ! -e "$ENGINE_DIR" ] || {
    echo "engines already exist: $ENGINE_DIR (remove the folder to rebuild)" >&2; exit 1;
}
if [ -e "$ONNX_DIR" ] && [ "$(cat "$STAMP" 2>/dev/null || true)" != "$STAMP_TEXT" ]; then
    echo "$ONNX_DIR holds an unfinished or other export (remove it to export again)" >&2
    exit 1
fi

# A step that fails or is interrupted removes the folder it was writing.
PARTIAL=""
trap '[ -z "$PARTIAL" ] || rm -rf "$PARTIAL"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "=============================================================="
echo " pi0.5 $SIZE -> TensorRT Edge-LLM $VERSION engines ($POLICY_CONFIG)"
echo " checkpoint: $CHECKPOINT"
echo " exporter:   $VENV"
echo " engines:    $ENGINE_DIR"
echo "=============================================================="

#------------------------------------------------------------------------------
# 1. Exporter, from the kept Edge-LLM source
#
# The [export] extra is all the export needs: it runs on the CPU, without
# Edge-LLM's native build.
#------------------------------------------------------------------------------
if [ -x "$VENV/bin/trtedge" ]; then
    echo "[1/4] reusing the exporter in $VENV"
else
    echo "[1/4] installing the exporter from $EDGELLM_SRC into $VENV..."
    mkdir -p "$(dirname "$VENV")"
    PARTIAL="$VENV"
    python3 -m venv "$VENV"
    PYTHONNOUSERSITE=1 "$VENV/bin/python" -m pip install --quiet --upgrade pip
    PYTHONNOUSERSITE=1 "$VENV/bin/python" -m pip install --quiet -e "$EDGELLM_SRC[export]"
    EXPORTER_VERSION="$(exporter_version)"
    [ "$EXPORTER_VERSION" = "$VERSION" ] || {
        echo "installed tensorrt-edgellm ${EXPORTER_VERSION:-(unknown)}, not $VERSION" >&2; exit 1;
    }
    PARTIAL=""
fi

#------------------------------------------------------------------------------
# 2. Export to FP16 ONNX
#------------------------------------------------------------------------------
if [ -e "$STAMP" ]; then
    echo "[2/4] reusing the export in $ONNX_DIR"
else
    echo "[2/4] exporting to FP16 ONNX in $ONNX_DIR..."
    mkdir -p "$(dirname "$ONNX_DIR")"
    PARTIAL="$ONNX_DIR"
    PYTHONNOUSERSITE=1 "$VENV/bin/trtedge" export "$CHECKPOINT" "$ONNX_DIR" \
        --dtype float16 --pi05-policy-config "$POLICY_CONFIG"
    printf '%s\n' "$STAMP_TEXT" > "$STAMP"
    PARTIAL=""
fi

#------------------------------------------------------------------------------
# 3. Normalization statistics
#------------------------------------------------------------------------------
NORM_STATS="$ONNX_DIR/assets/norm_stats.json"
if [ -f "$NORM_STATS" ]; then
    echo "[3/4] reusing $NORM_STATS"
else
    echo "[3/4] downloading the normalization statistics..."
    PARTIAL="$NORM_STATS.partial"
    curl --fail --location --silent --show-error --create-dirs --output "$NORM_STATS.partial" "$NORM_STATS_URL"
    mv "$NORM_STATS.partial" "$NORM_STATS"
    PARTIAL=""
fi

#------------------------------------------------------------------------------
# 4. Engines, built on this GPU
#------------------------------------------------------------------------------
echo "[4/4] building the engines in $ENGINE_DIR..."
PARTIAL="$ENGINE_DIR"
# TensorRT's builder opens its builder resource library by name, which outside the
# system search paths (an extracted SDK in TRT_PACKAGE_DIR) only LD_LIBRARY_PATH finds.
EDGELLM_PLUGIN_PATH="$PLUGIN" \
LD_LIBRARY_PATH="$EDGELLM/lib${TRT_PACKAGE_DIR:+:$TRT_PACKAGE_DIR/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$BUILDER" --onnx-dir "$ONNX_DIR" --engine-dir "$ENGINE_DIR"
PARTIAL=""

echo
echo "engines built in $ENGINE_DIR"
