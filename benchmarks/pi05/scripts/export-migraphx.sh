#!/usr/bin/env bash
#==============================================================================
# Exports the ONNX bundle the pi0.5 MIGraphX bench reads, from the LIBERO
# checkpoint in data/ckpts/pi05/lerobot/libero/, in two steps:
#   1. creates the Python environment export-onnx.py checks for (lerobot 0.6.0,
#      transformers 5.5.4, torch 2.8.0 for the CPU, onnx 1.19.0) in
#      ${XDG_CACHE_HOME:-~/.cache}/vai/venv-pi05-export, once; reference.py
#      runs in the same one
#   2. runs migraphx/export-onnx.py in it: FP16 prefill and step graphs, their
#      constants folded, into data/benchmarks/pi05/libero/, where the bench
#      reads them next to reference.py's files
#
# The export runs on the CPU and needs no MIGraphX: the bench compiles the
# graphs on its GPU. The environment is reused; a bundle already there is never
# overwritten.
#
# Needs Python 3.12 with venv (uv is used when found), a C++23 compiler for the
# input helper export-onnx.py builds from vAi's sources (CXX, else the first of
# g++-15, g++-14, clang++-20..18, clang++, c++ that has C++23), the checkpoint,
# the PaliGemma tokenizer and enough RAM for the FP32 checkpoint (14.5 GB).
# PI05_EXPORT_VENV names another environment.
#
#   ./benchmarks/pi05/scripts/export-migraphx.sh
#   ./benchmarks/pi05/scripts/export-migraphx.sh -- --no-constant-folding   # for the hipBLASLt GEMM path
#
# Arguments after -- go to export-onnx.py.
#==============================================================================
set -euo pipefail

[ -n "${HOME:-}" ] || { echo "HOME is not set" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/vai"
VENV="${PI05_EXPORT_VENV:-$CACHE_DIR/venv-pi05-export}"
DATASET=libero

# The versions export-onnx.py checks, and the rest of the environment it was run with.
TORCH_INDEX=https://download.pytorch.org/whl/cpu
TORCH_PACKAGES=(torch==2.8.0 torchvision==0.23.0)
PACKAGES=(lerobot==0.6.0 transformers==5.5.4 onnx==1.19.0 numpy==2.2.6 pillow)
CHECKED="lerobot==0.6.0 transformers==5.5.4 torch==2.8.0 onnx==1.19.0"

usage() {
    sed -n '2,/^#===/p' "$0" | sed 's/^#//'
    cat <<USAGE

Options:
  -h, --help    this text
USAGE
}

EXPORT_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --)        shift; EXPORT_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

CHECKPOINT="$ROOT/data/ckpts/pi05/lerobot/$DATASET"
TOKENIZER="$ROOT/data/tokenizers/paligemma/tokenizer.json"
OUTPUT="$ROOT/data/benchmarks/pi05/$DATASET"
BUNDLE=(prefill.fp16.onnx prefill.fp16.weights.bin step.fp16.onnx step.fp16.weights.bin metadata.json config.txt)

#------------------------------------------------------------------------------
# Prerequisites, checked before anything is installed or exported
#------------------------------------------------------------------------------
[ "$(uname -s)" = Linux ] || { echo "this export is set up for Linux only" >&2; exit 1; }
for file in model.safetensors config.json; do
    [ -f "$CHECKPOINT/$file" ] || {
        echo "missing $CHECKPOINT/$file (see benchmarks/pi05/README.md)" >&2; exit 1;
    }
done
[ -f "$TOKENIZER" ] || { echo "missing $TOKENIZER (see benchmarks/pi05/README.md)" >&2; exit 1; }
# A bundle sent elsewhere with -- --output DIR is checked by export-onnx.py itself.
for arg in ${EXPORT_ARGS[@]+"${EXPORT_ARGS[@]}"}; do
    case "$arg" in --output|--output=*|--data-root|--data-root=*) BUNDLE=() ;; esac
done
for file in ${BUNDLE[@]+"${BUNDLE[@]}"}; do
    [ ! -e "$OUTPUT/$file" ] || {
        echo "a bundle already exists in $OUTPUT ($file): remove its files to export again" >&2; exit 1;
    }
done

# The input helper uses C++23 explicit object parameters, which g++ 13 lacks.
cxx_ok() {
    command -v "$1" >/dev/null 2>&1 || return 1
    printf 'struct P { constexpr int v(this const P&) { return 0; } };\nstatic_assert(P{}.v() == 0);\n' \
        | "$1" -std=c++23 -fsyntax-only -x c++ - >/dev/null 2>&1
}
if [ -n "${CXX:-}" ]; then
    COMPILER="$CXX"
else
    COMPILER=""
    for candidate in g++-15 g++-14 clang++-20 clang++-19 clang++-18 clang++ c++; do
        if cxx_ok "$candidate"; then COMPILER="$candidate"; break; fi
    done
    [ -n "$COMPILER" ] || { echo "no C++23 compiler found: install g++-14 or clang++-18, or set CXX" >&2; exit 1; }
fi

if [ ! -x "$VENV/bin/python" ]; then
    PYTHON=""
    for candidate in python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1 \
           && "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
            PYTHON="$candidate"; break
        fi
    done
    command -v uv >/dev/null 2>&1 || [ -n "$PYTHON" ] || {
        echo "Python 3.12 not found (lerobot 0.6.0 needs it): install python3.12 or uv" >&2; exit 1;
    }
fi

# An interrupted environment install is removed, so the next run starts it over.
PARTIAL=""
trap '[ -z "$PARTIAL" ] || rm -rf "$PARTIAL"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "=============================================================="
echo " pi0.5 $DATASET -> MIGraphX ONNX bundle (FP16)"
echo " checkpoint: $CHECKPOINT"
echo " python:     $VENV"
echo " C++:        $COMPILER"
echo " bundle:     $OUTPUT"
echo "=============================================================="

#------------------------------------------------------------------------------
# 1. Python environment
#
# The CPU torch goes in first: lerobot would otherwise pull a CUDA build of
# several GB.
#------------------------------------------------------------------------------
if [ -x "$VENV/bin/python" ]; then
    echo "[1/2] reusing the environment in $VENV"
else
    echo "[1/2] creating the environment in $VENV..."
    mkdir -p "$(dirname "$VENV")"
    PARTIAL="$VENV"
    if command -v uv >/dev/null 2>&1; then
        uv venv --quiet "$VENV" --python 3.12
        uv pip install --quiet --python "$VENV/bin/python" "${TORCH_PACKAGES[@]}" --index-url "$TORCH_INDEX"
        uv pip install --quiet --python "$VENV/bin/python" "${PACKAGES[@]}"
    else
        "$PYTHON" -m venv "$VENV"
        "$VENV/bin/python" -m pip install --quiet --upgrade pip
        "$VENV/bin/python" -m pip install --quiet "${TORCH_PACKAGES[@]}" --index-url "$TORCH_INDEX"
        "$VENV/bin/python" -m pip install --quiet "${PACKAGES[@]}"
    fi
    PARTIAL=""
fi

# export-onnx.py checks them too, but only after loading torch; this says which to fix first.
for pin in $CHECKED; do
    name="${pin%%==*}" expected="${pin#*==}"
    found="$("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('$name'))" 2>/dev/null || true)"
    [ "${found%%+*}" = "$expected" ] || {
        echo "$VENV has $name ${found:-(not installed)}, not $expected: remove it to create it again" >&2; exit 1;
    }
done

#------------------------------------------------------------------------------
# 2. Export
#------------------------------------------------------------------------------
echo "[2/2] exporting (CPU, several minutes)..."
PYTHONNOUSERSITE=1 "$VENV/bin/python" "$ROOT/benchmarks/pi05/migraphx/export-onnx.py" \
    --cxx "$COMPILER" ${EXPORT_ARGS[@]+"${EXPORT_ARGS[@]}"}

echo
echo "bundle exported to $OUTPUT"
