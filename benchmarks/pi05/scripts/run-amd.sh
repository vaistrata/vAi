#!/usr/bin/env bash
#==============================================================================
# Runs the pi0.5 AMD pair on the migraphx workload, vAi then MIGraphX, with
# the GPU clock pinned to the top of its DPM table
# (power_dpm_force_performance_level=high), as benchmarks/SETUP.md asks.
#
# The level is set with sudo only when it is not high already, and set back
# to what it was when the script exits, however it exits. An APU shares its
# power with the CPU, so while each bench runs the GPU's actual clock and
# power (hwmon) are sampled, and their range while the GPU is busy is printed
# under the run.
#
#   ./benchmarks/pi05/scripts/run-amd.sh
#   ./benchmarks/pi05/scripts/run-amd.sh -- --atol 0.002 --rtol 0.001
#
# Arguments after -- go to both benches. BUILD_DIR names another build tree
# (default: build); AMD_CARD another card (default: the first amdgpu one).
#==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
BENCHES=(
    "bench-pi05-vai --comparison migraphx"
    "bench-pi05-migraphx"
)
SAMPLE_PERIOD=0.2   # seconds between clock samples
BUSY_PERCENT=50     # a sample counts as under load from this gpu_busy_percent

usage() {
    sed -n '2,/^#===/p' "$0" | sed 's/^#//'
}

BENCH_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --)        shift; BENCH_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for bench in "${BENCHES[@]}"; do
    binary="$BUILD_DIR/bin/${bench%% *}"
    [ -x "$binary" ] || {
        echo "missing $binary: build it first (cmake --build $BUILD_DIR --target ${bench%% *})" >&2
        exit 1
    }
done

#------------------------------------------------------------------------------
# The amdgpu card, found by driver, not number: indices move across boots.
#------------------------------------------------------------------------------
card="${AMD_CARD:-}"
if [ -z "$card" ]; then
    for c in /sys/class/drm/card*; do
        [ -e "$c/device/driver" ] || continue
        if [ "$(basename "$(readlink -f "$c/device/driver")")" = amdgpu ]; then
            card="$c"
            break
        fi
    done
fi
[ -n "$card" ] || { echo "no amdgpu card under /sys/class/drm" >&2; exit 1; }
device="$card/device"
level_file="$device/power_dpm_force_performance_level"
[ -f "$level_file" ] || { echo "missing $level_file" >&2; exit 1; }

hwmon="$(ls -d "$device"/hwmon/hwmon* 2>/dev/null | head -1 || true)"
freq_file="$hwmon/freq1_input"      # shader clock, Hz
power_file="$hwmon/power1_average"  # package power, uW
[ -n "$hwmon" ] && [ -f "$freq_file" ] || freq_file=""
[ -n "$hwmon" ] && [ -f "$power_file" ] || power_file=""

activeLevels() {
    for f in sclk mclk fclk; do
        [ -f "$device/pp_dpm_$f" ] || continue
        printf '%s %s  ' "$f" "$(sed -n 's/^[0-9]*: *\([0-9]*[A-Za-z]*\) *\*$/\1/p' "$device/pp_dpm_$f" | head -1)"
    done
    echo
}

#------------------------------------------------------------------------------
# Pin the level; restore it on exit
#------------------------------------------------------------------------------
previous="$(cat "$level_file")"
sampler=""
restore() {
    [ -z "$sampler" ] || kill "$sampler" 2>/dev/null || true
    if [ "$previous" != high ] && [ "$(cat "$level_file")" != "$previous" ]; then
        echo "$previous" | sudo tee "$level_file" >/dev/null
        echo "clock: $(basename "$card") level restored to $(cat "$level_file")"
    fi
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$previous" = high ]; then
    echo "clock: $(basename "$card") level already high"
else
    echo "clock: $(basename "$card") level $previous -> high (sudo)"
    echo high | sudo tee "$level_file" >/dev/null
    [ "$(cat "$level_file")" = high ] || { echo "could not set $level_file to high" >&2; exit 1; }
fi
echo "clock: active DPM levels: $(activeLevels)"

#------------------------------------------------------------------------------
# Each bench, with the clock sampled while it runs
#------------------------------------------------------------------------------
samples="$(mktemp)"
trap 'rm -f "$samples"; restore' EXIT

sample() {
    while :; do
        printf '%s %s %s\n' \
            "$(cat "$device/gpu_busy_percent" 2>/dev/null || echo 0)" \
            "$( [ -n "$freq_file" ] && cat "$freq_file" 2>/dev/null || echo 0)" \
            "$( [ -n "$power_file" ] && cat "$power_file" 2>/dev/null || echo 0)" >> "$samples"
        sleep "$SAMPLE_PERIOD"
    done
}

summary() {
    awk -v busy="$BUSY_PERCENT" '
        $1 >= busy { n++; f = $2 / 1e6; p = $3 / 1e6; fs += f; ps += p
                     if (n == 1 || f < fmin) fmin = f; if (n == 1 || f > fmax) fmax = f
                     if (n == 1 || p > pmax) pmax = p }
        END {
            if (!n) { print "clock under load: no samples at >= " busy "% busy"; exit }
            printf "clock under load (%d samples at >= %d%% busy): sclk %.0f-%.0f MHz (mean %.0f), power mean %.1f W, max %.1f W\n",
                   n, busy, fmin, fmax, fs / n, ps / n, pmax
        }' "$samples"
}

status=0
for bench in "${BENCHES[@]}"; do
    read -r -a command <<<"$bench"
    echo
    echo "=============================================================="
    echo " ${command[*]} ${BENCH_ARGS[*]}"
    echo "=============================================================="
    : > "$samples"
    sample &
    sampler=$!
    # The vAi bench asks which Vulkan device to use; stdin closed takes its default.
    if ! (cd "$REPO_ROOT" && "$BUILD_DIR/bin/${command[0]}" "${command[@]:1}" "${BENCH_ARGS[@]}" </dev/null); then
        status=1
    fi
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    sampler=""
    summary
    echo "clock: active DPM levels: $(activeLevels)"
done

exit $status
