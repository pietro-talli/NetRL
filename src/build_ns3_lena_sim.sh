#!/usr/bin/env bash
# build_ns3_lena_sim.sh
#
# Compile src/ns3_lena_sim.cc against a local 5G-LENA ns-3 build.
#
# Usage:
#   bash src/build_ns3_lena_sim.sh [--debug|--release] [--ns3-lena-dir PATH]
#   bash src/build_ns3_lena_sim.sh [--debug|--release] [PATH]
#
# Environment override:
#   NS3_LENA_DIR=/path/to/5g-lena/ns-3-dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/ns3_lena_sim.cc"
OUTPUT="$SCRIPT_DIR/ns3_lena_sim"

OPT_FLAGS="-O2"
NS3_LENA_DIR="${NS3_LENA_DIR:-/home/dianalab/Projects/5g-lena/ns-3-dev}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            OPT_FLAGS="-O0 -g"
            shift
            ;;
        --release)
            OPT_FLAGS="-O2"
            shift
            ;;
        --ns3-lena-dir)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --ns3-lena-dir requires a path argument"
                exit 1
            fi
            NS3_LENA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash src/build_ns3_lena_sim.sh [--debug|--release] [--ns3-lena-dir PATH]"
            echo "   or: bash src/build_ns3_lena_sim.sh [--debug|--release] [PATH]"
            echo "Default path: /home/dianalab/Projects/5g-lena/ns-3-dev"
            echo "Env override: NS3_LENA_DIR=/path/to/5g-lena/ns-3-dev"
            exit 0
            ;;
        *)
            # Support positional PATH for convenience.
            if [[ -z "${POSITIONAL_PATH_SET:-}" ]]; then
                NS3_LENA_DIR="$1"
                POSITIONAL_PATH_SET=1
                shift
            else
                echo "ERROR: Unknown argument '$1'"
                exit 1
            fi
            ;;
    esac
done

NS3_BUILD="$NS3_LENA_DIR/build"
NS3_INC="$NS3_BUILD/include"
NS3_LIB="$NS3_BUILD/lib"

if [ ! -d "$NS3_LIB" ]; then
    echo "ERROR: 5G-LENA build libraries not found at $NS3_LIB"
    echo "Build first:"
    echo "  cd $NS3_LENA_DIR && ./ns3 build"
    exit 1
fi

CORE_LIB="$(ls "$NS3_LIB"/libns3.*-core-default.so 2>/dev/null | head -n 1 || true)"
if [ -z "$CORE_LIB" ]; then
    echo "ERROR: Could not find libns3.*-core-default.so in $NS3_LIB"
    exit 1
fi

NS3_VER="$(basename "$CORE_LIB" | sed -E 's/^libns3\.([^-]+)-core-default\.so$/\1/')"

echo "=== NetRL ns3 5G-LENA simulation build ==="
echo "  Source  : $SOURCE"
echo "  ns3-dev : $NS3_LENA_DIR"
echo "  version : $NS3_VER"

CXX_STD="-std=c++20"

LIBS="\
    -lns3.${NS3_VER}-nr-default \
    -lns3.${NS3_VER}-lte-default \
    -lns3.${NS3_VER}-antenna-default \
    -lns3.${NS3_VER}-buildings-default \
    -lns3.${NS3_VER}-spectrum-default \
    -lns3.${NS3_VER}-internet-default \
    -lns3.${NS3_VER}-point-to-point-default \
    -lns3.${NS3_VER}-network-default \
    -lns3.${NS3_VER}-mobility-default \
    -lns3.${NS3_VER}-propagation-default \
    -lns3.${NS3_VER}-core-default"

g++ $CXX_STD $OPT_FLAGS \
    -I"$NS3_INC" \
    -L"$NS3_LIB" \
    -Wl,-rpath,"$NS3_LIB" \
    "$SOURCE" \
    $LIBS \
    -o "$OUTPUT"

echo "Built: $OUTPUT"

echo "Smoke test (QUIT) ..."
if printf 'QUIT\n' | timeout 30 "$OUTPUT" 2>/dev/null | grep -q "READY"; then
    echo "Smoke test PASSED"
else
    echo "Smoke test FAILED — binary returned unexpected output."
    echo "Run: printf 'QUIT\n' | $OUTPUT"
    exit 1
fi
