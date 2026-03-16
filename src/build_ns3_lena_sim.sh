#!/usr/bin/env bash
# build_ns3_lena_sim.sh
#
# Compile src/ns3_lena_sim.cc against a local 5G-LENA ns-3 build.
#
# Usage:
#   bash src/build_ns3_lena_sim.sh [--debug|--release]
#
# Environment override:
#   NS3_LENA_DIR=/path/to/5g-lena/ns-3-dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/ns3_lena_sim.cc"
OUTPUT="$SCRIPT_DIR/ns3_lena_sim"

OPT_FLAGS="-O2"
for arg in "$@"; do
    case "$arg" in
        --debug)   OPT_FLAGS="-O0 -g" ;;
        --release) OPT_FLAGS="-O2" ;;
    esac
done

NS3_LENA_DIR="${NS3_LENA_DIR:-/home/dianalab/Projects/5g-lena/ns-3-dev}"
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
