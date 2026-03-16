#!/usr/bin/env bash
# build_ns3_mmwave_sim.sh
#
# Compile src/ns3_mmwave_sim.cc against the local ns3-mmwave build.
#
# Usage:
#   bash src/build_ns3_mmwave_sim.sh [--debug|--release]
#
# Output:
#   src/ns3_mmwave_sim  (binary placed next to this script)
#
# Requirements:
#   - g++ with C++17 support (GCC >= 9)
#   - Compiled ns3-mmwave source tree at
#     /home/dianalab/Projects/ns3-mmwave/build
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/ns3_mmwave_sim.cc"
OUTPUT="$SCRIPT_DIR/ns3_mmwave_sim"

# --- Build mode ---
OPT_FLAGS="-O2"
for arg in "$@"; do
    case "$arg" in
        --debug)   OPT_FLAGS="-O0 -g" ;;
        --release) OPT_FLAGS="-O2"     ;;
    esac
done

echo "=== NetRL ns3 mmWave simulation build ==="
echo "  Source : $SOURCE"

# --- Detect ns3-mmwave build ------------------------------------------------
NS3_MMWAVE_DIR="/home/dianalab/Projects/ns3-mmwave"
NS3_BUILD="$NS3_MMWAVE_DIR/build"

if [ ! -f "$NS3_BUILD/lib/libns3.42-mmwave-default.so" ]; then
    echo "ERROR: ns3-mmwave build not found at $NS3_BUILD"
    echo "  Expected: $NS3_BUILD/lib/libns3.42-mmwave-default.so"
    echo "  Build ns3-mmwave first:"
    echo "    cd $NS3_MMWAVE_DIR && ./ns3 build"
    exit 1
fi

NS3_INC="$NS3_BUILD/include"
NS3_LIB="$NS3_BUILD/lib"
CXX_STD="-std=c++20"
echo "  ns3    : ns3-mmwave 3.42 at $NS3_BUILD"

# --- Libraries needed -------------------------------------------------------
# Include mmwave, lte, buildings + standard ns3 modules that mmwave depends on
LIBS="\
    -lns3.42-mmwave-default \
    -lns3.42-lte-default \
    -lns3.42-buildings-default \
    -lns3.42-spectrum-default \
    -lns3.42-antenna-default \
    -lns3.42-internet-default \
    -lns3.42-point-to-point-default \
    -lns3.42-network-default \
    -lns3.42-mobility-default \
    -lns3.42-propagation-default \
    -lns3.42-core-default"

# --- Compile ----------------------------------------------------------------
echo ""
echo "Compiling with $CXX_STD $OPT_FLAGS ..."

g++ $CXX_STD $OPT_FLAGS \
    -I"$NS3_INC" \
    -L"$NS3_LIB" \
    -Wl,-rpath,"$NS3_LIB" \
    "$SOURCE" \
    $LIBS \
    -o "$OUTPUT"

echo "Built: $OUTPUT"

# --- Smoke test -------------------------------------------------------------
echo ""
echo "Smoke test (QUIT) ..."
if printf 'QUIT\n' | timeout 30 "$OUTPUT" 2>/dev/null | grep -q "READY"; then
    echo "Smoke test PASSED"
else
    echo "Smoke test FAILED — binary returned unexpected output."
    echo "Run:  printf 'QUIT\n' | $OUTPUT"
    exit 1
fi
