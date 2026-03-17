#!/usr/bin/env bash
# build_ns3_multi_ue_sim.sh
#
# Compile src/ns3_wifi_multi_ue_sim.cc against the installed ns-3 library.
#
# Usage:
#   bash src/build_ns3_multi_ue_sim.sh [--debug|--release]
#
# Output:
#   src/ns3_wifi_multi_ue_sim  (binary placed next to this script)
#
# Requirements:
#   - g++ with C++20 support  (GCC >= 10, Clang >= 11)
#   - ns3 installed via pip  (pip install ns3)            [ns3 >= 3.43]
#     OR a compiled ns3-mmwave source tree at
#     /home/dianalab/Projects/ns3-mmwave/                 [ns3 3.42]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT_FLAGS="-O2"

for arg in "$@"; do
    case "$arg" in
        --debug)   OPT_FLAGS="-O0 -g" ;;
        --release) OPT_FLAGS="-O2"     ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "=== NetRL ns3 WiFi multi-UE simulation build ==="
echo "  Source : $SCRIPT_DIR/ns3_wifi_multi_ue_sim.cc"

# ---------------------------------------------------------------------------
# Detect ns3 installation
# Preference 1 : pip-installed ns3 (direct g++ compilation, C++20 required)
# Preference 2 : ns3-mmwave source build (C++17 may suffice for ns3 3.42)
# ---------------------------------------------------------------------------

NS3_PIP_PREFIX=$(python3 -c "
import importlib.util, os, sys
spec = importlib.util.find_spec('ns3')
if not spec or not spec.submodule_search_locations:
    sys.exit(0)
for p in list(spec.submodule_search_locations):
    candidate = os.path.join(os.path.dirname(p), 'ns3')
    if os.path.isfile(os.path.join(candidate, 'include', 'ns3', 'simulator.h')):
        print(candidate)
        sys.exit(0)
" 2>/dev/null || true)

NS3_MMWAVE_BUILD="/home/dianalab/Projects/ns3-mmwave/build"

if [[ -n "$NS3_PIP_PREFIX" ]]; then
    NS3_INC="$NS3_PIP_PREFIX/include"
    NS3_LIB="$NS3_PIP_PREFIX/lib64"
    NS3_VER=$(ls "$NS3_LIB"/libns3.*-core*.so 2>/dev/null | \
              grep -oP 'libns3\.\K[\d.]+' | head -1 || echo "44")
    CXX_STD="-std=c++20"
    LIBS="-lns3.${NS3_VER}-core -lns3.${NS3_VER}-network -lns3.${NS3_VER}-internet -lns3.${NS3_VER}-wifi -lns3.${NS3_VER}-mobility -lns3.${NS3_VER}-propagation"
    echo "  ns3    : pip-installed ns3-${NS3_VER} at $NS3_PIP_PREFIX"

elif [[ -d "$NS3_MMWAVE_BUILD/lib" ]]; then
    NS3_INC="$NS3_MMWAVE_BUILD/include"
    NS3_LIB="$NS3_MMWAVE_BUILD/lib"
    CXX_STD="-std=c++17"
    LIBS="-lns3.42-core-default -lns3.42-network-default -lns3.42-internet-default -lns3.42-wifi-default -lns3.42-mobility-default -lns3.42-propagation-default"
    echo "  ns3    : ns3-mmwave build at $NS3_MMWAVE_BUILD (ns3 3.42)"

else
    echo ""
    echo "ERROR: No usable ns3 installation found."
    echo "  Install via pip:  pip install ns3"
    echo "  Or build ns3-mmwave from source."
    exit 1
fi

echo ""
echo "Compiling with $CXX_STD $OPT_FLAGS ..."
g++ $CXX_STD $OPT_FLAGS \
    -I"$NS3_INC" \
    -L"$NS3_LIB" \
    -Wl,-rpath,"$NS3_LIB" \
    "$SCRIPT_DIR/ns3_wifi_multi_ue_sim.cc" \
    $LIBS \
    -o "$SCRIPT_DIR/ns3_wifi_multi_ue_sim"

echo "Built: $SCRIPT_DIR/ns3_wifi_multi_ue_sim"
echo ""
echo "Smoke test (QUIT) ..."
printf 'QUIT\n' | timeout 60 "$SCRIPT_DIR/ns3_wifi_multi_ue_sim" 2>/dev/null | grep -q READY && \
    echo "Smoke test PASSED" || echo "Smoke test FAILED (check stderr)"
