cmake_minimum_required(VERSION 3.16)
project(netrl_ns3_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate ns3
#
# Strategy 1: pip-installed ns3 (ns3 PyPI package ships CMake config).
#   The CMake config lives at:
#     <site-packages>/ns3/lib64/cmake/ns3/ns3Config.cmake
#
# Strategy 2: CMake package registry / system install.
# ---------------------------------------------------------------------------

# Detect pip-installed ns3 prefix via Python
execute_process(
    COMMAND python3 -c [=[
import importlib.util, os, sys
spec = importlib.util.find_spec('ns3')
if not spec or not spec.submodule_search_locations:
    sys.exit(1)
paths = list(spec.submodule_search_locations)
for p in paths:
    candidate = os.path.join(os.path.dirname(p), 'ns3', 'lib64', 'cmake', 'ns3')
    if os.path.isfile(os.path.join(candidate, 'ns3Config.cmake')):
        print(candidate)
        sys.exit(0)
sys.exit(1)
]=]
    OUTPUT_VARIABLE _NS3_PIP_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _NS3_PIP_RESULT
)

if(_NS3_PIP_RESULT EQUAL 0 AND EXISTS "${_NS3_PIP_CMAKE_DIR}")
    message(STATUS "ns3: found pip-installed CMake config at ${_NS3_PIP_CMAKE_DIR}")
    list(PREPEND CMAKE_PREFIX_PATH "${_NS3_PIP_CMAKE_DIR}")
endif()

find_package(ns3 REQUIRED COMPONENTS core network internet wifi mobility)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(ns3_wifi_sim "${CMAKE_CURRENT_SOURCE_DIR}/ns3_wifi_sim.cc")

target_link_libraries(ns3_wifi_sim PRIVATE
    ns3::core
    ns3::network
    ns3::internet
    ns3::wifi
    ns3::mobility
)

# Propagation module is often bundled with wifi in the CMake targets,
# but add it explicitly if available.
if(TARGET ns3::propagation)
    target_link_libraries(ns3_wifi_sim PRIVATE ns3::propagation)
endif()

# Ensure the runtime can find the ns3 shared libraries without
# requiring LD_LIBRARY_PATH.
set_target_properties(ns3_wifi_sim PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
