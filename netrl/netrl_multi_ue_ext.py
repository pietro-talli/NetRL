"""Package-local loader for the compiled NetRL multi-UE NS3 extension."""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import os


def _preload_ns3_shared_libs() -> str | None:
    spec = importlib.util.find_spec("ns3")
    if not spec or not spec.submodule_search_locations:
        return None

    ns3_base = list(spec.submodule_search_locations)[0]
    lib_dir = os.path.join(ns3_base, "lib64")
    if not os.path.isdir(lib_dir):
        return None

    load_mode = 0
    if hasattr(os, "RTLD_NOW"):
        load_mode |= os.RTLD_NOW
    if hasattr(os, "RTLD_GLOBAL"):
        load_mode |= os.RTLD_GLOBAL

    for module in ["core", "network", "internet", "mobility", "propagation", "wifi"]:
        matches = sorted(glob.glob(os.path.join(lib_dir, f"libns3.*-{module}.so")))
        if matches:
            ctypes.CDLL(matches[0], mode=load_mode)

    return lib_dir


try:
    from _netrl_multi_ue_ext import *  # noqa: F403
except ImportError:
    preloaded_lib_dir = _preload_ns3_shared_libs()
    try:
        from _netrl_multi_ue_ext import *  # noqa: F403
    except ImportError as exc:
        hint = ""
        if preloaded_lib_dir:
            hint = (
                "\nIf needed for direct binary loading, export:\n"
                f"  export LD_LIBRARY_PATH={preloaded_lib_dir}:$LD_LIBRARY_PATH"
            )
        raise ImportError(
            "Could not load _netrl_multi_ue_ext backend for netrl.netrl_multi_ue_ext. "
            "Reinstall with: pip install -e ."
            f"{hint}"
        ) from exc
