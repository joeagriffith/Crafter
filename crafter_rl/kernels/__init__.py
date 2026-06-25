"""Lazy loader for custom CUDA/C++ ops compiled into ``crafter_rl._C``.

Add ``.cu`` / ``.cpp`` sources under ``csrc/`` (see ``csrc/README.md``).
Building requires the CUDA Toolkit (``nvcc``) to be installed -- it is NOT
currently installed in this environment.

Importing this module never triggers a build. Call :func:`load_kernels`
explicitly to compile-and-load on demand; if there are no sources it returns
``None`` / raises a clear ``NotImplementedError`` instead of failing obscurely.
"""
import glob
import os

# csrc/ lives at the project root (three levels up from this file).
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSRC_DIR = os.path.join(_PKG_ROOT, "csrc")

_loaded = None


def _sources():
    """All .cu/.cpp sources currently present under csrc/."""
    return sorted(
        glob.glob(os.path.join(_CSRC_DIR, "*.cu"))
        + glob.glob(os.path.join(_CSRC_DIR, "*.cpp"))
    )


def load_kernels(verbose=False):
    """Lazily JIT-compile and load the custom ops as ``crafter_rl._C``.

    Returns the loaded extension module, or raises ``NotImplementedError`` if
    there are no sources yet (the common case today). Cached after first load.
    """
    global _loaded
    if _loaded is not None:
        return _loaded

    sources = _sources()
    if not sources:
        raise NotImplementedError(
            "No CUDA/C++ sources found under csrc/. Add .cu/.cpp files and "
            "ensure the CUDA Toolkit (nvcc) is installed, then call "
            "load_kernels() again. See csrc/README.md."
        )

    from torch.utils.cpp_extension import load  # imported lazily on purpose
    _loaded = load(name="crafter_rl_C", sources=sources, verbose=verbose)
    return _loaded
