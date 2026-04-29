"""JAX compatibility helpers."""
from __future__ import annotations

from typing import Optional


def _get_jax_config() -> Optional[object]:
    try:
        from jax import config as jax_config
        return jax_config
    except Exception:
        try:
            from jax.config import config as jax_config
            return jax_config
        except Exception:
            return None


def enable_x64() -> None:
    """Best-effort enablement of 64-bit mode across JAX versions."""
    jax_config = _get_jax_config()
    if jax_config is None:
        return

    try:
        jax_config.update("jax_enable_x64", True)
    except Exception:
        pass


def _get_jax_scipy_linalg() -> Optional[object]:
    try:
        from jax.scipy import linalg as jax_linalg
        return jax_linalg
    except Exception:
        return None


def jax_svd(*args, **kwargs):
    jax_linalg = _get_jax_scipy_linalg()
    if jax_linalg is None:
        raise ImportError("jax.scipy.linalg is unavailable")
    return jax_linalg.svd(*args, **kwargs)


def jax_cholesky(*args, **kwargs):
    jax_linalg = _get_jax_scipy_linalg()
    if jax_linalg is None:
        raise ImportError("jax.scipy.linalg is unavailable")
    return jax_linalg.cholesky(*args, **kwargs)


def jax_solve_triangular(*args, **kwargs):
    jax_linalg = _get_jax_scipy_linalg()
    if jax_linalg is None:
        raise ImportError("jax.scipy.linalg is unavailable")
    return jax_linalg.solve_triangular(*args, **kwargs)
