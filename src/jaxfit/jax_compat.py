"""JAX compatibility helpers."""
from __future__ import annotations

from typing import Callable, Optional, Tuple


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


def _get_jax_numpy_linalg() -> Optional[object]:
    try:
        import jax.numpy as jnp
        return jnp.linalg
    except Exception:
        return None


def _get_tree_flatten() -> Optional[Callable]:
    try:
        from jax.tree_util import tree_flatten
        return tree_flatten
    except Exception:
        try:
            from jax.tree import flatten as tree_flatten
            return tree_flatten
        except Exception:
            return None


def jax_tree_flatten(tree) -> Tuple[object, object]:
    tree_flatten = _get_tree_flatten()
    if tree_flatten is None:
        raise ImportError("jax.tree_util.tree_flatten is unavailable")
    return tree_flatten(tree)


def jax_svd(*args, **kwargs):
    jax_linalg = _get_jax_scipy_linalg()
    if jax_linalg is not None and hasattr(jax_linalg, "svd"):
        return jax_linalg.svd(*args, **kwargs)

    jax_np_linalg = _get_jax_numpy_linalg()
    if jax_np_linalg is None or not hasattr(jax_np_linalg, "svd"):
        raise ImportError("jax.scipy.linalg.svd is unavailable")
    return jax_np_linalg.svd(*args, **kwargs)


def jax_cholesky(a, lower: bool = False, **kwargs):
    jax_linalg = _get_jax_scipy_linalg()
    if jax_linalg is not None and hasattr(jax_linalg, "cholesky"):
        return jax_linalg.cholesky(a, lower=lower, **kwargs)

    jax_np_linalg = _get_jax_numpy_linalg()
    if jax_np_linalg is None or not hasattr(jax_np_linalg, "cholesky"):
        raise ImportError("jax.scipy.linalg.cholesky is unavailable")

    result = jax_np_linalg.cholesky(a)
    if lower:
        return result
    return result.swapaxes(-1, -2)


def _parse_transpose(trans) -> Tuple[bool, bool]:
    if trans in (0, None, "N", "n", False):
        return False, False
    if trans in (1, "T", "t", True):
        return True, False
    if trans in (2, "C", "c"):
        return True, True
    return False, False


def jax_solve_triangular(
    a,
    b,
    lower: bool = False,
    trans=0,
    unit_diagonal: bool = False,
    **kwargs,
):
    jax_linalg = _get_jax_scipy_linalg()
    if jax_linalg is not None and hasattr(jax_linalg, "solve_triangular"):
        return jax_linalg.solve_triangular(
            a,
            b,
            lower=lower,
            trans=trans,
            unit_diagonal=unit_diagonal,
            **kwargs,
        )

    try:
        from jax.lax.linalg import triangular_solve
    except Exception:
        raise ImportError("jax.scipy.linalg.solve_triangular is unavailable")

    transpose_a, conjugate_a = _parse_transpose(trans)
    squeeze = False
    if getattr(b, "ndim", None) == 1:
        b = b[..., None]
        squeeze = True

    result = triangular_solve(
        a,
        b,
        left_side=True,
        lower=lower,
        transpose_a=transpose_a,
        conjugate_a=conjugate_a,
        unit_diagonal=unit_diagonal,
    )
    if squeeze:
        result = result[..., 0]
    return result
