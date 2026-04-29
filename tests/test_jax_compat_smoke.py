import numpy as np
from numpy.testing import assert_allclose, assert_equal

import jax.numpy as jnp

from jaxfit import CurveFit


def _linear(x, a, b):
    return a * x + b


def test_enable_x64():
    arr = jnp.array([1.0])
    assert_equal(arr.dtype, jnp.float64)


def test_curve_fit_basic():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0])

    popt, pcov = CurveFit().curve_fit(_linear, x, y)

    assert_allclose(popt, [2.0, 1.0], rtol=1e-2, atol=1e-2)
    assert_equal(pcov.shape, (2, 2))


def test_curve_fit_bounds():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0])

    lb = np.array([0.0, 0.0])
    ub = np.array([3.0, 3.0])
    popt, _ = CurveFit().curve_fit(_linear, x, y, bounds=(lb, ub), method="trf")

    assert_allclose(np.minimum(np.maximum(popt, lb), ub), popt)


def test_curve_fit_sigma_vector():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0])
    sigma = np.array([1.0, 1.0, 1.0])

    popt, pcov = CurveFit().curve_fit(
        _linear,
        x,
        y,
        sigma=sigma,
        absolute_sigma=True,
        method="trf",
    )

    assert_allclose(popt, [2.0, 1.0], rtol=1e-2, atol=1e-2)
    assert_equal(pcov.shape, (2, 2))


def test_curve_fit_sigma_matrix():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0])
    sigma = np.diag([1.0, 1.0, 1.0])

    popt, pcov = CurveFit().curve_fit(
        _linear,
        x,
        y,
        sigma=sigma,
        absolute_sigma=True,
        method="trf",
    )

    assert_allclose(popt, [2.0, 1.0], rtol=1e-2, atol=1e-2)
    assert_equal(pcov.shape, (2, 2))


def test_curve_fit_data_mask():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0])
    data_mask = np.array([True, False, True])

    popt, pcov = CurveFit().curve_fit(
        _linear,
        x,
        y,
        data_mask=data_mask,
        method="trf",
    )

    assert_allclose(popt, [2.0, 1.0], rtol=1e-2, atol=1e-2)
    assert_equal(pcov.shape, (2, 2))
