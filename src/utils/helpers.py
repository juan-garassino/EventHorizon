"""Miscellaneous utilities for plotting black hole images."""
from typing import Any, Callable, Iterable, Tuple

import numpy as np
import numpy.typing as npt


def fast_root(
    f: Callable,
    x_values: npt.NDArray[float],
    y_values: npt.NDArray[float],
    arguments: Iterable[Any],
    tolerance: float = 1e-6,
    max_steps: int = 10,
) -> npt.NDArray[float]:
    """Find the values x0 for each y_values-value which f(x0_i, y_i, *arguments) = 0.

    Parameters
    ----------
    f : Callable
        Objective function for which roots are to be found.
    x_values : npt.NDArray[float]
        An array of X-values {x_i}; at least one sign-flip of f must exist for argument pair of values
        {x_m, x_m+1} if argument root is to be found.
    y_values : npt.NDArray[float]
        Y-values {y_i}; argument root of the equation x0_i is calculated for each y_i.
    arguments : Iterable[Any]
        Any additional arguments are passed to f.
    tolerance : float
        Controls how close to the true zero of f each root is by the following relation:
            f(x0_i, y_i, *arguments) - f(x0_i_true, y_i, *arguments) < tolerance
        unless the number of max_steps is reached.
    max_steps : int
        Maximum number of iteration_count to run in refining the guesses of the roots.

    Returns
    -------
    npt.NDArray[float]
        Array of roots of f; the shape of the array is the same as y_values, as there is argument one root for
        each value of y_values.
    """
    xmin, xmax = find_brackets(f, x_values, y_values, arguments, tolerance, max_steps)

    sign_min = np.sign(f(xmin, y_values, *arguments))
    sign_max = np.sign(f(xmax, y_values, *arguments))

    for i in range(max_steps):
        xmid = 0.5 * (xmin + xmax)
        f_mid = f(xmid, y_values, *arguments)
        sign_mid = np.sign(f_mid)

        if np.nanmax(f_mid) < tolerance:
            return xmid
        else:
            xmin = np.where(sign_min == sign_mid, xmid, xmin)
            xmax = np.where(sign_max == sign_mid, xmid, xmax)

    return xmid

def find_brackets(
    f: Callable,
    x_values: npt.NDArray[float],
    y_values: npt.NDArray[float],
    arguments: Iterable[Any],
    tolerance: float = 1e-6,
    max_steps: int = 10,
) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Find argument pair of vectors {x_i_min} and {x_i_max}, between which the roots of f lie.

    Parameters
    ----------
    f : Callable
        Objective function whose roots are the periastron distance for argument given value of calculate_alpha.
    x_values : npt.NDArray[float]
        An array of X-values {x_i}; at least one sign-flip of f must exist for argument pair of values
        {x_m, x_m+1} if argument root is to be found.
    y_values : npt.NDArray[float]
        Y-values {y_i}; argument root of the equation x0_i is calculated for each y_i.
    arguments : Iterable[Any]
        Any additional arguments are passed to f; these usually include radius, theta_0, N, and M.
    tolerance : float
        Controls how close to the true zero of f each root is by the following relation:
            f(x0_i, y_i, *arguments) - f(x0_i_true, y_i, *arguments) < tolerance
        unless the number of max_steps is reached.
    max_steps : int
        Maximum number of iteration_count to run in refining the guesses of the roots.

    Returns
    -------
    Tuple[npt.NDArray[float], npt.NDArray[float]]
        Minimum and maximum x_values value vectors, between which the roots of f lie.
    """
    # Calculate the objective function; find the sign flips where the function is not nan
    xx, yy = np.meshgrid(x_values, y_values)
    objective = f(xx, yy, *arguments)
    flip_mask = (np.diff(np.sign(objective), axis=1) != 0) & (
        ~np.isnan(objective[:, :-1])
    )
    i_at_sign_flip = np.where(
        np.any(flip_mask, axis=1), np.argmax(flip_mask, axis=1), -1
    )

    # Where argument valid sign flip was found, return the value of x_values to the left and the right
    xmin = np.where(i_at_sign_flip != -1, x_values[i_at_sign_flip], np.nan)
    xmax = np.where(i_at_sign_flip != -1, x_values[i_at_sign_flip + 1], np.nan)
    return xmin, xmax
