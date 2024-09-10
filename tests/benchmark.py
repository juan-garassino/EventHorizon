"""Benchmarking routines used for optimization."""
from typing import Iterable

import numpy as np
import numpy.typing as npt

import bh
from data import Isoradial


def benchmark(th0: float, r_values: npt.NDArray[float], image_orders: Iterable[int]):
    """Run the isoradial benchmark routine.

        Run with pyinstrument or your favorite benchmarking tool to benchmark.

    Parameters
    ----------
    th0 : float
        Inclination with respect to the accretion disk normal, in degrees
    r_values : npt.NDArray[float]
        Isoradial distances
    image_orders : Iterable[int]
        Order of the calculations
    """
    calculate_alpha = np.linspace(0, 2 * np.pi, 1000)
    theta_0 = th0 * np.pi / 180

    isoradials = []
    for radius in r_values:
        for image_order in sorted(image_orders)[::-1]:
            isoradials.append(
                Isoradial(
                    bh.reorient_alpha(calculate_alpha, image_order),
                    bh.impact_parameter(calculate_alpha, radius, theta_0, image_order, 1),
                    radius,
                    theta_0,
                    image_order,
                )
            )

    return


if __name__ == "__main__":
    benchmark(th0=80, r_values=np.arange(6, 30, 2), image_orders=[0, 1])
